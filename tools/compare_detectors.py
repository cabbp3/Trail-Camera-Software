import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/Users/brookebratcher/Desktop/chris/Trail Camera Software V 1.0")

from ai_detection import MegaDetectorV5
from speciesnet_wrapper import SpeciesNetWrapper
from database import TrailCamDatabase

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(folder: Path):
    images = []
    for root, _, files in os.walk(folder):
        for name in files:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                images.append(Path(root) / name)
    return images


def _build_tag_map(db: TrailCamDatabase):
    photos = db.get_all_photos(include_archived=True)
    path_to_tags = {}
    base_to_tags = defaultdict(list)
    for p in photos:
        photo_id = p.get("id")
        file_path = p.get("file_path")
        if not file_path or not photo_id:
            continue
        tags = db.get_tags(photo_id) or []
        path_to_tags[file_path] = tags
        base_to_tags[Path(file_path).name].append(tags)
    return path_to_tags, base_to_tags


def _get_tags_for_image(path: Path, path_map, base_map):
    tags = path_map.get(str(path))
    if tags is not None:
        return tags, False
    base_tags = base_map.get(path.name)
    if not base_tags:
        return None, False
    if len(base_tags) == 1:
        return base_tags[0], False
    # ambiguous
    return None, True


def _is_empty_tag(tags):
    if not tags:
        return False
    return any(t.lower() == "empty" for t in tags)


def _is_species_tagged(tags):
    if not tags:
        return False
    for t in tags:
        if t.lower() != "empty":
            return True
    return False


def _histogram(values, bins=10):
    if not values:
        return [0] * bins
    counts = [0] * bins
    for v in values:
        if v < 0:
            idx = 0
        elif v >= 1:
            idx = bins - 1
        else:
            idx = int(v * bins)
        counts[idx] += 1
    return counts


def _print_hist(title, counts):
    total = sum(counts) or 1
    print(f"{title} (0.0-1.0, {len(counts)} bins)")
    for i, c in enumerate(counts):
        lo = i / len(counts)
        hi = (i + 1) / len(counts)
        pct = (c / total) * 100
        print(f"  {lo:.2f}-{hi:.2f}: {c} ({pct:.1f}%)")


def _run_megadetector(detector: MegaDetectorV5, image_path: str, threshold: float):
    detections = detector.detect(image_path)
    boxes = []
    for d in detections:
        if d.confidence >= threshold:
            boxes.append({"confidence": float(d.confidence)})
    return boxes


def _run_speciesnet(wrapper: SpeciesNetWrapper, image_path: str, threshold: float):
    wrapper.DETECTION_CONF_THRESHOLD = threshold
    result = wrapper.detect_and_classify(image_path)
    boxes = result.get("detections", []) or []
    return boxes


def threshold_sweep(images, tags_lookup, wrapper: SpeciesNetWrapper):
    thresholds = [round(0.1 + i * 0.05, 2) for i in range(11)]
    best = None
    for t in thresholds:
        tp = fp = fn = 0
        for img in images:
            tags, _ = tags_lookup(img)
            if tags is None:
                continue
            boxes = _run_speciesnet(wrapper, str(img), t)
            has_det = len(boxes) > 0
            if _is_empty_tag(tags):
                if has_det:
                    fp += 1
            elif _is_species_tagged(tags):
                if has_det:
                    tp += 1
                else:
                    fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        if not best or f1 > best["f1"]:
            best = {"threshold": t, "precision": precision, "recall": recall, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn}
        print(f"Threshold {t:.2f} | TP {tp} FP {fp} FN {fn} | P {precision:.3f} R {recall:.3f} F1 {f1:.3f}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Compare MegaDetector v5 vs SpeciesNet detections")
    parser.add_argument("folder", help="Folder of images")
    parser.add_argument("--csv", help="Write per-image results to CSV")
    parser.add_argument("--sample", type=int, default=0, help="Random sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sweep", action="store_true", help="Run SpeciesNet threshold sweep")
    parser.add_argument("--md-threshold", type=float, default=0.85)
    parser.add_argument("--sn-threshold", type=float, default=0.15)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return 1

    images = _list_images(folder)
    if not images:
        print("No images found")
        return 1

    if args.sample and args.sample > 0 and args.sample < len(images):
        random.seed(args.seed)
        images = random.sample(images, args.sample)

    db = TrailCamDatabase()
    path_map, base_map = _build_tag_map(db)
    def tags_lookup(p):
        return _get_tags_for_image(p, path_map, base_map)

    detector = MegaDetectorV5(confidence_threshold=args.md_threshold)
    wrapper = SpeciesNetWrapper()
    wrapper.initialize()

    md_conf = []
    sn_conf = []
    md_counts = []
    sn_counts = []

    md_has = set()
    sn_has = set()

    md_fp = md_fn = 0
    sn_fp = sn_fn = 0
    md_empty_total = 0
    md_species_total = 0
    sn_empty_total = 0
    sn_species_total = 0
    unknown_tags = 0
    ambiguous_tags = 0

    rows = []

    for img in images:
        tags, ambiguous = tags_lookup(img)
        if tags is None:
            if ambiguous:
                ambiguous_tags += 1
            else:
                unknown_tags += 1
        is_empty = _is_empty_tag(tags) if tags is not None else False
        is_species = _is_species_tagged(tags) if tags is not None else False

        md_boxes = _run_megadetector(detector, str(img), args.md_threshold)
        sn_boxes = _run_speciesnet(wrapper, str(img), args.sn_threshold)

        md_counts.append(len(md_boxes))
        sn_counts.append(len(sn_boxes))
        md_conf.extend([b["confidence"] for b in md_boxes])
        sn_conf.extend([b["confidence"] for b in sn_boxes])

        if md_boxes:
            md_has.add(str(img))
        if sn_boxes:
            sn_has.add(str(img))

        if tags is not None:
            if is_empty:
                md_empty_total += 1
                sn_empty_total += 1
                if md_boxes:
                    md_fp += 1
                if sn_boxes:
                    sn_fp += 1
            elif is_species:
                md_species_total += 1
                sn_species_total += 1
                if not md_boxes:
                    md_fn += 1
                if not sn_boxes:
                    sn_fn += 1

        if args.csv:
            rows.append({
                "image": str(img),
                "tags": ";".join(tags) if tags else "",
                "md_count": len(md_boxes),
                "sn_count": len(sn_boxes),
                "md_max_conf": max([b["confidence"] for b in md_boxes], default=0),
                "sn_max_conf": max([b["confidence"] for b in sn_boxes], default=0),
                "empty_tag": int(is_empty),
                "species_tag": int(is_species),
            })

    total = len(images)
    both = len(md_has & sn_has)
    either = len(md_has | sn_has)
    agree_jaccard = both / either if either else 0
    agree_all = both / total if total else 0

    print("\n=== Detector Comparison Summary ===")
    print(f"Images analyzed: {total}")
    print(f"Unknown tags: {unknown_tags} | Ambiguous tags: {ambiguous_tags}")
    print(f"MegaDetector boxes total: {sum(md_counts)} | avg per image: {sum(md_counts)/total:.2f}")
    print(f"SpeciesNet boxes total: {sum(sn_counts)} | avg per image: {sum(sn_counts)/total:.2f}")
    print(f"Agreement (both detect on same images): {both}/{total} = {agree_all:.3f}")
    print(f"Agreement Jaccard (both/either): {both}/{either} = {agree_jaccard:.3f}")

    print("\n=== False Positives / Negatives (Tag-based) ===")
    print(f"MegaDetector FP on Empty: {md_fp}/{md_empty_total} = {(md_fp/md_empty_total if md_empty_total else 0):.3f}")
    print(f"MegaDetector FN on Species: {md_fn}/{md_species_total} = {(md_fn/md_species_total if md_species_total else 0):.3f}")
    print(f"SpeciesNet FP on Empty: {sn_fp}/{sn_empty_total} = {(sn_fp/sn_empty_total if sn_empty_total else 0):.3f}")
    print(f"SpeciesNet FN on Species: {sn_fn}/{sn_species_total} = {(sn_fn/sn_species_total if sn_species_total else 0):.3f}")

    print("\n=== Confidence Histograms ===")
    _print_hist("MegaDetector confidences", _histogram(md_conf, bins=10))
    _print_hist("SpeciesNet confidences", _histogram(sn_conf, bins=10))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"\nWrote CSV: {args.csv}")

    if args.sweep:
        print("\n=== SpeciesNet Threshold Sweep ===")
        best = threshold_sweep(images, tags_lookup, wrapper)
        if best:
            print(
                f"Best threshold: {best['threshold']:.2f} | F1 {best['f1']:.3f} "
                f"(P {best['precision']:.3f}, R {best['recall']:.3f})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
