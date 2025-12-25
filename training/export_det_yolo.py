"""
Export detection annotations from the database to a YOLO-format dataset.

Usage:
  python training/export_det_yolo.py --db ~/.trailcam/trailcam.db --out exports/det_dataset --val_split 0.2

What it does:
- Reads boxes from the annotation_boxes table.
- By default, only exports HUMAN-VERIFIED boxes ("subject", "deer_head").
- AI-generated boxes ("ai_subject", "ai_deer_head") are excluded unless --include_ai is specified.
- Copies images into train/val/images and writes matching YOLO .txt label files in train/val/labels.
- Writes a dataset YAML (dataset.yaml) with class names for ultralytics YOLO.

Note on label noise:
- Even human-verified labels may contain errors (missed animals, sloppy boxes, wrong species).
- Consider using label smoothing during training to handle this.
- The model can be used to flag uncertain predictions for human review (active learning).

Requirements: pillow
"""
import argparse
import os
import random
import shutil
import sqlite3
from pathlib import Path

from PIL import Image


CLASSES = ["subject", "deer_head"]

# Human-verified labels only (default for training)
HUMAN_LABEL_MAP = {
    "subject": 0,
    "deer_head": 1,
}

# AI-generated labels (excluded by default, can be included with --include_ai)
AI_LABEL_MAP = {
    "ai_subject": 0,
    "ai_deer_head": 1,
}


def fetch_boxes(db_path: Path, include_ai: bool = False):
    """Fetch annotation boxes from database.

    Args:
        db_path: Path to SQLite database
        include_ai: If True, include AI-generated boxes. Default False (human-only).

    Returns:
        Tuple of (items dict, stats dict)
    """
    # Build label map based on options
    label_map = dict(HUMAN_LABEL_MAP)
    if include_ai:
        label_map.update(AI_LABEL_MAP)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id AS photo_id, p.file_path, b.label, b.x1, b.y1, b.x2, b.y2
        FROM annotation_boxes b
        JOIN photos p ON p.id = b.photo_id
        WHERE p.file_path != '' AND p.file_path IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    items = {}
    stats = {
        "human_subject": 0,
        "human_deer_head": 0,
        "ai_subject": 0,
        "ai_deer_head": 0,
        "skipped_ai": 0,
        "skipped_unknown": 0,
    }

    for r in rows:
        lbl = (r["label"] or "").strip().lower()

        # Track statistics
        if lbl == "subject":
            stats["human_subject"] += 1
        elif lbl == "deer_head":
            stats["human_deer_head"] += 1
        elif lbl == "ai_subject":
            stats["ai_subject"] += 1
            if not include_ai:
                stats["skipped_ai"] += 1
                continue
        elif lbl == "ai_deer_head":
            stats["ai_deer_head"] += 1
            if not include_ai:
                stats["skipped_ai"] += 1
                continue

        # Normalize label for lookup (handle case variations)
        lbl_normalized = lbl.lower()
        if lbl_normalized == "subject":
            cls_id = 0
        elif lbl_normalized == "deer_head":
            cls_id = 1
        elif lbl_normalized == "ai_subject" and include_ai:
            cls_id = 0
        elif lbl_normalized == "ai_deer_head" and include_ai:
            cls_id = 1
        else:
            stats["skipped_unknown"] += 1
            continue

        items.setdefault(r["file_path"], []).append({
            "cls": cls_id,
            "x1": float(r["x1"]),
            "y1": float(r["y1"]),
            "x2": float(r["x2"]),
            "y2": float(r["y2"]),
        })

    return items, stats


def write_yolo(dataset_dir: Path, items: dict, val_split: float = 0.2):
    imgs = list(items.keys())
    random.shuffle(imgs)
    split_idx = int(len(imgs) * (1 - val_split))
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]
    for split, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for src in split_imgs:
            try:
                img = Image.open(src)
                w, h = img.size
            except Exception:
                continue
            dst_img = img_dir / Path(src).name
            try:
                shutil.copy2(src, dst_img)
            except Exception:
                continue
            lines = []
            for b in items[src]:
                cx = (b["x1"] + b["x2"]) / 2.0
                cy = (b["y1"] + b["y2"]) / 2.0
                bw = (b["x2"] - b["x1"])
                bh = (b["y2"] - b["y1"])
                lines.append(f'{b["cls"]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
            dst_lbl = lbl_dir / (dst_img.stem + ".txt")
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
    # dataset yaml
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {dataset_dir}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"names:\n"
        + "\n".join([f"  {i}: {cls}" for i, cls in enumerate(CLASSES)]),
        encoding="utf-8",
    )
    return train_imgs, val_imgs


def main():
    ap = argparse.ArgumentParser(description="Export detection annotations to YOLO format")
    ap.add_argument("--db", default=str(Path.home() / ".trailcam" / "trailcam.db"),
                    help="Path to SQLite database")
    ap.add_argument("--out", default="exports/det_dataset",
                    help="Output directory for YOLO dataset")
    ap.add_argument("--val_split", type=float, default=0.2,
                    help="Fraction of data for validation (default: 0.2)")
    ap.add_argument("--include_ai", action="store_true",
                    help="Include AI-generated boxes (not recommended for training)")
    args = ap.parse_args()

    print(f"[export] Database: {args.db}")
    print(f"[export] Include AI boxes: {args.include_ai}")

    items, stats = fetch_boxes(Path(args.db), include_ai=args.include_ai)

    # Print statistics
    print(f"\n[export] Box statistics:")
    print(f"  Human 'subject' boxes:    {stats['human_subject']}")
    print(f"  Human 'deer_head' boxes:  {stats['human_deer_head']}")
    print(f"  AI 'ai_subject' boxes:    {stats['ai_subject']}")
    print(f"  AI 'ai_deer_head' boxes:  {stats['ai_deer_head']}")
    if stats['skipped_ai'] > 0:
        print(f"  ⚠️  Skipped AI boxes:      {stats['skipped_ai']} (use --include_ai to include)")
    if stats['skipped_unknown'] > 0:
        print(f"  ⚠️  Skipped unknown labels: {stats['skipped_unknown']}")

    total_boxes = stats['human_subject'] + stats['human_deer_head']
    if args.include_ai:
        total_boxes += stats['ai_subject'] + stats['ai_deer_head']
    print(f"  Total boxes for export:   {total_boxes}")

    if not items:
        print("\n[export] No boxes found to export.")
        return

    dataset_dir = Path(args.out)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    train_imgs, val_imgs = write_yolo(dataset_dir, items, val_split=args.val_split)

    print(f"\n[export] YOLO dataset written to {dataset_dir}")
    print(f"[export] Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")
    print(f"[export] Classes: {CLASSES}")
    print(f"[export] YAML: {dataset_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
