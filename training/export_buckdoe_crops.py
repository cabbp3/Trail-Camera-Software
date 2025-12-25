"""
Export deer head crops with buck/doe labels for training a sex classifier.

Usage:
  python training/export_buckdoe_crops.py --db ~/.trailcam/trailcam.db --out exports/buckdoe_crops

What it does:
- Finds photos tagged as "buck" or "doe" that have deer_head bounding boxes
- Crops the deer_head region from each photo
- Organizes into train/val splits with class folders (buck/, doe/)
- Suitable for training with train_classifier.py or similar

Requirements: pillow
"""
import argparse
import os
import random
import sqlite3
from pathlib import Path

from PIL import Image


def fetch_buckdoe_with_heads(db_path: Path):
    """
    Fetch photos that have both a buck/doe tag AND a deer_head box.

    Returns list of dicts with: file_path, sex, x1, y1, x2, y2 (normalized coords)
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get photos with buck/doe tags and deer_head boxes
    cur.execute("""
        SELECT DISTINCT
            p.id,
            p.file_path,
            LOWER(t.tag_name) as sex,
            b.x1, b.y1, b.x2, b.y2
        FROM photos p
        JOIN tags t ON t.photo_id = p.id
        JOIN annotation_boxes b ON b.photo_id = p.id
        WHERE LOWER(t.tag_name) IN ('buck', 'doe')
          AND LOWER(b.label) = 'deer_head'
          AND p.file_path IS NOT NULL
          AND p.file_path != ''
    """)

    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        items.append({
            "photo_id": r["id"],
            "file_path": r["file_path"],
            "sex": r["sex"],
            "x1": float(r["x1"]),
            "y1": float(r["y1"]),
            "x2": float(r["x2"]),
            "y2": float(r["y2"]),
        })

    return items


def crop_and_save(item: dict, output_dir: Path, index: int, padding: float = 0.1) -> bool:
    """
    Crop deer_head from image and save to class folder.

    Args:
        item: Dict with file_path, sex, x1, y1, x2, y2
        output_dir: Base output directory (should have buck/ and doe/ subdirs)
        index: Unique index for filename
        padding: Extra padding around box as fraction of box size (default 10%)

    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(item["file_path"])
        w, h = img.size

        # Convert normalized coords to pixels
        x1 = int(item["x1"] * w)
        y1 = int(item["y1"] * h)
        x2 = int(item["x2"] * w)
        y2 = int(item["y2"] * h)

        # Add padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * padding)
        pad_y = int(box_h * padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Crop
        crop = img.crop((x1, y1, x2, y2))

        # Save to class folder
        class_dir = output_dir / item["sex"]
        class_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        orig_name = Path(item["file_path"]).stem
        out_path = class_dir / f"{orig_name}_{index:04d}.jpg"

        # Handle duplicates
        counter = 1
        while out_path.exists():
            out_path = class_dir / f"{orig_name}_{index:04d}_{counter}.jpg"
            counter += 1

        crop.save(out_path, "JPEG", quality=95)
        return True

    except Exception as e:
        print(f"  Error processing {item['file_path']}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Export deer head crops for buck/doe classification")
    ap.add_argument("--db", default=str(Path.home() / ".trailcam" / "trailcam.db"),
                    help="Path to SQLite database")
    ap.add_argument("--out", default="exports/buckdoe_crops",
                    help="Output directory")
    ap.add_argument("--val_split", type=float, default=0.2,
                    help="Fraction for validation (default: 0.2)")
    ap.add_argument("--padding", type=float, default=0.1,
                    help="Padding around deer_head box as fraction (default: 0.1)")
    ap.add_argument("--min_size", type=int, default=32,
                    help="Minimum crop size in pixels (default: 32)")
    args = ap.parse_args()

    print(f"[export] Database: {args.db}")
    print(f"[export] Output: {args.out}")
    print(f"[export] Val split: {args.val_split}")

    # Fetch data
    items = fetch_buckdoe_with_heads(Path(args.db))

    if not items:
        print("[export] No buck/doe photos with deer_head boxes found.")
        return

    # Count by class
    bucks = [i for i in items if i["sex"] == "buck"]
    does = [i for i in items if i["sex"] == "doe"]

    print(f"\n[export] Found:")
    print(f"  Buck crops: {len(bucks)}")
    print(f"  Doe crops:  {len(does)}")

    # Shuffle and split
    random.shuffle(bucks)
    random.shuffle(does)

    buck_split = int(len(bucks) * (1 - args.val_split))
    doe_split = int(len(does) * (1 - args.val_split))

    train_items = bucks[:buck_split] + does[:doe_split]
    val_items = bucks[buck_split:] + does[doe_split:]

    random.shuffle(train_items)
    random.shuffle(val_items)

    print(f"\n[export] Split:")
    print(f"  Train: {len(train_items)} ({len([i for i in train_items if i['sex']=='buck'])} buck, {len([i for i in train_items if i['sex']=='doe'])} doe)")
    print(f"  Val:   {len(val_items)} ({len([i for i in val_items if i['sex']=='buck'])} buck, {len([i for i in val_items if i['sex']=='doe'])} doe)")

    # Create output directories
    out_dir = Path(args.out)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    # Clear existing
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Process train
    print(f"\n[export] Exporting train crops...")
    train_success = 0
    for idx, item in enumerate(train_items):
        if crop_and_save(item, train_dir, idx, padding=args.padding):
            train_success += 1

    # Process val
    print(f"[export] Exporting val crops...")
    val_success = 0
    for idx, item in enumerate(val_items):
        if crop_and_save(item, val_dir, idx, padding=args.padding):
            val_success += 1

    print(f"\n[export] Done!")
    print(f"  Train crops saved: {train_success}")
    print(f"  Val crops saved:   {val_success}")
    print(f"  Output directory:  {out_dir}")
    print(f"\n[export] To train, run:")
    print(f"  python training/train_classifier.py \\")
    print(f"    --train_dir {out_dir}/train \\")
    print(f"    --val_dir {out_dir}/val \\")
    print(f"    --out outputs/buckdoe.onnx \\")
    print(f"    --labels outputs/buckdoe_labels.txt")


if __name__ == "__main__":
    main()
