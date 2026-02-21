#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import TrailCamDatabase

SPECIES_OPTIONS = [
    "",
    "Armadillo",
    "Bobcat",
    "Chipmunk",
    "Coyote",
    "Deer",
    "Dog",
    "Empty",
    "Flicker",
    "Fox",
    "Ground Hog",
    "House Cat",
    "Opossum",
    "Other",
    "Other Bird",
    "Otter",
    "Person",
    "Quail",
    "Rabbit",
    "Raccoon",
    "Skunk",
    "Squirrel",
    "Turkey",
    "Turkey Buzzard",
    "Unknown",
    "Vehicle",
    "Verification",
]


def is_head_box(box: Dict) -> bool:
    label = str(box.get("label", "")).lower()
    return "head" in label


def main():
    parser = argparse.ArgumentParser(description="Cleanup box species / photo tags desync.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    db = TrailCamDatabase()
    try:
        try:
            db.backup_before_batch_operation()
        except Exception:
            pass

        species_set = set(s for s in SPECIES_OPTIONS if s)
        try:
            species_set.update(db.list_custom_species())
        except Exception:
            pass

        photos = db.get_all_photos(include_archived=True)

        stats = {
            "photos": 0,
            "boxes_cleared_for_empty": 0,
            "box_species_seeded": 0,
            "tags_updated": 0,
            "boxes_updated": 0,
        }

        for photo in photos:
            pid = photo.get("id")
            if not pid:
                continue
            stats["photos"] += 1
            tags = db.get_tags(pid)
            boxes = db.get_boxes(pid)

            if "Empty" in tags and boxes:
                if not args.dry_run:
                    db.set_boxes(pid, [])
                    db.update_photo_tags(pid, ["Empty"])
                stats["boxes_cleared_for_empty"] += 1
                continue

            if not boxes:
                continue

            subject_boxes = [b for b in boxes if not is_head_box(b)]
            photo_species = [t for t in tags if t in species_set and t not in ("Empty", "Unknown", "Verification")]

            seeded = False
            if len(subject_boxes) == 1 and (not subject_boxes[0].get("species")) and len(photo_species) == 1:
                subject_boxes[0]["species"] = photo_species[0]
                seeded = True

            box_species = set(
                b.get("species") for b in boxes
                if b.get("species") and b.get("species") != "Unknown"
            )

            new_tags: List[str] = [t for t in tags if t not in species_set]
            if box_species:
                new_tags.extend(sorted(box_species))

            if set(new_tags) != set(tags):
                if not args.dry_run:
                    db.update_photo_tags(pid, new_tags)
                stats["tags_updated"] += 1

            if seeded:
                if not args.dry_run:
                    db.set_boxes(pid, boxes)
                stats["box_species_seeded"] += 1
                stats["boxes_updated"] += 1

        print("Cleanup complete")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if args.dry_run:
            print("(dry-run: no changes written)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
