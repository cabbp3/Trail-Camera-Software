#!/usr/bin/env python3
"""
Run AI species suggestions on all unlabeled photos.
"""
import sys
import os

# Change to script directory so model paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from database import TrailCamDatabase
from ai_suggester import CombinedSuggester

def main():
    db = TrailCamDatabase()
    suggester = CombinedSuggester()

    if not suggester.ready:
        print("ERROR: AI model not available. Check models/ folder.")
        sys.exit(1)

    print(f"AI model ready: CLIP={suggester.clip and suggester.clip.ready}, ONNX={suggester.onnx and suggester.onnx.ready}")
    print(f"Buck/Doe model ready: {suggester.buckdoe_ready}")

    # Get all photos
    all_photos = db.get_all_photos()

    # Find photos without tags OR without suggestions
    unlabeled = []
    for p in all_photos:
        tags = db.get_tags(p["id"])
        has_suggestion = p.get("suggested_tag")
        if not tags and not has_suggestion:  # No tags and no suggestion
            unlabeled.append(p)

    print(f"\nFound {len(unlabeled)} unlabeled photos out of {len(all_photos)} total.")

    if not unlabeled:
        print("Nothing to process!")
        return

    species_count = 0
    sex_count = 0
    errors = 0

    for i, p in enumerate(unlabeled, 1):
        path = p.get("file_path")
        if not path or not os.path.exists(path):
            errors += 1
            continue

        # Run species prediction
        result = suggester.predict(path)
        if result:
            label, conf = result
            db.set_suggested_tag(p["id"], label, conf)
            species_count += 1

            # If Deer with buck/doe model, also suggest sex
            if label == "Deer" and suggester.buckdoe_ready:
                sex_result = suggester.predict_sex(path)
                if sex_result:
                    sex_label, sex_conf = sex_result
                    db.set_suggested_sex(p["id"], sex_label, sex_conf)
                    sex_count += 1

        # Progress update every 50 photos
        if i % 50 == 0:
            print(f"  Processed {i}/{len(unlabeled)}...")

    print(f"\nDone!")
    print(f"  Species suggestions: {species_count}")
    print(f"  Buck/doe suggestions: {sex_count}")
    if errors:
        print(f"  Errors (missing files): {errors}")

if __name__ == "__main__":
    main()
