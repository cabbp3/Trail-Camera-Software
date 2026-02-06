"""One-time comparison: run SpeciesNet on all labeled photos and find mismatches.

Usage:
    cd "Trail Camera Software V 1.0"
    source .venv/bin/activate
    python tools/compare_labels_vs_speciesnet.py

Outputs:
    ~/Desktop/speciesnet_mismatches.csv  (mismatches only)
    ~/Desktop/speciesnet_full_results.csv (all results)
    Console summary of match rate

Can be interrupted and resumed — skips photos already in the output CSV.
"""

import csv
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import TrailCamDatabase
from speciesnet_wrapper import SpeciesNetWrapper
import user_config

# Species labels used in the app (matches SPECIES_OPTIONS in label_tool.py)
SPECIES_SET = {
    "Armadillo", "Bobcat", "Chipmunk", "Coyote", "Deer", "Dog", "Empty",
    "Flicker", "Fox", "Ground Hog", "House Cat", "Opossum", "Other",
    "Other Bird", "Otter", "Person", "Quail", "Rabbit", "Raccoon",
    "Skunk", "Squirrel", "Turkey", "Turkey Buzzard", "Unknown", "Vehicle",
    "Verification",
}

MISMATCH_CSV = Path.home() / "Desktop" / "speciesnet_mismatches.csv"
FULL_CSV = Path.home() / "Desktop" / "speciesnet_full_results.csv"


def load_already_processed(csv_path):
    """Load photo IDs already processed (for resume support)."""
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done.add(int(row["photo_id"]))
                except (ValueError, KeyError):
                    pass
    return done


def get_human_species(db, photo_id):
    """Get the human-confirmed species label for a photo from tags table."""
    tags = db.get_tags(photo_id)
    species_tags = [t for t in tags if t in SPECIES_SET]
    if species_tags:
        # Return the first species tag (most photos have one)
        return species_tags[0]
    return None


def main():
    print("=== SpeciesNet vs Human Labels Comparison ===\n")

    # Clear old mismatch review queues
    queue_files = [
        Path.home() / ".trailcam" / "misclassified_review_queue.json",
        Path.home() / ".trailcam" / "claude_review_queue.json",
        Path.home() / ".trailcam" / "v6_missed_review_queue.json",
    ]
    for qf in queue_files:
        if qf.exists():
            qf.unlink()
            print(f"Cleared old queue: {qf.name}")

    # Initialize database
    db = TrailCamDatabase()

    # Initialize SpeciesNet
    state = user_config.get_speciesnet_state()
    print(f"Geofencing state: {state or '(none)'}")

    wrapper = SpeciesNetWrapper(state=state, geofence=bool(state))
    print("Loading SpeciesNet model (first time may download ~500 MB)...")
    if not wrapper.initialize():
        print(f"ERROR: Failed to load SpeciesNet: {wrapper.error_message}")
        sys.exit(1)
    print("SpeciesNet loaded.\n")

    # Get all photos
    all_photos = db.get_all_photos(include_archived=True)
    print(f"Total photos in database: {len(all_photos)}")

    # Filter to photos with human species labels
    labeled_photos = []
    for p in all_photos:
        pid = p.get("id")
        species = get_human_species(db, pid)
        if species and species not in ("Empty", "Verification", "Unknown", "Other"):
            labeled_photos.append((p, species))

    print(f"Photos with human species labels: {len(labeled_photos)}")

    # Check for resume
    already_done = load_already_processed(FULL_CSV)
    remaining = [(p, s) for p, s in labeled_photos if p["id"] not in already_done]
    if already_done:
        print(f"Already processed: {len(already_done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All photos already processed. Check the CSV files on your Desktop.")
        return

    # Open CSV files for appending
    full_is_new = not FULL_CSV.exists() or not already_done
    mismatch_is_new = not MISMATCH_CSV.exists() or not already_done

    full_file = open(FULL_CSV, "a", newline="")
    mismatch_file = open(MISMATCH_CSV, "a", newline="")

    fieldnames = ["photo_id", "file_path", "original_name", "human_label",
                  "speciesnet_label", "speciesnet_raw", "speciesnet_confidence", "match"]

    full_writer = csv.DictWriter(full_file, fieldnames=fieldnames)
    mismatch_writer = csv.DictWriter(mismatch_file, fieldnames=fieldnames)

    if full_is_new:
        full_writer.writeheader()
    if mismatch_is_new:
        mismatch_writer.writeheader()

    # Process photos
    matches = 0
    mismatches = 0
    errors = 0
    start_time = time.time()

    try:
        for i, (photo, human_label) in enumerate(remaining):
            pid = photo["id"]
            file_path = photo.get("file_path", "")

            if not file_path or not os.path.exists(file_path):
                errors += 1
                continue

            try:
                result = wrapper.detect_and_classify(file_path)
                sn_species = result.get("app_species") or "Empty"
                sn_raw = result.get("raw_prediction") or ""
                sn_conf = result.get("prediction_score", 0)
            except Exception as e:
                print(f"  Error on photo {pid}: {e}")
                errors += 1
                continue

            is_match = human_label.lower() == sn_species.lower()
            if is_match:
                matches += 1
            else:
                mismatches += 1

            row = {
                "photo_id": pid,
                "file_path": file_path,
                "original_name": photo.get("original_name", ""),
                "human_label": human_label,
                "speciesnet_label": sn_species,
                "speciesnet_raw": sn_raw,
                "speciesnet_confidence": f"{sn_conf:.3f}",
                "match": "yes" if is_match else "no",
            }

            full_writer.writerow(row)
            if not is_match:
                mismatch_writer.writerow(row)

            # Progress every 50 photos
            processed = i + 1
            if processed % 50 == 0 or processed == len(remaining):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - processed) / rate if rate > 0 else 0
                total_done = len(already_done) + processed
                print(f"  [{total_done}/{len(labeled_photos)}] "
                      f"matches={matches} mismatches={mismatches} errors={errors} "
                      f"({rate:.1f} photos/sec, ETA: {eta/60:.0f} min)")
                # Flush to disk
                full_file.flush()
                mismatch_file.flush()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved. Run again to resume.")
    finally:
        full_file.close()
        mismatch_file.close()

    # Summary
    total = matches + mismatches
    match_rate = (matches / total * 100) if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Photos compared:   {total}")
    print(f"Matches:           {matches} ({match_rate:.1f}%)")
    print(f"Mismatches:        {mismatches} ({100-match_rate:.1f}%)")
    print(f"Errors/missing:    {errors}")
    print(f"\nFull results:      {FULL_CSV}")
    print(f"Mismatches only:   {MISMATCH_CSV}")


if __name__ == "__main__":
    main()
