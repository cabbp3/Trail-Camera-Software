#!/usr/bin/env python3
"""
Fix Cloud Photo Timestamps

Downloads photos from R2, reads EXIF timestamps, and updates Supabase.
Fixes photos that have upload timestamps instead of actual capture timestamps.
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from PIL.ExifTags import TAGS


def get_exif_datetime(image_path: Path):
    """Read DateTimeOriginal from EXIF data.

    Returns ISO datetime string like: 2026-01-20T19:06:54
    Returns None if no EXIF timestamp found.
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()

            if not exif_data:
                return None

            # Look for DateTimeOriginal (preferred) or DateTime
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal' and value:
                    # Format: "2026:01:20 19:06:54" -> "2026-01-20T19:06:54"
                    return value.replace(':', '-', 2).replace(' ', 'T')

            # Fallback to DateTime
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTime' and value:
                    return value.replace(':', '-', 2).replace(' ', 'T')

    except Exception as e:
        print(f"  EXIF read error: {e}")

    return None


def main():
    from r2_storage import R2Storage
    from supabase_rest import get_client

    print("=== Fix Cloud Photo Timestamps ===\n")

    # Connect to services
    r2 = R2Storage()
    if not r2.is_configured():
        print("ERROR: R2 not configured")
        return

    client = get_client()
    if not client:
        print("ERROR: Supabase not configured")
        return

    # Get all photos from Supabase
    print("Fetching photos from Supabase...")
    result = client.table("photos_sync").select("file_hash,date_taken,original_name").execute(fetch_all=True)
    photos = result.data
    print(f"Found {len(photos)} photos\n")

    # Identify photos with suspicious timestamps (filename-based patterns)
    # These have timestamps that look like they came from the filename parsing
    suspicious = []
    for photo in photos:
        date_taken = photo.get("date_taken", "")
        original_name = photo.get("original_name", "")

        # If the date_taken looks like it was derived from filename (has specific patterns)
        # or if the original_name has the timestamp pattern but date_taken doesn't match
        if original_name and "T" in original_name and "_" in original_name:
            # Filename has timestamp pattern - these might have wrong dates
            suspicious.append(photo)

    print(f"Found {len(suspicious)} photos with filename-style names (may have wrong timestamps)")

    if not suspicious:
        print("No suspicious timestamps found.")
        return

    # Ask user to proceed
    response = input(f"\nDownload and check {len(suspicious)} photos? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    fixed = 0
    no_exif = 0
    same = 0
    errors = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for i, photo in enumerate(suspicious):
            file_hash = photo["file_hash"]
            old_date = photo.get("date_taken", "")
            original_name = photo.get("original_name", "")

            print(f"\n[{i+1}/{len(suspicious)}] {original_name}")
            print(f"  Current timestamp: {old_date}")

            # Download photo
            r2_key = f"photos/{file_hash}.jpg"
            local_path = tmp_path / f"{file_hash}.jpg"

            if not r2.download_photo(r2_key, local_path):
                # Try .jpeg
                r2_key = f"photos/{file_hash}.jpeg"
                if not r2.download_photo(r2_key, local_path):
                    print(f"  ERROR: Could not download from R2")
                    errors += 1
                    continue

            # Read EXIF
            exif_date = get_exif_datetime(local_path)

            if not exif_date:
                print(f"  No EXIF timestamp found")
                no_exif += 1
                continue

            print(f"  EXIF timestamp: {exif_date}")

            if exif_date == old_date:
                print(f"  Already correct")
                same += 1
                continue

            # Update Supabase
            print(f"  Updating to: {exif_date}")
            try:
                update_result = client.table("photos_sync").upsert(
                    [{"file_hash": file_hash, "date_taken": exif_date}],
                    on_conflict="file_hash"
                ).execute()
                fixed += 1
            except Exception as e:
                print(f"  ERROR updating Supabase: {e}")
                errors += 1

            # Clean up
            if local_path.exists():
                local_path.unlink()

    print(f"\n=== Summary ===")
    print(f"Fixed: {fixed}")
    print(f"Already correct: {same}")
    print(f"No EXIF data: {no_exif}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
