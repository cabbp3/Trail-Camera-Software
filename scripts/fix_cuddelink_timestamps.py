#!/usr/bin/env python3
"""
Fix CuddeLink photo timestamps using EXIF DateTimeOriginal.

The CuddeLink filename timestamps are when photos were uploaded to their servers,
not when the trail camera actually captured the image. This script:
1. Fetches CuddeLink photos from Supabase
2. Downloads each from R2
3. Reads the EXIF DateTimeOriginal (actual capture time)
4. Updates Supabase with the correct date_taken
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import boto3
import requests
from botocore.config import Config
from PIL import Image
from PIL.ExifTags import TAGS


def get_r2_client():
    """Create R2 client from config."""
    config_path = Path.home() / ".trailcam" / "r2_config.json"
    with open(config_path) as f:
        config = json.load(f)

    return boto3.client(
        's3',
        endpoint_url=config['endpoint_url'],
        aws_access_key_id=config['access_key_id'],
        aws_secret_access_key=config['secret_access_key'],
        config=Config(signature_version='s3v4'),
        region_name='auto'
    ), config['bucket_name']


def get_exif_datetime(image_path: Path):
    """Read DateTimeOriginal from EXIF data.

    Returns ISO datetime string like: 2026-01-20T19:06:54
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if not exif_data:
            return None

        # Look for DateTimeOriginal (tag 36867) or DateTime (tag 306)
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal' and value:
                # Format: "2026:01:20 19:06:54" -> "2026-01-20T19:06:54"
                return value.replace(':', '-', 2).replace(' ', 'T')

        # Fallback to DateTime if DateTimeOriginal not found
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTime' and value:
                return value.replace(':', '-', 2).replace(' ', 'T')

    except Exception as e:
        print(f"  EXIF error: {e}")

    return None


def main():
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser(description='Fix CuddeLink timestamps from EXIF')
    parser.add_argument('--limit', type=int, default=0, help='Max photos to process (0=all)')
    parser.add_argument('--local-only', action='store_true', help='Only fix local DB, not cloud')
    args = parser.parse_args()

    # Supabase config
    supabase_url = 'https://iwvehmthbjcvdqjqxtty.supabase.co'
    supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml3dmVobXRoYmpjdmRxanF4dHR5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjYyMDI0NDQsImV4cCI6MjA4MTc3ODQ0NH0._z6WAfUBP_Qda0IcjTS_LEI_J7r147BrmSib3dyneLE'

    headers = {
        'apikey': supabase_key,
        'Authorization': f'Bearer {supabase_key}',
        'Content-Type': 'application/json',
        'Prefer': 'count=exact',
    }

    # Get R2 client
    s3_client, bucket = get_r2_client()

    # Connect to local database
    db_path = Path.home() / '.trailcam' / 'trailcam.db'
    local_db = sqlite3.connect(str(db_path))
    local_db.row_factory = sqlite3.Row

    # Fetch ALL CuddeLink photos with pagination
    print("Fetching CuddeLink photos from Supabase...")
    all_photos = []
    offset = 0
    page_size = 1000

    while True:
        resp = requests.get(
            f'{supabase_url}/rest/v1/photos_sync?select=file_hash,original_name,date_taken&original_name=like.202*T*&order=date_taken.desc&offset={offset}&limit={page_size}',
            headers=headers
        )
        page = resp.json()
        if not page:
            break
        all_photos.extend(page)
        print(f"  Fetched {len(all_photos)} photos...")
        offset += page_size
        if len(page) < page_size:
            break

    photos = all_photos
    if args.limit > 0:
        photos = photos[:args.limit]

    print(f"Found {len(all_photos)} CuddeLink photos, processing {len(photos)}\n")

    updated = 0
    failed = 0
    skipped = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, photo in enumerate(photos, 1):
            file_hash = photo['file_hash']
            original_name = photo['original_name']
            current_date = photo['date_taken']

            print(f"[{i}/{len(photos)}] {file_hash[:12]}...")
            print(f"  Current: {current_date}")

            # Download from R2
            try:
                local_path = Path(tmp_dir) / f"{file_hash}.jpg"
                s3_client.download_file(bucket, f"photos/{file_hash}.jpg", str(local_path))
            except Exception as e:
                print(f"  Download failed: {e}")
                failed += 1
                continue

            # Get EXIF timestamp
            exif_time = get_exif_datetime(local_path)

            if not exif_time:
                print(f"  No EXIF timestamp - skipping")
                skipped += 1
                local_path.unlink(missing_ok=True)
                continue

            print(f"  EXIF: {exif_time}")

            # Check if different (compare without seconds for tolerance)
            current_prefix = current_date[:16] if current_date else ""
            exif_prefix = exif_time[:16] if exif_time else ""

            if current_prefix == exif_prefix:
                print(f"  Already correct - skipping")
                skipped += 1
                local_path.unlink(missing_ok=True)
                continue

            # Update Supabase (unless --local-only)
            if not args.local_only:
                try:
                    resp = requests.patch(
                        f'{supabase_url}/rest/v1/photos_sync?file_hash=eq.{file_hash}',
                        headers=headers,
                        json={'date_taken': exif_time}
                    )
                    if resp.status_code < 400:
                        print(f"  CLOUD UPDATED: {current_date} -> {exif_time}")
                    else:
                        print(f"  Cloud update failed: {resp.status_code}")
                except Exception as e:
                    print(f"  Cloud update error: {e}")

            # Update local database
            try:
                cursor = local_db.cursor()
                cursor.execute(
                    "UPDATE photos SET date_taken = ? WHERE file_hash = ?",
                    (exif_time, file_hash)
                )
                if cursor.rowcount > 0:
                    print(f"  LOCAL UPDATED")
                local_db.commit()
            except Exception as e:
                print(f"  Local update error: {e}")

            updated += 1

            # Clean up
            local_path.unlink(missing_ok=True)

    local_db.close()

    print(f"\n{'='*40}")
    print(f"Updated: {updated}")
    print(f"Skipped (already correct or no EXIF): {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
