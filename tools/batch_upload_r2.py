#!/usr/bin/env python3
"""
Batch Upload to Cloudflare R2

Uploads photos from the local library to R2 cloud storage.
Uses file_hash for filenames (deduplication + security).

Usage:
    python tools/batch_upload_r2.py --username brooke --thumbnails-only
    python tools/batch_upload_r2.py --username brooke --full-photos
    python tools/batch_upload_r2.py --username brooke --both

Options:
    --username      Your username (required)
    --thumbnails-only   Upload only thumbnails (~90MB total)
    --full-photos   Upload full resolution photos (~5GB)
    --both          Upload both thumbnails and full photos
    --limit N       Only upload first N photos (for testing)

Note: Automatically skips files that already exist in R2 (by hash).
"""

import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from r2_storage import R2Storage
from database import TrailCamDatabase


def format_size(bytes_size):
    """Format bytes as human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def format_time(seconds):
    """Format seconds as human readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m"


def main():
    parser = argparse.ArgumentParser(description='Batch upload photos to R2')
    parser.add_argument('--username', required=True, help='Your username')
    parser.add_argument('--thumbnails-only', action='store_true', help='Upload only thumbnails')
    parser.add_argument('--full-photos', action='store_true', help='Upload full resolution photos')
    parser.add_argument('--both', action='store_true', help='Upload both thumbnails and full photos')
    parser.add_argument('--limit', type=int, help='Limit number of photos to upload')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')

    args = parser.parse_args()

    # Validate mode
    if not (args.thumbnails_only or args.full_photos or args.both):
        print("Error: Specify --thumbnails-only, --full-photos, or --both")
        sys.exit(1)

    upload_thumbs = args.thumbnails_only or args.both
    upload_full = args.full_photos or args.both

    print(f"=== R2 Batch Upload ===")
    print(f"Username: {args.username}")
    print(f"Upload thumbnails: {upload_thumbs}")
    print(f"Upload full photos: {upload_full}")
    if args.limit:
        print(f"Limit: {args.limit} photos")
    print()

    # Initialize
    storage = R2Storage()
    if not storage.is_configured():
        print("Error: R2 not configured. Check ~/.trailcam/r2_config.json")
        sys.exit(1)

    db = TrailCamDatabase()

    # Get all photos (filter out archived)
    all_photos = db.get_all_photos()
    photos = [p for p in all_photos if not p.get('archived', 0)]
    if args.limit:
        photos = photos[:args.limit]

    print(f"Found {len(photos)} photos to process")

    # Calculate estimated size
    total_thumb_size = 0
    total_photo_size = 0

    for photo in photos:
        if upload_thumbs:
            thumb_path_str = photo.get('thumbnail_path')
            if thumb_path_str:
                thumb_path = Path(thumb_path_str)
                if thumb_path.exists():
                    total_thumb_size += thumb_path.stat().st_size

        if upload_full:
            file_path_str = photo.get('file_path')
            if file_path_str:
                photo_path = Path(file_path_str)
                if photo_path.exists():
                    total_photo_size += photo_path.stat().st_size

    print(f"Estimated upload size:")
    if upload_thumbs:
        print(f"  Thumbnails: {format_size(total_thumb_size)}")
    if upload_full:
        print(f"  Full photos: {format_size(total_photo_size)}")
    print(f"  Total: {format_size(total_thumb_size + total_photo_size)}")
    print()

    # Confirm
    if not args.yes:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            sys.exit(0)

    print()

    # Upload
    uploaded_thumbs = 0
    uploaded_photos = 0
    skipped = 0
    errors = 0
    bytes_uploaded = 0
    start_time = time.time()

    for i, photo in enumerate(photos):
        file_hash = photo.get('file_hash')
        if not file_hash:
            errors += 1
            continue

        progress = f"[{i+1}/{len(photos)}]"

        # Upload thumbnail (using hash-based filename)
        if upload_thumbs:
            thumb_path_str = photo.get('thumbnail_path')
            if thumb_path_str:
                thumb_path = Path(thumb_path_str)
                if thumb_path.exists():
                    r2_key = f"users/{args.username}/thumbnails/{file_hash}_thumb.jpg"

                    # Always check if exists (deduplication)
                    if storage.check_exists(r2_key):
                        skipped += 1
                    else:
                        result = storage.upload_file(thumb_path, r2_key)
                        if result:
                            uploaded_thumbs += 1
                            bytes_uploaded += thumb_path.stat().st_size
                        else:
                            errors += 1

        # Upload full photo (using hash-based filename)
        if upload_full:
            file_path_str = photo.get('file_path')
            if file_path_str:
                photo_path = Path(file_path_str)
                if photo_path.exists():
                    r2_key = f"users/{args.username}/photos/{file_hash}.jpg"

                    # Always check if exists (deduplication)
                    if storage.check_exists(r2_key):
                        skipped += 1
                    else:
                        result = storage.upload_file(photo_path, r2_key)
                        if result:
                            uploaded_photos += 1
                            bytes_uploaded += photo_path.stat().st_size
                        else:
                            errors += 1

        # Progress update every 10 photos
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = bytes_uploaded / elapsed if elapsed > 0 else 0
            eta = (total_thumb_size + total_photo_size - bytes_uploaded) / rate if rate > 0 else 0
            print(f"{progress} Uploaded: {format_size(bytes_uploaded)} | "
                  f"Rate: {format_size(rate)}/s | ETA: {format_time(eta)}")

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=== Upload Complete ===")
    print(f"Thumbnails uploaded: {uploaded_thumbs}")
    print(f"Full photos uploaded: {uploaded_photos}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total uploaded: {format_size(bytes_uploaded)}")
    print(f"Time: {format_time(elapsed)}")
    if elapsed > 0:
        print(f"Average rate: {format_size(bytes_uploaded/elapsed)}/s")

    # Check bucket stats
    stats = storage.get_bucket_stats()
    print()
    print(f"R2 bucket now: {stats['object_count']} objects, {stats['total_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
