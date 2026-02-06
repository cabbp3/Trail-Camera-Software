#!/usr/bin/env python3
"""
One-time script to update existing photos with full datetime from their filenames.
"""

import os
import re
import requests


def get_env(name: str) -> str:
    """Get required environment variable."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def parse_cuddelink_datetime(filename: str):
    """Parse datetime from CuddeLink filename.

    Filename format: 2026-01-20T16_01_52.430510-8.jpeg
    Returns ISO datetime string like: 2026-01-20T16:01:52
    """
    match = re.match(r'(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})', filename)
    if match:
        date_part = match.group(1)
        hour = match.group(2)
        minute = match.group(3)
        second = match.group(4)
        return f"{date_part}T{hour}:{minute}:{second}"
    return None


def main():
    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_KEY")

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
    }

    # Fetch all photos
    print("[Backfill] Fetching photos from Supabase...")
    all_photos = []
    offset = 0
    page_size = 1000

    while True:
        resp = requests.get(
            f"{supabase_url}/rest/v1/photos_sync?select=file_hash,photo_key,date_taken&limit={page_size}&offset={offset}",
            headers=headers,
            timeout=30
        )
        if resp.status_code != 200:
            print(f"[Backfill] Error fetching photos: {resp.status_code}")
            break

        data = resp.json()
        all_photos.extend(data)

        if len(data) < page_size:
            break
        offset += page_size

    print(f"[Backfill] Found {len(all_photos)} photos")

    # Find photos that need datetime update (date_taken doesn't have 'T' or is just a date)
    to_update = []
    for photo in all_photos:
        date_taken = photo.get("date_taken", "")
        photo_key = photo.get("photo_key", "")
        file_hash = photo.get("file_hash")

        # Skip if already has time component
        if "T" in date_taken and ":" in date_taken:
            continue

        # Extract filename from photo_key (format: YYYY/MM/DD/filename.jpg)
        if "/" in photo_key:
            filename = photo_key.split("/")[-1]
        else:
            filename = photo_key

        # Try to parse datetime from filename
        new_datetime = parse_cuddelink_datetime(filename)
        if new_datetime and file_hash:
            to_update.append({
                "file_hash": file_hash,
                "old_date": date_taken,
                "new_datetime": new_datetime,
                "filename": filename,
            })

    print(f"[Backfill] {len(to_update)} photos need datetime update")

    if not to_update:
        print("[Backfill] All photos already have proper datetimes!")
        return

    # Update each photo
    updated = 0
    failed = 0

    for i, photo in enumerate(to_update, 1):
        try:
            resp = requests.patch(
                f"{supabase_url}/rest/v1/photos_sync?file_hash=eq.{photo['file_hash']}",
                headers=headers,
                json={
                    "date_taken": photo["new_datetime"],
                    "original_name": photo["filename"],
                },
                timeout=30
            )
            if resp.status_code < 400:
                print(f"[{i}/{len(to_update)}] Updated {photo['file_hash'][:12]}... {photo['old_date']} -> {photo['new_datetime']}")
                updated += 1
            else:
                print(f"[{i}/{len(to_update)}] Failed {photo['file_hash'][:12]}...: {resp.status_code}")
                failed += 1
        except Exception as e:
            print(f"[{i}/{len(to_update)}] Failed {photo['file_hash'][:12]}...: {e}")
            failed += 1

    print(f"\n[Backfill] Complete! Updated: {updated}, Failed: {failed}")


if __name__ == "__main__":
    main()
