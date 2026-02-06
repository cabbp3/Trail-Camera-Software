#!/usr/bin/env python3
"""
Fix date_taken values in Supabase by directly updating them from local SQLite.
Uses batch updates for speed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from database import TrailCamDatabase
from supabase_rest import _load_config

def main():
    print("Fix date_taken values in Supabase")
    print("=" * 50)

    config = _load_config()
    if not config:
        print("ERROR: Supabase config not found!")
        return 1

    url = config['url']
    key = config.get('key') or config.get('anon_key')

    headers = {
        'apikey': key,
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
    }

    db = TrailCamDatabase()
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT file_hash, date_taken FROM photos
        WHERE file_hash IS NOT NULL AND date_taken IS NOT NULL
    ''')
    local_photos = cursor.fetchall()
    print(f"Found {len(local_photos)} photos to update")

    # Process in batches using concurrent requests
    import concurrent.futures

    def update_one(row):
        file_hash = row['file_hash']
        date_taken = row['date_taken']
        if date_taken and ' ' in date_taken and 'T' not in date_taken:
            date_taken = date_taken.replace(' ', 'T')

        resp = requests.patch(
            f"{url}/rest/v1/photos_sync?file_hash=eq.{file_hash}",
            headers=headers,
            json={'date_taken': date_taken},
            timeout=10
        )
        return resp.status_code in (200, 204)

    updated = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(update_one, row): row for row in local_photos}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if future.result():
                updated += 1
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{len(local_photos)}")

    print(f"\nDone! Updated {updated} photos.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
