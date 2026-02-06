#!/usr/bin/env python3
"""
Force a full sync to Supabase to fix date_taken values.

This will push ALL photos from local SQLite to Supabase, overwriting
any incorrect date_taken values that were set by the GitHub workflow.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import TrailCamDatabase
from supabase_rest import get_client

def main():
    print("=" * 60)
    print("Force Full Sync to Supabase")
    print("=" * 60)
    print()
    print("This will push ALL photos from your local database to Supabase,")
    print("overwriting any incorrect date_taken values.")
    print()

    # Initialize database
    db = TrailCamDatabase()

    # Get Supabase client
    client = get_client()
    if not client or not client.is_configured():
        print("ERROR: Supabase is not configured!")
        print("Please set up Supabase credentials first.")
        return 1

    # Test connection
    if not client.test_connection():
        print("ERROR: Cannot connect to Supabase!")
        return 1

    print("Connected to Supabase successfully.")
    print()

    # Clear last sync time to force full sync
    print("Clearing last sync time to force full sync...")
    try:
        cursor = db.conn.cursor()
        cursor.execute("DELETE FROM sync_state WHERE key = 'last_push'")
        db.conn.commit()
        print("Last sync time cleared.")
    except Exception as e:
        print(f"Note: Could not clear sync state (may not exist yet): {e}")

    print()
    print("Starting full sync...")
    print("-" * 60)

    def progress_callback(step, total, message):
        print(f"[{step}/{total}] {message}")

    try:
        counts = db.push_to_supabase(client, progress_callback=progress_callback, force_full=True)
        print("-" * 60)
        print()
        print("Sync completed!")
        print(f"  Photos synced: {counts.get('photos', 0)}")
        print(f"  Tags synced: {counts.get('tags', 0)}")
        print(f"  Deer metadata synced: {counts.get('deer_metadata', 0)}")
        print()
        print("The date_taken values in Supabase should now match your local database.")
        print("Android app should show correct times after refreshing.")
        return 0
    except Exception as e:
        print(f"ERROR: Sync failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
