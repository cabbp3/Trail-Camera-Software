"""
Windows Database Fix Script

Run this on the Windows computer to:
1. Diagnose database issues
2. Remove accidentally imported thumbnail files
3. Calculate file hashes for cross-computer sync
4. Pull labels from Supabase using hash-based matching

Usage:
    cd "C:\Users\mbroo\OneDrive\Desktop\Trail Camera Software V 1.0"
    python windows_fix.py
"""
import sqlite3
import os
import hashlib
from pathlib import Path


def get_db_path():
    """Get the database path."""
    home = Path.home()
    return home / ".trailcam" / "trailcam.db"


def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    try:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def ensure_file_hash_column(conn):
    """Make sure the file_hash column exists."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(photos)")
    cols = {row[1] for row in cursor.fetchall()}
    if "file_hash" not in cols:
        print("Adding file_hash column to database...")
        cursor.execute("ALTER TABLE photos ADD COLUMN file_hash TEXT")
        conn.commit()


def diagnose(conn):
    """Show diagnostic information about the database."""
    cursor = conn.cursor()

    print("=" * 60)
    print("DATABASE DIAGNOSTICS")
    print("=" * 60)

    # Total photos
    cursor.execute("SELECT COUNT(*) FROM photos")
    total = cursor.fetchone()[0]
    print(f"\nTotal photos in database: {total}")

    # Check for orphaned entries (file doesn't exist)
    cursor.execute("SELECT id, file_path, original_name FROM photos")
    orphaned = []
    for row in cursor.fetchall():
        if not os.path.exists(row[1]):
            orphaned.append(row)

    print(f"\n--- Orphaned entries (file missing): {len(orphaned)} ---")
    if orphaned:
        print("  These are database entries where the photo file no longer exists.")
        print("  Examples:")
        for row in orphaned[:5]:
            print(f"    {row[2]} -> {row[1]}")
        if len(orphaned) > 5:
            print(f"    ... and {len(orphaned) - 5} more")

    # Sample of original_name values
    print("\n--- Sample original_name values ---")
    cursor.execute("SELECT original_name, date_taken, camera_model FROM photos LIMIT 10")
    for row in cursor.fetchall():
        print(f"  {row[0]} | {row[1]} | {row[2]}")

    # Check for thumbnail files
    cursor.execute("SELECT COUNT(*) FROM photos WHERE original_name LIKE '%_thumb%'")
    thumb_count = cursor.fetchone()[0]
    print(f"\n--- Thumbnail files detected: {thumb_count} ---")

    if thumb_count > 0:
        cursor.execute("SELECT original_name FROM photos WHERE original_name LIKE '%_thumb%' LIMIT 5")
        print("  Examples:")
        for row in cursor.fetchall():
            print(f"    {row[0]}")

    # Check file hash status
    cursor.execute("SELECT COUNT(*) FROM photos WHERE file_hash IS NOT NULL")
    with_hash = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM photos WHERE file_hash IS NULL")
    without_hash = cursor.fetchone()[0]
    print(f"\n--- File hash status ---")
    print(f"  Photos with hash: {with_hash}")
    print(f"  Photos without hash: {without_hash}")

    # Tags count
    cursor.execute("SELECT COUNT(*) FROM tags")
    tags = cursor.fetchone()[0]
    print(f"\n--- Total tags in database: {tags} ---")

    # Tag distribution
    cursor.execute("""
        SELECT tag_name, COUNT(*) as cnt
        FROM tags
        GROUP BY tag_name
        ORDER BY cnt DESC
        LIMIT 10
    """)
    print("\nTop tags:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    print("\n" + "=" * 60)
    return thumb_count, len(orphaned)


def remove_orphans(conn):
    """Remove database entries where the photo file no longer exists."""
    cursor = conn.cursor()

    # Find orphaned entries
    cursor.execute("SELECT id, file_path, original_name FROM photos")
    orphaned = []
    for row in cursor.fetchall():
        if not os.path.exists(row[1]):
            orphaned.append(row)

    if not orphaned:
        print("No orphaned entries to remove.")
        return 0

    print(f"\nFound {len(orphaned)} orphaned database entries (files deleted):")
    for o in orphaned[:5]:
        print(f"  {o[2]}")
    if len(orphaned) > 5:
        print(f"  ... and {len(orphaned) - 5} more")

    confirm = input("\nRemove these orphaned entries from the database? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled.")
        return 0

    orphan_ids = [o[0] for o in orphaned]

    # Delete in batches to avoid SQL issues
    for i in range(0, len(orphan_ids), 100):
        batch = orphan_ids[i:i+100]
        placeholders = ','.join('?' * len(batch))

        # Delete related records
        cursor.execute(f"DELETE FROM tags WHERE photo_id IN ({placeholders})", batch)
        cursor.execute(f"DELETE FROM deer_metadata WHERE photo_id IN ({placeholders})", batch)
        try:
            cursor.execute(f"DELETE FROM deer_additional WHERE photo_id IN ({placeholders})", batch)
        except:
            pass
        try:
            cursor.execute(f"DELETE FROM annotation_boxes WHERE photo_id IN ({placeholders})", batch)
        except:
            pass
        # Delete photos
        cursor.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", batch)

    conn.commit()
    print(f"Removed {len(orphaned)} orphaned entries from database.")
    return len(orphaned)


def remove_thumbnails(conn):
    """Remove thumbnail files from the database."""
    cursor = conn.cursor()

    # Find thumbnail photos
    cursor.execute("""
        SELECT id, file_path, original_name FROM photos
        WHERE original_name LIKE '%_thumb%'
    """)
    thumbs = cursor.fetchall()

    if not thumbs:
        print("No thumbnail files to remove.")
        return 0

    print(f"\nFound {len(thumbs)} thumbnail files to remove:")
    for t in thumbs[:5]:
        print(f"  ID {t[0]}: {t[1]}")
    if len(thumbs) > 5:
        print(f"  ... and {len(thumbs) - 5} more")

    confirm = input("\nRemove these from the database? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled.")
        return 0

    thumb_ids = [t[0] for t in thumbs]
    placeholders = ','.join('?' * len(thumb_ids))

    # Delete tags first (foreign key)
    cursor.execute(f"DELETE FROM tags WHERE photo_id IN ({placeholders})", thumb_ids)
    cursor.execute(f"DELETE FROM deer_metadata WHERE photo_id IN ({placeholders})", thumb_ids)
    cursor.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", thumb_ids)

    conn.commit()
    print(f"Removed {len(thumbs)} thumbnail entries from database.")
    return len(thumbs)


def calculate_hashes(conn):
    """Calculate file hashes for photos that don't have one."""
    cursor = conn.cursor()

    cursor.execute("SELECT id, file_path FROM photos WHERE file_hash IS NULL")
    rows = cursor.fetchall()

    if not rows:
        print("All photos already have file hashes.")
        return 0

    print(f"\nCalculating file hashes for {len(rows)} photos...")

    count = 0
    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(rows)}")

        file_path = row[1]
        if file_path and os.path.exists(file_path):
            file_hash = calculate_file_hash(file_path)
            if file_hash:
                cursor.execute("UPDATE photos SET file_hash = ? WHERE id = ?",
                               (file_hash, row[0]))
                count += 1

    conn.commit()
    print(f"Calculated {count} file hashes.")
    return count


def pull_from_supabase(conn):
    """Pull labels from Supabase using hash-based matching."""
    print("\n" + "=" * 60)
    print("PULLING LABELS FROM SUPABASE")
    print("=" * 60)

    try:
        import supabase_rest
        import json

        # Read credentials from settings
        settings_path = Path.home() / ".trailcam" / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
            url = settings.get("supabase_url", "")
            key = settings.get("supabase_key", "")
        else:
            url = ""
            key = ""

        if not url or not key:
            print("\nSupabase credentials not found in settings.")
            print("Please run the app and configure Supabase first, or enter manually:")
            url = input("Supabase URL: ").strip()
            key = input("Supabase Key: ").strip()

        if not url or not key:
            print("Credentials required. Skipping sync.")
            return 0

        print(f"\nConnecting to Supabase...")
        client = supabase_rest.create_client(url, key)

        if not client.test_connection():
            print("Failed to connect to Supabase. Check credentials.")
            return 0

        print("Connected!")

        cursor = conn.cursor()

        # Build a lookup of local photos by file_hash
        print("\nBuilding local hash lookup...")
        cursor.execute("SELECT id, file_hash FROM photos WHERE file_hash IS NOT NULL")
        local_by_hash = {row[1]: row[0] for row in cursor.fetchall()}
        print(f"  Local photos with hashes: {len(local_by_hash)}")

        # Pull tags from Supabase
        print("\nPulling tags from Supabase...")
        result = client.table("tags").select("*").execute()

        if result.error:
            print(f"Error: {result.error}")
            return 0

        print(f"  Found {len(result.data)} tags in Supabase")

        tags_applied = 0
        tags_matched_by_hash = 0
        for row in result.data:
            file_hash = row.get("file_hash")
            photo_id = None

            # Try to match by file_hash
            if file_hash and file_hash in local_by_hash:
                photo_id = local_by_hash[file_hash]
                tags_matched_by_hash += 1

            if photo_id:
                cursor.execute("""
                    INSERT OR IGNORE INTO tags (photo_id, tag_name) VALUES (?, ?)
                """, (photo_id, row["tag_name"]))
                if cursor.rowcount > 0:
                    tags_applied += 1

        print(f"  Matched by file_hash: {tags_matched_by_hash}")
        print(f"  Tags applied: {tags_applied}")

        # Pull deer_metadata
        print("\nPulling deer metadata from Supabase...")
        result = client.table("deer_metadata").select("*").execute()

        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"  Found {len(result.data)} deer metadata entries in Supabase")

            deer_applied = 0
            for row in result.data:
                file_hash = row.get("file_hash")
                photo_id = None

                if file_hash and file_hash in local_by_hash:
                    photo_id = local_by_hash[file_hash]

                if photo_id and row.get("deer_id"):
                    cursor.execute("""
                        INSERT OR REPLACE INTO deer_metadata
                        (photo_id, deer_id, age_class, left_points_min, left_points_max,
                         right_points_min, right_points_max)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (photo_id, row.get("deer_id"), row.get("age_class"),
                          row.get("left_points_min"), row.get("left_points_max"),
                          row.get("right_points_min"), row.get("right_points_max")))
                    if cursor.rowcount > 0:
                        deer_applied += 1

            print(f"  Deer metadata applied: {deer_applied}")

        conn.commit()

        # Show final tag counts
        print("\n--- Current tag counts in local database ---")
        cursor.execute("""
            SELECT tag_name, COUNT(*) as cnt FROM tags
            GROUP BY tag_name ORDER BY cnt DESC LIMIT 10
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")

        return tags_applied

    except ImportError:
        print("supabase_rest.py not found. Make sure you're in the right directory.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    print("\n" + "=" * 60)
    print("TRAIL CAMERA SOFTWARE - WINDOWS DATABASE FIX")
    print("=" * 60)

    db_path = get_db_path()
    print(f"\nDatabase path: {db_path}")

    if not db_path.exists():
        print("Database not found! Run the app first to create it.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Ensure file_hash column exists
        ensure_file_hash_column(conn)

        # Step 1: Diagnose
        thumb_count, orphan_count = diagnose(conn)

        # Step 2: Remove orphaned entries (files that were deleted)
        if orphan_count > 0:
            print("\n" + "-" * 60)
            remove_orphans(conn)

        # Step 3: Remove thumbnails if found
        if thumb_count > 0:
            print("\n" + "-" * 60)
            remove_thumbnails(conn)

        # Step 4: Calculate file hashes
        print("\n" + "-" * 60)
        calculate_hashes(conn)

        # Step 5: Pull from Supabase using hash matching
        print("\n" + "-" * 60)
        pull_from_supabase(conn)

        print("\n" + "=" * 60)
        print("DONE!")
        print("=" * 60)
        print("\nThe sync now uses file content hashes to match photos.")
        print("This works even when filenames differ between computers.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
