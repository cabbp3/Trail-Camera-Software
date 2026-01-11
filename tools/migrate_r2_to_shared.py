#!/usr/bin/env python3
"""
Migrate R2 files from user-specific paths to shared structure.

Moves files from:
  users/brooke/photos/{hash}.jpg -> photos/{hash}.jpg
  users/brooke/thumbnails/{hash}_thumb.jpg -> thumbnails/{hash}_thumb.jpg

This is a server-side copy operation (fast, no download/upload).
"""

import sys
import argparse
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from r2_storage import R2Storage


def migrate_prefix(storage, source_prefix: str, dest_prefix: str, dry_run: bool = True):
    """
    Migrate all files from source_prefix to dest_prefix.

    Args:
        storage: R2Storage instance
        source_prefix: e.g. "users/brooke/photos/"
        dest_prefix: e.g. "photos/"
        dry_run: If True, only print what would be done

    Returns:
        (copied, skipped, errors) counts
    """
    copied = 0
    skipped = 0
    errors = 0

    print(f"\nMigrating: {source_prefix} -> {dest_prefix}")
    print("-" * 60)

    # List all objects with source prefix
    paginator = storage.client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=storage.bucket_name, Prefix=source_prefix):
        for obj in page.get('Contents', []):
            source_key = obj['Key']

            # Extract filename from source key
            filename = source_key.replace(source_prefix, '')
            dest_key = f"{dest_prefix}{filename}"

            # Check if destination already exists
            try:
                storage.client.head_object(Bucket=storage.bucket_name, Key=dest_key)
                # Already exists
                skipped += 1
                if skipped <= 5:
                    print(f"  SKIP (exists): {dest_key}")
                elif skipped == 6:
                    print(f"  ... (skipping further 'exists' messages)")
                continue
            except:
                pass  # Doesn't exist, proceed with copy

            if dry_run:
                print(f"  WOULD COPY: {source_key} -> {dest_key}")
                copied += 1
            else:
                try:
                    # Server-side copy within same bucket
                    storage.client.copy_object(
                        Bucket=storage.bucket_name,
                        CopySource={'Bucket': storage.bucket_name, 'Key': source_key},
                        Key=dest_key
                    )
                    copied += 1
                    if copied <= 20 or copied % 100 == 0:
                        print(f"  COPIED [{copied}]: {filename}")
                except Exception as e:
                    errors += 1
                    print(f"  ERROR: {source_key} - {e}")

    return copied, skipped, errors


def delete_prefix(storage, prefix: str, dry_run: bool = True):
    """Delete all files under a prefix."""
    deleted = 0
    errors = 0

    print(f"\nDeleting: {prefix}")
    print("-" * 60)

    paginator = storage.client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=storage.bucket_name, Prefix=prefix):
        objects = page.get('Contents', [])
        if not objects:
            continue

        if dry_run:
            for obj in objects:
                print(f"  WOULD DELETE: {obj['Key']}")
                deleted += 1
        else:
            # Delete in batches of 1000 (S3 limit)
            keys_to_delete = [{'Key': obj['Key']} for obj in objects]
            try:
                storage.client.delete_objects(
                    Bucket=storage.bucket_name,
                    Delete={'Objects': keys_to_delete}
                )
                deleted += len(keys_to_delete)
                print(f"  DELETED batch of {len(keys_to_delete)} files")
            except Exception as e:
                errors += len(keys_to_delete)
                print(f"  ERROR deleting batch: {e}")

    return deleted, errors


def main():
    parser = argparse.ArgumentParser(description='Migrate R2 files to shared structure')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Only show what would be done (default: True)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually perform the migration')
    parser.add_argument('--delete-old', action='store_true',
                        help='Delete old user-specific files after migration')
    parser.add_argument('--username', default='brooke',
                        help='Username to migrate from (default: brooke)')
    args = parser.parse_args()

    dry_run = not args.execute

    print("=" * 60)
    print("R2 Migration: User-specific -> Shared Structure")
    print("=" * 60)

    if dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***")
        print("Use --execute to actually perform the migration\n")
    else:
        print("\n*** EXECUTE MODE - Files will be copied ***\n")

    storage = R2Storage()
    if not storage.is_configured():
        print("ERROR: R2 storage not configured")
        return 1

    # Define migrations
    migrations = [
        (f"users/{args.username}/thumbnails/", "thumbnails/"),
        (f"users/{args.username}/photos/", "photos/"),
    ]

    total_copied = 0
    total_skipped = 0
    total_errors = 0

    for source, dest in migrations:
        copied, skipped, errors = migrate_prefix(storage, source, dest, dry_run)
        total_copied += copied
        total_skipped += skipped
        total_errors += errors

    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"  Copied:  {total_copied}")
    print(f"  Skipped: {total_skipped} (already exist at destination)")
    print(f"  Errors:  {total_errors}")

    if dry_run:
        print(f"\nTo execute this migration, run:")
        print(f"  python {__file__} --execute")

    # Handle deletion of old files
    if args.delete_old and not dry_run and total_errors == 0:
        print("\n" + "=" * 60)
        print("CLEANUP: Deleting old user-specific files")
        print("=" * 60)

        for source, _ in migrations:
            deleted, del_errors = delete_prefix(storage, source, dry_run=False)
            print(f"  {source}: deleted {deleted}, errors {del_errors}")
    elif args.delete_old and dry_run:
        print("\nWould also delete old files with --delete-old --execute")

    return 0 if total_errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
