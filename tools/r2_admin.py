#!/usr/bin/env python3
"""
R2 Admin Tool

View and manage cloud storage contents.
Shows all users, their uploads, and storage usage.

Usage:
    python tools/r2_admin.py              # Show summary
    python tools/r2_admin.py --users      # List all users
    python tools/r2_admin.py --user brooke  # Show specific user's files
    python tools/r2_admin.py --delete users/brooke/photos/test.jpg  # Delete a file
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from r2_storage import R2Storage


def format_size(bytes_size):
    """Format bytes as human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def get_all_objects(storage):
    """Get all objects in the bucket with metadata."""
    if not storage.client:
        return []

    try:
        objects = []
        paginator = storage.client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=storage.bucket_name):
            for obj in page.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'modified': obj['LastModified']
                })

        return objects
    except Exception as e:
        print(f"Error listing objects: {e}")
        return []


def show_summary(storage):
    """Show overall bucket summary."""
    print("=" * 60)
    print("CLOUDFLARE R2 ADMIN - BUCKET SUMMARY")
    print("=" * 60)

    objects = get_all_objects(storage)

    if not objects:
        print("\nBucket is empty.")
        return

    # Analyze by user
    users = defaultdict(lambda: {'photos': 0, 'thumbnails': 0, 'other': 0, 'size': 0})

    for obj in objects:
        key = obj['key']
        size = obj['size']

        parts = key.split('/')
        if len(parts) >= 2 and parts[0] == 'users':
            username = parts[1]
            users[username]['size'] += size

            if 'photos/' in key:
                users[username]['photos'] += 1
            elif 'thumbnails/' in key:
                users[username]['thumbnails'] += 1
            else:
                users[username]['other'] += 1
        else:
            users['_root']['size'] += size
            users['_root']['other'] += 1

    total_size = sum(obj['size'] for obj in objects)

    print(f"\nTotal objects: {len(objects)}")
    print(f"Total size: {format_size(total_size)}")
    print(f"Free tier remaining: {format_size(10 * 1024 * 1024 * 1024 - total_size)}")

    print(f"\nUsers: {len([u for u in users if u != '_root'])}")
    print("-" * 60)
    print(f"{'Username':<20} {'Photos':>8} {'Thumbs':>8} {'Size':>12}")
    print("-" * 60)

    for username in sorted(users.keys()):
        if username == '_root':
            continue
        data = users[username]
        print(f"{username:<20} {data['photos']:>8} {data['thumbnails']:>8} {format_size(data['size']):>12}")

    if users['_root']['size'] > 0:
        print(f"{'(root files)':<20} {'-':>8} {'-':>8} {format_size(users['_root']['size']):>12}")

    print("-" * 60)


def list_users(storage):
    """List all users."""
    print("Users in bucket:")
    print("-" * 40)

    objects = get_all_objects(storage)
    users = set()

    for obj in objects:
        parts = obj['key'].split('/')
        if len(parts) >= 2 and parts[0] == 'users':
            users.add(parts[1])

    if not users:
        print("No users found.")
        return

    for user in sorted(users):
        print(f"  - {user}")

    print(f"\nTotal: {len(users)} users")


def show_user(storage, username):
    """Show details for a specific user."""
    print(f"Files for user: {username}")
    print("=" * 60)

    objects = get_all_objects(storage)
    user_objects = [o for o in objects if o['key'].startswith(f"users/{username}/")]

    if not user_objects:
        print("No files found for this user.")
        return

    photos = [o for o in user_objects if '/photos/' in o['key']]
    thumbs = [o for o in user_objects if '/thumbnails/' in o['key']]

    print(f"\nPhotos: {len(photos)}")
    for obj in photos[:10]:  # Show first 10
        print(f"  {obj['key'].split('/')[-1]} ({format_size(obj['size'])})")
    if len(photos) > 10:
        print(f"  ... and {len(photos) - 10} more")

    print(f"\nThumbnails: {len(thumbs)}")
    if thumbs:
        print(f"  {len(thumbs)} thumbnail files")

    total = sum(o['size'] for o in user_objects)
    print(f"\nTotal size: {format_size(total)}")


def delete_file(storage, key):
    """Delete a file from R2."""
    if not storage.check_exists(key):
        print(f"File not found: {key}")
        return

    confirm = input(f"Delete {key}? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return

    if storage.delete_photo(key):
        print(f"Deleted: {key}")
    else:
        print("Delete failed.")


def main():
    parser = argparse.ArgumentParser(description='R2 Admin Tool')
    parser.add_argument('--users', action='store_true', help='List all users')
    parser.add_argument('--user', type=str, help='Show specific user details')
    parser.add_argument('--delete', type=str, help='Delete a file by key')

    args = parser.parse_args()

    storage = R2Storage()
    if not storage.is_configured():
        print("Error: R2 not configured. Check ~/.trailcam/r2_config.json")
        sys.exit(1)

    if args.users:
        list_users(storage)
    elif args.user:
        show_user(storage, args.user)
    elif args.delete:
        delete_file(storage, args.delete)
    else:
        show_summary(storage)


if __name__ == "__main__":
    main()
