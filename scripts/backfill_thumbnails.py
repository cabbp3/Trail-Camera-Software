#!/usr/bin/env python3
"""
One-time script to generate thumbnails for photos that were uploaded without them.
Lists all photos in R2, checks for missing thumbnails, and generates them.
"""

import io
import os
import tempfile
from pathlib import Path

import boto3
from botocore.config import Config
from PIL import Image


def get_env(name: str) -> str:
    """Get required environment variable."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def create_thumbnail(image_bytes: bytes, max_size: int = 400) -> bytes:
    """Create a thumbnail from image bytes. Returns JPEG bytes."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=75, optimize=True)
        return buffer.getvalue()


def main():
    # Load config from environment
    endpoint = get_env("R2_ENDPOINT")
    access_key = get_env("R2_ACCESS_KEY")
    secret_key = get_env("R2_SECRET_KEY")
    bucket = get_env("R2_BUCKET")

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    # List all photos
    print("[Backfill] Listing photos in R2...")
    photos = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix='photos/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.jpg'):
                photos.append(key)

    print(f"[Backfill] Found {len(photos)} photos")

    # List all thumbnails
    print("[Backfill] Listing existing thumbnails...")
    existing_thumbs = set()
    for page in paginator.paginate(Bucket=bucket, Prefix='thumbnails/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Extract hash from thumbnails/{hash}_thumb.jpg
            if key.endswith('_thumb.jpg'):
                hash_part = key.replace('thumbnails/', '').replace('_thumb.jpg', '')
                existing_thumbs.add(hash_part)

    print(f"[Backfill] Found {len(existing_thumbs)} existing thumbnails")

    # Find photos missing thumbnails
    missing = []
    for photo_key in photos:
        # Extract hash from photos/{hash}.jpg
        hash_part = photo_key.replace('photos/', '').replace('.jpg', '')
        if hash_part not in existing_thumbs:
            missing.append((photo_key, hash_part))

    print(f"[Backfill] {len(missing)} photos missing thumbnails")

    if not missing:
        print("[Backfill] All photos have thumbnails!")
        return

    # Generate and upload missing thumbnails
    created = 0
    failed = 0

    for i, (photo_key, hash_part) in enumerate(missing, 1):
        try:
            print(f"[{i}/{len(missing)}] Processing {hash_part}...")

            # Download photo
            response = s3_client.get_object(Bucket=bucket, Key=photo_key)
            photo_bytes = response['Body'].read()

            # Create thumbnail
            thumb_bytes = create_thumbnail(photo_bytes)

            # Upload thumbnail
            thumb_key = f"thumbnails/{hash_part}_thumb.jpg"
            s3_client.put_object(
                Bucket=bucket,
                Key=thumb_key,
                Body=thumb_bytes,
                ContentType='image/jpeg'
            )

            created += 1
            print(f"  Created thumbnail: {thumb_key}")

        except Exception as e:
            failed += 1
            print(f"  Failed: {e}")

    print(f"\n[Backfill] Complete! Created: {created}, Failed: {failed}")


if __name__ == "__main__":
    main()
