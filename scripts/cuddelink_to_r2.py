#!/usr/bin/env python3
"""
CuddeLink to R2 Sync Script

Downloads photos from CuddeLink and uploads them to Cloudflare R2.
Designed to run via GitHub Actions on a schedule.

Environment variables required:
  CUDDELINK_EMAIL, CUDDELINK_PASSWORD
  R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
  SUPABASE_URL, SUPABASE_KEY
"""

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import boto3
import requests
from botocore.config import Config
from PIL import Image

# CuddeLink URLs
BASE_URL = "https://camp.cuddeback.com"
LOGIN_URL = f"{BASE_URL}/Identity/Account/Login"
PHOTOS_URL = f"{BASE_URL}/photos"
APPLY_FILTER_URL = f"{BASE_URL}/photos/applyFilter"
DOWNLOAD_VIEW_URL = f"{BASE_URL}/photos/downloadphotoview"
DOWNLOAD_FILE_URL = f"{BASE_URL}/photos/Download"

# Browser-like headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_env(name: str) -> str:
    """Get required environment variable."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def extract_verification_token(html: str) -> str | None:
    """Extract CSRF token from login page."""
    patterns = [
        r'name="__RequestVerificationToken"[^>]*value="([^"]+)"',
        r"name='__RequestVerificationToken'[^>]*value='([^']+)'",
        r'value="([^"]+)"[^>]*name="__RequestVerificationToken"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_photo_ids(html: str) -> list[str]:
    """Extract photo IDs from photos page HTML."""
    ids = set()
    patterns = [
        r'id="photoId\d+"\s+value="([^"]+)"',
        r'value="([^"]+)"\s+id="photoId\d+"',
        r'data-photo-id=["\']([^"\']+)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE):
            ids.add(match.group(1))
    return list(ids)


def cuddelink_login(session: requests.Session, email: str, password: str) -> None:
    """Login to CuddeLink."""
    print("[CuddeLink] Fetching login page...")
    resp = session.get(LOGIN_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    token = extract_verification_token(resp.text)

    payload = {
        "Input.Email": email,
        "Input.Password": password,
        "Input.RememberMe": "false",
    }
    if token:
        payload["__RequestVerificationToken"] = token

    print("[CuddeLink] Logging in...")
    login_headers = {
        **HEADERS,
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": BASE_URL,
        "Referer": LOGIN_URL,
    }
    resp = session.post(LOGIN_URL, data=payload, headers=login_headers, timeout=20)

    if resp.status_code >= 400:
        raise RuntimeError(f"Login failed: {resp.status_code}")

    # Verify login by accessing photos page
    resp = session.get(PHOTOS_URL, headers=HEADERS, timeout=20)
    if "login" in resp.url.lower():
        raise RuntimeError("Login failed - redirected back to login page")

    print("[CuddeLink] Login successful!")


def apply_date_filter(session: requests.Session, start_date: str, end_date: str) -> None:
    """Apply date filter to photos page."""
    filter_value = f"date;{start_date};{end_date}"
    headers = {
        **HEADERS,
        "Content-Type": "application/x-www-form-urlencoded",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": PHOTOS_URL,
    }
    print(f"[CuddeLink] Applying date filter: {start_date} to {end_date}")
    resp = session.post(APPLY_FILTER_URL, data={"filter": filter_value}, headers=headers, timeout=30)
    resp.raise_for_status()


def fetch_photo_ids(session: requests.Session) -> list[str]:
    """Get photo IDs from current filtered view."""
    resp = session.get(PHOTOS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return extract_photo_ids(resp.text)


def download_photos_zip(session: requests.Session, photo_ids: list[str], dest_dir: Path) -> Path:
    """Request and download zip file of photos."""
    print(f"[CuddeLink] Requesting download for {len(photo_ids)} photos...")

    payload = {"photoIds": json.dumps(photo_ids)}
    headers = {
        **HEADERS,
        "Referer": PHOTOS_URL,
        "Origin": BASE_URL,
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }

    resp = session.post(DOWNLOAD_VIEW_URL, data=payload, headers=headers, timeout=120)
    resp.raise_for_status()

    # Check if direct zip response
    if "application/zip" in resp.headers.get("Content-Type", ""):
        zip_path = dest_dir / "photos.zip"
        zip_path.write_bytes(resp.content)
        return zip_path

    # Otherwise we get a GUID
    guid = resp.text.strip().strip('"')
    if not guid:
        raise RuntimeError("Empty download GUID received")

    # Download zip using GUID
    print(f"[CuddeLink] Downloading zip file...")
    download_url = f"{DOWNLOAD_FILE_URL}?fileGuid={guid}&filename=CuddebackPhotos"
    resp = session.get(download_url, headers=HEADERS, timeout=600, stream=True)
    resp.raise_for_status()

    zip_path = dest_dir / "photos.zip"
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[CuddeLink] Downloaded {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    return zip_path


def extract_images(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract image files from zip."""
    images = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith((".jpg", ".jpeg", ".png")):
                zf.extract(name, dest_dir)
                images.append(dest_dir / name)
    print(f"[CuddeLink] Extracted {len(images)} images")
    return images


def check_photo_exists(supabase_url: str, supabase_key: str, file_hash: str) -> bool:
    """Check if photo hash already exists in Supabase."""
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    url = f"{supabase_url}/rest/v1/photos_sync?file_hash=eq.{file_hash}&select=file_hash&limit=1"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return len(data) > 0
    except Exception as e:
        print(f"[Supabase] Warning: Could not check for duplicate: {e}")

    return False


def create_thumbnail(image_path: Path, max_size: int = 400) -> bytes:
    """Create a thumbnail from an image. Returns JPEG bytes."""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (handles RGBA, etc.)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        # Calculate new size maintaining aspect ratio
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))

        # Resize with high quality
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=75, optimize=True)
        return buffer.getvalue()


def get_exif_datetime(image_path: Path):
    """Read DateTimeOriginal from EXIF data.

    Returns ISO datetime string like: 2026-01-20T19:06:54
    Returns None if no EXIF timestamp found.
    """
    try:
        from PIL.ExifTags import TAGS

        with Image.open(image_path) as img:
            # Try _getexif() first (works for JPEG)
            exif_data = None
            try:
                exif_data = img._getexif()
            except AttributeError:
                pass

            # Try getexif() as fallback (newer Pillow API)
            if not exif_data:
                try:
                    exif_obj = img.getexif()
                    if exif_obj:
                        exif_data = dict(exif_obj)
                except Exception:
                    pass

            if not exif_data:
                print(f"    [EXIF] No EXIF data found in {image_path.name}")
                return None

            # Look for DateTimeOriginal (tag 36867) or DateTime (tag 306)
            datetime_original = exif_data.get(36867)  # DateTimeOriginal
            datetime_tag = exif_data.get(306)  # DateTime

            # Also try by name in case tag IDs differ
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal' and value and not datetime_original:
                    datetime_original = value
                elif tag == 'DateTime' and value and not datetime_tag:
                    datetime_tag = value

            # Use DateTimeOriginal if available, else DateTime
            timestamp = datetime_original or datetime_tag

            if timestamp:
                # Format: "2026:01:20 19:06:54" -> "2026-01-20T19:06:54"
                result = str(timestamp).replace(':', '-', 2).replace(' ', 'T')
                return result

            print(f"    [EXIF] EXIF data exists but no timestamp tags found in {image_path.name}")

    except Exception as e:
        print(f"[EXIF] Failed to read timestamp from {image_path.name}: {e}")

    return None


def parse_cuddelink_datetime(filename: str, fallback_date: str) -> str:
    """Parse datetime from CuddeLink filename (DEPRECATED - use ocr_timestamp_from_image).

    This is kept as a fallback but the filename timestamp is when CuddeLink
    processed the photo, NOT when the trail camera captured it.
    """
    # Try to extract datetime from filename
    # Pattern: YYYY-MM-DDTHH_MM_SS.microseconds-sequence.ext
    match = re.match(r'(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})', filename)
    if match:
        date_part = match.group(1)
        hour = int(match.group(2))
        minute = match.group(3)
        second = match.group(4)

        # Convert UTC to Central time (UTC-6)
        # Note: This is approximate since filename is upload time, not capture time
        utc_dt = datetime.fromisoformat(f"{date_part}T{hour:02d}:{minute}:{second}")
        central_dt = utc_dt - timedelta(hours=6)

        return central_dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Fallback to just the date
    return fallback_date


def upload_to_r2(s3_client, bucket: str, file_path: Path, r2_key: str) -> bool:
    """Upload file to R2."""
    try:
        s3_client.upload_file(str(file_path), bucket, r2_key)
        return True
    except Exception as e:
        print(f"[R2] Upload failed: {e}")
        return False


def upload_bytes_to_r2(s3_client, bucket: str, data: bytes, r2_key: str) -> bool:
    """Upload bytes to R2."""
    try:
        s3_client.put_object(Bucket=bucket, Key=r2_key, Body=data, ContentType='image/jpeg')
        return True
    except Exception as e:
        print(f"[R2] Upload failed: {e}")
        return False


def save_to_supabase(supabase_url: str, supabase_key: str, file_hash: str, datetime_taken: str, filename: str) -> bool:
    """Save photo metadata to Supabase."""
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    # Extract just the date part for the photo_key path
    date_part = datetime_taken.split("T")[0]
    photo_key = f"{date_part.replace('-', '/')}/{filename}"

    data = {
        "file_hash": file_hash,
        "photo_key": photo_key,
        "date_taken": datetime_taken,
        "original_name": filename,
        "updated_at": datetime.utcnow().isoformat(),
    }

    try:
        # Use file_hash as conflict key to prevent duplicates
        resp = requests.post(
            f"{supabase_url}/rest/v1/photos_sync?on_conflict=file_hash",
            headers=headers,
            json=data,
            timeout=30,
        )
        return resp.status_code < 400
    except Exception as e:
        print(f"[Supabase] Save failed: {e}")
        return False


def process_day(session: requests.Session, day_str: str, s3_client, config: dict) -> tuple[int, int]:
    """Process a single day's photos. Returns (uploaded, skipped)."""
    print(f"\n[Sync] Processing {day_str}...")

    apply_date_filter(session, day_str, day_str)
    photo_ids = fetch_photo_ids(session)

    if not photo_ids:
        print(f"[Sync] No photos for {day_str}")
        return 0, 0

    print(f"[Sync] Found {len(photo_ids)} photos")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Download zip
        zip_path = download_photos_zip(session, photo_ids, tmp_dir)

        # Extract images
        images = extract_images(zip_path, tmp_dir)

        uploaded = 0
        skipped = 0

        for img_path in images:
            # Calculate hash
            file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()

            # Check if exists
            if check_photo_exists(config["supabase_url"], config["supabase_key"], file_hash):
                print(f"  Skipping duplicate: {img_path.name}")
                skipped += 1
                continue

            # Upload full image to R2
            r2_key = f"photos/{file_hash}.jpg"
            if upload_to_r2(s3_client, config["r2_bucket"], img_path, r2_key):
                # Generate and upload thumbnail
                try:
                    thumb_data = create_thumbnail(img_path)
                    thumb_key = f"thumbnails/{file_hash}_thumb.jpg"
                    upload_bytes_to_r2(s3_client, config["r2_bucket"], thumb_data, thumb_key)
                    print(f"  Uploaded: {img_path.name} + thumbnail")
                except Exception as e:
                    print(f"  Uploaded: {img_path.name} (thumbnail failed: {e})")

                # Get datetime from EXIF (preferred) or filename (fallback)
                datetime_taken = get_exif_datetime(img_path)
                if datetime_taken:
                    print(f"    EXIF timestamp: {datetime_taken}")
                else:
                    datetime_taken = parse_cuddelink_datetime(img_path.name, day_str)
                    print(f"    Filename timestamp (fallback): {datetime_taken}")

                # Save metadata
                save_to_supabase(
                    config["supabase_url"],
                    config["supabase_key"],
                    file_hash,
                    datetime_taken,
                    img_path.name,
                )
                uploaded += 1
            else:
                print(f"  Failed: {img_path.name}")

        return uploaded, skipped


def main():
    parser = argparse.ArgumentParser(description="Sync CuddeLink photos to R2")
    parser.add_argument("--days-back", type=int, default=1, help="Days to look back")
    args = parser.parse_args()

    # Load config from environment
    config = {
        "cuddelink_email": get_env("CUDDELINK_EMAIL"),
        "cuddelink_password": get_env("CUDDELINK_PASSWORD"),
        "r2_endpoint": get_env("R2_ENDPOINT"),
        "r2_access_key": get_env("R2_ACCESS_KEY"),
        "r2_secret_key": get_env("R2_SECRET_KEY"),
        "r2_bucket": get_env("R2_BUCKET"),
        "supabase_url": get_env("SUPABASE_URL"),
        "supabase_key": get_env("SUPABASE_KEY"),
    }

    # Set up R2 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=config["r2_endpoint"],
        aws_access_key_id=config["r2_access_key"],
        aws_secret_access_key=config["r2_secret_key"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    # Login to CuddeLink
    session = requests.Session()
    cuddelink_login(session, config["cuddelink_email"], config["cuddelink_password"])

    # Process each day
    today = date.today()
    total_uploaded = 0
    total_skipped = 0

    for days_ago in range(args.days_back, -1, -1):
        day = today - timedelta(days=days_ago)
        day_str = day.strftime("%Y-%m-%d")

        uploaded, skipped = process_day(session, day_str, s3_client, config)
        total_uploaded += uploaded
        total_skipped += skipped

    print(f"\n[Sync] Complete! Uploaded: {total_uploaded}, Skipped: {total_skipped}")


if __name__ == "__main__":
    main()
