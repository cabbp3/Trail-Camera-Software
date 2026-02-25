"""
Cloudflare R2 Storage Integration

Upload and download photos to/from Cloudflare R2 for web/mobile access.
Uses S3-compatible API via boto3.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional
import hashlib

logger = logging.getLogger(__name__)


def _get_bundled_config_path() -> Path:
    """Get path to bundled config, works for dev and PyInstaller bundle."""
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        return Path(sys._MEIPASS) / "cloud_config.json"
    # Running in development
    return Path(__file__).parent / "cloud_config.json"


# Config file locations (checked in order)
USER_CONFIG_PATH = Path.home() / ".trailcam" / "r2_config.json"
BUNDLED_CONFIG_PATH = _get_bundled_config_path()


class R2Storage:
    """Cloudflare R2 storage client for photo uploads/downloads."""

    def __init__(self):
        self.client = None
        self.bucket_name = None
        self._load_config()

    def _load_config(self):
        """Load R2 credentials from config file.

        Checks in order:
        1. User config (~/.trailcam/r2_config.json)
        2. Bundled config (cloud_config.json in app folder)
        """
        config = None

        # Try user config first
        if USER_CONFIG_PATH.exists():
            try:
                with open(USER_CONFIG_PATH) as f:
                    config = json.load(f)
                logger.info("Using user R2 config")
            except Exception as e:
                logger.warning(f"Failed to load user config: {e}")

        # Fall back to bundled config
        if not config and BUNDLED_CONFIG_PATH.exists():
            try:
                with open(BUNDLED_CONFIG_PATH) as f:
                    data = json.load(f)
                    config = data.get("r2", data)  # Handle nested or flat format
                logger.info("Using bundled R2 config")
            except Exception as e:
                logger.warning(f"Failed to load bundled config: {e}")

        if not config:
            logger.warning("No R2 config found")
            return

        try:
            import boto3
            from botocore.config import Config

            self.bucket_name = config["bucket_name"]

            self.client = boto3.client(
                "s3",
                endpoint_url=config["endpoint_url"],
                aws_access_key_id=config["access_key_id"],
                aws_secret_access_key=config["secret_access_key"],
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"}
                )
            )
            logger.info(f"R2 storage initialized for bucket: {self.bucket_name}")

        except ImportError:
            logger.error("boto3 not installed. Run: pip install boto3")
        except Exception as e:
            logger.error(f"Failed to initialize R2 storage: {e}")

    def is_configured(self) -> bool:
        """Check if R2 is properly configured."""
        return self.client is not None and self.bucket_name is not None

    def upload_photo(self, local_path: Path, user_id: str, photo_id: str) -> Optional[str]:
        """
        Upload a photo to R2.

        Args:
            local_path: Path to the local photo file
            user_id: User identifier (kept for API compatibility, not used)
            photo_id: Unique photo identifier (should be file hash)

        Returns:
            R2 key (path) if successful, None if failed

        Note: Uses shared structure photos/{photo_id}.jpg (no user prefix)
        """
        if not self.is_configured():
            logger.error("R2 not configured")
            return None

        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return None

        # Determine file extension
        ext = local_path.suffix.lower() or ".jpg"

        # Build R2 key - shared structure (no user prefix)
        r2_key = f"photos/{photo_id}{ext}"

        try:
            # Determine content type
            content_type = "image/jpeg"
            if ext == ".png":
                content_type = "image/png"
            elif ext in (".heic", ".heif"):
                content_type = "image/heic"

            self.client.upload_file(
                str(local_path),
                self.bucket_name,
                r2_key,
                ExtraArgs={"ContentType": content_type}
            )
            logger.info(f"Uploaded: {r2_key}")
            return r2_key

        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return None

    def upload_thumbnail(self, local_path: Path, user_id: str, photo_id: str) -> Optional[str]:
        """
        Upload a thumbnail to R2.

        Args:
            local_path: Path to the thumbnail file
            user_id: User identifier (kept for API compatibility, not used)
            photo_id: Unique photo identifier (should be file hash)

        Returns:
            R2 key if successful, None if failed

        Note: Uses shared structure thumbnails/{photo_id}_thumb.jpg (no user prefix)
        """
        if not self.is_configured():
            logger.error("R2 not configured")
            return None

        # Shared structure - no user prefix
        r2_key = f"thumbnails/{photo_id}_thumb.jpg"

        try:
            self.client.upload_file(
                str(local_path),
                self.bucket_name,
                r2_key,
                ExtraArgs={"ContentType": "image/jpeg"}
            )
            logger.info(f"Uploaded thumbnail: {r2_key}")
            return r2_key

        except Exception as e:
            logger.error(f"Failed to upload thumbnail {local_path}: {e}")
            return None

    def upload_file(self, local_path: Path, r2_key: str, content_type: str = "image/jpeg") -> bool:
        """
        Upload a file to R2 with a specific key.

        Args:
            local_path: Path to the local file
            r2_key: Full R2 key (path) for the file

        Returns:
            True if successful
        """
        if not self.is_configured():
            return False

        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return False

        try:
            # Determine content type from extension
            ext = local_path.suffix.lower()
            if ext == ".png":
                content_type = "image/png"
            elif ext in (".heic", ".heif"):
                content_type = "image/heic"
            else:
                content_type = "image/jpeg"

            self.client.upload_file(
                str(local_path),
                self.bucket_name,
                r2_key,
                ExtraArgs={"ContentType": content_type}
            )
            logger.debug(f"Uploaded: {r2_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {r2_key}: {e}")
            return False

    def upload_bytes(self, data: bytes, r2_key: str, content_type: str = "image/jpeg") -> bool:
        """
        Upload raw bytes to R2.

        Args:
            data: Bytes to upload
            r2_key: Full R2 path
            content_type: MIME type

        Returns:
            True if successful
        """
        if not self.is_configured():
            return False

        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=r2_key,
                Body=data,
                ContentType=content_type
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload bytes to {r2_key}: {e}")
            return False

    def download_photo(self, r2_key: str, local_path: Path) -> bool:
        """
        Download a photo from R2.

        Args:
            r2_key: R2 key (path) of the photo
            local_path: Where to save locally

        Returns:
            True if successful
        """
        if not self.is_configured():
            return False

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket_name, r2_key, str(local_path))
            logger.info(f"Downloaded: {r2_key} -> {local_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {r2_key}: {e}")
            return False

    def get_signed_url(self, r2_key: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generate a signed URL for temporary access to a photo.

        Args:
            r2_key: R2 key of the photo
            expires_in: URL validity in seconds (default 1 hour)

        Returns:
            Signed URL string, or None if failed
        """
        if not self.is_configured():
            return None

        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": r2_key},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {r2_key}: {e}")
            return None

    def delete_photo(self, r2_key: str) -> bool:
        """Delete a photo from R2."""
        if not self.is_configured():
            return False

        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=r2_key)
            logger.info(f"Deleted: {r2_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {r2_key}: {e}")
            return False

    def list_photos(self, user_id: str = None, prefix: str = "photos/") -> list:
        """
        List photos in shared storage.

        Args:
            user_id: User identifier (kept for API compatibility, not used)
            prefix: Path prefix (default "photos/")

        Returns:
            List of R2 keys

        Note: Uses shared structure - no user prefix
        """
        if not self.is_configured():
            return []

        # Shared structure - no user prefix
        full_prefix = prefix

        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=full_prefix
            )

            keys = []
            for obj in response.get("Contents", []):
                keys.append(obj["Key"])

            return keys

        except Exception as e:
            logger.error(f"Failed to list photos: {e}")
            return []

    def check_exists(self, r2_key: str) -> bool:
        """Check if a file exists in R2."""
        if not self.is_configured():
            return False

        try:
            self.client.head_object(Bucket=self.bucket_name, Key=r2_key)
            return True
        except self.client.exceptions.NoSuchKey:
            return False
        except Exception as e:
            # 404 comes back as ClientError with code 404
            error_code = getattr(e, 'response', {}).get('Error', {}).get('Code', '')
            if error_code in ('404', 'NoSuchKey'):
                return False
            logger.warning(f"R2 check_exists unexpected error for {r2_key}: {e}")
            return False

    def get_uploaded_photo_hashes(self) -> set:
        """Return set of file_hashes that exist in R2 photos/ prefix.

        Uses one paginated list_objects_v2 call instead of per-photo HEAD requests.
        Strips extension from key to extract hash, so both .jpg and .jpeg are found.
        """
        if not self.is_configured():
            return set()
        try:
            hashes = set()
            paginator = self.client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix="photos/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]  # e.g. "photos/abc123.jpg"
                    name = key.split("/", 1)[1] if "/" in key else key
                    file_hash = name.rsplit(".", 1)[0] if "." in name else name
                    hashes.add(file_hash)
            return hashes
        except Exception as e:
            logger.error(f"Failed to list uploaded photo hashes: {e}")
            return set()

    def get_bucket_stats(self) -> dict:
        """Get basic bucket statistics."""
        if not self.is_configured():
            return {"error": "Not configured"}

        try:
            paginator = self.client.get_paginator('list_objects_v2')
            total_size = 0
            count = 0
            for page in paginator.paginate(Bucket=self.bucket_name):
                for obj in page.get("Contents", []):
                    total_size += obj.get("Size", 0)
                    count += 1

            return {
                "object_count": count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {"error": str(e)}


def test_connection():
    """Test R2 connection and print status."""
    print("Testing R2 connection...")

    storage = R2Storage()

    if not storage.is_configured():
        print("ERROR: R2 not configured. Check ~/.trailcam/r2_config.json")
        return False

    print(f"Bucket: {storage.bucket_name}")

    # Try to get bucket stats
    stats = storage.get_bucket_stats()
    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return False

    print(f"Objects in bucket: {stats['object_count']}")
    print(f"Total size: {stats['total_size_mb']} MB")
    print("SUCCESS: R2 connection working!")
    return True


if __name__ == "__main__":
    test_connection()
