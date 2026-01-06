"""
Cloudflare R2 Storage Integration

Upload and download photos to/from Cloudflare R2 for web/mobile access.
Uses S3-compatible API via boto3.
"""

import json
import logging
from pathlib import Path
from typing import Optional
import hashlib

logger = logging.getLogger(__name__)

# Config file locations (checked in order)
USER_CONFIG_PATH = Path.home() / ".trailcam" / "r2_config.json"
BUNDLED_CONFIG_PATH = Path(__file__).parent / "cloud_config.json"


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
            user_id: User identifier for organizing storage
            photo_id: Unique photo identifier

        Returns:
            R2 key (path) if successful, None if failed
        """
        if not self.is_configured():
            logger.error("R2 not configured")
            return None

        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return None

        # Determine file extension
        ext = local_path.suffix.lower() or ".jpg"

        # Build R2 key
        r2_key = f"users/{user_id}/photos/{photo_id}{ext}"

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
            user_id: User identifier
            photo_id: Unique photo identifier

        Returns:
            R2 key if successful, None if failed
        """
        if not self.is_configured():
            logger.error("R2 not configured")
            return None

        r2_key = f"users/{user_id}/thumbnails/{photo_id}_thumb.jpg"

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

    def list_photos(self, user_id: str, prefix: str = "photos/") -> list:
        """
        List photos for a user.

        Args:
            user_id: User identifier
            prefix: Sub-path within user folder (default "photos/")

        Returns:
            List of R2 keys
        """
        if not self.is_configured():
            return []

        full_prefix = f"users/{user_id}/{prefix}"

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
        except:
            return False

    def get_bucket_stats(self) -> dict:
        """Get basic bucket statistics."""
        if not self.is_configured():
            return {"error": "Not configured"}

        try:
            response = self.client.list_objects_v2(Bucket=self.bucket_name)

            total_size = 0
            count = 0
            for obj in response.get("Contents", []):
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
