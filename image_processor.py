"""
Image processing module for EXIF extraction and file operations.
"""
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# EXIF tag numbers (direct values for Pillow 11.3.0 compatibility)
EXIF_MODEL = 0x0110  # 272 - Camera model
EXIF_DATETIME = 0x0132  # 306 - DateTime
EXIF_DATETIME_ORIGINAL = 0x9003  # 36867 - DateTimeOriginal
EXIF_USER_COMMENT = 0x9286  # 37510 - UserComment (contains camera ID for CuddeLink)


def _sanitize_camera_model(value: str) -> Optional[str]:
    """Sanitize camera_model to prevent corrupted EXIF data from causing sync issues.

    Returns None if the value is clearly garbage (mostly non-printable chars).
    """
    if not value:
        return None

    # Convert to string and strip
    value = str(value).strip()
    if not value:
        return None

    # Remove null bytes and other control characters
    cleaned = ''.join(c for c in value if c.isprintable() or c in ' \t')
    cleaned = cleaned.strip()

    if not cleaned:
        return None

    # Check if it's mostly garbage (less than 50% alphanumeric/space)
    alnum_count = sum(1 for c in cleaned if c.isalnum() or c in ' -_')
    if len(cleaned) > 0 and alnum_count / len(cleaned) < 0.5:
        return None

    # Reject if it looks like date fragments (common corruption pattern)
    if len(cleaned) > 3 and cleaned[0].isdigit() and '/' in cleaned:
        return None

    # Truncate very long values (likely corruption)
    if len(cleaned) > 50:
        cleaned = cleaned[:50]

    return cleaned if cleaned else None


def get_library_path() -> Path:
    """Get the trail camera library path based on OS."""
    home = Path.home()
    if os.name == 'nt':  # Windows
        return Path("C:/TrailCamLibrary")
    else:  # macOS and Linux
        return home / "TrailCamLibrary"


def extract_exif_data(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract date/time and camera model from EXIF data.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (date_taken, camera_model) or (None, None) if not found
    """
    try:
        with Image.open(image_path) as img:
            exifdata = img.getexif()

            if exifdata is None:
                return None, None

            date_taken = None
            camera_model = None

            # Get camera model from IFD0 (main EXIF)
            if EXIF_MODEL in exifdata:
                camera_model = _sanitize_camera_model(exifdata[EXIF_MODEL])

            # DateTime (tag 306) is in IFD0 - use as fallback
            datetime_fallback = exifdata.get(EXIF_DATETIME)

            # DateTimeOriginal (tag 36867) is in the EXIF sub-IFD (0x8769)
            # This is where the ACTUAL capture time is stored
            datetime_original = None
            try:
                exif_ifd = exifdata.get_ifd(0x8769)  # EXIF IFD
                if exif_ifd:
                    datetime_original = exif_ifd.get(EXIF_DATETIME_ORIGINAL)
            except (AttributeError, Exception):
                pass  # get_ifd not available in older Pillow

            # Try deprecated _getexif() as fallback (merges all IFDs)
            if not datetime_original:
                try:
                    old_exif = img._getexif()
                    if old_exif:
                        datetime_original = old_exif.get(EXIF_DATETIME_ORIGINAL)
                        if not datetime_fallback:
                            datetime_fallback = old_exif.get(EXIF_DATETIME)
                        if not camera_model:
                            camera_model = _sanitize_camera_model(old_exif.get(EXIF_MODEL))
                except (AttributeError, Exception):
                    pass

            # Use DateTimeOriginal (capture time) if available, else DateTime
            timestamp = datetime_original or datetime_fallback
            if timestamp:
                try:
                    dt = datetime.strptime(str(timestamp), "%Y:%m:%d %H:%M:%S")
                    date_taken = dt.isoformat()
                except (ValueError, TypeError):
                    pass

            # If no date found in EXIF, try file modification time
            if date_taken is None:
                try:
                    mtime = os.path.getmtime(image_path)
                    dt = datetime.fromtimestamp(mtime)
                    date_taken = dt.isoformat()
                except OSError:
                    pass

            return date_taken, camera_model

    except Exception as e:
        logger.warning(f"Error extracting EXIF from {image_path}: {e}")
        # Fallback to file modification time
        try:
            mtime = os.path.getmtime(image_path)
            dt = datetime.fromtimestamp(mtime)
            return dt.isoformat(), None
        except OSError:
            return None, None


def _get_cuddelink_user_comment(image_path: str) -> Optional[str]:
    """Extract UserComment field from CuddeLink EXIF data."""
    import re
    try:
        with Image.open(image_path) as img:
            exifdata = img.getexif()
            if exifdata is None:
                return None

            # Get IFD (EXIF sub-IFD) for UserComment
            exif_ifd = exifdata.get_ifd(0x8769)  # ExifOffset IFD
            if not exif_ifd:
                return None

            user_comment = exif_ifd.get(EXIF_USER_COMMENT)
            if not user_comment:
                return None

            # UserComment can be bytes or string
            if isinstance(user_comment, bytes):
                try:
                    user_comment = user_comment.decode('utf-8', errors='ignore').strip('\x00')
                except:
                    user_comment = str(user_comment)

            return str(user_comment)
    except Exception as e:
        logger.debug(f"Error extracting UserComment from {image_path}: {e}")
        return None


def extract_cuddelink_camera_id(image_path: str) -> Optional[str]:
    """Extract camera ID (location name) from CuddeLink EXIF UserComment field.

    CuddeLink cameras store metadata in UserComment like:
    "MR=C.1,AD=8/11/2025,...,ID=SALT LICK E,LO=002,MA=1328AD88A36E,..."

    Args:
        image_path: Path to the image file

    Returns:
        Camera ID string (e.g., "SALT LICK E") or None if not found
    """
    import re
    user_comment = _get_cuddelink_user_comment(image_path)
    if not user_comment:
        return None

    match = re.search(r'ID=([^,]+)', user_comment)
    if match:
        return match.group(1).strip()
    return None


def extract_cuddelink_mac_address(image_path: str) -> Optional[str]:
    """Extract camera MAC address (unique hardware ID) from CuddeLink EXIF.

    The MAC address is unique to each physical camera and doesn't change
    when the camera is moved to a new location.

    Args:
        image_path: Path to the image file

    Returns:
        MAC address string (e.g., "1328AD88A36E") or None if not found
    """
    import re
    user_comment = _get_cuddelink_user_comment(image_path)
    if not user_comment:
        return None

    match = re.search(r'MA=([^,]+)', user_comment)
    if match:
        return match.group(1).strip()
    return None


def generate_new_filename(date_taken: Optional[str], camera_model: Optional[str]) -> str:
    """Generate new filename in format: YYYY-MM-DD_HH-MM-SS_CameraModel.jpg
    
    Args:
        date_taken: ISO format datetime string
        camera_model: Camera model name
        
    Returns:
        New filename
    """
    if date_taken:
        try:
            dt = datetime.fromisoformat(date_taken.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            dt = datetime.now()
    else:
        dt = datetime.now()
    
    # Format: YYYY-MM-DD_HH-MM-SS
    date_str = dt.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Clean camera model for filename
    if camera_model:
        # Remove invalid filename characters
        model_clean = "".join(c for c in camera_model if c.isalnum() or c in (' ', '-', '_'))
        model_clean = model_clean.strip().replace(' ', '_')
        if model_clean:
            filename = f"{date_str}_{model_clean}.jpg"
        else:
            filename = f"{date_str}.jpg"
    else:
        filename = f"{date_str}.jpg"
    
    return filename


def get_destination_path(date_taken: Optional[str]) -> Path:
    """Get destination path based on date (Year/Month structure).
    
    Args:
        date_taken: ISO format datetime string
        
    Returns:
        Path object for destination directory
    """
    library_path = get_library_path()
    
    if date_taken:
        try:
            dt = datetime.fromisoformat(date_taken.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            dt = datetime.now()
    else:
        dt = datetime.now()
    
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    
    dest_dir = library_path / year / month
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    return dest_dir


def import_photo(source_path: str) -> Tuple[str, str, str, str]:
    """Import a photo from source to library.
    
    Args:
        source_path: Path to source image file
        
    Returns:
        Tuple of (new_file_path, original_name, date_taken, camera_model)
    """
    source_path = Path(source_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Extract EXIF data
    date_taken, camera_model = extract_exif_data(str(source_path))
    
    # Generate new filename
    new_filename = generate_new_filename(date_taken, camera_model)
    
    # Get destination directory
    dest_dir = get_destination_path(date_taken)
    
    # Destination file path
    dest_path = dest_dir / new_filename
    
    # Handle duplicate filenames
    counter = 1
    original_dest = dest_path
    while dest_path.exists():
        stem = original_dest.stem
        dest_path = dest_dir / f"{stem}_{counter:03d}.jpg"
        counter += 1
    
    # Copy file
    shutil.copy2(source_path, dest_path)
    
    return (str(dest_path), source_path.name, date_taken or "", camera_model or "")


def create_thumbnail(image_path: str, size: Tuple[int, int] = (250, 250),
                     force: bool = False) -> Optional[str]:
    """Create a thumbnail for an image.

    Args:
        image_path: Path to source image
        size: Thumbnail size (width, height)
        force: If True, regenerate even if thumbnail exists

    Returns:
        Path to thumbnail file, or None if failed
    """
    try:
        thumb_dir = get_library_path() / ".thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(image_path)
        thumb_filename = f"{source_path.stem}_thumb.jpg"
        thumb_path = thumb_dir / thumb_filename

        # Skip if thumbnail already exists (unless forced)
        if thumb_path.exists() and not force:
            return str(thumb_path)

        with Image.open(image_path) as img:
            # Convert RGBA/P mode to RGB for JPEG compatibility
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.thumbnail(size, Image.Resampling.LANCZOS)
            # Lower quality (75) and optimize for smaller file size
            img.save(thumb_path, "JPEG", quality=75, optimize=True)

        return str(thumb_path)
    except Exception as e:
        logger.warning(f"Error creating thumbnail for {image_path}: {e}")
        return None

