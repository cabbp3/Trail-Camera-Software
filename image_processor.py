"""
Image processing module for EXIF extraction and file operations.
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from PIL import Image
from PIL.ExifTags import TAGS

# EXIF tag numbers (direct values for Pillow 11.3.0 compatibility)
EXIF_MODEL = 0x0110  # 272 - Camera model
EXIF_DATETIME = 0x0132  # 306 - DateTime
EXIF_DATETIME_ORIGINAL = 0x9003  # 36867 - DateTimeOriginal


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
            
            for tag_id, value in exifdata.items():
                # Use direct tag ID comparison for Pillow 11.3.0 compatibility
                if tag_id == EXIF_DATETIME_ORIGINAL:
                    # Format: "YYYY:MM:DD HH:MM:SS"
                    try:
                        dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                        date_taken = dt.isoformat()
                    except (ValueError, TypeError):
                        pass
                
                elif tag_id == EXIF_DATETIME:
                    # Use DateTime if DateTimeOriginal not found
                    if date_taken is None:
                        try:
                            dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                            date_taken = dt.isoformat()
                        except (ValueError, TypeError):
                            pass
                
                elif tag_id == EXIF_MODEL:
                    camera_model = str(value).strip()
            
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
        print(f"Error extracting EXIF from {image_path}: {e}")
        # Fallback to file modification time
        try:
            mtime = os.path.getmtime(image_path)
            dt = datetime.fromtimestamp(mtime)
            return dt.isoformat(), None
        except OSError:
            return None, None


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


def create_thumbnail(image_path: str, size: Tuple[int, int] = (250, 250)) -> Optional[str]:
    """Create a thumbnail for an image.
    
    Args:
        image_path: Path to source image
        size: Thumbnail size (width, height)
        
    Returns:
        Path to thumbnail file, or None if failed
    """
    try:
        thumb_dir = get_library_path() / ".thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = Path(image_path)
        thumb_filename = f"{source_path.stem}_thumb.jpg"
        thumb_path = thumb_dir / thumb_filename
        
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumb_path, "JPEG", quality=85)
        
        return str(thumb_path)
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return None

