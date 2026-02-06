"""
Site detector for trail camera photos using OCR.

Trail cameras typically burn a text overlay into images that includes:
- Date and time
- Camera/site name (e.g., "RAYS LINE 003", "SALT LICK E 002")
- Temperature

This module extracts that text to identify which site a photo came from.
This is more reliable than visual similarity since the camera explicitly
labels the photos.
"""
import os
import re
from typing import Optional, Dict, List, Tuple
from pathlib import Path

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# Mapping from camera overlay text patterns to database site names
# The camera overlay often differs from the human-friendly database name
CAMERA_TO_SITE_MAP = {
    "RAYS LINE": "Ray's Line",
    "RAY'S LINE": "Ray's Line",
    "SALT LICK": "Salt Lick",
    "WB 27": "WB 27",
    "WB27": "WB 27",
    # Note: "WEST OF ROAD" is ambiguous - could be West Triangle or West Salt Lick
    # We'll need to handle this specially or use visual features to disambiguate
    "WEST OF ROAD": "West Triangle",  # Default mapping, may need refinement
}

# Known site name patterns (will be auto-populated from database)
KNOWN_SITES = [
    "RAYS LINE",
    "RAY'S LINE",
    "SALT LICK",
    "WB 27",
    "WEST OF ROAD",
]


class SiteDetector:
    """
    Detects camera site from trail camera photo text overlays.

    Uses OCR to read the text burned into images by the camera,
    then matches against known site names.
    """

    def __init__(self, known_sites: List[str] = None):
        """
        Initialize the detector.

        Args:
            known_sites: List of known site name patterns to match against.
                        If None, uses the default KNOWN_SITES list.
        """
        self._ready = OCR_AVAILABLE
        self._sites = known_sites or KNOWN_SITES

        if not self._ready:
            print("[SiteDetector] OCR not available - install pytesseract")

    @property
    def ready(self) -> bool:
        return self._ready

    def set_known_sites(self, sites: List[str]):
        """Update the list of known site names."""
        self._sites = sites

    def detect_site(self, image_path: str) -> Optional[Tuple[str, float]]:
        """
        Detect site name from image text overlay.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (site_name, confidence) or None if not detected
        """
        if not self._ready or not os.path.exists(image_path):
            return None

        try:
            text = self._extract_overlay_text(image_path)
            if not text:
                return None

            return self._match_site(text)

        except Exception as e:
            print(f"[SiteDetector] Error processing {image_path}: {e}")
            return None

    def _extract_overlay_text(self, image_path: str) -> str:
        """
        Extract text from the camera's overlay region.

        Most trail cameras put text at the bottom of the image.
        Some use the top. We check both.
        """
        with Image.open(image_path) as img:
            w, h = img.size

            # Try bottom 12% first (most common location)
            bottom = img.crop((0, int(h * 0.88), w, h))
            text = pytesseract.image_to_string(bottom, config='--psm 6')

            # If no good text, try top 12%
            if not self._has_site_hint(text):
                top = img.crop((0, 0, w, int(h * 0.12)))
                top_text = pytesseract.image_to_string(top, config='--psm 6')
                if self._has_site_hint(top_text):
                    text = top_text

        return text.strip()

    def _has_site_hint(self, text: str) -> bool:
        """Check if text contains any hint of a site name."""
        upper = text.upper()
        for site in self._sites:
            if site.upper() in upper:
                return True
        # Also check for common patterns
        patterns = ['LINE', 'LICK', 'ROAD', 'TRIANGLE', 'STAND', 'FEEDER']
        return any(p in upper for p in patterns)

    def _match_site(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Match extracted text against known sites using camera-to-site mapping.

        Returns:
            Tuple of (database_site_name, confidence) or None
        """
        upper = text.upper()

        # Score each known pattern
        best_pattern = None
        best_score = 0

        for pattern in self._sites:
            pattern_upper = pattern.upper()

            # Exact substring match
            if pattern_upper in upper:
                # Give higher score to longer matches (more specific)
                score = len(pattern) / 20.0  # Normalize to ~0-1
                score = min(1.0, score + 0.5)  # Boost for exact match

                if score > best_score:
                    best_score = score
                    best_pattern = pattern

        if best_pattern:
            # Map camera pattern to database site name
            db_site = CAMERA_TO_SITE_MAP.get(best_pattern.upper(), best_pattern)
            return (db_site, best_score)

        return None

    def detect_sites_batch(
        self,
        image_paths: List[str],
        progress_callback=None
    ) -> Dict[str, Optional[Tuple[str, float]]]:
        """
        Detect sites for multiple images.

        Args:
            image_paths: List of paths to process
            progress_callback: Optional callback(current, total, message)

        Returns:
            Dict mapping path -> (site, confidence) or None
        """
        results = {}
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            results[path] = self.detect_site(path)

            if progress_callback and (i + 1) % 20 == 0:
                detected = sum(1 for v in results.values() if v)
                progress_callback(i + 1, total, f"{detected} sites detected...")

        return results


def load_known_sites_from_db(db) -> List[str]:
    """
    Load known site names from the database.

    Uses camera_location values that have been manually set.
    """
    photos = db.get_all_photos()
    sites = set()

    for p in photos:
        loc = (p.get('camera_location') or '').strip()
        if loc:
            sites.add(loc)

    return list(sites)


def run_site_detection(
    db,
    detector: SiteDetector = None,
    progress_callback=None
) -> Dict[str, any]:
    """
    Run OCR site detection on unlabeled photos.

    Args:
        db: TrailCamDatabase instance
        detector: SiteDetector instance (created if None)
        progress_callback: Optional callback(current, total, message)

    Returns:
        Results dict with detection statistics
    """
    # Load known sites from database
    known_sites = load_known_sites_from_db(db)

    if not known_sites:
        return {"error": "No labeled photos. Please label some photos with camera locations first."}

    # Create detector
    if detector is None:
        detector = SiteDetector(known_sites)
    else:
        detector.set_known_sites(known_sites)

    if not detector.ready:
        return {"error": "OCR not available. Install pytesseract."}

    print(f"[SiteDetector] Known sites: {known_sites}")

    # Get unlabeled photos
    photos = db.get_all_photos()
    unlabeled = [p for p in photos if not (p.get('camera_location') or '').strip()]

    if progress_callback:
        progress_callback(0, len(unlabeled), "Detecting sites via OCR...")

    # Process each photo
    detected = 0
    failed = 0
    by_site = {s: 0 for s in known_sites}

    for i, photo in enumerate(unlabeled):
        path = photo.get('file_path')
        if not path or not os.path.exists(path):
            failed += 1
            continue

        result = detector.detect_site(path)

        if result:
            site_name, confidence = result

            # Find matching site in database format
            matched = None
            for s in known_sites:
                if site_name.upper() in s.upper() or s.upper() in site_name.upper():
                    matched = s
                    break

            if matched:
                # Get or create site
                site = db.get_site_by_name(matched)
                if site:
                    db.set_photo_site_suggestion(photo['id'], site['id'], confidence)
                else:
                    site_id = db.create_site(matched, confirmed=True)
                    db.set_photo_site_suggestion(photo['id'], site_id, confidence)

                detected += 1
                by_site[matched] = by_site.get(matched, 0) + 1
        else:
            failed += 1

        if progress_callback and (i + 1) % 20 == 0:
            progress_callback(i + 1, len(unlabeled), f"{detected} sites detected...")

    return {
        "detected": detected,
        "failed": failed,
        "total_unlabeled": len(unlabeled),
        "known_sites": known_sites,
        "by_site": by_site
    }


# Quick test
if __name__ == "__main__":
    print("Testing SiteDetector...")

    if not OCR_AVAILABLE:
        print("OCR not available - install pytesseract")
    else:
        detector = SiteDetector()

        # Find test images
        import subprocess
        result = subprocess.run(
            ['find', os.path.expanduser('~/TrailCamLibrary/2025'), '-name', '*.jpg'],
            capture_output=True, text=True
        )
        paths = result.stdout.strip().split('\n')[:10]

        print(f"\nTesting on {len(paths)} images:")
        for p in paths:
            if p and os.path.exists(p):
                result = detector.detect_site(p)
                name = os.path.basename(p)
                if result:
                    print(f"  {name}: {result[0]} (conf: {result[1]:.2f})")
                else:
                    print(f"  {name}: No site detected")
