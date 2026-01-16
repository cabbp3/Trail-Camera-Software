"""
Combined site identifier for trail camera photos.

Uses a hybrid approach:
1. First tries OCR to read camera text overlays (most reliable)
2. Falls back to semantic embeddings when OCR fails (works on any image)

This handles cameras that do and don't burn location text into photos.
"""
import os
from typing import Optional, Dict, List, Tuple

# Import both approaches
from site_detector import SiteDetector, CAMERA_TO_SITE_MAP, OCR_AVAILABLE
from site_embedder import SemanticSiteEmbedder, TORCH_AVAILABLE
import numpy as np


class SiteIdentifier:
    """
    Identifies camera sites using OCR with semantic fallback.

    Best of both worlds:
    - OCR is 91% accurate when camera text is present
    - Semantic embeddings provide 70% accuracy as fallback
    """

    def __init__(self, db=None):
        """
        Initialize with both detection methods.

        Args:
            db: Optional database for loading known sites and building references
        """
        self._ocr = SiteDetector() if OCR_AVAILABLE else None
        self._semantic = SemanticSiteEmbedder() if TORCH_AVAILABLE else None
        self._db = db

        # Site reference embeddings (built from labeled photos)
        self._site_refs = {}  # site_name -> embedding

        self._ready = (self._ocr and self._ocr.ready) or (self._semantic and self._semantic.ready)

        methods = []
        if self._ocr and self._ocr.ready:
            methods.append("OCR")
        if self._semantic and self._semantic.ready:
            methods.append("Semantic")
        print(f"[SiteIdentifier] Ready with: {', '.join(methods) or 'None'}")

    @property
    def ready(self) -> bool:
        return self._ready

    def build_site_references(self, labeled_photos: List[Dict], samples_per_site: int = 20):
        """
        Build reference embeddings for each site from labeled photos.

        This enables the semantic fallback by learning what each site looks like.

        Args:
            labeled_photos: List of photo dicts with 'camera_location' and 'file_path'
            samples_per_site: Number of photos to use per site for reference
        """
        if not self._semantic or not self._semantic.ready:
            print("[SiteIdentifier] Semantic embeddings not available, skipping reference build")
            return

        # Group by location
        by_loc = {}
        for p in labeled_photos:
            loc = (p.get('camera_location') or '').strip()
            if loc and p.get('file_path'):
                if loc not in by_loc:
                    by_loc[loc] = []
                by_loc[loc].append(p)

        print(f"[SiteIdentifier] Building references for {len(by_loc)} sites...")

        for loc, photos in by_loc.items():
            # Sample photos
            import random
            sample = random.sample(photos, min(samples_per_site, len(photos)))

            # Extract embeddings
            embs = []
            for p in sample:
                emb = self._semantic.extract_embedding(p['file_path'])
                if emb is not None:
                    embs.append(emb)

            if embs:
                # Average and normalize
                ref = np.mean(embs, axis=0)
                ref = ref / (np.linalg.norm(ref) + 1e-8)
                self._site_refs[loc] = ref
                print(f"  {loc}: reference from {len(embs)} photos")

    def identify_site(
        self,
        image_path: str,
        min_semantic_confidence: float = 0.75
    ) -> Optional[Tuple[str, float, str]]:
        """
        Identify the site for an image.

        Args:
            image_path: Path to the image
            min_semantic_confidence: Minimum similarity for semantic match

        Returns:
            Tuple of (site_name, confidence, method) or None
            method is "ocr" or "semantic"
        """
        if not os.path.exists(image_path):
            return None

        # Method 1: Try OCR first (most reliable)
        if self._ocr and self._ocr.ready:
            result = self._ocr.detect_site(image_path)
            if result:
                site, conf = result
                return (site, conf, "ocr")

        # Method 2: Fall back to semantic embeddings
        if self._semantic and self._semantic.ready and self._site_refs:
            emb = self._semantic.extract_embedding(image_path)
            if emb is not None:
                # Find best matching site
                best_site = None
                best_sim = 0

                for site, ref in self._site_refs.items():
                    sim = float(np.dot(emb, ref))
                    if sim > best_sim:
                        best_sim = sim
                        best_site = site

                if best_site and best_sim >= min_semantic_confidence:
                    return (best_site, best_sim, "semantic")

        return None

    def identify_batch(
        self,
        image_paths: List[str],
        progress_callback=None
    ) -> Dict[str, Optional[Tuple[str, float, str]]]:
        """
        Identify sites for multiple images.

        Returns:
            Dict mapping path -> (site, confidence, method) or None
        """
        results = {}
        total = len(image_paths)

        ocr_count = 0
        semantic_count = 0
        failed_count = 0

        for i, path in enumerate(image_paths):
            result = self.identify_site(path)
            results[path] = result

            if result:
                if result[2] == "ocr":
                    ocr_count += 1
                else:
                    semantic_count += 1
            else:
                failed_count += 1

            if progress_callback and (i + 1) % 20 == 0:
                progress_callback(
                    i + 1, total,
                    f"OCR: {ocr_count}, Semantic: {semantic_count}, Failed: {failed_count}"
                )

        return results


def run_site_identification(
    db,
    progress_callback=None
) -> Dict[str, any]:
    """
    Run hybrid site identification on unlabeled photos.

    Args:
        db: TrailCamDatabase instance
        progress_callback: Optional callback(current, total, message)

    Returns:
        Results dict
    """
    # Get all photos
    photos = db.get_all_photos()
    labeled = [p for p in photos if (p.get('camera_location') or '').strip()]
    unlabeled = [p for p in photos if not (p.get('camera_location') or '').strip()]

    if not labeled:
        return {"error": "No labeled photos. Please label some photos with camera locations first."}

    # Create identifier and build references
    identifier = SiteIdentifier(db)

    if not identifier.ready:
        return {"error": "No identification methods available. Install pytesseract and/or PyTorch."}

    if progress_callback:
        progress_callback(0, 1, "Building site references from labeled photos...")

    identifier.build_site_references(labeled)

    # Process unlabeled photos
    if progress_callback:
        progress_callback(0, len(unlabeled), "Identifying sites...")

    ocr_count = 0
    semantic_count = 0
    failed_count = 0
    by_site = {}

    for i, photo in enumerate(unlabeled):
        path = photo.get('file_path')
        if not path:
            failed_count += 1
            continue

        result = identifier.identify_site(path)

        if result:
            site_name, confidence, method = result

            # Save to database
            site = db.get_site_by_name(site_name)
            if site:
                db.set_photo_site_suggestion(photo['id'], site['id'], confidence)
            else:
                site_id = db.create_site(site_name, confirmed=True)
                db.set_photo_site_suggestion(photo['id'], site_id, confidence)

            by_site[site_name] = by_site.get(site_name, 0) + 1

            if method == "ocr":
                ocr_count += 1
            else:
                semantic_count += 1
        else:
            failed_count += 1

        if progress_callback and (i + 1) % 20 == 0:
            progress_callback(i + 1, len(unlabeled),
                f"OCR: {ocr_count}, Semantic: {semantic_count}, Failed: {failed_count}")

    return {
        "total_unlabeled": len(unlabeled),
        "ocr_detected": ocr_count,
        "semantic_detected": semantic_count,
        "failed": failed_count,
        "by_site": by_site,
        "labeled_count": len(labeled)
    }


if __name__ == "__main__":
    print("Testing SiteIdentifier...")

    identifier = SiteIdentifier()

    if identifier.ready:
        # Quick test on a few images
        import subprocess
        result = subprocess.run(
            ['find', os.path.expanduser('~/TrailCamLibrary/2025'), '-name', '*.jpg'],
            capture_output=True, text=True
        )
        paths = result.stdout.strip().split('\n')[:10]

        print(f"\nTesting on {len([p for p in paths if p])} images:")
        for p in paths:
            if p and os.path.exists(p):
                result = identifier.identify_site(p)
                name = os.path.basename(p)
                if result:
                    site, conf, method = result
                    print(f"  {name}: {site} ({method}, {conf:.2f})")
                else:
                    print(f"  {name}: Not identified")
