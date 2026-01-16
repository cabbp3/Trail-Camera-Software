"""
Site clustering for trail camera photos.

Uses image embeddings and DBSCAN clustering to automatically group photos
by camera location/site based on visual similarity of backgrounds.
"""
import os
import struct
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Optional imports - will check availability
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    models = None
    transforms = None
    Image = None

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    DBSCAN = None
    StandardScaler = None


class SiteClusterer:
    """
    Extracts structural features from trail camera photos for site clustering.

    Focuses on stable landmarks like tree trunks (vertical edges) that remain
    consistent across photos from the same camera location, even hours apart.
    """

    EMBEDDING_VERSION = "edges_v2"
    EMBEDDING_DIM = 256  # Edge-based feature dimension

    def __init__(self):
        self._ready = False
        try:
            import cv2
            self._cv2 = cv2
            self._ready = True
            print(f"[SiteClusterer] Edge-based clustering ready")
        except ImportError:
            print("[SiteClusterer] OpenCV not available, falling back to basic mode")
            self._ready = True  # Will use basic PIL-based approach

    @property
    def ready(self) -> bool:
        return self._ready

    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract structural features focused on stable landmarks (trees, edges).

        Looks for:
        - Strong vertical lines (tree trunks)
        - Edge structure in outer regions (not center where animals appear)
        - Large-scale features that don't change with wind/weather
        """
        if not os.path.exists(image_path):
            return None

        try:
            if hasattr(self, '_cv2'):
                return self._extract_edge_features(image_path)
            else:
                return self._extract_basic_features(image_path)
        except Exception as e:
            print(f"[SiteClusterer] Feature extraction failed for {image_path}: {e}")
            return None

    def _extract_edge_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract edge-based features using OpenCV."""
        cv2 = self._cv2

        # Load and resize image
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Resize to standard size for consistent features
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise (ignore small twigs)
        # Larger kernel = focus on bigger features
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Edge detection with higher thresholds to get only strong edges
        edges = cv2.Canny(blurred, 50, 150)

        # Focus on outer regions (left 25%, right 25%, top 30%)
        # This is where trees/landmarks are, not center where animals move
        h, w = edges.shape

        # Create mask for outer regions
        mask = np.zeros_like(edges)
        mask[:, :int(w*0.25)] = 1  # Left 25%
        mask[:, int(w*0.75):] = 1  # Right 25%
        mask[:int(h*0.3), :] = 1   # Top 30%

        # Apply mask
        masked_edges = edges * mask

        # Detect lines using Hough transform (finds straight lines like tree trunks)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50,
                                minLineLength=50, maxLineGap=10)

        # Build feature vector from line characteristics
        features = []

        # Feature 1: Histogram of edge positions (where are the major edges?)
        # Divide image into grid and count edge pixels
        grid_h, grid_w = 8, 8
        for i in range(grid_h):
            for j in range(grid_w):
                y1, y2 = i * h // grid_h, (i+1) * h // grid_h
                x1, x2 = j * w // grid_w, (j+1) * w // grid_w
                cell_edges = masked_edges[y1:y2, x1:x2]
                features.append(np.sum(cell_edges) / 255.0 / ((y2-y1) * (x2-x1)))

        # Feature 2: Vertical line positions and angles
        vertical_bins = np.zeros(16)  # 16 horizontal position bins
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is mostly vertical (tree trunk)
                if abs(x2 - x1) < abs(y2 - y1) * 0.3:  # Within 17 degrees of vertical
                    # Bin by horizontal position
                    center_x = (x1 + x2) / 2
                    bin_idx = int(center_x / w * 15)
                    bin_idx = min(15, max(0, bin_idx))
                    # Weight by line length
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    vertical_bins[bin_idx] += length / 100.0

        features.extend(vertical_bins.tolist())

        # Feature 3: Edge density in different regions
        regions = [
            (0, h//3, 0, w//3),      # Top-left
            (0, h//3, w//3, 2*w//3), # Top-center
            (0, h//3, 2*w//3, w),    # Top-right
            (h//3, 2*h//3, 0, w//3), # Mid-left
            (h//3, 2*h//3, 2*w//3, w), # Mid-right
        ]
        for y1, y2, x1, x2 in regions:
            region = edges[y1:y2, x1:x2]
            features.append(np.sum(region) / 255.0 / ((y2-y1) * (x2-x1)))

        # Feature 4: Overall edge orientation histogram
        # Use Sobel to get gradient directions
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        angles = np.arctan2(sobely, sobelx)
        magnitudes = np.sqrt(sobelx**2 + sobely**2)

        # Bin angles into histogram (weighted by magnitude)
        angle_bins = 16
        angle_hist = np.zeros(angle_bins)
        for i in range(angle_bins):
            low = -np.pi + i * 2 * np.pi / angle_bins
            high = -np.pi + (i+1) * 2 * np.pi / angle_bins
            mask = (angles >= low) & (angles < high) & (magnitudes > 20)
            angle_hist[i] = np.sum(magnitudes[mask])
        angle_hist = angle_hist / (np.sum(angle_hist) + 1e-8)
        features.extend(angle_hist.tolist())

        # Convert to numpy array and normalize
        emb = np.array(features, dtype=np.float32)

        # Pad or truncate to fixed size
        if len(emb) < self.EMBEDDING_DIM:
            emb = np.pad(emb, (0, self.EMBEDDING_DIM - len(emb)))
        else:
            emb = emb[:self.EMBEDDING_DIM]

        # L2 normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def _extract_basic_features(self, image_path: str) -> Optional[np.ndarray]:
        """Fallback feature extraction using PIL only."""
        img = Image.open(image_path).convert("L")  # Grayscale
        img = img.resize((64, 48))

        # Simple grid-based intensity features
        arr = np.array(img, dtype=np.float32) / 255.0

        # Compute edge approximation using differences
        dx = np.abs(np.diff(arr, axis=1))
        dy = np.abs(np.diff(arr, axis=0))

        # Grid features from edges
        features = []
        h, w = arr.shape
        for i in range(8):
            for j in range(8):
                y1, y2 = i * h // 8, (i+1) * h // 8
                x1, x2 = j * w // 8, (j+1) * w // 8
                if x2 <= dx.shape[1] and y2 <= dx.shape[0]:
                    features.append(np.mean(dx[y1:y2, x1:x2]))
                if x2 <= dy.shape[1] and y2 <= dy.shape[0]:
                    features.append(np.mean(dy[y1:y2, x1:min(x2, dy.shape[1])]))

        emb = np.array(features, dtype=np.float32)
        if len(emb) < self.EMBEDDING_DIM:
            emb = np.pad(emb, (0, self.EMBEDDING_DIM - len(emb)))
        else:
            emb = emb[:self.EMBEDDING_DIM]

        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        """Convert numpy embedding to bytes for database storage."""
        return embedding.tobytes()

    def bytes_to_embedding(self, data: bytes) -> np.ndarray:
        """Convert bytes back to numpy embedding."""
        return np.frombuffer(data, dtype=np.float32)

    def cluster_embeddings(
        self,
        embeddings: List[Tuple[int, np.ndarray]],
        eps: float = 0.3,
        min_samples: int = 5
    ) -> Dict[int, int]:
        """
        Cluster photo embeddings using DBSCAN.

        Args:
            embeddings: List of (photo_id, embedding_vector) tuples
            eps: DBSCAN distance threshold (lower = tighter clusters)
            min_samples: Minimum photos to form a cluster

        Returns:
            Dict mapping photo_id -> cluster_id (-1 = noise/unassigned)
        """
        if not SKLEARN_AVAILABLE:
            print("[SiteClusterer] sklearn not available for clustering")
            return {}

        if len(embeddings) < min_samples:
            print(f"[SiteClusterer] Not enough photos ({len(embeddings)}) for clustering")
            return {}

        photo_ids = [e[0] for e in embeddings]
        vectors = np.array([e[1] for e in embeddings])

        # DBSCAN clustering with cosine-like distance (since embeddings are normalized)
        # Using eps on normalized vectors: smaller eps = tighter clusters
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(vectors)

        return {pid: int(label) for pid, label in zip(photo_ids, labels)}

    def compute_cluster_confidence(
        self,
        cluster_embeddings: List[Tuple[int, np.ndarray]]
    ) -> Dict[int, float]:
        """
        Compute confidence score for each photo in a cluster.

        Confidence is based on distance to cluster centroid -
        photos closer to center are more confidently assigned.

        Returns:
            Dict mapping photo_id -> confidence (0.0 to 1.0)
        """
        if not cluster_embeddings:
            return {}

        vectors = np.array([e[1] for e in cluster_embeddings])
        centroid = vectors.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Distance from centroid (0 = at center, higher = further)
        distances = np.linalg.norm(vectors - centroid, axis=1)

        # Convert to confidence: closer = higher confidence
        # Use exponential decay so photos near center get high confidence
        max_dist = distances.max() if distances.max() > 0 else 1.0
        confidences = np.exp(-2 * distances / max_dist)

        return {pid: float(conf) for (pid, _), conf in zip(cluster_embeddings, confidences)}

    def find_representative_photo(
        self,
        cluster_embeddings: List[Tuple[int, np.ndarray]]
    ) -> Optional[int]:
        """
        Find the most representative photo for a cluster (closest to centroid).

        Args:
            cluster_embeddings: List of (photo_id, embedding) for photos in cluster

        Returns:
            photo_id of the most representative photo
        """
        if not cluster_embeddings:
            return None

        vectors = np.array([e[1] for e in cluster_embeddings])
        centroid = vectors.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Find photo closest to centroid
        distances = np.linalg.norm(vectors - centroid, axis=1)
        best_idx = int(np.argmin(distances))
        return cluster_embeddings[best_idx][0]


def run_site_clustering(
    db,
    clusterer: SiteClusterer = None,
    k_neighbors: int = 10,
    strong_threshold: float = 0.25,
    min_strong_matches: int = 2,
    progress_callback=None
) -> Dict[str, any]:
    """
    Match unlabeled photos to verified (human-labeled) camera locations.

    Only suggests locations that already exist from human labels - does NOT create new sites.
    Requires 2+ STRONG matches (very similar photos) from the same location.
    This focuses on key characteristics (trees, branches) that should match closely.

    Args:
        db: TrailCamDatabase instance
        clusterer: SiteClusterer instance (created if None)
        k_neighbors: Number of nearest labeled photos to consider
        strong_threshold: Max distance for a "strong" match (lower = stricter)
        min_strong_matches: Minimum strong matches from same location required
        progress_callback: Optional callback(current, total, message)

    Returns:
        Dict with matching results
    """
    if clusterer is None:
        clusterer = SiteClusterer()

    if not clusterer.ready:
        return {"error": "Clusterer not ready - check OpenCV installation"}

    # Step 1: Get labeled photos (those with camera_location set)
    all_photos = db.get_all_photos()
    labeled_photos = [p for p in all_photos if p.get("camera_location") and p["camera_location"].strip()]
    unlabeled_photos = [p for p in all_photos if not p.get("camera_location") or not p["camera_location"].strip()]

    if not labeled_photos:
        return {"error": "No labeled photos found. Please label some photos with camera locations first."}

    # Get unique locations
    locations = set(p["camera_location"].strip() for p in labeled_photos)
    print(f"[SiteClusterer] Found {len(labeled_photos)} labeled photos across {len(locations)} locations")
    print(f"[SiteClusterer] Locations: {', '.join(sorted(locations))}")

    # Step 2: Compute embeddings for photos that don't have them
    photos_needing_embeddings = db.get_photos_without_embeddings()
    total_embed = len(photos_needing_embeddings)

    if progress_callback:
        progress_callback(0, total_embed, "Computing image features...")

    for i, photo in enumerate(photos_needing_embeddings):
        path = photo.get("file_path") or photo.get("thumbnail_path")
        if path and os.path.exists(path):
            emb = clusterer.extract_embedding(path)
            if emb is not None:
                db.save_embedding(photo["id"], clusterer.embedding_to_bytes(emb),
                                  clusterer.EMBEDDING_VERSION)

        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(i + 1, total_embed, f"Extracted {i + 1}/{total_embed} features")

    # Step 3: Load all embeddings
    if progress_callback:
        progress_callback(0, 1, "Loading embeddings...")

    raw_embeddings = db.get_all_embeddings()
    # Handle both old (pid, data) and new (pid, data, version) format
    all_embeddings = {}
    for item in raw_embeddings:
        if len(item) >= 2:
            pid, data = item[0], item[1]
            all_embeddings[pid] = clusterer.bytes_to_embedding(data)

    # Build reference set from labeled photos
    # Each entry: (photo_id, embedding, camera_location)
    labeled_refs = []
    for p in labeled_photos:
        if p["id"] in all_embeddings:
            labeled_refs.append((p["id"], all_embeddings[p["id"]], p["camera_location"].strip()))

    if len(labeled_refs) < k_neighbors:
        return {"error": f"Need at least {k_neighbors} labeled photos with embeddings. Have {len(labeled_refs)}."}

    # Step 4: For each unlabeled photo, find k nearest labeled photos
    if progress_callback:
        progress_callback(0, len(unlabeled_photos), "Matching photos to locations...")

    suggestions_made = 0
    no_match_count = 0
    results_by_location = {loc: 0 for loc in locations}

    for i, photo in enumerate(unlabeled_photos):
        pid = photo["id"]
        if pid not in all_embeddings:
            continue

        emb = all_embeddings[pid]

        # Calculate distance to all labeled photos
        distances = []
        for ref_pid, ref_emb, ref_loc in labeled_refs:
            dist = np.linalg.norm(emb - ref_emb)
            distances.append((dist, ref_loc, ref_pid))

        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[0])

        # Take k nearest neighbors
        k_nearest = distances[:k_neighbors]

        # Count STRONG matches and ALL matches per location
        location_strong_counts = {}  # How many strong matches per location
        location_total_counts = {}   # How many total matches per location (for consensus)
        location_best_distance = {}  # Best (min) distance per location

        for dist, loc, ref_pid in k_nearest:
            # Track best distance per location
            if loc not in location_best_distance or dist < location_best_distance[loc]:
                location_best_distance[loc] = dist
            # Count total matches (for consensus check)
            location_total_counts[loc] = location_total_counts.get(loc, 0) + 1
            # Count strong matches
            if dist < strong_threshold:
                location_strong_counts[loc] = location_strong_counts.get(loc, 0) + 1

        # Find best location by: 1) most strong matches, 2) most total matches, 3) closest distance
        best_location = None
        best_strong_count = 0
        best_total_count = 0
        best_distance = float('inf')

        all_locations = set(location_strong_counts.keys()) | set(location_total_counts.keys())
        for loc in all_locations:
            strong_count = location_strong_counts.get(loc, 0)
            total_count = location_total_counts.get(loc, 0)
            dist = location_best_distance.get(loc, float('inf'))

            # Prioritize strong matches, then total count, then distance
            if strong_count > best_strong_count:
                best_strong_count = strong_count
                best_total_count = total_count
                best_location = loc
                best_distance = dist
            elif strong_count == best_strong_count:
                if total_count > best_total_count:
                    best_total_count = total_count
                    best_location = loc
                    best_distance = dist
                elif total_count == best_total_count and dist < best_distance:
                    best_location = loc
                    best_distance = dist

        # Count top 5 closest for consensus check
        top5_location_counts = {}
        for dist, loc, ref_pid in distances[:5]:  # Only top 5 closest
            top5_location_counts[loc] = top5_location_counts.get(loc, 0) + 1

        # Find consensus location (most common in top 5)
        consensus_location = max(top5_location_counts.keys(), key=lambda x: top5_location_counts[x]) if top5_location_counts else None
        consensus_count = top5_location_counts.get(consensus_location, 0) if consensus_location else 0

        # Accept if: 2+ strong matches OR 4/5 closest neighbors agree
        has_strong_match = best_strong_count >= min_strong_matches
        has_consensus = consensus_count >= 4  # 4 of 5 closest neighbors

        # Use the location from whichever rule matched
        match_location = None
        if has_strong_match:
            match_location = best_location
        elif has_consensus:
            match_location = consensus_location

        if match_location:
            # Calculate confidence based on number of strong matches and distance
            confidence = float(min(1.0, best_strong_count / 3) * 0.5 + (1.0 - best_distance / strong_threshold) * 0.5)

            # Store suggestion in database
            site = db.get_site_by_name(match_location)
            if site:
                db.set_photo_site_suggestion(pid, site["id"], confidence)
            else:
                # Create site for this verified location
                site_id = db.create_site(match_location, confirmed=True)
                db.set_photo_site_suggestion(pid, site_id, confidence)

            suggestions_made += 1
            results_by_location[match_location] = results_by_location.get(match_location, 0) + 1
        else:
            no_match_count += 1

        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1, len(unlabeled_photos),
                            f"Processed {i + 1}/{len(unlabeled_photos)} ({suggestions_made} suggestions)")

    if progress_callback:
        progress_callback(len(unlabeled_photos), len(unlabeled_photos), "Done!")

    return {
        "suggestions_made": suggestions_made,
        "no_match_count": no_match_count,
        "total_unlabeled": len(unlabeled_photos),
        "labeled_count": len(labeled_photos),
        "locations": list(locations),
        "results_by_location": results_by_location
    }


def suggest_cluster_parameters(db) -> Dict[str, float]:
    """
    Analyze the database and return info about labeled photos.

    Returns info about labeled/unlabeled counts and suggested parameters.
    """
    all_photos = db.get_all_photos()
    photo_count = len(all_photos)

    # Count labeled vs unlabeled
    labeled_photos = [p for p in all_photos if p.get("camera_location") and p["camera_location"].strip()]
    unlabeled_photos = [p for p in all_photos if not p.get("camera_location") or not p["camera_location"].strip()]

    # Get unique locations
    locations = {}
    for p in labeled_photos:
        loc = p["camera_location"].strip()
        locations[loc] = locations.get(loc, 0) + 1

    # Default parameters for k-NN matching
    k_neighbors = 5
    match_threshold = 0.35
    min_matches = 3

    return {
        "k_neighbors": k_neighbors,
        "match_threshold": match_threshold,
        "min_matches": min_matches,
        "photo_count": photo_count,
        "labeled_count": len(labeled_photos),
        "unlabeled_count": len(unlabeled_photos),
        "locations": locations,
        "location_count": len(locations)
    }
