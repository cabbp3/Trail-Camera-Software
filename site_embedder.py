"""
Semantic site embedder for trail camera photos.

Uses a pre-trained vision model to create scene embeddings that are robust
to lighting, weather, seasons, and animal presence. This replaces the
edge-based approach in site_clustering.py with semantic understanding.

Key improvements over edge-based:
1. Masks out detected animals before computing embedding
2. Uses pre-trained features that understand "forest", "field", etc.
3. More robust to lighting/IR mode changes
4. Captures semantic scene similarity, not just edge positions
"""
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Check for dependencies
try:
    import torch
    import torch.nn as nn
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
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class SemanticSiteEmbedder:
    """
    Creates semantic embeddings of trail camera scenes for site matching.

    Uses MobileNetV2 features (pre-trained on ImageNet) which understand
    scene content semantically. Masks out animal regions to focus on
    the static background.
    """

    EMBEDDING_VERSION = "semantic_v1"
    EMBEDDING_DIM = 1280  # MobileNetV2 feature dimension

    def __init__(self, onnx_path: Optional[str] = None, use_gpu: bool = False):
        """
        Initialize the embedder.

        Args:
            onnx_path: Path to ONNX model. If None, uses PyTorch model.
            use_gpu: Whether to use GPU (if available)
        """
        self._ready = False
        self._model = None
        self._session = None
        self._transform = None
        self._device = "cpu"

        # Try ONNX first (faster, no PyTorch needed)
        if onnx_path and ONNX_AVAILABLE and os.path.exists(onnx_path):
            self._init_onnx(onnx_path)
        elif TORCH_AVAILABLE:
            self._init_pytorch(use_gpu)
        else:
            print("[SemanticSiteEmbedder] Neither PyTorch nor ONNX available")

    def _init_pytorch(self, use_gpu: bool):
        """Initialize with PyTorch MobileNetV2."""
        try:
            # Use MobileNetV2 - small (14MB) but effective
            self._model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

            # Remove classification head - we want features only
            # MobileNetV2 features are in model.features, then avgpool
            self._model.classifier = nn.Identity()

            # Set to eval mode
            self._model.eval()

            # Move to GPU if requested and available
            if use_gpu and torch.cuda.is_available():
                self._device = "cuda"
                self._model = self._model.to(self._device)

            # Standard ImageNet preprocessing
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            self._ready = True
            print(f"[SemanticSiteEmbedder] PyTorch MobileNetV2 ready (device: {self._device})")

        except Exception as e:
            print(f"[SemanticSiteEmbedder] PyTorch init failed: {e}")

    def _init_onnx(self, onnx_path: str):
        """Initialize with ONNX model."""
        try:
            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(onnx_path, providers=providers)
            self._ready = True
            print(f"[SemanticSiteEmbedder] ONNX model ready: {onnx_path}")
        except Exception as e:
            print(f"[SemanticSiteEmbedder] ONNX init failed: {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    def extract_embedding(
        self,
        image_path: str,
        mask_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
        boxes_are_pixels: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract semantic embedding from an image.

        Args:
            image_path: Path to the image file
            mask_boxes: Optional list of (x1, y1, x2, y2) boxes to mask out.
                       These are typically animal detections from MegaDetector.
            boxes_are_pixels: If True, boxes are in pixel coords (default).
                             If False, boxes are 0-1 normalized.

        Returns:
            1280-dim normalized embedding vector, or None on failure
        """
        if not self._ready or not os.path.exists(image_path):
            return None

        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = img.size

            # Convert pixel boxes to normalized if needed
            if mask_boxes and boxes_are_pixels:
                normalized_boxes = []
                for x1, y1, x2, y2 in mask_boxes:
                    # Normalize to 0-1 range
                    normalized_boxes.append((
                        x1 / orig_w if orig_w > 0 else 0,
                        y1 / orig_h if orig_h > 0 else 0,
                        x2 / orig_w if orig_w > 0 else 0,
                        y2 / orig_h if orig_h > 0 else 0
                    ))
                mask_boxes = normalized_boxes

            # Mask out animal regions if provided
            if mask_boxes:
                img = self._mask_regions(img, mask_boxes)

            # Option: Focus on edges only (mask center where animals usually are)
            # This can help when animal boxes aren't available
            # img = self._focus_on_edges(img, center_mask_pct=0.4)

            # Extract features
            if self._model is not None:
                return self._extract_pytorch(img)
            elif self._session is not None:
                return self._extract_onnx(img)
            else:
                return None

        except Exception as e:
            print(f"[SemanticSiteEmbedder] Extraction failed for {image_path}: {e}")
            return None

    def _mask_regions(
        self,
        img: "Image.Image",
        boxes: List[Tuple[float, float, float, float]]
    ) -> "Image.Image":
        """
        Mask out regions of the image (replace with neutral gray).

        This removes animals/people so the embedding focuses on background.
        Uses a neutral gray that won't bias the features.
        """
        import numpy as np
        from PIL import Image as PILImage

        arr = np.array(img)
        h, w = arr.shape[:2]

        # Neutral gray value (ImageNet mean-ish)
        neutral = np.array([128, 128, 128], dtype=np.uint8)

        for box in boxes:
            x1, y1, x2, y2 = box
            # Convert normalized coords to pixel coords
            px1 = max(0, int(x1 * w))
            py1 = max(0, int(y1 * h))
            px2 = min(w, int(x2 * w))
            py2 = min(h, int(y2 * h))

            # Expand box slightly to ensure animal is fully covered
            pad_x = int((px2 - px1) * 0.1)
            pad_y = int((py2 - py1) * 0.1)
            px1 = max(0, px1 - pad_x)
            py1 = max(0, py1 - pad_y)
            px2 = min(w, px2 + pad_x)
            py2 = min(h, py2 + pad_y)

            # Replace with neutral gray
            arr[py1:py2, px1:px2] = neutral

        return PILImage.fromarray(arr)

    def _focus_on_edges(
        self,
        img: "Image.Image",
        center_mask_pct: float = 0.4
    ) -> "Image.Image":
        """
        Mask out the center of the image to focus on edges/corners.

        Trail cameras typically show animals in the center. The fixed
        background (trees, landmarks) is more visible at the edges.

        Args:
            img: Input image
            center_mask_pct: Fraction of center to mask (0.4 = middle 40%)
        """
        import numpy as np
        from PIL import Image as PILImage

        arr = np.array(img)
        h, w = arr.shape[:2]

        # Calculate center region
        margin_x = int(w * (1 - center_mask_pct) / 2)
        margin_y = int(h * (1 - center_mask_pct) / 2)

        # Mask center with neutral gray
        neutral = np.array([128, 128, 128], dtype=np.uint8)
        arr[margin_y:h-margin_y, margin_x:w-margin_x] = neutral

        return PILImage.fromarray(arr)

    def _extract_pytorch(self, img: "Image.Image") -> np.ndarray:
        """Extract features using PyTorch model."""
        # Transform image
        tensor = self._transform(img).unsqueeze(0)  # Add batch dim
        tensor = tensor.to(self._device)

        # Extract features
        with torch.no_grad():
            # MobileNetV2 with Identity classifier outputs (batch, 1280)
            # after the adaptive avg pool
            features = self._model.features(tensor)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)

        # Convert to numpy and normalize
        emb = features.cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)

    def _extract_onnx(self, img: "Image.Image") -> np.ndarray:
        """Extract features using ONNX runtime."""
        # Resize and preprocess
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std

        # Transpose to (C, H, W) and add batch dim
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, 0).astype(np.float32)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: arr})

        # Normalize output
        emb = outputs[0][0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)

    def embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        """Convert embedding to bytes for database storage."""
        return embedding.astype(np.float32).tobytes()

    def bytes_to_embedding(self, data: bytes) -> np.ndarray:
        """Convert bytes back to embedding."""
        return np.frombuffer(data, dtype=np.float32)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute similarity between two embeddings.

        Returns cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite).
        Since embeddings are L2 normalized, this is just the dot product.
        """
        return float(np.dot(emb1, emb2))

    def export_to_onnx(self, output_path: str):
        """
        Export the PyTorch model to ONNX format for bundling.

        Args:
            output_path: Where to save the .onnx file
        """
        if not TORCH_AVAILABLE or self._model is None:
            print("[SemanticSiteEmbedder] Cannot export - PyTorch model not loaded")
            return False

        try:
            # Create dummy input
            dummy = torch.randn(1, 3, 224, 224)

            # Create a wrapper that includes avgpool
            class FeatureExtractor(nn.Module):
                def __init__(self, mobilenet):
                    super().__init__()
                    self.features = mobilenet.features
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))

                def forward(self, x):
                    x = self.features(x)
                    x = self.pool(x)
                    return x.flatten(1)

            wrapper = FeatureExtractor(self._model)
            wrapper.eval()

            # Export
            torch.onnx.export(
                wrapper,
                dummy,
                output_path,
                input_names=["image"],
                output_names=["embedding"],
                dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
                opset_version=11
            )

            print(f"[SemanticSiteEmbedder] Exported to {output_path}")
            return True

        except Exception as e:
            print(f"[SemanticSiteEmbedder] Export failed: {e}")
            return False


def run_semantic_site_matching(
    db,
    embedder: SemanticSiteEmbedder = None,
    similarity_threshold: float = 0.85,
    min_matches: int = 3,
    progress_callback=None
) -> Dict[str, any]:
    """
    Match unlabeled photos to known camera locations using semantic embeddings.

    This is an improved version of run_site_clustering that uses semantic
    understanding instead of edge-based features.

    Args:
        db: TrailCamDatabase instance
        embedder: SemanticSiteEmbedder instance (created if None)
        similarity_threshold: Min cosine similarity for a match (0.85 = very similar)
        min_matches: Minimum number of similar labeled photos needed
        progress_callback: Optional callback(current, total, message)

    Returns:
        Dict with matching results
    """
    if embedder is None:
        embedder = SemanticSiteEmbedder()

    if not embedder.ready:
        return {"error": "Embedder not ready - check PyTorch/torchvision installation"}

    # Get photos
    all_photos = db.get_all_photos()
    labeled = [p for p in all_photos if p.get("camera_location") and p["camera_location"].strip()]
    unlabeled = [p for p in all_photos if not p.get("camera_location") or not p["camera_location"].strip()]

    if not labeled:
        return {"error": "No labeled photos. Please label some photos with camera locations first."}

    locations = set(p["camera_location"].strip() for p in labeled)
    print(f"[SemanticSite] {len(labeled)} labeled photos, {len(locations)} locations")

    # Step 1: Compute embeddings for all photos
    # Check which need computing (version mismatch or missing)
    total_photos = len(all_photos)
    computed = 0

    if progress_callback:
        progress_callback(0, total_photos, "Computing semantic embeddings...")

    for i, photo in enumerate(all_photos):
        pid = photo["id"]

        # Check if we have an up-to-date embedding
        existing = db.get_embedding(pid)
        if existing:
            _, version = existing
            if version == embedder.EMBEDDING_VERSION:
                continue  # Already have semantic embedding

        # Get animal boxes to mask out (from annotation_boxes table)
        boxes = db.get_boxes(pid) if hasattr(db, 'get_boxes') else []
        mask_boxes = []
        for box in boxes:
            # Boxes are stored as x1, y1, x2, y2 in pixel coordinates
            # We need to get image dimensions to normalize
            x1, y1, x2, y2 = box.get('x1', 0), box.get('y1', 0), box.get('x2', 0), box.get('y2', 0)
            # Store as pixel coords - we'll normalize in extract_embedding
            mask_boxes.append((x1, y1, x2, y2))

        # Extract embedding
        path = photo.get("file_path")
        if path and os.path.exists(path):
            emb = embedder.extract_embedding(path, mask_boxes=mask_boxes)
            if emb is not None:
                db.save_embedding(pid, embedder.embedding_to_bytes(emb), embedder.EMBEDDING_VERSION)
                computed += 1

        if progress_callback and (i + 1) % 20 == 0:
            progress_callback(i + 1, total_photos, f"Computed {computed} new embeddings...")

    # Step 2: Load all embeddings
    if progress_callback:
        progress_callback(0, 1, "Loading embeddings...")

    raw = db.get_all_embeddings()
    all_emb = {}
    for pid, data, version in raw:
        if version == embedder.EMBEDDING_VERSION:
            all_emb[pid] = embedder.bytes_to_embedding(data)

    # Build reference set
    labeled_refs = []
    for p in labeled:
        if p["id"] in all_emb:
            labeled_refs.append((p["id"], all_emb[p["id"]], p["camera_location"].strip()))

    if len(labeled_refs) < min_matches:
        return {"error": f"Need at least {min_matches} labeled photos with embeddings. Have {len(labeled_refs)}."}

    # Step 3: Match unlabeled to labeled
    if progress_callback:
        progress_callback(0, len(unlabeled), "Matching photos to locations...")

    suggestions = 0
    no_match = 0
    by_location = {loc: 0 for loc in locations}

    for i, photo in enumerate(unlabeled):
        pid = photo["id"]
        if pid not in all_emb:
            continue

        emb = all_emb[pid]

        # Compute similarity to all labeled photos
        similarities = []
        for ref_id, ref_emb, ref_loc in labeled_refs:
            sim = embedder.compute_similarity(emb, ref_emb)
            similarities.append((sim, ref_loc, ref_id))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: -x[0])

        # Count matches above threshold per location
        location_matches = {}
        location_best_sim = {}

        for sim, loc, ref_id in similarities:
            if sim >= similarity_threshold:
                location_matches[loc] = location_matches.get(loc, 0) + 1
                if loc not in location_best_sim:
                    location_best_sim[loc] = sim

        # Find best location with enough matches
        best_loc = None
        best_count = 0
        best_sim = 0

        for loc, count in location_matches.items():
            if count >= min_matches:
                if count > best_count or (count == best_count and location_best_sim[loc] > best_sim):
                    best_loc = loc
                    best_count = count
                    best_sim = location_best_sim[loc]

        if best_loc:
            # Confidence based on number of matches and similarity
            confidence = min(1.0, (best_count / 5) * 0.5 + best_sim * 0.5)

            site = db.get_site_by_name(best_loc)
            if site:
                db.set_photo_site_suggestion(pid, site["id"], confidence)
            else:
                site_id = db.create_site(best_loc, confirmed=True)
                db.set_photo_site_suggestion(pid, site_id, confidence)

            suggestions += 1
            by_location[best_loc] = by_location.get(best_loc, 0) + 1
        else:
            no_match += 1

        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1, len(unlabeled), f"{suggestions} matches found...")

    if progress_callback:
        progress_callback(len(unlabeled), len(unlabeled), "Done!")

    return {
        "suggestions_made": suggestions,
        "no_match_count": no_match,
        "total_unlabeled": len(unlabeled),
        "labeled_count": len(labeled),
        "locations": list(locations),
        "results_by_location": by_location,
        "embeddings_computed": computed
    }


# Quick test
if __name__ == "__main__":
    print("Testing SemanticSiteEmbedder...")

    embedder = SemanticSiteEmbedder()
    if embedder.ready:
        print(f"Ready! Embedding dim: {embedder.EMBEDDING_DIM}")

        # Test on a sample image if available
        test_path = os.path.expanduser("~/TrailCamLibrary")
        if os.path.exists(test_path):
            # Find first jpg
            for root, dirs, files in os.walk(test_path):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg')):
                        path = os.path.join(root, f)
                        print(f"Testing on: {path}")
                        emb = embedder.extract_embedding(path)
                        if emb is not None:
                            print(f"Embedding shape: {emb.shape}")
                            print(f"Embedding norm: {np.linalg.norm(emb):.4f}")
                        break
                break
    else:
        print("Embedder not ready - check dependencies")
