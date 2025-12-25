"""
AI suggestion scaffolding (species tags).

Uses ONNX classifier trained on user's trail camera photos.
Models expected at:
  - models/species.onnx + labels.txt - Species classification
  - models/buckdoe.onnx - Buck/doe classification (optional)
  - models/reid.onnx - Deer re-ID embeddings (optional)
"""
import os
import sys
from typing import Optional, Tuple, List


def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and PyInstaller bundle."""
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    # Running in development
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

try:
    import onnxruntime as ort
    import numpy as np
    from PIL import Image
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None
    np = None
    Image = None

# Master list of valid species labels - predictions outside this set are rejected
VALID_SPECIES = {
    "Deer", "Empty", "Other", "Other_Mammal", "Other Mammal", "Turkey",
    "Bobcat", "Coyote", "Opossum", "Other Bird", "Person",
    "Quail", "Rabbit", "Raccoon", "Squirrel", "Vehicle", "Skunk"
}


class SpeciesSuggester:
    """
    ONNX species classifier trained on user's trail camera photos.

    Model expectations:
      - A classification ONNX model that takes a 1x3xHxW float input.
      - Outputs a single logits/probability vector.
      - labels.txt alongside the model lists labels in output order.
    """

    def __init__(self, model_path: str = None, labels_path: str = None):
        self.model_path = model_path or get_resource_path("models/species.onnx")
        self.labels_path = labels_path or get_resource_path("models/labels.txt")
        self.session = None
        self.labels = []
        self.input_size = (224, 224)
        self._ready = False
        if ONNX_AVAILABLE and os.path.exists(self.model_path):
            self._load_model()

    def _load_model(self):
        try:
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            if os.path.exists(self.labels_path):
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    raw_labels = [line.strip() for line in f if line.strip()]
                # Only keep labels that are in VALID_SPECIES
                self.labels = [lbl for lbl in raw_labels if lbl in VALID_SPECIES]
                if len(self.labels) != len(raw_labels):
                    invalid = [lbl for lbl in raw_labels if lbl not in VALID_SPECIES]
                    print(f"[AI] Warning: Filtered out invalid labels: {invalid}")
            if not self.labels:
                self.labels = ["Deer", "Turkey", "Coyote", "Raccoon", "Person", "Vehicle", "Empty"]
            self._ready = True
            print(f"[AI] Loaded species model with {len(self.labels)} labels")
        except Exception as exc:
            print(f"[AI] Failed to load species model: {exc}")
            self.session = None
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self, image_path: str) -> Optional[Tuple[str, float]]:
        """Predict species from an image. Returns (label, confidence) or None."""
        if not self.ready or not os.path.exists(image_path):
            return None
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(self.input_size)
            arr = np.array(img).astype("float32") / 255.0
            # Apply ImageNet normalization (same as training)
            mean = np.array([0.485, 0.456, 0.406], dtype="float32")
            std = np.array([0.229, 0.224, 0.225], dtype="float32")
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)
            arr = np.expand_dims(arr, 0)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: arr})
            logits = outputs[0][0]
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            idx = int(np.argmax(probs))
            label = self.labels[idx] if idx < len(self.labels) else None
            if label is None or label not in VALID_SPECIES:
                print(f"[AI] Rejected invalid prediction: {label}")
                return None
            conf = float(probs[idx])
            return label, conf
        except Exception as exc:
            print(f"[AI] Species prediction failed for {image_path}: {exc}")
            return None


class BuckDoeSuggester:
    """
    Buck/Doe classifier for deer images.
    Expects models/buckdoe.onnx that outputs class logits for [buck, doe].
    Best used on deer_head crops.
    """

    def __init__(self, model_path: str = None, labels_path: str = None):
        self.model_path = model_path or get_resource_path("models/buckdoe.onnx")
        self.labels_path = labels_path or get_resource_path("models/buckdoe_labels.txt")
        self.session = None
        self.labels = ["buck", "doe"]
        self._ready = False
        self.input_size = (224, 224)
        if ONNX_AVAILABLE and os.path.exists(self.model_path):
            self._load_model()

    def _load_model(self):
        try:
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            if os.path.exists(self.labels_path):
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = [line.strip() for line in f if line.strip()]
            self._ready = True
            print(f"[AI] Loaded buck/doe model")
        except Exception as exc:
            print(f"[AI] Failed to load buck/doe model: {exc}")
            self.session = None
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self, image_path: str) -> Optional[Tuple[str, float]]:
        """Predict buck or doe from a deer image (preferably head crop)."""
        if not self.ready or not os.path.exists(image_path):
            return None
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(self.input_size)
            arr = np.array(img).astype("float32") / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype="float32")
            std = np.array([0.229, 0.224, 0.225], dtype="float32")
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)
            arr = np.expand_dims(arr, 0)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: arr})
            logits = outputs[0][0]
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            idx = int(np.argmax(probs))
            label = self.labels[idx] if idx < len(self.labels) else str(idx)
            conf = float(probs[idx])
            return label, conf
        except Exception as exc:
            print(f"[AI] Buck/doe prediction failed for {image_path}: {exc}")
            return None


class ReIDSuggester:
    """
    Deer re-ID embedding loader (ONNX).
    Expects models/reid.onnx that outputs an embedding vector.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or get_resource_path("models/reid.onnx")
        self.session = None
        self._ready = False
        self.input_size = (224, 224)
        if ONNX_AVAILABLE and os.path.exists(self.model_path):
            self._load_model()

    def _load_model(self):
        try:
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_size = (shape[3], shape[2])
            self._ready = True
        except Exception as exc:
            print(f"[AI] Failed to load re-ID model: {exc}")
            self.session = None
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def embed(self, image_path: str) -> Optional[List[float]]:
        """Return a normalized embedding vector for a deer image."""
        if not self.ready or not os.path.exists(image_path):
            return None
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(self.input_size)
            arr = np.array(img).astype("float32") / 255.0
            arr = arr.transpose(2, 0, 1)
            arr = np.expand_dims(arr, 0)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: arr})
            emb = outputs[0][0]
            norm = float(np.linalg.norm(emb)) or 1.0
            return (emb / norm).tolist()
        except Exception as exc:
            print(f"[AI] ReID embedding failed for {image_path}: {exc}")
            return None


class CombinedSuggester:
    """Combined AI suggester with species, buck/doe, and re-ID models."""

    def __init__(self):
        self.species = SpeciesSuggester()
        self.buckdoe = BuckDoeSuggester()
        self.reid = ReIDSuggester()
        # Backwards compatibility aliases
        self.onnx = self.species

    @property
    def ready(self) -> bool:
        return self.species.ready

    @property
    def buckdoe_ready(self) -> bool:
        return self.buckdoe.ready

    @property
    def reid_ready(self) -> bool:
        return self.reid.ready

    def predict(self, image_path: str) -> Optional[Tuple[str, float]]:
        """Predict species from an image."""
        return self.species.predict(image_path)

    def predict_sex(self, image_path: str) -> Optional[Tuple[str, float]]:
        """Predict buck or doe from a deer image (preferably head crop)."""
        return self.buckdoe.predict(image_path)

    def embed_deer(self, image_path: str) -> Optional[List[float]]:
        """Get embedding for deer re-ID."""
        return self.reid.embed(image_path)


# Backwards compatibility - keep old class name working
ONNXSuggester = SpeciesSuggester
