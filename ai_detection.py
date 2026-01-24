"""AI-powered animal detection and identification for trail-cam photos"""

import os
import json
import requests
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configuration constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_API_TIMEOUT = 30
MAX_ZOOM_SCALE = 10.0
MIN_ZOOM_SCALE = 0.1

try:
    import torch
    from PIL import Image
    MEGADETECTOR_AVAILABLE = True
except ImportError:
    MEGADETECTOR_AVAILABLE = False
    print("Warning: torch not available. Install with: pip install torch torchvision")


@dataclass
class Detection:
    """Represents a single detection result"""
    category: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, width, height (normalized 0-1)


@dataclass
class AnimalIdentification:
    """Represents individual animal re-identification"""
    individual_id: str
    confidence: float
    species: str


class MegaDetectorV5:
    """Local MegaDetector v5 inference"""

    # Category mappings (0-indexed when loaded via torch.hub)
    CATEGORIES = {
        0: 'animal',
        1: 'person',
        2: 'vehicle'
    }
    
    # Species heuristics (can be improved with additional classifiers)
    SPECIES_KEYWORDS = {
        'deer': ['deer', 'whitetail', 'buck', 'doe'],
        'turkey': ['turkey', 'bird'],
        'coyote': ['coyote', 'canine', 'dog'],
    }
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_available = False
        self.error_message = None

        if MEGADETECTOR_AVAILABLE:
            self._load_model(model_path)
        else:
            self.error_message = "PyTorch not installed. Install with: pip install torch torchvision"

    # MegaDetector v5a download URLs (prefer newer version)
    MODEL_URLS = {
        'md_v5a.0.1.pt': "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.1.pt",
        'md_v5a.0.0.pt': "https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt",
    }
    MODEL_SIZE_MB = 281  # Approximate size for progress display

    def _download_model(self, model_path: Path) -> bool:
        """Download MegaDetector model from GitHub releases"""
        model_name = model_path.name
        url = self.MODEL_URLS.get(model_name, self.MODEL_URLS['md_v5a.0.0.pt'])

        print(f"Downloading MegaDetector v5a model ({self.MODEL_SIZE_MB} MB)...")
        print(f"From: {url}")

        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r  Downloaded {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)

            print(f"\n  Model saved to {model_path}")
            return True

        except Exception as e:
            print(f"\n  Error downloading model: {e}")
            self.error_message = f"Failed to download MegaDetector model: {e}"
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            return False

    def _load_model(self, model_path: Optional[str] = None):
        """Load MegaDetector model, downloading if necessary"""
        if model_path is None:
            # Prefer newer model version, fall back to older
            model_path = Path.home() / '.trailcam' / 'md_v5a.0.1.pt'
            if not model_path.exists():
                model_path = Path.home() / '.trailcam' / 'md_v5a.0.0.pt'

        self.model_path = Path(model_path)

        # Auto-download if not present
        if not self.model_path.exists():
            print(f"MegaDetector model not found at {self.model_path}")
            if not self._download_model(self.model_path):
                print("Failed to download MegaDetector model.")
                self.error_message = f"MegaDetector model not found at {self.model_path} and download failed"
                return

        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                       path=str(self.model_path), force_reload=False)
            self.model.to(self.device)
            self.model.conf = self.confidence_threshold
            self.is_available = True
            self.error_message = None
            print(f"MegaDetector loaded on {self.device}")
        except Exception as e:
            print(f"Error loading MegaDetector: {e}")
            self.error_message = f"Failed to load MegaDetector model: {e}"
            self.is_available = False
    
    def detect(self, image_path: str) -> List[Detection]:
        """Run detection on an image. Returns empty list if model not available."""
        if self.model is None:
            if self.error_message:
                print(f"MegaDetector not available: {self.error_message}")
            return []
        
        try:
            # Run inference
            results = self.model(image_path)
            detections = []
            
            # Parse results
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = box
                category_id = int(cls)
                category = self.CATEGORIES.get(category_id, 'unknown')

                # Get image dimensions for normalization
                with Image.open(image_path) as img:
                    w, h = img.size

                # Normalize bounding box
                bbox = (x1/w, y1/h, (x2-x1)/w, (y2-y1)/h)
                
                detections.append(Detection(
                    category=category,
                    confidence=float(conf),
                    bbox=bbox
                ))
            
            return detections
        except Exception as e:
            print(f"Error detecting in {image_path}: {e}")
            return []
    
    def classify_species(self, image_path: str, detection: Detection) -> str:
        """Simple species classification (can be enhanced)"""
        if detection.category != 'animal':
            return detection.category
        
        # Default to 'animal' - this is where you'd add a species classifier
        # For now, return generic 'animal' or use filename heuristics
        filename_lower = os.path.basename(image_path).lower()
        
        for species, keywords in self.SPECIES_KEYWORDS.items():
            if any(kw in filename_lower for kw in keywords):
                return species
        
        return 'animal'  # Generic fallback


class MegaDetectorV6:
    """MegaDetector v6 using Ultralytics YOLO - smaller and faster than v5"""

    # Category mappings (same as v5)
    CATEGORIES = {
        0: 'animal',
        1: 'person',
        2: 'vehicle'
    }

    # Download URL for v6 compact model
    MODEL_URL = "https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1"
    MODEL_SIZE_MB = 49  # Much smaller than v5's 268 MB

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.2):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_available = False
        self.error_message = None

        # Check for ultralytics
        try:
            from ultralytics import YOLO
            self._YOLO = YOLO
            self._load_model(model_path)
        except ImportError:
            self.error_message = "ultralytics not installed. Install with: pip install ultralytics"
            print(f"Warning: {self.error_message}")

    def _download_model(self, model_path: Path) -> bool:
        """Download MegaDetector v6 model from Zenodo"""
        print(f"Downloading MegaDetector v6 compact model ({self.MODEL_SIZE_MB} MB)...")
        print(f"From: {self.MODEL_URL}")

        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(self.MODEL_URL, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r  Downloaded {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)

            print(f"\n  Model saved to {model_path}")
            return True

        except Exception as e:
            print(f"\n  Error downloading model: {e}")
            self.error_message = f"Failed to download MegaDetector v6 model: {e}"
            if model_path.exists():
                model_path.unlink()
            return False

    def _load_model(self, model_path: Optional[str] = None):
        """Load MegaDetector v6 model, downloading if necessary"""
        if model_path is None:
            model_path = Path.home() / '.trailcam' / 'MDV6-yolov9-c.pt'

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            print(f"MegaDetector v6 model not found at {self.model_path}")
            if not self._download_model(self.model_path):
                return

        try:
            self.model = self._YOLO(str(self.model_path))
            self.is_available = True
            self.error_message = None
            print(f"MegaDetector v6 loaded ({self.MODEL_SIZE_MB} MB model)")
        except Exception as e:
            print(f"Error loading MegaDetector v6: {e}")
            self.error_message = f"Failed to load MegaDetector v6: {e}"
            self.is_available = False

    def detect(self, image_path: str) -> List[Detection]:
        """Run detection on an image. Returns empty list if model not available."""
        if self.model is None:
            if self.error_message:
                print(f"MegaDetector v6 not available: {self.error_message}")
            return []

        try:
            # Run inference with ultralytics
            results = self.model.predict(image_path, conf=self.confidence_threshold, verbose=False)
            detections = []

            # Get image dimensions
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            w, h = img.size

            # Parse results (ultralytics format)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    category = self.CATEGORIES.get(cls, 'unknown')

                    # Normalize bounding box to [0, 1]
                    bbox = (x1/w, y1/h, (x2-x1)/w, (y2-y1)/h)

                    detections.append(Detection(
                        category=category,
                        confidence=conf,
                        bbox=bbox
                    ))

            return detections
        except Exception as e:
            print(f"Error detecting in {image_path}: {e}")
            return []


class MegaDetectorV6ONNX:
    """MegaDetector v6 using ONNX - works without PyTorch for Windows builds"""

    CATEGORIES = {0: 'animal', 1: 'person', 2: 'vehicle'}
    MODEL_URL = "https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1"
    ONNX_SIZE_MB = 97

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.2):
        self.confidence_threshold = confidence_threshold
        self.session = None
        self.is_available = False
        self.error_message = None
        self.input_size = 640

        try:
            import onnxruntime as ort
            self._ort = ort
            self._load_model(model_path)
        except ImportError:
            self.error_message = "onnxruntime not installed. Install with: pip install onnxruntime"
            print(f"Warning: {self.error_message}")

    def _load_model(self, model_path: Optional[str] = None):
        """Load ONNX model"""
        if model_path is None:
            model_path = Path.home() / '.trailcam' / 'MDV6-yolov9-c.onnx'

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            self.error_message = f"ONNX model not found at {self.model_path}. Export using: model.export(format='onnx')"
            print(f"MegaDetector v6 ONNX: {self.error_message}")
            return

        try:
            self.session = self._ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.is_available = True
            self.error_message = None
            print(f"MegaDetector v6 ONNX loaded ({self.ONNX_SIZE_MB} MB model)")
        except Exception as e:
            print(f"Error loading MegaDetector v6 ONNX: {e}")
            self.error_message = f"Failed to load ONNX model: {e}"
            self.is_available = False

    def _nms(self, boxes, scores, iou_threshold=0.5):
        """Non-Maximum Suppression to filter overlapping boxes"""
        if len(boxes) == 0:
            return []

        # Convert to numpy arrays
        boxes = np.array(boxes)
        scores = np.array(scores)

        # Sort by score
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = area_i + area_j - inter

            iou = inter / (union + 1e-6)

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image_path: str) -> List[Detection]:
        """Run detection using ONNX model"""
        if self.session is None:
            if self.error_message:
                print(f"MegaDetector v6 ONNX not available: {self.error_message}")
            return []

        try:
            from PIL import Image as PILImage

            # Load and preprocess image
            img = PILImage.open(image_path).convert('RGB')
            orig_w, orig_h = img.size
            img_resized = img.resize((self.input_size, self.input_size))

            arr = np.array(img_resized).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            arr = np.expand_dims(arr, 0)  # Add batch dim

            # Run inference
            outputs = self.session.run(None, {self.input_name: arr})

            # Parse YOLOv9 output: [batch, 7, 8400] -> [7, 8400]
            # 7 = 4 (x_center, y_center, w, h) + 3 (animal, person, vehicle scores)
            output = outputs[0][0].T  # [8400, 7]

            boxes_raw = output[:, :4]  # x_center, y_center, w, h (in 640x640 space)
            class_scores = output[:, 4:]

            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)

            # Filter by confidence
            mask = confidences > self.confidence_threshold
            boxes_filtered = boxes_raw[mask]
            classes_filtered = class_ids[mask]
            confs_filtered = confidences[mask]

            if len(boxes_filtered) == 0:
                return []

            # Convert center format to corner format for NMS
            x_center, y_center, w, h = boxes_filtered.T
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            # Apply NMS per class
            detections = []
            for cls_id in range(3):  # animal, person, vehicle
                cls_mask = classes_filtered == cls_id
                if not np.any(cls_mask):
                    continue

                cls_boxes = boxes_xyxy[cls_mask]
                cls_confs = confs_filtered[cls_mask]

                keep = self._nms(cls_boxes, cls_confs, iou_threshold=0.5)

                for idx in keep:
                    box = cls_boxes[idx]
                    conf = cls_confs[idx]

                    # Convert from 640x640 to normalized [0,1] then to original image coords
                    x1_norm = box[0] / self.input_size
                    y1_norm = box[1] / self.input_size
                    x2_norm = box[2] / self.input_size
                    y2_norm = box[3] / self.input_size

                    # Bbox in (x, y, w, h) normalized format
                    bbox = (x1_norm, y1_norm, x2_norm - x1_norm, y2_norm - y1_norm)

                    detections.append(Detection(
                        category=self.CATEGORIES.get(cls_id, 'unknown'),
                        confidence=float(conf),
                        bbox=bbox
                    ))

            return detections

        except Exception as e:
            print(f"Error detecting in {image_path}: {e}")
            return []


class WildlifeAIClient:
    """Client for Wildlife.ai individual re-identification API"""
    
    API_URL = "https://api.wildlife.ai/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('WILDLIFE_AI_API_KEY')
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
    
    def identify_individual(self, image_crop: np.ndarray, species: str = 'deer') -> Optional[AnimalIdentification]:
        """Identify individual animal from cropped image"""
        if not self.api_key:
            print("Wildlife.ai API key not set. Set WILDLIFE_AI_API_KEY environment variable.")
            return None
        
        try:
            # Convert crop to JPEG bytes
            _, buffer = cv2.imencode('.jpg', image_crop)
            
            # Prepare multipart upload
            files = {'image': ('crop.jpg', buffer.tobytes(), 'image/jpeg')}
            data = {'species': species}
            
            # Call API
            response = self.session.post(
                f"{self.API_URL}/identify",
                files=files,
                data=data,
                timeout=DEFAULT_API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return AnimalIdentification(
                    individual_id=result.get('individual_id', 'Unknown'),
                    confidence=result.get('confidence', 0.0),
                    species=species
                )
            else:
                print(f"Wildlife.ai API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error calling Wildlife.ai API: {e}")
            return None


class AIDetectionManager:
    """Manages AI detection pipeline"""

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.detector = MegaDetectorV5(confidence_threshold=confidence_threshold)
        self.wildlife_client = WildlifeAIClient()
        self.confidence_threshold = confidence_threshold
    
    def process_image(self, image_path: str) -> Dict:
        """Process single image with AI detection"""
        results = {
            'image_path': image_path,
            'detections': [],
            'auto_tags': [],
            'individuals': []
        }
        
        # Run MegaDetector
        detections = self.detector.detect(image_path)
        
        if not detections:
            results['auto_tags'].append(('empty', 1.0))
            return results
        
        # Load image for cropping
        img = cv2.imread(image_path)
        if img is None:
            return results
        
        h, w = img.shape[:2]
        
        for detection in detections:
            # Add detection to results
            results['detections'].append({
                'category': detection.category,
                'confidence': detection.confidence,
                'bbox': detection.bbox
            })
            
            # Auto-tag if confidence is high enough
            if detection.confidence >= self.confidence_threshold:
                species = self.detector.classify_species(image_path, detection)
                results['auto_tags'].append((species, detection.confidence))
                
                # If it's a deer, try individual re-ID
                if species == 'deer' and self.wildlife_client.api_key:
                    # Crop bounding box
                    x, y, box_w, box_h = detection.bbox
                    x1, y1 = int(x * w), int(y * h)
                    x2, y2 = int((x + box_w) * w), int((y + box_h) * h)
                    
                    crop = img[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        # Call Wildlife.ai API
                        individual = self.wildlife_client.identify_individual(crop, 'deer')
                        if individual:
                            results['individuals'].append({
                                'individual_id': individual.individual_id,
                                'confidence': individual.confidence,
                                'species': individual.species,
                                'bbox': detection.bbox
                            })
        
        return results
    
    def process_folder(self, folder_path: str, callback=None) -> List[Dict]:
        """Process all images in a folder"""
        folder = Path(folder_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        results = []
        image_files = [f for f in folder.rglob('*') if f.suffix.lower() in image_extensions]
        
        for i, image_file in enumerate(image_files):
            result = self.process_image(str(image_file))
            results.append(result)
            
            if callback:
                callback(i + 1, len(image_files), result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save detection results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
