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
    
    # Category mappings
    CATEGORIES = {
        1: 'animal',
        2: 'person',
        3: 'vehicle'
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
        
        if MEGADETECTOR_AVAILABLE:
            self._load_model(model_path)
    
    # MegaDetector v5a download URL
    MODEL_URL = "https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
    MODEL_SIZE_MB = 281  # Approximate size for progress display

    def _download_model(self, model_path: Path) -> bool:
        """Download MegaDetector model from GitHub releases"""
        print(f"Downloading MegaDetector v5a model ({self.MODEL_SIZE_MB} MB)...")
        print(f"From: {self.MODEL_URL}")

        # Ensure directory exists
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
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            return False

    def _load_model(self, model_path: Optional[str] = None):
        """Load MegaDetector model, downloading if necessary"""
        if model_path is None:
            model_path = Path.home() / '.trailcam' / 'md_v5a.0.0.pt'

        self.model_path = Path(model_path)

        # Auto-download if not present
        if not self.model_path.exists():
            print(f"MegaDetector model not found at {self.model_path}")
            if not self._download_model(self.model_path):
                print("Failed to download MegaDetector model.")
                return

        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                       path=str(self.model_path), force_reload=False)
            self.model.to(self.device)
            self.model.conf = self.confidence_threshold
            print(f"MegaDetector loaded on {self.device}")
        except Exception as e:
            print(f"Error loading MegaDetector: {e}")
    
    def detect(self, image_path: str) -> List[Detection]:
        """Run detection on an image"""
        if self.model is None:
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
                img = Image.open(image_path)
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
