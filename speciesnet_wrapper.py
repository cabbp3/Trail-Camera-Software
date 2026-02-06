"""
SpeciesNet integration wrapper.

Wraps Google's SpeciesNet (cameratrapai) to match existing AI pipeline conventions.
Handles model initialization, species label mapping, and geofencing.

SpeciesNet replaces both MegaDetector (detection) and the custom ONNX species
classifier (classification) in a single ensemble call.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Map SpeciesNet scientific/label names to VALID_SPECIES in ai_suggester.py.
# SpeciesNet uses lowercase underscore-separated labels.
SPECIESNET_TO_APP_SPECIES = {
    # Deer
    "odocoileus_virginianus": "Deer",
    "odocoileus_hemionus": "Deer",
    "odocoileus": "Deer",
    "cervidae": "Deer",
    "cervus_elaphus": "Deer",
    # Turkey
    "meleagris_gallopavo": "Turkey",
    "meleagris": "Turkey",
    # Coyote
    "canis_latrans": "Coyote",
    # Fox
    "vulpes_vulpes": "Fox",
    "urocyon_cinereoargenteus": "Fox",
    "vulpes": "Fox",
    # Raccoon
    "procyon_lotor": "Raccoon",
    "procyon": "Raccoon",
    # Bobcat
    "lynx_rufus": "Bobcat",
    "lynx": "Bobcat",
    # Opossum
    "didelphis_virginiana": "Opossum",
    "didelphis": "Opossum",
    # Squirrel
    "sciurus_carolinensis": "Squirrel",
    "sciurus_niger": "Squirrel",
    "sciurus": "Squirrel",
    "tamiasciurus_hudsonicus": "Squirrel",
    # Rabbit
    "sylvilagus_floridanus": "Rabbit",
    "sylvilagus": "Rabbit",
    # Skunk
    "mephitis_mephitis": "Skunk",
    "mephitis": "Skunk",
    # Ground Hog
    "marmota_monax": "Ground Hog",
    "marmota": "Ground Hog",
    # Otter
    "lontra_canadensis": "Otter",
    "lontra": "Otter",
    # Quail
    "colinus_virginianus": "Quail",
    "colinus": "Quail",
    # Armadillo
    "dasypus_novemcinctus": "Armadillo",
    "dasypus": "Armadillo",
    # House Cat
    "felis_catus": "House Cat",
    "felis": "House Cat",
    # Dog
    "canis_lupus_familiaris": "Dog",
    # Chipmunk
    "tamias_striatus": "Chipmunk",
    "tamias": "Chipmunk",
    # Turkey Buzzard / Vulture
    "cathartes_aura": "Turkey Buzzard",
    "coragyps_atratus": "Turkey Buzzard",
    # Person
    "homo_sapiens": "Person",
    "homo": "Person",
    "person": "Person",
    # Flicker
    "colaptes_auratus": "Flicker",
    "colaptes": "Flicker",
    # Family-level mappings (when SpeciesNet only gets to family)
    "sciuridae": "Squirrel",
    "phasianidae": "Turkey",
    # Non-animal classes
    "blank": "Empty",
    "empty": "Empty",
    "vehicle": "Vehicle",
}

# Detection category mapping (SpeciesNet uses string keys)
DETECTION_CATEGORY_MAP = {
    "1": "ai_animal",
    "2": "ai_person",
    "3": "ai_vehicle",
}


class SpeciesNetWrapper:
    """Wraps Google SpeciesNet for trail camera species detection and classification.

    Provides detect_and_classify() which runs the full SpeciesNet ensemble
    (MegaDetector detection + EfficientNet V2 M classification) in one call.
    """

    def __init__(self, state: str = "", geofence: bool = True):
        self._model = None
        self._available = False
        self._error = None
        self._state = state
        self._geofence = geofence

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def error_message(self) -> Optional[str]:
        return self._error

    def initialize(self, progress_callback=None) -> bool:
        """Load SpeciesNet model, downloading weights on first use (~500 MB).

        Args:
            progress_callback: Optional callable(message: str, percent: float)

        Returns:
            True if model loaded successfully.
        """
        try:
            from speciesnet import DEFAULT_MODEL, SpeciesNet

            if progress_callback:
                progress_callback("Loading SpeciesNet model...", 0.0)

            self._model = SpeciesNet(
                model_name=DEFAULT_MODEL,
                components="all",
                geofence=self._geofence,
            )
            self._available = True
            self._error = None
            logger.info("SpeciesNet loaded successfully")
            return True

        except ImportError:
            self._error = (
                "speciesnet not installed. Install with: pip install speciesnet"
            )
            logger.error(self._error)
            return False
        except Exception as e:
            self._error = f"Failed to load SpeciesNet: {e}"
            logger.error(self._error)
            return False

    def detect_and_classify(self, image_path: str) -> Dict:
        """Run full SpeciesNet pipeline on a single image.

        Returns dict with:
            detections: list of {label, x1, y1, x2, y2, confidence}
            app_species: str mapped to VALID_SPECIES (or readable fallback)
            prediction_score: float confidence
            raw_prediction: str original SpeciesNet label
        """
        if not self._available or self._model is None:
            return {
                "detections": [],
                "app_species": None,
                "prediction_score": 0,
                "raw_prediction": None,
            }

        try:
            kwargs = {
                "filepaths": [image_path],
                "run_mode": "single_thread",
                "batch_size": 1,
                "progress_bars": False,
            }
            if self._state:
                kwargs["country"] = "USA"
                kwargs["admin1_region"] = self._state

            results = self._model.predict(**kwargs)

            predictions = results.get("predictions", [])
            if not predictions:
                return {
                    "detections": [],
                    "app_species": "Empty",
                    "prediction_score": 0.95,
                    "raw_prediction": "blank",
                }

            pred = predictions[0]

            # Convert detections to app box format
            raw_detections = pred.get("detections", [])
            boxes = []
            for det in raw_detections:
                cat = str(det.get("category", "1"))
                label = DETECTION_CATEGORY_MAP.get(cat, "ai_animal")
                bbox = det.get("bbox", [0, 0, 0, 0])
                # SpeciesNet bbox: [x_min, y_min, width, height] normalized
                x, y, w, h = bbox
                boxes.append({
                    "label": label,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "confidence": det.get("conf", 0),
                })

            raw_label = pred.get("prediction", "")
            app_species = self._map_species(raw_label)
            score = pred.get("prediction_score", 0)

            return {
                "detections": boxes,
                "app_species": app_species,
                "prediction_score": score if score else 0,
                "raw_prediction": raw_label,
            }

        except Exception as e:
            logger.error(f"SpeciesNet prediction failed for {image_path}: {e}")
            return {
                "detections": [],
                "app_species": None,
                "prediction_score": 0,
                "raw_prediction": None,
            }

    def _map_species(self, speciesnet_label: str) -> str:
        """Map a SpeciesNet label to the app's VALID_SPECIES set.

        SpeciesNet returns taxonomy strings like:
            uuid;Class;Order;Family;Genus;Species;Common Name
        e.g. "5c7ce...;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"

        Strategy:
        1. Parse taxonomy string → extract genus_species
        2. Direct lookup in SPECIESNET_TO_APP_SPECIES
        3. Try genus-level match
        4. Try common name match
        5. Fall back to common name or title-cased readable name
        """
        if not speciesnet_label:
            return "Empty"

        label_lower = speciesnet_label.lower().strip()

        # Direct lookup (handles simple labels like "blank", "empty", "person")
        if label_lower in SPECIESNET_TO_APP_SPECIES:
            return SPECIESNET_TO_APP_SPECIES[label_lower]

        # Parse semicolon-separated taxonomy: uuid;class;order;family;genus;species;common_name
        if ";" in label_lower:
            parts = label_lower.split(";")
            # Extract genus (index 4) and species (index 5) if available
            genus = parts[4].strip() if len(parts) > 4 else ""
            species = parts[5].strip() if len(parts) > 5 else ""
            common_name = parts[6].strip() if len(parts) > 6 else ""

            # Try genus_species lookup
            if genus and species:
                key = f"{genus}_{species}"
                if key in SPECIESNET_TO_APP_SPECIES:
                    return SPECIESNET_TO_APP_SPECIES[key]

            # Try genus-only lookup
            if genus and genus in SPECIESNET_TO_APP_SPECIES:
                return SPECIESNET_TO_APP_SPECIES[genus]

            # Try family-level lookup (index 3)
            family = parts[3].strip() if len(parts) > 3 else ""
            if family and family in SPECIESNET_TO_APP_SPECIES:
                return SPECIESNET_TO_APP_SPECIES[family]

            # Try common name keywords
            if common_name:
                common_lower = common_name.lower()
                for keyword, app_label in [
                    ("deer", "Deer"), ("turkey", "Turkey"), ("coyote", "Coyote"),
                    ("fox", "Fox"), ("raccoon", "Raccoon"), ("bobcat", "Bobcat"),
                    ("opossum", "Opossum"), ("squirrel", "Squirrel"),
                    ("rabbit", "Rabbit"), ("skunk", "Skunk"), ("groundhog", "Ground Hog"),
                    ("woodchuck", "Ground Hog"), ("otter", "Otter"),
                    ("quail", "Quail"), ("armadillo", "Armadillo"),
                    ("cat", "House Cat"), ("dog", "Dog"), ("vulture", "Turkey Buzzard"),
                    ("chipmunk", "Chipmunk"), ("human", "Person"),
                ]:
                    if keyword in common_lower:
                        return app_label
                # Return readable common name for unmapped species
                return common_name.title()

            # Return readable genus species
            if genus and species:
                return f"{genus.title()} {species.title()}"
            if genus:
                return genus.title()

        # Underscore-separated fallback (original format)
        parts = label_lower.split("_")
        if parts[0] in SPECIESNET_TO_APP_SPECIES:
            return SPECIESNET_TO_APP_SPECIES[parts[0]]

        return speciesnet_label.replace("_", " ").title()

    def set_state(self, state: str):
        """Update the US state for geofencing."""
        self._state = state
