"""
Stamp Reader - Pattern-based OCR for trail camera stamps

Automatically detects dates, times, temperatures, etc. from photo stamps.
Learns location names as you identify them.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QComboBox, QMessageBox, QFileDialog, QGroupBox,
    QListWidget, QListWidgetItem, QWidget, QScrollArea,
    QLineEdit, QFormLayout, QFrame
)

# Storage for learned patterns
PATTERNS_FILE = Path.home() / ".trailcam" / "stamp_patterns.json"


class StampField:
    """Represents a detected field in the stamp."""
    def __init__(self, text: str, field_type: str, confidence: float = 1.0):
        self.text = text
        self.field_type = field_type  # "date", "time", "temp", "location", "photo_num", "moon", "unknown"
        self.confidence = confidence


class StampReader(QDialog):
    """Dialog for reading and learning stamp patterns."""

    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Stamp Reader")
        self.resize(1000, 700)

        self.current_photo_path = None
        self.current_camera_model = None
        self.pixmap_item = None
        self.detected_fields: List[StampField] = []
        self.learned_patterns = self._load_patterns()

        self._setup_ui()

    def _load_patterns(self) -> Dict:
        """Load learned patterns from file."""
        if PATTERNS_FILE.exists():
            try:
                with open(PATTERNS_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {
            "locations": [],  # Known location names
            "camera_brands": ["Cuddeback", "Cuddelink", "STEALTH CAM", "Bushnell", "Moultrie"],
            "ignore_words": ["AM", "PM", "F", "C"],  # Words to skip
        }

    def _save_patterns(self):
        """Save learned patterns to file."""
        PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PATTERNS_FILE, 'w') as f:
            json.dump(self.learned_patterns, f, indent=2)

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # Left side - Image view
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Photo selection row
        photo_row = QHBoxLayout()

        # Camera dropdown
        photo_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(150)
        self.camera_combo.addItem("Select Camera...")
        self._populate_camera_samples()
        self.camera_combo.currentIndexChanged.connect(self._load_camera_sample)
        photo_row.addWidget(self.camera_combo)

        photo_row.addStretch()

        load_btn = QPushButton("Load Photo")
        load_btn.clicked.connect(self._load_photo)
        photo_row.addWidget(load_btn)

        use_current_btn = QPushButton("Use Current Photo")
        use_current_btn.clicked.connect(self._load_current_photo)
        photo_row.addWidget(use_current_btn)

        left_layout.addLayout(photo_row)

        # Photo info
        self.photo_label = QLabel("No photo loaded")
        self.photo_label.setStyleSheet("color: #888; padding: 5px;")
        left_layout.addWidget(self.photo_label)

        # Graphics view for stamp area
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumSize(500, 200)
        self.view.setMaximumHeight(250)
        self.view.setStyleSheet("background-color: #1a1a1a;")
        left_layout.addWidget(self.view)

        # OCR Result
        ocr_group = QGroupBox("Raw OCR Text")
        ocr_layout = QVBoxLayout(ocr_group)
        self.ocr_text_label = QLabel("Load a photo to see OCR results")
        self.ocr_text_label.setWordWrap(True)
        self.ocr_text_label.setStyleSheet("font-family: monospace; padding: 10px; background: #2a2a2a; border-radius: 5px;")
        self.ocr_text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ocr_layout.addWidget(self.ocr_text_label)
        left_layout.addWidget(ocr_group)

        # Detected fields
        detected_group = QGroupBox("Detected Fields")
        detected_layout = QVBoxLayout(detected_group)

        self.fields_list = QListWidget()
        self.fields_list.setMinimumHeight(150)
        detected_layout.addWidget(self.fields_list)

        left_layout.addWidget(detected_group)

        layout.addWidget(left_panel, stretch=2)

        # Right side - Controls and learning
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Instructions
        instructions = QLabel(
            "How it works:\n\n"
            "1. Select a camera to load a sample photo\n"
            "2. The system reads the stamp and auto-detects:\n"
            "   • Dates (01/15/2025)\n"
            "   • Times (10:30 AM)\n"
            "   • Temperatures (45°F)\n"
            "   • Photo numbers (001)\n"
            "3. Unknown text appears in yellow\n"
            "4. Click 'Mark as Location' to teach it\n"
            "5. It remembers for next time!"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #aaa; padding: 10px; background: #2a2a2a; border-radius: 5px;")
        right_layout.addWidget(instructions)

        # Unknown text actions
        unknown_group = QGroupBox("Unknown Text")
        unknown_layout = QVBoxLayout(unknown_group)

        self.unknown_combo = QComboBox()
        self.unknown_combo.addItem("Select unknown text...")
        unknown_layout.addWidget(self.unknown_combo)

        mark_location_btn = QPushButton("Mark as Location")
        mark_location_btn.setStyleSheet("background-color: #1565c0; color: white; padding: 8px;")
        mark_location_btn.clicked.connect(self._mark_as_location)
        unknown_layout.addWidget(mark_location_btn)

        mark_ignore_btn = QPushButton("Ignore This Text")
        mark_ignore_btn.clicked.connect(self._mark_as_ignore)
        unknown_layout.addWidget(mark_ignore_btn)

        right_layout.addWidget(unknown_group)

        # Known locations
        locations_group = QGroupBox("Known Locations")
        locations_layout = QVBoxLayout(locations_group)

        self.locations_list = QListWidget()
        self.locations_list.setMaximumHeight(150)
        self._refresh_locations_list()
        locations_layout.addWidget(self.locations_list)

        remove_loc_btn = QPushButton("Remove Selected")
        remove_loc_btn.clicked.connect(self._remove_location)
        locations_layout.addWidget(remove_loc_btn)

        right_layout.addWidget(locations_group)

        # Action buttons
        btn_row = QHBoxLayout()

        reread_btn = QPushButton("Re-read Stamp")
        reread_btn.clicked.connect(self._read_stamp)
        btn_row.addWidget(reread_btn)

        apply_btn = QPushButton("Apply to Database")
        apply_btn.setStyleSheet("background-color: #2e7d32; color: white; padding: 8px;")
        apply_btn.clicked.connect(self._apply_to_database)
        btn_row.addWidget(apply_btn)

        right_layout.addLayout(btn_row)

        right_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        right_layout.addWidget(close_btn)

        layout.addWidget(right_panel, stretch=1)

    def _populate_camera_samples(self):
        """Populate dropdown with camera models."""
        self.camera_samples = {}

        if not self.db:
            return

        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT camera_model, file_path
            FROM photos
            WHERE camera_model IS NOT NULL
            AND camera_model != ''
            AND LENGTH(camera_model) < 20
            AND camera_model NOT LIKE '%/%'
            AND camera_model NOT LIKE '%:%'
            GROUP BY camera_model
            ORDER BY COUNT(*) DESC
            LIMIT 15
        """)

        for row in cursor.fetchall():
            model = row[0].strip()
            path = row[1]
            if model and path and os.path.exists(path):
                self.camera_samples[model] = path
                # Count photos
                cursor.execute("SELECT COUNT(*) FROM photos WHERE camera_model = ?", (model,))
                count = cursor.fetchone()[0]
                self.camera_combo.addItem(f"{model} ({count} photos)")

    def _load_camera_sample(self, index: int):
        """Load sample photo for selected camera."""
        if index <= 0:
            return

        text = self.camera_combo.currentText()
        model = text.split(" (")[0] if " (" in text else text

        if model in self.camera_samples:
            self.current_camera_model = model
            self._display_photo(self.camera_samples[model])

    def _load_photo(self):
        """Load a photo from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Photo", "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self._display_photo(file_path)

    def _load_current_photo(self):
        """Load currently displayed photo from main app."""
        if self.parent() and hasattr(self.parent(), 'photos') and hasattr(self.parent(), 'index'):
            photos = self.parent().photos
            index = self.parent().index
            if photos and 0 <= index < len(photos):
                photo = photos[index]
                path = photo.get("file_path")
                if path and os.path.exists(path):
                    self.current_camera_model = photo.get("camera_model", "")
                    self._display_photo(path)
                    return
        QMessageBox.warning(self, "No Photo", "No current photo available.")

    def _display_photo(self, path: str):
        """Display photo stamp area and run OCR."""
        self.current_photo_path = path
        self.photo_label.setText(f"Photo: {os.path.basename(path)}")
        self.photo_label.setStyleSheet("color: white; padding: 5px;")

        # Load and show stamp area (bottom 20%)
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", f"Could not load: {path}")
            return

        # Crop to stamp area
        h = pixmap.height()
        stamp_height = int(h * 0.20)
        stamp_pixmap = pixmap.copy(0, h - stamp_height, pixmap.width(), stamp_height)

        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(stamp_pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(stamp_pixmap.rect().toRectF())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Run OCR
        self._read_stamp()

    def _read_stamp(self):
        """Read stamp using OCR and detect patterns."""
        if not self.current_photo_path:
            return

        try:
            import pytesseract
            from PIL import Image, ImageOps
        except ImportError:
            QMessageBox.warning(self, "Error", "pytesseract not installed.\nRun: pip install pytesseract")
            return

        try:
            img = Image.open(self.current_photo_path).convert("RGB")
            w, h = img.size
            # Crop bottom 20%
            stamp = img.crop((0, int(h * 0.80), w, h))
            gray = ImageOps.grayscale(stamp)
            raw_text = pytesseract.image_to_string(gray).strip()
        except Exception as e:
            self.ocr_text_label.setText(f"OCR Error: {e}")
            return

        self.ocr_text_label.setText(raw_text if raw_text else "(no text detected)")

        # Parse the text
        self._parse_stamp_text(raw_text)

    def _parse_stamp_text(self, text: str):
        """Parse OCR text and detect patterns."""
        self.detected_fields.clear()
        self.fields_list.clear()
        self.unknown_combo.clear()
        self.unknown_combo.addItem("Select unknown text...")

        if not text:
            return

        # Split into tokens (words and symbols)
        # Keep some things together like "10:30AM" or "01/15/2025"
        tokens = re.findall(r'\d{1,2}[:/]\d{2}(?:[:/]\d{2,4})?(?:\s*[AP]M)?|\d{1,2}/\d{1,2}/\d{2,4}|\d+°[FC]|\S+', text, re.IGNORECASE)

        for token in tokens:
            token = token.strip()
            if not token or len(token) < 2:
                continue

            field = self._classify_token(token)
            self.detected_fields.append(field)

            # Add to list widget
            item = QListWidgetItem(f"{field.field_type.upper()}: {field.text}")
            if field.field_type == "unknown":
                item.setForeground(QColor(255, 200, 0))  # Yellow for unknown
                self.unknown_combo.addItem(field.text)
            elif field.field_type == "location":
                item.setForeground(QColor(0, 200, 0))  # Green for location
            else:
                item.setForeground(QColor(150, 150, 255))  # Blue for auto-detected

            self.fields_list.addItem(item)

    def _classify_token(self, token: str) -> StampField:
        """Classify a token into a field type."""
        token_upper = token.upper().strip()
        token_clean = token.strip()

        # Date pattern: MM/DD/YYYY or similar
        if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', token_clean):
            return StampField(token_clean, "date")

        # Time pattern: HH:MM or HH:MM:SS with optional AM/PM
        if re.match(r'^\d{1,2}:\d{2}(:\d{2})?\s*([AP]M)?$', token_clean, re.IGNORECASE):
            return StampField(token_clean, "time")

        # Just AM or PM
        if token_upper in ["AM", "PM"]:
            return StampField(token_clean, "time")

        # Temperature: number with degree symbol
        if re.match(r'^\d+°[FC]$', token_clean, re.IGNORECASE):
            return StampField(token_clean, "temperature")

        # Photo number: 3-digit number
        if re.match(r'^\d{3}$', token_clean):
            return StampField(token_clean, "photo_num")

        # Moon phase symbols
        if any(c in token for c in "🌑🌒🌓🌔🌕🌖🌗🌘"):
            return StampField(token_clean, "moon")

        # Known camera brands
        for brand in self.learned_patterns.get("camera_brands", []):
            if brand.upper() in token_upper or token_upper in brand.upper():
                return StampField(token_clean, "brand")

        # Known locations
        for loc in self.learned_patterns.get("locations", []):
            if loc.upper() == token_upper or token_upper in loc.upper() or loc.upper() in token_upper:
                return StampField(token_clean, "location")

        # Ignore words
        if token_upper in self.learned_patterns.get("ignore_words", []):
            return StampField(token_clean, "ignore")

        # Single letters or very short - probably noise
        if len(token_clean) <= 1:
            return StampField(token_clean, "ignore")

        # Unknown - might be location
        return StampField(token_clean, "unknown")

    def _mark_as_location(self):
        """Mark selected unknown text as a location."""
        text = self.unknown_combo.currentText()
        if text == "Select unknown text..." or not text:
            QMessageBox.warning(self, "Select Text", "Please select unknown text from the dropdown.")
            return

        # Add to known locations
        locations = self.learned_patterns.get("locations", [])
        if text not in locations:
            locations.append(text)
            self.learned_patterns["locations"] = locations
            self._save_patterns()

        self._refresh_locations_list()
        self._read_stamp()  # Re-parse with new knowledge

        QMessageBox.information(self, "Location Added", f"'{text}' is now recognized as a location.")

    def _mark_as_ignore(self):
        """Mark selected text to be ignored."""
        text = self.unknown_combo.currentText()
        if text == "Select unknown text..." or not text:
            return

        ignore = self.learned_patterns.get("ignore_words", [])
        if text.upper() not in [w.upper() for w in ignore]:
            ignore.append(text)
            self.learned_patterns["ignore_words"] = ignore
            self._save_patterns()

        self._read_stamp()

    def _refresh_locations_list(self):
        """Refresh the known locations list."""
        self.locations_list.clear()
        for loc in sorted(self.learned_patterns.get("locations", [])):
            self.locations_list.addItem(loc)

    def _remove_location(self):
        """Remove selected location from known list."""
        item = self.locations_list.currentItem()
        if not item:
            return

        loc = item.text()
        locations = self.learned_patterns.get("locations", [])
        if loc in locations:
            locations.remove(loc)
            self.learned_patterns["locations"] = locations
            self._save_patterns()

        self._refresh_locations_list()
        self._read_stamp()

    def _apply_to_database(self):
        """Apply detected location to current photo in database."""
        if not self.current_photo_path or not self.db:
            QMessageBox.warning(self, "Error", "No photo loaded or database not available.")
            return

        # Find detected location
        location = None
        for field in self.detected_fields:
            if field.field_type == "location":
                location = field.text
                break

        if not location:
            QMessageBox.warning(self, "No Location", "No location was detected in this stamp.")
            return

        # Update database
        cursor = self.db.conn.cursor()
        cursor.execute(
            "UPDATE photos SET camera_location = ? WHERE file_path = ?",
            (location, self.current_photo_path)
        )
        self.db.conn.commit()

        QMessageBox.information(
            self, "Updated",
            f"Set camera_location to '{location}' for this photo."
        )


def get_learned_locations() -> List[str]:
    """Get list of learned location names."""
    if PATTERNS_FILE.exists():
        try:
            with open(PATTERNS_FILE) as f:
                data = json.load(f)
                return data.get("locations", [])
        except:
            pass
    return []
