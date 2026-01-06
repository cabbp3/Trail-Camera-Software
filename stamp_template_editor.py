"""
Stamp Template Editor

Allows users to define regions on trail camera photo stamps
to teach the app where to find specific information (location, date, time, etc.)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QPainter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QFileDialog,
    QGroupBox, QFormLayout, QComboBox, QWidget, QSplitter, QScrollArea
)

# Template storage location
TEMPLATES_DIR = Path.home() / ".trailcam" / "stamp_templates"


class DraggableBox(QGraphicsRectItem):
    """A box that can be drawn and resized on the image."""

    COLORS = [
        QColor(255, 0, 0, 150),    # Red
        QColor(0, 255, 0, 150),    # Green
        QColor(0, 0, 255, 150),    # Blue
        QColor(255, 255, 0, 150),  # Yellow
        QColor(255, 0, 255, 150),  # Magenta
        QColor(0, 255, 255, 150),  # Cyan
        QColor(255, 128, 0, 150),  # Orange
        QColor(128, 0, 255, 150),  # Purple
        QColor(255, 128, 128, 150),# Pink
        QColor(128, 255, 128, 150),# Light Green
    ]

    def __init__(self, rect: QRectF, index: int, parent=None):
        super().__init__(rect, parent)
        self.index = index
        color = self.COLORS[index % len(self.COLORS)]
        self.setPen(QPen(color, 3))
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)


class StampTemplateEditor(QDialog):
    """Dialog for creating and editing stamp templates."""

    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Stamp Template Editor")
        self.resize(1200, 800)

        self.current_photo_path = None
        self.pixmap_item = None
        self.boxes: List[DraggableBox] = []
        self.box_notes: List[str] = [""] * 10
        self.drawing = False
        self.draw_start = None
        self.temp_rect = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # Left side - Image view
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Photo selection
        photo_row = QHBoxLayout()
        self.photo_label = QLabel("No photo loaded")
        self.photo_label.setStyleSheet("color: #888;")
        photo_row.addWidget(self.photo_label)
        photo_row.addStretch()

        # Camera sample dropdown
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(150)
        self.camera_combo.addItem("Select Camera...")
        self._populate_camera_samples()
        self.camera_combo.currentIndexChanged.connect(self._load_camera_sample)
        photo_row.addWidget(self.camera_combo)

        load_btn = QPushButton("Load Photo")
        load_btn.clicked.connect(self._load_photo)
        photo_row.addWidget(load_btn)

        load_current_btn = QPushButton("Use Current Photo")
        load_current_btn.clicked.connect(self._load_current_photo)
        photo_row.addWidget(load_current_btn)

        left_layout.addLayout(photo_row)

        # Instructions
        instructions = QLabel(
            "Click and drag to draw boxes around stamp regions (up to 5).\n"
            "Each box should contain one piece of information (location, date, time, etc.)"
        )
        instructions.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        instructions.setWordWrap(True)
        left_layout.addWidget(instructions)

        # Graphics view for image
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumSize(700, 500)
        self.view.setStyleSheet("background-color: #1a1a1a;")
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.viewport().installEventFilter(self)
        left_layout.addWidget(self.view)

        # Zoom controls
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom:"))

        zoom_fit_btn = QPushButton("Fit")
        zoom_fit_btn.clicked.connect(self._zoom_fit)
        zoom_row.addWidget(zoom_fit_btn)

        zoom_stamp_btn = QPushButton("Zoom to Stamp Area")
        zoom_stamp_btn.clicked.connect(self._zoom_stamp)
        zoom_row.addWidget(zoom_stamp_btn)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(40)
        zoom_in_btn.clicked.connect(lambda: self.view.scale(1.2, 1.2))
        zoom_row.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedWidth(40)
        zoom_out_btn.clicked.connect(lambda: self.view.scale(0.8, 0.8))
        zoom_row.addWidget(zoom_out_btn)

        zoom_row.addStretch()

        clear_boxes_btn = QPushButton("Clear All Boxes")
        clear_boxes_btn.clicked.connect(self._clear_boxes)
        zoom_row.addWidget(clear_boxes_btn)

        left_layout.addLayout(zoom_row)

        layout.addWidget(left_panel, stretch=2)

        # Right side - Box list and notes
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Template name
        name_group = QGroupBox("Template Info")
        name_layout = QFormLayout(name_group)

        self.template_name = QLineEdit()
        self.template_name.setPlaceholderText("e.g., Cuddeback J-Series")
        name_layout.addRow("Template Name:", self.template_name)

        self.camera_model = QLineEdit()
        self.camera_model.setPlaceholderText("e.g., Cuddeback")
        name_layout.addRow("Camera Model:", self.camera_model)

        right_layout.addWidget(name_group)

        # Box definitions - scrollable for 10 boxes
        boxes_group = QGroupBox("Box Definitions (up to 10)")
        boxes_outer_layout = QVBoxLayout(boxes_group)

        boxes_scroll = QScrollArea()
        boxes_scroll.setWidgetResizable(True)
        boxes_scroll.setMaximumHeight(350)
        boxes_container = QWidget()
        boxes_layout = QVBoxLayout(boxes_container)
        boxes_layout.setContentsMargins(0, 0, 0, 0)

        self.box_widgets = []
        for i in range(10):
            box_widget = self._create_box_widget(i)
            boxes_layout.addWidget(box_widget)
            self.box_widgets.append(box_widget)

        boxes_scroll.setWidget(boxes_container)
        boxes_outer_layout.addWidget(boxes_scroll)
        right_layout.addWidget(boxes_group)

        # Saved templates
        templates_group = QGroupBox("Saved Templates")
        templates_layout = QVBoxLayout(templates_group)

        self.templates_list = QListWidget()
        self.templates_list.itemDoubleClicked.connect(self._load_template)
        templates_layout.addWidget(self.templates_list)

        templates_btn_row = QHBoxLayout()
        load_template_btn = QPushButton("Load")
        load_template_btn.clicked.connect(self._load_selected_template)
        templates_btn_row.addWidget(load_template_btn)

        delete_template_btn = QPushButton("Delete")
        delete_template_btn.clicked.connect(self._delete_template)
        templates_btn_row.addWidget(delete_template_btn)

        templates_layout.addLayout(templates_btn_row)
        right_layout.addWidget(templates_group)

        # Action buttons
        btn_row = QHBoxLayout()

        save_btn = QPushButton("Save Template")
        save_btn.setStyleSheet("background-color: #2e7d32; color: white; padding: 10px;")
        save_btn.clicked.connect(self._save_template)
        btn_row.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        right_layout.addLayout(btn_row)

        layout.addWidget(right_panel, stretch=1)

        # Load existing templates
        self._refresh_templates_list()

    def _populate_camera_samples(self):
        """Populate dropdown with sample photos from each camera model."""
        self.camera_samples = {}

        if not self.db:
            return

        cursor = self.db.conn.cursor()
        # Get one sample photo per camera model (clean names only)
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
                # Clean up model name for display
                display_name = model
                self.camera_samples[display_name] = path
                self.camera_combo.addItem(f"{display_name} ({self._count_photos(model)} photos)")

    def _count_photos(self, model: str) -> int:
        """Count photos for a camera model."""
        if not self.db:
            return 0
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos WHERE camera_model = ?", (model,))
        return cursor.fetchone()[0]

    def _load_camera_sample(self, index: int):
        """Load a sample photo for the selected camera."""
        if index <= 0:  # Skip "Select Camera..." option
            return

        # Get the model name from the combo text
        text = self.camera_combo.currentText()
        model = text.split(" (")[0] if " (" in text else text

        if model in self.camera_samples:
            path = self.camera_samples[model]
            self._display_photo(path)
            # Pre-fill camera model field
            self.camera_model.setText(model)

    def _create_box_widget(self, index: int) -> QWidget:
        """Create a widget for defining one box."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 2, 0, 2)

        # Color indicator
        colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple", "pink", "lightgreen"]
        color_label = QLabel(f"Box {index + 1}")
        color_label.setStyleSheet(f"color: {colors[index]}; font-weight: bold; min-width: 50px;")
        layout.addWidget(color_label)

        # Field type dropdown
        field_type = QComboBox()
        field_type.addItems([
            "(not used)",
            "Location",
            "Date",
            "Time",
            "Temperature",
            "Moon Phase",
            "Camera Name",
            "Camera ID/Serial",
            "Photo Number",
            "Battery Level",
            "Brand Logo",
            "Other"
        ])
        field_type.setMinimumWidth(100)
        layout.addWidget(field_type)

        # Notes field
        notes = QLineEdit()
        notes.setPlaceholderText("Additional notes...")
        layout.addWidget(notes)

        # Delete button
        delete_btn = QPushButton("X")
        delete_btn.setFixedWidth(30)
        delete_btn.clicked.connect(lambda: self._delete_box(index))
        layout.addWidget(delete_btn)

        widget.field_type = field_type
        widget.notes = notes
        widget.color_label = color_label

        return widget

    def _load_photo(self):
        """Load a photo from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Photo", "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self._display_photo(file_path)

    def _load_current_photo(self):
        """Load the currently displayed photo from the main app."""
        if self.parent() and hasattr(self.parent(), 'photos') and hasattr(self.parent(), 'index'):
            photos = self.parent().photos
            index = self.parent().index
            if photos and 0 <= index < len(photos):
                path = photos[index].get("file_path")
                if path and os.path.exists(path):
                    self._display_photo(path)
                    return
        QMessageBox.warning(self, "No Photo", "No current photo available.")

    def _display_photo(self, path: str):
        """Display a photo in the editor."""
        self.current_photo_path = path
        self.photo_label.setText(os.path.basename(path))
        self.photo_label.setStyleSheet("color: white;")

        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", f"Could not load image: {path}")
            return

        # Clear scene and add image
        self.scene.clear()
        self.boxes.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(pixmap.rect().toRectF())

        # Zoom to stamp area by default
        self._zoom_stamp()

    def _zoom_fit(self):
        """Fit the entire image in view."""
        if self.pixmap_item:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _zoom_stamp(self):
        """Zoom to the bottom stamp area."""
        if self.pixmap_item:
            rect = self.scene.sceneRect()
            # Show bottom 25% of image
            stamp_rect = QRectF(0, rect.height() * 0.75, rect.width(), rect.height() * 0.25)
            self.view.fitInView(stamp_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def eventFilter(self, obj, event):
        """Handle mouse events for drawing boxes."""
        if obj == self.view.viewport() and self.pixmap_item:
            if event.type() == event.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton and len(self.boxes) < 10:
                    self.drawing = True
                    self.draw_start = self.view.mapToScene(event.pos())
                    return True

            elif event.type() == event.Type.MouseMove and self.drawing:
                current = self.view.mapToScene(event.pos())
                if self.temp_rect:
                    self.scene.removeItem(self.temp_rect)

                rect = QRectF(self.draw_start, current).normalized()
                self.temp_rect = QGraphicsRectItem(rect)
                color = DraggableBox.COLORS[len(self.boxes) % 10]
                self.temp_rect.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
                self.scene.addItem(self.temp_rect)
                return True

            elif event.type() == event.Type.MouseButtonRelease and self.drawing:
                self.drawing = False
                if self.temp_rect:
                    self.scene.removeItem(self.temp_rect)
                    self.temp_rect = None

                end = self.view.mapToScene(event.pos())
                rect = QRectF(self.draw_start, end).normalized()

                # Only create box if it has meaningful size
                if rect.width() > 10 and rect.height() > 10:
                    self._add_box(rect)
                return True

        return super().eventFilter(obj, event)

    def _add_box(self, rect: QRectF):
        """Add a new box to the scene."""
        if len(self.boxes) >= 10:
            QMessageBox.warning(self, "Limit Reached", "Maximum 10 boxes allowed.")
            return

        index = len(self.boxes)
        box = DraggableBox(rect, index)
        self.scene.addItem(box)
        self.boxes.append(box)

        # Update UI
        self._update_box_widgets()

    def _delete_box(self, index: int):
        """Delete a box by index."""
        if 0 <= index < len(self.boxes):
            box = self.boxes[index]
            self.scene.removeItem(box)
            self.boxes.pop(index)

            # Re-index remaining boxes
            for i, box in enumerate(self.boxes):
                box.index = i
                color = DraggableBox.COLORS[i % 10]
                box.setPen(QPen(color, 3))
                box.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))

            self._update_box_widgets()

    def _clear_boxes(self):
        """Clear all boxes."""
        for box in self.boxes:
            self.scene.removeItem(box)
        self.boxes.clear()
        self._update_box_widgets()

    def _update_box_widgets(self):
        """Update the box widget visibility based on drawn boxes."""
        colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple", "pink", "lightgreen"]
        for i, widget in enumerate(self.box_widgets):
            if i < len(self.boxes):
                widget.setEnabled(True)
                widget.color_label.setStyleSheet(
                    f"color: {colors[i]}; font-weight: bold;"
                )
            else:
                widget.setEnabled(False)
                widget.field_type.setCurrentIndex(0)
                widget.notes.clear()

    def _save_template(self):
        """Save the current template."""
        name = self.template_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a template name.")
            return

        if not self.boxes:
            QMessageBox.warning(self, "Error", "Please draw at least one box.")
            return

        if not self.pixmap_item:
            QMessageBox.warning(self, "Error", "Please load a photo first.")
            return

        # Get image dimensions for relative coordinates
        img_rect = self.scene.sceneRect()
        img_w, img_h = img_rect.width(), img_rect.height()

        # Build template data
        template = {
            "name": name,
            "camera_model": self.camera_model.text().strip(),
            "boxes": []
        }

        for i, box in enumerate(self.boxes):
            rect = box.rect()
            # Convert to relative coordinates (0-1)
            box_data = {
                "x1": rect.x() / img_w,
                "y1": rect.y() / img_h,
                "x2": (rect.x() + rect.width()) / img_w,
                "y2": (rect.y() + rect.height()) / img_h,
                "field_type": self.box_widgets[i].field_type.currentText(),
                "notes": self.box_widgets[i].notes.text()
            }
            template["boxes"].append(box_data)

        # Save to file
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
        template_path = TEMPLATES_DIR / f"{safe_name}.json"

        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)

        QMessageBox.information(self, "Saved", f"Template saved: {template_path.name}")
        self._refresh_templates_list()

    def _refresh_templates_list(self):
        """Refresh the list of saved templates."""
        self.templates_list.clear()

        if TEMPLATES_DIR.exists():
            for path in sorted(TEMPLATES_DIR.glob("*.json")):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    item = QListWidgetItem(data.get("name", path.stem))
                    item.setData(Qt.ItemDataRole.UserRole, str(path))
                    self.templates_list.addItem(item)
                except:
                    pass

    def _load_selected_template(self):
        """Load the selected template."""
        item = self.templates_list.currentItem()
        if item:
            self._load_template(item)

    def _load_template(self, item: QListWidgetItem):
        """Load a template from file."""
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not os.path.exists(path):
            return

        try:
            with open(path) as f:
                template = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load template: {e}")
            return

        # Set template info
        self.template_name.setText(template.get("name", ""))
        self.camera_model.setText(template.get("camera_model", ""))

        # Clear existing boxes
        self._clear_boxes()

        # If we have an image loaded, draw the boxes
        if self.pixmap_item:
            img_rect = self.scene.sceneRect()
            img_w, img_h = img_rect.width(), img_rect.height()

            for i, box_data in enumerate(template.get("boxes", [])[:10]):
                # Convert relative coords to absolute
                rect = QRectF(
                    box_data["x1"] * img_w,
                    box_data["y1"] * img_h,
                    (box_data["x2"] - box_data["x1"]) * img_w,
                    (box_data["y2"] - box_data["y1"]) * img_h
                )
                self._add_box(rect)

                # Set field type and notes
                if i < len(self.box_widgets):
                    field_type = box_data.get("field_type", "(not used)")
                    idx = self.box_widgets[i].field_type.findText(field_type)
                    if idx >= 0:
                        self.box_widgets[i].field_type.setCurrentIndex(idx)
                    self.box_widgets[i].notes.setText(box_data.get("notes", ""))
        else:
            QMessageBox.information(
                self, "Template Loaded",
                "Template loaded. Load a photo to see the boxes."
            )

    def _delete_template(self):
        """Delete the selected template."""
        item = self.templates_list.currentItem()
        if not item:
            return

        path = item.data(Qt.ItemDataRole.UserRole)
        name = item.text()

        reply = QMessageBox.question(
            self, "Delete Template",
            f"Delete template '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(path)
                self._refresh_templates_list()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not delete: {e}")


def get_templates() -> List[Dict]:
    """Get all saved stamp templates."""
    templates = []
    if TEMPLATES_DIR.exists():
        for path in TEMPLATES_DIR.glob("*.json"):
            try:
                with open(path) as f:
                    templates.append(json.load(f))
            except:
                pass
    return templates
