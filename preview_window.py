"""
Preview window for viewing and tagging individual photos — PyQt6 version (FINAL)
"""
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSplitter,
    QDialog,
    QGroupBox,
    QTextEdit,
    QMessageBox,
    QSlider,
    QLineEdit,
    QComboBox,
    QFormLayout,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QCheckBox,
    QGestureEvent,
    QPinchGesture,
    QTextBrowser,
)
from PyQt6.QtGui import QPixmap, QPainter, QShortcut, QKeySequence, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PIL import Image, ImageOps, ImageEnhance
from typing import List, Optional
from database import TrailCamDatabase


class ImageGraphicsView(QGraphicsView):
    zoom_changed = pyqtSignal(float)  # emits current scale factor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        self.scale_factor = 1.15
        self.current_scale = 1.0
        self.is_fit_to_window = True
        self.grabGesture(Qt.GestureType.PinchGesture)
        self.min_scale = 0.1
        self._ImageGraphicsView__prev_pos = None  # Initialize for mouse drag tracking

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.toggle_fit_100()
        super().mouseDoubleClickEvent(event)

    def zoom_in(self):
        if self.current_scale < 10.0:
            self.is_fit_to_window = False
            self.scale(self.scale_factor, self.scale_factor)
            self.current_scale *= self.scale_factor
            self._update_drag_mode()
            self.zoom_changed.emit(self.current_scale)

    def zoom_out(self):
        if self.current_scale > self.min_scale + 1e-3:
            self.is_fit_to_window = False
            next_scale = self.current_scale / self.scale_factor
            if next_scale < self.min_scale:
                factor = self.min_scale / self.current_scale
                self.scale(factor, factor)
                self.current_scale = self.min_scale
            else:
                self.scale(1 / self.scale_factor, 1 / self.scale_factor)
                self.current_scale = next_scale
            self._update_drag_mode()
            self.zoom_changed.emit(self.current_scale)

    def zoom_fit(self):
        if self.scene() and self.scene().items():
            self.resetTransform()
            self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.current_scale = self.transform().m11()
            self.min_scale = self.current_scale
            self.is_fit_to_window = True
            self._update_drag_mode()
            self.zoom_changed.emit(self.current_scale)

    def zoom_100(self):
        self.resetTransform()
        self.current_scale = 1.0
        self.is_fit_to_window = False
        self._update_drag_mode()
        if self.scene() and self.scene().items():
            self.centerOn(self.scene().itemsBoundingRect().center())
        self.zoom_changed.emit(self.current_scale)

    def toggle_fit_100(self):
        if self.is_fit_to_window:
            self.zoom_100()
        else:
            self.zoom_fit()

    def set_zoom_level(self, scale_factor: float):
        """Set an absolute zoom level (1.0 = 100%)."""
        scale_factor = max(self.min_scale, min(scale_factor, 10.0))
        self.is_fit_to_window = False
        self.resetTransform()
        self.scale(scale_factor, scale_factor)
        self.current_scale = scale_factor
        self._update_drag_mode()
        self.zoom_changed.emit(self.current_scale)

    def _update_drag_mode(self):
        if self.transform().m11() > 1.05:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    # CLICK-AND-DRAG PANNING
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.__prev_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            if self.__prev_pos is not None:
                delta = event.position() - self.__prev_pos
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self.__prev_pos = event.position()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            return self.gestureEvent(event)
        return super().event(event)

    def gestureEvent(self, event: QGestureEvent):
        pinch = event.gesture(Qt.GestureType.PinchGesture)
        if pinch:
            if pinch.state() == Qt.GestureState.GestureUpdated:
                if pinch.scaleFactor() > 1.0:
                    self.zoom_in()
                else:
                    self.zoom_out()
            event.accept()
            return True
        return False


class PreviewWindow(QDialog):
    photo_updated = pyqtSignal()
    photo_changed = pyqtSignal(int)

    def __init__(self, photo_id: int, db: TrailCamDatabase, photo_list: List = None,
                 current_index: int = 0, parent=None, auto_enhance_default: bool = True):
        super().__init__(parent)
        self.photo_id = photo_id
        self.db = db
        self.photo_list = photo_list or []
        self.current_index = current_index
        self.initial_deer_id = None
        self.last_autofill_ids = []
        self.last_autofill_deer_id = None
        self.auto_enhance_enabled = auto_enhance_default
        self.setWindowTitle("Photo Preview")
        self.resize(1400, 1000)

        layout = QHBoxLayout(self)
        self.splitter = QSplitter()
        layout.addWidget(self.splitter)

        # Image viewer container with top bar
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Image viewer
        self.view = ImageGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(6, 6, 6, 6)
        self.fullscreen_btn = QPushButton("Full Screen")
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        fit_btn_left = QPushButton("Fit")
        fit_btn_left.clicked.connect(self.view.zoom_fit)
        full_btn_left = QPushButton("100%")
        full_btn_left.clicked.connect(self.view.zoom_100)
        self.enhance_btn = QPushButton("Enhance")
        self.enhance_btn.setCheckable(True)
        self.enhance_btn.setChecked(True)
        self.enhance_btn.clicked.connect(self.toggle_enhance)
        self._update_enhance_button()
        top_bar.addWidget(self.fullscreen_btn)
        top_bar.addWidget(fit_btn_left)
        top_bar.addWidget(full_btn_left)
        top_bar.addWidget(self.enhance_btn)
        top_bar.addStretch()
        # Show Details button floated to the far right near the panel
        self.show_details_top_btn = QPushButton("Show Details")
        self.show_details_top_btn.clicked.connect(lambda: self.toggle_panel_btn.setChecked(False))
        top_bar.addWidget(self.show_details_top_btn)
        self.top_bar_widget = QWidget()
        self.top_bar_widget.setLayout(top_bar)
        left_layout.addWidget(self.top_bar_widget)

        left_layout.addWidget(self.view)
        self.splitter.addWidget(left_container)

        # Right panel with slide toggle
        self.right_panel = QWidget()
        self.right_panel.setMaximumWidth(560)
        self.right_panel.setMinimumWidth(300)
        right_layout = QVBoxLayout(self.right_panel)

        # Navigation + panel toggle
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn.clicked.connect(self.show_next)
        self.toggle_panel_btn = QPushButton("Hide Details")
        self.toggle_panel_btn.setCheckable(True)
        self.toggle_panel_btn.toggled.connect(self.toggle_details_panel)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.toggle_panel_btn)
        right_layout.addLayout(nav_layout)

        # Tags
        tag_group = QGroupBox("Quick Tags")
        tag_layout = QHBoxLayout()
        self.tag_dropdown = QComboBox()
        self.tag_dropdown.addItems(["Deer", "Turkey", "Coyote", "Person", "Vehicle", "Empty"])
        self.custom_tag_input = QLineEdit()
        self.custom_tag_input.setPlaceholderText("Custom tag...")
        tag_apply_btn = QPushButton("Toggle Tag")
        tag_apply_btn.clicked.connect(self.apply_dropdown_tag)
        tag_layout.addWidget(self.tag_dropdown)
        tag_layout.addWidget(self.custom_tag_input)
        tag_layout.addWidget(tag_apply_btn)
        tag_group.setLayout(tag_layout)
        right_layout.addWidget(tag_group)
        self.tag_buttons = {}

        # Deer metadata
        deer_group = QGroupBox("Deer Details")
        deer_form = QFormLayout()
        self.deer_id_input = QComboBox()
        self.deer_id_input.setEditable(True)
        self.deer_id_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.deer_id_input.setPlaceholderText("Deer ID")
        self._populate_deer_id_dropdown()
        self.profile_btn = QPushButton("Buck Profile")
        self.profile_btn.clicked.connect(self.open_buck_profile)
        self.age_combo = QComboBox()
        self.age_combo.addItems(["Unknown", "Fawn", "Yearling", "2.5", "3.5", "4.5+"])
        self.left_points_min = QSpinBox()
        self.right_points_min = QSpinBox()
        self.ab_points_min = QSpinBox()
        self.ab_points_max = QSpinBox()
        for spin in [self.left_points_min, self.right_points_min, self.ab_points_min, self.ab_points_max]:
            spin.setRange(0, 30)
            spin.setSingleStep(1)
            spin.setValue(0)
        self.left_uncertain = QCheckBox("Uncertain")
        self.right_uncertain = QCheckBox("Uncertain")
        self.ab_plus = QCheckBox("+")
        self.ab_uncertain = QCheckBox("Uncertain")
        self.season_combo = QComboBox()
        self.season_combo.addItem("Auto")
        # Add a reasonable range of seasons around current year
        current_year = datetime.now().year
        for yr in range(current_year + 1, current_year - 6, -1):
            self.season_combo.addItem(self.db.format_season_label(yr))
        deer_form.addRow("Deer ID:", self.deer_id_input)
        deer_form.addRow("", self.profile_btn)
        apply_profile_btn = QPushButton("Apply Profile to Season Photos")
        apply_profile_btn.clicked.connect(self.apply_profile_to_season_photos)
        deer_form.addRow("", apply_profile_btn)
        merge_btn = QPushButton("Merge Buck IDs")
        merge_btn.clicked.connect(self.merge_buck_ids_dialog)
        deer_form.addRow("", merge_btn)
        deer_form.addRow("Age Class:", self.age_combo)
        deer_form.addRow("Left Typical Points:", self._make_typical_row(self.left_points_min, self.left_uncertain))
        deer_form.addRow("Right Typical Points:", self._make_typical_row(self.right_points_min, self.right_uncertain))
        deer_form.addRow("Abnormal Points:", self._make_range_row(self.ab_points_min, self.ab_points_max, self.ab_plus, self.ab_uncertain))
        deer_form.addRow("Antler Season (May-Apr):", self.season_combo)
        deer_group.setLayout(deer_form)
        right_layout.addWidget(deer_group)
        self.deer_id_input.currentTextChanged.connect(self.update_additional_enabled)
        self.ab_uncertain.toggled.connect(lambda checked: self._apply_uncertain_state(self.ab_uncertain, self.ab_plus, self.ab_points_max, checked))
        self.ab_plus.toggled.connect(lambda checked: self._apply_plus_state(self.ab_plus, self.ab_points_max, checked))

        # Additional deer list
        additional_group = QGroupBox("Additional Deer")
        self.additional_group = additional_group
        additional_layout = QVBoxLayout()
        add_row1 = QHBoxLayout()
        self.additional_deer_input = QLineEdit()
        self.additional_deer_input.setPlaceholderText("Deer ID...")
        self.additional_age_combo = QComboBox()
        self.additional_age_combo.addItems(["Unknown", "Fawn", "Yearling", "2.5", "3.5", "4.5+"])
        add_row1.addWidget(self.additional_deer_input)
        add_row1.addWidget(self.additional_age_combo)

        add_row2 = QHBoxLayout()
        self.additional_left_min = QSpinBox()
        self.additional_right_min = QSpinBox()
        self.additional_ab_min = QSpinBox()
        self.additional_ab_max = QSpinBox()
        self.additional_ab_plus = QCheckBox("+")
        self.additional_left_uncertain = QCheckBox("Uncertain")
        self.additional_right_uncertain = QCheckBox("Uncertain")
        self.additional_ab_uncertain = QCheckBox("Uncertain")
        for spin in [self.additional_left_min, self.additional_right_min, self.additional_ab_min, self.additional_ab_max]:
            spin.setRange(0, 30)
            spin.setSingleStep(1)
            spin.setValue(0)
        add_row2.addWidget(QLabel("L min"))
        add_row2.addWidget(self.additional_left_min)
        add_row2.addWidget(self.additional_left_uncertain)
        add_row2.addWidget(QLabel("R min"))
        add_row2.addWidget(self.additional_right_min)
        add_row2.addWidget(self.additional_right_uncertain)
        add_row2.addWidget(QLabel("Ab min"))
        add_row2.addWidget(self.additional_ab_min)
        add_row2.addWidget(QLabel("Ab max"))
        add_row2.addWidget(self.additional_ab_max)
        add_row2.addWidget(self.additional_ab_plus)
        add_row2.addWidget(self.additional_ab_uncertain)

        add_btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_additional_deer)
        add_btn_row.addStretch()
        add_btn_row.addWidget(add_btn)
        # Wire uncertain toggles for additional deer
        self.additional_left_uncertain.toggled.connect(lambda checked: None)
        self.additional_right_uncertain.toggled.connect(lambda checked: None)
        self.additional_ab_uncertain.toggled.connect(lambda checked: self._apply_uncertain_state(self.additional_ab_uncertain, self.additional_ab_plus, self.additional_ab_max, checked))
        self.additional_ab_plus.toggled.connect(lambda checked: self._apply_plus_state(self.additional_ab_plus, self.additional_ab_max, checked))

        self.additional_list = QListWidget()
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected_deer)
        additional_layout.addLayout(add_row1)
        additional_layout.addLayout(add_row2)
        additional_layout.addLayout(add_btn_row)
        additional_layout.addWidget(self.additional_list)
        additional_layout.addWidget(remove_btn)
        additional_group.setLayout(additional_layout)
        right_layout.addWidget(additional_group)
        self.additional_group.setEnabled(False)

        # Notes
        notes_label = QLabel("Notes:")
        right_layout.addWidget(notes_label)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(120)
        right_layout.addWidget(self.notes_edit)
        save_notes_btn = QPushButton("Save Details")
        save_notes_btn.clicked.connect(self.save_details)
        right_layout.addWidget(save_notes_btn)

        # Camera location and key characteristics
        meta_group = QGroupBox("Photo Attributes")
        meta_layout = QFormLayout()
        self.camera_location_input = QLineEdit()
        self.camera_location_input.setPlaceholderText("Camera location / name")
        self.key_char_dropdown = QComboBox()
        self.key_char_dropdown.setEditable(True)
        self.key_char_dropdown.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.key_char_dropdown.setPlaceholderText("Select or type characteristic")
        self._populate_key_char_dropdown()
        self.key_characteristics_input = QTextEdit()
        self.key_characteristics_input.setPlaceholderText("Key characteristics (e.g., sticker on G2, scar on shoulder)")
        self.key_characteristics_input.setMaximumHeight(80)
        meta_layout.addRow("Camera Location:", self.camera_location_input)
        key_char_row = QHBoxLayout()
        add_key_btn = QPushButton("Add")
        add_key_btn.clicked.connect(self.add_key_characteristic_from_dropdown)
        key_char_row.addWidget(self.key_char_dropdown)
        key_char_row.addWidget(add_key_btn)
        key_char_container = QWidget()
        key_char_container.setLayout(key_char_row)
        meta_layout.addRow("Key Characteristics:", key_char_container)
        meta_layout.addRow("", self.key_characteristics_input)
        self.broken_side_combo = QComboBox()
        self.broken_side_combo.addItems(["None", "Left", "Right", "Both", "Unknown"])
        self.broken_note = QTextEdit()
        self.broken_note.setMaximumHeight(60)
        self.broken_note.setPlaceholderText("Notes on broken antler (date observed, details)")
        meta_layout.addRow("Broken Antler:", self.broken_side_combo)
        meta_layout.addRow("Broken Notes:", self.broken_note)
        # Suggested tag display
        suggested_row = QHBoxLayout()
        self.suggested_label = QLabel("Suggested: none")
        self.apply_suggested_btn = QPushButton("Apply")
        self.apply_suggested_btn.clicked.connect(self.apply_suggested_tag)
        suggested_row.addWidget(self.suggested_label)
        suggested_row.addStretch()
        suggested_row.addWidget(self.apply_suggested_btn)
        meta_layout.addRow(suggested_row)
        meta_group.setLayout(meta_layout)
        right_layout.addWidget(meta_group)
        # Undo autofill button
        self.undo_autofill_btn = QPushButton("Undo Last Autofill")
        self.undo_autofill_btn.setEnabled(False)
        self.undo_autofill_btn.clicked.connect(self.undo_autofill)
        right_layout.addWidget(self.undo_autofill_btn)

        # Favorite / Delete
        action_layout = QHBoxLayout()
        self.fav_btn = QPushButton("Favorite")
        self.fav_btn.setCheckable(True)
        self.fav_btn.clicked.connect(self.save_details)
        delete_btn = QPushButton("Delete Photo")
        delete_btn.clicked.connect(self.delete_current_photo)
        action_layout.addWidget(self.fav_btn)
        action_layout.addWidget(delete_btn)
        right_layout.addLayout(action_layout)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.view.zoom_fit)
        full_btn = QPushButton("100%")
        full_btn.clicked.connect(self.view.zoom_100)
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.view.zoom_in)
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.view.zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)   # 10%
        self.zoom_slider.setMaximum(500)  # 500%
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        zoom_layout.addWidget(fit_btn)
        zoom_layout.addWidget(full_btn)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        right_layout.addLayout(zoom_layout)
        self.view.zoom_changed.connect(self.update_zoom_ui)

        right_layout.addStretch()
        self.splitter.addWidget(self.right_panel)
        # right_panel already assigned above
        self.splitter.setStretchFactor(0, 12)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([1800, 320])

        self.load_photo()
        self.update_navigation_state()
        self._prev_splitter_sizes = self.splitter.sizes()
        # Start with details hidden for larger photo view
        self.toggle_panel_btn.setChecked(True)
        self.show_details_top_btn.show()

    def load_photo(self):
        path = self.db.get_photo_path(self.photo_id)
        if path and os.path.exists(path):
            pixmap = self._load_pixmap_with_enhance(path)
            self.pixmap_item.setPixmap(pixmap)
            self.view.zoom_fit()
        self.refresh_tag_state()
        self.refresh_notes_and_favorite()
        self.refresh_deer_metadata()
        self.refresh_additional_deer()
        self.refresh_photo_attributes()
        self.undo_autofill_btn.setEnabled(False)
        self.last_autofill_ids = []
        self.last_autofill_deer_id = None
        self.update_navigation_state()
        self.update_zoom_ui(self.view.current_scale)
        self.update_additional_enabled()
        # Fullscreen shortcut
        self.fs_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fs_shortcut.activated.connect(self.toggle_fullscreen)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.view.zoom_fit()

    def show_previous(self):
        if self.photo_list and self.current_index > 0:
            self.current_index -= 1
            self.photo_id = self.photo_list[self.current_index]
            self.load_photo()
            self.photo_changed.emit(self.photo_id)

    def show_next(self):
        if self.photo_list and self.current_index < len(self.photo_list) - 1:
            self.current_index += 1
            self.photo_id = self.photo_list[self.current_index]
            self.load_photo()
            self.photo_changed.emit(self.photo_id)

    def toggle_tag(self, tag_name: str):
        current_tags = self.db.get_photo_tags(self.photo_id)
        if tag_name in current_tags:
            current_tags.remove(tag_name)
        else:
            current_tags.append(tag_name)
        self.db.update_photo_tags(self.photo_id, current_tags)
        self.refresh_tag_state()
        self.photo_updated.emit()

    def delete_current_photo(self):
        reply = QMessageBox.question(
            self,
            "Delete Photo",
            "Permanently delete this photo from disk and database?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            path = self.db.get_photo_path(self.photo_id)
            if path and os.path.exists(path):
                os.remove(path)
            self.db.delete_photo(self.photo_id)
            self.photo_updated.emit()
            self.close()

    def refresh_tag_state(self):
        """Sync quick tag dropdown (no-op for now)."""
        # Tags remain stored in DB; dropdown is just a quick toggle.
        pass

    def refresh_notes_and_favorite(self):
        """Sync notes text and favorite state."""
        self.fav_btn.blockSignals(True)
        self.fav_btn.setChecked(self.db.is_favorite(self.photo_id))
        self.fav_btn.blockSignals(False)
        self.notes_edit.blockSignals(True)
        self.notes_edit.setText(self.db.get_notes(self.photo_id) or "")
        self.notes_edit.blockSignals(False)
    
    def refresh_photo_attributes(self):
        """Sync camera location and key characteristics."""
        photo = self.db.get_photo_by_id(self.photo_id) or {}
        self.camera_location_input.blockSignals(True)
        self.camera_location_input.setText(photo.get("camera_location") or "")
        self.camera_location_input.blockSignals(False)
        self.key_characteristics_input.blockSignals(True)
        self.key_characteristics_input.setPlainText(photo.get("key_characteristics") or "")
        self.key_characteristics_input.blockSignals(False)
        # Broken antler
        meta = self.db.get_deer_metadata(self.photo_id)
        side = meta.get("broken_antler_side") or "None"
        if self.broken_side_combo.findText(side) == -1:
            side = "None"
        self.broken_side_combo.blockSignals(True)
        self.broken_side_combo.setCurrentText(side)
        self.broken_side_combo.blockSignals(False)
        self.broken_note.blockSignals(True)
        self.broken_note.setPlainText(meta.get("broken_antler_note") or "")
        self.broken_note.blockSignals(False)
        # Suggested tag
        suggested = photo.get("suggested_tag")
        conf = photo.get("suggested_confidence")
        if suggested:
            conf_text = f" ({conf:.2f})" if conf is not None else ""
            self.suggested_label.setText(f"Suggested: {suggested}{conf_text}")
            self.apply_suggested_btn.setEnabled(True)
        else:
            self.suggested_label.setText("Suggested: none")
            self.apply_suggested_btn.setEnabled(False)

    def refresh_deer_metadata(self):
        """Sync deer metadata fields."""
        meta = self.db.get_deer_metadata(self.photo_id)
        self._populate_deer_id_dropdown()
        self.deer_id_input.blockSignals(True)
        self.deer_id_input.setCurrentText(meta.get("deer_id") or "")
        self.deer_id_input.blockSignals(False)
        self.initial_deer_id = meta.get("deer_id")
        self.age_combo.blockSignals(True)
        age_value = meta.get("age_class") or "Unknown"
        if self.age_combo.findText(age_value) == -1:
            age_value = "Unknown"
        self.age_combo.setCurrentText(age_value)
        self.age_combo.blockSignals(False)
        # Points
        self._set_spin_value(self.left_points_min, meta.get("left_points_min"))
        self._set_spin_value(self.right_points_min, meta.get("right_points_min"))
        self.left_uncertain.blockSignals(True)
        self.right_uncertain.blockSignals(True)
        self.left_uncertain.setChecked(meta.get("left_points_uncertain", False))
        self.right_uncertain.setChecked(meta.get("right_points_uncertain", False))
        self.left_uncertain.blockSignals(False)
        self.right_uncertain.blockSignals(False)
        self._set_spin_value(self.ab_points_min, meta.get("abnormal_points_min"))
        self._set_spin_value(self.ab_points_max, meta.get("abnormal_points_max"))
        self._sync_plus_from_values(self.ab_plus, self.ab_points_max, meta.get("abnormal_points_min"), meta.get("abnormal_points_max"))
        self.season_combo.blockSignals(True)
        season_year = self.db.get_season_year(self.photo_id)
        if season_year is None:
            self.season_combo.setCurrentText("Auto")
        else:
            label = self.db.format_season_label(season_year)
            if self.season_combo.findText(label) == -1:
                self.season_combo.addItem(label)
            self.season_combo.setCurrentText(label)
        self.season_combo.blockSignals(False)
        self.update_additional_enabled()
        # Set uncertain toggles for abnormal
        self._sync_plus_from_values(self.ab_plus, self.ab_points_max, meta.get("abnormal_points_min"), meta.get("abnormal_points_max"))
        self._sync_uncertain_from_values(self.ab_uncertain, self.ab_points_max, meta.get("abnormal_points_min"), meta.get("abnormal_points_max"))
        # Autofill from buck profile if available
        if meta.get("deer_id"):
            self._apply_buck_profile(meta.get("deer_id"), season_year)

    def refresh_additional_deer(self):
        """Sync additional deer list UI from DB."""
        self.additional_list.blockSignals(True)
        self.additional_list.clear()
        for entry in self.db.get_additional_deer(self.photo_id):
            deer_id = entry.get("deer_id") or ""
            if deer_id:
                display = self._format_additional_entry(entry)
                item = QListWidgetItem(display)
                item.setData(Qt.ItemDataRole.UserRole, entry)
                self.additional_list.addItem(item)
        self.additional_list.blockSignals(False)

    def update_navigation_state(self):
        """Enable/disable navigation buttons based on position."""
        has_list = bool(self.photo_list)
        if hasattr(self, "prev_btn"):
            self.prev_btn.setEnabled(has_list and self.current_index > 0)
        if hasattr(self, "next_btn"):
            self.next_btn.setEnabled(has_list and self.current_index < len(self.photo_list) - 1)

    def save_details(self):
        """Persist notes, favorite, and deer metadata."""
        self.db.set_notes(self.photo_id, self.notes_edit.toPlainText())
        self.db.set_favorite(self.photo_id, self.fav_btn.isChecked())
        deer_id = self.deer_id_input.currentText().strip() or None
        age_class = self.age_combo.currentText()
        if age_class == "Unknown":
            age_class = None
        # Normalize abnormal range so a single value is easy (copy to both min/max)
        self._normalize_range_spins(self.ab_points_min, self.ab_points_max)
        lp_min, lp_unc = self._get_typical_values(self.left_points_min, self.left_uncertain)
        rp_min, rp_unc = self._get_typical_values(self.right_points_min, self.right_uncertain)
        ab_min, ab_max = self._get_range_values(self.ab_points_min, self.ab_points_max, self.ab_plus, self.ab_uncertain)
        season_text = self.season_combo.currentText()
        if season_text == "Auto":
            season_year = self.db.compute_season_year(self.db.get_photo_by_id(self.photo_id).get("date_taken"))
        else:
            try:
                season_year = int(season_text.split("-")[0])
            except ValueError:
                season_year = None
        self.db.set_season_year(self.photo_id, season_year)
        self.db.set_deer_metadata(
            self.photo_id,
            deer_id=deer_id,
            age_class=age_class,
            left_points_min=lp_min,
            right_points_min=rp_min,
            left_points_uncertain=lp_unc,
            right_points_uncertain=rp_unc,
            abnormal_points_min=ab_min,
            abnormal_points_max=ab_max,
            broken_antler_side=self.broken_side_combo.currentText() if self.broken_side_combo.currentText() != "None" else None,
            broken_antler_note=self.broken_note.toPlainText().strip(),
        )
        # Update buck profile for this season
        photo_row = self.db.get_photo_by_id(self.photo_id) or {}
        deer_meta = self.db.get_deer_metadata(self.photo_id)
        self.db.update_buck_profile_from_photo(deer_id, season_year, photo_row, deer_meta)
        # Save camera location and key characteristics
        self.db.update_photo_attributes(
            self.photo_id,
            camera_location=self.camera_location_input.text().strip(),
            key_characteristics=self._normalize_tags_text(self.key_characteristics_input.toPlainText()),
        )
        # Save additional deer list
        extra_entries = []
        for i in range(self.additional_list.count()):
            item = self.additional_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole) or {}
            deer_val = data.get("deer_id") or item.text().strip()
            if deer_val:
                extra_entries.append(data)
        if not deer_id:
            extra_entries = []
        self.db.set_additional_deer(self.photo_id, extra_entries)
        # Propagate deer_id to other photos if newly assigned
        if deer_id and deer_id != self.initial_deer_id:
            reply = QMessageBox.question(
                self,
                "Propagate Deer ID",
                "Apply this deer ID to other photos that currently have no deer ID? "
                "Only the deer ID will be copied; age/antler details remain per photo.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                updated_ids = self.db.propagate_deer_id_to_empty(deer_id, exclude_photo_id=self.photo_id)
                self.last_autofill_ids = updated_ids
                self.last_autofill_deer_id = deer_id
                if updated_ids:
                    self.undo_autofill_btn.setEnabled(True)
                    QMessageBox.information(
                        self,
                        "Autofill applied",
                        f"Applied Deer ID to {len(updated_ids)} photo(s). You can undo using 'Undo Last Autofill'."
                    )
            self.initial_deer_id = deer_id
        self.photo_updated.emit()

    def on_zoom_slider_changed(self, value: int):
        """Handle zoom slider change; value is percent."""
        self.view.set_zoom_level(value / 100.0)

    def update_zoom_ui(self, scale: float):
        """Sync zoom slider/label when zoom changes."""
        percent = int(scale * 100)
        self.zoom_label.setText(f"{percent}%")
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), percent)))
        self.zoom_slider.blockSignals(False)

    def add_additional_deer(self):
        """Add a new additional deer entry from input."""
        deer_id = self.additional_deer_input.text().strip()
        if not deer_id:
            return
        age = self.additional_age_combo.currentText()
        if age == "Unknown":
            age = None
        # Normalize abnormal range so a single value is easy
        self._normalize_range_spins(self.additional_ab_min, self.additional_ab_max)
        lp_min, lp_unc = self._get_typical_values(self.additional_left_min, self.additional_left_uncertain)
        rp_min, rp_unc = self._get_typical_values(self.additional_right_min, self.additional_right_uncertain)
        ab_min, ab_max = self._get_range_values(self.additional_ab_min, self.additional_ab_max, self.additional_ab_plus, self.additional_ab_uncertain)
        entry = {
            "deer_id": deer_id,
            "age_class": age,
            "left_points_min": lp_min,
            "right_points_min": rp_min,
            "left_points_uncertain": lp_unc,
            "right_points_uncertain": rp_unc,
            "abnormal_points_min": ab_min,
            "abnormal_points_max": ab_max,
        }
        # Avoid duplicates in UI
        for i in range(self.additional_list.count()):
            existing = self.additional_list.item(i).data(Qt.ItemDataRole.UserRole) or {}
            if existing.get("deer_id") == deer_id:
                self.additional_deer_input.clear()
                return
        item = QListWidgetItem(self._format_additional_entry(entry))
        item.setData(Qt.ItemDataRole.UserRole, entry)
        self.additional_list.addItem(item)
        self.additional_deer_input.clear()
        self.additional_age_combo.setCurrentIndex(0)
        for spin in [self.additional_left_min, self.additional_right_min, self.additional_ab_min, self.additional_ab_max]:
            spin.setValue(0)
        self.save_details()

    def remove_selected_deer(self):
        """Remove selected additional deer entries."""
        selected = self.additional_list.selectedItems()
        if not selected:
            return
        for item in selected:
            row = self.additional_list.row(item)
            self.additional_list.takeItem(row)
        self.save_details()

    def update_additional_enabled(self):
        """Enable additional-deer group only if primary deer is set."""
        has_primary = bool(self.deer_id_input.currentText().strip())
        self.additional_group.setEnabled(has_primary)
    
    def undo_autofill(self):
        """Undo last deer_id autofill propagation."""
        if not self.last_autofill_ids:
            return
        self.db.clear_deer_id(self.last_autofill_ids)
        self.last_autofill_ids = []
        self.last_autofill_deer_id = None
        self.undo_autofill_btn.setEnabled(False)
        QMessageBox.information(self, "Undo", "Autofill of deer ID has been undone for the affected photos.")

    def apply_dropdown_tag(self):
        """Toggle the selected dropdown tag on/off for this photo."""
        tag = self.custom_tag_input.text().strip() or self.tag_dropdown.currentText()
        self.toggle_tag(tag)

    def apply_suggested_tag(self):
        """Apply suggested tag if available."""
        text = self.suggested_label.text().replace("Suggested:", "").strip()
        if text and text.lower() != "none":
            tag = text.split(" ")[0].strip()  # take the label part
            self.toggle_tag(tag)

    def toggle_enhance(self, checked: bool):
        """Toggle auto-enhance and reload current photo."""
        self.auto_enhance_enabled = checked
        self._update_enhance_button()
        self.load_photo()

    def _update_enhance_button(self):
        """Update enhance button label to reflect state."""
        if self.auto_enhance_enabled:
            self.enhance_btn.setText("Enhance: ON")
        else:
            self.enhance_btn.setText("Enhance: OFF")

    def _none_if_zero(self, value: int) -> Optional[int]:
        return None if value == 0 else value

    def _set_spin_value(self, spin: QSpinBox, val: Optional[int]):
        spin.blockSignals(True)
        spin.setValue(val if val is not None else 0)
        spin.blockSignals(False)

    @staticmethod
    def _normalize_tags_text(text: str) -> str:
        """Normalize comma/line separated tags to a clean comma-separated string."""
        parts = []
        for part in text.replace("\n", ",").split(","):
            p = part.strip()
            if p:
                parts.append(p)
        # dedupe preserve order
        seen = set()
        result = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return ", ".join(result)

    def _populate_key_char_dropdown(self):
        """Fill key characteristic dropdown with existing tokens from DB."""
        options = []
        try:
            for photo in self.db.get_all_photos():
                kc = photo.get("key_characteristics") or ""
                for tok in self._normalize_tags_text(kc).split(","):
                    t = tok.strip()
                    if t:
                        options.append(t)
        except Exception:
            options = []
        unique = []
        seen = set()
        for t in options:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        self.key_char_dropdown.clear()
        self.key_char_dropdown.addItem("")  # empty placeholder
        self.key_char_dropdown.addItems(unique)

    def add_key_characteristic_from_dropdown(self):
        """Append selected/typed characteristic into the field."""
        text = self.key_char_dropdown.currentText().strip()
        if not text:
            return
        existing = self._normalize_tags_text(self.key_characteristics_input.toPlainText())
        parts = [p.strip() for p in existing.split(",") if p.strip()]
        if text not in parts:
            parts.append(text)
        self.key_characteristics_input.setPlainText(", ".join(parts))

    def _populate_deer_id_dropdown(self):
        """Fill deer ID dropdown from existing IDs in DB."""
        try:
            ids = set()
            for photo in self.db.get_all_photos():
                meta = self.db.get_deer_metadata(photo["id"])
                did = meta.get("deer_id") or ""
                if did:
                    ids.add(did)
                for add in self.db.get_additional_deer(photo["id"]):
                    if add.get("deer_id"):
                        ids.add(add["deer_id"])
            sorted_ids = sorted(ids)
        except Exception:
            sorted_ids = []
        current = self.deer_id_input.currentText() if isinstance(self.deer_id_input, QComboBox) else ""
        self.deer_id_input.blockSignals(True)
        self.deer_id_input.clear()
        self.deer_id_input.addItem("")  # placeholder
        for did in sorted_ids:
            self.deer_id_input.addItem(did)
        if current:
            self.deer_id_input.setCurrentText(current)
        self.deer_id_input.blockSignals(False)

    def _all_deer_ids(self):
        ids = set()
        for photo in self.db.get_all_photos():
            meta = self.db.get_deer_metadata(photo["id"])
            if meta.get("deer_id"):
                ids.add(meta["deer_id"])
            for add in self.db.get_additional_deer(photo["id"]):
                if add.get("deer_id"):
                    ids.add(add["deer_id"])
        return ids

    def _note_original_name(self, original_id: str, new_id: str):
        """Append a locked note indicating original buck name on affected photos."""
        if not original_id or not new_id:
            return
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT p.id, p.notes FROM photos p
            JOIN deer_metadata dm ON dm.photo_id = p.id
            WHERE dm.deer_id = ?
        """, (new_id,))
        rows = cursor.fetchall()
        note_snip = f"[Merged: originally {original_id}]"
        for row in rows:
            notes = row["notes"] or ""
            if note_snip not in notes:
                updated = (notes + " " + note_snip).strip()
                self.db.set_notes(row["id"], updated)

    def _apply_buck_profile(self, deer_id: str, season_year: Optional[int]):
        """Autofill fields from buck profile for the given season."""
        profile = self.db.get_buck_profile(deer_id, season_year)
        if not profile:
            return
        # Typical points
        if profile.get("left_points_min") is not None:
            self._set_spin_value(self.left_points_min, profile.get("left_points_min"))
            self.left_uncertain.setChecked(bool(profile.get("left_points_uncertain")))
        if profile.get("right_points_min") is not None:
            self._set_spin_value(self.right_points_min, profile.get("right_points_min"))
            self.right_uncertain.setChecked(bool(profile.get("right_points_uncertain")))
        # Abnormal min/max
        self._set_spin_value(self.ab_points_min, profile.get("abnormal_points_min"))
        self._set_spin_value(self.ab_points_max, profile.get("abnormal_points_max"))
        # Key characteristics
        kc = profile.get("key_characteristics") or ""
        if kc:
            self.key_characteristics_input.setPlainText(self._normalize_tags_text(kc))
        # Camera locations merged
        cam = profile.get("camera_locations") or ""
        if cam and not self.camera_location_input.text().strip():
            self.camera_location_input.setText(cam.split(",")[0].strip())

    def open_buck_profile(self):
        """Show a simple profile dialog with seasons and changes."""
        deer_id = self.deer_id_input.currentText().strip()
        if not deer_id:
            QMessageBox.information(self, "Buck Profile", "Set a Deer ID first.")
            return
        summaries = self.db.get_buck_season_summaries(deer_id)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Buck Profile: {deer_id}")
        layout = QVBoxLayout(dlg)
        info = QTextBrowser()
        lines = []
        prev = None
        for s in summaries:
            label = self.db.format_season_label(s["season_year"]) if s.get("season_year") else "Unknown season"
            lines.append(f"<b>{label}</b> ({s.get('photo_count',0)} photos)")
            lines.append(f"  Typical L/R: {s.get('left_points_min') or '?'} / {s.get('right_points_min') or '?'}")
            lines.append(f"  Abnormal L/R: {s.get('left_ab_points_min') or '?'} / {s.get('right_ab_points_min') or '?'}")
            lines.append(f"  Abnormal total: {s.get('abnormal_points_min') or '?'}–{s.get('abnormal_points_max') or '?'}")
            lines.append(f"  Cameras: {s.get('camera_locations') or ''}")
            lines.append(f"  Key characteristics: {self._normalize_tags_text(s.get('key_characteristics') or '')}")
            if prev:
                diffs = []
                for key in ["left_points_min", "right_points_min", "left_ab_points_min", "right_ab_points_min", "abnormal_points_min", "abnormal_points_max"]:
                    if s.get(key) != prev.get(key) and s.get(key) is not None:
                        diffs.append(f"{key} changed from {prev.get(key)} to {s.get(key)}")
                if diffs:
                    lines.append("  Changes vs previous season:")
                    for d in diffs:
                        lines.append(f"    - {d}")
            lines.append("")
            prev = s
        info.setHtml("<br>".join(lines) if lines else "No data yet.")
        layout.addWidget(info)
        btns = QHBoxLayout()
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        btns.addStretch()
        btns.addWidget(close)
        layout.addLayout(btns)
        dlg.resize(500, 400)
        dlg.exec()

    def apply_profile_to_season_photos(self):
        """Apply current buck season profile to all photos for that buck/season."""
        deer_id = self.deer_id_input.currentText().strip()
        if not deer_id:
            QMessageBox.information(self, "Apply Profile", "Set a Deer ID first.")
            return
        season_year = self.db.get_season_year(self.photo_id)
        if season_year is None:
            QMessageBox.information(self, "Apply Profile", "Season is unknown for this photo.")
            return
        confirm = QMessageBox.question(
            self,
            "Apply Profile",
            f"Apply {deer_id}'s profile to all photos in {self.db.format_season_label(season_year)}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        count = self.db.apply_profile_to_season_photos(deer_id, season_year)
        QMessageBox.information(self, "Apply Profile", f"Applied profile to {count} photo(s) for {self.db.format_season_label(season_year)}.")

    def merge_buck_ids_dialog(self):
        """Merge one buck ID into a new/target ID and note originals."""
        deer_ids = sorted({d for d in self._all_deer_ids() if d})
        if len(deer_ids) < 1:
            QMessageBox.information(self, "Merge Buck IDs", "No buck IDs found to merge.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Merge Buck IDs")
        layout = QFormLayout(dlg)
        src_combo = QComboBox()
        src_combo.addItems(deer_ids)
        tgt_combo = QComboBox()
        tgt_combo.addItems(deer_ids)
        new_name = QLineEdit()
        new_name.setPlaceholderText("New buck ID (required)")
        layout.addRow("Source (will be merged):", src_combo)
        layout.addRow("Target (keep data):", tgt_combo)
        layout.addRow("New Buck ID:", new_name)
        buttons = QHBoxLayout()
        ok_btn = QPushButton("Merge")
        cancel_btn = QPushButton("Cancel")
        buttons.addStretch()
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addRow(buttons)

        def do_merge():
            src = src_combo.currentText().strip()
            tgt = tgt_combo.currentText().strip()
            new_id = new_name.text().strip()
            if not src or not tgt or not new_id:
                QMessageBox.warning(dlg, "Merge", "Source, target, and new buck ID are required.")
                return
            confirm = QMessageBox.question(
                dlg,
                "Confirm Merge",
                f"Merge '{src}' into '{tgt}' and rename to '{new_id}'?\nA note will be added to photos from the source ID.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            # First merge src into tgt
            affected = self.db.merge_deer_ids(src, tgt)
            # Then rename tgt to new_id if different
            if new_id != tgt:
                self.db.merge_deer_ids(tgt, new_id)
            # Add note to photos that had src
            self._note_original_name(src, new_id)
            QMessageBox.information(dlg, "Merge", f"Merged {src} into {new_id}. Photos updated: {affected}")
            self._populate_deer_id_dropdown()
            dlg.accept()

        ok_btn.clicked.connect(do_merge)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()

    def _format_additional_entry(self, entry: dict) -> str:
        deer_id = entry.get("deer_id") or ""
        age = entry.get("age_class") or "Unknown"
        lp_min = entry.get("left_points_min")
        rp_min = entry.get("right_points_min")
        lp_unc = entry.get("left_points_uncertain")
        rp_unc = entry.get("right_points_uncertain")
        ab_min = entry.get("abnormal_points_min")
        ab_max = entry.get("abnormal_points_max")
        def fmt_typical(lo, uncertain):
            if lo is None:
                return "?"
            suffix = "?" if uncertain else ""
            return f"{lo}{suffix}"
        def fmt_range(lo, hi):
            if lo is None and hi is None:
                return "?"
            if lo is None:
                return f"?-{hi}"
            if hi is None:
                return f"{lo}+"
            if lo == hi:
                return f"{lo}"
            return f"{lo}-{hi}"
        return f"{deer_id} | Age: {age} | L:{fmt_typical(lp_min, lp_unc)} R:{fmt_typical(rp_min, rp_unc)} Ab:{fmt_range(ab_min, ab_max)}"

    def _make_range_row(
        self,
        spin_min: QSpinBox,
        spin_max: QSpinBox,
        plus_checkbox: Optional[QCheckBox] = None,
        uncertain_checkbox: Optional[QCheckBox] = None,
    ) -> QWidget:
        """Build a compact widget containing min/max spins and optional + / uncertain toggles."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Min"))
        layout.addWidget(spin_min)
        layout.addWidget(QLabel("Max"))
        layout.addWidget(spin_max)
        if plus_checkbox:
            layout.addWidget(plus_checkbox)
        if uncertain_checkbox:
            layout.addWidget(uncertain_checkbox)
        layout.addStretch()
        return container

    def _normalize_range_spins(self, spin_min: QSpinBox, spin_max: QSpinBox):
        """If only one side is set, copy it to the other; enforce min <= max."""
        lo = spin_min.value()
        hi = spin_max.value()
        if lo and not hi:
            hi = lo
        elif hi and not lo:
            lo = hi
        if hi and lo and lo > hi:
            lo, hi = hi, lo
        spin_min.setValue(lo)
        spin_max.setValue(hi)

    def _make_typical_row(self, spin_min: QSpinBox, uncertain_checkbox: QCheckBox) -> QWidget:
        """Single value + uncertain toggle for typical points."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Min"))
        layout.addWidget(spin_min)
        layout.addWidget(uncertain_checkbox)
        layout.addStretch()
        return container

    def _get_typical_values(self, spin_min: QSpinBox, uncertain_checkbox: QCheckBox) -> tuple:
        """Return (min, uncertain) for typical points."""
        val = spin_min.value()
        return (None if val == 0 else val, uncertain_checkbox.isChecked())

    def _get_range_values(self, spin_min: QSpinBox, spin_max: QSpinBox, plus_checkbox: Optional[QCheckBox], uncertain_checkbox: Optional[QCheckBox]) -> tuple:
        """Return (min, max) with None for empty. If uncertain is False, treat as exact."""
        lo = spin_min.value()
        hi = spin_max.value()
        uncertain = uncertain_checkbox.isChecked() if uncertain_checkbox else False
        if not uncertain:
            # Exact value if provided
            return (None if lo == 0 else lo, None if lo == 0 else lo)
        if plus_checkbox and plus_checkbox.isChecked():
            return (None if lo == 0 else lo, None)
        if lo and not hi:
            hi = lo
        elif hi and not lo:
            lo = hi
        if hi and lo and lo > hi:
            lo, hi = hi, lo
        return (None if lo == 0 else lo, None if hi == 0 else hi)

    def _apply_plus_state(self, checkbox: QCheckBox, max_spin: QSpinBox, checked: bool):
        """Toggle plus mode; disable max spin when checked."""
        max_spin.setEnabled(not checked)
        if checked:
            max_spin.setValue(0)

    def _sync_plus_from_values(self, checkbox: QCheckBox, max_spin: QSpinBox, min_val: Optional[int], max_val: Optional[int]):
        """Set plus checkbox based on stored values."""
        is_plus = min_val is not None and max_val is None
        checkbox.blockSignals(True)
        checkbox.setChecked(is_plus)
        checkbox.blockSignals(False)
        self._apply_plus_state(checkbox, max_spin, is_plus)

    def _apply_uncertain_state(self, uncertain: QCheckBox, plus_checkbox: Optional[QCheckBox], max_spin: QSpinBox, checked: bool):
        """Enable/disable range inputs based on uncertain toggle."""
        max_spin.setEnabled(checked and (plus_checkbox is None or not plus_checkbox.isChecked()))
        if plus_checkbox:
            plus_checkbox.setEnabled(checked)
            if not checked:
                plus_checkbox.setChecked(False)
        if not checked:
            max_spin.setValue(0)

    def _sync_uncertain_from_values(self, checkbox: QCheckBox, max_spin: QSpinBox, min_val: Optional[int], max_val: Optional[int]):
        """Set uncertain checkbox based on stored values (range or plus)."""
        is_uncertain = (min_val is not None and max_val is None) or (min_val is not None and max_val is not None and min_val != max_val)
        checkbox.blockSignals(True)
        checkbox.setChecked(is_uncertain)
        checkbox.blockSignals(False)
        self._apply_uncertain_state(checkbox, None, max_spin, is_uncertain)

    def toggle_fullscreen(self):
        """Toggle full screen for the preview window."""
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_btn.setText("Full Screen")
            self.top_bar_widget.show()
            if hasattr(self, "right_panel"):
                self.right_panel.show()
            if hasattr(self, "_prev_splitter_sizes") and self._prev_splitter_sizes:
                self.splitter.setSizes(self._prev_splitter_sizes)
            self.show_details_top_btn.setVisible(self.toggle_panel_btn.isChecked())
        else:
            self._prev_splitter_sizes = self.splitter.sizes()
            self.showFullScreen()
            self.fullscreen_btn.setText("Exit Full Screen")
            self.top_bar_widget.hide()
            if hasattr(self, "right_panel"):
                self.right_panel.hide()
            self.splitter.setSizes([self.width(), 0])
            self.show_details_top_btn.show()

    def toggle_details_panel(self, hidden: bool):
        """Hide/show the right details panel."""
        if hidden:
            self.toggle_panel_btn.setText("Show Details")
            self.right_panel.hide()
            self._prev_splitter_sizes = self.splitter.sizes()
            self.splitter.setSizes([self.width(), 0])
            self.show_details_top_btn.show()
        else:
            self.toggle_panel_btn.setText("Hide Details")
            self.right_panel.show()
            self.show_details_top_btn.hide()
            if hasattr(self, "_prev_splitter_sizes") and self._prev_splitter_sizes:
                self.splitter.setSizes(self._prev_splitter_sizes)
            else:
                self.splitter.setSizes([1500, 400])

    def _load_pixmap_with_enhance(self, path: str) -> QPixmap:
        """Load image with optional auto-enhance."""
        if not self.auto_enhance_enabled:
            return QPixmap(path)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = ImageOps.autocontrast(img, cutoff=1)
                img = ImageEnhance.Color(img).enhance(1.1)
                img = ImageEnhance.Contrast(img).enhance(1.05)
                img = ImageEnhance.Brightness(img).enhance(1.02)
                data = img.tobytes("raw", "RGB")
                qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGB888)
                return QPixmap.fromImage(qimg)
        except Exception as exc:
            logger.warning(f"Auto-enhance failed for {path}: {exc}")
            return QPixmap(path)
