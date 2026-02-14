"""
Head Direction Annotation Window

A dedicated window for annotating deer head direction by drawing a line
from the top of the skull to the tip of the nose.

Navigates box-by-box with auto-zoom to each box.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QWidget, QSizePolicy, QMessageBox, QFrame,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSlider
)
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal, QEvent
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush, QWheelEvent

from database import TrailCamDatabase
from PIL import Image
import math
import random


class HeadAnnotationView(QGraphicsView):
    """Graphics view for displaying image and drawing head direction lines."""

    zoom_changed = pyqtSignal(float)
    line_drawn = pyqtSignal(float, float, float, float)  # x1, y1, x2, y2 in relative coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)

        self.pixmap_item = None
        self.original_pixmap = None
        self.img_width = 0
        self.img_height = 0

        # Current box being annotated
        self.current_box = None

        # Zoom state
        self.scale_factor = 1.15
        self.current_scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0

        # Drawing state
        self.drawing = False
        self.draw_start = None
        self.temp_line = None

        # Pan state (for right-click/middle-click pan)
        self.panning = False
        self.pan_start = None

        # Enable pinch-to-zoom gesture
        self.grabGesture(Qt.GestureType.PinchGesture)

    def set_image_and_box(self, file_path: str, box: dict):
        """Load image and focus on a specific box."""
        try:
            img = Image.open(file_path).convert("RGB")
            self.img_width, self.img_height = img.size

            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qimg)
            self.current_box = box

            self._rebuild_scene()

            # Set min_scale to fit level
            self._set_fit_as_min_scale()

            self._zoom_to_box()
        except Exception as e:
            self.scene_obj.clear()
            text_item = self.scene_obj.addText(f"Error loading image: {e}")
            text_item.setDefaultTextColor(QColor(255, 100, 100))

    def _set_fit_as_min_scale(self):
        """Calculate and set min_scale to the fit-to-view level."""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        # Temporarily fit to view to get the scale
        self.resetTransform()
        self.fitInView(self.scene_obj.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.min_scale = self.transform().m11()
        # Reset for actual zoom
        self.resetTransform()

    def _rebuild_scene(self):
        """Rebuild the scene with image and current box overlay."""
        self.scene_obj.clear()

        if self.original_pixmap is None or self.original_pixmap.isNull():
            return

        self.pixmap_item = self.scene_obj.addPixmap(self.original_pixmap)

        if self.current_box is None:
            return

        box = self.current_box
        has_head = box.get("head_x1") is not None

        # Draw the box - cyan for current focus
        box_color = QColor(0, 255, 255)
        pen = QPen(box_color, 4)

        x1 = box["x1"] * self.img_width
        y1 = box["y1"] * self.img_height
        x2 = box["x2"] * self.img_width
        y2 = box["y2"] * self.img_height

        self.scene_obj.addRect(x1, y1, x2 - x1, y2 - y1, pen)

        # Draw head line if exists
        if has_head:
            line_pen = QPen(QColor(255, 0, 255), 4)
            sx = box["head_x1"] * self.img_width
            sy = box["head_y1"] * self.img_height
            ex = box["head_x2"] * self.img_width
            ey = box["head_y2"] * self.img_height

            self.scene_obj.addLine(sx, sy, ex, ey, line_pen)
            self._draw_arrow(ex, ey, sx, sy)

            brush = QBrush(QColor(255, 0, 255))
            self.scene_obj.addEllipse(sx - 6, sy - 6, 12, 12, line_pen, brush)

    def _draw_arrow(self, x2, y2, x1, y1):
        """Draw arrow head at end of line."""
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_size = 15

        pen = QPen(QColor(255, 0, 255), 4)

        p1_x = x2 - arrow_size * math.cos(angle - math.pi / 6)
        p1_y = y2 - arrow_size * math.sin(angle - math.pi / 6)
        p2_x = x2 - arrow_size * math.cos(angle + math.pi / 6)
        p2_y = y2 - arrow_size * math.sin(angle + math.pi / 6)

        self.scene_obj.addLine(x2, y2, p1_x, p1_y, pen)
        self.scene_obj.addLine(x2, y2, p2_x, p2_y, pen)

    def _zoom_to_box(self):
        """Zoom and center on the current box with padding."""
        if self.current_box is None or self.original_pixmap is None or self.original_pixmap.isNull():
            return

        box = self.current_box
        x1 = box["x1"] * self.img_width
        y1 = box["y1"] * self.img_height
        x2 = box["x2"] * self.img_width
        y2 = box["y2"] * self.img_height

        # Add small padding around the box (15% on each side) for tight zoom
        width = x2 - x1
        height = y2 - y1
        padding_x = width * 0.15
        padding_y = height * 0.15

        rect = QRectF(
            x1 - padding_x,
            y1 - padding_y,
            width + padding_x * 2,
            height + padding_y * 2
        )

        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.current_scale = self.transform().m11()
        self.zoom_changed.emit(self.current_scale)

    def zoom_in(self):
        if self.current_scale < self.max_scale:
            self.scale(self.scale_factor, self.scale_factor)
            self.current_scale *= self.scale_factor
            self.zoom_changed.emit(self.current_scale)

    def zoom_out(self):
        # Don't zoom below fit level
        if self.current_scale <= self.min_scale + 0.01:
            return

        next_scale = self.current_scale / self.scale_factor
        if next_scale <= self.min_scale:
            # Snap to fit
            self.zoom_fit()
        else:
            self.scale(1 / self.scale_factor, 1 / self.scale_factor)
            self.current_scale = next_scale
            self.zoom_changed.emit(self.current_scale)

    def zoom_fit(self):
        """Fit the whole image in view."""
        if self.scene_obj.items():
            self.resetTransform()
            self.fitInView(self.scene_obj.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.current_scale = self.transform().m11()
            self.zoom_changed.emit(self.current_scale)

    def zoom_to_box(self):
        """Re-zoom to the current box."""
        self._zoom_to_box()

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            return self.gestureEvent(event)
        return super().event(event)

    def gestureEvent(self, event):
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Start drawing line
            scene_pos = self.mapToScene(event.pos())
            self.draw_start = scene_pos
            self.drawing = True
            pen = QPen(QColor(255, 0, 255, 180), 4)
            self.temp_line = self.scene_obj.addLine(
                scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y(), pen
            )
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            # Start panning
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.temp_line:
            scene_pos = self.mapToScene(event.pos())
            self.temp_line.setLine(
                self.draw_start.x(), self.draw_start.y(),
                scene_pos.x(), scene_pos.y()
            )
        elif self.panning and self.pan_start:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            scene_pos = self.mapToScene(event.pos())

            if self.temp_line:
                self.scene_obj.removeItem(self.temp_line)
                self.temp_line = None

            if self.img_width > 0 and self.img_height > 0:
                x1 = self.draw_start.x() / self.img_width
                y1 = self.draw_start.y() / self.img_height
                x2 = scene_pos.x() / self.img_width
                y2 = scene_pos.y() / self.img_height

                x1 = max(0, min(1, x1))
                y1 = max(0, min(1, y1))
                x2 = max(0, min(1, x2))
                y2 = max(0, min(1, y2))

                self.line_drawn.emit(x1, y1, x2, y2)

            self.drawing = False
            self.draw_start = None
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.panning = False
            self.pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click to clear head line."""
        if event.button() == Qt.MouseButton.LeftButton and self.current_box:
            if self.current_box.get("head_x1") is not None:
                self.current_box["head_x1"] = None
                self.current_box["head_y1"] = None
                self.current_box["head_x2"] = None
                self.current_box["head_y2"] = None
                self.current_box["_modified"] = True
                self._rebuild_scene()
        super().mouseDoubleClickEvent(event)


class HeadAnnotationWindow(QDialog):
    """Window for annotating deer head direction, box by box."""

    QUICK_NOTES = [
        ("Outside frame", "Head outside photo frame"),
        ("Obstructed: object", "Head obstructed by object"),
        ("Obstructed: body", "Head obstructed by body (walking away)"),
        ("Facing away", "Facing away - line estimated"),
        ("Low Quality", "Low quality photo - cannot annotate"),
        ("NOT A DEER", "Species error - not a deer"),
    ]

    def __init__(self, db: TrailCamDatabase, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Head Direction Annotation")
        self.setMinimumSize(1000, 800)

        # Build queue of individual boxes needing annotation
        self._build_box_queue()
        self.current_index = 0

        # History of completed boxes (for going back)
        self.history = []

        self._setup_ui()
        self._load_current()

    def _build_box_queue(self):
        """Build a shuffled queue of individual boxes needing annotation."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT ab.id, ab.photo_id, p.file_path,
                   ab.x1, ab.y1, ab.x2, ab.y2,
                   ab.head_x1, ab.head_y1, ab.head_x2, ab.head_y2, ab.head_notes
            FROM annotation_boxes ab
            JOIN photos p ON ab.photo_id = p.id
            JOIN tags t ON p.id = t.photo_id
            WHERE t.tag_name = 'Deer'
              AND ab.label IN ('subject', 'ai_animal')
              AND ab.head_x1 IS NULL
              AND (ab.head_notes IS NULL OR ab.head_notes = '')
        """)

        self.box_queue = []
        for row in cursor.fetchall():
            self.box_queue.append({
                "id": row[0],
                "photo_id": row[1],
                "file_path": row[2],
                "x1": row[3], "y1": row[4], "x2": row[5], "y2": row[6],
                "head_x1": row[7], "head_y1": row[8], "head_x2": row[9], "head_y2": row[10],
                "head_notes": row[11],
                "_modified": False
            })

        # Shuffle for variety across cameras and lighting
        random.shuffle(self.box_queue)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Draw line from TOP OF SKULL to NOSE TIP. Left-click and drag to draw.\n"
            "Right-click drag or middle-click drag to pan. Scroll wheel or pinch to zoom.\n"
            "Double-click to clear line. Use quick notes if head cannot be annotated."
        )
        instructions.setStyleSheet("color: #aaa; padding: 5px;")
        layout.addWidget(instructions)

        # Top bar
        top_bar = QHBoxLayout()

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        top_bar.addWidget(self.progress_label)

        top_bar.addStretch()

        # Zoom controls
        zoom_label = QLabel("Zoom:")
        top_bar.addWidget(zoom_label)

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedWidth(30)
        zoom_out_btn.clicked.connect(self._zoom_out)
        top_bar.addWidget(zoom_out_btn)

        self.zoom_display = QLabel("100%")
        self.zoom_display.setFixedWidth(50)
        self.zoom_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_bar.addWidget(self.zoom_display)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(30)
        zoom_in_btn.clicked.connect(self._zoom_in)
        top_bar.addWidget(zoom_in_btn)

        fit_btn = QPushButton("Fit Image")
        fit_btn.clicked.connect(self._zoom_fit)
        top_bar.addWidget(fit_btn)

        box_btn = QPushButton("Fit Box")
        box_btn.clicked.connect(self._zoom_to_box)
        top_bar.addWidget(box_btn)

        layout.addLayout(top_bar)

        # Graphics view
        self.view = HeadAnnotationView()
        self.view.zoom_changed.connect(self._update_zoom_display)
        self.view.line_drawn.connect(self._on_line_drawn)
        layout.addWidget(self.view, stretch=1)

        # Status bar
        status_frame = QFrame()
        status_frame.setStyleSheet("background-color: #2a2a2a; padding: 5px;")
        status_layout = QHBoxLayout(status_frame)

        self.status_label = QLabel()
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        self.total_label = QLabel()
        status_layout.addWidget(self.total_label)

        layout.addWidget(status_frame)

        # Quick notes
        quick_notes_layout = QHBoxLayout()
        quick_notes_label = QLabel("Quick notes:")
        quick_notes_layout.addWidget(quick_notes_label)

        for label, note_text in self.QUICK_NOTES:
            btn = QPushButton(label)
            btn.setStyleSheet("padding: 5px 10px;")
            btn.clicked.connect(lambda checked, t=note_text: self._add_quick_note(t))
            quick_notes_layout.addWidget(btn)

        quick_notes_layout.addStretch()
        layout.addLayout(quick_notes_layout)

        # Notes
        notes_label = QLabel("Notes (marks as reviewed):")
        layout.addWidget(notes_label)

        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(60)
        self.notes_edit.setPlaceholderText("Add notes if head cannot be annotated...")
        layout.addWidget(self.notes_edit)

        # Buttons
        button_layout = QHBoxLayout()

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self._go_prev)
        button_layout.addWidget(self.prev_btn)

        self.save_btn = QPushButton("Save & Next")
        self.save_btn.setStyleSheet("background-color: #2a5a2a; font-weight: bold;")
        self.save_btn.clicked.connect(self._save_and_next)
        button_layout.addWidget(self.save_btn)

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._go_next)
        button_layout.addWidget(self.skip_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def _add_quick_note(self, note_text):
        current = self.notes_edit.toPlainText().strip()
        if current:
            self.notes_edit.setPlainText(f"{current}; {note_text}")
        else:
            self.notes_edit.setPlainText(note_text)

    def _zoom_in(self):
        self.view.zoom_in()

    def _zoom_out(self):
        self.view.zoom_out()

    def _zoom_fit(self):
        self.view.zoom_fit()

    def _zoom_to_box(self):
        self.view.zoom_to_box()

    def _update_zoom_display(self, scale: float):
        self.zoom_display.setText(f"{int(scale * 100)}%")

    def _on_line_drawn(self, x1, y1, x2, y2):
        """Handle line drawn - apply to current box."""
        if not self.box_queue or self.current_index >= len(self.box_queue):
            return

        box = self.box_queue[self.current_index]
        box["head_x1"] = x1
        box["head_y1"] = y1
        box["head_x2"] = x2
        box["head_y2"] = y2
        box["_modified"] = True

        # Update view
        self.view.current_box = box
        self.view._rebuild_scene()
        self._update_status()

    def _load_current(self):
        if not self.box_queue:
            self.view.scene_obj.clear()
            text = self.view.scene_obj.addText("All deer boxes have been annotated!")
            text.setDefaultTextColor(QColor(100, 255, 100))
            self.progress_label.setText("Done!")
            self.status_label.setText("")
            self.prev_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.skip_btn.setEnabled(False)
            return

        if self.current_index >= len(self.box_queue):
            self.current_index = len(self.box_queue) - 1

        box = self.box_queue[self.current_index]
        self.view.set_image_and_box(box["file_path"], box)

        self.progress_label.setText(f"Box {self.current_index + 1} / {len(self.box_queue)}  (History: {len(self.history)})")
        self._update_status()
        self._update_total()

        # Enable Previous if we can go back in queue OR if there's history
        self.prev_btn.setEnabled(self.current_index > 0 or len(self.history) > 0)
        self.skip_btn.setEnabled(self.current_index < len(self.box_queue) - 1)
        self.save_btn.setEnabled(True)

        self.notes_edit.clear()

    def _update_status(self):
        if not self.box_queue or self.current_index >= len(self.box_queue):
            return
        box = self.box_queue[self.current_index]
        has_line = box.get("head_x1") is not None
        status = "Line drawn" if has_line else "Needs annotation"
        self.status_label.setText(status)

    def _update_total(self):
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN head_x1 IS NOT NULL OR (head_notes IS NOT NULL AND head_notes != '') THEN 1 ELSE 0 END) as done
            FROM annotation_boxes ab
            JOIN photos p ON ab.photo_id = p.id
            JOIN tags t ON p.id = t.photo_id
            WHERE t.tag_name = 'Deer'
              AND ab.label IN ('subject', 'ai_animal')
        """)
        row = cursor.fetchone()
        total = row[0] or 0
        done = row[1] or 0
        self.total_label.setText(f"Total: {done}/{total} done")

    def _save_current(self):
        if not self.box_queue or self.current_index >= len(self.box_queue):
            return

        box = self.box_queue[self.current_index]
        notes = self.notes_edit.toPlainText().strip() or None

        if box.get("head_x1") is not None:
            x1 = max(0.0, min(1.0, float(box["head_x1"])))
            y1 = max(0.0, min(1.0, float(box["head_y1"])))
            x2 = max(0.0, min(1.0, float(box["head_x2"])))
            y2 = max(0.0, min(1.0, float(box["head_y2"])))
            box["head_x1"] = x1
            box["head_y1"] = y1
            box["head_x2"] = x2
            box["head_y2"] = y2
            self.db.set_box_head_line(
                box["id"],
                x1, y1,
                x2, y2,
                notes
            )
        elif notes:
            self.db.set_box_head_line(box["id"], None, None, None, None, notes)

        box["_modified"] = False

    def _save_and_next(self):
        self._save_current()

        # Remove current box from queue and add to history
        if self.box_queue and self.current_index < len(self.box_queue):
            box = self.box_queue[self.current_index]
            if box.get("head_x1") is not None or self.notes_edit.toPlainText().strip():
                # Add to history before removing
                self.history.append(self.box_queue.pop(self.current_index))
                # Don't increment index since we removed current item
                if self.current_index >= len(self.box_queue):
                    self.current_index = max(0, len(self.box_queue) - 1)
            else:
                # No annotation made, just move to next
                if self.current_index < len(self.box_queue) - 1:
                    self.current_index += 1

        self._load_current()

    def _go_prev(self):
        """Go back - either in queue or into history."""
        if self.current_index > 0:
            # Still have items before in queue
            self.current_index -= 1
            self._load_current()
        elif self.history:
            # Go back into history - move last history item back to front of queue
            box = self.history.pop()
            # Re-fetch from database to get current state
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT ab.id, ab.photo_id, p.file_path,
                       ab.x1, ab.y1, ab.x2, ab.y2,
                       ab.head_x1, ab.head_y1, ab.head_x2, ab.head_y2, ab.head_notes
                FROM annotation_boxes ab
                JOIN photos p ON ab.photo_id = p.id
                WHERE ab.id = ?
            """, (box["id"],))
            row = cursor.fetchone()
            if row:
                refreshed_box = {
                    "id": row[0],
                    "photo_id": row[1],
                    "file_path": row[2],
                    "x1": row[3], "y1": row[4], "x2": row[5], "y2": row[6],
                    "head_x1": row[7], "head_y1": row[8], "head_x2": row[9], "head_y2": row[10],
                    "head_notes": row[11],
                    "_modified": False
                }
                self.box_queue.insert(0, refreshed_box)
                self.current_index = 0
            self._load_current()

    def _go_next(self):
        if self.current_index < len(self.box_queue) - 1:
            self.current_index += 1
            self._load_current()

    def closeEvent(self, event):
        # Save any pending changes
        if self.box_queue and self.current_index < len(self.box_queue):
            box = self.box_queue[self.current_index]
            if box.get("_modified"):
                self._save_current()
        super().closeEvent(event)
