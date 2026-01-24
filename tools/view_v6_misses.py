#!/usr/bin/env python3
"""
View photos where MegaDetector v6 missed detections that v5 found.
Shows missed detection boxes in BLUE.

Usage:
    python tools/view_v6_misses.py
"""

import sys
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QMessageBox
)
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QFont
from PyQt6.QtCore import Qt, QRectF


class V6MissViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MegaDetector v6 Missed Detections Review")
        self.resize(1400, 900)

        # Load the queue
        self.queue_path = Path.home() / '.trailcam' / 'v6_missed_review_queue.json'
        if not self.queue_path.exists():
            QMessageBox.critical(self, "Error", f"Queue file not found:\n{self.queue_path}")
            sys.exit(1)

        with open(self.queue_path) as f:
            self.queue = json.load(f)

        self.current_index = 0

        self._setup_ui()
        self._load_current()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel - list
        left = QVBoxLayout()
        left.addWidget(QLabel(f"<b>{len(self.queue)} photos with missed detections</b>"))

        self.list_widget = QListWidget()
        for i, item in enumerate(self.queue):
            name = Path(item['file_path']).name
            missed = len(item['missed_boxes'])
            self.list_widget.addItem(f"{i+1}. {name} ({missed} missed)")
        self.list_widget.currentRowChanged.connect(self._on_select)
        self.list_widget.setMaximumWidth(400)
        left.addWidget(self.list_widget)

        # Legend
        legend = QLabel(
            "<b>Box Colors:</b><br>"
            "🔵 <span style='color:blue'>BLUE = v5 found, v6 missed</span><br>"
            "(These are what v6 failed to detect)"
        )
        left.addWidget(legend)

        layout.addLayout(left)

        # Right panel - image
        right = QVBoxLayout()

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        right.addWidget(self.info_label)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        right.addWidget(self.view, 1)

        # Navigation
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self._prev)
        nav.addWidget(self.prev_btn)

        self.counter_label = QLabel()
        nav.addWidget(self.counter_label)

        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.next_btn)

        nav.addStretch()

        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_in_btn.clicked.connect(lambda: self.view.scale(1.25, 1.25))
        nav.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_out_btn.clicked.connect(lambda: self.view.scale(0.8, 0.8))
        nav.addWidget(self.zoom_out_btn)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self._fit_view)
        nav.addWidget(self.fit_btn)

        right.addLayout(nav)
        layout.addLayout(right, 1)

    def _load_current(self):
        if not self.queue:
            return

        item = self.queue[self.current_index]
        file_path = item['file_path']

        # Update list selection
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(self.current_index)
        self.list_widget.blockSignals(False)

        # Update counter
        self.counter_label.setText(f"{self.current_index + 1} / {len(self.queue)}")

        # Update info
        self.info_label.setText(
            f"<b>{Path(file_path).name}</b><br>"
            f"v5 found: {item['v5_detections']} animals | "
            f"v6 found: {item['v6_detections']} animals | "
            f"<span style='color:blue'><b>Missed: {len(item['missed_boxes'])}</b></span>"
        )

        # Load image
        self.scene.clear()
        if not Path(file_path).exists():
            self.info_label.setText(f"<span style='color:red'>File not found: {file_path}</span>")
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.info_label.setText(f"<span style='color:red'>Failed to load: {file_path}</span>")
            return

        self.scene.addPixmap(pixmap)
        w, h = pixmap.width(), pixmap.height()

        # Draw missed boxes in BLUE
        blue_pen = QPen(QColor(0, 100, 255))  # Bright blue
        blue_pen.setWidth(max(3, int(min(w, h) * 0.005)))  # Scale line width

        for box in item['missed_boxes']:
            bbox = box['bbox']  # (x, y, w, h) normalized
            x = bbox[0] * w
            y = bbox[1] * h
            bw = bbox[2] * w
            bh = bbox[3] * h

            rect = QGraphicsRectItem(x, y, bw, bh)
            rect.setPen(blue_pen)
            rect.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.scene.addItem(rect)

            # Add label
            conf = box.get('confidence', 0)
            label_text = f"MISSED ({conf:.0%})"
            label = self.scene.addSimpleText(label_text)
            label.setPos(x, y - 25)
            label.setBrush(QBrush(QColor(0, 100, 255)))
            font = QFont()
            font.setPointSize(max(12, int(min(w, h) * 0.015)))
            font.setBold(True)
            label.setFont(font)

        self._fit_view()

    def _fit_view(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current()

    def _next(self):
        if self.current_index < len(self.queue) - 1:
            self.current_index += 1
            self._load_current()

    def _on_select(self, row):
        if 0 <= row < len(self.queue):
            self.current_index = row
            self._load_current()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self._prev()
        elif event.key() == Qt.Key.Key_Right:
            self._next()
        else:
            super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    window = V6MissViewer()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
