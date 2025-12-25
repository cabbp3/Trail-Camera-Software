"""Compare up to four photos side-by-side with independent zoom."""
import os
from typing import List
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QHBoxLayout,
    QGraphicsScene,
    QGraphicsPixmapItem,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from database import TrailCamDatabase
from preview_window import ImageGraphicsView


class ComparePane(QWidget):
    """Pane holding one image with independent zoom controls."""

    def __init__(self, title: str, pixmap: QPixmap):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel(title)
        layout.addWidget(self.title)

        self.view = ImageGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)

        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.view.zoom_fit)
        full_btn = QPushButton("100%")
        full_btn.clicked.connect(self.view.zoom_100)
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.clicked.connect(self.view.zoom_out)
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.clicked.connect(self.view.zoom_in)
        self.zoom_label = QLabel("100%")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.valueChanged.connect(self.on_slider_change)
        self.view.zoom_changed.connect(self.on_zoom_change)

        for w in [fit_btn, full_btn, zoom_out_btn, zoom_in_btn, self.zoom_label, self.zoom_slider]:
            controls.addWidget(w)

        layout.addLayout(controls)
        self.view.zoom_fit()

    def on_slider_change(self, value: int):
        self.view.set_zoom_level(value / 100.0)

    def on_zoom_change(self, scale: float):
        percent = int(scale * 100)
        self.zoom_label.setText(f"{percent}%")
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), percent)))
        self.zoom_slider.blockSignals(False)


class CompareWindow(QDialog):
    """Dialog to compare up to four photos side-by-side."""

    def __init__(self, photo_ids: List[int], db: TrailCamDatabase, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Compare Photos")
        self.resize(1600, 1000)
        grid = QGridLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(6)

        for idx, pid in enumerate(photo_ids[:4]):
            photo = self.db.get_photo_by_id(pid)
            if not photo:
                continue
            path = photo.get("file_path")
            if not path or not os.path.exists(path):
                continue
            pixmap = QPixmap(path)
            title = photo.get("original_name") or os.path.basename(path)
            pane = ComparePane(title, pixmap)
            row = idx // 2
            col = idx % 2
            grid.addWidget(pane, row, col)

        for i in range(2):
            grid.setColumnStretch(i, 1)
            grid.setRowStretch(i, 1)
