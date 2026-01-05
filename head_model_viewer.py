"""
Head Model Visualization Viewer

View the prediction visualizations from head keypoint model training.
Green = ground truth, Red = model prediction.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from pathlib import Path


class HeadModelViewer(QDialog):
    """Viewer for head keypoint model visualizations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Head Keypoint Model v1.0 - Predictions")
        self.setMinimumSize(900, 700)

        # Find visualization images
        self.viz_dir = Path(__file__).parent / "outputs" / "head_keypoints_v1" / "visualizations"
        self.images = sorted(self.viz_dir.glob("prediction_*.png")) if self.viz_dir.exists() else []
        self.current_index = 0

        self._setup_ui()
        self._load_current()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title and legend
        title = QLabel("Head Keypoint Model Predictions")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        legend = QLabel("Green = Ground Truth (your annotation)  |  Red = Model Prediction")
        legend.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(legend)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.image_label.setMinimumSize(600, 500)
        layout.addWidget(self.image_label, stretch=1)

        # Progress
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.progress_label)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self._go_prev)
        nav_layout.addWidget(self.prev_btn)

        nav_layout.addStretch()

        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

        # Stats frame
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background-color: #2a2a2a; padding: 10px; border-radius: 5px;")
        stats_layout = QVBoxLayout(stats_frame)

        stats_title = QLabel("Model Performance (v1.0)")
        stats_title.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(stats_title)

        stats_text = QLabel(
            "Training samples: 119 clean annotations\n"
            "Skull error: ~15% (30px on 200px crop)\n"
            "Nose error: ~21% (42px on 200px crop)\n\n"
            "The model is learning to find heads but needs more training data for precision."
        )
        stats_text.setStyleSheet("color: #aaa;")
        stats_layout.addWidget(stats_text)

        layout.addWidget(stats_frame)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def _load_current(self):
        if not self.images:
            self.image_label.setText("No visualizations found.\n\nRun training first:\npython training/train_head_keypoints.py")
            self.progress_label.setText("0 / 0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return

        # Load image
        img_path = self.images[self.current_index]
        pixmap = QPixmap(str(img_path))

        # Scale to fit
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

        # Update progress
        self.progress_label.setText(f"Sample {self.current_index + 1} / {len(self.images)}")

        # Update buttons
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.images) - 1)

    def _go_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current()

    def _go_next(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self._load_current()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._load_current()
