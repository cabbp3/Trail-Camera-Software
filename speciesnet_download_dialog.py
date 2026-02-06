"""First-time SpeciesNet model download dialog with progress feedback."""

import logging

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


class SpeciesNetDownloadWorker(QThread):
    """Background thread for downloading/initializing SpeciesNet models."""

    progress = pyqtSignal(str, float)  # message, percent (0-100)
    finished = pyqtSignal(bool, str)  # success, error_message

    def __init__(self, wrapper, parent=None):
        super().__init__(parent)
        self.wrapper = wrapper

    def run(self):
        try:
            success = self.wrapper.initialize(
                progress_callback=lambda msg, pct: self.progress.emit(msg, pct)
            )
            if success:
                self.finished.emit(True, "")
            else:
                self.finished.emit(
                    False, self.wrapper.error_message or "Unknown error"
                )
        except Exception as e:
            logger.error(f"SpeciesNet download failed: {e}")
            self.finished.emit(False, str(e))


class SpeciesNetDownloadDialog(QDialog):
    """Dialog shown during first-time SpeciesNet model download."""

    def __init__(self, wrapper, parent=None):
        super().__init__(parent)
        self.wrapper = wrapper
        self.success = False

        self.setWindowTitle("Setting Up SpeciesNet")
        self.setMinimumWidth(450)
        self.setWindowModality(Qt.WindowModality.WindowModal)

        layout = QVBoxLayout(self)

        self.info_label = QLabel(
            "SpeciesNet needs to download AI models on first use.\n"
            "This may take a few minutes depending on your internet speed.\n\n"
            "Model size: approximately 500 MB"
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.status_label = QLabel("Ready to download")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate until we get progress
        layout.addWidget(self.progress_bar)

        self.start_btn = QPushButton("Download and Setup")
        self.start_btn.clicked.connect(self._start_download)
        layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)

    def _start_download(self):
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Downloading SpeciesNet models...")

        self.worker = SpeciesNetDownloadWorker(self.wrapper, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, message, percent):
        self.status_label.setText(message)
        if percent > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(percent))

    def _on_finished(self, success, error):
        self.success = success
        if success:
            self.status_label.setText("SpeciesNet is ready!")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.accept()
        else:
            self.status_label.setText(f"Setup failed: {error}")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)
            QMessageBox.warning(
                self,
                "SpeciesNet Setup Failed",
                f"Could not set up SpeciesNet:\n\n{error}\n\n"
                "You can try again later from Tools > Run AI.",
            )
