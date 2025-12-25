"""
Dialog for finding and removing duplicate photos.
"""
import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QWidget, QGridLayout, QCheckBox, QGroupBox,
    QMessageBox, QProgressDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from database import TrailCamDatabase


class HashCalculationThread(QThread):
    """Thread for calculating file hashes without blocking UI."""
    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal(dict)  # hash_map: {hash: [photo_dicts]}
    error = pyqtSignal(str)
    
    def __init__(self, photos: List[Dict], db: TrailCamDatabase):
        super().__init__()
        self.photos = photos
        self.db = db
        self.cancelled = False
    
    def run(self):
        """Calculate hashes for all photos."""
        hash_map = {}  # {hash: [photo_dicts]}
        total = len(self.photos)
        
        for i, photo in enumerate(self.photos):
            if self.cancelled:
                break
            
            file_path = photo['file_path']
            self.progress.emit(i + 1, total, os.path.basename(file_path))
            
            if not os.path.exists(file_path):
                continue
            
            try:
                # Calculate MD5 hash
                file_hash = self.calculate_hash(file_path)
                
                if file_hash not in hash_map:
                    hash_map[file_hash] = []
                hash_map[file_hash].append(photo)
                
            except Exception as e:
                self.error.emit(f"Error hashing {os.path.basename(file_path)}: {str(e)}")
        
        # Filter to only duplicates (hash with 2+ photos)
        duplicates = {h: photos for h, photos in hash_map.items() if len(photos) > 1}
        self.finished.emit(duplicates)
    
    def calculate_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for duplicate detection."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def cancel(self):
        """Cancel hash calculation."""
        self.cancelled = True


class DuplicateThumbnailWidget(QWidget):
    """Widget for displaying a duplicate photo with checkbox."""
    
    def __init__(self, photo: Dict, thumb_path: str = None):
        super().__init__()
        self.photo = photo
        self.thumb_path = thumb_path
        self.setFixedSize(120, 150)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Checkbox
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(False)  # Default to unchecked (will be kept)
        layout.addWidget(self.checkbox)
        
        # Thumbnail
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(110, 110)
        layout.addWidget(self.image_label)
        
        # File info
        file_name = os.path.basename(photo['file_path'])
        if len(file_name) > 15:
            file_name = file_name[:12] + "..."
        self.info_label = QLabel(file_name)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        self.load_thumbnail()
    
    def load_thumbnail(self):
        """Load thumbnail image."""
        try:
            if self.thumb_path and os.path.exists(self.thumb_path):
                pixmap = QPixmap(self.thumb_path)
            elif os.path.exists(self.photo['file_path']):
                pixmap = QPixmap(self.photo['file_path'])
            else:
                return
            
            if not pixmap.isNull():
                scaled = pixmap.scaled(110, 110, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(scaled)
        except Exception as e:
            print(f"Error loading thumbnail: {e}")


class DuplicateDialog(QDialog):
    """Dialog for selecting and removing duplicate photos."""
    
    def __init__(self, duplicates: Dict[str, List[Dict]], db: TrailCamDatabase, parent=None):
        super().__init__(parent)
        self.duplicates = duplicates
        self.db = db
        self.thumbnail_widgets = []
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("Remove Duplicate Photos")
        self.setMinimumSize(900, 600)
        
        main_layout = QVBoxLayout(self)
        
        # Instructions
        info_label = QLabel(
            "Found duplicate photos (identical files). Select which ones to delete.\n"
            "Checked photos will be deleted. Unchecked photos will be kept."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)
        
        # Scroll area for duplicate groups
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Create a group for each set of duplicates
        self.duplicate_groups = []  # List of lists of thumbnail widgets (one per duplicate group)
        
        for file_hash, photos in self.duplicates.items():
            group = QGroupBox(f"Duplicate Group ({len(photos)} identical photos)")
            group_layout = QVBoxLayout()
            
            # Thumbnails grid
            thumb_grid = QGridLayout()
            thumb_grid.setSpacing(10)
            
            # Sort photos by date (newest first)
            sorted_photos = sorted(
                photos,
                key=lambda p: p.get('date_taken', '') or p.get('import_date', ''),
                reverse=True
            )
            
            group_widgets = []
            for i, photo in enumerate(sorted_photos):
                row = i // 4
                col = i % 4
                
                thumb_widget = DuplicateThumbnailWidget(
                    photo,
                    photo.get('thumbnail_path')
                )
                thumb_grid.addWidget(thumb_widget, row, col)
                group_widgets.append(thumb_widget)
                self.thumbnail_widgets.append(thumb_widget)
                
                # Show file path and date
                file_path = photo['file_path']
                date_str = photo.get('date_taken', 'Unknown date')
                tooltip = f"{file_path}\nDate: {date_str}"
                thumb_widget.setToolTip(tooltip)
            
            self.duplicate_groups.append(group_widgets)
            
            group_layout.addLayout(thumb_grid)
            group.setLayout(group_layout)
            scroll_layout.addWidget(group)
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        
        # Set default: keep newest in each group
        self.keep_newest()
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Auto-select buttons
        keep_newest_btn = QPushButton("Keep Newest (Auto)")
        keep_newest_btn.clicked.connect(self.keep_newest)
        button_layout.addWidget(keep_newest_btn)
        
        keep_oldest_btn = QPushButton("Keep Oldest (Auto)")
        keep_oldest_btn.clicked.connect(self.keep_oldest)
        button_layout.addWidget(keep_oldest_btn)
        
        button_layout.addStretch()
        
        # Select all / None
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_none)
        button_layout.addWidget(select_none_btn)
        
        button_layout.addStretch()
        
        # Delete button
        delete_btn = QPushButton("Delete Selected")
        delete_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                font-weight: bold;
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        delete_btn.clicked.connect(self.delete_selected)
        button_layout.addWidget(delete_btn)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        main_layout.addLayout(button_layout)
    
    def keep_newest(self):
        """Keep only the newest photo in each duplicate group."""
        for group_widgets in self.duplicate_groups:
            if group_widgets:
                # First widget is newest (sorted by date descending)
                group_widgets[0].checkbox.setChecked(False)  # Keep newest
                # Delete all others
                for w in group_widgets[1:]:
                    w.checkbox.setChecked(True)  # Delete others
    
    def keep_oldest(self):
        """Keep only the oldest photo in each duplicate group."""
        for group_widgets in self.duplicate_groups:
            if group_widgets:
                # Last widget is oldest (sorted by date descending)
                for w in group_widgets[:-1]:
                    w.checkbox.setChecked(True)  # Delete others
                group_widgets[-1].checkbox.setChecked(False)  # Keep oldest
    
    def select_all(self):
        """Select all photos for deletion."""
        for widget in self.thumbnail_widgets:
            widget.checkbox.setChecked(True)
    
    def select_none(self):
        """Deselect all photos."""
        for widget in self.thumbnail_widgets:
            widget.checkbox.setChecked(False)
    
    def delete_selected(self):
        """Delete selected duplicate photos."""
        selected_photos = [
            widget.photo for widget in self.thumbnail_widgets
            if widget.checkbox.isChecked()
        ]
        
        if not selected_photos:
            QMessageBox.information(self, "No Selection", "No photos selected for deletion.")
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to permanently delete {len(selected_photos)} photo(s)?\n\n"
            "This action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Delete files and remove from database
        deleted_count = 0
        error_count = 0
        
        for photo in selected_photos:
            try:
                file_path = photo['file_path']
                
                # Delete file from disk
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Delete thumbnail if exists
                thumb_path = photo.get('thumbnail_path')
                if thumb_path and os.path.exists(thumb_path):
                    os.remove(thumb_path)
                
                # Remove from database
                photo_id = self.db.get_photo_id(file_path)
                if photo_id:
                    self.db.remove_photo(photo_id)
                
                deleted_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"Error deleting {photo.get('file_path', 'unknown')}: {e}")
        
        # Show result
        if error_count > 0:
            QMessageBox.warning(
                self, "Deletion Complete",
                f"Deleted {deleted_count} photo(s).\n"
                f"Errors occurred with {error_count} photo(s)."
            )
        else:
            QMessageBox.information(
                self, "Deletion Complete",
                f"Successfully deleted {deleted_count} photo(s)."
            )
        
        self.accept()

