"""
Main window for trail camera photo organizer.
"""
import os
from pathlib import Path
from typing import Optional, Dict, List, Union
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QFileDialog, QMessageBox,
    QScrollArea, QProgressDialog, QDateEdit, QGroupBox, QDialog,
    QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont
from PIL import Image
from database import TrailCamDatabase
from image_processor import import_photo, create_thumbnail, get_library_path, extract_exif_data
from preview_window import PreviewWindow
from compare_window import CompareWindow
from training.label_tool import TrainerWindow
from duplicate_dialog import DuplicateDialog, HashCalculationThread
from ai_suggester import CombinedSuggester


def find_image_files(folder: Path):
    """Find all JPG/JPEG files in folder (case-insensitive)."""
    # Find files with common case variations, then filter case-insensitively
    # This is more efficient than rglob("*") on large directories
    all_candidates = (
        list(folder.rglob("*.jpg")) + list(folder.rglob("*.JPG")) +
        list(folder.rglob("*.jpeg")) + list(folder.rglob("*.JPEG"))
    )
    # Filter for valid files and ensure case-insensitive matching
    image_extensions = {'.jpg', '.jpeg'}
    jpg_files = [
        f for f in all_candidates 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    # Remove duplicates (in case of case variations)
    return list(dict.fromkeys(jpg_files))


class ImportThread(QThread):
    """Thread for importing photos without blocking UI."""
    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, folder_path: str, db: TrailCamDatabase):
        super().__init__()
        self.folder_path = folder_path
        self.db = db
        self.cancelled = False
    
    def run(self):
        """Import all JPG/JPEG files from folder."""
        folder = Path(self.folder_path)
        jpg_files = find_image_files(folder)
        
        total = len(jpg_files)
        imported = 0
        
        for i, jpg_file in enumerate(jpg_files):
            if self.cancelled:
                break
            
            try:
                self.progress.emit(i + 1, total, jpg_file.name)
                
                # Check if already imported
                new_path, original_name, date_taken, camera_model = import_photo(str(jpg_file))
                
                # Check if photo already exists in database
                existing_id = self.db.get_photo_id(new_path)
                if existing_id:
                    continue
                
                # Create thumbnail
                thumb_path = create_thumbnail(new_path)
                
                # Add to database
                self.db.add_photo(new_path, original_name, date_taken, camera_model, thumb_path)
                imported += 1
                
            except Exception as e:
                self.error.emit(f"Error importing {jpg_file.name}: {str(e)}")
        
        self.finished.emit()
    
    def cancel(self):
        """Cancel import operation."""
        self.cancelled = True


class CheckableComboBox(QComboBox):
    """Combo box that supports multi-select via checkboxes."""

    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        if self.lineEdit():
            self.lineEdit().setReadOnly(True)
        self.view().pressed.connect(self.handle_item_pressed)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

    def addItem(self, text, userData=None):
        super().addItem(text, userData)
        index = self.model().index(self.count() - 1, 0)
        self.model().setData(index, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.update_display_text()

    def addItems(self, texts: List[str]):
        for text in texts:
            self.addItem(text)

    def handle_item_pressed(self, index):
        item = self.model().itemFromIndex(index)
        if item is None:
            return
        checked = item.checkState() == Qt.CheckState.Checked
        item.setCheckState(Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked)

        # Handle "All" logic: selecting All clears others; selecting others clears All
        if index.row() == 0 and self.itemText(0).lower() == "all":
            # clear all others
            for i in range(1, self.count()):
                self.model().item(i).setCheckState(Qt.CheckState.Unchecked)
        else:
            # uncheck "All"
            if self.count() > 0 and self.itemText(0).lower() == "all":
                all_item = self.model().item(0)
                all_item.setCheckState(Qt.CheckState.Unchecked)
        self.update_display_text()
        self.selection_changed.emit()

    def selected_items(self) -> List[str]:
        items = []
        for i in range(self.count()):
            item = self.model().item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                text = self.itemText(i)
                if text.lower() == "all":
                    continue
                items.append(text)
        return items

    def clear_checks(self):
        for i in range(self.count()):
            item = self.model().item(i)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.update_display_text()

    def set_checked_items(self, texts: List[str]):
        """Check items matching provided texts."""
        text_set = set(texts)
        for i in range(self.count()):
            item = self.model().item(i)
            if not item:
                continue
            if self.itemText(i) in text_set:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
        # If none selected, optionally check "All"? we leave unchecked = no filter
        self.update_display_text()

    def update_display_text(self):
        selected = self.selected_items()
        display = ", ".join(selected) if selected else "All"
        if self.lineEdit():
            self.lineEdit().setText(display)


class ThumbnailWidget(QWidget):
    """Widget for displaying a single thumbnail."""
    clicked = pyqtSignal(str)  # Emits file_path when clicked
    
    def __init__(self, file_path: str, thumb_path: Optional[str] = None):
        super().__init__()
        self.file_path = file_path
        self.thumb_path = thumb_path
        self.is_highlighted = False
        self.is_selected = False
        self.thumbnail_size = 200  # Clean 200x200 size
        self.setFixedSize(self.thumbnail_size, self.thumbnail_size)
        
        # Create layout once
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)  # Don't stretch - maintain aspect ratio
        layout.addWidget(self.image_label)
        
        self.setLayout(layout)
        self.update_style()
        self.load_thumbnail()
    
    def update_style(self):
        """Update widget style based on highlight state."""
        if self.is_highlighted or self.is_selected:
            color = "#FF9500" if self.is_selected else "#007AFF"
            style = f"""
                QWidget {{
                    border: 3px solid {color};
                    border-radius: 6px;
                    background-color: #f8f8f8;
                }}
                QWidget:hover {{
                    border: 3px solid {color};
                    background-color: #f0f0f0;
                }}
            """
        else:
            style = """
                QWidget {
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    background-color: #f8f8f8;
                }
                QWidget:hover {
                    border: 2px solid #007AFF;
                    background-color: #f5f5f5;
                }
            """
        self.setStyleSheet(style)
    
    def set_highlighted(self, highlighted: bool):
        """Set highlight state."""
        self.is_highlighted = highlighted
        self.update_style()
    
    def set_selected(self, selected: bool):
        """Set compare selection state."""
        self.is_selected = selected
        self.update_style()
    
    def load_thumbnail(self):
        """Load thumbnail image with proper aspect ratio and centering."""
        try:
            if self.thumb_path and os.path.exists(self.thumb_path):
                pixmap = QPixmap(self.thumb_path)
            elif os.path.exists(self.file_path):
                # Load full image and scale down
                pixmap = QPixmap(self.file_path)
            else:
                return
            
            if not pixmap.isNull():
                # Calculate size maintaining aspect ratio
                # Leave some padding for border (8px on each side = 16px total)
                max_size = self.thumbnail_size - 16
                
                # Get original dimensions
                orig_width = pixmap.width()
                orig_height = pixmap.height()
                
                # Calculate scaling to fit within max_size while maintaining aspect ratio
                if orig_width > orig_height:
                    # Landscape or square
                    scaled_width = max_size
                    scaled_height = int((orig_height * max_size) / orig_width)
                else:
                    # Portrait
                    scaled_height = max_size
                    scaled_width = int((orig_width * max_size) / orig_height)
                
                # Scale with high-quality transformation
                scaled = pixmap.scaled(
                    scaled_width, 
                    scaled_height, 
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Set pixmap - QLabel will center it automatically due to AlignCenter
                self.image_label.setPixmap(scaled)
        except Exception as e:
            print(f"Error loading thumbnail: {e}")
    
    def mousePressEvent(self, event):
        """Handle mouse click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.file_path)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.db = TrailCamDatabase()
        self.import_thread = None
        self.current_photos = []
        self.thumbnail_widgets = {}  # Map file_path -> ThumbnailWidget
        self.preview_window = None  # Reference to current preview window
        self.compare_mode = False
        self.compare_selection = []  # list of file paths
        self.suggester = CombinedSuggester()
        self.auto_enhance_all = True
        self._labeler = None
        self.init_ui()
        self.load_photos()
        # Check if library is empty and offer to import
        self.check_library_empty()
        # Open the labeler as the primary workflow
        QTimer.singleShot(0, self.open_labeler)
    
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("Trail Camera Photo Organizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Menu bar
        self.create_menu_bar()
        
        # Search bar
        search_group = QGroupBox("Search & Filter")
        search_layout = QHBoxLayout()
        
        # Tag filter
        search_layout.addWidget(QLabel("Tag:"))
        self.tag_combo = CheckableComboBox()
        self.tag_combo.addItem("All")
        self.tag_combo.addItems(["Deer", "Buck", "Doe", "Fawn", "Turkey", "Coyote", "Trash"])
        self.tag_combo.selection_changed.connect(self.apply_filters)
        search_layout.addWidget(self.tag_combo)
        
        # Deer ID filter
        search_layout.addWidget(QLabel("Deer ID:"))
        self.deer_id_input = QLineEdit()
        self.deer_id_input.setPlaceholderText("Enter deer ID...")
        self.deer_id_input.textChanged.connect(self.apply_filters)
        search_layout.addWidget(self.deer_id_input)

        # Age class filter
        search_layout.addWidget(QLabel("Age:"))
        self.age_combo = CheckableComboBox()
        self.age_combo.addItem("All")
        for age in ["Fawn", "Yearling", "2.5", "3.5", "4.5+", "Unknown"]:
            self.age_combo.addItem(age)
        self.age_combo.selection_changed.connect(self.apply_filters)
        search_layout.addWidget(self.age_combo)

        # Antler season filter (May–Apr)
        search_layout.addWidget(QLabel("Antler Season:"))
        self.season_combo = CheckableComboBox()
        self.season_combo.addItem("All")
        # Populate later from DB
        self.season_combo.selection_changed.connect(self.apply_filters)
        search_layout.addWidget(self.season_combo)

        # Suggested tag filter (AI)
        search_layout.addWidget(QLabel("Suggested:"))
        self.suggested_combo = CheckableComboBox()
        self.suggested_combo.addItem("All")
        self.suggested_combo.selection_changed.connect(self.apply_filters)
        search_layout.addWidget(self.suggested_combo)

        # Date range
        search_layout.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        self.date_from.setCalendarPopup(True)
        self.date_from.dateChanged.connect(self.apply_filters)
        search_layout.addWidget(self.date_from)
        
        search_layout.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.dateChanged.connect(self.apply_filters)
        search_layout.addWidget(self.date_to)

        # Compare controls
        compare_box = QGroupBox("Compare")
        compare_layout = QVBoxLayout()
        compare_row = QHBoxLayout()
        self.compare_toggle = QPushButton("Select for Compare")
        self.compare_toggle.setCheckable(True)
        self.compare_toggle.setToolTip("Toggle compare selection mode (up to 4 photos)")
        self.compare_toggle.toggled.connect(self.on_compare_toggle)
        compare_btn = QPushButton("Open Compare")
        compare_btn.clicked.connect(self.launch_compare_from_selection)
        compare_row.addWidget(self.compare_toggle)
        compare_row.addWidget(compare_btn)
        compare_layout.addLayout(compare_row)
        self.compare_list = QListWidget()
        compare_layout.addWidget(self.compare_list)
        compare_box.setLayout(compare_layout)
        search_layout.addWidget(compare_box)
        
        # Clear button
        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self.clear_filters)
        search_layout.addWidget(clear_btn)
        
        search_layout.addStretch()
        search_group.setLayout(search_layout)
        main_layout.addWidget(search_group)
        
        # Thumbnail grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(15)  # Increased spacing for larger thumbnails
        self.grid_layout.setContentsMargins(15, 15, 15, 15)
        
        scroll_area.setWidget(self.grid_widget)
        main_layout.addWidget(scroll_area)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        import_action = file_menu.addAction("Import Folder...")
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self.import_folder)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        remove_duplicates_action = tools_menu.addAction("Remove Duplicate/Identical Photos...")
        remove_duplicates_action.triggered.connect(self.remove_duplicates)
        compare_action = tools_menu.addAction("Compare Photos...")
        compare_action.triggered.connect(self.compare_photos)
        suggest_action = tools_menu.addAction("Suggest Tags (AI)...")
        suggest_action.triggered.connect(self.run_ai_suggestions)
        suggest_all_action = tools_menu.addAction("Suggest Tags (AI) — All Photos")
        suggest_all_action.triggered.connect(self.run_ai_suggestions_all)
        labeler_action = tools_menu.addAction("Open Labeler")
        labeler_action.triggered.connect(self.open_labeler)

        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        self.enhance_toggle_action = settings_menu.addAction("Auto Enhance All")
        self.enhance_toggle_action.setCheckable(True)
        self.enhance_toggle_action.setChecked(True)
        self.enhance_toggle_action.toggled.connect(self.toggle_global_enhance)
    
    def import_folder(self):
        """Import photos from a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Import")
        if not folder:
            return
        
        # Check if folder contains JPG/JPEG files
        folder_path = Path(folder)
        jpg_files = find_image_files(folder_path)
        
        if not jpg_files:
            QMessageBox.information(self, "No Images", 
                                  "No JPG/JPEG files found in the selected folder.")
            return
        
        reply = QMessageBox.question(
            self, "Import Photos",
            f"Found {len(jpg_files)} JPG file(s). Import them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.start_import(folder)
    
    def start_import(self, folder_path: str):
        """Start importing photos."""
        # Create progress dialog
        self.progress_dialog = QProgressDialog("Importing photos...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        
        # Create and start import thread
        self.import_thread = ImportThread(folder_path, self.db)
        self.import_thread.progress.connect(self.update_progress)
        self.import_thread.finished.connect(self.import_finished)
        self.import_thread.error.connect(self.import_error)
        self.progress_dialog.canceled.connect(self.import_thread.cancel)
        
        self.import_thread.start()
    
    def update_progress(self, current: int, total: int, filename: str):
        """Update progress dialog."""
        self.progress_dialog.setMaximum(total)
        self.progress_dialog.setValue(current)
        self.progress_dialog.setLabelText(f"Importing: {filename}")
    
    def import_finished(self):
        """Handle import completion."""
        self.progress_dialog.close()
        self.statusBar().showMessage("Import completed", 3000)
        self.load_photos()
        QMessageBox.information(self, "Import Complete", 
                              "Photos have been imported successfully!")
    
    def import_error(self, error_msg: str):
        """Handle import error."""
        QMessageBox.warning(self, "Import Error", error_msg)
    
    def clear_filters(self):
        """Clear all search filters."""
        self.tag_combo.clear_checks()
        self.deer_id_input.clear()
        if hasattr(self, "age_combo"):
            self.age_combo.clear_checks()
        if hasattr(self, "season_combo"):
            self.season_combo.clear_checks()
        if hasattr(self, "suggested_combo"):
            self.suggested_combo.clear_checks()
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        self.date_to.setDate(QDate.currentDate())
        self.load_photos()
    
    def apply_filters(self):
        """Apply search filters."""
        tags = self.tag_combo.selected_items()
        if not tags:
            tags = None
        
        deer_id = self.deer_id_input.text().strip()
        if not deer_id:
            deer_id = None

        age_class = self.age_combo.selected_items()
        if not age_class:
            age_class = None

        season_selection = self.season_combo.selected_items()
        season_year = None
        if season_selection:
            season_year = []
            for season_text in season_selection:
                try:
                    season_year.append(int(season_text.split("-")[0]))
                except ValueError:
                    continue
            if not season_year:
                season_year = None
        
        suggested_tags = self.suggested_combo.selected_items()
        if not suggested_tags:
            suggested_tags = None
        
        camera_location = None
        # If we add a UI later, hook here; for now leave None
        
        date_from = self.date_from.date().toString("yyyy-MM-dd")
        date_to = self.date_to.date().toString("yyyy-MM-dd")
        
        # Search photos
        self.current_photos = self.db.search_photos(
            tag=tags,
            deer_id=deer_id,
            age_class=age_class,
            season_year=season_year,
            suggested_tag=suggested_tags,
            date_from=date_from,
            date_to=date_to
        )
        
        self.display_photos()
    
    def load_photos(self):
        """Load all photos from database and verify they exist on disk."""
        all_photos = self.db.get_all_photos()
        # Filter out photos that no longer exist on disk
        self.current_photos = [
            photo for photo in all_photos 
            if os.path.exists(photo['file_path'])
        ]
        self.populate_season_filter()
        self.populate_suggested_filter()
        
        # Remove photos from database that no longer exist
        missing_photos = [
            photo for photo in all_photos 
            if photo not in self.current_photos
        ]
        if missing_photos:
            # Optionally clean up missing photos from database
            # For now, just display what exists
            pass
        
        self.display_photos()
    
    def check_library_empty(self):
        """Check if library is empty and optionally prompt for import."""
        if len(self.current_photos) == 0:
            # Check if library folder exists and has any photos
            library_path = get_library_path()
            if library_path.exists():
                # Check for any JPG/JPEG files in library
                existing_files = find_image_files(library_path)
                if existing_files:
                    # Library has files but not in database - offer to scan
                    reply = QMessageBox.question(
                        self, "Library Found",
                        f"Found {len(existing_files)} photo(s) in library folder but not in database.\n"
                        "Would you like to scan and add them to the database?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.scan_library_folder()
                else:
                    # Library folder exists but is empty
                    reply = QMessageBox.question(
                        self, "Empty Library",
                        "Your photo library is empty.\n"
                        "Would you like to import photos from a folder?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.import_folder()
            else:
                # Library folder doesn't exist - offer to import
                reply = QMessageBox.question(
                    self, "Welcome",
                    "Welcome to Trail Camera Photo Organizer!\n"
                    "Your library is empty. Would you like to import photos?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.import_folder()
    
    def scan_library_folder(self):
        """Scan library folder and add existing photos to database."""
        library_path = get_library_path()
        if not library_path.exists():
            return
        
        existing_files = find_image_files(library_path)
        if not existing_files:
            return
        
        # Create progress dialog
        self.progress_dialog = QProgressDialog("Scanning library folder...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        
        total = len(existing_files)
        added = 0
        
        for i, jpg_file in enumerate(existing_files):
            if self.progress_dialog.wasCanceled():
                break
            
            self.progress_dialog.setMaximum(total)
            self.progress_dialog.setValue(i + 1)
            self.progress_dialog.setLabelText(f"Scanning: {jpg_file.name}")
            
            try:
                file_path = str(jpg_file)
                # Check if already in database
                existing_id = self.db.get_photo_id(file_path)
                if existing_id:
                    continue
                
                # Extract EXIF data
                date_taken, camera_model = extract_exif_data(file_path)
                
                # Create thumbnail
                thumb_path = create_thumbnail(file_path)
                
                # Add to database
                self.db.add_photo(file_path, jpg_file.name, date_taken or "", camera_model or "", thumb_path)
                added += 1
                
            except Exception as e:
                print(f"Error scanning {jpg_file.name}: {e}")
        
        self.progress_dialog.close()
        self.statusBar().showMessage(f"Added {added} photo(s) to database", 3000)
        self.load_photos()
    
    def display_photos(self):
        """Display photos in grid."""
        # Clear existing thumbnails
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.thumbnail_widgets.clear()
        
        # Add thumbnails (5 columns)
        cols = 5
        for i, photo in enumerate(self.current_photos):
            row = i // cols
            col = i % cols
            
            thumb_widget = ThumbnailWidget(
                photo['file_path'],
                photo.get('thumbnail_path')
            )
            thumb_widget.clicked.connect(self.on_thumbnail_clicked)
            self.grid_layout.addWidget(thumb_widget, row, col)
            self.thumbnail_widgets[photo['file_path']] = thumb_widget
            thumb_widget.set_selected(photo['file_path'] in self.compare_selection)
        
        # Update status
        count = len(self.current_photos)
        self.statusBar().showMessage(f"Displaying {count} photo(s)")

    def on_thumbnail_clicked(self, file_path: str):
        """Handle thumbnail click: preview or compare selection."""
        if self.compare_mode:
            self.toggle_compare_selection(file_path)
            return
        self.show_preview(file_path)

    def show_preview(self, file_path: str):
        """Open the full-size preview window — 100% working version"""
        photo_id = self.db.get_photo_id(file_path)
        if photo_id is None:
            return

        # Close any old preview window
        if self.preview_window:
            self.preview_window.close()

        # Build a clean list of photo IDs (integers only)
        photo_id_list: List[int] = []
        for photo in self.current_photos:
            pid = photo.get('id') or self.db.get_photo_id(photo['file_path'])
            if pid is None:
                continue
            try:
                photo_id_list.append(int(pid))
            except (TypeError, ValueError):
                continue

        # Find our current position
        current_index = photo_id_list.index(photo_id) if photo_id in photo_id_list else 0

        # Open the preview window
        self.preview_window = PreviewWindow(
            photo_id=photo_id,
            db=self.db,
            photo_list=photo_id_list,
            current_index=current_index,
            parent=self,
            auto_enhance_default=self.auto_enhance_all
        )

        # Connect signals
        self.preview_window.photo_updated.connect(self.apply_filters)
        self.preview_window.photo_changed.connect(self.on_preview_photo_changed)
        self.preview_window.finished.connect(self.on_preview_closed)

        # Show it
        self.preview_window.show()
        self.preview_window.raise_()
        self.preview_window.activateWindow()

        # Highlight the thumbnail
        self.update_thumbnail_highlight(file_path)

    def on_preview_photo_changed(self, photo_id: int):
        """Handle photo change in preview window."""
        # Find the photo by ID and update highlight
        for photo in self.current_photos:
            if self.db.get_photo_id(photo['file_path']) == photo_id:
                self.update_thumbnail_highlight(photo['file_path'])
                break
    
    def on_preview_closed(self):
        """Handle preview window closing."""
        self.clear_thumbnail_highlights()
        self.preview_window = None
    
    def update_thumbnail_highlight(self, file_path: str):
        """Update thumbnail highlight for given file path."""
        self.clear_thumbnail_highlights()
        if file_path in self.thumbnail_widgets:
            self.thumbnail_widgets[file_path].set_highlighted(True)
        # keep compare selection styling
        for path in self.compare_selection:
            if path in self.thumbnail_widgets:
                self.thumbnail_widgets[path].set_selected(True)
    
    def clear_thumbnail_highlights(self):
        """Clear all thumbnail highlights."""
        for thumb_widget in self.thumbnail_widgets.values():
            thumb_widget.set_highlighted(False)
            thumb_widget.set_selected(False)
        # Re-apply compare selection state
        for path in self.compare_selection:
            if path in self.thumbnail_widgets:
                self.thumbnail_widgets[path].set_selected(True)
    
    def remove_duplicates(self):
        """Find and remove duplicate photos."""
        # Get all photos from database
        all_photos = self.db.get_all_photos()
        
        if len(all_photos) < 2:
            QMessageBox.information(
                self, "No Duplicates",
                "Not enough photos to check for duplicates.\n"
                "Need at least 2 photos in the library."
            )
            return
        
        # Filter to only photos that exist on disk
        existing_photos = [p for p in all_photos if os.path.exists(p['file_path'])]
        
        if len(existing_photos) < 2:
            QMessageBox.information(
                self, "No Duplicates",
                "Not enough existing photos to check for duplicates."
            )
            return
        
        # Create progress dialog
        self.progress_dialog = QProgressDialog(
            "Calculating file hashes to find duplicates...", 
            "Cancel", 0, 0, self
        )
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        
        # Create and start hash calculation thread
        self.hash_thread = HashCalculationThread(existing_photos, self.db)
        self.hash_thread.progress.connect(self.update_hash_progress)
        self.hash_thread.finished.connect(self.on_hash_finished)
        self.hash_thread.error.connect(self.hash_error)
        self.progress_dialog.canceled.connect(self.hash_thread.cancel)
        
        self.hash_thread.start()
    
    def update_hash_progress(self, current: int, total: int, filename: str):
        """Update hash calculation progress."""
        self.progress_dialog.setMaximum(total)
        self.progress_dialog.setValue(current)
        self.progress_dialog.setLabelText(f"Checking: {filename}")
    
    def on_hash_finished(self, duplicates: Dict[str, List[Dict]]):
        """Handle hash calculation completion."""
        self.progress_dialog.close()
        
        if not duplicates:
            QMessageBox.information(
                self, "No Duplicates Found",
                "No duplicate photos found in your library."
            )
            return
        
        # Count total duplicate photos
        total_duplicates = sum(len(photos) - 1 for photos in duplicates.values())
        
        # Show duplicate dialog
        dialog = DuplicateDialog(duplicates, self.db, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Refresh the photo grid
            self.load_photos()
            self.statusBar().showMessage(
                f"Removed duplicate photos. Displaying {len(self.current_photos)} photo(s)",
                5000
            )
    
    def hash_error(self, error_msg: str):
        """Handle hash calculation error."""
        QMessageBox.warning(self, "Hash Calculation Error", error_msg)
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.import_thread and self.import_thread.isRunning():
            reply = QMessageBox.question(
                self, "Import in Progress",
                "Import is still in progress. Do you want to cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.import_thread.cancel()
                self.import_thread.wait()
            else:
                event.ignore()
                return
        
        self.db.close()
        event.accept()

    def populate_season_filter(self):
        """Populate antler season dropdown from database values."""
        if not hasattr(self, "season_combo"):
            return
        current_values = self.season_combo.selected_items()
        seasons = self.db.get_seasons()
        self.season_combo.blockSignals(True)
        self.season_combo.clear()
        self.season_combo.addItem("All")
        for season in seasons:
            self.season_combo.addItem(self.db.format_season_label(season))
        # Restore selections if possible
        if current_values:
            valid = [val for val in current_values if self.season_combo.findText(val) >= 0]
            self.season_combo.set_checked_items(valid)
        else:
            self.season_combo.clear_checks()
        self.season_combo.blockSignals(False)

    def populate_suggested_filter(self):
        """Populate suggested tag dropdown from database values."""
        if not hasattr(self, "suggested_combo"):
            return
        current_values = self.suggested_combo.selected_items()
        suggestions = self.db.get_suggested_tags()
        self.suggested_combo.blockSignals(True)
        self.suggested_combo.clear()
        self.suggested_combo.addItem("All")
        for tag in suggestions:
            self.suggested_combo.addItem(tag)
        if current_values:
            valid = [val for val in current_values if self.suggested_combo.findText(val) >= 0]
            self.suggested_combo.set_checked_items(valid)
        else:
            self.suggested_combo.clear_checks()
        self.suggested_combo.blockSignals(False)

    def compare_photos(self):
        """Legacy compare launcher (unused)."""
        self.launch_compare_from_selection()

    def on_compare_toggle(self, checked: bool):
        self.compare_mode = checked
        if checked:
            self.statusBar().showMessage("Compare mode: click thumbnails to select up to 4 photos")
        else:
            self.statusBar().showMessage("Compare mode off")

    def toggle_compare_selection(self, file_path: str):
        """Add/remove a photo from compare selection."""
        if file_path in self.compare_selection:
            self.compare_selection.remove(file_path)
        else:
            if len(self.compare_selection) >= 4:
                QMessageBox.information(self, "Compare Limit", "You can compare up to 4 photos at once.")
                return
            self.compare_selection.append(file_path)
        self.update_compare_list()
        if file_path in self.thumbnail_widgets:
            self.thumbnail_widgets[file_path].set_selected(file_path in self.compare_selection)

    def update_compare_list(self):
        """Refresh UI list of selected compare items."""
        self.compare_list.clear()
        for path in self.compare_selection:
            display = os.path.basename(path)
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.compare_list.addItem(item)

    def launch_compare_from_selection(self):
        """Launch compare window based on current selection."""
        if len(self.compare_selection) < 2:
            QMessageBox.information(self, "Select Photos", "Select at least 2 photos (up to 4) using compare mode.")
            return
        photo_ids = []
        for path in self.compare_selection:
            pid = self.db.get_photo_id(path)
            if pid:
                photo_ids.append(pid)
        if len(photo_ids) < 2:
            QMessageBox.warning(self, "Selection Error", "Could not resolve selected photos.")
            return
        compare_window = CompareWindow(photo_ids=photo_ids[:4], db=self.db, parent=self)
        compare_window.show()

    def open_labeler(self):
        """Open the integrated labeler (trainer UI)."""
        if not self._labeler:
            self._labeler = TrainerWindow()
        self._labeler.show()
        self._labeler.raise_()
        self._labeler.activateWindow()

    def run_ai_suggestions(self):
        """Run AI species suggestions over current photos."""
        if not self.suggester or not self.suggester.ready:
            QMessageBox.information(
                self,
                "AI Model Not Available",
                "No AI model found. Place a species model at models/species.onnx "
                "with labels.txt, then retry."
            )
            return
        if not self.current_photos:
            QMessageBox.information(self, "No Photos", "No photos available to run suggestions.")
            return

        updated = 0
        for photo in self.current_photos:
            path = photo.get("file_path")
            if not path or not os.path.exists(path):
                continue
            result = self.suggester.predict(path)
            if result:
                label, conf = result
                pid = photo.get("id") or self.db.get_photo_id(path)
                if pid:
                    self.db.set_suggested_tag(pid, label, conf)
                    updated += 1
        QMessageBox.information(
            self,
            "Suggestions Complete",
            f"AI suggestions stored for {updated} photo(s).\n"
            "You can open photos to review and apply tags."
        )

    def run_ai_suggestions_all(self):
        """Run AI suggestions over all photos in the database."""
        if not self.suggester or not self.suggester.ready:
            QMessageBox.information(
                self,
                "AI Model Not Available",
                "No AI model found. Place a species model at models/species.onnx "
                "with labels.txt, then retry."
            )
            return
        all_photos = self.db.get_all_photos()
        if not all_photos:
            QMessageBox.information(self, "No Photos", "No photos available to run suggestions.")
            return
        updated = 0
        for photo in all_photos:
            path = photo.get("file_path")
            if not path or not os.path.exists(path):
                continue
            result = self.suggester.predict(path)
            if result:
                label, conf = result
                pid = photo.get("id") or self.db.get_photo_id(path)
                if pid:
                    self.db.set_suggested_tag(pid, label, conf)
                    updated += 1
        QMessageBox.information(
            self,
            "Suggestions Complete",
            f"AI suggestions stored for {updated} photo(s) across the library.\n"
            "You can open photos to review and apply tags."
        )
    
    def toggle_global_enhance(self, checked: bool):
        """Toggle default auto-enhance for preview windows."""
        self.auto_enhance_all = checked

    def set_quick_filters(self, tag: str = None, age: str = None):
        """Apply quick filter presets."""
        if tag:
            self.tag_combo.set_checked_items([tag])
        if age:
            self.age_combo.set_checked_items([age])
        self.apply_filters()

    def set_current_season_filter(self):
        """Set filter to the current antler season."""
        current_date = QDate.currentDate().toString("yyyy-MM-dd")
        season_year = self.db.compute_season_year(current_date)
        label = self.db.format_season_label(season_year)
        self.season_combo.set_checked_items([label])
        self.apply_filters()
