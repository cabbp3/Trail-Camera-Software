"""
Trail Cam Organizer - Main UI

Simplified photo browser for end users.
Features: Browse photos, filter/sort, accept AI labels, basic tagging.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QComboBox, QPushButton,
    QToolButton, QFrame, QScrollArea, QFormLayout, QLineEdit,
    QSpinBox, QMessageBox, QFileDialog, QProgressBar, QCheckBox,
    QMenu, QMenuBar, QApplication, QDialog, QDialogButtonBox,
    QGroupBox, QSlider, QGraphicsScene, QButtonGroup, QSizePolicy,
    QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QRectF
from PyQt6.QtGui import QPixmap, QAction, QShortcut, QKeySequence, QIcon, QColor, QPainter, QPen, QBrush

from database import TrailCamDatabase as Database
import image_processor
from preview_window import PreviewWindow, ImageGraphicsView
from compare_window import CompareWindow
from ai_detection import MegaDetectorV5
from ai_suggester import SpeciesSuggester, BuckDoeSuggester
from cuddelink_downloader import download_new_photos, check_server_status

# Common species for quick labeling
QUICK_SPECIES = ["Empty", "Deer", "Turkey", "Raccoon", "Coyote", "Squirrel"]

# Full species options (matches Trainer app)
SPECIES_OPTIONS = ["", "Coyote", "Deer", "Empty", "Person", "Raccoon", "Turkey", "Unknown", "Vehicle"]

# Tags that are not species
SEX_TAGS = {"buck", "doe"}

logger = logging.getLogger(__name__)


# Dark theme stylesheet
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
}
QMenuBar {
    background-color: #2d2d2d;
    color: #d4d4d4;
}
QMenuBar::item:selected {
    background-color: #3d3d3d;
}
QMenu {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #444;
}
QMenu::item:selected {
    background-color: #0078d4;
}
QListWidget {
    background-color: #000000;
    border: 1px solid #3c3c3c;
    color: #ffffff;
}
QListWidget::item {
    background-color: #000000;
    color: #ffffff;
}
QListWidget::item:selected {
    background-color: #0078d4;
    color: #ffffff;
}
QListWidget::item:hover {
    background-color: #1a1a1a;
}
QPushButton, QToolButton {
    background-color: #3c3c3c;
    border: 1px solid #555;
    color: #d4d4d4;
    padding: 5px 12px;
    border-radius: 3px;
}
QPushButton:hover, QToolButton:hover {
    background-color: #4a4a4a;
}
QPushButton:pressed, QToolButton:pressed {
    background-color: #0078d4;
}
QPushButton:disabled {
    background-color: #2d2d2d;
    color: #666;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #3c3c3c;
    border: 1px solid #555;
    color: #d4d4d4;
    padding: 4px;
    border-radius: 3px;
}
QComboBox QAbstractItemView {
    background-color: #3c3c3c;
    color: #d4d4d4;
    selection-background-color: #0078d4;
}
QScrollArea {
    border: none;
}
QSplitter::handle {
    background-color: #3c3c3c;
}
QLabel {
    color: #d4d4d4;
}
QGroupBox {
    border: 1px solid #444;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QProgressBar {
    border: 1px solid #555;
    border-radius: 3px;
    background: #333;
    text-align: center;
}
QProgressBar::chunk {
    background: #0078d4;
    border-radius: 2px;
}
"""


class AIWorker(QThread):
    """Background worker for AI detection."""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal()
    photo_done = pyqtSignal(int)  # photo_id

    def __init__(self, db, detector, species_suggester, photo_ids):
        super().__init__()
        self.db = db
        self.detector = detector
        self.species_suggester = species_suggester
        self.photo_ids = photo_ids
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        total = len(self.photo_ids)
        for i, pid in enumerate(self.photo_ids):
            if self._stop:
                break
            photo = self.db.get_photo_by_id(pid)
            if not photo:
                continue
            self.progress.emit(i + 1, total, f"Processing {Path(photo['file_path']).name}")

            # Run MegaDetector
            if self.detector:
                try:
                    detections = self.detector.detect(photo['file_path'])
                    self.db.set_boxes(pid, detections)
                except Exception as e:
                    logger.error(f"Detection error: {e}")

            # Run species classifier on boxes
            if self.species_suggester:
                try:
                    boxes = self.db.get_boxes(pid)
                    for box in boxes:
                        if box.get('label') == 'ai_animal':
                            species, conf = self.species_suggester.suggest(
                                photo['file_path'],
                                (box['x1'], box['y1'], box['x2'], box['y2'])
                            )
                            if species:
                                self.db.set_box_species(box['id'], species, conf)
                except Exception as e:
                    logger.error(f"Suggestion error: {e}")

            self.photo_done.emit(pid)

        self.finished.emit()


class OrganizerWindow(QMainWindow):
    """Main window for Trail Cam Organizer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trail Cam Organizer")
        self.setMinimumSize(1200, 700)

        # Core components
        self.db = Database()
        self.detector = None
        self.species_suggester = None
        self._load_ai_models()

        # State
        self.photos = []
        self.current_photo = None
        self.current_index = -1
        self._loading = False
        self._show_boxes = False
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._do_save)

        # UI setup
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._load_photos()

        # Apply dark theme
        self.setStyleSheet(DARK_STYLE)

    def _load_ai_models(self):
        """Load AI detection and classification models."""
        try:
            self.detector = MegaDetectorV5()
            logger.info("Loaded MegaDetector")
        except Exception as e:
            logger.warning(f"Could not load MegaDetector: {e}")

        try:
            self.species_suggester = SpeciesSuggester()
            logger.info(f"Loaded species model with {len(self.species_suggester.labels)} labels")
        except Exception as e:
            logger.warning(f"Could not load species model: {e}")

    def _setup_ui(self):
        """Build the main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Filter bar at top
        self._setup_filter_bar()
        main_layout.addWidget(self.filter_bar)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter

        # Left: Photo list
        self._setup_photo_list()
        splitter.addWidget(self.photo_list_container)

        # Center: Photo preview
        self._setup_preview()
        splitter.addWidget(self.preview_container)

        # Right: Info panel
        self._setup_info_panel()
        splitter.addWidget(self.info_panel)

        splitter.setSizes([220, 700, 320])
        main_layout.addWidget(splitter, stretch=1)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 4px 8px; background: #2d2d2d;")
        main_layout.addWidget(self.status_label)

    def _setup_filter_bar(self):
        """Create the filter/sort bar."""
        self.filter_bar = QWidget()
        self.filter_bar.setMaximumHeight(32)
        self.filter_bar.setStyleSheet("background-color: #2d2d2d; border-bottom: 1px solid #444;")

        layout = QHBoxLayout(self.filter_bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        label_style = "QLabel { font-size: 11px; color: #aaa; }"
        combo_style = "QComboBox { padding: 2px 4px; font-size: 11px; min-width: 80px; }"

        # Species filter
        layout.addWidget(self._label("Species:", label_style))
        self.species_filter = QComboBox()
        self.species_filter.setStyleSheet(combo_style)
        self.species_filter.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.species_filter)

        # Sex filter
        layout.addWidget(self._label("Sex:", label_style))
        self.sex_filter = QComboBox()
        self.sex_filter.setStyleSheet(combo_style)
        self.sex_filter.addItems(["All", "Buck", "Doe", "Unknown"])
        self.sex_filter.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.sex_filter)

        # Deer ID filter
        layout.addWidget(self._label("Deer:", label_style))
        self.deer_filter = QComboBox()
        self.deer_filter.setStyleSheet(combo_style)
        self.deer_filter.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.deer_filter)

        # Site filter
        layout.addWidget(self._label("Site:", label_style))
        self.site_filter = QComboBox()
        self.site_filter.setStyleSheet(combo_style)
        self.site_filter.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.site_filter)

        # Year filter
        layout.addWidget(self._label("Year:", label_style))
        self.year_filter = QComboBox()
        self.year_filter.setStyleSheet(combo_style)
        self.year_filter.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.year_filter)

        # Collection filter
        layout.addWidget(self._label("Collection:", label_style))
        self.collection_filter = QComboBox()
        self.collection_filter.setStyleSheet(combo_style)
        self.collection_filter.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.collection_filter)

        layout.addSpacing(16)

        # Sort
        layout.addWidget(self._label("Sort:", label_style))
        self.sort_combo = QComboBox()
        self.sort_combo.setStyleSheet(combo_style)
        self.sort_combo.addItem("Date (Newest)", "date_desc")
        self.sort_combo.addItem("Date (Oldest)", "date_asc")
        self.sort_combo.addItem("Location", "location")
        self.sort_combo.addItem("Species", "species")
        self.sort_combo.addItem("Deer ID", "deer_id")
        self.sort_combo.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.sort_combo)

        # Archive filter
        layout.addWidget(self._label("Show:", label_style))
        self.archive_filter = QComboBox()
        self.archive_filter.setStyleSheet(combo_style)
        self.archive_filter.addItem("Active", "active")
        self.archive_filter.addItem("Archived", "archived")
        self.archive_filter.addItem("All", "all")
        self.archive_filter.currentIndexChanged.connect(self._apply_filters)
        layout.addWidget(self.archive_filter)

        layout.addStretch()

        # Photo count
        self.photo_count_label = QLabel("0 photos")
        self.photo_count_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.photo_count_label)

    def _label(self, text, style):
        """Helper to create styled labels."""
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    def _setup_photo_list(self):
        """Create the photo list panel."""
        self.photo_list_container = QWidget()
        layout = QVBoxLayout(self.photo_list_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.photo_list = QListWidget()
        self.photo_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.photo_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.photo_list.setAlternatingRowColors(False)
        self.photo_list.setMinimumWidth(200)
        self.photo_list.setStyleSheet("QListWidget { font-size: 11px; }")
        self.photo_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.photo_list)

    def _setup_preview(self):
        """Create the photo preview area with zoom/pan support."""
        self.preview_container = QWidget()
        layout = QVBoxLayout(self.preview_container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Preview image using ImageGraphicsView for zoom/pan
        self.scene = QGraphicsScene()
        self.view = ImageGraphicsView()
        self.view.setScene(self.scene)
        self.view.setMinimumSize(400, 300)
        self.view.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.current_pixmap = None
        self.current_boxes = []  # Store detection boxes
        layout.addWidget(self.view, stretch=1)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(8)

        fit_btn = QPushButton("Fit")
        fit_btn.setMaximumWidth(40)
        fit_btn.clicked.connect(self.view.zoom_fit)
        zoom_layout.addWidget(fit_btn)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setSingleStep(5)
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.valueChanged.connect(lambda v: self.view.set_zoom_level(v / 100.0))
        self.view.zoom_changed.connect(self._sync_zoom_slider)
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(40)
        zoom_layout.addWidget(self.zoom_label)

        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)

        # AI suggestion banner
        self.suggestion_banner = QFrame()
        self.suggestion_banner.setStyleSheet("""
            QFrame { background-color: #2a4a6a; border-radius: 4px; padding: 4px; }
        """)
        banner_layout = QHBoxLayout(self.suggestion_banner)
        banner_layout.setContentsMargins(8, 4, 8, 4)

        self.suggestion_label = QLabel("AI suggests: ")
        self.suggestion_label.setStyleSheet("color: #8cf; font-weight: bold;")
        banner_layout.addWidget(self.suggestion_label)

        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setStyleSheet("background-color: #4a8; border: none;")
        self.accept_btn.clicked.connect(self._accept_suggestion)
        banner_layout.addWidget(self.accept_btn)

        self.reject_btn = QPushButton("Reject")
        self.reject_btn.setStyleSheet("background-color: #a44; border: none;")
        self.reject_btn.clicked.connect(self._reject_suggestion)
        banner_layout.addWidget(self.reject_btn)

        banner_layout.addStretch()
        self.suggestion_banner.hide()
        layout.addWidget(self.suggestion_banner)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.clicked.connect(self._go_prev)
        nav_layout.addWidget(self.prev_btn)

        nav_layout.addStretch()

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

    def _sync_zoom_slider(self, scale):
        """Sync zoom slider with current view scale."""
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(scale * 100))
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"{int(scale * 100)}%")

    def _setup_info_panel(self):
        """Create the photo info/labeling panel (similar to Trainer app)."""
        self.info_panel = QScrollArea()
        self.info_panel.setWidgetResizable(True)
        self.info_panel.setMinimumWidth(300)
        self.info_panel.setMaximumWidth(420)
        self.info_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Box tabs for multiple subjects (like Trainer)
        self.current_box_index = 0
        self.box_tab_bar = QTabWidget()
        self.box_tab_bar.setTabPosition(QTabWidget.TabPosition.North)
        self.box_tab_bar.setDocumentMode(True)
        self.box_tab_bar.currentChanged.connect(self._on_box_tab_switched)
        self.box_tab_bar.setMaximumHeight(30)
        self.box_tab_bar.setStyleSheet("QTabBar::tab { padding: 4px 12px; }")
        self.box_tab_bar.hide()  # Only show when multiple boxes
        layout.addWidget(self.box_tab_bar)

        # File info (compact)
        info_group = QGroupBox("Photo Info")
        info_layout = QFormLayout(info_group)
        info_layout.setSpacing(4)

        self.filename_label = QLabel("-")
        self.filename_label.setWordWrap(True)
        self.filename_label.setStyleSheet("font-size: 10px; color: #888;")
        info_layout.addRow("File:", self.filename_label)

        self.date_label = QLabel("-")
        info_layout.addRow("Date:", self.date_label)

        self.camera_label = QLabel("-")
        info_layout.addRow("Camera:", self.camera_label)

        layout.addWidget(info_group)

        # Labeling section
        label_group = QGroupBox("Labels")
        label_layout = QVBoxLayout(label_group)
        label_layout.setSpacing(8)

        # Species row
        species_row = QHBoxLayout()
        species_row.addWidget(QLabel("Species:"))
        self.species_combo = QComboBox()
        self.species_combo.setEditable(True)
        self.species_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.species_combo.setMinimumHeight(28)
        self.species_combo.setCompleter(None)
        self.species_combo.currentTextChanged.connect(self._schedule_save)
        species_row.addWidget(self.species_combo, 1)
        label_layout.addLayout(species_row)

        # Sex toggle buttons (like Trainer)
        sex_label_row = QHBoxLayout()
        sex_label_row.addWidget(QLabel("Sex:"))
        sex_label_row.addStretch()
        label_layout.addLayout(sex_label_row)

        self.sex_group = QButtonGroup()
        self.sex_group.setExclusive(False)  # Allow manual toggle control
        self.sex_buttons = {}
        sex_row = QHBoxLayout()
        sex_row.setSpacing(4)
        for label in ["Buck", "Doe", "Unknown"]:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(28)
            btn.setMinimumWidth(60)
            btn.setStyleSheet("QToolButton:checked { background:#446; color:white; }")
            self.sex_group.addButton(btn)
            self.sex_buttons[label] = btn
            sex_row.addWidget(btn)
            btn.clicked.connect(lambda checked, lbl=label: self._on_sex_clicked(lbl, checked))
        sex_row.addStretch()
        label_layout.addLayout(sex_row)

        # Age row
        age_row = QHBoxLayout()
        age_row.addWidget(QLabel("Age:"))
        self.age_combo = QComboBox()
        self.age_combo.addItems(["", "Fawn", "1.5", "2.5", "3.5", "4.5", "5.5+", "Mature"])
        self.age_combo.currentIndexChanged.connect(self._schedule_save)
        age_row.addWidget(self.age_combo, 1)
        label_layout.addLayout(age_row)

        # Deer ID row
        deer_row = QHBoxLayout()
        deer_row.addWidget(QLabel("Deer ID:"))
        self.deer_id_combo = QComboBox()
        self.deer_id_combo.setEditable(True)
        self.deer_id_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.deer_id_combo.setMinimumHeight(28)
        self.deer_id_combo.setMinimumWidth(120)
        self.deer_id_combo.view().setMinimumWidth(200)
        self.deer_id_combo.currentTextChanged.connect(self._schedule_save)
        deer_row.addWidget(self.deer_id_combo, 1)
        label_layout.addLayout(deer_row)

        # Quick buck ID buttons (like Trainer)
        self.quick_buck_btns = []
        quick_buck_label = QLabel("Recent Buck IDs:")
        quick_buck_label.setStyleSheet("font-size: 10px; color: #888;")
        label_layout.addWidget(quick_buck_label)

        for row_idx in range(2):  # 2 rows of 4 buttons
            row_layout = QHBoxLayout()
            row_layout.setSpacing(2)
            for col_idx in range(4):
                btn = QToolButton()
                btn.setCheckable(False)
                btn.clicked.connect(self._on_quick_buck_clicked)
                btn.setMinimumWidth(40)
                btn.setMaximumWidth(100)
                btn.setMinimumHeight(22)
                btn.setStyleSheet("font-size: 11px; padding: 2px;")
                btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
                btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                self.quick_buck_btns.append(btn)
                row_layout.addWidget(btn)
            row_layout.addStretch()
            label_layout.addLayout(row_layout)

        # Location row
        loc_row = QHBoxLayout()
        loc_row.addWidget(QLabel("Location:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)
        self.camera_combo.currentTextChanged.connect(self._schedule_save)
        loc_row.addWidget(self.camera_combo, 1)
        label_layout.addLayout(loc_row)

        layout.addWidget(label_group)

        # Quick Species Labels
        quick_group = QGroupBox("Quick Species")
        quick_layout = QVBoxLayout(quick_group)
        quick_layout.setSpacing(4)

        # Two rows of quick species buttons
        quick_row1 = QHBoxLayout()
        quick_row1.setSpacing(4)
        for species in QUICK_SPECIES[:3]:
            btn = QPushButton(species)
            btn.setMaximumHeight(26)
            btn.clicked.connect(lambda checked, s=species: self._quick_label(s))
            quick_row1.addWidget(btn)
        quick_layout.addLayout(quick_row1)

        quick_row2 = QHBoxLayout()
        quick_row2.setSpacing(4)
        for species in QUICK_SPECIES[3:]:
            btn = QPushButton(species)
            btn.setMaximumHeight(26)
            btn.clicked.connect(lambda checked, s=species: self._quick_label(s))
            quick_row2.addWidget(btn)
        quick_layout.addLayout(quick_row2)

        layout.addWidget(quick_group)

        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        action_layout.setSpacing(4)

        btn_row1 = QHBoxLayout()
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.clicked.connect(self._compare_photos)
        btn_row1.addWidget(self.compare_btn)

        self.archive_btn = QPushButton("Archive")
        self.archive_btn.clicked.connect(self._archive_photo)
        btn_row1.addWidget(self.archive_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_labels)
        btn_row1.addWidget(self.clear_btn)
        action_layout.addLayout(btn_row1)

        layout.addWidget(action_group)
        layout.addStretch()

        self.info_panel.setWidget(content)

    def _on_sex_clicked(self, label, checked):
        """Handle sex toggle button click."""
        # Uncheck other buttons (exclusive behavior)
        for lbl, btn in self.sex_buttons.items():
            if lbl != label:
                btn.setChecked(False)
        # Toggle the clicked button
        self.sex_buttons[label].setChecked(True)
        self._schedule_save()

    def _on_box_tab_switched(self, index):
        """Handle switching between box tabs."""
        self.current_box_index = index
        self._highlight_current_box()

    def _highlight_current_box(self):
        """Highlight the currently selected box in the view."""
        # This highlights the current box in the scene
        pass  # Box highlighting handled during display

    def _on_quick_buck_clicked(self):
        """Handle quick buck ID button click."""
        btn = self.sender()
        if btn and btn.text():
            self.deer_id_combo.setCurrentText(btn.text())

    def _update_recent_buck_buttons(self):
        """Update quick buck ID buttons with recent IDs."""
        recent_ids = self.db.list_recent_bucks(limit=8)
        for i, btn in enumerate(self.quick_buck_btns):
            if i < len(recent_ids):
                btn.setText(recent_ids[i])
                btn.setVisible(True)
            else:
                btn.setText("")
                btn.setVisible(False)

    def _setup_menu(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        import_action = QAction("Import Photos...", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self._import_photos)
        file_menu.addAction(import_action)

        add_folder_action = QAction("Add Photo Location...", self)
        add_folder_action.triggered.connect(self._add_photo_location)
        file_menu.addAction(add_folder_action)

        file_menu.addSeparator()

        # CuddeLink
        cudde_action = QAction("Download from CuddeLink...", self)
        cudde_action.triggered.connect(self._download_cuddelink)
        file_menu.addAction(cudde_action)

        cudde_setup_action = QAction("Setup CuddeLink Credentials...", self)
        cudde_setup_action.triggered.connect(self._setup_cuddelink)
        file_menu.addAction(cudde_setup_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        self.show_boxes_action = QAction("Show Boxes", self)
        self.show_boxes_action.setCheckable(True)
        self.show_boxes_action.setChecked(False)
        self.show_boxes_action.setShortcut("Ctrl+B")
        self.show_boxes_action.triggered.connect(self._toggle_boxes)
        view_menu.addAction(self.show_boxes_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        ai_action = QAction("Run AI on All Photos", self)
        ai_action.triggered.connect(self._run_ai_all)
        tools_menu.addAction(ai_action)

        ai_current_action = QAction("Run AI on Current Photo", self)
        ai_current_action.setShortcut("Ctrl+R")
        ai_current_action.triggered.connect(self._run_ai_current)
        tools_menu.addAction(ai_current_action)

        tools_menu.addSeparator()

        find_dupes_action = QAction("Find Duplicates...", self)
        find_dupes_action.triggered.connect(self._find_duplicates)
        tools_menu.addAction(find_dupes_action)

        tools_menu.addSeparator()

        # Bulk archive submenu
        archive_menu = tools_menu.addMenu("Bulk Archive")

        archive_empty_action = QAction("Archive All Empty", self)
        archive_empty_action.triggered.connect(lambda: self._bulk_archive("Empty"))
        archive_menu.addAction(archive_empty_action)

        archive_doe_action = QAction("Archive All Doe", self)
        archive_doe_action.triggered.connect(lambda: self._bulk_archive("Doe"))
        archive_menu.addAction(archive_doe_action)

        archive_turkey_action = QAction("Archive All Turkey", self)
        archive_turkey_action.triggered.connect(lambda: self._bulk_archive("Turkey"))
        archive_menu.addAction(archive_turkey_action)

        archive_raccoon_action = QAction("Archive All Raccoon", self)
        archive_raccoon_action.triggered.connect(lambda: self._bulk_archive("Raccoon"))
        archive_menu.addAction(archive_raccoon_action)

        archive_squirrel_action = QAction("Archive All Squirrel", self)
        archive_squirrel_action.triggered.connect(lambda: self._bulk_archive("Squirrel"))
        archive_menu.addAction(archive_squirrel_action)

        archive_menu.addSeparator()

        archive_filtered_action = QAction("Archive Current Filtered View", self)
        archive_filtered_action.triggered.connect(self._bulk_archive_filtered)
        archive_menu.addAction(archive_filtered_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        QShortcut(QKeySequence("Left"), self, activated=self._go_prev)
        QShortcut(QKeySequence("Right"), self, activated=self._go_next)
        QShortcut(QKeySequence("Space"), self, activated=self._open_preview)
        QShortcut(QKeySequence("A"), self, activated=self._accept_suggestion)

    def _load_photos(self):
        """Load photos from database and populate filters."""
        self.photos = self.db.get_all_photos()
        self._populate_filter_options()
        self._apply_filters()

    def _populate_filter_options(self):
        """Populate filter dropdowns with available values."""
        # Block signals during population
        for combo in [self.species_filter, self.deer_filter, self.site_filter,
                      self.year_filter, self.collection_filter]:
            combo.blockSignals(True)

        # Species - get from database like Trainer does
        self.species_filter.clear()
        self.species_filter.addItem("All Species", "")

        # Build species list from defaults + custom + all tags
        species_set = set(s for s in SPECIES_OPTIONS if s)
        try:
            # Add custom species
            for s in self.db.list_custom_species():
                if s and len(s) >= 2 and s.lower() not in SEX_TAGS:
                    species_set.add(s)
            # Add all tags that look like species
            all_tags = self.db.get_all_distinct_tags()
            skip_tags = SEX_TAGS | {"buck", "doe", "favorite"}
            for t in all_tags:
                if t and len(t) >= 2 and t.lower() not in skip_tags:
                    species_set.add(t)
        except Exception as e:
            logger.error(f"Error getting species: {e}")

        species_list = sorted(species_set)
        for s in species_list:
            self.species_filter.addItem(s, s)

        # Also populate species combo in info panel
        self.species_combo.blockSignals(True)
        self.species_combo.clear()
        self.species_combo.addItems([""] + species_list)
        self.species_combo.blockSignals(False)

        # Deer IDs - query from deer_metadata table
        self.deer_filter.clear()
        self.deer_filter.addItem("All Deer", "")

        deer_ids = set()
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT DISTINCT deer_id FROM deer_metadata WHERE deer_id IS NOT NULL AND deer_id != ''")
            deer_ids.update(row[0] for row in cursor.fetchall())
            cursor.execute("SELECT DISTINCT deer_id FROM deer_additional WHERE deer_id IS NOT NULL AND deer_id != ''")
            deer_ids.update(row[0] for row in cursor.fetchall())
        except Exception as e:
            logger.error(f"Error getting deer IDs: {e}")

        deer_ids_list = sorted(deer_ids)
        for d in deer_ids_list:
            self.deer_filter.addItem(d, d)

        self.deer_id_combo.blockSignals(True)
        self.deer_id_combo.clear()
        self.deer_id_combo.addItems([""] + deer_ids_list)
        self.deer_id_combo.blockSignals(False)

        # Sites
        self.site_filter.clear()
        self.site_filter.addItem("All Sites", "")
        sites = sorted(set(
            p.get('camera_location') for p in self.photos
            if p.get('camera_location')
        ))
        for s in sites:
            self.site_filter.addItem(s, s)

        self.camera_combo.clear()
        self.camera_combo.addItems([""] + sites)

        # Years (antler year: May-April)
        self.year_filter.clear()
        self.year_filter.addItem("All Years", "")
        years = sorted(set(
            p.get('season_year') for p in self.photos
            if p.get('season_year')
        ), reverse=True)
        for y in years:
            self.year_filter.addItem(f"{y}-{y+1}", y)

        # Collections
        self.collection_filter.clear()
        self.collection_filter.addItem("All Collections", "")
        collections = sorted(set(
            p.get('collection') for p in self.photos
            if p.get('collection')
        ))
        for c in collections:
            self.collection_filter.addItem(c, c)

        # Restore signals
        for combo in [self.species_filter, self.deer_filter, self.site_filter,
                      self.year_filter, self.collection_filter]:
            combo.blockSignals(False)

    def _apply_filters(self):
        """Filter and sort photos, update list."""
        filtered = self.photos.copy()

        # Apply filters
        species = self.species_filter.currentData()
        if species:
            filtered = [p for p in filtered if species in (p.get('tags') or '')]

        sex = self.sex_filter.currentText()
        if sex and sex != "All":
            sex_val = sex.lower() if sex != "Unknown" else ""
            filtered = [p for p in filtered if p.get('deer_sex') == sex_val]

        deer_id = self.deer_filter.currentData()
        if deer_id:
            filtered = [p for p in filtered if p.get('deer_id') == deer_id]

        site = self.site_filter.currentData()
        if site:
            filtered = [p for p in filtered if p.get('camera_location') == site]

        year = self.year_filter.currentData()
        if year:
            filtered = [p for p in filtered if p.get('season_year') == year]

        collection = self.collection_filter.currentData()
        if collection:
            filtered = [p for p in filtered if p.get('collection') == collection]

        archive = self.archive_filter.currentData()
        if archive == "active":
            filtered = [p for p in filtered if not p.get('archived')]
        elif archive == "archived":
            filtered = [p for p in filtered if p.get('archived')]

        # Apply sort
        sort_key = self.sort_combo.currentData()
        if sort_key == "date_desc":
            filtered.sort(key=lambda p: p.get('date_taken') or '', reverse=True)
        elif sort_key == "date_asc":
            filtered.sort(key=lambda p: p.get('date_taken') or '')
        elif sort_key == "location":
            filtered.sort(key=lambda p: p.get('camera_location') or '')
        elif sort_key == "species":
            filtered.sort(key=lambda p: p.get('tags') or '')
        elif sort_key == "deer_id":
            filtered.sort(key=lambda p: p.get('deer_id') or '')

        # Update list
        self.filtered_photos = filtered
        self._populate_photo_list()
        self.photo_count_label.setText(f"{len(filtered)} photos")

    def _populate_photo_list(self):
        """Populate the photo list widget."""
        self.photo_list.clear()

        for photo in self.filtered_photos:
            display, is_suggestion = self._build_photo_label(photo)

            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, photo['id'])
            item.setToolTip(Path(photo.get('file_path', '')).name)

            # Red text for AI suggestions (unverified)
            if is_suggestion:
                item.setForeground(QColor(200, 50, 50))

            self.photo_list.addItem(item)

        # Select first item
        if self.photo_list.count() > 0:
            self.photo_list.setCurrentRow(0)

    def _build_photo_label(self, photo):
        """Build list label: date + most specific identifier.
        Returns (display_string, is_suggestion) tuple."""
        pid = photo.get('id')

        # Format date as m/d/yyyy h:mm AM/PM
        date_taken = photo.get('date_taken')
        if date_taken:
            try:
                if 'T' in date_taken:
                    dt = datetime.fromisoformat(date_taken.replace('Z', '+00:00'))
                else:
                    dt = datetime.strptime(date_taken, '%Y-%m-%d %H:%M:%S')
                label = dt.strftime('%-m/%-d/%Y %-I:%M %p')
            except Exception:
                label = date_taken
        else:
            label = Path(photo.get('file_path', '')).name

        # Get the most specific identifier: deer ID > Buck/Doe > Species > AI suggestion
        detail = ''
        is_suggestion = False

        try:
            # First priority: Deer ID
            deer_meta = self.db.get_deer_metadata(pid)
            deer_id = (deer_meta.get('deer_id') or '').strip() if deer_meta else ''
            if deer_id:
                detail = deer_id
            else:
                # Second priority: Buck or Doe tag
                tags = set(self.db.get_tags(pid))
                if 'Buck' in tags:
                    detail = 'Buck'
                elif 'Doe' in tags:
                    detail = 'Doe'
                else:
                    # Third priority: Species tag
                    species_tags = [t for t in tags if t not in ['Buck', 'Doe']]
                    if species_tags:
                        detail = species_tags[0]
                    else:
                        # Fourth priority: AI suggestion (not verified)
                        suggested = photo.get('suggested_tag', '')
                        if suggested:
                            detail = suggested
                            is_suggestion = True
        except Exception:
            pass

        if detail:
            display = f'{label} - {detail}'
        else:
            display = label

        return display, is_suggestion

    def _on_selection_changed(self):
        """Handle photo selection change."""
        items = self.photo_list.selectedItems()
        if not items:
            return

        item = items[0]
        photo_id = item.data(Qt.ItemDataRole.UserRole)
        self._load_photo(photo_id)

    def _load_photo(self, photo_id):
        """Load and display a photo."""
        self._loading = True

        photo = self.db.get_photo_by_id(photo_id)
        if not photo:
            self._loading = False
            return

        self.current_photo = photo
        self.current_index = self.photo_list.currentRow()

        # Update preview using QGraphicsView
        self.scene.clear()
        if photo.get('file_path') and os.path.exists(photo['file_path']):
            pixmap = QPixmap(photo['file_path'])
            if not pixmap.isNull():
                self.current_pixmap = pixmap
                self.scene.addPixmap(pixmap)
                self.scene.setSceneRect(QRectF(pixmap.rect()))

                # Load and draw subject boxes
                self.current_boxes = self.db.get_boxes(photo['id'])
                self._draw_boxes_on_scene()

                # Update box tabs
                self._update_box_tabs()

                # Fit to view
                self.view.zoom_fit()
        else:
            # Show placeholder text
            text_item = self.scene.addText("Photo not found")
            text_item.setDefaultTextColor(QColor("#888"))

        # Update info
        self.filename_label.setText(Path(photo.get('file_path', '')).name)

        date_str = ""
        if photo.get('date_taken'):
            try:
                dt = datetime.fromisoformat(photo['date_taken'])
                date_str = dt.strftime("%B %d, %Y %I:%M %p")
            except:
                date_str = photo['date_taken']
        self.date_label.setText(date_str)

        self.camera_label.setText(photo.get('camera_location') or '-')

        # Get tags and deer metadata
        tags_list = self.db.get_tags(photo['id'])
        deer_meta = self.db.get_deer_metadata(photo['id'])

        # Update labels - species is first non-Buck/Doe tag
        species = ''
        for tag in tags_list:
            if tag not in ['Buck', 'Doe']:
                species = tag
                break
        self.species_combo.setCurrentText(species)

        # Sex from tags - update toggle buttons
        for btn in self.sex_buttons.values():
            btn.setChecked(False)
        if 'Buck' in tags_list:
            self.sex_buttons['Buck'].setChecked(True)
        elif 'Doe' in tags_list:
            self.sex_buttons['Doe'].setChecked(True)

        # Age from deer_metadata
        age = deer_meta.get('age_class') or ''
        age_idx = self.age_combo.findText(age)
        self.age_combo.setCurrentIndex(max(0, age_idx))

        # Deer ID from deer_metadata
        self.deer_id_combo.setCurrentText(deer_meta.get('deer_id') or '')
        self.camera_combo.setCurrentText(photo.get('camera_location') or '')

        # Update quick buck buttons
        self._update_recent_buck_buttons()

        # Show AI suggestion if present
        suggested = photo.get('suggested_tag')
        if suggested:
            conf = photo.get('suggested_confidence', 0) * 100
            self.suggestion_label.setText(f"AI suggests: {suggested} ({conf:.0f}%)")
            self.suggestion_banner.show()
        else:
            self.suggestion_banner.hide()

        # Update archive button
        if photo.get('archived'):
            self.archive_btn.setText("Unarchive")
        else:
            self.archive_btn.setText("Archive")

        self._loading = False
        self.status_label.setText(f"Photo {self.current_index + 1} of {len(self.filtered_photos)}")

    def _draw_boxes_on_scene(self):
        """Draw detection boxes on the scene."""
        if not self.current_pixmap or not self._show_boxes:
            return

        img_w = self.current_pixmap.width()
        img_h = self.current_pixmap.height()

        for i, box in enumerate(self.current_boxes):
            if box.get('label') not in ['subject', 'ai_animal']:
                continue

            # Convert normalized coords to pixels
            x1 = box['x1'] * img_w
            y1 = box['y1'] * img_h
            x2 = box['x2'] * img_w
            y2 = box['y2'] * img_h

            rect = QRectF(x1, y1, x2 - x1, y2 - y1)

            # Color based on whether this is the selected box
            if i == self.current_box_index:
                pen = QPen(QColor("#00ff00"))  # Green for selected
                pen.setWidth(4)
            else:
                pen = QPen(QColor("#ffaa00"))  # Orange for others
                pen.setWidth(3)

            self.scene.addRect(rect, pen)

            # Add label text
            species = box.get('species') or box.get('suggested_species') or ''
            if species:
                text = self.scene.addText(f"{i+1}: {species}")
                text.setDefaultTextColor(QColor("#fff"))
                text.setPos(x1, y1 - 20)

    def _update_box_tabs(self):
        """Update the box tab bar based on detected subjects."""
        self.box_tab_bar.blockSignals(True)
        self.box_tab_bar.clear()

        subject_boxes = [b for b in self.current_boxes if b.get('label') in ['subject', 'ai_animal']]

        if len(subject_boxes) > 1:
            for i, box in enumerate(subject_boxes):
                species = box.get('species') or box.get('suggested_species') or '?'
                self.box_tab_bar.addTab(QWidget(), f"Subject {i+1}: {species}")
            self.box_tab_bar.show()
            self.box_tab_bar.setCurrentIndex(min(self.current_box_index, len(subject_boxes) - 1))
        else:
            self.box_tab_bar.hide()

        self.box_tab_bar.blockSignals(False)

    def _schedule_save(self):
        """Schedule a save after a brief delay (debounce)."""
        if self._loading:
            return
        self._save_timer.start(500)

    def _do_save(self):
        """Save current photo labels."""
        if not self.current_photo or self._loading:
            return

        pid = self.current_photo['id']

        # Get values
        species = self.species_combo.currentText().strip()
        # Get sex from toggle buttons
        sex = ""
        for label, btn in self.sex_buttons.items():
            if btn.isChecked():
                sex = label
                break
        age = self.age_combo.currentText()
        deer_id = self.deer_id_combo.currentText().strip()
        location = self.camera_combo.currentText().strip()

        # Save species as tag
        if species:
            self.db.add_tag(pid, species)

        # Save sex as tag (Buck/Doe)
        if sex and sex not in ["", "Unknown"]:
            self.db.add_tag(pid, sex)

        # Save deer metadata (deer_id and age_class only)
        self.db.set_deer_metadata(pid, deer_id=deer_id, age_class=age if age else None)

        # Save location
        if location:
            self.db.update_photo_attributes(pid, camera_location=location)

        # Clear suggestion if user made a choice
        if species and self.current_photo.get('suggested_tag'):
            self.db.set_suggested_tag(pid, None, None)
            self.suggestion_banner.hide()

        self.status_label.setText("Saved")

    def _accept_suggestion(self):
        """Accept the AI suggestion."""
        if not self.current_photo:
            return

        suggested = self.current_photo.get('suggested_tag')
        if suggested:
            self.species_combo.setCurrentText(suggested)
            self._do_save()
            self._go_next()

    def _reject_suggestion(self):
        """Reject/clear the AI suggestion."""
        if not self.current_photo:
            return

        self.db.set_suggested_tag(self.current_photo['id'], None, None)
        self.suggestion_banner.hide()

    def _go_prev(self):
        """Go to previous photo."""
        if self.current_index > 0:
            self.photo_list.setCurrentRow(self.current_index - 1)

    def _go_next(self):
        """Go to next photo."""
        if self.current_index < self.photo_list.count() - 1:
            self.photo_list.setCurrentRow(self.current_index + 1)

    def _on_preview_click(self, event):
        """Handle click on preview - open full size."""
        self._open_preview()

    def _open_preview(self):
        """Open full-size preview window."""
        if not self.current_photo:
            return

        try:
            preview = PreviewWindow(
                self.current_photo['file_path'],
                self.db,
                self.current_photo['id'],
                parent=self
            )
            preview.show()
        except Exception as e:
            logger.error(f"Error opening preview: {e}")

    def _compare_photos(self):
        """Open compare window with selected photos."""
        items = self.photo_list.selectedItems()
        if len(items) < 2:
            QMessageBox.information(self, "Compare", "Select 2-4 photos to compare")
            return

        photo_ids = [item.data(Qt.ItemDataRole.UserRole) for item in items[:4]]
        photos = [self.db.get_photo_by_id(pid) for pid in photo_ids]
        paths = [p['file_path'] for p in photos if p]

        if len(paths) >= 2:
            compare = CompareWindow(paths, parent=self)
            compare.show()

    def _archive_photo(self):
        """Toggle archive status of current photo."""
        if not self.current_photo:
            return

        pid = self.current_photo['id']
        currently_archived = self.current_photo.get('archived', False)

        if currently_archived:
            self.db.unarchive_photo(pid)
        else:
            self.db.archive_photo(pid)

        # Refresh
        self._load_photos()

    def _clear_labels(self):
        """Clear all labels from current photo."""
        if not self.current_photo:
            return

        reply = QMessageBox.question(
            self, "Clear Labels",
            "Clear all labels from this photo?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            pid = self.current_photo['id']
            self.db.update_photo_tags(pid, [])
            self.db.set_deer_metadata(pid, deer_id=None, age_class=None)
            self._load_photo(pid)

    def _import_photos(self):
        """Import photos from a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if not folder:
            return

        # Find images
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        files = []
        for root, dirs, filenames in os.walk(folder):
            for f in filenames:
                if Path(f).suffix.lower() in extensions:
                    files.append(os.path.join(root, f))

        if not files:
            QMessageBox.information(self, "Import", "No photos found in folder")
            return

        # Import
        imported = 0
        for f in files:
            try:
                new_path, original_name, date_taken, camera_model = image_processor.import_photo(f)
                self.db.add_photo(new_path, original_name, date_taken, camera_model)
                imported += 1
            except Exception as e:
                logger.error(f"Import error for {f}: {e}")

        QMessageBox.information(self, "Import", f"Imported {imported} photos")
        self._load_photos()

    def _add_photo_location(self):
        """Scan a folder for photos and add them to the database (without copying)."""
        folder = QFileDialog.getExistingDirectory(self, "Select Photo Folder to Scan")
        if not folder:
            return

        # Find images
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        files = []
        for root, dirs, filenames in os.walk(folder):
            for f in filenames:
                if Path(f).suffix.lower() in extensions:
                    files.append(os.path.join(root, f))

        if not files:
            QMessageBox.information(self, "Scan", "No photos found in folder")
            return

        # Add photos to database (in-place, no copying)
        added = 0
        for f in files:
            try:
                # Check if already in database
                if self.db.get_photo_by_path(f):
                    continue
                # Extract EXIF
                date_taken, camera_model = image_processor.extract_exif_data(f)
                self.db.add_photo(f, Path(f).name, date_taken or "", camera_model or "")
                added += 1
            except Exception as e:
                logger.error(f"Error adding {f}: {e}")

        QMessageBox.information(self, "Scan Complete", f"Added {added} new photos from:\n{folder}")
        self._load_photos()

    def _run_ai_all(self):
        """Run AI detection on all unprocessed photos."""
        # Get photos without AI boxes
        unprocessed = [p['id'] for p in self.photos if not self.db.get_boxes(p['id'])]

        if not unprocessed:
            QMessageBox.information(self, "AI Detection", "All photos already processed")
            return

        reply = QMessageBox.question(
            self, "AI Detection",
            f"Run AI detection on {len(unprocessed)} photos?\nThis may take a while.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Start worker
        self.ai_worker = AIWorker(self.db, self.detector, self.species_suggester, unprocessed)
        self.ai_worker.progress.connect(self._on_ai_progress)
        self.ai_worker.finished.connect(self._on_ai_finished)
        self.ai_worker.start()

    def _run_ai_current(self):
        """Run AI on current photo."""
        if not self.current_photo:
            return

        pid = self.current_photo['id']
        path = self.current_photo['file_path']

        self.status_label.setText("Running AI...")
        QApplication.processEvents()

        try:
            if self.detector:
                detections = self.detector.detect(path)
                self.db.set_boxes(pid, detections)

            if self.species_suggester:
                boxes = self.db.get_boxes(pid)
                for box in boxes:
                    if box.get('label') == 'ai_animal':
                        species, conf = self.species_suggester.suggest(
                            path, (box['x1'], box['y1'], box['x2'], box['y2'])
                        )
                        if species:
                            self.db.set_box_species(box['id'], species, conf)
                            self.db.set_suggested_tag(pid, species, conf)

            self._load_photo(pid)
            self.status_label.setText("AI complete")
        except Exception as e:
            logger.error(f"AI error: {e}")
            self.status_label.setText(f"AI error: {e}")

    def _on_ai_progress(self, current, total, msg):
        """Handle AI progress updates."""
        self.status_label.setText(f"AI: {current}/{total} - {msg}")

    def _on_ai_finished(self):
        """Handle AI completion."""
        self.status_label.setText("AI detection complete")
        self._load_photos()

    def _find_duplicates(self):
        """Open duplicate finder dialog."""
        try:
            from duplicate_dialog import DuplicateDialog
            dialog = DuplicateDialog(self.db, parent=self)
            dialog.exec()
        except Exception as e:
            logger.error(f"Error opening duplicate dialog: {e}")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Trail Cam Organizer",
            "Trail Cam Organizer\n\n"
            "Organize and label your trail camera photos with AI assistance.\n\n"
            "Version 2.0"
        )

    def _toggle_boxes(self):
        """Toggle display of subject boxes on the preview."""
        self._show_boxes = self.show_boxes_action.isChecked()
        if self.current_photo:
            self._load_photo(self.current_photo['id'])

    def _quick_label(self, species):
        """Apply a quick species label to the current photo."""
        if not self.current_photo:
            return

        pid = self.current_photo['id']
        self.db.add_tag(pid, species)

        # Clear AI suggestion if we're labeling
        if self.current_photo.get('suggested_tag'):
            self.db.set_suggested_tag(pid, None, None)

        # Update UI and go to next
        self.species_combo.setCurrentText(species)
        self.suggestion_banner.hide()
        self.status_label.setText(f"Labeled: {species}")

        # Advance to next photo
        self._go_next()

    def _quick_label_sex(self, sex):
        """Apply Buck or Doe label (requires Deer species)."""
        if not self.current_photo:
            return

        pid = self.current_photo['id']

        # Add Deer tag if not already present
        tags = self.db.get_tags(pid)
        if 'Deer' not in tags:
            self.db.add_tag(pid, 'Deer')

        # Add Buck or Doe tag
        self.db.add_tag(pid, sex)

        # Clear AI suggestion
        if self.current_photo.get('suggested_tag'):
            self.db.set_suggested_tag(pid, None, None)

        # Update UI
        self.species_combo.setCurrentText('Deer')
        # Update sex toggle buttons
        for lbl, btn in self.sex_buttons.items():
            btn.setChecked(lbl == sex)
        self.suggestion_banner.hide()
        self.status_label.setText(f"Labeled: Deer - {sex}")

        # Advance to next photo
        self._go_next()

    def _bulk_archive(self, species):
        """Archive all photos with the specified species tag."""
        # Find all photos with this species
        matching = []
        for photo in self.photos:
            if photo.get('archived'):
                continue
            tags = self.db.get_tags(photo['id'])
            if species in tags:
                matching.append(photo['id'])

        if not matching:
            QMessageBox.information(self, "Bulk Archive", f"No active photos found with tag: {species}")
            return

        reply = QMessageBox.question(
            self, "Bulk Archive",
            f"Archive {len(matching)} photos tagged as '{species}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.db.archive_photos(matching)
            QMessageBox.information(self, "Bulk Archive", f"Archived {len(matching)} photos")
            self._load_photos()

    def _bulk_archive_filtered(self):
        """Archive all photos in the current filtered view."""
        if not self.filtered_photos:
            QMessageBox.information(self, "Bulk Archive", "No photos in current filter")
            return

        # Only archive non-archived photos
        to_archive = [p['id'] for p in self.filtered_photos if not p.get('archived')]

        if not to_archive:
            QMessageBox.information(self, "Bulk Archive", "No active photos in current filter")
            return

        reply = QMessageBox.question(
            self, "Bulk Archive",
            f"Archive {len(to_archive)} photos from current filtered view?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.db.archive_photos(to_archive)
            QMessageBox.information(self, "Bulk Archive", f"Archived {len(to_archive)} photos")
            self._load_photos()

    def _setup_cuddelink(self):
        """Show dialog to set up CuddeLink credentials."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("TrailCam", "Organizer")
        current_email = settings.value("cuddelink_email", "")

        dialog = QDialog(self)
        dialog.setWindowTitle("CuddeLink Credentials")
        dialog.setMinimumWidth(350)

        layout = QVBoxLayout(dialog)

        info_label = QLabel("Enter your CuddeLink account credentials.\nThese will be saved for future downloads.")
        layout.addWidget(info_label)

        form = QFormLayout()
        email_edit = QLineEdit(current_email)
        email_edit.setPlaceholderText("email@example.com")
        form.addRow("Email:", email_edit)

        password_edit = QLineEdit()
        password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        password_edit.setPlaceholderText("Your CuddeLink password")
        form.addRow("Password:", password_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            email = email_edit.text().strip()
            password = password_edit.text()

            if not email or not password:
                QMessageBox.warning(self, "CuddeLink", "Please enter both email and password.")
                return

            settings.setValue("cuddelink_email", email)
            settings.setValue("cuddelink_password", password)
            QMessageBox.information(self, "CuddeLink", "Credentials saved successfully!")

    def _download_cuddelink(self):
        """Download photos from CuddeLink."""
        from PyQt6.QtCore import QSettings, QDate
        settings = QSettings("TrailCam", "Organizer")

        email = settings.value("cuddelink_email", "")
        password = settings.value("cuddelink_password", "")

        if not email or not password:
            reply = QMessageBox.question(
                self, "CuddeLink",
                "No CuddeLink credentials found.\n\nWould you like to set them up now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._setup_cuddelink()
                email = settings.value("cuddelink_email", "")
                password = settings.value("cuddelink_password", "")
            if not email or not password:
                return

        # Date range dialog
        date_dialog = QDialog(self)
        date_dialog.setWindowTitle("CuddeLink Date Range")
        date_layout = QVBoxLayout(date_dialog)

        date_layout.addWidget(QLabel("Select date range to download:"))

        from PyQt6.QtWidgets import QDateEdit
        form = QFormLayout()

        # Default: last 7 days
        end_date = QDate.currentDate()
        start_date = end_date.addDays(-7)

        start_edit = QDateEdit(start_date)
        start_edit.setCalendarPopup(True)
        form.addRow("From:", start_edit)

        end_edit = QDateEdit(end_date)
        end_edit.setCalendarPopup(True)
        form.addRow("To:", end_edit)

        date_layout.addLayout(form)

        date_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        date_buttons.accepted.connect(date_dialog.accept)
        date_buttons.rejected.connect(date_dialog.reject)
        date_layout.addWidget(date_buttons)

        if date_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        start = start_edit.date().toString("yyyy-MM-dd")
        end = end_edit.date().toString("yyyy-MM-dd")

        # Progress dialog
        self.status_label.setText("Connecting to CuddeLink...")
        QApplication.processEvents()

        try:
            # Get library path for download destination
            dest = image_processor.get_library_path()

            # Download photos
            result = download_new_photos(
                email=email,
                password=password,
                dest_dir=str(dest),
                start_date=start,
                end_date=end
            )

            if result.get('error'):
                error = result['error']
                # Check if it's a credential error
                if "credentials" in error.lower() or "password" in error.lower() or "invalid" in error.lower() or "login" in error.lower():
                    reply = QMessageBox.question(
                        self, "CuddeLink Login Failed",
                        f"Login failed: {error}\n\nWould you like to update your credentials?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self._setup_cuddelink()
                else:
                    QMessageBox.warning(self, "CuddeLink", f"Download failed: {error}")
                self.status_label.setText("CuddeLink download failed")
                return

            downloaded = result.get('downloaded', 0)
            skipped = result.get('skipped', 0)

            if downloaded == 0 and skipped == 0:
                QMessageBox.information(self, "CuddeLink", "No new photos found to download.")
            else:
                # Import downloaded photos to database
                new_files = result.get('files', [])
                imported = 0
                for f in new_files:
                    try:
                        if self.db.get_photo_by_path(f):
                            continue
                        date_taken, camera_model = image_processor.extract_exif_data(f)
                        self.db.add_photo(f, Path(f).name, date_taken or "", camera_model or "")
                        imported += 1
                    except Exception as e:
                        logger.error(f"Import error: {e}")

                msg = f"Downloaded: {downloaded} photos\nSkipped (already have): {skipped}\nImported to library: {imported}"
                QMessageBox.information(self, "CuddeLink", msg)
                self._load_photos()

            self.status_label.setText("CuddeLink download complete")

        except Exception as e:
            logger.error(f"CuddeLink error: {e}")
            error = str(e)
            # Check if it's a credential error
            if "credentials" in error.lower() or "password" in error.lower() or "invalid" in error.lower() or "login" in error.lower():
                reply = QMessageBox.question(
                    self, "CuddeLink Login Failed",
                    f"Login failed: {error}\n\nWould you like to update your credentials?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._setup_cuddelink()
            else:
                QMessageBox.warning(self, "CuddeLink", f"Error: {error}")
            self.status_label.setText("CuddeLink error")

    def resizeEvent(self, event):
        """Handle resize - update preview scaling."""
        super().resizeEvent(event)
        if self.current_photo:
            self._load_photo(self.current_photo['id'])

    def closeEvent(self, event):
        """Handle window close."""
        self._save_timer.stop()
        if hasattr(self, 'ai_worker') and self.ai_worker.isRunning():
            self.ai_worker.stop()
            self.ai_worker.wait()
        event.accept()


def main():
    """Entry point for Trail Cam Organizer."""
    app = QApplication(sys.argv)
    app.setApplicationName("Trail Cam Organizer")

    window = OrganizerWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
