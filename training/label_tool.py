"""
Labeler UI (now primary app).

- Step through photos, set species/sex/deer_id/age/points/camera/location/notes.
- Bulk assign buck IDs, compare selected, export CSVs.
- File/Tools/Settings menu items for imports, AI suggestions, duplicates (basic).
"""
import os
import sys
import shutil
import tempfile
import hashlib
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from version import __version__
import updater
import user_config
from sync_manager import SyncManager
from r2_upload_queue import R2UploadManager
from speciesnet_wrapper import SpeciesNetWrapper
from speciesnet_download_dialog import SpeciesNetDownloadDialog
from ai_detection import MEGADETECTOR_AVAILABLE
from dialogs import (
    AIOptionsDialog,
    TightwadComparisonDialog,
    UserSetupDialog,
    SupabaseCredentialsDialog,
    CuddeLinkCredentialsDialog,
)

logger = logging.getLogger(__name__)

from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal, QCoreApplication, QSettings, QPoint, QRect, QSize, QThread, QDate, QEventLoop
from PyQt6.QtGui import QPixmap, QIcon, QAction, QPen, QShortcut, QKeySequence, QColor, QBrush, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QTreeWidget,
    QTreeWidgetItem,
    QScrollArea,
    QMenuBar,
    QMenu,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QSplitter,
    QFrame,
    QSizePolicy,
    QButtonGroup,
    QToolButton,
    QSlider,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsRectItem,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QTextBrowser,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QProgressDialog,
    QProgressBar,
    QInputDialog,
    QAbstractItemView,
    QTableWidget,
    QTableWidgetItem,
    QLayout,
    QWidgetItem,
    QDateEdit,
    QTabWidget,
    QGridLayout,
    QGroupBox,
    QRadioButton,
)
import PyQt6  # used for plugin path detection
import sysconfig
from PyQt6.QtGui import QStandardItemModel, QStandardItem

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database import TrailCamDatabase  # noqa: E402
from preview_window import ImageGraphicsView  # reuse zoom/pan behavior


class SupabaseAuthDialog(QDialog):
    """Simple Supabase auth dialog (sign in / sign up)."""

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self._signup_mode = False
        self.result_mode = None

        self.setWindowTitle("Trail Camera Organizer — Sign In")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        welcome = QLabel("Sign in to sync photos and labels across devices.")
        welcome.setWordWrap(True)
        welcome.setStyleSheet("margin-bottom: 8px;")
        layout.addWidget(welcome)

        form = QFormLayout()
        self.email_edit = QLineEdit()
        self.email_edit.setPlaceholderText("Email")
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Password")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.display_name_label = QLabel("Display Name")
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setPlaceholderText("Your name")

        form.addRow("Email", self.email_edit)
        form.addRow("Password", self.password_edit)
        form.addRow(self.display_name_label, self.display_name_edit)
        layout.addLayout(form)

        self.display_name_label.hide()
        self.display_name_edit.hide()

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        layout.addWidget(self.error_label)

        btn_row = QHBoxLayout()
        self.sign_in_btn = QPushButton("Sign In")
        self.sign_up_btn = QPushButton("Sign Up")
        self.continue_btn = QPushButton("Browse Only")
        btn_row.addWidget(self.sign_in_btn)
        btn_row.addWidget(self.sign_up_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.continue_btn)
        layout.addLayout(btn_row)

        self.forgot_btn = QPushButton("Forgot Password?")
        self.forgot_btn.setFlat(True)
        self.forgot_btn.setStyleSheet("color: #4a9eff; text-decoration: underline; border: none;")
        layout.addWidget(self.forgot_btn)

        self.sign_in_btn.clicked.connect(self._on_sign_in)
        self.sign_up_btn.clicked.connect(self._on_sign_up)
        self.continue_btn.clicked.connect(self._on_continue)
        self.forgot_btn.clicked.connect(self._on_forgot_password)

    def _set_signup_mode(self, enabled: bool):
        self._signup_mode = enabled
        if enabled:
            self.display_name_label.show()
            self.display_name_edit.show()
        else:
            self.display_name_label.hide()
            self.display_name_edit.hide()

    def _on_sign_in(self):
        email = self.email_edit.text().strip()
        password = self.password_edit.text().strip()
        if not email or not password:
            self.error_label.setText("Email and password are required.")
            return
        result = self.client.sign_in(email, password)
        if result.get("ok") and self.client.is_authenticated:
            self.result_mode = "signed_in"
            self.accept()
        else:
            msg = result.get("error", "Check credentials.")
            self.error_label.setText(f"Login failed: {msg}")

    def _on_sign_up(self):
        if not self._signup_mode:
            self._set_signup_mode(True)
            self.error_label.setText("")
            return
        email = self.email_edit.text().strip()
        password = self.password_edit.text().strip()
        display_name = self.display_name_edit.text().strip()
        if not email or not password or not display_name:
            self.error_label.setText("Email, password, and display name are required.")
            return
        result = self.client.sign_up(email, password, display_name)
        if result.get("ok") and self.client.is_authenticated:
            self.result_mode = "signed_in"
            self.accept()
        elif result.get("ok"):
            self.error_label.setText("Sign up succeeded. Check email to confirm.")
        else:
            msg = result.get("error", "Check details.")
            self.error_label.setText(f"Sign up failed: {msg}")

    def _on_forgot_password(self):
        email = self.email_edit.text().strip()
        if not email:
            self.error_label.setText("Enter your email address first.")
            return
        result = self.client.recover_password(email)
        if result.get("ok"):
            self.error_label.setStyleSheet("color: green;")
            self.error_label.setText("Password reset email sent. Check your inbox.")
        else:
            self.error_label.setStyleSheet("color: red;")
            self.error_label.setText(f"Failed: {result.get('error', 'Unknown error')}")

    def _on_continue(self):
        self.result_mode = "continue"
        self.accept()


class CheckableComboBox(QComboBox):
    """A combo box with checkable items for multi-select filtering."""
    selectionChanged = pyqtSignal()  # Emitted when selection changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self._all_text = "All"
        self._placeholder = "Select..."
        self.view().pressed.connect(self._on_item_pressed)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText(self._placeholder)
        self._updating = False

    def set_all_text(self, text: str):
        """Set the text for the 'All' option."""
        self._all_text = text
        if self._model.rowCount() > 0:
            self._model.item(0).setText(text)

    def add_item(self, text: str, data=None, checked: bool = False):
        """Add a checkable item."""
        item = QStandardItem(text)
        item.setCheckable(True)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setData(data if data is not None else text, Qt.ItemDataRole.UserRole)
        self._model.appendRow(item)
        self._update_display()

    def add_items(self, items: list):
        """Add multiple items. Items can be strings or (text, data) tuples."""
        for item in items:
            if isinstance(item, tuple):
                self.add_item(item[0], item[1])
            else:
                self.add_item(item, item)

    def clear(self):
        """Clear all items."""
        self._model.clear()

    def _on_item_pressed(self, index):
        """Handle item click - toggle checkbox."""
        item = self._model.itemFromIndex(index)
        if item is None:
            return

        self._updating = True
        if item.checkState() == Qt.CheckState.Checked:
            item.setCheckState(Qt.CheckState.Unchecked)
        else:
            item.setCheckState(Qt.CheckState.Checked)

        # If "All" is clicked, affect all other items
        if index.row() == 0:
            all_checked = item.checkState() == Qt.CheckState.Checked
            for i in range(1, self._model.rowCount()):
                self._model.item(i).setCheckState(
                    Qt.CheckState.Checked if all_checked else Qt.CheckState.Unchecked
                )
        else:
            # If any item is unchecked, uncheck "All"
            # If all items are checked, check "All"
            all_checked = all(
                self._model.item(i).checkState() == Qt.CheckState.Checked
                for i in range(1, self._model.rowCount())
            )
            self._model.item(0).setCheckState(
                Qt.CheckState.Checked if all_checked else Qt.CheckState.Unchecked
            )

        self._updating = False
        self._update_display()
        self.selectionChanged.emit()

    def _update_display(self):
        """Update the displayed text based on selection."""
        if self._updating:
            return
        selected = self.get_checked_data()
        if not selected or len(selected) == self._model.rowCount() - 1:
            # None selected or all selected
            self.lineEdit().setText(self._all_text)
        elif len(selected) == 1:
            # Single selection - show the item text
            for i in range(1, self._model.rowCount()):
                item = self._model.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    self.lineEdit().setText(item.text())
                    break
        else:
            # Multiple selections - show count
            self.lineEdit().setText(f"{len(selected)} selected")

    def get_checked_data(self) -> list:
        """Get list of data values for checked items (excluding 'All')."""
        result = []
        for i in range(1, self._model.rowCount()):
            item = self._model.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                result.append(item.data(Qt.ItemDataRole.UserRole))
        return result

    def get_checked_texts(self) -> list:
        """Get list of text values for checked items (excluding 'All')."""
        result = []
        for i in range(1, self._model.rowCount()):
            item = self._model.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                result.append(item.text())
        return result

    def set_checked_data(self, data_list: list):
        """Set which items are checked by their data values."""
        self._updating = True
        data_set = set(data_list)
        for i in range(1, self._model.rowCount()):
            item = self._model.item(i)
            item_data = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(
                Qt.CheckState.Checked if item_data in data_set else Qt.CheckState.Unchecked
            )
        # Update "All" checkbox
        all_checked = all(
            self._model.item(i).checkState() == Qt.CheckState.Checked
            for i in range(1, self._model.rowCount())
        ) if self._model.rowCount() > 1 else True
        self._model.item(0).setCheckState(
            Qt.CheckState.Checked if all_checked else Qt.CheckState.Unchecked
        )
        self._updating = False
        self._update_display()

    def select_all(self):
        """Select all items."""
        self._updating = True
        for i in range(self._model.rowCount()):
            self._model.item(i).setCheckState(Qt.CheckState.Checked)
        self._updating = False
        self._update_display()

    def hidePopup(self):
        """Override to keep popup open when clicking items."""
        # Don't hide - let user check multiple items
        pass

    def mousePressEvent(self, event):
        """Show/hide popup on click."""
        if self.view().isVisible():
            self.view().hide()
        else:
            self.showPopup()


def get_model_version(model_type: str = "species") -> str:
    """Read model version from models/version.txt."""
    version_file = Path(__file__).resolve().parent.parent / "models" / "version.txt"
    if not version_file.exists():
        return "unknown"
    try:
        for line in version_file.read_text().strip().split("\n"):
            if "=" in line:
                key, val = line.split("=", 1)
                if key.strip() == model_type:
                    return val.strip()
    except Exception:
        pass
    return "unknown"


class AnnotatableView(ImageGraphicsView):
    """Image view with simple box drawing support."""
    box_created = pyqtSignal(object)  # emits dict with scene coords
    box_delete_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.draw_mode = None
        self._start_pos = None
        self._temp_rect = None

    def set_draw_mode(self, mode: Optional[str]):
        self.draw_mode = mode
        self._start_pos = None
        if self._temp_rect and self.scene():
            self.scene().removeItem(self._temp_rect)
            self._temp_rect = None

    def mousePressEvent(self, event):
        if self.draw_mode and event.button() == Qt.MouseButton.LeftButton:
            self._start_pos = self.mapToScene(event.pos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draw_mode and self._start_pos is not None:
            end = self.mapToScene(event.pos())
            rect = QRectF(self._start_pos, end).normalized()
            if self._temp_rect is None:
                pen = QPen(Qt.GlobalColor.red)
                pen.setWidth(5)
                self._temp_rect = self.scene().addRect(rect, pen)
            else:
                self._temp_rect.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.draw_mode and event.button() == Qt.MouseButton.LeftButton and self._start_pos is not None:
            end = self.mapToScene(event.pos())
            rect = QRectF(self._start_pos, end).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self.box_created.emit({"label": self.draw_mode, "rect": rect})
            if self._temp_rect and self.scene():
                self.scene().removeItem(self._temp_rect)
            self._temp_rect = None
            self._start_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.scene():
                for it in self.scene().selectedItems():
                    if hasattr(it, "idx"):
                        self.box_delete_requested.emit(it.idx)
                        event.accept()
                        return
        super().keyPressEvent(event)


class DraggableBoxItem(QGraphicsRectItem):
    """Rect item with simple corner-resize and move; calls back on change."""

    def __init__(self, idx: int, rect: QRectF, pen: QPen, on_change, parent=None):
        super().__init__(rect)
        self.idx = idx
        self.on_change = on_change
        self.setPen(pen)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.handle_size = 6.0
        self.active_handle = None  # 0=TL,1=TR,2=BR,3=BL

    def _handles(self):
        r = self.rect()
        return [
            r.topLeft(),
            r.topRight(),
            r.bottomRight(),
            r.bottomLeft(),
        ]

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        # Draw small handles at corners
        painter.save()
        painter.setBrush(self.pen().color())
        for pt in self._handles():
            painter.drawRect(pt.x() - self.handle_size / 2, pt.y() - self.handle_size / 2, self.handle_size, self.handle_size)
        painter.restore()

    def mousePressEvent(self, event):
        # Check if near a handle
        pos = event.pos()
        for i, h in enumerate(self._handles()):
            if (h - pos).manhattanLength() <= self.handle_size * 1.5:
                self.active_handle = i
                event.accept()
                return
        self.active_handle = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.active_handle is not None:
            r = QRectF(self.rect())
            p = event.pos()
            if self.active_handle == 0:  # TL
                r.setTopLeft(p)
            elif self.active_handle == 1:  # TR
                r.setTopRight(p)
            elif self.active_handle == 2:  # BR
                r.setBottomRight(p)
            elif self.active_handle == 3:  # BL
                r.setBottomLeft(p)
            r = r.normalized()
            if r.width() > 2 and r.height() > 2:
                self.setRect(r)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.active_handle is not None:
            self.active_handle = None
        # Notify on any change
        if callable(self.on_change):
            scene_rect = self.mapRectToScene(self.rect())
            self.on_change(self.idx, scene_rect)
from compare_window import CompareWindow
from ai_suggester import CombinedSuggester
from duplicate_dialog import DuplicateDialog, HashCalculationThread
from stamp_reader import StampReader
from image_processor import import_photo, create_thumbnail, extract_cuddelink_camera_id, extract_cuddelink_mac_address
from cuddelink_downloader import download_new_photos, check_server_status

# Ensure Qt finds the platform plugin (Cocoa on macOS, qwindows on Windows) and image plugins
def _ensure_qt_plugin_paths():
    is_windows = sys.platform == "win32"
    is_macos = sys.platform == "darwin"

    # Platform-specific plugin patterns
    if is_windows:
        platform_glob = "qwindows*.dll"
        imageformat_glob = "qjpeg*.dll"
    else:  # macOS/Linux
        platform_glob = "libqcocoa*.dylib" if is_macos else "libqxcb*.so"
        imageformat_glob = "libqjpeg*.dylib" if is_macos else "libqjpeg*.so"

    # macOS: Hardcode known-good user path first (common when PyQt6 was installed with --user)
    if is_macos:
        user_platform = Path.home() / "Library" / "Python" / "3.9" / "lib" / "python" / "site-packages" / "PyQt6" / "Qt6" / "plugins" / "platforms"
        user_root = user_platform.parent if user_platform.parent.name == "plugins" else user_platform.parent
        if user_platform.exists() and not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(user_platform)
        if user_root.exists() and not os.environ.get("QT_PLUGIN_PATH"):
            os.environ["QT_PLUGIN_PATH"] = str(user_root)

    platform_candidates = []
    root_candidates = []
    # PyQt6 location (current import)
    root = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    root_candidates.append(root)
    platform_candidates.append(root / "platforms")
    # User site-packages (pip user install)
    user_base = sysconfig.get_config_var("userbase") or str(Path.home())
    user_purelib = Path(sysconfig.get_path("purelib", vars={"base": user_base}))
    root_candidates.append(user_purelib / "PyQt6" / "Qt6" / "plugins")
    platform_candidates.append(user_purelib / "PyQt6" / "Qt6" / "plugins" / "platforms")

    # OS-specific paths
    if is_macos:
        # Homebrew/System user path fallback (common macOS location)
        hb_root = Path.home() / "Library" / "Python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "lib" / "python" / "site-packages" / "PyQt6" / "Qt6" / "plugins"
        root_candidates.append(hb_root)
        platform_candidates.append(hb_root / "platforms")
    elif is_windows:
        # Windows: Check AppData local site-packages
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            win_root = Path(appdata) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages" / "PyQt6" / "Qt6" / "plugins"
            root_candidates.append(win_root)
            platform_candidates.append(win_root / "platforms")

    # Virtualenv site-packages (explicit)
    try:
        venv = Path(sys.prefix)
        if is_windows:
            vroot = venv / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "plugins"
        else:
            vroot = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "PyQt6" / "Qt6" / "plugins"
        root_candidates.append(vroot)
        platform_candidates.append(vroot / "platforms")
    except Exception:
        pass

    if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        for c in platform_candidates:
            if c.exists() and any(c.glob(platform_glob)):
                os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(c))
                break
    if not os.environ.get("QT_PLUGIN_PATH"):
        for c in root_candidates:
            if c.exists() and (c / "imageformats").exists() and any((c / "imageformats").glob(imageformat_glob)):
                os.environ.setdefault("QT_PLUGIN_PATH", str(c))
                break

_ensure_qt_plugin_paths()


# Fixed species list - alphabetized, admin-only changes
# This is the complete list of valid species for the application
SPECIES_OPTIONS = [
    "",  # Empty option first
    "Armadillo",
    "Bobcat",
    "Chipmunk",
    "Coyote",
    "Deer",
    "Dog",
    "Empty",
    "Flicker",
    "Fox",
    "Ground Hog",
    "House Cat",
    "Opossum",
    "Other",
    "Other Bird",
    "Otter",
    "Person",
    "Quail",
    "Rabbit",
    "Raccoon",
    "Skunk",
    "Squirrel",
    "Turkey",
    "Turkey Buzzard",
    "Unknown",
    "Vehicle",
    "Verification",
]
SEX_TAGS = {"buck", "doe"}

# Master list of valid species labels - matches SPECIES_OPTIONS (excluding empty string)
VALID_SPECIES = set(s for s in SPECIES_OPTIONS if s)
SEX_OPTIONS = ["", "Buck", "Doe", "Unknown"]
AGE_OPTIONS = ["", "1.5", "2.5", "3.5", "4.5", "5.5+", "Fawn", "Mature", "Unknown"]

# Simple modern QSS theme for a cleaner, less “Win95” look
APP_STYLE = """
QMainWindow, QWidget {
    background: #1f252b;
    color: #f2f4f8;
    font-family: 'Helvetica Neue', 'Segoe UI', sans-serif;
    font-size: 12px;
}
QLabel { color: #ffffff; font-weight: 500; }
QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
    background: #2a3138;
    border: 1px solid #3a4148;
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: #0fb6d6;
    color: #f2f4f8;
}
QComboBox QAbstractItemView {
    background: #2a3138;
    selection-background-color: #0fb6d6;
}
QPushButton, QToolButton {
    background: #0fb6d6;
    color: #0b1723;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
}
QPushButton:hover, QToolButton:hover { background: #11c7eb; }
QPushButton:disabled, QToolButton:disabled {
    background: #3a4148;
    color: #8b95a1;
}
QToolButton:checked {
    background: #f2f4f8;
    color: #0b1723;
}
QListWidget {
    background: #1a2026;
    alternate-background-color: #202831;
    border: 1px solid #2f3640;
    padding: 4px;
    color: #f2f4f8;
}
QScrollBar:vertical {
    background: #1f252b;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #2f8fad;
    min-height: 20px;
    border-radius: 4px;
}
QSplitter::handle {
    background: #2f3640;
}
QMenuBar, QMenu {
    background: #1f252b;
    color: #f2f4f8;
}
QMenu::item:selected {
    background: #0fb6d6;
    color: #0b1723;
}
QTreeWidget {
    background: #1a2026;
    alternate-background-color: #202831;
    border: 1px solid #2f3640;
}
"""


class ClickableLabel(QLabel):
    """Simple clickable thumbnail label."""

    def __init__(self, idx: int, on_click, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.on_click = on_click

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and callable(self.on_click):
            self.on_click(self.idx)
        super().mousePressEvent(event)


class FlowLayout(QLayout):
    """A flow layout that arranges widgets left-to-right, wrapping to new lines."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._spacing = 4

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def setSpacing(self, spacing):
        self._spacing = spacing

    def spacing(self):
        return self._spacing

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect)

    def _do_layout(self, rect):
        x = rect.x()
        y = rect.y()
        line_height = 0
        for item in self._items:
            widget = item.widget()
            if widget is None:
                continue
            space = self._spacing
            next_x = x + item.sizeHint().width() + space
            if next_x - space > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space
                next_x = x + item.sizeHint().width() + space
                line_height = 0
            item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = next_x
            line_height = max(line_height, item.sizeHint().height())


class AIWorker(QThread):
    """Background worker thread for AI processing."""

    # Signals emitted during processing
    progress = pyqtSignal(int, int, str)  # current, total, message
    photo_processed = pyqtSignal(dict)  # {id, species, species_conf, sex, sex_conf, boxes_added, heads_added}
    finished_all = pyqtSignal(dict)  # counts dict with all stats
    error = pyqtSignal(str)  # error message

    def __init__(self, photos: list, db, speciesnet_wrapper, buckdoe_suggester=None,
                 species_suggester=None, detector=None, parent=None, options=None):
        super().__init__(parent)
        self.photos = photos
        self.db = db
        self.speciesnet = speciesnet_wrapper
        self.buckdoe_suggester = buckdoe_suggester
        self.species_suggester = species_suggester  # Custom ONNX species model
        self.detector = detector  # Standalone MegaDetector (when not using SpeciesNet's)
        self._cancelled = False
        # Options control which AI steps to run
        self.options = options or {
            "detect_boxes": True,
            "species_id": True,
            "deer_head_boxes": True,
            "buck_doe": True
        }

    def cancel(self):
        self._cancelled = True

    def run(self):
        """Run AI processing in background thread."""
        try:
            self._run_impl()
        except Exception as e:
            logger.error(f"AIWorker crashed: {e}", exc_info=True)
            self.error.emit(f"AI processing crashed: {e}")

    def _run_impl(self):
        """Internal implementation of run() — uses SpeciesNet for detection + classification."""
        total = len(self.photos)
        counts = {"detect": 0, "species": 0, "heads": 0, "sex": 0}

        for i, p in enumerate(self.photos):
            if self._cancelled:
                break

            self.progress.emit(i, total, f"Processing {i + 1} of {total}...")

            result = {
                "id": p.get("id"),
                "species": None,
                "species_conf": 0,
                "sex": None,
                "sex_conf": 0,
                "boxes_added": False,
                "heads_added": False
            }

            try:
                pid = p.get("id")
                file_path = p.get("file_path")

                # Step 0: Check if this is a verification photo (small file < 15 KB)
                if file_path and os.path.exists(file_path):
                    try:
                        size_kb = os.path.getsize(file_path) / 1024
                        if size_kb < 15:
                            self.db.set_suggested_tag(pid, "Verification", 0.95)
                            result["species"] = "Verification"
                            result["species_conf"] = 0.95
                            counts["species"] += 1
                            self.photo_processed.emit(result)
                            continue
                    except Exception:
                        pass

                if not file_path or not os.path.exists(file_path):
                    self.photo_processed.emit(result)
                    continue

                use_md = self.detector is not None and self.options.get("use_megadetector")
                use_custom_cls = self.species_suggester is not None and self.options.get("use_custom_classifier")

                # --- Detection ---
                has_existing_boxes = False
                try:
                    has_existing_boxes = self.db.has_detection_boxes(pid)
                except Exception:
                    pass

                boxes = []
                app_species = None
                score = 0

                if not has_existing_boxes:
                    if use_md:
                        # Standalone MegaDetector v6
                        detections = self.detector.detect(file_path)
                        cat_map = {"animal": "ai_animal", "person": "ai_person", "vehicle": "ai_vehicle"}
                        for det in detections:
                            boxes.append({
                                "label": cat_map.get(det.category, "ai_animal"),
                                "x1": det.bbox[0], "y1": det.bbox[1],
                                "x2": det.bbox[0] + det.bbox[2], "y2": det.bbox[1] + det.bbox[3],
                                "confidence": det.confidence,
                            })
                    elif self.speciesnet:
                        # SpeciesNet detection + classification
                        sn_result = self.speciesnet.detect_and_classify(file_path)
                        boxes = sn_result.get("detections", [])
                        app_species = sn_result.get("app_species")
                        score = sn_result.get("prediction_score", 0)

                    if boxes:
                        self.db.set_boxes(pid, boxes)
                        result["boxes_added"] = True
                        counts["detect"] += 1
                elif not use_md and self.speciesnet:
                    # Already have boxes but still need species from SpeciesNet
                    sn_result = self.speciesnet.detect_and_classify(file_path)
                    app_species = sn_result.get("app_species")
                    score = sn_result.get("prediction_score", 0)

                # --- Classification ---
                check_boxes = self.db.get_boxes(pid) if pid else boxes
                has_person = any(b.get("label") == "ai_person" for b in check_boxes)
                has_vehicle = any(b.get("label") == "ai_vehicle" for b in check_boxes)
                animal_boxes = [b for b in check_boxes if b.get("label") in ("ai_animal", "ai_subject", "subject")]

                label, conf = None, None

                if has_person:
                    label, conf = "Person", 0.95
                elif has_vehicle:
                    label, conf = "Vehicle", 0.95
                elif animal_boxes:
                    if use_custom_cls:
                        # Custom ONNX species model — classify per box via crop
                        best_species, best_conf = None, 0
                        for box in animal_boxes:
                            box_id = box.get("id")
                            crop_result = self._classify_box_custom(file_path, box)
                            if crop_result and box_id:
                                sp, sc = crop_result
                                self.db.set_box_ai_suggestion(box_id, sp, sc)
                                if sc > best_conf:
                                    best_species, best_conf = sp, sc
                        label = best_species or "Empty"
                        conf = best_conf or 0.5
                    elif len(animal_boxes) > 1 and self.speciesnet:
                        # Multiple boxes: SpeciesNet per-box classification
                        per_box = self.speciesnet.classify_per_box(file_path, animal_boxes)
                        best_species, best_conf = None, 0
                        for box, box_pred in zip(animal_boxes, per_box):
                            box_id = box.get("id")
                            box_sp = box_pred.get("app_species")
                            box_sc = box_pred.get("prediction_score", 0)
                            if box_id and box_sp:
                                self.db.set_box_ai_suggestion(box_id, box_sp, box_sc)
                            if box_sc > best_conf and box_sp:
                                best_species, best_conf = box_sp, box_sc
                        label = best_species or app_species
                        conf = best_conf or score
                    elif app_species and app_species not in ("Empty",):
                        # Single box: use whole-image SpeciesNet prediction
                        label, conf = app_species, score
                        box_id = animal_boxes[0].get("id")
                        if box_id:
                            self.db.set_box_ai_suggestion(box_id, app_species, score)
                    else:
                        label, conf = app_species or "Empty", score or 0.95
                elif not check_boxes:
                    label, conf = "Empty", 0.95
                else:
                    label, conf = app_species or "Empty", score or 0.95

                if label and pid:
                    self.db.set_suggested_tag(pid, label, conf)
                    result["species"] = label
                    result["species_conf"] = conf
                    counts["species"] += 1

                # Buck/doe for deer
                if label == "Deer" and self.options.get("buck_doe"):
                    if self.buckdoe_suggester and self.buckdoe_suggester.buckdoe_ready:
                        sex_count = self._predict_sex_for_boxes(p)
                        if sex_count > 0:
                            counts["sex"] += sex_count
                            result["sex"] = "predicted"
                            result["sex_conf"] = sex_count

            except Exception as e:
                logger.warning(f"AI processing failed for photo {p.get('id')}: {e}")

            self.photo_processed.emit(result)

        self.finished_all.emit(counts)

    def _classify_box_custom(self, file_path: str, box: dict):
        """Classify a single detection box using the custom ONNX species model.

        Returns (species, confidence) tuple or None.
        """
        try:
            from PIL import Image
            import tempfile
            img = Image.open(file_path).convert("RGB")
            w, h = img.size
            x1 = int(box.get("x1", 0) * w)
            y1 = int(box.get("y1", 0) * h)
            x2 = int(box.get("x2", 0) * w)
            y2 = int(box.get("y2", 0) * h)
            if x2 - x1 < 32 or y2 - y1 < 32:
                return None
            crop = img.crop((x1, y1, x2, y2))
            crop_path = tempfile.mkstemp(suffix=".jpg")[1]
            try:
                crop.save(crop_path, "JPEG", quality=90)
                result = self.species_suggester.predict(crop_path)
                return result  # (species, confidence) or None
            finally:
                try:
                    os.unlink(crop_path)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Custom classifier failed for box in {file_path}: {e}")
            return None

    def _ensure_detection_boxes(self, photo: dict, detector, names) -> bool:
        """Run detector on photo if it has no boxes."""
        pid = photo.get("id")
        if not pid:
            return False
        try:
            if self.db.has_detection_boxes(pid):
                return True
        except Exception:
            pass
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return False
        boxes = self._detect_boxes_for_path(path, detector, names, conf_thresh=0.25)
        if boxes:
            try:
                self.db.set_boxes(pid, boxes)
                return True
            except Exception:
                pass
        return False

    def _detect_boxes_for_path(self, path: str, detector, names, conf_thresh=0.25):
        """Run detector and return boxes."""
        if detector is None:
            return []
        try:
            results = detector(path, verbose=False)
            boxes = []
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < conf_thresh:
                        continue
                    cls_id = int(box.cls[0])
                    label = names.get(cls_id, f"class_{cls_id}") if names else f"class_{cls_id}"
                    # Normalize coordinates
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    boxes.append({
                        "label": f"ai_{label}",
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": conf
                    })
            return boxes
        except Exception:
            return []

    def _best_crop_for_photo(self, photo: dict):
        """Return (temp_file_path, pixel_area) of the best crop or (None, None)."""
        from PIL import Image

        pid = photo.get("id")
        if not pid:
            return None, None
        boxes = []
        try:
            boxes = self.db.get_boxes(pid)
        except Exception:
            boxes = []
        if not boxes:
            return None, None
        # Prefer deer_head, then ai_animal, then subject
        chosen = None
        for b in boxes:
            if b.get("label") == "deer_head":
                chosen = b
                break
        if chosen is None:
            for b in boxes:
                if b.get("label") == "ai_animal":
                    chosen = b
                    break
        if chosen is None:
            for b in boxes:
                if str(b.get("label")).endswith("subject"):
                    chosen = b
                    break
        if chosen is None:
            return None, None
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return None, None
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            x1 = int(chosen["x1"] * w); x2 = int(chosen["x2"] * w)
            y1 = int(chosen["y1"] * h); y2 = int(chosen["y2"] * h)
            x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
            if x2 - x1 < 5 or y2 - y1 < 5:
                return None, None
            pixel_area = (x2 - x1) * (y2 - y1)
            tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
            img.crop((x1, y1, x2, y2)).save(tmp, "JPEG", quality=90)
            return tmp, pixel_area
        except Exception:
            return None, None

    def _crop_for_box(self, photo: dict, box: dict):
        """Return (temp_file_path, pixel_area) for a specific box or (None, None)."""
        from PIL import Image

        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return None, None
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            x1 = int(box["x1"] * w); x2 = int(box["x2"] * w)
            y1 = int(box["y1"] * h); y2 = int(box["y2"] * h)
            x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
            if x2 - x1 < 5 or y2 - y1 < 5:
                return None, None
            pixel_area = (x2 - x1) * (y2 - y1)
            tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
            img.crop((x1, y1, x2, y2)).save(tmp, "JPEG", quality=90)
            return tmp, pixel_area
        except Exception:
            return None, None

    def _best_head_crop_for_photo(self, photo: dict):
        """Return a temp file path of deer crop for buck/doe classification, or None.

        Priority order:
        1. deer_head (human-labeled)
        2. ai_deer_head (AI-detected head)
        3. subject box with species=Deer (human-labeled body)
        4. ai_animal box (AI-detected body) - buck/doe v2.0 was trained on these
        """
        from PIL import Image

        pid = photo.get("id")
        if not pid:
            return None
        boxes = []
        try:
            boxes = self.db.get_boxes(pid)
        except Exception:
            boxes = []
        if not boxes:
            return None
        # Priority 1: deer_head boxes (human-labeled)
        chosen = None
        for b in boxes:
            if b.get("label") == "deer_head":
                chosen = b
                break
        # Priority 2: ai_deer_head (AI-detected head)
        if chosen is None:
            for b in boxes:
                if b.get("label") == "ai_deer_head":
                    chosen = b
                    break
        # Priority 3: subject box with species=Deer (human-labeled body)
        if chosen is None:
            for b in boxes:
                if b.get("label") == "subject" and b.get("species", "").lower() == "deer":
                    chosen = b
                    break
        # Priority 4: ai_animal box with deer species (buck/doe v2.0 trained on these)
        # Only use ai_animal if species is deer or unset (caller should verify deer)
        if chosen is None:
            for b in boxes:
                if b.get("label") == "ai_animal":
                    box_species = (b.get("species") or "").lower()
                    if box_species in ("deer", ""):
                        chosen = b
                        break
        if chosen is None:
            return None
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return None
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            x1 = int(chosen["x1"] * w); x2 = int(chosen["x2"] * w)
            y1 = int(chosen["y1"] * h); y2 = int(chosen["y2"] * h)
            # Add 10% padding
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x); x2 = min(w, x2 + pad_x)
            y1 = max(0, y1 - pad_y); y2 = min(h, y2 + pad_y)
            if x2 - x1 < 32 or y2 - y1 < 32:
                return None
            tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
            img.crop((x1, y1, x2, y2)).save(tmp, "JPEG", quality=90)
            return tmp
        except Exception:
            return None

    def _add_deer_head_boxes(self, photo: dict, detector, names):
        """Add deer head boxes if deer head detector is available."""
        # This would use a separate deer head detector model
        # For now, skip if not available
        pass

    def _predict_sex_for_boxes(self, photo: dict) -> int:
        """Run buck/doe prediction on all deer boxes in a photo.

        Returns the number of boxes that got sex predictions.
        Stores predictions at box level using set_box_sex.
        """
        if not self.buckdoe_suggester or not self.buckdoe_suggester.buckdoe_ready:
            return 0

        pid = photo.get("id")
        if not pid:
            return 0

        boxes = self.db.get_boxes(pid)
        if not boxes:
            return 0

        # Find deer/animal boxes that need sex prediction
        deer_boxes = []
        head_boxes = {}  # Map head boxes by id for association

        for box in boxes:
            species = (box.get("species") or "").lower()
            label = (box.get("label") or "").lower()
            ai_species = (box.get("ai_suggested_species") or "").lower()

            # Skip if already has sex prediction
            if box.get("sex") and box.get("sex") != "Unknown":
                continue

            # Identify deer boxes (by species or AI suggestion)
            if species == "deer" or ai_species == "deer" or label == "ai_animal":
                deer_boxes.append(box)

            # Track head boxes for potential association
            if label in ("deer_head", "ai_deer_head"):
                head_boxes[box["id"]] = box

        if not deer_boxes:
            return 0

        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return 0

        count = 0
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            w, h = img.size

            for deer_box in deer_boxes:
                crop_path = None
                try:
                    # Try to find associated head box within this deer box
                    best_head = None
                    deer_x1, deer_y1 = deer_box["x1"], deer_box["y1"]
                    deer_x2, deer_y2 = deer_box["x2"], deer_box["y2"]

                    for hbox in head_boxes.values():
                        hx1, hy1, hx2, hy2 = hbox["x1"], hbox["y1"], hbox["x2"], hbox["y2"]
                        # Check if head box center is within deer box
                        hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                        if deer_x1 <= hcx <= deer_x2 and deer_y1 <= hcy <= deer_y2:
                            best_head = hbox
                            break

                    # Crop either head box (preferred) or deer box
                    box_to_crop = best_head if best_head else deer_box
                    x1 = int(box_to_crop["x1"] * w)
                    x2 = int(box_to_crop["x2"] * w)
                    y1 = int(box_to_crop["y1"] * h)
                    y2 = int(box_to_crop["y2"] * h)

                    # Add padding for context
                    pad_x = int((x2 - x1) * 0.1)
                    pad_y = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - pad_x)
                    x2 = min(w, x2 + pad_x)
                    y1 = max(0, y1 - pad_y)
                    y2 = min(h, y2 + pad_y)

                    if x2 - x1 < 32 or y2 - y1 < 32:
                        continue

                    crop_path = Path(tempfile.mkstemp(suffix=".jpg")[1])
                    img.crop((x1, y1, x2, y2)).save(crop_path, "JPEG", quality=90)

                    # Run prediction
                    sex_res = self.buckdoe_suggester.predict_sex(str(crop_path))
                    if sex_res:
                        sex_label, sex_conf = sex_res
                        self.db.set_box_sex(deer_box["id"], sex_label, sex_conf)
                        count += 1

                finally:
                    if crop_path:
                        try:
                            Path(crop_path).unlink(missing_ok=True)
                        except Exception:
                            pass

        except Exception as e:
            logger.warning(f"Error in _predict_sex_for_boxes: {e}")

        return count


class CloudCheckWorker(QThread):
    """Background worker for checking cloud photos without blocking UI."""

    # Signals
    finished = pyqtSignal(list)  # list of cloud-only photos
    error = pyqtSignal(str)  # error message

    def __init__(self, local_hashes: set, parent=None):
        super().__init__(parent)
        self.local_hashes = local_hashes

    def run(self):
        """Run cloud check in background thread."""
        try:
            from supabase_rest import get_cloud_photos_not_local
            cloud_only = get_cloud_photos_not_local(self.local_hashes)
            self.finished.emit(cloud_only)
        except Exception as e:
            logger.error(f"Cloud check worker error: {e}", exc_info=True)
            self.error.emit(str(e))


class TrainerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrailCam Trainer")
        icon_path = Path(__file__).resolve().parent.parent / "ChatGPT Image Dec 5, 2025, 07_07_24 PM.png"
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            self.setWindowIcon(icon)
        self.db = TrailCamDatabase()
        # Don't overwrite labels.txt - it should match the trained model's classes
        # self._write_species_labels_file()
        self.suggester = CombinedSuggester()
        self.speciesnet_wrapper = SpeciesNetWrapper(
            state=user_config.get_speciesnet_state(),
            geofence=True,
        )
        self.auto_enhance_all = True
        self.photos = self._sorted_photos(self.db.get_all_photos())
        # Start at most recent photo (photos are sorted oldest to newest)
        self.index = len(self.photos) - 1 if self.photos else 0
        self.in_review_mode = False
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_current)
        self._known_hashes = None  # lazy-populated MD5 set for duplicate skip

        # Periodic WAL checkpoint timer - reduces data loss risk on crash
        # Checkpoints every 5 minutes (300,000 ms)
        self.wal_checkpoint_timer = QTimer(self)
        self.wal_checkpoint_timer.timeout.connect(self._periodic_wal_checkpoint)
        self.wal_checkpoint_timer.start(300000)  # 5 minutes
        # Photo list cache to reduce database calls
        self._photos_cache = None
        self._photos_cache_time = 0

        self._save_pending = False
        self.current_boxes = []
        self.box_items = []
        self.box_mode = None
        self.boxes_hidden = False
        self.box_tab_widgets = []  # Tab widget references for per-box labeling
        self.ai_review_mode = False
        self.ai_review_queue = []
        self.ai_reviewed_photos = set()  # Track reviewed photo IDs for green highlighting
        self._advancing_review = False  # Guard against recursive advance calls
        self._detector_warned = False

        # Integrated queue mode (replaces modal dialogs)
        self.queue_mode = False  # True when viewing a filtered queue
        self._loading_photo_data = False  # Flag to prevent auto-advance during photo load
        self.queue_type = None  # 'species', 'sex', 'boxes', etc.
        self.queue_photo_ids = []  # List of photo IDs in the queue
        self.queue_data = {}  # Extra data per photo (e.g., suggested species, confidence)
        self.queue_reviewed = set()  # Photo IDs that have been reviewed
        self.queue_pre_filter_index = 0  # Remember position before entering queue

        # Background AI processing
        self.ai_worker = None  # AIWorker instance when running
        self.ai_processing = False  # True while AI is running in background

        # Automatic cloud sync
        self.sync_manager = SyncManager(self.db, self._get_supabase_client_silent)
        self.sync_manager.status_changed.connect(self._on_sync_status_changed)
        self.sync_manager.sync_completed.connect(self._on_sync_completed)
        self.sync_manager.sync_failed.connect(self._on_sync_failed)

        # Automatic R2 photo upload
        self.r2_manager = R2UploadManager(self._get_r2_storage)
        self.r2_manager.status_changed.connect(self._on_r2_status_changed)
        self.r2_manager.upload_completed.connect(self._on_r2_upload_completed)

        # Session-based recently applied species (for quick buttons)
        self._session_recent_species = []  # Species applied this session, most recent first

        # Photos marked for comparison (easier than Ctrl+Click)
        self.marked_for_compare = set()  # Set of photo IDs

        self.scene = QGraphicsScene()
        self.view = AnnotatableView()
        self.view.setScene(self.scene)
        self.view.setMinimumSize(320, 240)
        self.view.box_delete_requested.connect(self._delete_box_by_idx)
        self.current_pixmap = None
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setSingleStep(5)
        self.zoom_slider.valueChanged.connect(lambda v: self.view.set_zoom_level(v / 100.0))
        self.view.zoom_changed.connect(self._sync_zoom_slider)
        self.view.box_created.connect(self._on_box_created)

        self.species_combo = QComboBox()
        self.species_combo.setEditable(True)
        self.species_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.species_combo.setMinimumHeight(28)
        # Disable aggressive autocompletion; let the user type freely
        self.species_combo.setCompleter(None)
        # Add button for multi-species support
        self.add_species_btn = QToolButton()
        self.add_species_btn.setText("+")
        self.add_species_btn.setToolTip("Add species (allows multiple species per photo)")
        self.add_species_btn.clicked.connect(self._add_species_tag)
        # Label showing current species tags
        self.current_species_label = QLabel("")
        self.current_species_label.setStyleSheet("color: #6a6; font-size: 11px;")
        # Recent species quick buttons (applied tags only)
        self.recent_species_btns = []
        for _ in range(10):
            btn = QToolButton()
            btn.setCheckable(False)
            btn.clicked.connect(self._on_quick_species_clicked)
            btn.setMinimumWidth(60)
            self.recent_species_btns.append(btn)
        self._populate_species_dropdown()

        self._setup_nav_shortcuts()

        self.sex_group = QButtonGroup()
        self.sex_group.setExclusive(False)  # Allow manual toggle control
        self.sex_buttons = {}
        sex_row = QHBoxLayout()
        for label in ["Buck", "Doe", "Unknown"]:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setStyleSheet("QToolButton:checked { background:#446; color:white; }")
            self.sex_group.addButton(btn)
            self.sex_buttons[label] = btn
            sex_row.addWidget(btn)
            btn.clicked.connect(lambda checked, lbl=label: self._on_sex_clicked(lbl, checked))
        # default Unknown
        self.sex_buttons["Unknown"].setChecked(True)
        # Sex suggestion display
        sex_row.addSpacing(12)
        self.sex_suggest_label = QLabel("Suggested: —")
        self.sex_suggest_label.setStyleSheet("color: #888; font-size: 11px;")
        self.apply_sex_suggest_btn = QToolButton()
        self.apply_sex_suggest_btn.setText("Apply")
        self.apply_sex_suggest_btn.setEnabled(False)
        self.apply_sex_suggest_btn.clicked.connect(self._apply_sex_suggestion)
        sex_row.addWidget(self.sex_suggest_label)
        sex_row.addWidget(self.apply_sex_suggest_btn)
        sex_row.addStretch()

        self.deer_id_edit = QComboBox()
        self.deer_id_edit.setEditable(True)
        self.deer_id_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.deer_id_edit.setPlaceholderText("Deer ID")
        self.deer_id_edit.setMinimumHeight(28)
        self.deer_id_edit.setMinimumWidth(120)
        self.deer_id_edit.setMaximumWidth(250)
        self.deer_id_edit.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # Make dropdown popup wider to show full deer IDs
        self.deer_id_edit.view().setMinimumWidth(200)
        self._populate_deer_id_dropdown()
        self.deer_id_edit.currentTextChanged.connect(self._apply_buck_profile_to_ui)
        # Handle both typed and programmatic changes (e.g., quick buttons) for autosave/defaults.
        self.deer_id_edit.currentTextChanged.connect(self.on_deer_id_changed)
        self.deer_id_edit.editTextChanged.connect(self.on_deer_id_changed)
        self.merge_btn = QPushButton("Merge Buck IDs")
        self.merge_btn.clicked.connect(self.merge_buck_ids_dialog)
        self.bulk_buck_btn = QPushButton("Set Buck ID on Selected")
        self.bulk_buck_btn.clicked.connect(self.apply_buck_to_selected)
        self.profile_btn = QPushButton("View Buck Profile")
        self.profile_btn.clicked.connect(self.open_buck_profile)
        # ensure additional combo shares IDs
        self.deer_id_edit.currentTextChanged.connect(self._populate_additional_deer_dropdown)

        # Additional buck (second deer) — collapsible
        self.add_buck_toggle = QToolButton()
        self.add_buck_toggle.setText("Add second buck in this photo")
        self.add_buck_toggle.setCheckable(True)
        self.add_buck_toggle.setChecked(False)
        self.add_buck_toggle.toggled.connect(self._toggle_additional_buck)
        self.add_buck_container = QWidget()
        add_layout = QFormLayout(self.add_buck_container)
        add_layout.setContentsMargins(0, 0, 0, 0)
        add_layout.setSpacing(4)
        self.add_deer_id_edit = QComboBox()
        self.add_deer_id_edit.setEditable(True)
        self.add_deer_id_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.add_deer_id_edit.setPlaceholderText("Second Deer ID")
        self.add_deer_id_edit.currentTextChanged.connect(self.schedule_save)
        self.add_age_combo = QComboBox()
        self.add_age_combo.addItems(AGE_OPTIONS)
        self.add_age_combo.currentIndexChanged.connect(self.schedule_save)
        self.add_left_min = QLineEdit()
        self.add_right_min = QLineEdit()
        self.add_left_min.setMaximumWidth(80)
        self.add_right_min.setMaximumWidth(80)
        self.add_left_min.setPlaceholderText("Min")
        self.add_right_min.setPlaceholderText("Min")
        self.add_left_uncertain = QCheckBox("Uncertain")
        self.add_right_uncertain = QCheckBox("Uncertain")
        for w in (self.add_left_min, self.add_right_min):
            w.textChanged.connect(self.schedule_save)
        self.add_left_uncertain.toggled.connect(self.schedule_save)
        self.add_right_uncertain.toggled.connect(self.schedule_save)
        add_layout.addRow("Second Deer ID:", self.add_deer_id_edit)
        add_layout.addRow("Age class:", self.add_age_combo)
        add_layout.addRow("Left typical points:", self._row_typical(self.add_left_min, self.add_left_uncertain))
        add_layout.addRow("Right typical points:", self._row_typical(self.add_right_min, self.add_right_uncertain))
        self.add_buck_container.setVisible(False)

        # Quick select last 9 bucks (3x3 grid)
        self.quick_buck_btns = []
        quick_grid = QVBoxLayout()
        quick_grid.setSpacing(4)
        recent_label = QLabel("Recent Buck IDs:")
        quick_grid.addWidget(recent_label)
        for row_idx in range(3):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(2)
            for col_idx in range(4):
                btn = QToolButton()
                btn.setCheckable(False)
                btn.clicked.connect(self._on_quick_buck_clicked)
                btn.setMinimumWidth(40)
                btn.setMaximumWidth(110)
                btn.setMinimumHeight(20)
                btn.setStyleSheet("font-size: 12px; padding: 2px;")
                btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
                btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                self.quick_buck_btns.append(btn)
                row_layout.addWidget(btn)
            row_layout.addStretch()
            quick_grid.addLayout(row_layout)
        self._update_recent_buck_buttons()
        self.age_combo = QComboBox()
        self.age_combo.addItems(AGE_OPTIONS)

        self.tags_edit = QLineEdit()
        # Camera Location dropdown (used for site clustering)
        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)
        self.camera_combo.setPlaceholderText("Select or type location")
        self.camera_combo.setMinimumWidth(150)
        self.camera_combo.setMinimumHeight(28)
        self._populate_camera_locations()
        self.char_dropdown = QComboBox()
        self.char_dropdown.setEditable(True)
        self.char_dropdown.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.char_dropdown.setPlaceholderText("Select or type characteristic")
        self._populate_char_dropdown()
        self.char_edit = QTextEdit()
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(0)
        self.notes_edit.hide()
        self.left_min = QLineEdit()
        self.right_min = QLineEdit()
        self.left_uncertain = QCheckBox("Uncertain")
        self.right_uncertain = QCheckBox("Uncertain")
        self.left_ab_min = QLineEdit()
        self.right_ab_min = QLineEdit()
        self.left_ab_unc = QCheckBox("Uncertain")
        self.right_ab_unc = QCheckBox("Uncertain")
        self.ab_min = QLineEdit()
        self.ab_max = QLineEdit()
        for w in (self.left_min, self.right_min, self.left_ab_min, self.right_ab_min, self.ab_min, self.ab_max):
            w.setPlaceholderText("e.g. 8")
            w.setMaximumWidth(80)

        # Box tabs container - shows tabs for each detection box
        self.current_box_index = 0  # Which box is currently being edited
        self.box_tab_bar = QTabWidget()
        self.box_tab_bar.setTabPosition(QTabWidget.TabPosition.North)
        self.box_tab_bar.setDocumentMode(True)  # Cleaner look
        self.box_tab_bar.currentChanged.connect(self._on_box_tab_switched)
        self.box_tab_bar.setMaximumHeight(30)
        self.box_tab_bar.setStyleSheet("QTabBar::tab { padding: 4px 12px; }")

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(10, 12, 10, 12)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Add box tab bar at top of form with delete button
        box_row_widget = QWidget()
        box_row = QHBoxLayout(box_row_widget)
        box_row.setContentsMargins(0, 0, 0, 0)
        box_row.setSpacing(6)
        box_row.addWidget(self.box_tab_bar, 1)
        self.add_subject_btn = QPushButton("+")
        self.add_subject_btn.setMaximumWidth(30)
        self.add_subject_btn.setToolTip("Add new subject box (draw on image)")
        self.add_subject_btn.clicked.connect(self._start_add_subject)
        box_row.addWidget(self.add_subject_btn)
        self.clear_labels_btn = QPushButton("Clear")
        self.clear_labels_btn.setMaximumWidth(50)
        self.clear_labels_btn.setToolTip("Clear all labels from all subjects")
        self.clear_labels_btn.clicked.connect(self._clear_all_labels)
        box_row.addWidget(self.clear_labels_btn)
        self.delete_box_btn = QPushButton("Delete")
        self.delete_box_btn.setMaximumWidth(60)
        self.delete_box_btn.setToolTip("Delete current subject")
        self.delete_box_btn.clicked.connect(self._delete_current_box)
        box_row.addWidget(self.delete_box_btn)
        form.addRow("Subject:", box_row_widget)
        species_row_widget = QWidget()
        species_row = QHBoxLayout(species_row_widget)
        species_row.setContentsMargins(0, 0, 0, 0)
        species_row.setSpacing(6)
        species_row.addWidget(self.species_combo)
        species_row.addWidget(self.add_species_btn)
        species_row.addWidget(self.current_species_label)
        species_row.addStretch()
        # Suggested tag display + apply button + recent species quick buttons
        self.suggest_label = QLabel("Suggested: —")
        self.apply_suggest_btn = QToolButton()
        self.apply_suggest_btn.setText("Apply")
        self.apply_suggest_btn.clicked.connect(self._apply_suggestion)
        self.apply_all_suggest_btn = QToolButton()
        self.apply_all_suggest_btn.setText("Apply to All")
        self.apply_all_suggest_btn.setToolTip("Apply current species to all boxes on this photo")
        self.apply_all_suggest_btn.clicked.connect(self._apply_all_suggestions)
        self.accept_all_btn = QToolButton()
        self.accept_all_btn.setText("Accept All")
        self.accept_all_btn.setToolTip("Accept AI suggestion for all boxes on this photo")
        self.accept_all_btn.setStyleSheet("background-color: #4a4; color: white;")
        self.accept_all_btn.clicked.connect(self._accept_all_suggestions)
        # Quick buttons reuse the same recent species buttons
        suggest_row = QVBoxLayout()
        suggest_row.setContentsMargins(0, 0, 0, 0)
        suggest_row.setSpacing(4)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)
        top_row.addWidget(self.suggest_label)
        top_row.addWidget(self.apply_suggest_btn)
        top_row.addWidget(self.apply_all_suggest_btn)
        top_row.addWidget(self.accept_all_btn)
        top_row.addStretch()
        suggest_row.addLayout(top_row)
        form.addRow("Species:", species_row_widget)
        self.suggest_row_widget = QWidget()
        form.addRow("", self.suggest_row_widget)
        self.suggest_row_widget.setLayout(suggest_row)
        # Two-row quick species buttons (10 most recent)
        quick_species_container = QWidget()
        quick_species_layout = QVBoxLayout(quick_species_container)
        quick_species_layout.setContentsMargins(0, 0, 0, 0)
        quick_species_layout.setSpacing(4)
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(4)
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(4)
        for idx, btn in enumerate(self.recent_species_btns):
            (row1 if idx < 5 else row2).addWidget(btn)
        # Add "Other..." button to type custom species
        self.other_species_btn = QToolButton()
        self.other_species_btn.setText("Other...")
        self.other_species_btn.setMinimumWidth(60)
        self.other_species_btn.clicked.connect(self._on_other_species_clicked)
        row2.addWidget(self.other_species_btn)
        row1.addStretch()
        row2.addStretch()
        quick_species_layout.addLayout(row1)
        quick_species_layout.addLayout(row2)
        form.addRow("Recent species:", quick_species_container)
        sex_row_widget = QWidget()
        sex_row_widget.setLayout(sex_row)
        form.addRow("Sex (buck/doe):", sex_row_widget)
        form.addRow("Deer ID:", self.deer_id_edit)
        bulk_row = QHBoxLayout()
        bulk_row.addWidget(self.merge_btn)
        bulk_row.addWidget(self.bulk_buck_btn)
        bulk_row.addWidget(self.profile_btn)
        self.bulk_container = QWidget()
        self.bulk_container.setLayout(bulk_row)
        form.addRow("", self.bulk_container)
        form.addRow(self.add_buck_toggle)
        form.addRow(self.add_buck_container)
        self.quick_buck_container = QWidget()
        self.quick_buck_container.setLayout(quick_grid)
        form.addRow("", self.quick_buck_container)
        # Box tools (split into two rows to avoid crowding)
        box_row_top = QHBoxLayout()
        box_row_bottom = QHBoxLayout()
        self.box_subject_btn = QPushButton("Draw Subject")
        self.box_subject_btn.setCheckable(True)
        self.box_subject_btn.clicked.connect(lambda: self._set_box_mode("subject"))
        self.box_head_btn = QPushButton("Draw Deer Head")
        self.box_head_btn.setCheckable(True)
        self.box_head_btn.clicked.connect(lambda: self._set_box_mode("deer_head"))
        self.box_clear_btn = QPushButton("Clear Boxes")
        self.box_clear_btn.clicked.connect(self._clear_boxes)
        self.box_clear_ai_btn = QPushButton("Clear AI Boxes")
        self.box_clear_ai_btn.clicked.connect(self._clear_ai_boxes)
        self.box_ai_btn = QPushButton("AI Detect Boxes")
        self.box_ai_btn.clicked.connect(self.run_ai_boxes)
        self.box_accept_ai_btn = QPushButton("Accept AI Boxes")
        self.box_accept_ai_btn.clicked.connect(self.accept_ai_boxes)
        self.box_accept_all_ai_btn = QToolButton()
        self.box_accept_all_ai_btn.setText("Accept All AI")
        self.box_accept_all_ai_btn.clicked.connect(self.accept_all_ai_boxes)
        self.box_reject_ai_btn = QToolButton()
        self.box_reject_ai_btn.setText("Reject AI")
        self.box_reject_ai_btn.clicked.connect(self._clear_ai_boxes)
        self.box_edit_btn = QPushButton("Edit Boxes...")
        self.box_edit_btn.clicked.connect(self.edit_boxes)
        self.box_clear_all_ai_btn = QPushButton("Clear ALL AI")
        self.box_clear_all_ai_btn.clicked.connect(self.clear_all_ai_data)
        self.box_bulk_btn = QPushButton("AI Detect All")
        self.box_bulk_btn.clicked.connect(self.run_ai_boxes_all)
        self.box_toggle_btn = QToolButton()
        self.box_toggle_btn.setText("Hide Boxes")
        self.box_toggle_btn.setCheckable(True)
        self.box_toggle_btn.setChecked(False)
        self.box_toggle_btn.toggled.connect(self._toggle_boxes_visible)
        # Top row: drawing + running AI
        box_row_top.addWidget(self.box_subject_btn)
        box_row_top.addWidget(self.box_head_btn)
        box_row_top.addWidget(self.box_ai_btn)
        box_row_top.addWidget(self.box_bulk_btn)
        box_row_top.addWidget(self.box_toggle_btn)
        # Bottom row: review/accept/clear
        box_row_bottom.addWidget(self.box_accept_ai_btn)
        box_row_bottom.addWidget(self.box_reject_ai_btn)
        box_row_bottom.addWidget(self.box_accept_all_ai_btn)
        box_row_bottom.addWidget(self.box_clear_btn)
        box_row_bottom.addWidget(self.box_clear_ai_btn)
        box_row_bottom.addWidget(self.box_clear_all_ai_btn)
        box_row_bottom.addWidget(self.box_edit_btn)
        self.box_container_top = QWidget()
        self.box_container_top.setLayout(box_row_top)
        self.box_container_bottom = QWidget()
        self.box_container_bottom.setLayout(box_row_bottom)
        form.addRow("Boxes:", self.box_container_top)
        form.addRow("", self.box_container_bottom)
        # Hide box controls by default - only show during box queue review
        self.box_container_top.hide()
        self.box_container_bottom.hide()

        # Antler/age details (collapsible)
        self.antler_toggle = QToolButton()
        self.antler_toggle.setText("Antler details ▸")
        self.antler_toggle.setCheckable(True)
        self.antler_toggle.setChecked(False)
        self.antler_toggle.toggled.connect(self._toggle_antler_section)
        antler_layout = QFormLayout()
        antler_layout.setContentsMargins(0, 0, 0, 0)
        antler_layout.setSpacing(4)
        antler_layout.addRow("Age class:", self.age_combo)
        antler_layout.addRow("Left typical points:", self._row_typical(self.left_min, self.left_uncertain))
        antler_layout.addRow("Right typical points:", self._row_typical(self.right_min, self.right_uncertain))
        antler_layout.addRow("Left abnormal points:", self._row_typical(self.left_ab_min, self.left_ab_unc))
        antler_layout.addRow("Right abnormal points:", self._row_typical(self.right_ab_min, self.right_ab_unc))
        antler_layout.addRow("Abnormal points (min/max):", self._row_pair(self.ab_min, self.ab_max))
        self.antler_container = QWidget()
        self.antler_container.setLayout(antler_layout)
        self.antler_container.setVisible(False)
        form.addRow(self.antler_toggle)
        form.addRow("", self.antler_container)
        form.addRow("Tags (comma-separated):", self.tags_edit)
        # Camera location row with quick select buttons
        camera_row = QHBoxLayout()
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.setSpacing(4)
        camera_row.addWidget(self.camera_combo)
        self.location_buttons_layout = QHBoxLayout()
        self.location_buttons_layout.setSpacing(2)
        camera_row.addLayout(self.location_buttons_layout)
        camera_row.addStretch()
        camera_row_widget = QWidget()
        camera_row_widget.setLayout(camera_row)
        self._populate_location_buttons()
        form.addRow("Camera location:", camera_row_widget)
        key_char_row = QHBoxLayout()
        add_char_btn = QPushButton("Add")
        add_char_btn.clicked.connect(self.add_char_from_dropdown)
        key_char_row.addWidget(self.char_dropdown)
        key_char_row.addWidget(add_char_btn)
        self.key_char_container = QWidget()
        self.key_char_container.setLayout(key_char_row)
        form.addRow("Key characteristics:", self.key_char_container)
        # Tag display area - clickable tags with remove buttons
        self.char_tags_container = QWidget()
        self.char_tags_layout = FlowLayout(self.char_tags_container)
        self.char_tags_layout.setSpacing(4)
        form.addRow("", self.char_tags_container)
        # Hide the text edit - we use it for data storage only
        self.char_edit.hide()

        # Removed: Save, Save Next, Prev, Next, Export Training CSVs buttons
        self.compare_btn = QPushButton("Compare Selected")
        self.compare_btn.clicked.connect(self.compare_selected)
        # Multi-select toggle - allows clicking to select multiple photos
        self.multi_select_toggle = QToolButton()
        self.multi_select_toggle.setText("Select Multiple")
        self.multi_select_toggle.setCheckable(True)
        self.multi_select_toggle.setChecked(False)
        self.multi_select_toggle.toggled.connect(self._toggle_multi_select)
        # Select All button
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setToolTip("Select all photos in current filter")
        self.select_all_btn.clicked.connect(self._select_all_photos)
        # Favorite checkbox
        self.favorite_checkbox = QCheckBox("Favorite")
        self.favorite_checkbox.setToolTip("Favorites are protected from archiving")
        self.favorite_checkbox.stateChanged.connect(self._on_favorite_changed)
        # Archive/Unarchive buttons
        self.archive_btn = QPushButton("Archive")
        self.archive_btn.setToolTip("Archive selected photo(s) - hides from default view (favorites protected)")
        self.archive_btn.clicked.connect(self.archive_current_photo)
        self.unarchive_btn = QPushButton("Unarchive")
        self.unarchive_btn.setToolTip("Restore archived photo(s) to default view")
        self.unarchive_btn.clicked.connect(self.unarchive_current_photo)
        # Removed buttons: Select/Mark, Review Queue, Exit Review, Select All, Clear Selection, Set Species
        self.details_toggle = QToolButton()
        self.details_toggle.setText("Hide Details")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setChecked(False)
        self.details_toggle.toggled.connect(self._toggle_details_panel)

        nav = QHBoxLayout()
        nav.addWidget(self.compare_btn)
        nav.addWidget(self.multi_select_toggle)
        nav.addWidget(self.select_all_btn)
        nav.addWidget(self.favorite_checkbox)
        nav.addWidget(self.archive_btn)
        nav.addWidget(self.unarchive_btn)
        nav.addStretch()
        nav.addWidget(self.details_toggle)

        # Wrap form in a container with stretch to fill vertical space
        form_container = QWidget()
        form_container_layout = QVBoxLayout(form_container)
        form_container_layout.setContentsMargins(0, 0, 0, 0)
        form_container_layout.addWidget(form_widget)
        form_container_layout.addStretch(1)

        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setWidget(form_container)
        self.form_scroll.setMinimumWidth(320)
        self.form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        split = QSplitter()
        self.splitter = split
        self._last_split_sizes = [220, 700, 420]
        # Photo list for navigation / multi-select
        self.photo_list_widget = QListWidget()
        self.photo_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.photo_list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.photo_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.photo_list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.photo_list_widget.setUniformItemSizes(True)
        self.photo_list_widget.setAlternatingRowColors(True)
        self.photo_list_widget.setMinimumWidth(220)
        self.photo_list_widget.setStyleSheet("QListWidget { font-size: 11px; }")
        self.photo_list_widget.itemSelectionChanged.connect(self.on_photo_selection_changed)
        QShortcut(QKeySequence.StandardKey.SelectAll, self.photo_list_widget, activated=self.select_all_photos)
        QShortcut(QKeySequence("Ctrl+Shift+Down"), self.photo_list_widget, activated=self.select_to_end)
        QShortcut(QKeySequence("Ctrl+Shift+Up"), self.photo_list_widget, activated=self.select_to_start)
        self.photo_list_widget.itemPressed.connect(self._on_photo_item_pressed)
        self.photo_list_widget.itemClicked.connect(lambda item: setattr(self, "_last_click_row", self.photo_list_widget.row(item)))
        self._last_click_row = -1
        self._detector_warned = False
        # Suggestion filter
        self.suggest_filter_combo = QComboBox()
        self.suggest_filter_combo.setMinimumWidth(100)
        self.suggest_filter_combo.currentIndexChanged.connect(self._populate_photo_list)
        # Site filter
        self.site_filter_combo = QComboBox()
        self.site_filter_combo.setMinimumWidth(100)
        self.site_filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.site_filter_combo.view().setMinimumWidth(200)
        self.site_filter_combo.currentIndexChanged.connect(self._populate_photo_list)
        # Species filter (multi-select)
        self.species_filter_combo = CheckableComboBox()
        self.species_filter_combo.setMinimumWidth(120)
        self.species_filter_combo.view().setMinimumWidth(200)
        self.species_filter_combo.selectionChanged.connect(self._populate_photo_list)
        # Sex filter
        self.sex_filter_combo = QComboBox()
        self.sex_filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sex_filter_combo.view().setMinimumWidth(150)
        self.sex_filter_combo.currentIndexChanged.connect(self._populate_photo_list)
        # Deer ID filter
        self.deer_id_filter_combo = QComboBox()
        self.deer_id_filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.deer_id_filter_combo.view().setMinimumWidth(200)
        self.deer_id_filter_combo.currentIndexChanged.connect(self._populate_photo_list)
        # Year filter (antler year: May-April)
        self.year_filter_combo = QComboBox()
        self.year_filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.year_filter_combo.view().setMinimumWidth(120)
        self.year_filter_combo.currentIndexChanged.connect(self._populate_photo_list)
        # Collection/Farm filter
        self.collection_filter_combo = QComboBox()
        self.collection_filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.collection_filter_combo.view().setMinimumWidth(150)
        self.collection_filter_combo.currentIndexChanged.connect(self._populate_photo_list)
        # Sort order
        self.sort_combo = QComboBox()
        self.sort_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sort_combo.addItem("Date (Newest)", "date_desc")
        self.sort_combo.addItem("Date (Oldest)", "date_asc")
        self.sort_combo.addItem("Location", "location")
        self.sort_combo.addItem("Species", "species")
        self.sort_combo.addItem("Deer ID", "deer_id")
        self.sort_combo.currentIndexChanged.connect(self._populate_photo_list)
        # Archive/Favorites filter (combined)
        self.archive_filter_combo = QComboBox()
        self.archive_filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.archive_filter_combo.addItem("Active Photos", "active")
        self.archive_filter_combo.addItem("Favorites", "favorites")
        self.archive_filter_combo.addItem("Archived", "archived")
        self.archive_filter_combo.addItem("All Photos", "all")
        self.archive_filter_combo.currentIndexChanged.connect(self._populate_photo_list)
        self._populate_species_filter_options()
        self._populate_sex_filter_options()
        self._populate_deer_id_filter_options()
        self._populate_year_filter_options()
        self._populate_collection_filter_options()

        # Single row filter bar - ultra compact, spread horizontally
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(8, 0, 8, 0)
        filter_layout.setSpacing(4)
        for combo in [self.species_filter_combo, self.sex_filter_combo, self.deer_id_filter_combo,
                      self.site_filter_combo, self.year_filter_combo, self.collection_filter_combo,
                      self.sort_combo, self.archive_filter_combo]:
            combo.setMaximumHeight(22)
            combo.setMinimumWidth(90)
            combo.setStyleSheet("QComboBox { padding: 1px 3px; font-size: 11px; }")
        label_style = "QLabel { font-size: 11px; }"
        filters = [("Species:", self.species_filter_combo), ("Sex:", self.sex_filter_combo),
                   ("Deer ID:", self.deer_id_filter_combo), ("Loc:", self.site_filter_combo),
                   ("Year:", self.year_filter_combo), ("Col:", self.collection_filter_combo),
                   ("Sort:", self.sort_combo), ("Show:", self.archive_filter_combo)]
        for i, (text, combo) in enumerate(filters):
            lbl = QLabel(text)
            lbl.setStyleSheet(label_style)
            filter_layout.addWidget(lbl)
            filter_layout.addWidget(combo)
            if i < len(filters) - 1:
                filter_layout.addSpacing(12)  # Fixed spacing between filter pairs
        filter_layout.addStretch()  # Push everything left, remaining space on right

        self.filter_row_container = QWidget()
        self.filter_row_container.setLayout(filter_layout)
        self.filter_row_container.setMaximumHeight(26)
        self.filter_row_container.setStyleSheet("QWidget { background-color: #2d2d2d; border-bottom: 1px solid #444; }")
        self._populate_site_filter_options()

        # Queue control panel (hidden by default, shown when in queue mode)
        self.queue_panel = QFrame()
        self.queue_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        self.queue_panel.setStyleSheet("QFrame { background-color: #2a4a6a; border-radius: 4px; padding: 4px; }")
        queue_layout = QVBoxLayout(self.queue_panel)
        queue_layout.setContentsMargins(8, 6, 8, 6)
        queue_layout.setSpacing(4)

        # Queue header with title and count
        queue_header = QHBoxLayout()
        self.queue_title_label = QLabel("Review Queue")
        self.queue_title_label.setStyleSheet("font-weight: bold; color: white; font-size: 12px;")
        self.queue_count_label = QLabel("0 / 0")
        self.queue_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        queue_header.addWidget(self.queue_title_label)
        queue_header.addStretch()
        queue_header.addWidget(self.queue_count_label)
        queue_layout.addLayout(queue_header)

        # Suggestion info label
        self.queue_suggestion_label = QLabel("")
        self.queue_suggestion_label.setStyleSheet("color: #8cf; font-size: 11px;")
        self.queue_suggestion_label.setWordWrap(True)
        queue_layout.addWidget(self.queue_suggestion_label)

        # AI processing progress bar (hidden when not processing)
        self.queue_progress_bar = QProgressBar()
        self.queue_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background: #333;
                height: 16px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #4a8;
                border-radius: 2px;
            }
        """)
        self.queue_progress_bar.setTextVisible(True)
        self.queue_progress_bar.hide()  # Hidden until AI processing starts
        queue_layout.addWidget(self.queue_progress_bar)

        # Multiple species checkbox (disables auto-advance)
        self.queue_multi_species = QCheckBox("Multiple species in photo")
        self.queue_multi_species.setStyleSheet("color: #aaa; font-size: 11px;")
        queue_layout.addWidget(self.queue_multi_species)

        # Queue action buttons
        queue_btn_layout = QHBoxLayout()
        queue_btn_layout.setSpacing(4)
        self.queue_accept_btn = QPushButton("Accept (A)")
        self.queue_accept_btn.setStyleSheet("background-color: #4a4; color: white; font-weight: bold;")
        self.queue_accept_btn.clicked.connect(self._queue_accept)
        self.queue_reject_btn = QPushButton("Reject (R)")
        self.queue_reject_btn.setStyleSheet("background-color: #a44; color: white;")
        self.queue_reject_btn.clicked.connect(self._queue_reject)
        self.queue_exit_btn = QPushButton("Exit Queue")
        self.queue_exit_btn.setStyleSheet("background-color: #666; color: white;")
        self.queue_exit_btn.clicked.connect(self._exit_queue_mode)
        queue_btn_layout.addWidget(self.queue_accept_btn)
        queue_btn_layout.addWidget(self.queue_reject_btn)
        queue_btn_layout.addWidget(self.queue_exit_btn)
        queue_layout.addLayout(queue_btn_layout)

        self.queue_panel.hide()  # Hidden by default

        left_panel = QWidget()
        left_panel.setMinimumWidth(150)  # Prevent panel from disappearing
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        # filter_row_container moved to top pane
        left_layout.addWidget(self.queue_panel)
        left_layout.addWidget(self.photo_list_widget, 1)
        self._populate_photo_list()
        # preview strip frame
        preview_frame = QFrame()
        preview_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        preview_frame.setMaximumHeight(140)
        self.preview_layout = QHBoxLayout(preview_frame)
        self.preview_layout.setContentsMargins(4, 4, 4, 4)
        self.preview_layout.setSpacing(6)
        # Toggle to collapse/expand previews
        self.preview_frame = preview_frame
        self.preview_toggle = QToolButton()
        self.preview_toggle.setText("Hide previews")
        self.preview_toggle.setCheckable(True)
        self.preview_toggle.setChecked(True)
        self.preview_toggle.toggled.connect(self._toggle_previews)

        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.preview_toggle)
        image_layout.addWidget(preview_frame)
        image_layout.addWidget(self.view)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom"))
        zoom_row.addWidget(self.zoom_slider)
        image_layout.addLayout(zoom_row)

        split.addWidget(left_panel)
        split.addWidget(image_container)
        split.addWidget(self.form_scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        # Prevent panels from collapsing completely
        split.setCollapsible(0, False)
        split.setCollapsible(1, False)
        split.setCollapsible(2, False)

        container = QWidget()
        layout = QVBoxLayout(container)
        # Menu bar with core actions
        menubar = QMenuBar()
        file_menu = QMenu("File", self)
        import_action = file_menu.addAction("Import Folder...")
        import_action.triggered.connect(self.import_folder)
        import_sd_action = file_menu.addAction("Import from SD Card...")
        import_sd_action.triggered.connect(self.import_from_sd_card)
        cudde_action = file_menu.addAction("Download from CuddeLink...")
        cudde_action.triggered.connect(self.download_cuddelink)
        cudde_setup_action = file_menu.addAction("Setup CuddeLink Credentials...")
        cudde_setup_action.triggered.connect(self.setup_cuddelink_credentials)
        cudde_status_action = file_menu.addAction("Check CuddeLink Status...")
        cudde_status_action.triggered.connect(self.check_cuddelink_status)
        file_menu.addSeparator()
        push_cloud_action = file_menu.addAction("Push to Cloud...")
        push_cloud_action.triggered.connect(self.push_to_cloud)
        pull_cloud_action = file_menu.addAction("Pull from Cloud...")
        pull_cloud_action.triggered.connect(self.pull_from_cloud)
        file_menu.addSeparator()
        sign_out_action = file_menu.addAction("Sign Out")
        sign_out_action.triggered.connect(self._sign_out)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        menubar.addMenu(file_menu)
        self.tools_menu = QMenu("Tools", self)
        compare_action = self.tools_menu.addAction("Compare Selected")
        compare_action.triggered.connect(self.compare_selected)
        dup_action = self.tools_menu.addAction("Remove Duplicate/Identical Photos...")
        dup_action.triggered.connect(self.remove_duplicates)
        prune_action = self.tools_menu.addAction("Prune Missing File Entries")
        prune_action.triggered.connect(self.prune_missing_files)
        clear_ai_action = self.tools_menu.addAction("Clear All AI Boxes/Suggestions")
        clear_ai_action.triggered.connect(self.clear_all_ai_data)
        ai_action = self.tools_menu.addAction("Suggest Tags (AI)...")
        ai_action.triggered.connect(self.run_ai_suggestions)
        rerun_ai_action = self.tools_menu.addAction("Re-run AI on Selection...")
        rerun_ai_action.triggered.connect(self.rerun_ai_on_selection)
        detect_missing_boxes_action = self.tools_menu.addAction("Detect Boxes for Tagged Photos...")
        detect_missing_boxes_action.triggered.connect(self.detect_boxes_for_tagged_photos)
        self.tools_menu.addSeparator()
        sex_suggest_action = self.tools_menu.addAction("Suggest Buck/Doe (AI) on Deer Photos...")
        sex_suggest_action.triggered.connect(self.run_sex_suggestions_on_deer)
        verification_action = self.tools_menu.addAction("Detect Verification Photos...")
        verification_action.triggered.connect(self.detect_verification_photos)

        # === REVIEW QUEUES (grouped together) ===
        self.tools_menu.addSeparator()
        review_label = self.tools_menu.addAction("── Review Queues ──")
        review_label.setEnabled(False)

        species_review_action = self.tools_menu.addAction("Review Species Suggestions...")
        species_review_action.triggered.connect(self.review_species_suggestions_integrated)

        sex_review_action = self.tools_menu.addAction("Review Buck/Doe Suggestions...")
        sex_review_action.triggered.connect(self.review_sex_suggestions)

        site_review_action = self.tools_menu.addAction("Review Site Suggestions...")
        site_review_action.triggered.connect(self.review_site_suggestions)

        label_suggestion_action = self.tools_menu.addAction("Review Label Suggestions...")
        label_suggestion_action.triggered.connect(self.review_label_suggestions)

        self.claude_review_action = self.tools_menu.addAction("Claude Review Queue...")
        self.claude_review_action.triggered.connect(self.review_claude_queue)
        self._update_claude_queue_menu()

        mislabel_action = self.tools_menu.addAction("Find Potential Mislabels...")
        mislabel_action.triggered.connect(self.find_potential_mislabels)

        # Check for species mislabel review queue
        mislabel_queue_file = os.path.expanduser("~/.trailcam/species_mislabel_review.json")
        if os.path.exists(mislabel_queue_file):
            try:
                with open(mislabel_queue_file, 'r') as f:
                    queue_data = json.load(f)
                    count = queue_data.get('total', len(queue_data.get('items', [])))
                if count > 0:
                    mislabel_review_action = self.tools_menu.addAction(f"★ Review Mislabeled Species ({count})...")
                    mislabel_review_action.triggered.connect(self.review_mislabeled_species)
            except Exception:
                pass

        # Check for special one-time queue
        self._check_special_queue_menu()

        # === ANNOTATION TOOLS ===
        self.tools_menu.addSeparator()
        annotation_label = self.tools_menu.addAction("── Annotation Tools ──")
        annotation_label.setEnabled(False)
        stamp_reader_action = self.tools_menu.addAction("Stamp Reader...")
        stamp_reader_action.triggered.connect(self.open_stamp_reader)

        self.tools_menu.addSeparator()
        profiles_action = self.tools_menu.addAction("Buck Profiles")
        profiles_action.triggered.connect(self.open_buck_profiles_list)

        # === SITE MANAGEMENT ===
        self.tools_menu.addSeparator()
        site_label = self.tools_menu.addAction("── Sites ──")
        site_label.setEnabled(False)
        auto_site_action = self.tools_menu.addAction("Auto-Detect Sites...")
        auto_site_action.triggered.connect(self.run_site_clustering)
        manage_sites_action = self.tools_menu.addAction("Manage Sites...")
        manage_sites_action.triggered.connect(self.manage_sites)
        manage_cameras_action = self.tools_menu.addAction("Manage Cameras...")
        manage_cameras_action.triggered.connect(self.manage_cameras)

        # Store AI-related actions for simple mode hiding
        self.advanced_menu_actions = [
            clear_ai_action, ai_action, sex_suggest_action,
            review_label, species_review_action,
            sex_review_action, site_review_action, profiles_action,
            site_label, auto_site_action, manage_sites_action
        ]

        # === CLOUD SYNC ===
        self.tools_menu.addSeparator()
        cloud_label = self.tools_menu.addAction("── Cloud Sync ──")
        cloud_label.setEnabled(False)
        cloud_status_action = self.tools_menu.addAction("Cloud Status...")
        cloud_status_action.triggered.connect(self.show_cloud_status)
        upload_thumbs_action = self.tools_menu.addAction("Upload Thumbnails to Cloud...")
        upload_thumbs_action.triggered.connect(lambda: self.upload_to_cloud(thumbnails_only=True))
        upload_all_action = self.tools_menu.addAction("Upload All Photos to Cloud...")
        upload_all_action.triggered.connect(lambda: self.upload_to_cloud(thumbnails_only=False))
        download_cloud_action = self.tools_menu.addAction("Download New Photos from Cloud...")
        download_cloud_action.triggered.connect(self._check_for_new_cloud_photos)
        change_username_action = self.tools_menu.addAction("Change Username...")
        change_username_action.triggered.connect(self.change_username)
        admin_panel_action = self.tools_menu.addAction("Admin: View All Users...")
        admin_panel_action.triggered.connect(self.show_cloud_admin)

        menubar.addMenu(self.tools_menu)
        settings_menu = QMenu("Settings", self)

        # User settings (name & hunting club)
        user_settings_action = settings_menu.addAction("User Settings...")
        user_settings_action.triggered.connect(self.show_user_settings)
        settings_menu.addSeparator()

        self.enhance_toggle_action = settings_menu.addAction("Auto Enhance All")
        self.enhance_toggle_action.setCheckable(True)
        self.enhance_toggle_action.setChecked(True)
        self.enhance_toggle_action.toggled.connect(self.toggle_global_enhance)
        settings_menu.addSeparator()
        photo_locations_action = settings_menu.addAction("Photo Storage Locations...")
        photo_locations_action.triggered.connect(self._manage_photo_locations)
        settings_menu.addSeparator()
        supabase_setup_action = settings_menu.addAction("Setup Supabase Cloud...")
        supabase_setup_action.triggered.connect(self.setup_supabase_credentials)
        reset_sync_prefs_action = settings_menu.addAction("Reset Cloud Sync Preferences...")
        reset_sync_prefs_action.triggered.connect(self._reset_cloud_sync_preferences)
        settings_menu.addSeparator()
        auto_archive_action = settings_menu.addAction("Auto-Archive Settings...")
        auto_archive_action.triggered.connect(self._show_auto_archive_settings)
        settings_menu.addSeparator()
        ai_model_settings_action = settings_menu.addAction("AI Model Settings...")
        ai_model_settings_action.triggered.connect(self._show_ai_model_settings)
        menubar.addMenu(settings_menu)

        # Help menu
        help_menu = QMenu("Help", self)
        check_updates_action = help_menu.addAction("Check for Updates...")
        check_updates_action.triggered.connect(self._check_for_updates)
        help_menu.addSeparator()
        about_action = help_menu.addAction("About Trail Camera Software")
        about_action.triggered.connect(self._show_about)
        menubar.addMenu(help_menu)

        layout.setMenuBar(menubar)
        layout.addWidget(self.filter_row_container)  # Top filter pane
        layout.addWidget(split)
        layout.addLayout(nav)
        self.setCentralWidget(container)

        # Add sync status indicator to status bar
        self.sync_status_label = QLabel("Cloud: Synced")
        self.sync_status_label.setStyleSheet("color: green; padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.sync_status_label)

        # Add R2 upload status indicator
        self.r2_status_label = QLabel("R2: Synced")
        self.r2_status_label.setStyleSheet("color: green; padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.r2_status_label)
        # Add user auth status indicator
        self.user_status_label = QLabel("User: Not logged in")
        self.user_status_label.setStyleSheet("color: gray; padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.user_status_label)
        self._update_auth_status()
        QTimer.singleShot(0, self._maybe_show_login_dialog)

        if not self.photos:
            QMessageBox.information(self, "Trainer", "No photos found in the database.")
        else:
            self.load_photo()
        self._populate_photo_list()

        # Autosave on edits
        self.species_combo.currentIndexChanged.connect(self._on_species_changed)
        # Also connect activated signal - fires when user explicitly selects an item from dropdown
        # This is needed because currentIndexChanged doesn't fire if clicking same item
        self.species_combo.activated.connect(self._on_species_changed)
        # Don't connect editTextChanged - it fires on every keystroke and causes glitches
        # Instead, use schedule_save for typed text (debounced)
        self.species_combo.editTextChanged.connect(self.schedule_save)
        self.age_combo.currentIndexChanged.connect(self.schedule_save)
        self.tags_edit.textChanged.connect(self.schedule_save)
        self.camera_combo.currentTextChanged.connect(self.schedule_save)
        self.char_edit.textChanged.connect(self.schedule_save)
        self.notes_edit.textChanged.connect(self.schedule_save)
        self.left_min.textChanged.connect(self.schedule_save)
        self.right_min.textChanged.connect(self.schedule_save)
        self.left_ab_min.textChanged.connect(self.schedule_save)
        self.right_ab_min.textChanged.connect(self.schedule_save)
        self.ab_min.textChanged.connect(self.schedule_save)
        self.ab_max.textChanged.connect(self.schedule_save)
        self.left_uncertain.toggled.connect(self.schedule_save)
        self.right_uncertain.toggled.connect(self.schedule_save)
        self.left_ab_unc.toggled.connect(self.schedule_save)
        self.right_ab_unc.toggled.connect(self.schedule_save)

        # Load saved settings (window state, simple mode)
        self._load_settings()

        # Check cloud storage on startup (delayed so UI loads first)
        QTimer.singleShot(2000, self._check_cloud_storage_warning)

    def _periodic_wal_checkpoint(self):
        """Periodically checkpoint WAL to reduce data loss risk on crash."""
        try:
            if hasattr(self, 'db') and self.db:
                self.db.checkpoint_wal()
        except Exception as e:
            logger.warning(f"Periodic WAL checkpoint failed: {e}")

    def _check_cloud_storage_warning(self):
        """Check cloud storage and warn if over 8GB (admin only)."""
        try:
            # Only warn admins
            config = user_config.get_config()
            if not config.get('is_admin', False):
                return

            from r2_storage import R2Storage
            storage = R2Storage()
            if not storage.is_configured():
                return

            stats = storage.get_bucket_stats()
            if 'error' in stats:
                return

            total_mb = stats.get('total_size_mb', 0)
            total_gb = total_mb / 1024

            # Limits
            WARNING_GB = 8
            LIMIT_GB = 20

            if total_gb >= LIMIT_GB:
                QMessageBox.critical(
                    self,
                    "Cloud Storage LIMIT REACHED",
                    f"Cloud storage is at {total_gb:.1f} GB!\n\n"
                    f"LIMIT: {LIMIT_GB} GB\n\n"
                    "You need to delete files or upgrade storage."
                )
            elif total_gb >= WARNING_GB:
                QMessageBox.warning(
                    self,
                    "Cloud Storage Warning",
                    f"Cloud storage is at {total_gb:.1f} GB.\n\n"
                    f"Warning threshold: {WARNING_GB} GB\n"
                    f"Limit: {LIMIT_GB} GB\n\n"
                    "Consider managing storage soon."
                )
        except Exception as e:
            logger.debug(f"Cloud storage check failed: {e}")

    def _load_settings(self):
        """Load saved window settings."""
        settings = QSettings("TrailCam", "Trainer")

        # Restore window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Restore maximized state
        if settings.value("maximized", False, type=bool):
            self.showMaximized()

    def _save_settings(self):
        """Save window settings."""
        settings = QSettings("TrailCam", "Trainer")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("maximized", self.isMaximized())

    def showEvent(self, event):
        """Handle window show - prompt for cloud sync and CuddeLink on first show."""
        super().showEvent(event)
        # Only run once on first show
        if not hasattr(self, '_shown_once'):
            self._shown_once = True
            # Use a timer to run after the window is fully shown
            QTimer.singleShot(500, self._run_startup_prompts)

    def _run_startup_prompts(self):
        """Run startup tasks with progress dialog."""
        # First check if user needs to set up username/hunting club
        self._check_user_setup_on_startup()
        # Then run startup tasks with progress
        QTimer.singleShot(100, self._run_startup_with_progress)

    def _check_user_setup_on_startup(self):
        """Prompt for username and hunting clubs if not set."""
        from user_config import get_username, get_hunting_clubs, is_admin

        username = get_username()
        clubs = get_hunting_clubs()

        # Show setup if no username, or no clubs selected (unless admin)
        if not username or (not clubs and not is_admin()):
            self._show_user_setup_dialog()

    def _reset_filters_to_defaults(self):
        """Reset all filters to their default state (show all non-archived photos)."""
        # Block signals to prevent multiple photo list rebuilds
        self.collection_filter_combo.blockSignals(True)
        self.collection_filter_combo.setCurrentIndex(0)  # "All Collections"
        self.collection_filter_combo.blockSignals(False)

        if hasattr(self.species_filter_combo, 'select_all'):
            self.species_filter_combo.select_all()
        else:
            self.species_filter_combo.blockSignals(True)
            self.species_filter_combo.setCurrentIndex(0)
            self.species_filter_combo.blockSignals(False)

        self.archive_filter_combo.blockSignals(True)
        self.archive_filter_combo.setCurrentIndex(0)  # "Active" (non-archived) - index 0
        self.archive_filter_combo.blockSignals(False)

        self.sex_filter_combo.blockSignals(True)
        self.sex_filter_combo.setCurrentIndex(0)  # "All"
        self.sex_filter_combo.blockSignals(False)

        self.site_filter_combo.blockSignals(True)
        self.site_filter_combo.setCurrentIndex(0)  # "All Sites"
        self.site_filter_combo.blockSignals(False)

        self.year_filter_combo.blockSignals(True)
        self.year_filter_combo.setCurrentIndex(0)  # "All Years"
        self.year_filter_combo.blockSignals(False)

        self.deer_id_filter_combo.blockSignals(True)
        self.deer_id_filter_combo.setCurrentIndex(0)  # "All"
        self.deer_id_filter_combo.blockSignals(False)

        self.suggest_filter_combo.blockSignals(True)
        self.suggest_filter_combo.setCurrentIndex(0)  # "All"
        self.suggest_filter_combo.blockSignals(False)

    def _navigate_to_photo_by_id(self, photo_id: int):
        """Navigate to a specific photo by its database ID, bypassing all filters."""
        # Reload all photos including archived
        self.photos = self._sorted_photos(self.db.search_photos(include_archived=True))

        # Find the target photo index in the full list
        target_index = None
        for i, p in enumerate(self.photos):
            if p.get("id") == photo_id:
                target_index = i
                break

        if target_index is None:
            print(f"[WARNING] Photo {photo_id} not found in database")
            return

        # Reset all filters to defaults
        self._reset_filters_to_defaults()

        # Set archive filter to show ALL photos (including archived)
        self.archive_filter_combo.blockSignals(True)
        self.archive_filter_combo.setCurrentIndex(3)  # "All Photos"
        self.archive_filter_combo.blockSignals(False)

        # Set the index BEFORE populating
        self.index = target_index

        # Temporarily set a flag to bypass the "jump to first filtered" behavior
        self._navigating_to_specific_photo = True

        # Populate the list - this will rebuild filters but our flag prevents index change
        self._populate_photo_list()

        self._navigating_to_specific_photo = False

        # Make sure we're on the right photo
        self.index = target_index
        self.load_photo()

        # Also select it in the list widget
        for row in range(self.photo_list_widget.count()):
            item = self.photo_list_widget.item(row)
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx == target_index:
                self.photo_list_widget.blockSignals(True)
                self.photo_list_widget.setCurrentItem(item)
                item.setSelected(True)
                self.photo_list_widget.scrollToItem(item)
                self.photo_list_widget.blockSignals(False)
                break

    def _run_startup_with_progress(self):
        """Run startup tasks with a unified progress dialog.

        Combines label sync and cloud photo checks into one streamlined flow.
        Uses background thread to avoid blocking UI.
        """
        # Reset all filters to defaults on startup
        self._reset_filters_to_defaults()

        # Check if we have cloud configured
        has_supabase = False
        try:
            import json
            from pathlib import Path
            config_paths = [
                Path(__file__).parent.parent / "cloud_config.json",
                Path(__file__).parent / "cloud_config.json",
                Path.cwd() / "cloud_config.json",
            ]
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    if config.get("supabase", {}).get("url") and config.get("supabase", {}).get("anon_key"):
                        has_supabase = True
                        break
        except:
            pass

        if not has_supabase:
            # No cloud configured, nothing to do
            return

        # Worker thread for cloud sync
        class StartupSyncWorker(QThread):
            progress = pyqtSignal(str)  # status message
            finished = pyqtSignal(list)  # photos needing thumbnails
            error = pyqtSignal(str)

            def __init__(self, db, get_client_func, parent=None):
                super().__init__(parent)
                self.db = db
                self.get_client_func = get_client_func

            def run(self):
                try:
                    self.progress.emit("Syncing labels from cloud...")
                    client = self.get_client_func()
                    if client:
                        self.db.pull_from_supabase(client)

                    self.progress.emit("Checking for photos to download...")
                    # Get ALL cloud-only photos - let user decide thumbnails vs full
                    cloud_only_photos = self.db.get_cloud_only_photos()
                    self.finished.emit(cloud_only_photos)
                except Exception as e:
                    logger.error(f"Startup sync failed: {e}")
                    self.error.emit(str(e))

        # Create unified startup dialog
        startup_dlg = QDialog(self)
        startup_dlg.setWindowTitle("Syncing with Cloud")
        startup_dlg.setFixedWidth(400)
        startup_dlg.setWindowFlags(startup_dlg.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)
        layout = QVBoxLayout(startup_dlg)

        status_label = QLabel("Connecting to cloud...")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Indeterminate while syncing
        layout.addWidget(progress_bar)

        result = {"photos_to_download": [], "done": False}

        def on_progress(msg):
            status_label.setText(msg)

        def on_finished(photos):
            result["photos_to_download"] = photos
            result["done"] = True
            startup_dlg.close()

        def on_error(err):
            result["done"] = True
            startup_dlg.close()

        # Start worker
        worker = StartupSyncWorker(self.db, self._get_supabase_client, startup_dlg)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()

        startup_dlg.exec()

        # Wait for worker if dialog was closed early
        if worker.isRunning():
            worker.wait(10000)

        # Refresh photo list after sync
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_photo_list()

        # Mark that we've done the cloud check so it doesn't run again
        self._cloud_only_prompted = True

        # If there are photos to download, show single combined prompt
        if result["photos_to_download"]:
            self._prompt_cloud_download(result["photos_to_download"])

    def _prompt_cloud_download(self, photos_to_download: list):
        """Show a single combined prompt to download photos from cloud with progress bar."""
        count = len(photos_to_download)

        # Create download dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Download from Cloud")
        dlg.setFixedWidth(450)
        layout = QVBoxLayout(dlg)

        # Info label
        info_label = QLabel(
            f"<b>{count} photo{'s' if count != 1 else ''}</b> {'are' if count != 1 else 'is'} "
            f"available in the cloud but not on this device."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Download options
        option_label = QLabel("What would you like to download?")
        layout.addWidget(option_label)

        thumbs_only_radio = QRadioButton("Thumbnails only (fast, for browsing)")
        layout.addWidget(thumbs_only_radio)

        full_photos_radio = QRadioButton("Full photos (slower, for editing/viewing)")
        full_photos_radio.setChecked(True)  # Default to full photos
        layout.addWidget(full_photos_radio)

        # Progress bar (hidden initially)
        progress_bar = QProgressBar()
        progress_bar.setMaximum(count)
        progress_bar.setValue(0)
        progress_bar.setVisible(False)
        layout.addWidget(progress_bar)

        status_label = QLabel("")
        status_label.setVisible(False)
        layout.addWidget(status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        download_btn = QPushButton("Download")
        skip_btn = QPushButton("Skip")
        btn_layout.addWidget(download_btn)
        btn_layout.addWidget(skip_btn)
        layout.addLayout(btn_layout)

        # Worker thread for downloading
        class CloudDownloadWorker(QThread):
            progress = pyqtSignal(int, int, str)  # current, total, message
            finished = pyqtSignal(int, int)  # downloaded, failed

            def __init__(self, photos, db, full_photos=False, parent=None):
                super().__init__(parent)
                self.photos = photos
                self.db = db
                self.full_photos = full_photos
                self._cancelled = False

            def cancel(self):
                self._cancelled = True

            def run(self):
                from r2_storage import R2Storage
                from pathlib import Path
                from datetime import datetime as dt

                r2 = R2Storage()
                if not r2.is_configured():
                    self.finished.emit(0, len(self.photos))
                    return

                photo_dir = Path.home() / "TrailCamLibrary"
                thumb_dir = photo_dir / ".thumbnails"
                thumb_dir.mkdir(parents=True, exist_ok=True)

                downloaded = 0
                failed = 0
                total = len(self.photos)

                for i, photo in enumerate(self.photos):
                    if self._cancelled:
                        break

                    file_hash = photo.get("file_hash")
                    photo_id = photo.get("id")
                    if not file_hash:
                        failed += 1
                        continue

                    mode = "photo" if self.full_photos else "thumbnail"
                    self.progress.emit(i + 1, total, f"Downloading {mode} {i + 1} of {total}...")

                    try:
                        # Always download thumbnail
                        thumb_key = f"thumbnails/{file_hash}_thumb.jpg"
                        thumb_path = thumb_dir / f"{file_hash}_thumb.jpg"
                        if not thumb_path.exists():
                            r2.download_photo(thumb_key, thumb_path)

                        if photo_id and thumb_path.exists():
                            self.db.update_thumbnail_path(photo_id, str(thumb_path))

                        if self.full_photos:
                            # Download full photo
                            date_taken = photo.get("date_taken", "")
                            original_name = photo.get("original_name", f"{file_hash}.jpg")

                            # Determine destination folder by date
                            year, month = "Unknown", "Unknown"
                            if date_taken:
                                try:
                                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                                        try:
                                            parsed = dt.strptime(date_taken[:19], fmt)
                                            year = str(parsed.year)
                                            month = f"{parsed.month:02d}"
                                            break
                                        except ValueError:
                                            continue
                                except Exception:
                                    pass

                            dest_folder = photo_dir / year / month
                            dest_folder.mkdir(parents=True, exist_ok=True)
                            dest_path = dest_folder / original_name

                            def _lookup_path(path_obj):
                                try:
                                    lock = getattr(self.db, "_lock", None)
                                    if lock:
                                        with lock:
                                            cursor = self.db.conn.cursor()
                                            cursor.execute("SELECT id, file_hash FROM photos WHERE file_path = ?", (str(path_obj),))
                                            return cursor.fetchone()
                                    cursor = self.db.conn.cursor()
                                    cursor.execute("SELECT id, file_hash FROM photos WHERE file_path = ?", (str(path_obj),))
                                    return cursor.fetchone()
                                except Exception:
                                    return None

                            def _lookup_hash_path(fh):
                                try:
                                    lock = getattr(self.db, "_lock", None)
                                    if lock:
                                        with lock:
                                            cursor = self.db.conn.cursor()
                                            cursor.execute(
                                                "SELECT file_path FROM photos WHERE file_hash = ? AND file_path IS NOT NULL AND file_path NOT LIKE 'cloud://%' LIMIT 1",
                                                (fh,)
                                            )
                                            return cursor.fetchone()
                                    cursor = self.db.conn.cursor()
                                    cursor.execute(
                                        "SELECT file_path FROM photos WHERE file_hash = ? AND file_path IS NOT NULL AND file_path NOT LIKE 'cloud://%' LIMIT 1",
                                        (fh,)
                                    )
                                    return cursor.fetchone()
                                except Exception:
                                    return None

                            def _disambiguate_path(path_obj, fh):
                                base = path_obj.stem
                                suffix = path_obj.suffix or ".jpg"
                                candidate = path_obj.with_name(f"{base}_{fh[:8]}{suffix}")
                                if not candidate.exists():
                                    row = _lookup_path(candidate)
                                    if not row:
                                        return candidate
                                idx = 1
                                while True:
                                    candidate = path_obj.with_name(f"{base}_{fh[:8]}_{idx}{suffix}")
                                    if not candidate.exists():
                                        row = _lookup_path(candidate)
                                        if not row:
                                            return candidate
                                    idx += 1

                            row = _lookup_path(dest_path)
                            if row and row[0] != photo_id and row[1] != file_hash:
                                dest_path = _disambiguate_path(dest_path, file_hash)
                            elif dest_path.exists() and not row:
                                existing = _lookup_hash_path(file_hash)
                                if existing and existing[0]:
                                    dest_path = Path(existing[0])
                                else:
                                    dest_path = _disambiguate_path(dest_path, file_hash)
                            elif not row and not dest_path.exists():
                                row = _lookup_path(dest_path)
                                if row and row[0] != photo_id:
                                    dest_path = _disambiguate_path(dest_path, file_hash)

                            if dest_path.exists():
                                # File already downloaded — just update DB path
                                if photo_id:
                                    self.db.update_file_path(photo_id, str(dest_path))
                            else:
                                # Download from R2
                                photo_key = f"photos/{file_hash}.jpg"
                                success = r2.download_photo(photo_key, dest_path)
                                if not success:
                                    # Try .jpeg extension
                                    photo_key = f"photos/{file_hash}.jpeg"
                                    success = r2.download_photo(photo_key, dest_path)

                                if success and photo_id:
                                    self.db.update_file_path(photo_id, str(dest_path))
                                elif not success:
                                    failed += 1
                                    continue

                        downloaded += 1
                    except Exception as e:
                        logger.error(f"Failed to download {file_hash}: {e}")
                        failed += 1

                self.finished.emit(downloaded, failed)

        result = {"cancelled": False, "worker": None}

        def on_skip():
            result["cancelled"] = True
            if result["worker"] and result["worker"].isRunning():
                result["worker"].cancel()
                result["worker"].wait(2000)
            dlg.close()

        skip_btn.clicked.connect(on_skip)

        def on_progress(current, total, msg):
            progress_bar.setValue(current)
            status_label.setText(msg)

        def on_finished(downloaded, failed):
            mode = "photos" if result.get("full_photos") else "thumbnails"
            if result["cancelled"]:
                info_label.setText(f"Cancelled. Downloaded {downloaded} of {count}.")
            else:
                if failed > 0:
                    info_label.setText(f"Done! Downloaded {downloaded} {mode}, {failed} failed.")
                else:
                    info_label.setText(f"Done! Downloaded {downloaded} {mode}.")
            skip_btn.setText("Close")
            download_btn.setEnabled(False)
            progress_bar.setValue(count)
            # Hide radio buttons after download
            thumbs_only_radio.setVisible(False)
            full_photos_radio.setVisible(False)
            option_label.setVisible(False)

        def on_download():
            # Check which mode is selected
            full_photos = full_photos_radio.isChecked()
            result["full_photos"] = full_photos

            # Switch to download mode
            download_btn.setEnabled(False)
            skip_btn.setText("Cancel")
            progress_bar.setVisible(True)
            status_label.setVisible(True)
            thumbs_only_radio.setEnabled(False)
            full_photos_radio.setEnabled(False)

            mode = "full photos" if full_photos else "thumbnails"
            info_label.setText(f"Downloading {mode}...")

            # Start worker thread
            worker = CloudDownloadWorker(photos_to_download, self.db, full_photos, dlg)
            worker.progress.connect(on_progress)
            worker.finished.connect(on_finished)
            result["worker"] = worker
            worker.start()

        download_btn.clicked.connect(on_download)

        dlg.exec()

        # Wait for worker to finish if still running
        if result["worker"] and result["worker"].isRunning():
            result["worker"].wait(5000)

        # Refresh photo list after downloads
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_photo_list()

    def _check_cloud_pull_on_startup(self):
        """Check if we should pull from cloud on startup."""
        settings = QSettings("TrailCam", "Trainer")

        # Check if we have Supabase credentials (bundled or user-configured)
        has_supabase = False
        try:
            import json
            from pathlib import Path
            config_paths = [
                Path(__file__).parent.parent / "cloud_config.json",
                Path(__file__).parent / "cloud_config.json",
                Path.cwd() / "cloud_config.json",
            ]
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    if config.get("supabase", {}).get("url") and config.get("supabase", {}).get("anon_key"):
                        has_supabase = True
                        break
        except:
            pass

        # Also check user settings as fallback
        if not has_supabase:
            url = settings.value("supabase_url", "")
            key = settings.value("supabase_key", "")
            has_supabase = bool(url and key)

        if not has_supabase:
            return  # No cloud configured, skip

        # Check the "always" setting
        always_pull = settings.value("cloud_always_pull_on_open", "")

        if always_pull == "yes":
            self._do_cloud_pull_silent()
        elif always_pull == "no":
            pass  # User chose to never pull
        else:
            # Ask the user
            self._prompt_cloud_pull()

    def _check_cuddelink_on_startup(self):
        """Check if we should download from CuddeLink on startup."""
        settings = QSettings("TrailCam", "Trainer")

        # Check if we have CuddeLink credentials
        email = settings.value("cuddelink_email", "")
        password = settings.value("cuddelink_password", "")
        if not email or not password:
            return  # No CuddeLink configured, skip

        # Check the "always" setting
        always_download = settings.value("cuddelink_always_download_on_open", "")

        if always_download == "yes":
            self.download_cuddelink()
        elif always_download == "no":
            pass  # User chose to never download on open
        else:
            # Ask the user
            self._prompt_cuddelink_download()

    def _prompt_cuddelink_download(self):
        """Show dialog asking whether to download from CuddeLink."""
        dialog = QDialog(self)
        dialog.setWindowTitle("CuddeLink Download")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        label = QLabel("Would you like to download new photos from CuddeLink?")
        label.setWordWrap(True)
        layout.addWidget(label)

        remember_check = QCheckBox("Remember my choice")
        layout.addWidget(remember_check)

        btn_layout = QHBoxLayout()
        yes_btn = QPushButton("Yes, Download")
        no_btn = QPushButton("No, Skip")
        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(no_btn)
        layout.addLayout(btn_layout)

        result = {"choice": None}

        def on_yes():
            result["choice"] = "yes"
            if remember_check.isChecked():
                settings = QSettings("TrailCam", "Trainer")
                settings.setValue("cuddelink_always_download_on_open", "yes")
            dialog.accept()

        def on_no():
            result["choice"] = "no"
            if remember_check.isChecked():
                settings = QSettings("TrailCam", "Trainer")
                settings.setValue("cuddelink_always_download_on_open", "no")
            dialog.accept()

        yes_btn.clicked.connect(on_yes)
        no_btn.clicked.connect(on_no)

        dialog.exec()

        if result["choice"] == "yes":
            self.download_cuddelink()

    def _prompt_cloud_pull(self):
        """Show dialog asking whether to pull from cloud."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Cloud Sync")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        label = QLabel("Would you like to download the latest labels from the cloud?")
        label.setWordWrap(True)
        layout.addWidget(label)

        remember_check = QCheckBox("Remember my choice")
        layout.addWidget(remember_check)

        btn_layout = QHBoxLayout()
        yes_btn = QPushButton("Yes, Download")
        no_btn = QPushButton("No, Skip")
        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(no_btn)
        layout.addLayout(btn_layout)

        result = {"choice": None}

        def on_yes():
            result["choice"] = "yes"
            if remember_check.isChecked():
                settings = QSettings("TrailCam", "Trainer")
                settings.setValue("cloud_always_pull_on_open", "yes")
            dialog.accept()

        def on_no():
            result["choice"] = "no"
            if remember_check.isChecked():
                settings = QSettings("TrailCam", "Trainer")
                settings.setValue("cloud_always_pull_on_open", "no")
            dialog.accept()

        yes_btn.clicked.connect(on_yes)
        no_btn.clicked.connect(on_no)

        dialog.exec()

        if result["choice"] == "yes":
            self._do_cloud_pull_silent()

    def _do_cloud_pull_silent(self):
        """Pull from cloud without showing the full dialog."""
        try:
            client = self._get_supabase_client()
            if not client:
                return

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()

            counts = self.db.pull_from_supabase(client)

            QApplication.restoreOverrideCursor()

            total = sum(counts.values())
            if total > 0:
                # Silently reload data without showing popup
                self.photos = self._sorted_photos(self.db.get_all_photos())
                self._populate_photo_list()
                if self.photos:
                    self.load_photo()

            # After pulling labels, check for new photos that only exist in cloud
            QTimer.singleShot(100, self._check_for_new_cloud_photos)

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Cloud Sync", f"Failed to pull from cloud:\n{str(e)}")

    def _check_for_new_cloud_photos(self):
        """Check if there are photos in cloud that don't exist locally, offer to download.

        Runs the cloud query in a background thread to avoid blocking the UI.
        """
        try:
            # Get local file hashes (fast - from local DB)
            local_hashes = self.db.get_all_file_hashes()

            # Create and start background worker for cloud check
            self._cloud_check_worker = CloudCheckWorker(local_hashes, self)
            self._cloud_check_worker.finished.connect(self._on_cloud_check_finished)
            self._cloud_check_worker.error.connect(self._on_cloud_check_error)
            self._cloud_check_worker.start()

        except Exception as e:
            logger.error(f"Error starting cloud check: {e}", exc_info=True)
            # Still check for existing cloud-only photos
            QTimer.singleShot(200, self._check_for_cloud_only_photos)

    def _on_cloud_check_finished(self, cloud_only: list):
        """Handle results from background cloud check."""
        # Clean up worker reference
        if hasattr(self, '_cloud_check_worker'):
            self._cloud_check_worker.deleteLater()
            del self._cloud_check_worker

        if not cloud_only:
            # No new photos in cloud - check for existing cloud-only photos
            QTimer.singleShot(200, self._check_for_cloud_only_photos)
            return

        # Show dialog asking user if they want to download
        count = len(cloud_only)
        reply = QMessageBox.question(
            self,
            "New Photos in Cloud",
            f"Found {count} photo{'s' if count != 1 else ''} in the cloud that {'are' if count != 1 else 'is'} not on this device.\n\n"
            f"Would you like to download {'them' if count != 1 else 'it'}?\n\n"
            "(Only thumbnails will be downloaded for viewing. Full photos remain in cloud.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Start download with progress dialog
            self._download_cloud_photos(cloud_only)

        # After handling, also check for existing cloud-only photos
        QTimer.singleShot(200, self._check_for_cloud_only_photos)

    def _on_cloud_check_error(self, error_msg: str):
        """Handle error from background cloud check."""
        # Clean up worker reference
        if hasattr(self, '_cloud_check_worker'):
            self._cloud_check_worker.deleteLater()
            del self._cloud_check_worker

        logger.error(f"Cloud check failed: {error_msg}")
        # Still check for existing cloud-only photos
        QTimer.singleShot(200, self._check_for_cloud_only_photos)

    def _check_for_cloud_only_photos(self):
        """Check for photos with cloud:// paths that need to be downloaded."""
        # Prevent showing prompt twice
        if getattr(self, '_cloud_only_prompted', False):
            return
        self._cloud_only_prompted = True

        try:
            # Get photos where file_path starts with cloud://
            cloud_only_photos = self.db.get_cloud_only_photos()

            if not cloud_only_photos:
                return  # All photos are downloaded

            count = len(cloud_only_photos)
            reply = QMessageBox.question(
                self,
                "Download Cloud Photos",
                f"{count} photo{'s are' if count != 1 else ' is'} stored in the cloud but not downloaded to this device.\n\n"
                f"Would you like to download {'them' if count != 1 else 'it'} now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._download_existing_cloud_photos(cloud_only_photos)

        except Exception as e:
            logger.error(f"Error checking for cloud-only photos: {e}", exc_info=True)

    def _download_existing_cloud_photos(self, photos: list):
        """Download full photos for existing cloud-only records."""
        from r2_storage import R2Storage
        from pathlib import Path

        # Create progress dialog
        self._cloud_dl_dialog = QDialog(self)
        self._cloud_dl_dialog.setWindowTitle("Downloading Photos from Cloud")
        self._cloud_dl_dialog.setMinimumWidth(400)
        self._cloud_dl_dialog.setModal(True)
        layout = QVBoxLayout(self._cloud_dl_dialog)

        self._cloud_dl_label = QLabel(f"Preparing to download {len(photos)} photos...")
        layout.addWidget(self._cloud_dl_label)

        self._cloud_dl_progress = QProgressBar()
        self._cloud_dl_progress.setMaximum(len(photos))
        self._cloud_dl_progress.setValue(0)
        layout.addWidget(self._cloud_dl_progress)

        cancel_btn = QPushButton("Cancel")
        layout.addWidget(cancel_btn)

        self._cloud_dl_cancelled = False

        def on_cancel():
            self._cloud_dl_cancelled = True
            if hasattr(self, '_cloud_dl_worker') and self._cloud_dl_worker.isRunning():
                self._cloud_dl_worker.cancel()
            self._cloud_dl_dialog.close()

        cancel_btn.clicked.connect(on_cancel)

        # Worker thread for downloading
        class ExistingCloudPhotoWorker(QThread):
            progress = pyqtSignal(int, int, str)  # current, total, message
            finished = pyqtSignal(int, int)  # downloaded, failed

            def __init__(self, photos, db, r2, photo_dir):
                super().__init__()
                self.photos = photos
                self.db = db
                self.r2 = r2
                self.photo_dir = photo_dir
                self._cancelled = False

            def cancel(self):
                self._cancelled = True

            def run(self):
                from datetime import datetime as dt
                downloaded = 0
                failed = 0
                total = len(self.photos)

                try:
                    for i, photo in enumerate(self.photos):
                        if self._cancelled:
                            break

                        try:
                            file_hash = photo.get("file_hash")
                            photo_id = photo.get("id")
                            if not file_hash or not photo_id:
                                failed += 1
                                continue

                            self.progress.emit(i + 1, total, f"Downloading {i + 1} of {total}...")

                            # Determine destination path
                            date_taken = photo.get("date_taken", "")
                            original_name = photo.get("original_name", f"{file_hash}.jpg")

                            year, month = "Unknown", "Unknown"
                            if date_taken:
                                try:
                                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                                        try:
                                            parsed = dt.strptime(date_taken[:19], fmt)
                                            year = str(parsed.year)
                                            month = f"{parsed.month:02d}"
                                            break
                                        except ValueError:
                                            continue
                                except Exception:
                                    pass

                            dest_folder = self.photo_dir / year / month
                            dest_folder.mkdir(parents=True, exist_ok=True)

                            dest_path = dest_folder / original_name
                            if dest_path.exists():
                                ext = Path(original_name).suffix or ".jpg"
                                dest_path = dest_folder / f"{file_hash}{ext}"

                            # Download from R2
                            r2_key = f"photos/{file_hash}.jpg"
                            download_success = self.r2.download_photo(r2_key, dest_path)
                            if not download_success:
                                # Try .jpeg
                                r2_key = f"photos/{file_hash}.jpeg"
                                download_success = self.r2.download_photo(r2_key, dest_path)

                            if download_success:
                                self.db.update_file_path(photo_id, str(dest_path))
                                downloaded += 1

                                # SAFEGUARD: Verify timestamp from local EXIF after download
                                # This catches cases where cloud had wrong timestamp
                                try:
                                    exif_date = self._read_exif_datetime(dest_path)
                                    if exif_date and exif_date != date_taken:
                                        # EXIF differs from cloud - update local DB (thread-safe)
                                        self.db.update_date_taken(photo_id, exif_date)
                                except Exception:
                                    pass  # Don't fail download if EXIF check fails
                            else:
                                failed += 1

                        except Exception as e:
                            logger.error(f"Error downloading photo {photo.get('file_hash', 'unknown')}: {e}")
                            failed += 1

                except Exception as e:
                    logger.error(f"Critical error in download worker: {e}", exc_info=True)

                self.finished.emit(downloaded, failed)

            def _read_exif_datetime(self, image_path):
                """Read DateTimeOriginal from EXIF."""
                try:
                    from PIL import Image
                    from PIL.ExifTags import TAGS
                    with Image.open(image_path) as img:
                        exif = img._getexif()
                        if exif:
                            dt_orig = exif.get(36867)  # DateTimeOriginal
                            if dt_orig:
                                return str(dt_orig).replace(':', '-', 2).replace(' ', 'T')
                except Exception:
                    pass
                return None

        photo_dir = Path.home() / "TrailCamLibrary"
        photo_dir.mkdir(parents=True, exist_ok=True)

        r2 = R2Storage()
        if not r2.is_configured():
            QMessageBox.warning(self, "Cloud Download", "R2 storage is not configured.")
            return

        self._cloud_dl_worker = ExistingCloudPhotoWorker(photos, self.db, r2, photo_dir)

        def on_progress(current, total, message):
            self._cloud_dl_label.setText(message)
            self._cloud_dl_progress.setValue(current)

        def on_finished(downloaded, failed):
            self._cloud_dl_dialog.close()

            if self._cloud_dl_cancelled:
                QMessageBox.information(self, "Download Cancelled",
                    f"Download cancelled.\n{downloaded} photos were downloaded before cancellation.")
            else:
                QMessageBox.information(self, "Download Complete",
                    f"Successfully downloaded {downloaded} photos.\n{failed} failed.")

            # Refresh photo list
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()
            if self.photos:
                self.load_photo()

        self._cloud_dl_worker.progress.connect(on_progress)
        self._cloud_dl_worker.finished.connect(on_finished)
        self._cloud_dl_worker.start()

        self._cloud_dl_dialog.show()

    def _download_cloud_photos(self, cloud_photos: list):
        """Download thumbnails for cloud-only photos with progress dialog."""
        from r2_storage import R2Storage
        from pathlib import Path

        # Create progress dialog
        self._cloud_dl_dialog = QDialog(self)
        self._cloud_dl_dialog.setWindowTitle("Downloading from Cloud")
        self._cloud_dl_dialog.setMinimumWidth(400)
        self._cloud_dl_dialog.setModal(True)
        layout = QVBoxLayout(self._cloud_dl_dialog)

        self._cloud_dl_label = QLabel(f"Preparing to download {len(cloud_photos)} photos...")
        layout.addWidget(self._cloud_dl_label)

        self._cloud_dl_progress = QProgressBar()
        self._cloud_dl_progress.setMaximum(len(cloud_photos))
        self._cloud_dl_progress.setValue(0)
        layout.addWidget(self._cloud_dl_progress)

        cancel_btn = QPushButton("Cancel")
        layout.addWidget(cancel_btn)

        self._cloud_dl_cancelled = False

        def on_cancel():
            self._cloud_dl_cancelled = True
            if hasattr(self, '_cloud_dl_worker') and self._cloud_dl_worker.isRunning():
                self._cloud_dl_worker.cancel()
            self._cloud_dl_dialog.close()

        cancel_btn.clicked.connect(on_cancel)

        # Create worker thread
        class CloudPhotoDownloadWorker(QThread):
            progress = pyqtSignal(int, int, str)  # current, total, message
            finished = pyqtSignal(int, int)  # downloaded, failed

            def __init__(self, photos, db, r2, thumb_dir, photo_dir):
                super().__init__()
                self.photos = photos
                self.db = db
                self.r2 = r2
                self.thumb_dir = thumb_dir
                self.photo_dir = photo_dir  # Base photo directory
                self._cancelled = False

            def cancel(self):
                self._cancelled = True

            def _get_photo_dest_path(self, photo):
                """Get destination path for photo based on date_taken."""
                from datetime import datetime as dt
                date_taken = photo.get("date_taken", "")
                file_hash = photo.get("file_hash")
                original_name = photo.get("original_name", f"{file_hash}.jpg")

                # Parse date to get year/month for folder structure
                year, month = "Unknown", "Unknown"
                if date_taken:
                    try:
                        # Handle various date formats
                        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                            try:
                                parsed = dt.strptime(date_taken[:19], fmt)
                                year = str(parsed.year)
                                month = f"{parsed.month:02d}"
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass

                # Create destination folder
                dest_folder = self.photo_dir / year / month
                dest_folder.mkdir(parents=True, exist_ok=True)

                # Use original filename, but ensure uniqueness with hash if needed
                dest_path = dest_folder / original_name
                if dest_path.exists():
                    # File exists, use hash-based name
                    ext = Path(original_name).suffix or ".jpg"
                    dest_path = dest_folder / f"{file_hash}{ext}"

                return dest_path

            def _read_exif_datetime(self, image_path):
                """Read DateTimeOriginal from EXIF to verify/correct cloud timestamps."""
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        exif = img._getexif()
                        if exif:
                            # DateTimeOriginal (preferred) or DateTime (fallback)
                            dt_orig = exif.get(36867) or exif.get(306)
                            if dt_orig:
                                return str(dt_orig).replace(':', '-', 2).replace(' ', 'T')
                except Exception:
                    pass
                return None

            def run(self):
                downloaded = 0
                failed = 0
                total = len(self.photos)

                try:
                    for i, photo in enumerate(self.photos):
                        if self._cancelled:
                            break

                        try:
                            file_hash = photo.get("file_hash")
                            if not file_hash:
                                failed += 1
                                continue

                            self.progress.emit(i + 1, total, f"Downloading {i + 1} of {total}...")

                            # Add photo record to local database
                            photo_id = self.db.add_cloud_photo(photo)
                            if not photo_id:
                                # Already exists
                                continue

                            # Download full photo from R2
                            photo_r2_key = f"photos/{file_hash}.jpg"
                            photo_dest = self._get_photo_dest_path(photo)

                            photo_downloaded = False
                            if self.r2.download_photo(photo_r2_key, photo_dest):
                                # Update file_path to local path
                                self.db.update_file_path(photo_id, str(photo_dest))
                                photo_downloaded = True
                            else:
                                # Try .jpeg extension
                                photo_r2_key = f"photos/{file_hash}.jpeg"
                                if self.r2.download_photo(photo_r2_key, photo_dest):
                                    self.db.update_file_path(photo_id, str(photo_dest))
                                    photo_downloaded = True

                            # CRITICAL: Verify timestamp from EXIF after download
                            # Cloud timestamps from CuddeLink may be upload times, not capture times
                            if photo_downloaded and photo_dest.exists():
                                try:
                                    exif_date = self._read_exif_datetime(photo_dest)
                                    cloud_date = photo.get("date_taken", "")
                                    if exif_date and exif_date != cloud_date:
                                        # EXIF differs from cloud - use EXIF (thread-safe)
                                        self.db.update_date_taken(photo_id, exif_date)
                                except Exception:
                                    pass  # Don't fail download if EXIF check fails

                            # Download thumbnail from R2
                            thumb_r2_key = f"thumbnails/{file_hash}_thumb.jpg"
                            thumb_path = self.thumb_dir / f"{file_hash}_thumb.jpg"

                            if self.r2.download_photo(thumb_r2_key, thumb_path):
                                self.db.update_thumbnail_path(photo_id, str(thumb_path))

                            downloaded += 1

                        except Exception as e:
                            logger.error(f"Error downloading photo {photo.get('file_hash', 'unknown')}: {e}")
                            failed += 1

                except Exception as e:
                    logger.error(f"Critical error in download worker: {e}", exc_info=True)

                self.finished.emit(downloaded, failed)

        # Set up paths
        photo_dir = Path.home() / "TrailCamLibrary"
        photo_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir = photo_dir / ".thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        r2 = R2Storage()
        if not r2.is_configured():
            QMessageBox.warning(self, "Cloud Download", "R2 storage is not configured.")
            return

        self._cloud_dl_worker = CloudPhotoDownloadWorker(cloud_photos, self.db, r2, thumb_dir, photo_dir)

        def on_progress(current, total, message):
            self._cloud_dl_label.setText(message)
            self._cloud_dl_progress.setValue(current)

        def on_finished(downloaded, failed):
            self._cloud_dl_dialog.close()

            if self._cloud_dl_cancelled:
                QMessageBox.information(self, "Download Cancelled",
                    f"Download cancelled.\n{downloaded} photos were added before cancellation.")
            else:
                QMessageBox.information(self, "Download Complete",
                    f"Successfully added {downloaded} photos from cloud.")

            # Refresh photo list
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()
            if self.photos:
                self.load_photo()

        self._cloud_dl_worker.progress.connect(on_progress)
        self._cloud_dl_worker.finished.connect(on_finished)
        self._cloud_dl_worker.start()

        self._cloud_dl_dialog.show()

    def _handle_cloud_photo(self, photo):
        """Handle viewing a cloud-only photo - show thumbnail without prompting."""
        thumb_path = photo.get("thumbnail_path")

        # Show thumbnail
        self.scene.clear()
        self.box_items = []
        if thumb_path and os.path.exists(thumb_path):
            pix = QPixmap(thumb_path)
            if not pix.isNull():
                self.current_pixmap = pix
                self.scene.addPixmap(pix)
                self.view.zoom_fit()
                # Add "Cloud Photo" overlay text
                text_item = self.scene.addText("[Cloud Photo - Thumbnail Only]")
                text_item.setDefaultTextColor(Qt.GlobalColor.yellow)
                font = text_item.font()
                font.setPointSize(14)
                font.setBold(True)
                text_item.setFont(font)
                text_item.setPos(10, 10)
        else:
            self.current_pixmap = None
            self.scene.addText(f"Cloud photo - no thumbnail available")

        # Load boxes and metadata even for cloud photos
        self.current_boxes = []
        try:
            if photo.get("id"):
                self.current_boxes = self.db.get_boxes(photo["id"])
        except Exception:
            pass
        self._draw_boxes()
        self._update_box_tab_bar()

        # Load other metadata
        pid = photo.get("id")
        if pid:
            tags = self.db.get_tags(pid)
            self.tags_edit.blockSignals(True)
            self.tags_edit.setText(", ".join(tags))
            self.tags_edit.blockSignals(False)
            self.favorite_checkbox.blockSignals(True)
            self.favorite_checkbox.setChecked(bool(photo.get("favorite")))
            self.favorite_checkbox.blockSignals(False)

    def _download_single_cloud_photo(self, photo):
        """Download a single cloud photo and update local database."""
        from r2_storage import R2Storage
        from pathlib import Path
        from datetime import datetime as dt

        file_hash = photo.get("file_hash")
        if not file_hash:
            QMessageBox.warning(self, "Error", "Photo has no file hash.")
            return

        r2 = R2Storage()
        if not r2.is_configured():
            QMessageBox.warning(self, "Error", "R2 storage is not configured.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        try:
            # Determine destination path based on date
            date_taken = photo.get("date_taken", "")
            original_name = photo.get("original_name", f"{file_hash}.jpg")

            year, month = "Unknown", "Unknown"
            if date_taken:
                try:
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            parsed = dt.strptime(date_taken[:19], fmt)
                            year = str(parsed.year)
                            month = f"{parsed.month:02d}"
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass

            photo_dir = Path.home() / "TrailCamLibrary" / year / month
            photo_dir.mkdir(parents=True, exist_ok=True)

            # Use original filename, fall back to hash if exists
            dest_path = photo_dir / original_name
            if dest_path.exists():
                ext = Path(original_name).suffix or ".jpg"
                dest_path = photo_dir / f"{file_hash}{ext}"

            # Download from R2
            r2_key = f"photos/{file_hash}.jpg"
            if r2.download_photo(r2_key, dest_path):
                # Update database with local path
                self.db.update_file_path(photo["id"], str(dest_path))

                # Update the photo dict in memory
                photo["file_path"] = str(dest_path)

                # Refresh photos list to get updated data
                self.photos = self._sorted_photos(self.db.get_all_photos())

                QApplication.restoreOverrideCursor()
                QMessageBox.information(self, "Success", f"Photo downloaded to:\n{dest_path}")

                # Reload the photo with the new local path
                self.load_photo()
            else:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "Error", "Failed to download photo from cloud.")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Error", f"Download failed: {e}")

    def closeEvent(self, event):
        """Save settings and push to cloud before closing."""
        try:
            self._save_settings()

            # Create daily backup if one hasn't been made today
            try:
                if self.db.check_daily_backup():
                    print("[Shutdown] Daily backup created")
            except Exception as e:
                print(f"[Shutdown] Daily backup failed: {e}")

            # Do a final sync before closing to ensure all changes are pushed
            if hasattr(self, 'sync_manager'):
                # Force a sync regardless of debounce timer
                if self.sync_manager._check_network():
                    try:
                        client = self._get_supabase_client_silent()
                        if client and client.is_configured():
                            print("[Shutdown] Pushing final changes to cloud...")
                            counts = self.db.push_to_supabase(client)
                            total = sum(counts.values())
                            if total > 0:
                                print(f"[Shutdown] Pushed {total} items to cloud")
                    except Exception as e:
                        print(f"[Shutdown] Cloud push failed: {e}")
                elif self.sync_manager._pending:
                    # Offline with pending changes
                    QMessageBox.information(
                        self,
                        "Offline Changes Pending",
                        "Some label changes could not be synced to the cloud because the app was offline.\n\n"
                        "Your changes are saved locally and will sync automatically the next time "
                        "you open the app with an internet connection."
                    )
        finally:
            # Close database cleanly (clears crash flag) — always runs
            if hasattr(self, 'db') and self.db:
                self.db.close()

        event.accept()
        QApplication.instance().quit()

    def _reset_cloud_sync_preferences(self):
        """Reset the cloud sync and CuddeLink preferences so the app will ask again."""
        settings = QSettings("TrailCam", "Trainer")
        settings.remove("cloud_always_pull_on_open")
        settings.remove("cloud_always_push_on_close")
        settings.remove("cuddelink_always_download_on_open")
        QMessageBox.information(
            self, "Startup Preferences",
            "Startup preferences have been reset.\n\n"
            "The app will ask you about cloud sync and CuddeLink downloads next time."
        )

    def _show_ai_model_settings(self):
        """Show dialog for configuring AI model settings."""
        dialog = QDialog(self)
        dialog.setWindowTitle("AI Model Settings")
        dialog.setMinimumWidth(450)
        layout = QVBoxLayout(dialog)

        # --- Detector choice ---
        det_group = QGroupBox("Animal Detection")
        det_layout = QVBoxLayout(det_group)
        det_desc = QLabel("Which model detects animals and draws bounding boxes:")
        det_desc.setStyleSheet("color: #888;")
        det_layout.addWidget(det_desc)

        current_det = user_config.get_detector_choice()
        det_md = QRadioButton("MegaDetector v6 (standalone, faster)")
        det_sn = QRadioButton("SpeciesNet MegaDetector (built-in)")
        if current_det == "megadetector":
            det_md.setChecked(True)
        else:
            det_sn.setChecked(True)
        det_layout.addWidget(det_sn)
        det_layout.addWidget(det_md)
        layout.addWidget(det_group)

        # --- Classifier choice ---
        cls_group = QGroupBox("Species Classification")
        cls_layout = QVBoxLayout(cls_group)
        cls_desc = QLabel("Which model identifies species from detected animals:")
        cls_desc.setStyleSheet("color: #888;")
        cls_layout.addWidget(cls_desc)

        current_cls = user_config.get_classifier_choice()
        cls_sn = QRadioButton("SpeciesNet classifier (2,000+ species)")
        cls_custom = QRadioButton("Custom ONNX model (trained on your photos)")

        # Show custom model status
        custom_ready = hasattr(self, 'suggester') and self.suggester and self.suggester.ready
        if not custom_ready:
            cls_custom.setText("Custom ONNX model (not available)")
            cls_custom.setEnabled(False)

        if current_cls == "custom" and custom_ready:
            cls_custom.setChecked(True)
        else:
            cls_sn.setChecked(True)
        cls_layout.addWidget(cls_sn)
        cls_layout.addWidget(cls_custom)
        layout.addWidget(cls_group)

        # --- Geofencing ---
        geo_group = QGroupBox("SpeciesNet Geofencing")
        geo_layout = QVBoxLayout(geo_group)
        geo_desc = QLabel("Setting your state helps avoid suggesting species\nnot found in your region.")
        geo_desc.setStyleSheet("color: #888;")
        geo_layout.addWidget(geo_desc)

        state_combo = QComboBox()
        state_combo.addItem("(None - no geofencing)", "")
        current_state = user_config.get_speciesnet_state()
        for name, abbr in sorted(user_config.US_STATES.items()):
            state_combo.addItem(f"{name} ({abbr})", abbr)
            if abbr == current_state:
                state_combo.setCurrentIndex(state_combo.count() - 1)
        geo_layout.addWidget(state_combo)
        layout.addWidget(geo_group)

        # --- Status ---
        layout.addSpacing(5)
        sn_status = "Loaded" if self.speciesnet_wrapper.is_available else "Not loaded"
        md_status = "Available" if MEGADETECTOR_AVAILABLE else "Not available"
        status_label = QLabel(f"SpeciesNet: {sn_status}  |  MegaDetector: {md_status}")
        status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(status_label)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Save detector choice
            user_config.set_detector_choice("megadetector" if det_md.isChecked() else "speciesnet")
            # Save classifier choice
            user_config.set_classifier_choice("custom" if cls_custom.isChecked() else "speciesnet")
            # Save geofencing
            new_state = state_combo.currentData()
            user_config.set_speciesnet_state(new_state)
            self.speciesnet_wrapper.set_state(new_state)

    def _show_auto_archive_settings(self):
        """Show dialog for configuring auto-archive after sync."""
        settings = QSettings("TrailCam", "Trainer")

        dialog = QDialog(self)
        dialog.setWindowTitle("Auto-Archive Settings")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # Enable checkbox
        enable_cb = QCheckBox("Enable auto-archive after sync")
        enable_cb.setChecked(settings.value("auto_archive_enabled", False, type=bool))
        layout.addWidget(enable_cb)

        layout.addSpacing(10)

        # Description
        desc = QLabel("When enabled, photos will be archived after sync if they\n"
                      "don't match your selected species. Favorites and unlabeled\n"
                      "photos are always kept.")
        desc.setStyleSheet("color: #888;")
        layout.addWidget(desc)

        layout.addSpacing(10)

        # Preset selection
        preset_group = QGroupBox("Keep photos tagged as:")
        preset_layout = QVBoxLayout(preset_group)

        bucks_only_rb = QRadioButton("Bucks Only")
        custom_rb = QRadioButton("Custom Selection")

        current_preset = settings.value("auto_archive_preset", "custom")
        if current_preset == "bucks_only":
            bucks_only_rb.setChecked(True)
        else:
            custom_rb.setChecked(True)

        preset_layout.addWidget(bucks_only_rb)
        preset_layout.addWidget(custom_rb)

        # Custom species checkboxes
        species_widget = QWidget()
        species_layout = QVBoxLayout(species_widget)
        species_layout.setContentsMargins(20, 5, 0, 0)

        # Get all species from database
        all_species = ["Buck", "Deer", "Doe"]  # Common ones first
        if self.db:
            db_species = self.db.get_all_species_tags()
            for s in db_species:
                if s not in all_species:
                    all_species.append(s)

        # Get currently saved species
        saved_species_str = settings.value("auto_archive_keep_species", "Buck")
        saved_species = [s.strip() for s in saved_species_str.split(",") if s.strip()]

        species_checkboxes = {}
        for species in all_species[:15]:  # Limit to top 15 species
            cb = QCheckBox(species)
            cb.setChecked(species in saved_species)
            species_layout.addWidget(cb)
            species_checkboxes[species] = cb

        preset_layout.addWidget(species_widget)
        layout.addWidget(preset_group)

        # Enable/disable species checkboxes based on preset
        def update_species_enabled():
            is_custom = custom_rb.isChecked()
            species_widget.setEnabled(is_custom)

        bucks_only_rb.toggled.connect(update_species_enabled)
        custom_rb.toggled.connect(update_species_enabled)
        update_species_enabled()

        layout.addSpacing(10)

        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def save_settings():
            settings.setValue("auto_archive_enabled", enable_cb.isChecked())

            if bucks_only_rb.isChecked():
                settings.setValue("auto_archive_preset", "bucks_only")
                settings.setValue("auto_archive_keep_species", "Buck")
            else:
                settings.setValue("auto_archive_preset", "custom")
                selected = [s for s, cb in species_checkboxes.items() if cb.isChecked()]
                settings.setValue("auto_archive_keep_species", ",".join(selected))

            dialog.accept()
            QMessageBox.information(self, "Auto-Archive",
                                    "Auto-archive settings saved.\n\n"
                                    "Photos will be archived after the next sync operation.")

        save_btn.clicked.connect(save_settings)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def run_auto_archive(self):
        """Run auto-archive based on current settings. Called after sync."""
        settings = QSettings("TrailCam", "Trainer")

        if not settings.value("auto_archive_enabled", False, type=bool):
            return 0

        preset = settings.value("auto_archive_preset", "custom")
        if preset == "bucks_only":
            keep_species = ["Buck"]
        else:
            keep_str = settings.value("auto_archive_keep_species", "")
            keep_species = [s.strip() for s in keep_str.split(",") if s.strip()]

        if not keep_species:
            return 0

        photo_ids = self.db.get_photos_for_auto_archive(keep_species)
        if photo_ids:
            self.db.archive_photos(photo_ids)
            logger.info(f"Auto-archived {len(photo_ids)} photos (kept species: {keep_species})")
            return len(photo_ids)
        return 0

    def prune_missing_files(self):
        """Remove DB entries whose files no longer exist; report summary."""
        if not self.db:
            return
        photos = list(self.db.get_all_photos())
        missing = []
        for p in photos:
            path = p.get("file_path")
            if not path or not os.path.exists(path):
                missing.append(p["id"])
        if not missing:
            QMessageBox.information(self, "Prune Missing Files", "No missing files detected.")
            return
        removed = 0
        for pid in missing:
            try:
                self.db.delete_photo(pid)
                removed += 1
            except Exception:
                continue
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_photo_list()
        if self.index >= len(self.photos):
            self.index = max(0, len(self.photos) - 1)
        if self.photos:
            self.load_photo()
        QMessageBox.information(
            self,
            "Prune Missing Files",
            f"Removed {removed} photo entries with missing files."
        )

    def clear_all_ai_data(self):
        """Remove all AI boxes and pending suggestions across the entire library."""
        if not self.db:
            return
        resp = QMessageBox.question(
            self,
            "Clear AI Data",
            "This will remove all AI-generated boxes and pending AI suggestions across all photos. "
            "Human labels/boxes will be kept. Continue?",
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        photos = list(self.db.get_all_photos())
        cleared_boxes = 0
        for p in photos:
            pid = p.get("id")
            try:
                boxes = self.db.get_boxes(pid)
                kept = [b for b in boxes if not str(b.get("label", "")).startswith("ai_")]
                if len(kept) != len(boxes):
                    self.db.set_boxes(pid, kept)
                    cleared_boxes += len(boxes) - len(kept)
                # Drop AI suggestions if stored; treat as tags prefixed or a suggestion table if present
                if hasattr(self.db, "clear_suggestion"):
                    self.db.clear_suggestion(pid)
            except Exception:
                continue
        # Refresh current view
        self.photos = self._sorted_photos(self.db.get_all_photos())
        if self.index >= len(self.photos):
            self.index = max(0, len(self.photos) - 1)
        if self.photos:
            self.load_photo()
        QMessageBox.information(
            self,
            "Clear AI Data",
            f"Removed {cleared_boxes} AI box(es). AI suggestions cleared where supported."
        )

    def load_photo(self):
        if not self.photos:
            return
        self._loading_photo_data = True  # Prevent auto-advance during load
        # Ensure index is within bounds
        if self.index < 0 or self.index >= len(self.photos):
            self.index = max(0, min(self.index, len(self.photos) - 1))
            if not self.photos:
                return
        photo = self.photos[self.index]
        path = photo.get("file_path")
        # Keep list selection in sync with current index (but preserve multi-selection)
        try:
            self.photo_list_widget.blockSignals(True)
            selected_count = len(self.photo_list_widget.selectedItems())
            # Only clear selection if single-select mode (not multi-selecting)
            if selected_count <= 1:
                self.photo_list_widget.clearSelection()
                # Find item by stored index (UserRole data), not by row position
                target_item = None
                for row in range(self.photo_list_widget.count()):
                    item = self.photo_list_widget.item(row)
                    if item and item.data(Qt.ItemDataRole.UserRole) == self.index:
                        target_item = item
                        break
                if target_item:
                    target_item.setSelected(True)
                    self.photo_list_widget.scrollToItem(target_item)
        finally:
            self.photo_list_widget.blockSignals(False)
        # Show position within filtered results
        filtered_indices = self._get_filtered_indices()
        try:
            filtered_pos = filtered_indices.index(self.index) + 1
            filtered_total = len(filtered_indices)
        except ValueError:
            filtered_pos = self.index + 1
            filtered_total = len(self.photos)
        self.setWindowTitle(f"Trainer ({filtered_pos}/{filtered_total}) - {os.path.basename(path)}")

        # Handle cloud-only photos (file_path starts with cloud://)
        if path and path.startswith("cloud://"):
            # Offer to download full photo from cloud
            self._handle_cloud_photo(photo)
            return  # _handle_cloud_photo will call load_photo again after download
        else:
            pix = QPixmap(path)

        self.scene.clear()
        self.box_items = []
        if not pix.isNull():
            self.current_pixmap = pix
            self.scene.addPixmap(pix)
            self.view.zoom_fit()
        else:
            self.current_pixmap = None
            self.scene.addText(f"Cannot load image: {path}")
        # load boxes
        self.current_boxes = []
        try:
            if photo.get("id"):
                self.current_boxes = self.db.get_boxes(photo["id"])
        except Exception as e:
            logger.debug(f"Failed to load boxes for photo {photo.get('id')}: {e}")
            self.current_boxes = []
        self._draw_boxes()
        # Update box tab bar
        self._update_box_tab_bar()
        # In queue mode (species review), keep at fit-to-window for full context
        # (zoom_fit already called above at line 2274)
        # Only auto-advance if photo not already reviewed (user may be clicking back to view)
        if self.ai_review_mode and not self._photo_has_ai_boxes(self.current_boxes):
            if photo.get("id") not in self.ai_reviewed_photos:
                self._advance_ai_review()

        # Attempt to auto-fill camera info/date if missing
        self._maybe_autofill_exif(photo)
        self._maybe_autofill_stamp(photo)

        # Load metadata
        pid = photo["id"]
        tags = self.db.get_tags(pid)
        # Also show pending label suggestions with visual indicator
        suggestions = self.db.get_suggestions_for_photo(pid)
        pending_names = [s["tag_name"] for s in suggestions if s["status"] == "pending" and s["tag_name"] not in tags]
        display_tags = list(tags) + [f"{name}?" for name in pending_names]
        self.tags_edit.blockSignals(True)
        self.tags_edit.setText(", ".join(display_tags))
        self.tags_edit.blockSignals(False)
        if pending_names:
            self.tags_edit.setStyleSheet("color: #e6d54a;")
        else:
            self.tags_edit.setStyleSheet("")

        # Update favorite checkbox
        self.favorite_checkbox.blockSignals(True)
        self.favorite_checkbox.setChecked(bool(photo.get("favorite")))
        self.favorite_checkbox.blockSignals(False)

        deer = self.db.get_deer_metadata(pid)
        self._populate_deer_id_dropdown()
        self.deer_id_edit.setCurrentText(deer.get("deer_id") or "")
        self.age_combo.setCurrentText(deer.get("age_class") or "")
        self._set_int_field(self.left_min, deer.get("left_points_min"))
        self.left_uncertain.setChecked(deer.get("left_points_uncertain", False))
        self._set_int_field(self.right_min, deer.get("right_points_min"))
        self.right_uncertain.setChecked(deer.get("right_points_uncertain", False))
        self._set_int_field(self.left_ab_min, deer.get("left_ab_points_min"))
        self.left_ab_unc.setChecked(deer.get("left_ab_points_uncertain", False))
        self._set_int_field(self.right_ab_min, deer.get("right_ab_points_min"))
        self.right_ab_unc.setChecked(deer.get("right_ab_points_uncertain", False))
        self._set_int_field(self.ab_min, deer.get("abnormal_points_min"))
        self._set_int_field(self.ab_max, deer.get("abnormal_points_max"))

        # species: use box species when boxes exist, else use stored tags (suggestions require user approval)
        species_options = set(SPECIES_OPTIONS)
        try:
            species_options.update(self.db.list_custom_species())
        except Exception as e:
            logger.debug(f"Failed to list custom species: {e}")
        has_subject_boxes = any(not self._is_head_box(b) for b in self.current_boxes) if self.current_boxes else False
        current_species = [t for t in tags if t in species_options]
        if has_subject_boxes:
            species = ""
            if 0 <= self.current_box_index < len(self.current_boxes):
                species = self.current_boxes[self.current_box_index].get("species") or ""
        else:
            # Show first species in combo, all in label
            species = current_species[0] if current_species else ""
        self.species_combo.blockSignals(True)
        self.species_combo.setCurrentText(species)
        self.species_combo.blockSignals(False)
        # Track original saved species to know if user added NEW tag
        self._original_saved_species = set(current_species)
        self._update_current_species_label(current_species)

        # crude sex inference from tags
        sex = ""
        for t in tags:
            if t.lower() in ("buck", "doe"):
                sex = t.capitalize()
        if sex not in ("Buck", "Doe", "Unknown"):
            sex = "Unknown"
        self._set_sex(sex)

        # Set camera location
        cam_loc = photo.get("camera_location") or ""
        self.camera_combo.setCurrentText(cam_loc)
        self.char_edit.setPlainText(photo.get("key_characteristics") or "")
        self._refresh_char_tags()
        self.notes_edit.setPlainText(photo.get("notes") or "")
        # Apply profile autofill for this buck/season if present
        self._apply_buck_profile_to_ui()
        # Default species/sex when a buck ID exists
        if deer.get("deer_id"):
            if not (self.species_combo.currentText().strip()):
                self.species_combo.blockSignals(True)
                self.species_combo.setCurrentText("Deer")
                self.species_combo.blockSignals(False)
            current_sex = self._get_sex()
            if current_sex == "Unknown":
                self._set_sex("Buck")
        self._load_previews()
        self._update_suggestion_display(photo)
        self._update_sex_suggestion_display(photo)
        # Load additional buck (second deer)
        extras = self.db.get_additional_deer(pid)
        if extras:
            ex = extras[0]
            self.add_buck_toggle.setChecked(True)
            self.add_deer_id_edit.setCurrentText(ex.get("deer_id") or "")
            self.add_age_combo.setCurrentText(ex.get("age_class") or "")
            self._set_int_field(self.add_left_min, ex.get("left_points_min"))
            self._set_int_field(self.add_right_min, ex.get("right_points_min"))
            self.add_left_uncertain.setChecked(ex.get("left_points_uncertain", False))
            self.add_right_uncertain.setChecked(ex.get("right_points_uncertain", False))
        else:
            self.add_buck_toggle.setChecked(False)
            self.add_deer_id_edit.setCurrentText("")
            self.add_age_combo.setCurrentText("")
            self.add_left_min.clear()
            self.add_right_min.clear()
            self.add_left_uncertain.setChecked(False)
            self.add_right_uncertain.setChecked(False)

        # Update queue UI if in queue mode
        if self.queue_mode:
            self._update_queue_ui()

        # Update mark button state for current photo
        self._update_mark_button_state()

        # Preload thumbnails for nearby photos in the filtered list
        self._preload_nearby_thumbnails()

        self._loading_photo_data = False  # Done loading, allow auto-advance

    def _preload_nearby_thumbnails(self, buffer: int = 10):
        """Preload thumbnails for photos adjacent to current one in filtered list.

        This ensures smooth scrolling by generating thumbnails for nearby photos
        based on the current sort/filter order.
        """
        from image_processor import create_thumbnail

        filtered = self._filtered_photos()
        if not filtered:
            return

        # Find current position in filtered list
        current_pos = None
        for i, (idx, _) in enumerate(filtered):
            if idx == self.index:
                current_pos = i
                break

        if current_pos is None:
            return

        # Get range of photos to preload (buffer before and after)
        start = max(0, current_pos - buffer)
        end = min(len(filtered), current_pos + buffer + 1)

        # Generate thumbnails for nearby photos (skips existing ones)
        for i in range(start, end):
            if i == current_pos:
                continue  # Skip current photo
            _, photo = filtered[i]
            path = photo.get("file_path")
            if path and os.path.exists(path):
                create_thumbnail(path, file_hash=photo.get("file_hash"))  # Will skip if already exists

    def _get_current_username(self):
        """Get current username from config (cached per session)."""
        if not hasattr(self, '_cached_username'):
            from user_config import get_username
            self._cached_username = get_username() or "unknown"
        return self._cached_username

    def _clear_cached_username(self):
        """Clear cached username so it re-resolves from config/session."""
        if hasattr(self, '_cached_username'):
            delattr(self, '_cached_username')

    def _save_tags_role_aware(self, photo_id, tags):
        """Save tags with role awareness: owners write tags directly, members write suggestions."""
        username = self._get_current_username()
        role = self.db.get_user_role_for_photo(photo_id, username)

        if role in ('owner', 'editor'):
            # Owner: write directly to tags as before
            self.db.update_photo_tags(photo_id=photo_id, tags=tags)
        else:
            # Member: calculate new tags vs existing, add as suggestions
            existing_tags = set(self.db.get_tags(photo_id))
            new_tags = set(tags) - existing_tags
            for tag_name in new_tags:
                self.db.add_label_suggestion(photo_id, tag_name, username)

    def _add_tag_role_aware(self, photo_id, tag_name):
        """Add a single tag with role awareness."""
        username = self._get_current_username()
        role = self.db.get_user_role_for_photo(photo_id, username)

        if role in ('owner', 'editor'):
            self.db.add_tag(photo_id, tag_name)
        else:
            self.db.add_label_suggestion(photo_id, tag_name, username)

    def save_current(self):
        photo = self._current_photo()
        if not photo:
            return
        # Prevent saving during photo load - UI fields may be in transitional state
        if getattr(self, '_loading_photo_data', False):
            return
        pid = photo["id"]
        species = self.species_combo.currentText().strip()
        if species.lower() in SEX_TAGS:
            species = ""
        sex = self._get_sex()
        deer_id = self.deer_id_edit.currentText().strip()
        age = self.age_combo.currentText().strip()
        # Get tags from UI field, or from database if field is empty/hidden
        tags_text = self.tags_edit.text().strip()
        if tags_text:
            tags = [t.strip() for t in tags_text.split(",") if t.strip()]
        else:
            # Field is empty (likely hidden in simple mode) - get from database
            tags = self.db.get_tags(pid)
        # Remove prior sex tags so we don't end up with both buck and doe
        tags = [t for t in tags if t.lower() not in SEX_TAGS]
        # Build species set for validation
        species_set = set(SPECIES_OPTIONS)
        try:
            species_set.update(self.db.list_custom_species())
        except Exception:
            pass

        # For photos WITHOUT boxes, species selection replaces the current species tag
        # For photos WITH boxes, tags are derived from box species (soft rollup)
        has_subject_boxes = any(not self._is_head_box(b) for b in self.current_boxes) if self.current_boxes else False

        if not has_subject_boxes:
            # No boxes - this is photo-level species, replace all species tags with current selection
            # BUT preserve Verification tag unless explicitly changing to something else
            had_verification = "Verification" in tags
            non_species_tags = [t for t in tags if t not in species_set]
            tags = non_species_tags
            if species and (species in species_set or len(species) >= 3):
                tags.append(species)
            elif had_verification and not species:
                # Preserve Verification if no new species selected
                tags.append("Verification")
        else:
            # Has boxes - box species is the source of truth
            subject_boxes = [b for b in (self.current_boxes or []) if not self._is_head_box(b)]
            if species:
                if len(subject_boxes) == 1:
                    subject_boxes[0]["species"] = species
                elif self.queue_mode:
                    for b in subject_boxes:
                        if not b.get("species"):
                            b["species"] = species
            # Persist box changes before deriving tags
            try:
                self.db.set_boxes(pid, self.current_boxes)
            except Exception as exc:
                logger.error(f"Box save failed: {exc}")
            # Derive tags from boxes (soft rollup) and remove photo-level species tags
            non_species_tags = [t for t in tags if t not in species_set and t not in ("Empty", "Verification")]
            box_species = [b.get("species") for b in subject_boxes if b.get("species") and b.get("species") != "Unknown"]
            tags = non_species_tags + sorted(set(box_species))

        # Add sex tag if applicable
        if sex.lower() in ("buck", "doe") and sex not in tags:
            tags.append(sex)
        # If a real species exists, remove Empty/Unknown tags
        real_species = [t for t in tags if t in species_set and t not in ("Empty", "Unknown")]
        if real_species:
            tags = [t for t in tags if t not in ("Empty", "Unknown")]
        # If Empty is in tags, it should be the ONLY tag (clear everything else)
        if "Empty" in tags:
            tags = ["Empty"]
            if self.current_boxes:
                self.current_boxes = []
                try:
                    self.db.set_boxes(pid, [])
                except Exception as exc:
                    logger.error(f"Box clear failed: {exc}")
        self._save_tags_role_aware(pid, tags)
        # Update UI to reflect tag changes (block signals to prevent re-triggering save)
        self.tags_edit.blockSignals(True)
        self.tags_edit.setText(", ".join(tags))
        self.tags_edit.blockSignals(False)
        # If user tags as Empty or Unknown, clear species from all boxes (removes AI suggestions)
        if "Empty" in tags or "Unknown" in tags:
            for box in self.current_boxes:
                box["species"] = None
                box["species_conf"] = None
            self._draw_boxes()  # Refresh overlay
        self.db.set_deer_metadata(
            photo_id=pid,
            deer_id=deer_id or None,
            age_class=age or None,
            left_points_min=self._to_int(self.left_min.text()),
            right_points_min=self._to_int(self.right_min.text()),
            left_points_uncertain=self.left_uncertain.isChecked(),
            right_points_uncertain=self.right_uncertain.isChecked(),
            left_ab_points_min=self._to_int(self.left_ab_min.text()),
            right_ab_points_min=self._to_int(self.right_ab_min.text()),
            left_ab_points_uncertain=self.left_ab_unc.isChecked(),
            right_ab_points_uncertain=self.right_ab_unc.isChecked(),
            abnormal_points_min=self._to_int(self.ab_min.text()),
            abnormal_points_max=self._to_int(self.ab_max.text()),
        )
        self.db.update_photo_attributes(
            photo_id=pid,
            camera_location=self.camera_combo.currentText().strip(),
            key_characteristics=self._normalize_tags_text(self.char_edit.toPlainText()),
        )
        self.db.set_notes(pid, self.notes_edit.toPlainText())
        # Save boxes
        try:
            self.db.set_boxes(pid, self.current_boxes)
        except Exception as exc:
            logger.error(f"Box save failed: {exc}")
        # Save second buck if toggled
        extras = []
        if self.add_buck_toggle.isChecked():
            add_id = self.add_deer_id_edit.currentText().strip()
            if add_id:
                extras.append({
                    "deer_id": add_id,
                    "age_class": self.add_age_combo.currentText().strip() or None,
                    "left_points_min": self._to_int(self.add_left_min.text()),
                    "right_points_min": self._to_int(self.add_right_min.text()),
                    "left_points_uncertain": self.add_left_uncertain.isChecked(),
                    "right_points_uncertain": self.add_right_uncertain.isChecked(),
                    "left_ab_points_min": None,
                    "right_ab_points_min": None,
                    "left_ab_points_uncertain": False,
                    "right_ab_points_uncertain": False,
                    "abnormal_points_min": None,
                    "abnormal_points_max": None,
                })
        self.db.set_additional_deer(pid, extras)
        # Only clear suggestion if a NEW species tag was added (not already saved)
        original_species = getattr(self, "_original_saved_species", set())
        is_new_species = species and species not in original_species
        if is_new_species:
            # Official tag applied; clear suggestion
            self.db.set_suggested_tag(pid, None, None)
            # Bump to session recent species list
            if species in self._session_recent_species:
                self._session_recent_species.remove(species)
            self._session_recent_species.insert(0, species)
            self._session_recent_species = self._session_recent_species[:20]  # Keep max 20
            # Note: Custom species addition disabled - using fixed species list
            # Update original species so subsequent saves don't re-trigger
            self._original_saved_species.add(species)
        # Refresh camera location dropdown if new location was added
        cam_loc = self.camera_combo.currentText().strip()
        if cam_loc and self.camera_combo.findText(cam_loc) == -1:
            self._populate_camera_locations()
        self._update_recent_species_buttons()
        # refresh in-memory photo metadata
        photo.update(self.db.get_photo_by_id(pid) or {})
        self._bump_recent_buck(deer_id)
        self._update_recent_buck_buttons()
        self._update_photo_list_item(self.index)
        # Queue for cloud sync
        if hasattr(self, 'sync_manager'):
            self.sync_manager.queue_change()
        # If in review mode and resolved, remove from queue and advance
        if self.in_review_mode and self._current_photo_resolved(pid):
            # Remove photo from list at current index
            try:
                if 0 <= self.index < len(self.photos):
                    self.photos.pop(self.index)
            except Exception as e:
                logger.warning(f"Failed to remove photo from list at index {self.index}: {e}")
            # Adjust index if we were at the end
            if self.index >= len(self.photos):
                self.index = max(0, len(self.photos) - 1)
            # Rebuild the photo list to refresh UserRole indices
            self._populate_photo_list()
            self._exit_review_queue_if_done()
            if self.photos:
                self.load_photo()

    def retrain_model(self):
        """Invoke the ONNX export script to refresh the species model."""
        script = Path(__file__).resolve().parent / "export_species_onnx.py"
        db_path = Path.home() / ".trailcam" / "trailcam.db"
        out_path = Path(__file__).resolve().parent.parent / "models" / "species.onnx"
        labels_path = Path(__file__).resolve().parent.parent / "models" / "labels.txt"
        cmd = [
            sys.executable,
            str(script),
            "--db",
            str(db_path),
            "--out",
            str(out_path),
            "--labels",
            str(labels_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            msg = proc.stdout.strip() or "Model retrained."
            QMessageBox.information(self, "Retrain Model", msg)
            # Reload suggester so the new model is picked up.
            # Note: training script already outputs correct labels.txt
            self.suggester = CombinedSuggester()
        except subprocess.CalledProcessError as exc:
            err = exc.stderr or exc.stdout or str(exc)
            QMessageBox.warning(self, "Retrain Model Failed", err[:1000])

    def save_and_next(self):
        self.save_current()
        self.next_photo()

    def schedule_save(self):
        """Debounced autosave."""
        self.save_timer.start(300)

    def _on_species_changed(self):
        """Handle species combo changes - save immediately to ensure changes persist."""
        if getattr(self, '_in_species_changed', False):
            return
        self._in_species_changed = True
        try:
            # Save immediately instead of debouncing to ensure tag changes are saved
            self.save_timer.stop()

            # Update box species and tab name
            species = self.species_combo.currentText().strip()
            # Don't treat sex tags as species
            if species.lower() in SEX_TAGS:
                return
            # Debug: trace species changes in queue mode
            if self.queue_mode:
                has_boxes = hasattr(self, "current_boxes") and bool(self.current_boxes)
                print(f"[DEBUG] _on_species_changed: species='{species}', queue_mode={self.queue_mode}, has_boxes={has_boxes}")
            if hasattr(self, "current_boxes") and self.current_boxes:
                if self.current_box_index < len(self.current_boxes):
                    self.current_boxes[self.current_box_index]["species"] = species
                    # Clear sex if species is not Deer (sex only applies to deer)
                    if species and species.lower() != "deer":
                        self.current_boxes[self.current_box_index]["sex"] = None
                        self.current_boxes[self.current_box_index]["sex_conf"] = None
                    self._update_box_tab_name(self.current_box_index)

            self.save_current()

            # In queue mode, auto-advance when ALL boxes are labeled (not just current box)
            if self.queue_mode and not self._loading_photo_data:
                # Don't auto-advance if "Multiple species" is checked
                if hasattr(self, 'queue_multi_species') and self.queue_multi_species.isChecked():
                    print("[DEBUG] queue advance blocked: multi_species checked")
                    return
                species = self.species_combo.currentText().strip()
                if species:  # Only check if a species was actually selected
                    # Only advance when ALL boxes are labeled
                    all_labeled = self._all_boxes_labeled()
                    print(f"[DEBUG] queue advance check: species='{species}', all_labeled={all_labeled}")
                    if not all_labeled:
                        print("[DEBUG] queue advance blocked: not all boxes labeled")
                        return  # Still have unlabeled boxes
                    current_pid = None
                    if self.photos and self.index < len(self.photos):
                        current_pid = self.photos[self.index].get("id")
                    if current_pid:
                        # Check if already reviewed (prevents double-advance from multiple signals)
                        if current_pid in self.queue_reviewed:
                            return
                        # Mark as reviewed for green highlighting
                        self.queue_reviewed.add(current_pid)
                        # Clear AI suggestion since user made a decision
                        self.db.set_suggested_tag(current_pid, None, None)
                        self._mark_current_list_item_reviewed()
                    self._queue_advance()
        finally:
            self._in_species_changed = False

    def _get_filtered_indices(self) -> List[int]:
        """Get list of photo indices that pass current filters."""
        return [idx for idx, _ in self._filtered_photos()]

    def _current_photo(self):
        """Return current photo dict or None if out of bounds."""
        if not self.photos or self.index < 0 or self.index >= len(self.photos):
            return None
        return self.photos[self.index]

    def prev_photo(self):
        if not self.photos:
            return
        # Always save current box data before navigating
        self._save_current_box_data()
        if self.save_timer.isActive():
            self.save_timer.stop()
        self.save_current()
        # Navigate within filtered photos only
        filtered_indices = self._get_filtered_indices()
        if not filtered_indices:
            return
        try:
            current_pos = filtered_indices.index(self.index)
            new_pos = (current_pos - 1) % len(filtered_indices)
            self.index = filtered_indices[new_pos]
        except ValueError:
            # Current index not in filtered list, go to first filtered photo
            self.index = filtered_indices[0]
        self.load_photo()

    def next_photo(self):
        if not self.photos:
            return
        # Always save current box data before navigating
        self._save_current_box_data()
        if self.save_timer.isActive():
            self.save_timer.stop()
        self.save_current()
        # Navigate within filtered photos only
        filtered_indices = self._get_filtered_indices()
        if not filtered_indices:
            return
        try:
            current_pos = filtered_indices.index(self.index)
            new_pos = (current_pos + 1) % len(filtered_indices)
            self.index = filtered_indices[new_pos]
        except ValueError:
            # Current index not in filtered list, go to first filtered photo
            self.index = filtered_indices[0]
        self.load_photo()

    def keyPressEvent(self, event):
        """Allow Up/Down to navigate photos even when focus is in form fields (except multiline notes)."""
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            fw = self.focusWidget()
            # Don't steal arrows from multiline text areas where caret movement is expected.
            if not isinstance(fw, (QTextEdit, QTextBrowser)):
                if event.key() == Qt.Key.Key_Up:
                    self.prev_photo()
                else:
                    self.next_photo()
                event.accept()
                return
        super().keyPressEvent(event)

    def _setup_nav_shortcuts(self):
        """Install Up/Down shortcuts that work even when form fields have focus."""
        up = QAction(self)
        up.setShortcut(Qt.Key.Key_Up)
        up.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        up.triggered.connect(lambda: self._handle_nav(-1))
        self.addAction(up)

        down = QAction(self)
        down.setShortcut(Qt.Key.Key_Down)
        down.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        down.triggered.connect(lambda: self._handle_nav(1))
        self.addAction(down)

        # Queue mode shortcuts (A=Accept, R=Reject, Escape=Exit)
        accept_action = QAction(self)
        accept_action.setShortcut(Qt.Key.Key_A)
        accept_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        accept_action.triggered.connect(self._queue_accept_if_active)
        self.addAction(accept_action)

        reject_action = QAction(self)
        reject_action.setShortcut(Qt.Key.Key_R)
        reject_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        reject_action.triggered.connect(self._queue_reject_if_active)
        self.addAction(reject_action)

        exit_queue_action = QAction(self)
        exit_queue_action.setShortcut(Qt.Key.Key_Escape)
        exit_queue_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        exit_queue_action.triggered.connect(self._exit_queue_if_active)
        self.addAction(exit_queue_action)

        # Mark for compare shortcut (M key)
        mark_action = QAction(self)
        mark_action.setShortcut(Qt.Key.Key_M)
        mark_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        mark_action.triggered.connect(self._mark_if_not_typing)
        self.addAction(mark_action)

    def _mark_if_not_typing(self):
        """Toggle mark for compare if not typing in a field."""
        if not self._is_typing_in_field():
            self.toggle_mark_for_compare()

    def _is_typing_in_field(self) -> bool:
        """Check if user is typing in a text field (should ignore hotkeys)."""
        fw = self.focusWidget()
        if isinstance(fw, (QLineEdit, QTextEdit, QComboBox)):
            return True
        # Check if focus is in a combo box's line edit (editable combo)
        if fw and fw.parent() and isinstance(fw.parent(), QComboBox):
            return True
        return False

    def _queue_accept_if_active(self):
        """Accept queue item if in queue mode and focus is not in a text field."""
        if not self.queue_mode:
            return
        if self._is_typing_in_field():
            return  # Let typing work normally
        self._queue_accept()

    def _queue_reject_if_active(self):
        """Reject queue item if in queue mode and focus is not in a text field."""
        if not self.queue_mode:
            return
        if self._is_typing_in_field():
            return
        self._queue_reject()

    def _exit_queue_if_active(self):
        """Exit queue if in queue mode."""
        if self.queue_mode:
            self._exit_queue_mode()

    def _handle_nav(self, delta: int):
        fw = self.focusWidget()
        # Don't override arrow keys inside multiline text editors where caret movement matters.
        if isinstance(fw, (QTextEdit, QTextBrowser)):
            return
        if delta < 0:
            self.prev_photo()
        else:
            self.next_photo()

    # ====== Boxes / annotations ======
    def _set_box_mode(self, mode: Optional[str]):
        self.box_mode = mode
        self.box_subject_btn.setChecked(mode == "subject")
        self.box_head_btn.setChecked(mode == "deer_head")
        self.view.set_draw_mode(mode)

    def _start_add_subject(self):
        """Enable draw mode to add a new subject box."""
        self._set_box_mode("subject")
        # Show a brief status message
        self.statusBar().showMessage("Draw a box around the subject on the image", 5000)

    def _clear_boxes(self):
        self.current_boxes = []
        if self.scene:
            for it in self.box_items:
                self.scene.removeItem(it)
        self.box_items = []
        self._persist_boxes()

    def _clear_ai_boxes(self):
        self.current_boxes = [b for b in self.current_boxes if not str(b.get("label", "")).startswith("ai_")]
        self._draw_boxes()
        self._persist_boxes()
        if self.ai_review_mode and not self._photo_has_ai_boxes(self.current_boxes):
            self._advance_ai_review()

    def _clear_all_labels(self):
        """Clear all labels (species, sex, age, deer_id, points) from all subject boxes and photo tags."""
        # Clear labels from all subject boxes (if any)
        for box in self.current_boxes:
            # Skip head boxes
            if self._is_head_box(box):
                continue
            box["species"] = ""
            box["sex"] = ""
            box["deer_id"] = ""
            box["age_class"] = ""
            box["left_points_min"] = None
            box["right_points_min"] = None
            box["left_points_uncertain"] = False
            box["right_points_uncertain"] = False

        # Persist changes
        self._persist_boxes()

        # Clear the form fields
        self.species_combo.setCurrentText("")
        for btn in self.sex_buttons.values():
            btn.setChecked(False)
        self.deer_id_edit.setCurrentText("")
        self.age_combo.setCurrentText("")
        self.left_min.clear()
        self.right_min.clear()
        self.left_uncertain.setChecked(False)
        self.right_uncertain.setChecked(False)
        self.tags_edit.clear()  # Clear tags field
        self.current_species_label.setText("")  # Clear species label

        # Also clear photo-level data from database
        if self.photos and self.index < len(self.photos):
            photo = self.photos[self.index]
            pid = photo.get("id")
            if pid:
                self.db.update_photo_tags(pid, [])  # Clear all tags
                self.db.set_suggested_tag(pid, None, None)
                # Also clear in-memory photo data
                photo["suggested_tag"] = None
                photo["suggested_tag_confidence"] = None
                photo["suggested_sex"] = None
                photo["suggested_sex_confidence"] = None
                # Clear deer metadata
                self.db.set_deer_metadata(
                    pid,
                    deer_id=None,
                    age_class=None,
                    left_points_min=None,
                    right_points_min=None,
                    left_points_uncertain=False,
                    right_points_uncertain=False,
                    left_ab_points_min=None,
                    right_ab_points_min=None,
                    left_ab_points_uncertain=False,
                    right_ab_points_uncertain=False,
                    abnormal_points_min=None,
                    abnormal_points_max=None
                )
                # Clear additional deer entries
                self.db.set_additional_deer(pid, [])

        # Update UI
        self._update_box_tab_bar()
        self._draw_boxes()
        # Update current item in photo list without rebuilding entire list
        current_item = self.photo_list_widget.currentItem()
        if current_item and self.photos and self.index < len(self.photos):
            photo = self.photos[self.index]
            # Update item text to show cleared state
            filename = os.path.basename(photo.get("file_path", ""))
            current_item.setText(filename)  # Just show filename, no labels
        self.statusBar().showMessage("Cleared all labels", 3000)

    # --- Detection helpers ---
    def _get_megadetector(self):
        """Load and cache MegaDetector v6 model."""
        if hasattr(self, "_megadetector") and self._megadetector is not None:
            return self._megadetector

        try:
            from ai_detection import MegaDetectorV6
            self._megadetector = MegaDetectorV6()
            self._megadetector_available = self._megadetector.is_available
            if not self._megadetector_available:
                self._megadetector_error = self._megadetector.error_message
                self._megadetector = None
                return None
            return self._megadetector
        except Exception as e:
            logger.error(f"Failed to load MegaDetector v6: {e}")
            self._megadetector_available = False
            self._megadetector_error = str(e)
            self._megadetector = None
            return None

    def _detect_boxes_megadetector(self, path: str, conf_thresh: float = 0.2):
        """Detect animals/people/vehicles using MegaDetector v6."""
        model = self._get_megadetector()
        if model is None:
            return []
        try:
            detections = model.detect(path)
            cat_map = {"animal": "ai_animal", "person": "ai_person", "vehicle": "ai_vehicle"}
            boxes = []
            for det in detections:
                if det.confidence < conf_thresh:
                    continue
                x, y, w, h = det.bbox
                boxes.append({
                    'label': cat_map.get(det.category, 'ai_subject'),
                    'x1': x,
                    'y1': y,
                    'x2': x + w,
                    'y2': y + h,
                    'confidence': det.confidence
                })
            return boxes
        except Exception as e:
            logger.warning(f"MegaDetector failed on {path}: {e}")
            return []

    def _detect_deer_heads_for_path(self, path: str, detector=None, names=None, conf_thresh: float = 0.25):
        """Detect deer heads using custom YOLO model (separate from MegaDetector)."""
        if detector is None:
            return []
        boxes = []
        try:
            res = detector(path, imgsz=640, device="cpu", verbose=False)[0]
            preds = res.boxes
            if preds is not None and preds.xyxy is not None and len(preds) > 0:
                w, h = res.orig_shape[1], res.orig_shape[0]
                nmap = names or getattr(res, "names", {}) or {}
                for xyxy, cls, conf in zip(preds.xyxy.cpu().numpy(), preds.cls.cpu().numpy(), preds.conf.cpu().numpy()):
                    if conf < conf_thresh:
                        continue
                    x1, y1, x2, y2 = xyxy.tolist()
                    x1 = max(0.0, min(w, x1)); x2 = max(0.0, min(w, x2))
                    y1 = max(0.0, min(h, y1)); y2 = max(0.0, min(h, y2))
                    if x2 - x1 < 6 or y2 - y1 < 6:
                        continue
                    cls_name = nmap.get(int(cls), "").lower()
                    # Only keep deer head detections
                    if "deer" in cls_name and "head" in cls_name:
                        boxes.append({
                            "label": "ai_deer_head",
                            "x1": x1 / w,
                            "y1": y1 / h,
                            "x2": x2 / w,
                            "y2": y2 / h,
                            "confidence": float(conf),
                        })
        except Exception:
            pass
        return boxes

    def _detect_boxes_for_path(self, path: str, detector=None, names=None, conf_thresh: float = 0.2):
        """Run MegaDetector for animal detection. Deer heads are added separately after species confirmation."""
        # Primary detection: MegaDetector for animals/people/vehicles
        boxes = self._detect_boxes_megadetector(path, conf_thresh=conf_thresh)
        # NOTE: Deer head detection is NOT done here - it's only run after species is confirmed as Deer
        return boxes

    def _add_deer_head_boxes(self, photo: dict, detector=None, names=None):
        """Add deer head boxes to a photo that's confirmed as Deer. Call after species classification."""
        if detector is None:
            return
        pid = photo.get("id")
        path = photo.get("file_path")
        if not pid or not path or not os.path.exists(path):
            return
        # Get existing boxes
        try:
            boxes = self.db.get_boxes(pid)
        except Exception:
            boxes = []
        # Check if deer heads already exist
        if any(b.get("label") in ("deer_head", "ai_deer_head") for b in boxes):
            return  # Already has deer head boxes
        # Detect deer heads
        deer_heads = self._detect_deer_heads_for_path(path, detector=detector, names=names, conf_thresh=0.25)
        if deer_heads:
            boxes.extend(deer_heads)
            try:
                self.db.set_boxes(pid, boxes)
            except Exception:
                pass

    def _toggle_boxes_visible(self, hidden: bool):
        self.boxes_hidden = hidden
        self.box_toggle_btn.setText("Show Boxes" if hidden else "Hide Boxes")
        self._draw_boxes()

    def _toggle_antler_section(self, checked: bool):
        """Show/hide the antler/age fields."""
        if hasattr(self, "antler_container"):
            self.antler_container.setVisible(checked)
        if hasattr(self, "antler_toggle"):
            self.antler_toggle.setText("Antler details ▼" if checked else "Antler details ▸")

    def _on_box_tab_switched(self, tab_index: int):
        """Handle when user switches to a different box tab.

        Saves current box data, then loads the new box's data into form fields.
        Tab index is converted to actual box index (head boxes don't have tabs).
        """
        if tab_index < 0 or not hasattr(self, "current_boxes"):
            return

        # Handle "Photo" tab case (no boxes) - load photo-level tags into species combo
        if not self.current_boxes:
            self.current_box_index = 0
            # Load photo-level species tag into combo
            if self.photos and self.index < len(self.photos):
                photo = self.photos[self.index]
                tags = self.db.get_tags(photo.get("id"))
                # Find species tag (from SPECIES_OPTIONS)
                species_tag = ""
                for t in tags:
                    if t in SPECIES_OPTIONS:
                        species_tag = t
                        break
                self.species_combo.blockSignals(True)
                self.species_combo.setCurrentText(species_tag)
                self.species_combo.blockSignals(False)
            return

        # Convert tab index to actual box index
        box_index = self._tab_index_to_box_index(tab_index)
        if box_index >= len(self.current_boxes):
            return

        # Save current box data before switching
        if hasattr(self, "current_box_index") and self.current_box_index < len(self.current_boxes):
            self._save_current_box_data()

        # Update current box index
        self.current_box_index = box_index

        # Load new box data into form
        self._load_box_data(box_index)

        # Highlight the selected box on the image
        if hasattr(self, "box_items") and self.box_items:
            for i, item in enumerate(self.box_items):
                if hasattr(item, "idx"):  # DraggableBoxItem
                    item.setSelected(item.idx == box_index)

        # In queue mode (species review), zoom to the box for better visibility
        if self.queue_mode:
            self._zoom_to_box(box_index)

    def _zoom_to_box(self, box_idx: int, padding: float = 0.15):
        """Zoom and center the view on a specific detection box.

        Args:
            box_idx: Index of the box in current_boxes
            padding: Extra padding around the box (0.15 = 15% padding)
        """
        if not hasattr(self, "current_boxes") or box_idx >= len(self.current_boxes):
            return
        if not self.current_pixmap:
            return

        box = self.current_boxes[box_idx]
        x1, y1 = box.get("x1", 0), box.get("y1", 0)
        x2, y2 = box.get("x2", 1), box.get("y2", 1)

        # Convert normalized coords to pixel coords
        w = self.current_pixmap.width()
        h = self.current_pixmap.height()
        px1, py1 = x1 * w, y1 * h
        px2, py2 = x2 * w, y2 * h

        # Add padding
        box_w = px2 - px1
        box_h = py2 - py1
        pad_w = box_w * padding
        pad_h = box_h * padding

        # Create padded rect (clamped to image bounds)
        rect = QRectF(
            max(0, px1 - pad_w),
            max(0, py1 - pad_h),
            min(w, box_w + 2 * pad_w),
            min(h, box_h + 2 * pad_h)
        )

        # Zoom to fit the box with padding
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _save_current_box_data(self):
        """Save form field values to the current box."""
        if not hasattr(self, "current_boxes") or not self.current_boxes:
            return
        if self.current_box_index >= len(self.current_boxes):
            return

        box = self.current_boxes[self.current_box_index]

        # Save species
        species = self.species_combo.currentText().strip()
        box["species"] = species

        # Save sex (only for Deer - clear for other species)
        deer_id = self.deer_id_edit.currentText().strip()
        if species.lower() == "deer":
            sex = self._get_sex()
            # If deer_id is set, auto-set to Buck
            if deer_id and sex == "Unknown":
                sex = "Buck"
                self._set_sex("Buck")
            box["sex"] = sex
            # Clear sex_conf when user has confirmed sex (not Unknown) - removes from review queue
            if sex in ("Buck", "Doe"):
                box["sex_conf"] = None
        else:
            box["sex"] = None
            box["sex_conf"] = None

        # Save deer ID
        box["deer_id"] = deer_id

        # Save age
        box["age_class"] = self.age_combo.currentText().strip()

        # Save antler points
        box["left_points_min"] = self._get_int_field(self.left_min)
        box["right_points_min"] = self._get_int_field(self.right_min)
        box["left_points_uncertain"] = self.left_uncertain.isChecked()
        box["right_points_uncertain"] = self.right_uncertain.isChecked()

        # Persist boxes to database
        self._persist_boxes()

        # Update tab name
        self._update_box_tab_name(self.current_box_index)

    def _load_box_data(self, box_idx: int):
        """Load box data into the form fields."""
        if not hasattr(self, "current_boxes") or box_idx >= len(self.current_boxes):
            return

        box = self.current_boxes[box_idx]

        # Block signals to prevent save loops
        self.species_combo.blockSignals(True)
        self.deer_id_edit.blockSignals(True)
        self.age_combo.blockSignals(True)

        # Load species
        self.species_combo.setCurrentText(box.get("species", ""))

        # Load sex
        sex = box.get("sex", "Unknown")
        if sex not in ("Buck", "Doe"):
            sex = "Unknown"
        self._set_sex(sex)

        # Load deer ID
        self.deer_id_edit.setCurrentText(box.get("deer_id", ""))

        # Load age
        self.age_combo.setCurrentText(box.get("age_class", ""))

        # Load antler points
        self._set_int_field(self.left_min, box.get("left_points_min"))
        self._set_int_field(self.right_min, box.get("right_points_min"))
        self.left_uncertain.setChecked(box.get("left_points_uncertain", False))
        self.right_uncertain.setChecked(box.get("right_points_uncertain", False))

        # Unblock signals
        self.species_combo.blockSignals(False)
        self.deer_id_edit.blockSignals(False)
        self.age_combo.blockSignals(False)

    def _is_head_box(self, box: dict) -> bool:
        """Check if a box is a deer head box (not a subject box)."""
        label = str(box.get("label", "")).lower()
        return "deer_head" in label or "head" in label

    def _box_sort_priority(self, box: dict) -> int:
        """Return sort priority for a box (lower = higher priority).

        Priority order:
        0 = Deer with buck ID (highest)
        1 = Buck
        2 = Doe
        3 = Deer (general)
        4 = Other species (lowest)
        """
        species = (box.get("species") or "").lower()
        sex = (box.get("sex") or "").lower()
        deer_id = box.get("deer_id") or ""

        # Deer with buck ID = highest priority
        if deer_id and species == "deer":
            return 0
        # Buck
        if sex == "buck" or species == "deer" and sex == "buck":
            return 1
        # Doe
        if sex == "doe" or species == "deer" and sex == "doe":
            return 2
        # Deer (general, no sex specified)
        if species == "deer":
            return 3
        # Other species
        return 4

    def _get_subject_boxes(self) -> list:
        """Get only subject boxes (not head boxes) for tab display, sorted by priority.

        Priority: Deer with buck ID > Bucks > Does > Deer > Other species
        """
        if not hasattr(self, "current_boxes") or not self.current_boxes:
            return []
        boxes = [b for b in self.current_boxes if not self._is_head_box(b)]
        # Sort by priority (stable sort preserves original order for equal priorities)
        boxes.sort(key=self._box_sort_priority)
        return boxes

    def _get_subject_box_indices(self) -> list:
        """Get indices of subject boxes in current_boxes, sorted by same priority as _get_subject_boxes."""
        if not hasattr(self, "current_boxes") or not self.current_boxes:
            return []
        # Get indices with their boxes for sorting
        indexed_boxes = [(i, b) for i, b in enumerate(self.current_boxes) if not self._is_head_box(b)]
        # Sort by same priority as _get_subject_boxes
        indexed_boxes.sort(key=lambda x: self._box_sort_priority(x[1]))
        return [i for i, b in indexed_boxes]

    def _tab_index_to_box_index(self, tab_idx: int) -> int:
        """Convert tab index to actual box index in current_boxes."""
        indices = self._get_subject_box_indices()
        if 0 <= tab_idx < len(indices):
            return indices[tab_idx]
        return 0

    def _box_index_to_tab_index(self, box_idx: int) -> int:
        """Convert box index to tab index (-1 if box is a head box)."""
        indices = self._get_subject_box_indices()
        if box_idx in indices:
            return indices.index(box_idx)
        return -1

    def _update_box_tab_bar(self):
        """Update the box tab bar based on current_boxes (excluding head boxes)."""
        if not hasattr(self, "box_tab_bar"):
            return

        self.box_tab_bar.blockSignals(True)
        self.box_tab_bar.clear()

        subject_boxes = self._get_subject_boxes()

        if not subject_boxes:
            # No subject boxes - add single "Photo" tab
            self.box_tab_bar.addTab(QWidget(), "Photo")
            self.current_box_index = 0
            self.box_tab_bar.blockSignals(False)
            return

        # Create a tab for each subject box (skip head boxes)
        for tab_idx, box in enumerate(subject_boxes):
            box_num = tab_idx + 1
            species = box.get("species", "")
            if species:
                tab_name = f"Subject {box_num}: {species}"
            else:
                tab_name = f"Subject {box_num}"
            self.box_tab_bar.addTab(QWidget(), tab_name)

        # Select first tab and set corresponding box index
        self.box_tab_bar.setCurrentIndex(0)
        self.current_box_index = self._tab_index_to_box_index(0)
        self.box_tab_bar.blockSignals(False)

        # Load first box data
        self._load_box_data(self.current_box_index)

    def _update_box_tab_name(self, box_idx: int):
        """Update a single box tab's name based on its data."""
        if not hasattr(self, "box_tab_bar"):
            return
        if box_idx >= len(self.current_boxes):
            return

        # Convert box index to tab index (head boxes don't have tabs)
        tab_idx = self._box_index_to_tab_index(box_idx)
        if tab_idx < 0 or tab_idx >= self.box_tab_bar.count():
            return  # This is a head box or invalid index

        box = self.current_boxes[box_idx]
        box_num = tab_idx + 1  # Tab number, not box number
        species = box.get("species", "")
        if species:
            tab_name = f"Subject {box_num}: {species}"
        else:
            tab_name = f"Subject {box_num}"
        self.box_tab_bar.setTabText(tab_idx, tab_name)

    def _get_int_field(self, field) -> int:
        """Get integer value from a QLineEdit, or None if empty/invalid."""
        try:
            text = field.text().strip()
            if text:
                return int(text)
        except (ValueError, AttributeError):
            pass
        return None

    def _on_box_species_changed_from_form(self):
        """Handle species change from the main form - update current box."""
        if not hasattr(self, "current_boxes") or not self.current_boxes:
            return
        if self.current_box_index >= len(self.current_boxes):
            return

        species = self.species_combo.currentText().strip()
        self.current_boxes[self.current_box_index]["species"] = species
        self._update_box_tab_name(self.current_box_index)
        self._draw_boxes()  # Update box labels on image

    def _delete_current_box(self):
        """Delete the currently selected box."""
        if not hasattr(self, "current_boxes") or not self.current_boxes:
            return

        # Get the actual box index (not tab index)
        box_idx = self.current_box_index
        if box_idx < 0 or box_idx >= len(self.current_boxes):
            return

        # Don't allow deleting if only one subject box left
        subject_boxes = self._get_subject_boxes()
        if len(subject_boxes) <= 1 and not self._is_head_box(self.current_boxes[box_idx]):
            return

        # Remove the box
        del self.current_boxes[box_idx]

        # Persist changes
        self._persist_boxes()

        # Update current box index
        if self.current_boxes:
            subject_indices = self._get_subject_box_indices()
            if subject_indices:
                self.current_box_index = subject_indices[0]
            else:
                self.current_box_index = 0
        else:
            self.current_box_index = 0

        # Refresh UI
        self._update_box_tab_bar()
        self._draw_boxes()

    def _on_box_created(self, payload: dict):
        if not self.current_pixmap:
            return
        rect: QRectF = payload["rect"]
        label = payload.get("label", "subject")
        w = self.current_pixmap.width()
        h = self.current_pixmap.height()
        if w <= 0 or h <= 0:
            return
        x1 = max(0.0, min(1.0, rect.left() / w))
        y1 = max(0.0, min(1.0, rect.top() / h))
        x2 = max(0.0, min(1.0, rect.right() / w))
        y2 = max(0.0, min(1.0, rect.bottom() / h))
        self.current_boxes.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        self._draw_boxes()
        self._persist_boxes()
        # Update box tab bar
        self._update_box_tab_bar()
        # Exit draw mode so boxes can be selected/edited
        self._set_box_mode(None)
        if self.ai_review_mode and not self._photo_has_ai_boxes(self.current_boxes):
            self._advance_ai_review()

    def _draw_boxes(self):
        if not self.scene or not self.current_pixmap:
            return
        for it in self.box_items:
            self.scene.removeItem(it)
        self.box_items = []
        if self.boxes_hidden:
            return
        w = self.current_pixmap.width()
        h = self.current_pixmap.height()

        # Scale line thickness and font size based on image size
        # Use smaller dimension as reference for consistent scaling
        ref_size = min(w, h)
        line_width = max(2, int(ref_size * 0.004))  # 0.4% of image, min 2px
        font_size = max(12, int(ref_size * 0.018))  # 1.8% of image, min 12pt
        label_offset = max(16, int(ref_size * 0.025))  # 2.5% of image, min 16px

        # Build mapping from box to sorted subject number
        # Subject boxes are sorted by priority (deer first, etc.)
        sorted_subjects = self._get_subject_boxes()
        box_to_subject_num = {}
        for subj_idx, subj_box in enumerate(sorted_subjects):
            # Find this box in current_boxes by matching ID (preferred) or coordinates (fallback)
            subj_id = subj_box.get("id")
            for orig_idx, orig_box in enumerate(self.current_boxes):
                # Match by ID first (reliable), then by coordinates (fallback)
                if subj_id is not None and orig_box.get("id") == subj_id:
                    box_to_subject_num[orig_idx] = subj_idx + 1
                    break
                elif subj_id is None and (
                    orig_box.get("x1") == subj_box.get("x1") and
                    orig_box.get("y1") == subj_box.get("y1") and
                    orig_box.get("x2") == subj_box.get("x2") and
                    orig_box.get("y2") == subj_box.get("y2")
                ):
                    box_to_subject_num[orig_idx] = subj_idx + 1
                    break

        def _on_change(idx, scene_rect: QRectF):
            if idx < 0 or idx >= len(self.current_boxes):
                return
            bx = self.current_boxes[idx]
            bx["x1"] = max(0.0, min(1.0, scene_rect.left() / w))
            bx["y1"] = max(0.0, min(1.0, scene_rect.top() / h))
            bx["x2"] = max(0.0, min(1.0, scene_rect.right() / w))
            bx["y2"] = max(0.0, min(1.0, scene_rect.bottom() / h))
            self._persist_boxes()
        for idx, b in enumerate(self.current_boxes):
            rect = QRectF(b["x1"] * w, b["y1"] * h, (b["x2"] - b["x1"]) * w, (b["y2"] - b["y1"]) * h)
            lbl = b.get("label")
            if str(lbl) == "ai_deer_head":
                pen = QPen(Qt.GlobalColor.magenta)
                box_color = Qt.GlobalColor.magenta
            elif str(lbl).startswith("ai_"):
                pen = QPen(Qt.GlobalColor.yellow)
                box_color = Qt.GlobalColor.yellow
            elif lbl == "deer_head":
                pen = QPen(Qt.GlobalColor.red)
                box_color = Qt.GlobalColor.red
            else:
                pen = QPen(Qt.GlobalColor.green)
                box_color = Qt.GlobalColor.green
            pen.setWidth(line_width)
            item = DraggableBoxItem(idx, rect, pen, _on_change)
            self.scene.addItem(item)
            self.box_items.append(item)

            # Add subject label text (Subject 1, Subject 2, etc.) with species and sex if known
            # Use sorted subject number if this is a subject box, otherwise use original index
            box_num = box_to_subject_num.get(idx, idx + 1)
            species = b.get("species", "")
            sex = b.get("sex", "")
            # If no per-box species, check for photo-level suggestion
            if not species and self.photos and self.index < len(self.photos):
                photo = self.photos[self.index]
                species = photo.get("suggested_tag", "")
                if species:
                    species = f"{species}?"  # Add ? to indicate it's a suggestion
            # Build label with species and sex
            if species and sex and sex != "Unknown":
                label_text = f"Subject {box_num}: {species} ({sex})"
            elif species:
                label_text = f"Subject {box_num}: {species}"
            elif sex and sex != "Unknown":
                label_text = f"Subject {box_num}: ({sex})"
            else:
                label_text = f"Subject {box_num}"

            # Create text item ABOVE the box for visibility
            text_item = self.scene.addSimpleText(label_text)
            font = QFont("Arial", font_size, QFont.Weight.Bold)
            text_item.setFont(font)
            text_item.setBrush(QBrush(box_color))
            # Position above the box (scaled offset)
            label_y = rect.top() - label_offset
            if label_y < 0:  # If box is near top of image, put label inside
                label_y = rect.top() + 5
            text_item.setPos(rect.left() + 5, label_y)
            text_item.setZValue(100)  # Draw on top
            self.box_items.append(text_item)

    def _persist_boxes(self):
        """Persist boxes immediately for the current photo."""
        if not self.photos or self.index >= len(self.photos):
            return
        pid = self.photos[self.index].get("id")
        if not pid:
            return
        try:
            self.db.set_boxes(pid, self.current_boxes)
            # Sync box species to photo tags - tags should exactly match current box species
            box_species = set(box.get("species") for box in self.current_boxes if box.get("species") and box.get("species") != "Unknown")
            current_tags = self.db.get_tags(pid)

            # Build new tags list:
            # - Keep non-species tags (any custom tags not in SPECIES_OPTIONS)
            # - Replace species tags with current box species
            # - BUT preserve Empty/Verification tags (photo-level labels that shouldn't be auto-removed)
            species_set = set(SPECIES_OPTIONS)
            had_verification = "Verification" in current_tags
            had_empty = "Empty" in current_tags
            new_tags = [t for t in current_tags if t not in species_set]  # Keep non-species tags

            if box_species:
                # Add current box species (removes Empty/Unknown implicitly since they won't be in box_species)
                new_tags.extend(sorted(box_species))

            # ALWAYS preserve Verification tag - it's a special photo-level label
            # that should never be removed by box species sync
            if had_verification and "Verification" not in new_tags:
                new_tags.append("Verification")

            # Preserve Empty tag when no boxes exist (Empty photos have boxes cleared)
            if had_empty and not box_species and "Empty" not in new_tags:
                new_tags.append("Empty")

            if set(new_tags) != set(current_tags):
                self.db.update_photo_tags(pid, new_tags)
                # Update UI (block signals to prevent re-triggering save)
                self.tags_edit.blockSignals(True)
                self.tags_edit.setText(", ".join(new_tags))
                self.tags_edit.blockSignals(False)
            # Queue for cloud sync
            if hasattr(self, 'sync_manager'):
                self.sync_manager.queue_change()
        except Exception as exc:
            logger.error(f"Box save failed: {exc}")

    def _delete_box_by_idx(self, idx: int):
        """Remove a box by index (used by Delete key)."""
        if idx < 0 or idx >= len(self.current_boxes):
            return
        del self.current_boxes[idx]
        self._draw_boxes()
        self._persist_boxes()
        # Update box tab bar
        self._update_box_tab_bar()
        if self.ai_review_mode and not self._photo_has_ai_boxes(self.current_boxes):
            self._advance_ai_review()

    def exit_review_modes(self):
        """Leave any review mode and show full library."""
        self.in_review_mode = False
        self.ai_review_mode = False
        self.ai_review_queue = []
        self._advancing_review = False
        # Hide box controls when exiting review
        self.box_container_top.hide()
        self.box_container_bottom.hide()
        self.photos = self._sorted_photos(self.db.get_all_photos())
        if not self.photos:
            self.photo_list_widget.clear()
            return
        if self.index >= len(self.photos):
            self.index = max(0, len(self.photos) - 1)
        self._populate_photo_list()
        self.load_photo()

    def _compute_species_labels(self) -> List[str]:
        """Collect all species labels used (defaults + custom + tags excluding buck/doe)."""
        species = set(SPECIES_OPTIONS)
        try:
            species.update(self.db.list_custom_species())
        except Exception:
            pass
        try:
            for p in self.db.get_all_photos():
                for t in self.db.get_tags(p["id"]):
                    if t and t.lower() not in SEX_TAGS:
                        species.add(t)
        except Exception:
            pass
        # Remove empties, keep stable order
        cleaned = [s for s in species if s is not None]
        cleaned = [s.strip() for s in cleaned if s.strip()]
        return sorted(set(cleaned))

    def _write_species_labels_file(self):
        """Write models/labels.txt with ONLY valid species labels.

        Uses VALID_SPECIES as the master list to prevent corrupted/partial
        entries from polluting the labels file.
        """
        # Only write labels that are in the approved VALID_SPECIES set
        labels = sorted(VALID_SPECIES)
        models_dir = Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        labels_path = models_dir / "labels.txt"
        try:
            with labels_path.open("w", encoding="utf-8") as f:
                for lbl in labels:
                    f.write(lbl + "\n")
            logger.info(f"Wrote {len(labels)} valid species to labels.txt")
        except Exception as exc:
            logger.error(f"Failed to write labels.txt: {exc}")

    # ====== Suggestion review ======
    def _species_set(self) -> set:
        species = set(SPECIES_OPTIONS)
        try:
            species.update(self.db.list_custom_species())
        except Exception:
            pass
        return species

    def _gather_pending_suggestions(self) -> List[dict]:
        """Return photos with a suggested tag that is not yet applied as a species."""
        pending = []
        species_set = self._species_set()
        for p in self.db.get_all_photos():
            pid = p["id"]
            tags = set(self.db.get_tags(pid))
            sugg = p.get("suggested_tag")
            conf = p.get("suggested_confidence")
            has_species = any(t in species_set for t in tags)
            if sugg and not has_species:
                pending.append({"photo": p, "suggest": sugg, "conf": conf if conf is not None else 0.0})
        # sort by highest confidence first
        pending.sort(key=lambda x: x["conf"] if x["conf"] is not None else 0.0, reverse=True)
        return pending

    def enter_review_queue(self):
        """Switch main list to pending suggestions and work through them; resolved items disappear."""
        pending = self._gather_pending_suggestions()
        if not pending:
            QMessageBox.information(self, "Review Queue", "No pending suggestions to review.")
            return
        self.in_review_mode = True
        self.photos = [p["photo"] for p in pending]
        self._populate_photo_list()
        self.index = 0
        self.load_photo()

    def _exit_review_queue_if_done(self):
        if self.in_review_mode and not self.photos:
            self.in_review_mode = False
            QMessageBox.information(self, "Review Queue", "All pending suggestions resolved.")
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()
            if self.photos:
                self.index = 0
                self.load_photo()

    def _current_photo_resolved(self, pid: int) -> bool:
        """Resolved if species tag applied or suggestion cleared."""
        species_set = self._species_set()
        tags = set(self.db.get_tags(pid))
        if any(t in species_set for t in tags):
            return True
        p = self.db.get_photo_by_id(pid)
        if p and not p.get("suggested_tag"):
            return True
        return False

    def suggest_and_review(self):
        """Run AI suggestions on unlabeled photos, then open a review dialog."""
        if not self.suggester or not self.suggester.ready:
            QMessageBox.information(self, "AI Model Not Available", "AI model not loaded.")
            return
        species_set = self._species_set()
        new_preds = 0
        for p in self.db.get_all_photos():
            pid = p["id"]
            tags = set(self.db.get_tags(pid))
            has_species = any(t in species_set for t in tags)
            if has_species:
                continue
            res = self.suggester.predict(p.get("file_path"))
            if res:
                label, conf = res
                self.db.set_suggested_tag(pid, label, conf)
                new_preds += 1
        pending = self._gather_pending_suggestions()
        if not pending:
            QMessageBox.information(self, "AI Suggestions", "No pending suggestions to review.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Review AI Suggestions ({len(pending)})")
        layout = QVBoxLayout(dlg)
        self.suggest_list = QListWidget()
        for item in pending:
            p = item["photo"]
            label = item["suggest"]
            conf = item["conf"]
            percent = f"{conf*100:.1f}%" if conf is not None else "n/a"
            text = f"{percent} {label} — {Path(p['file_path']).name}"
            lw_item = QListWidgetItem(text)
            lw_item.setData(Qt.ItemDataRole.UserRole, item)
            self.suggest_list.addItem(lw_item)
        layout.addWidget(self.suggest_list)

        # Detail/editor for selected suggestion
        detail = QHBoxLayout()
        self.suggest_label_combo = QComboBox()
        # Include all known species plus suggested labels
        all_species = sorted(self._species_set() | {i["suggest"] for i in pending if i.get("suggest")})
        self.suggest_label_combo.addItems(all_species)
        self.suggest_conf = QLabel("Conf: n/a")
        detail.addWidget(QLabel("Label:"))
        detail.addWidget(self.suggest_label_combo)
        detail.addWidget(self.suggest_conf)
        detail.addStretch()
        layout.addLayout(detail)

        btn_row = QHBoxLayout()
        accept_btn = QPushButton("Accept")
        reject_btn = QPushButton("Reject")
        open_btn = QPushButton("Open in Labeler")
        close_btn = QPushButton("Close")
        btn_row.addWidget(accept_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addWidget(open_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        def _update_detail():
            sel = self.suggest_list.currentItem()
            if not sel:
                self.suggest_label_combo.setCurrentText("")
                self.suggest_conf.setText("Conf: n/a")
                return
            data = sel.data(Qt.ItemDataRole.UserRole)
            label = data.get("suggest") or ""
            conf = data.get("conf")
            if label and self.suggest_label_combo.findText(label) == -1:
                self.suggest_label_combo.addItem(label)
            self.suggest_label_combo.setCurrentText(label)
            self.suggest_conf.setText(f"Conf: {conf*100:.1f}%" if conf is not None else "Conf: n/a")

        def _apply_accept():
            selected = self.suggest_list.selectedItems()
            if not selected:
                return
            chosen_label = self.suggest_label_combo.currentText().strip()
            if not chosen_label:
                return
            rows = [self.suggest_list.row(sel) for sel in selected]
            min_row = min(rows) if rows else 0
            for sel in selected:
                data = sel.data(Qt.ItemDataRole.UserRole)
                p = data["photo"]
                pid = p["id"]
                tags = set(self.db.get_tags(pid))
                tags.add(chosen_label)
                self.db.update_photo_tags(pid, list(tags))
                # Change of label counts as overriding suggestion; clear suggestion
                self.db.set_suggested_tag(pid, None, None)
            for sel in selected:
                self.suggest_list.takeItem(self.suggest_list.row(sel))
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()
            if self.suggest_list.count() == 0:
                dlg.accept()
                return
            next_row = min(min_row, self.suggest_list.count() - 1)
            self.suggest_list.setCurrentRow(next_row)
            _update_detail()

        def _apply_reject():
            selected = self.suggest_list.selectedItems()
            if not selected:
                return
            rows = [self.suggest_list.row(sel) for sel in selected]
            min_row = min(rows) if rows else 0
            for sel in selected:
                data = sel.data(Qt.ItemDataRole.UserRole)
                pid = data["photo"]["id"]
                self.db.set_suggested_tag(pid, None, None)
            for sel in selected:
                self.suggest_list.takeItem(self.suggest_list.row(sel))
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()
            if self.suggest_list.count() == 0:
                dlg.accept()
                return
            next_row = min(min_row, self.suggest_list.count() - 1)
            self.suggest_list.setCurrentRow(next_row)
            _update_detail()

        def _open_in_labeler():
            selected = self.suggest_list.selectedItems()
            if not selected:
                return
            data = selected[0].data(Qt.ItemDataRole.UserRole)
            pid = data["photo"]["id"]
            dlg.accept()
            self._select_photo_by_id(pid)

        def _close():
            dlg.accept()

        self.suggest_list.currentItemChanged.connect(lambda _c, _p: _update_detail())
        self.suggest_list.itemSelectionChanged.connect(_update_detail)
        if self.suggest_list.count():
            self.suggest_list.setCurrentRow(0)
        _update_detail()

        accept_btn.clicked.connect(_apply_accept)
        reject_btn.clicked.connect(_apply_reject)
        open_btn.clicked.connect(_open_in_labeler)
        close_btn.clicked.connect(_close)

        dlg.exec()

    def export_csvs(self):
        if not self.photos:
            return
        export_dir = Path(__file__).resolve().parent / "exports"
        export_dir.mkdir(exist_ok=True)
        cls_path = export_dir / "classifier.csv"
        reid_path = export_dir / "reid.csv"

        with cls_path.open("w", encoding="utf-8") as f:
            f.write("path,species,sex\n")
            for p in self.photos:
                species = p.get("suggested_tag") or ""
                tags = self.db.get_tags(p["id"])
                for t in tags:
                    if t in SPECIES_OPTIONS and not species:
                        species = t
                sex = ""
                for t in tags:
                    if t.lower() in ("buck", "doe"):
                        sex = t.capitalize()
                f.write(f"{p['file_path']},{species},{sex}\n")

        with reid_path.open("w", encoding="utf-8") as f:
            f.write("path,deer_id\n")
            for p in self.photos:
                deer = self.db.get_deer_metadata(p["id"])
                deer_id = deer.get("deer_id") or ""
                if deer_id:
                    f.write(f"{p['file_path']},{deer_id}\n")

        QMessageBox.information(self, "Exported", f"Wrote:\n{cls_path}\n{reid_path}")

    @staticmethod
    def _sorted_photos(photos: list) -> list:
        """Sort photos by date_taken (ascending), fallback to file_path."""
        from datetime import datetime
        def key(p):
            dt = p.get("date_taken") or ""
            try:
                parsed = datetime.fromisoformat(dt)
                # Strip timezone info to avoid naive vs aware comparison errors
                if parsed.tzinfo is not None:
                    parsed = parsed.replace(tzinfo=None)
            except Exception:
                parsed = None
            return (parsed or datetime.min, p.get("file_path") or "")
        return sorted(photos, key=key)

    def _get_all_photos_cached(self, force_refresh: bool = False) -> list:
        """Get all photos with caching to reduce database calls.

        Cache is valid for 5 seconds unless force_refresh is True.
        """
        import time
        now = time.time()
        if force_refresh or self._photos_cache is None or (now - self._photos_cache_time) > 5:
            self._photos_cache = self.db.get_all_photos()
            self._photos_cache_time = now
        return self._photos_cache

    def _invalidate_photos_cache(self):
        """Invalidate the photos cache after modifications."""
        self._photos_cache = None
        self._photos_cache_time = 0

    def _refresh_photos(self, force: bool = False):
        """Refresh self.photos from database with sorting.

        Uses cache for efficiency. Call with force=True after modifications.
        """
        photos = self._get_all_photos_cached(force_refresh=force)
        self.photos = self._sorted_photos(photos)
        if self.index >= len(self.photos):
            self.index = max(0, len(self.photos) - 1)

    def _maybe_autofill_exif(self, photo: dict):
        """Try to read EXIF to set camera model and date if missing."""
        from PIL import Image, ExifTags
        pid = photo["id"]
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return
        need_date = not photo.get("date_taken")
        try:
            img = Image.open(path)
            exif = img.getexif()
            if exif:
                exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            else:
                exif_data = {}
        except Exception:
            exif_data = {}

        # Camera model (just store in camera_info, NOT camera_location)
        cam = None
        for key in ("Model", "Make"):
            if exif_data.get(key):
                cam = exif_data.get(key)
                break
        if cam:
            self.db.set_camera_model(pid, str(cam))

        # Date taken
        dt = None
        for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            if exif_data.get(key):
                dt = str(exif_data[key]).replace(":", "-", 2).replace(" ", "T")
                break
        if need_date and dt:
            self.db.set_date_taken(pid, dt)
            # refresh
            photo.update(self.db.get_photo_by_id(pid) or {})

    def _maybe_autofill_stamp(self, photo: dict):
        """Attempt to read bottom stamp for camera model if available."""
        try:
            import pytesseract  # optional
        except Exception:
            return  # silently skip if OCR not installed
        path = photo.get("file_path")
        pid = photo.get("id")
        if not path or not os.path.exists(path):
            return
        try:
            from PIL import Image, ImageOps
            img = Image.open(path).convert("RGB")
            w, h = img.size
            stamp = img.crop((0, int(h * 0.82), w, h))  # bottom ~18%
            gray = ImageOps.grayscale(stamp)
            text = pytesseract.image_to_string(gray)
            text = text.strip()
            if text:
                # Heuristic: take first non-empty line as camera model (NOT location)
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if lines:
                    cam = lines[0]
                    self.db.set_camera_model(pid, cam)
        except Exception:
            return

    def _load_previews(self):
        """Show small previews for neighboring photos."""
        # Clear layout
        while self.preview_layout.count():
            item = self.preview_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if not self.photos:
            return
        indices = [self.index - 2, self.index - 1, self.index + 1, self.index + 2]
        total = len(self.photos)
        for idx in indices:
            actual = idx % total
            p = self.photos[actual]
            thumb = ClickableLabel(actual, self._go_to_index)
            thumb.setFixedSize(120, 90)
            thumb.setStyleSheet("background:#222; border:1px solid #444;")
            pix = QPixmap(p.get("file_path", ""))
            if not pix.isNull():
                thumb.setPixmap(pix.scaled(120, 90, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            thumb.setToolTip(os.path.basename(p.get("file_path","")))
            self.preview_layout.addWidget(thumb)

    def _go_to_index(self, idx: int):
        """Jump to a specific photo index (used by clickable previews)."""
        if not self.photos:
            return
        self.index = idx % len(self.photos)
        self.load_photo()

    @staticmethod
    def _to_int(text: str):
        try:
            return int(text.strip())
        except Exception:
            return None

    @staticmethod
    def _set_int_field(widget: QLineEdit, value):
        widget.setText("" if value is None else str(value))

    @staticmethod
    def _row_pair(left_widget: QWidget, right_widget: QWidget) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(left_widget)
        layout.addWidget(QLabel(" / "))
        layout.addWidget(right_widget)
        layout.addStretch()
        return w

    @staticmethod
    def _row_typical(min_widget: QWidget, uncertain: QCheckBox) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Min"))
        layout.addWidget(min_widget)
        layout.addWidget(uncertain)
        layout.addStretch()
        return w

    def _get_sex(self) -> str:
        for label, btn in self.sex_buttons.items():
            if btn.isChecked():
                return label
        return "Unknown"

    def _set_sex(self, value: str):
        val = value if value in self.sex_buttons else "Unknown"
        for label, btn in self.sex_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(label == val)
            btn.blockSignals(False)

    def _toggle_previews(self, visible: bool):
        """Show/hide the preview strip to save space."""
        self.preview_frame.setVisible(visible)
        self.preview_toggle.setText("Hide previews" if visible else "Show previews")

    def _toggle_details_panel(self, hide: bool):
        """Collapse/expand the right-side details pane to give more photo space."""
        if not hasattr(self, "splitter") or not hasattr(self, "form_scroll"):
            return
        if hide:
            self.details_toggle.setText("◀ Show Details")
            self.details_toggle.setStyleSheet("QToolButton { font-size: 14px; font-weight: bold; padding: 8px 16px; background-color: #4a90d9; color: white; border-radius: 4px; }")
            self._last_split_sizes = self.splitter.sizes()
            # Actually hide the details panel
            self.form_scroll.setVisible(False)
        else:
            self.details_toggle.setText("Hide Details")
            self.details_toggle.setStyleSheet("")  # Reset to default style
            # Show the details panel
            self.form_scroll.setVisible(True)
            if getattr(self, "_last_split_sizes", None):
                self.splitter.setSizes(self._last_split_sizes)
            else:
                self.splitter.setSizes([220, 900, 400])

    def _toggle_multi_select(self, enabled: bool):
        """Toggle multi-select mode - clicking photos toggles their selection."""
        if enabled:
            self.photo_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            self.multi_select_toggle.setText("Select Multiple ✓")
            self.multi_select_toggle.setStyleSheet("QToolButton { background-color: #4a90d9; color: white; padding: 4px 8px; border-radius: 4px; }")
        else:
            # Clear selection and refresh to fix text colors
            self.photo_list_widget.clearSelection()
            self.photo_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.multi_select_toggle.setText("Select Multiple")
            self.multi_select_toggle.setStyleSheet("")
            # Re-select current item
            if 0 <= self.index < self.photo_list_widget.count():
                for i in range(self.photo_list_widget.count()):
                    item = self.photo_list_widget.item(i)
                    if item and item.data(Qt.ItemDataRole.UserRole) == self.index:
                        self.photo_list_widget.setCurrentItem(item)
                        break

    def _select_all_photos(self):
        """Select all photos in the current filtered list."""
        # Enable multi-select mode if not already
        if not self.multi_select_toggle.isChecked():
            self.multi_select_toggle.setChecked(True)

        # Select all items in the list
        self.photo_list_widget.selectAll()

        count = len(self.photo_list_widget.selectedItems())
        self.statusBar().showMessage(f"Selected {count} photos", 3000)

    def _toggle_additional_buck(self, enabled: bool):
        """Show/hide the second-buck fields."""
        self.add_buck_container.setVisible(enabled)
        if enabled:
            # ensure dropdown is synced
            self._populate_additional_deer_dropdown()
        else:
            # clear fields when hiding
            self.add_deer_id_edit.setCurrentText("")
            self.add_age_combo.setCurrentText("")
            self.add_left_min.clear()
            self.add_right_min.clear()
            self.add_left_uncertain.setChecked(False)
            self.add_right_uncertain.setChecked(False)

    def _update_suggestion_display(self, photo: dict):
        """Show AI suggestion with confidence, require explicit apply."""
        tag = photo.get("suggested_tag") or ""
        conf = photo.get("suggested_confidence")
        self.current_suggested_tag = tag
        self.current_suggested_conf = conf

        # Check for queue comparison data (old_tag vs new_suggestion)
        photo_id = photo.get("id")
        comparison = getattr(self, '_queue_comparison_data', {}).get(photo_id)

        if comparison:
            # Handle verification queue format (current_species, prediction)
            if 'current_species' in comparison:
                current = comparison.get('current_species', '?')
                prediction = comparison.get('prediction', '?')
                confidence = comparison.get('confidence', 0)
                second = comparison.get('second_best', '')
                second_conf = comparison.get('second_conf', 0)
                self.suggest_label.setText(f"Label: {current} → AI: {prediction} ({confidence}%) vs {second} ({second_conf}%)")
                self.current_suggested_tag = prediction
            else:
                # Handle misclassified queue format (old_tag, new_suggestion)
                old_tag = comparison.get('old_tag', '?')
                new_suggestion = comparison.get('new_suggestion', '?')
                confidence = comparison.get('confidence', 0)
                self.suggest_label.setText(f"Label: {old_tag} → AI: {new_suggestion} ({confidence}%)")
                self.current_suggested_tag = new_suggestion
            self.apply_suggest_btn.setEnabled(True)
        elif tag:
            if conf is not None:
                pct = int(conf * 100) if conf <= 1 else int(conf)
                self.suggest_label.setText(f"Suggested: {tag} ({pct}%)")
            else:
                self.suggest_label.setText(f"Suggested: {tag}")
            self.apply_suggest_btn.setEnabled(True)
        else:
            self.suggest_label.setText("Suggested: —")
            self.apply_suggest_btn.setEnabled(False)

    def _apply_suggestion(self):
        """User-approval step: apply suggested tag into species field."""
        tag = getattr(self, "current_suggested_tag", "") or ""
        if not tag:
            return
        self.species_combo.blockSignals(True)
        self.species_combo.setCurrentText(tag)
        self.species_combo.blockSignals(False)
        self.schedule_save()

    def _apply_all_suggestions(self):
        """Apply label (or AI suggestion if no labels) to all subject boxes.

        Priority: Any existing box label > AI suggestion
        Note: Deer ID is NOT copied since each buck must have unique ID.
        Head boxes are skipped (they're internal only).
        """
        subject_boxes = self._get_subject_boxes()

        if not subject_boxes:
            QMessageBox.information(self, "Apply to All", "No subjects in this photo.")
            return

        # Priority: current form selection > existing box label > AI suggestion
        species = self.species_combo.currentText().strip()

        if not species:
            # Check if ANY box has a saved label
            for box in subject_boxes:
                box_species = (box.get("species") or "").strip()
                if box_species:
                    species = box_species
                    break  # Use the first labeled box

        if not species:
            # No labels found - fall back to AI suggestion
            species = getattr(self, "current_suggested_tag", "") or ""

        if not species:
            QMessageBox.information(self, "Apply to All", "No species to apply. Label a subject or wait for AI suggestion.")
            return

        sex = self._get_sex()
        age = self.age_combo.currentText().strip()
        left_pts = self._get_int_field(self.left_min)
        right_pts = self._get_int_field(self.right_min)
        left_unc = self.left_uncertain.isChecked()
        right_unc = self.right_uncertain.isChecked()

        # Apply to all subject boxes (skip head boxes, skip deer_id)
        for idx in self._get_subject_box_indices():
            box = self.current_boxes[idx]
            box["species"] = species
            box["sex"] = sex
            box["age_class"] = age
            box["left_points_min"] = left_pts
            box["right_points_min"] = right_pts
            box["left_points_uncertain"] = left_unc
            box["right_points_uncertain"] = right_unc
            # NOTE: deer_id is intentionally NOT copied

        # Update form field to match
        self.species_combo.blockSignals(True)
        self.species_combo.setCurrentText(species)
        self.species_combo.blockSignals(False)

        # Persist boxes
        self._persist_boxes()

        # Save species as a tag (critical for queue filtering)
        self.save_current()

        # Clear the suggested_tag so photo doesn't reappear in queue
        if self.photos and self.index < len(self.photos):
            current_pid = self.photos[self.index].get("id")
            if current_pid:
                self.db.set_suggested_tag(current_pid, None, None)
                self.queue_reviewed.add(current_pid)

        # Update tab bar to show new species on all tabs
        self._update_box_tab_bar()

        # Redraw boxes to show labels
        self._draw_boxes()

        # In queue mode, mark current as reviewed and advance to next photo
        if self.queue_mode:
            self._mark_current_list_item_reviewed()
            self._queue_advance()

    def _accept_all_suggestions(self):
        """Accept AI suggestion and apply to all boxes on this photo."""
        subject_boxes = self._get_subject_boxes()

        if not subject_boxes:
            QMessageBox.information(self, "Accept All", "No subjects in this photo.")
            return

        # Get AI suggestion
        species = getattr(self, "current_suggested_tag", "") or ""

        if not species:
            QMessageBox.information(self, "Accept All", "No AI suggestion to accept.")
            return

        # Apply AI suggestion to all subject boxes
        for idx in self._get_subject_box_indices():
            box = self.current_boxes[idx]
            box["species"] = species

        # Update form field to match
        self.species_combo.setCurrentText(species)

        # Persist boxes
        self._persist_boxes()

        # Save species as a tag
        self.save_current()

        # Clear the suggested_tag so photo doesn't reappear in queue
        if self.photos and self.index < len(self.photos):
            current_pid = self.photos[self.index].get("id")
            if current_pid:
                self.db.set_suggested_tag(current_pid, None, None)
                self.queue_reviewed.add(current_pid)

        # Update tab bar to show new species on all tabs
        self._update_box_tab_bar()

        # Redraw boxes to show labels
        self._draw_boxes()

        # In queue mode, mark current as reviewed and advance to next photo
        if self.queue_mode:
            self._mark_current_list_item_reviewed()
            self._queue_advance()

    def _all_boxes_labeled(self) -> bool:
        """Check if all subject boxes have a species label."""
        subject_boxes = self._get_subject_boxes()
        if not subject_boxes:
            return True  # No boxes = nothing to label
        for box in subject_boxes:
            species = box.get("species") or ""
            if not species.strip():
                return False
        return True

    def _update_sex_suggestion_display(self, photo: dict):
        """Show AI sex suggestion (buck/doe) with confidence."""
        sex = photo.get("suggested_sex") or ""
        conf = photo.get("suggested_sex_confidence")
        self.current_suggested_sex = sex
        self.current_suggested_sex_conf = conf
        # Only show if not already set to buck/doe
        current_sex = self._get_sex()
        already_set = current_sex in ("Buck", "Doe")
        if sex and not already_set:
            if conf is not None:
                pct = int(conf * 100) if conf <= 1 else int(conf)
                self.sex_suggest_label.setText(f"Suggested: {sex.title()} ({pct}%)")
            else:
                self.sex_suggest_label.setText(f"Suggested: {sex.title()}")
            self.apply_sex_suggest_btn.setEnabled(True)
        else:
            self.sex_suggest_label.setText("Suggested: —")
            self.apply_sex_suggest_btn.setEnabled(False)

    def _apply_sex_suggestion(self):
        """User-approval step: apply suggested sex (buck/doe)."""
        sex = getattr(self, "current_suggested_sex", "") or ""
        if not sex:
            return
        # Normalize to title case for button matching
        sex_title = sex.title()
        if sex_title in self.sex_buttons:
            self.sex_buttons[sex_title].setChecked(True)
            self._sync_species_on_sex()
            self.schedule_save()
            # Clear suggestion display after applying
            self.sex_suggest_label.setText("Suggested: —")
            self.apply_sex_suggest_btn.setEnabled(False)

    def _on_sex_clicked(self, label: str, checked: bool):
        """Handle sex button clicks - allow toggling off to no selection."""
        if checked:
            # Uncheck other buttons, keep this one checked
            for lbl, btn in self.sex_buttons.items():
                if lbl != label:
                    btn.setChecked(False)
        else:
            # Clicking already-checked button - clear all (no selection)
            for btn in self.sex_buttons.values():
                btn.setChecked(False)

        self._sync_species_on_sex()
        self.save_current()

    def _sync_species_on_sex(self):
        """If Buck or Doe selected, set species to Deer. If Unknown, default to Deer if empty."""
        sex = self._get_sex()
        if sex in ("Buck", "Doe"):
            self.species_combo.blockSignals(True)
            self.species_combo.setCurrentText("Deer")
            self.species_combo.blockSignals(False)
        elif sex == "Unknown" and not self.species_combo.currentText().strip():
            self.species_combo.blockSignals(True)
            self.species_combo.setCurrentText("Deer")
            self.species_combo.blockSignals(False)

    @staticmethod
    def _normalize_tags_text(text: str) -> str:
        """Normalize comma/line separated tags to a clean comma-separated string."""
        parts = []
        for part in text.replace("\n", ",").split(","):
            p = part.strip()
            if p:
                parts.append(p)
        seen = set()
        out = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return ", ".join(out)

    def _populate_char_dropdown(self):
        """Fill characteristic dropdown from existing DB values."""
        try:
            options = []
            for photo in self._get_all_photos_cached():
                kc = photo.get("key_characteristics") or ""
                norm = self._normalize_tags_text(kc)
                for tok in norm.split(","):
                    t = tok.strip()
                    if t:
                        options.append(t)
            seen = set()
            unique = []
            for t in options:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
        except Exception:
            unique = []
        self.char_dropdown.clear()
        self.char_dropdown.addItem("")  # placeholder
        self.char_dropdown.addItems(unique)

    def _populate_camera_locations(self):
        """Fill camera location dropdown with existing locations."""
        if not hasattr(self, 'camera_combo'):
            return
        current_text = self.camera_combo.currentText()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem("")  # Empty = no location
        try:
            # Get unique camera locations from photos
            locations = set()
            for photo in self._get_all_photos_cached():
                loc = photo.get("camera_location")
                if loc and loc.strip():
                    locations.add(loc.strip())
            # Also get site names
            for site in self.db.get_all_sites(include_unconfirmed=True):
                name = site.get("name")
                if name and name.strip():
                    locations.add(name.strip())
            for loc in sorted(locations):
                self.camera_combo.addItem(loc)
        except Exception:
            pass
        # Try to restore selection
        if current_text:
            idx = self.camera_combo.findText(current_text)
            if idx >= 0:
                self.camera_combo.setCurrentIndex(idx)
            else:
                self.camera_combo.setCurrentText(current_text)
        self.camera_combo.blockSignals(False)
        # Also refresh quick select buttons
        self._populate_location_buttons()

    def _populate_location_buttons(self):
        """Create quick select buttons for known camera locations."""
        if not hasattr(self, 'location_buttons_layout'):
            return
        # Clear existing buttons
        while self.location_buttons_layout.count():
            item = self.location_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Get known locations from database
        try:
            locations = set()
            for photo in self._get_all_photos_cached():
                loc = photo.get("camera_location")
                if loc and loc.strip():
                    locations.add(loc.strip())
            # Create a button for each location
            for loc in sorted(locations):
                btn = QPushButton(loc)
                btn.setMaximumHeight(24)
                btn.setStyleSheet("padding: 2px 6px; font-size: 11px;")
                btn.clicked.connect(lambda checked, l=loc: self._quick_set_location(l))
                self.location_buttons_layout.addWidget(btn)
        except Exception:
            pass

    def _quick_set_location(self, location: str):
        """Set camera location and auto-save."""
        self.camera_combo.setCurrentText(location)
        self.schedule_save()

    def add_char_from_dropdown(self):
        """Append selected/typed characteristic into the field."""
        text = self.char_dropdown.currentText().strip()
        if not text:
            return
        existing = self._normalize_tags_text(self.char_edit.toPlainText())
        parts = [p.strip() for p in existing.split(",") if p.strip()]
        if text not in parts:
            parts.append(text)
        self.char_edit.setPlainText(", ".join(parts))
        self._refresh_char_tags()
        self.char_dropdown.setCurrentText("")

    def _refresh_char_tags(self):
        """Refresh the clickable tag buttons for key characteristics."""
        # Clear existing tags
        while self.char_tags_layout.count():
            item = self.char_tags_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get current characteristics
        text = self.char_edit.toPlainText()
        parts = [p.strip() for p in text.split(",") if p.strip()]

        # Create a button for each characteristic
        for char in parts:
            btn = QPushButton(f"{char} ✕")
            btn.setStyleSheet("""
                QPushButton {
                    background: #446;
                    color: white;
                    border: none;
                    border-radius: 10px;
                    padding: 4px 8px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background: #668;
                }
            """)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, c=char: self._remove_char_tag(c))
            self.char_tags_layout.addWidget(btn)

    def _remove_char_tag(self, char: str):
        """Remove a characteristic tag."""
        text = self.char_edit.toPlainText()
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if char in parts:
            parts.remove(char)
        self.char_edit.setPlainText(", ".join(parts))
        self._refresh_char_tags()
        self.save_current()

    def _populate_deer_id_dropdown(self):
        """Fill deer ID dropdown from existing IDs."""
        try:
            ids = set()
            for photo in self._get_all_photos_cached():
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
        current = self.deer_id_edit.currentText() if isinstance(self.deer_id_edit, QComboBox) else ""
        self.deer_id_edit.blockSignals(True)
        self.deer_id_edit.clear()
        self.deer_id_edit.addItem("")
        for did in sorted_ids:
            self.deer_id_edit.addItem(did)
        if current:
            self.deer_id_edit.setCurrentText(current)
        self.deer_id_edit.blockSignals(False)
        self._populate_additional_deer_dropdown()

    def _populate_additional_deer_dropdown(self):
        """Keep the second-buck dropdown in sync with known IDs."""
        try:
            ids = set()
            for photo in self._get_all_photos_cached():
                meta = self.db.get_deer_metadata(photo["id"])
                if meta.get("deer_id"):
                    ids.add(meta["deer_id"])
                for add in self.db.get_additional_deer(photo["id"]):
                    if add.get("deer_id"):
                        ids.add(add["deer_id"])
            sorted_ids = sorted(ids)
        except Exception:
            sorted_ids = []
        current = self.add_deer_id_edit.currentText() if isinstance(self.add_deer_id_edit, QComboBox) else ""
        self.add_deer_id_edit.blockSignals(True)
        self.add_deer_id_edit.clear()
        self.add_deer_id_edit.addItem("")
        for did in sorted_ids:
            self.add_deer_id_edit.addItem(did)
        if current:
            self.add_deer_id_edit.setCurrentText(current)
        self.add_deer_id_edit.blockSignals(False)

    def _populate_suggest_filter_options(self):
        """Fill the suggested-species filter combo."""
        if not hasattr(self, "suggest_filter_combo"):
            return
        current = self.suggest_filter_combo.currentData()
        self.suggest_filter_combo.blockSignals(True)
        self.suggest_filter_combo.clear()
        self.suggest_filter_combo.addItem("All photos", "")
        self.suggest_filter_combo.addItem("Has suggestion", "__has__")
        try:
            suggestions = sorted({p.get("suggested_tag") for p in self._get_all_photos_cached() if p.get("suggested_tag")})
        except Exception:
            suggestions = []
        for s in suggestions:
            if s:
                self.suggest_filter_combo.addItem(f"Suggested: {s}", s)
        idx = self.suggest_filter_combo.findData(current)
        self.suggest_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.suggest_filter_combo.blockSignals(False)

    def _populate_species_filter_options(self):
        """Fill the species filter combo with counts (contextual to other filters)."""
        if not hasattr(self, "species_filter_combo"):
            return
        # Save current selections
        current_selections = self.species_filter_combo.get_checked_data() if hasattr(self.species_filter_combo, 'get_checked_data') else []
        self.species_filter_combo._updating = True
        self.species_filter_combo.clear()
        # Add "All Species" as first item (always checked when all are selected)
        self.species_filter_combo.add_item("All Species", "", checked=True)
        self.species_filter_combo.set_all_text("All Species")
        try:
            # Count photos by species within current filter context
            context_photos = self._get_context_filtered_photos(exclude_filter='species')
            species_counts = {}
            unlabeled_count = 0
            for photo in context_photos:
                tags = set(self.db.get_tags(photo["id"]))
                species_tags = tags & VALID_SPECIES
                if not species_tags:
                    unlabeled_count += 1
                for sp in species_tags:
                    species_counts[sp] = species_counts.get(sp, 0) + 1
            if unlabeled_count > 0:
                self.species_filter_combo.add_item(f"Unlabeled ({unlabeled_count})", "__unlabeled__", checked=True)
            # Add only species with photos in current context
            for sp in sorted(species_counts.keys()):
                self.species_filter_combo.add_item(f"{sp} ({species_counts[sp]})", sp, checked=True)
        except Exception:
            self.species_filter_combo.add_item("Unlabeled", "__unlabeled__", checked=True)
            self.species_filter_combo.add_item("Deer", "Deer", checked=True)
        # Restore previous selections if any were deselected
        if current_selections:
            self.species_filter_combo.set_checked_data(current_selections)
        self.species_filter_combo._updating = False
        self.species_filter_combo._update_display()

    def _populate_sex_filter_options(self):
        """Fill the sex filter combo with counts (contextual to other filters)."""
        if not hasattr(self, "sex_filter_combo"):
            return
        current = self.sex_filter_combo.currentData()
        self.sex_filter_combo.blockSignals(True)
        self.sex_filter_combo.clear()
        self.sex_filter_combo.addItem("All", "")
        try:
            context_photos = self._get_context_filtered_photos(exclude_filter='sex')
            buck_count = 0
            doe_count = 0
            unknown_count = 0
            for photo in context_photos:
                tags = set(self.db.get_tags(photo["id"]))
                if "Buck" in tags:
                    buck_count += 1
                elif "Doe" in tags:
                    doe_count += 1
                elif "Unknown" in tags:
                    unknown_count += 1
            # Only add options with photos in current context
            if buck_count > 0:
                self.sex_filter_combo.addItem(f"Buck ({buck_count})", "Buck")
            if doe_count > 0:
                self.sex_filter_combo.addItem(f"Doe ({doe_count})", "Doe")
            if unknown_count > 0:
                self.sex_filter_combo.addItem(f"Unknown ({unknown_count})", "Unknown")
        except Exception:
            self.sex_filter_combo.addItem("Buck", "Buck")
            self.sex_filter_combo.addItem("Doe", "Doe")
            self.sex_filter_combo.addItem("Unknown", "Unknown")
        idx = self.sex_filter_combo.findData(current)
        self.sex_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.sex_filter_combo.blockSignals(False)

    def _populate_deer_id_filter_options(self):
        """Fill the deer ID filter combo with deer IDs (contextual to other filters)."""
        if not hasattr(self, "deer_id_filter_combo"):
            return
        current = self.deer_id_filter_combo.currentData()
        self.deer_id_filter_combo.blockSignals(True)
        self.deer_id_filter_combo.clear()
        self.deer_id_filter_combo.addItem("All Deer IDs", "")
        try:
            context_photos = self._get_context_filtered_photos(exclude_filter='deer_id')
            id_counts = {}
            has_id_count = 0
            no_id_count = 0
            for photo in context_photos:
                meta = self.db.get_deer_metadata(photo["id"])
                deer_id = meta.get("deer_id")
                if deer_id:
                    has_id_count += 1
                    id_counts[deer_id] = id_counts.get(deer_id, 0) + 1
                else:
                    no_id_count += 1
            # Only add options with photos in current context
            if has_id_count > 0:
                self.deer_id_filter_combo.addItem(f"Has ID ({has_id_count})", "__has_id__")
            if no_id_count > 0:
                self.deer_id_filter_combo.addItem(f"No ID ({no_id_count})", "__no_id__")
            for did in sorted(id_counts.keys()):
                self.deer_id_filter_combo.addItem(f"{did} ({id_counts[did]})", did)
        except Exception:
            self.deer_id_filter_combo.addItem("Has ID", "__has_id__")
            self.deer_id_filter_combo.addItem("No ID", "__no_id__")
        idx = self.deer_id_filter_combo.findData(current)
        self.deer_id_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.deer_id_filter_combo.blockSignals(False)

    def _populate_site_filter_options(self):
        """Fill the site filter combo with locations (contextual to other filters)."""
        if not hasattr(self, "site_filter_combo"):
            return
        current = self.site_filter_combo.currentData()
        self.site_filter_combo.blockSignals(True)
        self.site_filter_combo.clear()
        self.site_filter_combo.addItem("All locations", None)
        try:
            context_photos = self._get_context_filtered_photos(exclude_filter='site')
            # Count photos by camera_location
            location_counts = {}
            unassigned_count = 0
            for photo in context_photos:
                loc = photo.get("camera_location")
                if loc and loc.strip():
                    loc = loc.strip()
                    location_counts[loc] = location_counts.get(loc, 0) + 1
                else:
                    unassigned_count += 1
            # Add unassigned first if any
            if unassigned_count > 0:
                self.site_filter_combo.addItem(f"Unassigned ({unassigned_count})", "__unassigned__")
            # Add sorted locations with counts
            for loc in sorted(location_counts.keys()):
                count = location_counts[loc]
                label = f"{loc} ({count})"
                self.site_filter_combo.addItem(label, loc)
        except Exception:
            self.site_filter_combo.addItem("Unassigned", "__unassigned__")
        idx = self.site_filter_combo.findData(current)
        self.site_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.site_filter_combo.blockSignals(False)

    def _populate_year_filter_options(self):
        """Fill the year filter combo with years (contextual to other filters)."""
        if not hasattr(self, "year_filter_combo"):
            return
        current = self.year_filter_combo.currentData()
        self.year_filter_combo.blockSignals(True)
        self.year_filter_combo.clear()
        self.year_filter_combo.addItem("All Years", None)
        try:
            context_photos = self._get_context_filtered_photos(exclude_filter='year')
            # Count photos by antler year
            year_counts = {}
            for photo in context_photos:
                date_taken = photo.get("date_taken")
                if date_taken:
                    season_year = TrailCamDatabase.compute_season_year(date_taken)
                    if season_year:
                        year_counts[season_year] = year_counts.get(season_year, 0) + 1
            # Add sorted years (newest first) with counts and display format
            for year in sorted(year_counts.keys(), reverse=True):
                count = year_counts[year]
                label = f"{year}-{year + 1} ({count})"
                self.year_filter_combo.addItem(label, year)
        except Exception:
            pass
        idx = self.year_filter_combo.findData(current)
        self.year_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.year_filter_combo.blockSignals(False)

    def _populate_collection_filter_options(self):
        """Fill the collection filter combo (contextual to other filters)."""
        if not hasattr(self, "collection_filter_combo"):
            return
        current = self.collection_filter_combo.currentData()
        self.collection_filter_combo.blockSignals(True)
        self.collection_filter_combo.clear()
        self.collection_filter_combo.addItem("All Collections", None)
        try:
            context_photos = self._get_context_filtered_photos(exclude_filter='collection')
            # Count photos by collection
            collection_counts = {}
            unassigned_count = 0
            for photo in context_photos:
                coll = photo.get("collection") or ""
                if coll:
                    collection_counts[coll] = collection_counts.get(coll, 0) + 1
                else:
                    unassigned_count += 1
            # Add collections sorted by count (descending)
            for coll, count in sorted(collection_counts.items(), key=lambda x: -x[1]):
                self.collection_filter_combo.addItem(f"{coll} ({count})", coll)
            # Add unassigned option
            if unassigned_count > 0:
                self.collection_filter_combo.addItem(f"Unassigned ({unassigned_count})", "__unassigned__")
        except Exception:
            pass
        idx = self.collection_filter_combo.findData(current)
        self.collection_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.collection_filter_combo.blockSignals(False)

    def _filtered_photos(self):
        """Apply all filters to in-memory photo list."""
        result = list(enumerate(self.photos))

        # If navigating to a specific photo, bypass ALL filters
        if getattr(self, '_navigating_to_specific_photo', False):
            return result

        # If in queue mode, filter to only queue photos (bypass other filters)
        if self.queue_mode and self.queue_photo_ids:
            queue_set = set(self.queue_photo_ids)
            filtered = []
            for idx, p in result:
                if p.get("id") in queue_set:
                    filtered.append((idx, p))
            # Sort by queue order (preserve original queue order)
            id_to_order = {pid: i for i, pid in enumerate(self.queue_photo_ids)}
            filtered.sort(key=lambda x: id_to_order.get(x[1].get("id"), 999999))
            return filtered

        # Apply archive/favorites filter
        if hasattr(self, "archive_filter_combo"):
            archive_flt = self.archive_filter_combo.currentData()
            if archive_flt == "active":
                # Show only non-archived photos (default)
                result = [(idx, p) for idx, p in result if not p.get("archived")]
            elif archive_flt == "archived":
                # Show only archived photos
                result = [(idx, p) for idx, p in result if p.get("archived")]
            elif archive_flt == "favorites":
                # Show only favorites (non-archived)
                result = [(idx, p) for idx, p in result if p.get("favorite") and not p.get("archived")]
            # "all" shows everything, no filter needed

        sort_key = self.sort_combo.currentData() if hasattr(self, "sort_combo") else None
        needs_tags = False
        needs_deer = False

        if hasattr(self, "species_filter_combo"):
            if hasattr(self.species_filter_combo, 'get_checked_data'):
                selected_species = self.species_filter_combo.get_checked_data()
                total_options = self.species_filter_combo._model.rowCount() - 1 if hasattr(self.species_filter_combo, '_model') else 0
                if selected_species and len(selected_species) < total_options:
                    needs_tags = True
            else:
                species_flt = self.species_filter_combo.currentData()
                if species_flt:
                    needs_tags = True

        if hasattr(self, "sex_filter_combo"):
            if self.sex_filter_combo.currentData():
                needs_tags = True

        if hasattr(self, "deer_id_filter_combo"):
            if self.deer_id_filter_combo.currentData():
                needs_deer = True

        if sort_key == "species":
            needs_tags = True
        if sort_key == "deer_id":
            needs_deer = True

        tags_map = {}
        deer_map = {}
        if needs_tags or needs_deer:
            photo_ids = [p.get("id") for _, p in result if p.get("id") is not None]
            if needs_tags:
                tags_map = self.db.get_all_tags_batch(photo_ids)
            if needs_deer:
                deer_map = self.db.get_all_deer_metadata_batch(photo_ids)

        # Apply species filter (multi-select)
        if hasattr(self, "species_filter_combo"):
            # Check if it's a multi-select combo
            if hasattr(self.species_filter_combo, 'get_checked_data'):
                selected_species = self.species_filter_combo.get_checked_data()
                # Get total number of options (excluding "All")
                total_options = self.species_filter_combo._model.rowCount() - 1 if hasattr(self.species_filter_combo, '_model') else 0
                # Only filter if not all selected
                if selected_species and len(selected_species) < total_options:
                    filtered = []
                    selected_set = set(selected_species)
                    for idx, p in result:
                        tags = set(tags_map.get(p["id"], []))
                        species_tags = tags & VALID_SPECIES
                        # Check if "__unlabeled__" is selected
                        if "__unlabeled__" in selected_set and not species_tags:
                            filtered.append((idx, p))
                        elif species_tags & selected_set:
                            filtered.append((idx, p))
                    result = filtered
            else:
                # Fallback for regular combo
                species_flt = self.species_filter_combo.currentData()
                if species_flt:
                    filtered = []
                    for idx, p in result:
                        tags = set(tags_map.get(p["id"], []))
                        species_tags = tags & VALID_SPECIES
                        if species_flt == "__unlabeled__":
                            if not species_tags:
                                filtered.append((idx, p))
                        elif species_flt in species_tags:
                            filtered.append((idx, p))
                    result = filtered

        # Apply sex filter
        if hasattr(self, "sex_filter_combo"):
            sex_flt = self.sex_filter_combo.currentData()
            if sex_flt:
                filtered = []
                for idx, p in result:
                    tags = set(tags_map.get(p["id"], []))
                    photo_sex = ""
                    if "Buck" in tags:
                        photo_sex = "Buck"
                    elif "Doe" in tags:
                        photo_sex = "Doe"
                    elif "Unknown" in tags:
                        photo_sex = "Unknown"
                    if photo_sex == sex_flt:
                        filtered.append((idx, p))
                result = filtered

        # Apply deer ID filter
        if hasattr(self, "deer_id_filter_combo"):
            deer_flt = self.deer_id_filter_combo.currentData()
            if deer_flt:
                filtered = []
                for idx, p in result:
                    meta = deer_map.get(p["id"], {})
                    deer_id = meta.get("deer_id") or ""
                    if deer_flt == "__has_id__":
                        if deer_id:
                            filtered.append((idx, p))
                    elif deer_flt == "__no_id__":
                        if not deer_id:
                            filtered.append((idx, p))
                    elif deer_id == deer_flt:
                        filtered.append((idx, p))
                result = filtered

        # Apply suggestion filter
        if hasattr(self, "suggest_filter_combo"):
            flt = self.suggest_filter_combo.currentData()
            if flt:
                filtered = []
                for idx, p in result:
                    tag = p.get("suggested_tag")
                    if flt == "__has__":
                        if tag:
                            filtered.append((idx, p))
                    elif tag == flt:
                        filtered.append((idx, p))
                result = filtered

        # Apply site filter (check camera_location field)
        if hasattr(self, "site_filter_combo"):
            site_flt = self.site_filter_combo.currentData()
            if site_flt is not None:
                filtered = []
                for idx, p in result:
                    camera_loc = p.get("camera_location")
                    camera_loc = camera_loc.strip() if camera_loc else None
                    if site_flt == "__unassigned__":
                        # Unassigned = no camera_location set
                        if not camera_loc:
                            filtered.append((idx, p))
                    elif camera_loc == site_flt:
                        # Match camera_location string
                        filtered.append((idx, p))
                result = filtered

        # Apply year filter (antler year: May-April)
        if hasattr(self, "year_filter_combo"):
            year_flt = self.year_filter_combo.currentData()
            if year_flt is not None:
                filtered = []
                for idx, p in result:
                    date_taken = p.get("date_taken")
                    if date_taken:
                        season_year = TrailCamDatabase.compute_season_year(date_taken)
                        if season_year == year_flt:
                            filtered.append((idx, p))
                result = filtered

        # Apply collection filter
        if hasattr(self, "collection_filter_combo"):
            coll_flt = self.collection_filter_combo.currentData()
            if coll_flt is not None:
                filtered = []
                for idx, p in result:
                    photo_coll = p.get("collection") or ""
                    if coll_flt == "__unassigned__":
                        if not photo_coll:
                            filtered.append((idx, p))
                    elif photo_coll == coll_flt:
                        filtered.append((idx, p))
                result = filtered

        # Apply sorting
        if hasattr(self, "sort_combo"):
            sort_key = self.sort_combo.currentData()
            if sort_key == "date_desc":
                result.sort(key=lambda x: x[1].get("date_taken") or "", reverse=True)
            elif sort_key == "date_asc":
                result.sort(key=lambda x: x[1].get("date_taken") or "")
            elif sort_key == "location":
                result.sort(key=lambda x: (x[1].get("camera_location") or "zzz", x[1].get("date_taken") or ""))
            elif sort_key == "species":
                def get_species(p):
                    tags = tags_map.get(p["id"], [])
                    species_tags = set(tags) & VALID_SPECIES
                    return sorted(species_tags)[0] if species_tags else "zzz"
                result.sort(key=lambda x: (get_species(x[1]), x[1].get("date_taken") or ""))
            elif sort_key == "deer_id":
                def get_deer_id(p):
                    meta = deer_map.get(p["id"], {})
                    return meta.get("deer_id") or "zzz"
                result.sort(key=lambda x: (get_deer_id(x[1]), x[1].get("date_taken") or ""))

        return result

    def _get_context_filtered_photos(self, exclude_filter: str = None) -> list:
        """Get photos filtered by all active filters EXCEPT the specified one.

        This enables contextual filter dropdowns - showing only options that
        exist in the current filter context (e.g., only species found in the
        selected collection).

        Args:
            exclude_filter: One of 'species', 'sex', 'deer_id', 'site', 'year',
                          'collection', 'archive', 'suggest'. That filter is skipped.

        Returns:
            List of photo dicts matching all other active filters.
        """
        result = list(self.photos)

        # Skip if in queue mode - queue overrides all filters
        if self.queue_mode and self.queue_photo_ids:
            queue_set = set(self.queue_photo_ids)
            return [p for p in result if p.get("id") in queue_set]

        # Apply archive filter
        if exclude_filter != 'archive' and hasattr(self, "archive_filter_combo"):
            archive_flt = self.archive_filter_combo.currentData()
            if archive_flt == "active":
                result = [p for p in result if not p.get("archived")]
            elif archive_flt == "archived":
                result = [p for p in result if p.get("archived")]

        # Pre-fetch batch data to avoid N+1 queries
        needs_tags = False
        needs_deer = False

        if exclude_filter != 'species' and hasattr(self, "species_filter_combo"):
            if hasattr(self.species_filter_combo, 'get_checked_data'):
                selected_species = self.species_filter_combo.get_checked_data()
                total_options = self.species_filter_combo._model.rowCount() - 1 if hasattr(self.species_filter_combo, '_model') else 0
                if selected_species and len(selected_species) < total_options:
                    needs_tags = True
            else:
                if self.species_filter_combo.currentData():
                    needs_tags = True

        if exclude_filter != 'sex' and hasattr(self, "sex_filter_combo"):
            if self.sex_filter_combo.currentData():
                needs_tags = True

        if exclude_filter != 'deer_id' and hasattr(self, "deer_id_filter_combo"):
            if self.deer_id_filter_combo.currentData():
                needs_deer = True

        tags_map = {}
        deer_map = {}
        if needs_tags or needs_deer:
            photo_ids = [p.get("id") for p in result if p.get("id") is not None]
            if needs_tags:
                tags_map = self.db.get_all_tags_batch(photo_ids)
            if needs_deer:
                deer_map = self.db.get_all_deer_metadata_batch(photo_ids)

        # Apply species filter (multi-select)
        if exclude_filter != 'species' and hasattr(self, "species_filter_combo"):
            if hasattr(self.species_filter_combo, 'get_checked_data'):
                selected_species = self.species_filter_combo.get_checked_data()
                total_options = self.species_filter_combo._model.rowCount() - 1 if hasattr(self.species_filter_combo, '_model') else 0
                if selected_species and len(selected_species) < total_options:
                    filtered = []
                    selected_set = set(selected_species)
                    for p in result:
                        tags = set(tags_map.get(p["id"], []))
                        species_tags = tags & VALID_SPECIES
                        if "__unlabeled__" in selected_set and not species_tags:
                            filtered.append(p)
                        elif species_tags & selected_set:
                            filtered.append(p)
                    result = filtered
            else:
                species_flt = self.species_filter_combo.currentData()
                if species_flt:
                    filtered = []
                    for p in result:
                        tags = set(tags_map.get(p["id"], []))
                        species_tags = tags & VALID_SPECIES
                        if species_flt == "__unlabeled__":
                            if not species_tags:
                                filtered.append(p)
                        elif species_flt in species_tags:
                            filtered.append(p)
                    result = filtered

        # Apply sex filter
        if exclude_filter != 'sex' and hasattr(self, "sex_filter_combo"):
            sex_flt = self.sex_filter_combo.currentData()
            if sex_flt:
                filtered = []
                for p in result:
                    tags = set(tags_map.get(p["id"], []))
                    photo_sex = ""
                    if "Buck" in tags:
                        photo_sex = "Buck"
                    elif "Doe" in tags:
                        photo_sex = "Doe"
                    elif "Unknown" in tags:
                        photo_sex = "Unknown"
                    if photo_sex == sex_flt:
                        filtered.append(p)
                result = filtered

        # Apply deer ID filter
        if exclude_filter != 'deer_id' and hasattr(self, "deer_id_filter_combo"):
            deer_flt = self.deer_id_filter_combo.currentData()
            if deer_flt:
                filtered = []
                for p in result:
                    meta = deer_map.get(p["id"], {})
                    deer_id = meta.get("deer_id") or ""
                    if deer_flt == "__has_id__":
                        if deer_id:
                            filtered.append(p)
                    elif deer_flt == "__no_id__":
                        if not deer_id:
                            filtered.append(p)
                    elif deer_id == deer_flt:
                        filtered.append(p)
                result = filtered

        # Apply suggestion filter
        if exclude_filter != 'suggest' and hasattr(self, "suggest_filter_combo"):
            flt = self.suggest_filter_combo.currentData()
            if flt:
                filtered = []
                for p in result:
                    tag = p.get("suggested_tag")
                    if flt == "__has__":
                        if tag:
                            filtered.append(p)
                    elif tag == flt:
                        filtered.append(p)
                result = filtered

        # Apply site filter
        if exclude_filter != 'site' and hasattr(self, "site_filter_combo"):
            site_flt = self.site_filter_combo.currentData()
            if site_flt is not None:
                filtered = []
                for p in result:
                    camera_loc = p.get("camera_location")
                    camera_loc = camera_loc.strip() if camera_loc else None
                    if site_flt == "__unassigned__":
                        if not camera_loc:
                            filtered.append(p)
                    elif camera_loc == site_flt:
                        filtered.append(p)
                result = filtered

        # Apply year filter
        if exclude_filter != 'year' and hasattr(self, "year_filter_combo"):
            year_flt = self.year_filter_combo.currentData()
            if year_flt is not None:
                filtered = []
                for p in result:
                    date_taken = p.get("date_taken")
                    if date_taken:
                        season_year = TrailCamDatabase.compute_season_year(date_taken)
                        if season_year == year_flt:
                            filtered.append(p)
                result = filtered

        # Apply collection filter
        if exclude_filter != 'collection' and hasattr(self, "collection_filter_combo"):
            coll_flt = self.collection_filter_combo.currentData()
            if coll_flt is not None:
                filtered = []
                for p in result:
                    photo_coll = p.get("collection") or ""
                    if coll_flt == "__unassigned__":
                        if not photo_coll:
                            filtered.append(p)
                    elif photo_coll == coll_flt:
                        filtered.append(p)
                result = filtered

        return result

    def apply_buck_to_selected(self):
        """Bulk-assign current Deer ID to all selected photos."""
        deer_id = self.deer_id_edit.currentText().strip()
        if not deer_id:
            QMessageBox.information(self, "Bulk Set", "Select a Deer ID first.")
            return
        selected = self.photo_list_widget.selectedItems()
        if not selected:
            QMessageBox.information(self, "Bulk Set", "Select one or more photos in the list.")
            return
        for item in selected:
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is None or idx < 0 or idx >= len(self.photos):
                continue
            pid = self.photos[idx]["id"]
            self.db.set_deer_metadata(photo_id=pid, deer_id=deer_id)
        QMessageBox.information(self, "Bulk Set", f"Assigned Deer ID '{deer_id}' to {len(selected)} photo(s).")
        self._populate_deer_id_dropdown()

    def _apply_buck_profile_to_ui(self):
        """Autofill fields from buck profile (season-aware)."""
        deer_id = self.deer_id_edit.currentText().strip()
        if not deer_id:
            return
        # Determine season_year from photo
        photo = self._current_photo()
        if not photo:
            return
        season_year = photo.get("season_year")
        if season_year is None and photo.get("date_taken"):
            season_year = TrailCamDatabase.compute_season_year(photo.get("date_taken"))
        profile = self.db.get_buck_profile(deer_id, season_year)
        if not profile:
            return
        # Typical
        if profile.get("left_points_min") is not None:
            self._set_int_field(self.left_min, profile.get("left_points_min"))
            self.left_uncertain.setChecked(bool(profile.get("left_points_uncertain")))
        if profile.get("right_points_min") is not None:
            self._set_int_field(self.right_min, profile.get("right_points_min"))
            self.right_uncertain.setChecked(bool(profile.get("right_points_uncertain")))
        # Abnormal
        if profile.get("abnormal_points_min") is not None:
            self._set_int_field(self.ab_min, profile.get("abnormal_points_min"))
        if profile.get("abnormal_points_max") is not None:
            self._set_int_field(self.ab_max, profile.get("abnormal_points_max"))
        # Key characteristics
        kc = profile.get("key_characteristics") or ""
        if kc:
            self.char_edit.setPlainText(self._normalize_tags_text(kc))
            self._refresh_char_tags()
        # Camera location (only fill if empty)
        cam = profile.get("camera_locations") or ""
        if cam and not self.camera_combo.currentText().strip():
            self.camera_combo.setCurrentText(cam.split(",")[0].strip())
        # Defaults when buck ID present
        if deer_id:
            if not self.species_combo.currentText().strip():
                self.species_combo.setCurrentText("Deer")
            if self._get_sex() == "Unknown":
                self._set_sex("Buck")

    def _populate_photo_list(self):
        """Fill the navigation list with photos (sorted)."""
        # Refresh all filter dropdowns with contextual options
        # UNLESS we're navigating to a specific photo (filters are bypassed anyway)
        if not getattr(self, '_navigating_to_specific_photo', False):
            self._populate_suggest_filter_options()
            self._populate_species_filter_options()
            self._populate_sex_filter_options()
            self._populate_deer_id_filter_options()
            self._populate_site_filter_options()
            self._populate_year_filter_options()
            self._populate_collection_filter_options()
        self.photo_list_widget.blockSignals(True)
        self.photo_list_widget.clear()
        filtered = self._filtered_photos()
        filtered_indices = [idx for idx, _ in filtered]

        # If current photo not in filtered set, jump to first filtered photo
        # UNLESS we're navigating to a specific photo (e.g., from Properties button)
        jumped = False
        if filtered_indices and self.index not in filtered_indices:
            if not getattr(self, '_navigating_to_specific_photo', False):
                self.index = filtered_indices[0]
                jumped = True

        target_item = None
        for row, (idx, p) in enumerate(filtered):
            display, is_suggestion = self._build_photo_label(idx)
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.photo_list_widget.addItem(item)
            if idx == self.index:
                target_item = item
            # Red text for AI suggestions (unverified)
            if is_suggestion:
                item.setForeground(QColor(200, 50, 50))  # Red
            # Green highlight for reviewed items in AI box review mode
            if self.ai_review_mode and p.get("id") in self.ai_reviewed_photos:
                item.setBackground(QColor(144, 238, 144))  # Light green
            # Green highlight for reviewed items in queue mode
            elif self.queue_mode and p.get("id") in self.queue_reviewed:
                item.setBackground(QColor(144, 238, 144))  # Light green

        self.photo_list_widget.blockSignals(False)

        # Set current item AFTER unblocking signals so it sticks
        if target_item:
            self.photo_list_widget.setCurrentItem(target_item)
            target_item.setSelected(True)
            self.photo_list_widget.scrollToItem(target_item)
            self.photo_list_widget.setFocus()
        elif self.photo_list_widget.count() > 0:
            first_item = self.photo_list_widget.item(0)
            self.photo_list_widget.setCurrentItem(first_item)
            first_item.setSelected(True)
            self.photo_list_widget.setFocus()
            jumped = True
            if filtered_indices:
                self.index = filtered_indices[0]

        # Load photo if we jumped to a different one (but not during init)
        if jumped and filtered_indices and hasattr(self, 'preview_layout'):
            self.load_photo()

    def on_photo_selection_changed(self):
        """Load the first selected photo; keep selection aligned."""
        # Guard against calls during init before UI is fully built
        if not hasattr(self, 'preview_layout'):
            return
        selected = self.photo_list_widget.selectedItems()
        if not selected:
            return
        idx = selected[0].data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        # Save pending changes before switching photos
        if self.save_timer.isActive():
            self.save_timer.stop()
            self.save_current()
        self.index = idx
        self.load_photo()

    def compare_selected(self):
        """Open compare window for selected photos (up to 4)."""
        selected = self.photo_list_widget.selectedItems()
        if not selected or len(selected) < 2:
            QMessageBox.information(self, "Compare", "Select at least two photos in the list.")
            return
        photo_ids = []
        for item in selected[:4]:
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is None or idx < 0 or idx >= len(self.photos):
                continue
            pid = self.photos[idx].get("id")
            if pid:
                photo_ids.append(pid)
        if len(photo_ids) < 2:
            QMessageBox.information(self, "Compare", "Need at least two valid photos.")
            return
        dlg = CompareWindow(photo_ids=photo_ids, db=self.db, parent=self)
        dlg.exec()

    def toggle_mark_for_compare(self):
        """Toggle mark on current photo for comparison."""
        if not self.photos or self.index < 0 or self.index >= len(self.photos):
            return
        photo_id = self.photos[self.index].get("id")
        if not photo_id:
            return
        if photo_id in self.marked_for_compare:
            self.marked_for_compare.discard(photo_id)
        else:
            if len(self.marked_for_compare) >= 4:
                QMessageBox.information(self, "Mark Limit", "Maximum 4 photos can be marked for comparison.")
                return
            self.marked_for_compare.add(photo_id)
        self._update_mark_button_state()
        self._update_photo_list_marks()

    def compare_marked_photos(self):
        """Open compare window with all marked photos."""
        if len(self.marked_for_compare) < 2:
            QMessageBox.information(self, "Compare", "Mark at least 2 photos first (press M or click 'Mark').")
            return
        photo_ids = list(self.marked_for_compare)[:4]
        dlg = CompareWindow(photo_ids=photo_ids, db=self.db, parent=self)
        dlg.exec()

    def clear_marked_photos(self):
        """Clear all photos marked for comparison."""
        self.marked_for_compare.clear()
        self._update_mark_button_state()
        self._update_photo_list_marks()

    def _update_mark_button_state(self):
        """No-op - mark buttons removed from UI."""
        pass

    def _update_photo_list_marks(self):
        """Update photo list to show which photos are marked."""
        for row in range(self.photo_list_widget.count()):
            item = self.photo_list_widget.item(row)
            if not item:
                continue
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is None or idx >= len(self.photos):
                continue
            photo_id = self.photos[idx].get("id")
            text = item.text()
            # Remove existing mark indicator
            if text.startswith("★ "):
                text = text[2:]
            # Add mark indicator if marked
            if photo_id and photo_id in self.marked_for_compare:
                item.setText("★ " + text)
                item.setBackground(QColor(60, 100, 60))  # Subtle green
            else:
                item.setText(text)
                item.setBackground(QColor(0, 0, 0, 0))  # Transparent

    def select_all_photos(self):
        """Select all photos in the list."""
        self.photo_list_widget.blockSignals(True)
        self.photo_list_widget.selectAll()
        self.photo_list_widget.blockSignals(False)
        self.on_photo_selection_changed()

    def clear_selection(self):
        """Clear list selection."""
        self.photo_list_widget.clearSelection()

    def _on_favorite_changed(self, state):
        """Toggle favorite status for current photo."""
        if not self.photos or self.index >= len(self.photos):
            return
        photo = self.photos[self.index]
        pid = photo.get("id")
        if not pid:
            return
        # In PyQt6, stateChanged passes int: 0=unchecked, 2=checked
        is_favorite = state == 2
        self.db.set_favorite(pid, is_favorite)
        photo["favorite"] = 1 if is_favorite else 0

    def archive_current_photo(self):
        """Archive the current photo or selected photos."""
        selected = self.photo_list_widget.selectedItems()
        if selected and len(selected) > 1:
            # Archive multiple selected photos
            photo_ids = []
            for item in selected:
                idx = item.data(Qt.ItemDataRole.UserRole)
                if idx is not None and idx < len(self.photos):
                    pid = self.photos[idx].get("id")
                    if pid:
                        photo_ids.append(pid)
            if photo_ids:
                self.db.archive_photos(photo_ids)
                # Update in-memory data (skip favorites - they're protected)
                archived_count = 0
                skipped_favorites = 0
                for p in self.photos:
                    if p.get("id") in photo_ids:
                        if p.get("favorite"):
                            skipped_favorites += 1
                        else:
                            p["archived"] = 1
                            archived_count += 1
                self._populate_photo_list()
                if skipped_favorites > 0:
                    QMessageBox.information(self, "Archived", f"Archived {archived_count} photos.\n{skipped_favorites} favorites were protected.")
                else:
                    QMessageBox.information(self, "Archived", f"Archived {archived_count} photos.")
        elif self.photos and 0 <= self.index < len(self.photos):
            # Archive single current photo
            photo = self.photos[self.index]
            if photo.get("favorite"):
                QMessageBox.information(self, "Protected", "This photo is a favorite and cannot be archived.")
                return
            pid = photo.get("id")
            if pid:
                self.db.archive_photo(pid)
                photo["archived"] = 1
                self._populate_photo_list()
                # Move to next photo if available
                if self.index < len(self.photos) - 1:
                    self.next_photo()

    def unarchive_current_photo(self):
        """Unarchive the current photo or selected photos."""
        selected = self.photo_list_widget.selectedItems()
        if selected and len(selected) > 1:
            # Unarchive multiple selected photos
            photo_ids = []
            for item in selected:
                idx = item.data(Qt.ItemDataRole.UserRole)
                if idx is not None and idx < len(self.photos):
                    pid = self.photos[idx].get("id")
                    if pid:
                        photo_ids.append(pid)
            if photo_ids:
                for pid in photo_ids:
                    self.db.unarchive_photo(pid)
                # Update in-memory data
                for p in self.photos:
                    if p.get("id") in photo_ids:
                        p["archived"] = 0
                self._populate_photo_list()
                QMessageBox.information(self, "Unarchived", f"Restored {len(photo_ids)} photos.")
        elif self.photos and 0 <= self.index < len(self.photos):
            # Unarchive single current photo
            photo = self.photos[self.index]
            pid = photo.get("id")
            if pid:
                self.db.unarchive_photo(pid)
                photo["archived"] = 0
                self._populate_photo_list()

    def _select_range(self, start: int, end: int):
        if self.photo_list_widget.count() == 0:
            return
        start = max(0, min(start, self.photo_list_widget.count() - 1))
        end = max(0, min(end, self.photo_list_widget.count() - 1))
        lo, hi = sorted([start, end])
        self.photo_list_widget.blockSignals(True)
        self.photo_list_widget.clearSelection()
        for i in range(lo, hi + 1):
            item = self.photo_list_widget.item(i)
            if item:
                item.setSelected(True)
        self.photo_list_widget.blockSignals(False)

    def select_to_end(self):
        """Select from current row to last."""
        row = self.photo_list_widget.currentRow()
        if row < 0:
            row = 0
        self._select_range(row, self.photo_list_widget.count() - 1)

    def select_to_start(self):
        """Select from current row to first."""
        row = self.photo_list_widget.currentRow()
        if row < 0:
            row = 0
        self._select_range(0, row)

    def _photo_has_ai_boxes(self, boxes: List[dict]) -> bool:
        return any(str(b.get("label", "")).startswith("ai_") for b in boxes)

    def _ai_queue_indices(self) -> List[int]:
        indices = []
        for idx, p in enumerate(self.photos):
            try:
                boxes = self.db.get_boxes(p["id"])
            except Exception:
                continue
            if self._photo_has_ai_boxes(boxes):
                indices.append(idx)
        return indices

    def start_ai_review(self):
        """Enter AI review mode: step through photos with AI boxes until resolved."""
        # Reload ALL photos first (clear any filters) so we find all AI boxes
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_photo_list()

        queue = self._ai_queue_indices()
        if not queue:
            QMessageBox.information(self, "AI Review", "No photos with AI boxes to review.")
            return

        # Disable queue mode if active — only one review system at a time
        if self.queue_mode:
            self.queue_mode = False
            self.queue_type = None
            self.queue_photo_ids = []
            self.queue_data = {}
            self.queue_reviewed = set()
            self.queue_panel.hide()

        self.ai_review_mode = True
        self.ai_review_queue = queue
        self.ai_reviewed_photos = set()  # Clear reviewed tracking
        self._advancing_review = False  # Reset guard
        # Show box controls during review
        self.box_container_top.show()
        self.box_container_bottom.show()
        # jump to first queued photo
        self.index = queue[0]
        self.load_photo()
        QMessageBox.information(self, "AI Review", f"Reviewing {len(queue)} photo(s) with AI boxes.\nUse Accept/Reject to move to next.")

    def _advance_ai_review(self):
        """Move to next photo in queue; exit if none left."""
        if not self.ai_review_mode:
            return
        # Prevent recursive calls (from load_photo -> _advance_ai_review -> load_photo)
        if self._advancing_review:
            return
        self._advancing_review = True
        try:
            # Mark current photo as reviewed for green highlighting
            if self.index < len(self.photos):
                current_pid = self.photos[self.index].get("id")
                if current_pid:
                    self.ai_reviewed_photos.add(current_pid)
            # Refresh list to show green highlighting on reviewed items
            self._populate_photo_list()
            remaining = [i for i in self._ai_queue_indices() if i > self.index]
            if not remaining:
                self.ai_review_mode = False
                self.ai_review_queue = []
                # Hide box controls when review complete
                self.box_container_top.hide()
                self.box_container_bottom.hide()
                QMessageBox.information(self, "AI Review", "AI review complete.")
                return
            self.index = remaining[0]
            self.load_photo()
        finally:
            self._advancing_review = False

    def _on_photo_item_pressed(self, item: QListWidgetItem):
        """Enable shift-click range selection (and Ctrl+Shift)."""
        if item is None:
            return
        row = self.photo_list_widget.row(item)
        mods = QApplication.keyboardModifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            anchor = self._last_click_row if self._last_click_row >= 0 else self.photo_list_widget.currentRow()
            if anchor < 0:
                anchor = row
            self._select_range(anchor, row)
            self._last_click_row = row
        else:
            # regular click, update anchor
            self._last_click_row = row

    def apply_species_to_selected(self):
        """Bulk-assign current species to selected photos."""
        species = self.species_combo.currentText().strip()
        if not species:
            QMessageBox.information(self, "Bulk Set", "Select a species first.")
            return
        selected = self.photo_list_widget.selectedItems()
        if not selected:
            QMessageBox.information(self, "Bulk Set", "Select one or more photos in the list.")
            return
        count = 0
        for item in selected:
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is None or idx < 0 or idx >= len(self.photos):
                continue
            pid = self.photos[idx]["id"]
            tags = set(self.db.get_tags(pid))
            tags.add(species)
            self.db.update_photo_tags(pid, list(tags))
            self.db.set_suggested_tag(pid, species, None)
            count += 1
        QMessageBox.information(self, "Bulk Set", f"Assigned species '{species}' to {count} photo(s).")
        self._populate_photo_list()

    def _build_photo_label(self, idx: int) -> tuple:
        """Build list label: date + most specific identifier (buck ID > buck/doe > species).
        Returns (display_string, is_suggestion) tuple."""
        if idx < 0 or idx >= len(self.photos):
            return ("", False)
        p = self.photos[idx]
        pid = p.get("id")
        # Format date as m/d/yyyy time
        date_taken = p.get("date_taken")
        if date_taken:
            try:
                from datetime import datetime
                # Parse ISO format or common formats
                if "T" in date_taken:
                    dt = datetime.fromisoformat(date_taken.replace("Z", "+00:00"))
                else:
                    dt = datetime.strptime(date_taken, "%Y-%m-%d %H:%M:%S")
                label = dt.strftime("%-m/%-d/%Y %-I:%M %p")
            except Exception:
                label = date_taken
        else:
            label = os.path.basename(p.get("file_path", ""))

        # Get the most specific identifier: buck ID > buck/doe > species
        detail = ""
        is_suggestion = False
        try:
            # First priority: Buck ID (verified)
            deer = self.db.get_deer_metadata(pid)
            deer_id = (deer.get("deer_id") or "").strip() if deer else ""
            if deer_id:
                detail = deer_id
            else:
                # Second priority: Buck or Doe tag (verified)
                tags = set(self.db.get_tags(pid))
                if "Buck" in tags:
                    detail = "Buck"
                elif "Doe" in tags:
                    detail = "Doe"
                else:
                    # Third priority: Species (verified)
                    species_labels = set(SPECIES_OPTIONS)
                    try:
                        species_labels.update(self.db.list_custom_species())
                    except Exception:
                        pass
                    species_found = [t for t in tags if t in species_labels]
                    if species_found:
                        detail = species_found[0]
                    else:
                        # Fourth priority: AI suggestion (not verified)
                        suggested = p.get("suggested_tag", "")
                        if suggested and suggested in species_labels:
                            detail = suggested
                            is_suggestion = True
        except Exception:
            pass

        # In queue mode, append the suggestion to the display
        queue_suffix = ""
        if self.queue_mode and pid in self.queue_data:
            data = self.queue_data[pid]
            if self.queue_type == "species":
                species = data.get("species", "")
                conf = data.get("conf", 0)
                if species:
                    queue_suffix = f" — {species} ({conf:.0%})"
            elif self.queue_type == "sex":
                sex = data.get("sex", "")
                conf = data.get("conf", 0)
                if sex:
                    queue_suffix = f" — {sex} ({conf:.0%})"

        if detail:
            display = f"{label} - {detail}{queue_suffix}"
        else:
            display = f"{label}{queue_suffix}"
        return display, is_suggestion

    def _update_photo_list_item(self, idx: int):
        """Update a single list item label without resetting selection."""
        for i in range(self.photo_list_widget.count()):
            item = self.photo_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == idx:
                display, is_suggestion = self._build_photo_label(idx)
                item.setText(display)
                if is_suggestion:
                    item.setForeground(QColor(200, 50, 50))  # Red for unverified
                else:
                    # Reset to default (no explicit color - lets selection styling work)
                    item.setData(Qt.ItemDataRole.ForegroundRole, None)
                break

    # closeEvent is defined earlier in the class - removed duplicate

    def _select_photo_by_id(self, photo_id: Optional[int]):
        """Select and load a photo by DB id."""
        if photo_id is None:
            return
        for idx, p in enumerate(self.photos):
            if p.get("id") == photo_id:
                self.photo_list_widget.blockSignals(True)
                self.photo_list_widget.clearSelection()
                item = self.photo_list_widget.item(idx)
                if item:
                    item.setSelected(True)
                self.photo_list_widget.blockSignals(False)
                self.index = idx
                self.load_photo()
                return

    def _import_files(self, files: List[Path], skip_hash: bool = True, collection: str = "",
                      progress_callback=None, site_mappings: dict = None) -> int:
        """Copy files into the library, add to DB, and build thumbnails.

        Args:
            files: List of file paths to import
            skip_hash: If True, skip files that match existing hashes
            collection: Collection/farm name to assign to imported photos
            progress_callback: Optional callable(current, total, filename) for progress updates.
                               If it returns False, import is cancelled.
            site_mappings: Optional dict mapping camera IDs to site names
        """
        if site_mappings is None:
            site_mappings = {}
        if self._known_hashes is None:
            self._known_hashes = self._load_known_hashes()
        imported = 0
        total = len(files)
        for i, file_path in enumerate(files):
            # Update progress if callback provided
            if progress_callback:
                progress_callback(i, total, file_path.name)
                # Check if cancelled (callback sets self._import_cancelled flag)
                if hasattr(self, '_import_cancelled') and self._import_cancelled:
                    break

            # Compute file hash BEFORE importing to check for duplicates
            file_hash = self._hash_file(file_path)

            # If hash calculation failed, skip the file to prevent duplicates
            if not file_hash:
                logger.warning(f"Skipping {file_path.name}: could not compute file hash")
                continue

            # Skip if hash already exists (prevents importing duplicate photos)
            if file_hash in self._known_hashes:
                continue
            # Also check database (catches duplicates across sessions)
            if self.db.photo_exists_by_hash(file_hash):
                self._known_hashes.add(file_hash)
                continue

            # Validate file is actually an image (skip AppleDouble files, corrupted files, etc.)
            try:
                from PIL import Image
                with Image.open(str(file_path)) as img:
                    img.verify()
            except Exception:
                logger.warning(f"Skipping non-image file: {file_path}")
                continue

            try:
                dest_path, original_name, date_taken, camera_model = import_photo(str(file_path))
            except Exception as exc:
                logger.error(f"Import failed for {file_path}: {exc}")
                continue
            if self.db.get_photo_id(dest_path):
                # Already imported by file path.
                continue

            # Extract camera ID from CuddeLink EXIF (before thumbnail creation in case file moves)
            camera_id = extract_cuddelink_camera_id(str(file_path))
            # Map camera ID to site name (use mapping if provided, else use raw camera ID)
            camera_location = site_mappings.get(camera_id, camera_id) if camera_id else None

            thumb_path = create_thumbnail(dest_path, file_hash=file_hash)
            try:
                # Pass file_hash to add_photo to store it immediately and enable duplicate detection
                photo_id = self.db.add_photo(
                    dest_path, original_name, date_taken or "", camera_model or "",
                    thumb_path, collection=collection, file_hash=file_hash
                )

                # If add_photo returned None, it was a duplicate (shouldn't happen due to earlier check, but safety net)
                if photo_id is None:
                    if file_hash:
                        self._known_hashes.add(file_hash)
                    continue

                # Set camera location if extracted from CuddeLink EXIF
                if camera_location and photo_id:
                    self.db.update_photo_attributes(photo_id, camera_location=camera_location)

                # Auto-detect verification photos (small file size < 15 KB)
                # These are camera test shots, not real photos - mark before AI runs
                # Only mark if file is a valid image (defense against corrupted/non-image files)
                if photo_id:
                    try:
                        file_size_kb = os.path.getsize(dest_path) / 1024
                        if file_size_kb < 15:
                            # Verify it's actually a valid image before marking as Verification
                            from PIL import Image
                            with Image.open(dest_path) as img:
                                img.verify()
                            self.db.set_suggested_tag(photo_id, "Verification", 0.99)
                    except:
                        pass  # Not a valid image or can't check - don't mark as Verification

                if file_hash:
                    self._known_hashes.add(file_hash)
                imported += 1

                # Queue for R2 upload (hash is already stored in database)
                if hasattr(self, 'r2_manager') and file_hash:
                    self.r2_manager.queue_photo(
                        photo_id=photo_id,
                        file_hash=file_hash,
                        file_path=Path(dest_path),
                        thumbnail_path=Path(thumb_path) if thumb_path else None
                    )
            except Exception as exc:
                logger.error(f"DB insert failed for {dest_path}: {exc}")

        # Start background R2 upload after import batch completes
        if imported > 0 and hasattr(self, 'r2_manager') and self.r2_manager.pending_count > 0:
            self.r2_manager.start_background_upload()

        return imported

    def _hash_file(self, path: Path) -> Optional[str]:
        """MD5 of a file; returns None on failure."""
        try:
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as exc:
            logger.warning(f"Hash failed for {path}: {exc}")
            return None

    def _load_known_hashes(self) -> set:
        """Compute hashes for existing photos once to skip duplicates upfront."""
        hashes = set()
        failures = 0
        for p in self.db.get_all_photos():
            fp = p.get("file_path")
            if not fp:
                continue
            try:
                h = self._hash_file(Path(fp))
                if h:
                    hashes.add(h)
            except Exception:
                failures += 1
                continue
        return hashes

    def _ensure_detection_boxes(self, photo: dict, detector=None, names=None) -> bool:
        """Run detector on photo if it has no boxes. Returns True if boxes exist after."""
        pid = photo.get("id")
        if not pid:
            return False
        # Check if photo already has boxes
        try:
            if self.db.has_detection_boxes(pid):
                return True
        except Exception:
            pass
        # Run detection
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return False
        boxes = self._detect_boxes_for_path(path, detector=detector, names=names, conf_thresh=0.25)
        if boxes:
            try:
                self.db.set_boxes(pid, boxes)
                return True
            except Exception:
                pass
        return False

    def _get_deer_head_detector(self):
        """Get a cached deer head detector (custom YOLO model). Returns (detector, names) tuple."""
        model_path = self._get_det_model_path()
        if not model_path:
            return None, None
        try:
            from ultralytics import YOLO
            if not hasattr(self, "_deer_head_detector") or getattr(self, "_deer_head_detector_path", None) != model_path:
                self._deer_head_detector = YOLO(str(model_path))
                self._deer_head_detector_path = model_path
            detector = self._deer_head_detector
            names = getattr(detector, "names", {}) if hasattr(detector, "names") else None
            return detector, names
        except Exception:
            return None, None

    # Keep old name for backwards compatibility
    def _get_detector_for_suggestions(self):
        return self._get_deer_head_detector()

    def _best_crop_for_photo(self, photo: dict):
        """Return (temp_file_path, pixel_area) of the best crop (deer_head > ai_animal > subject) or (None, None)."""
        pid = photo.get("id")
        if not pid:
            return None, None
        boxes = []
        try:
            boxes = self.db.get_boxes(pid)
        except Exception:
            boxes = []
        if not boxes:
            return None, None
        # prefer deer_head, then ai_animal (MegaDetector), then subject/ai_subject
        chosen = None
        for b in boxes:
            if b.get("label") == "deer_head":
                chosen = b
                break
        if chosen is None:
            for b in boxes:
                if b.get("label") == "ai_animal":
                    chosen = b
                    break
        if chosen is None:
            for b in boxes:
                if str(b.get("label")).endswith("subject"):
                    chosen = b
                    break
        if chosen is None:
            return None, None
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return None, None
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            w, h = img.size
            x1 = int(chosen["x1"] * w); x2 = int(chosen["x2"] * w)
            y1 = int(chosen["y1"] * h); y2 = int(chosen["y2"] * h)
            x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
            if x2 - x1 < 5 or y2 - y1 < 5:
                return None, None
            pixel_area = (x2 - x1) * (y2 - y1)
            import tempfile
            tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
            img.crop((x1, y1, x2, y2)).save(tmp, "JPEG", quality=90)
            return tmp, pixel_area
        except Exception:
            return None, None

    def _crop_box(self, photo: dict, box: dict) -> Optional[Path]:
        """Crop a specific box from a photo and return temp file path."""
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return None
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            w, h = img.size
            x1 = int(box["x1"] * w); x2 = int(box["x2"] * w)
            y1 = int(box["y1"] * h); y2 = int(box["y2"] * h)
            x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
            if x2 - x1 < 5 or y2 - y1 < 5:
                return None
            import tempfile
            tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
            img.crop((x1, y1, x2, y2)).save(tmp, "JPEG", quality=90)
            return tmp
        except Exception:
            return None

    def _best_head_crop_for_photo(self, photo: dict) -> Optional[Path]:
        """Return a temp file path of deer crop for buck/doe classification, or None.

        Priority order:
        1. deer_head (human-labeled)
        2. ai_deer_head (AI-detected head)
        3. subject box with species=Deer (human-labeled body)
        4. ai_animal box (AI-detected body) - buck/doe v2.0 was trained on these
        """
        pid = photo.get("id")
        if not pid:
            return None
        boxes = []
        try:
            boxes = self.db.get_boxes(pid)
        except Exception:
            boxes = []
        if not boxes:
            return None
        # Priority 1: deer_head boxes (human-labeled)
        chosen = None
        for b in boxes:
            if b.get("label") == "deer_head":
                chosen = b
                break
        # Priority 2: ai_deer_head (AI-detected head)
        if chosen is None:
            for b in boxes:
                if b.get("label") == "ai_deer_head":
                    chosen = b
                    break
        # Priority 3: subject box with species=Deer (human-labeled body)
        if chosen is None:
            for b in boxes:
                if b.get("label") == "subject" and b.get("species", "").lower() == "deer":
                    chosen = b
                    break
        # Priority 4: ai_animal box with deer species (buck/doe v2.0 trained on these)
        # Only use ai_animal if species is deer or unset (caller should verify deer)
        if chosen is None:
            for b in boxes:
                if b.get("label") == "ai_animal":
                    box_species = (b.get("species") or "").lower()
                    if box_species in ("deer", ""):
                        chosen = b
                        break
        if chosen is None:
            return None
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return None
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            w, h = img.size
            x1 = int(chosen["x1"] * w); x2 = int(chosen["x2"] * w)
            y1 = int(chosen["y1"] * h); y2 = int(chosen["y2"] * h)
            # Add 10% padding for better context
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x); x2 = min(w, x2 + pad_x)
            y1 = max(0, y1 - pad_y); y2 = min(h, y2 + pad_y)
            if x2 - x1 < 32 or y2 - y1 < 32:
                return None
            import tempfile
            tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
            img.crop((x1, y1, x2, y2)).save(tmp, "JPEG", quality=90)
            return tmp
        except Exception:
            return None

    def _predict_sex_for_deer_boxes(self, photo: dict) -> int:
        """Run buck/doe prediction on all deer boxes in a photo.

        Returns the number of boxes that got sex predictions.
        Only runs on boxes where species='Deer' (case-insensitive).
        Uses deer_head crops when available for better accuracy.
        """
        if not self.suggester or not self.suggester.buckdoe_ready:
            return 0

        pid = photo.get("id")
        if not pid:
            return 0

        boxes = self.db.get_boxes(pid)
        if not boxes:
            return 0

        # Find deer boxes that need sex prediction
        deer_boxes = []
        head_boxes = {}  # Map head boxes for potential association

        for box in boxes:
            species = (box.get("species") or "").lower()
            ai_species = (box.get("ai_suggested_species") or "").lower()
            label = (box.get("label") or "").lower()

            # Skip if already has sex prediction
            if box.get("sex"):
                continue

            # Identify deer boxes (by species or AI suggestion)
            if species == "deer" or ai_species == "deer":
                deer_boxes.append(box)

            # Track head boxes for potential association
            if label in ("deer_head", "ai_deer_head"):
                head_boxes[box["id"]] = box

        if not deer_boxes:
            return 0

        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return 0

        count = 0
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            w, h = img.size

            for deer_box in deer_boxes:
                crop_path = None
                try:
                    # Try to find associated head box within this deer box
                    best_head = None
                    deer_x1, deer_y1 = deer_box["x1"], deer_box["y1"]
                    deer_x2, deer_y2 = deer_box["x2"], deer_box["y2"]

                    for hbox in head_boxes.values():
                        hx1, hy1, hx2, hy2 = hbox["x1"], hbox["y1"], hbox["x2"], hbox["y2"]
                        # Check if head box center is within deer box
                        hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                        if deer_x1 <= hcx <= deer_x2 and deer_y1 <= hcy <= deer_y2:
                            best_head = hbox
                            break

                    # Crop either head box (preferred) or deer box
                    box_to_crop = best_head if best_head else deer_box
                    x1 = int(box_to_crop["x1"] * w)
                    x2 = int(box_to_crop["x2"] * w)
                    y1 = int(box_to_crop["y1"] * h)
                    y2 = int(box_to_crop["y2"] * h)

                    # Add padding for context
                    pad_x = int((x2 - x1) * 0.1)
                    pad_y = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - pad_x)
                    x2 = min(w, x2 + pad_x)
                    y1 = max(0, y1 - pad_y)
                    y2 = min(h, y2 + pad_y)

                    if x2 - x1 < 32 or y2 - y1 < 32:
                        continue

                    import tempfile
                    crop_path = Path(tempfile.mkstemp(suffix=".jpg")[1])
                    img.crop((x1, y1, x2, y2)).save(crop_path, "JPEG", quality=90)

                    # Run prediction
                    sex_res = self.suggester.predict_sex(str(crop_path))
                    if sex_res:
                        sex_label, sex_conf = sex_res
                        self.db.set_box_sex(deer_box["id"], sex_label, sex_conf)
                        count += 1

                finally:
                    if crop_path:
                        try:
                            Path(crop_path).unlink(missing_ok=True)
                        except Exception:
                            pass

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error in _predict_sex_for_deer_boxes: {e}")

        return count

    # ====== FILE/TOOLS/SETTINGS INTEGRATIONS ======
    def import_folder(self):
        """Import photos from a folder."""
        dlg = QFileDialog(self, "Select Folder to Import")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        # Add skip-duplicates checkbox when using non-native dialog; guard layout
        skip_dup_checkbox = QCheckBox("Skip duplicates (hash check)")
        layout = dlg.layout()
        if layout:
            layout.addWidget(skip_dup_checkbox)
        if not dlg.exec():
            return
        selected = dlg.selectedFiles()
        folder = selected[0] if selected else ""
        if not folder:
            return
        folder_path = Path(folder)
        jpg_files = self._find_image_files(folder_path)
        if not jpg_files:
            QMessageBox.information(self, "No Images", "No JPG/JPEG files found in the selected folder.")
            return

        # Ask for collection before import
        dialog = QDialog(self)
        dialog.setWindowTitle("Import Photos")
        dialog.setMinimumWidth(350)
        dlg_layout = QVBoxLayout(dialog)

        dlg_layout.addWidget(QLabel(f"Found {len(jpg_files)} photo(s) in:\n{folder_path.name}"))

        # Collection dropdown
        collection_layout = QHBoxLayout()
        collection_layout.addWidget(QLabel("Collection/Farm:"))
        collection_combo = QComboBox()
        collection_combo.setEditable(True)
        collection_combo.setPlaceholderText("Select or type a collection name")
        existing_collections = self.db.get_distinct_collections()
        collection_combo.addItems(existing_collections)
        if existing_collections:
            collection_combo.setCurrentIndex(0)  # Default to first collection
        collection_layout.addWidget(collection_combo)
        dlg_layout.addLayout(collection_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        import_btn = QPushButton("Import")
        import_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(import_btn)
        dlg_layout.addLayout(btn_layout)

        cancel_btn.clicked.connect(dialog.reject)
        import_btn.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_collection = collection_combo.currentText().strip()

        # Create progress dialog
        progress = QProgressDialog("Importing photos...", "Cancel", 0, len(jpg_files), self)
        progress.setWindowTitle("Import Progress")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._import_cancelled = False

        def update_progress(current, total, filename):
            if progress.wasCanceled():
                self._import_cancelled = True
                return
            progress.setValue(current)
            progress.setLabelText(f"Importing {current + 1} of {total}:\n{filename}")
            QApplication.processEvents()

        imported = self._import_files(jpg_files, skip_hash=skip_dup_checkbox.isChecked(), collection=selected_collection, progress_callback=update_progress)
        progress.setValue(len(jpg_files))
        progress.close()

        if self._import_cancelled:
            QMessageBox.information(self, "Import", f"Import cancelled. Imported {imported} photo(s) before cancellation.")
        else:
            QMessageBox.information(self, "Import", f"Imported {imported} photo(s).")

        if imported:
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_collection_filter_options()  # Refresh collection dropdown
            self._populate_photo_list()
            if self.photos:
                self.index = 0
                self.load_photo()

    def import_from_sd_card(self):
        """Detect and import from SD cards/removable drives."""
        import platform

        # Find mounted volumes (macOS/Linux: /Volumes, Windows: drive letters)
        removable_drives = []

        if platform.system() == "Darwin":  # macOS
            volumes_path = Path("/Volumes")
            if volumes_path.exists():
                for vol in volumes_path.iterdir():
                    # Skip the main system drive
                    if vol.name == "Macintosh HD" or vol.is_symlink():
                        continue
                    # Check if it looks like a camera card (has DCIM folder)
                    dcim_path = vol / "DCIM"
                    if dcim_path.exists():
                        # Count photos
                        photo_count = 0
                        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
                            photo_count += len(list(dcim_path.rglob(ext)))
                        removable_drives.append((vol.name, str(vol), photo_count, str(dcim_path)))
                    else:
                        # Still show the drive even without DCIM
                        removable_drives.append((vol.name, str(vol), 0, None))
        elif platform.system() == "Windows":
            import string
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if Path(drive).exists() and letter not in ["C"]:
                    dcim_path = Path(drive) / "DCIM"
                    if dcim_path.exists():
                        photo_count = 0
                        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
                            photo_count += len(list(dcim_path.rglob(ext)))
                        removable_drives.append((f"Drive {letter}:", drive, photo_count, str(dcim_path)))
                    else:
                        removable_drives.append((f"Drive {letter}:", drive, 0, None))

        if not removable_drives:
            QMessageBox.information(
                self,
                "No SD Cards Found",
                "No SD cards or removable drives were detected.\n\n"
                "Please insert an SD card and try again, or use 'Import Folder...' to browse manually."
            )
            return

        # Show dialog to select drive
        dialog = QDialog(self)
        dialog.setWindowTitle("Import from SD Card")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Select an SD card or removable drive to import from:"))

        # Create list of drives
        drive_list = QListWidget()
        drive_list.setMinimumHeight(150)
        for name, path, count, dcim in removable_drives:
            if dcim and count > 0:
                item_text = f"{name}  —  {count} photos found"
            elif dcim:
                item_text = f"{name}  —  DCIM folder found (scanning...)"
            else:
                item_text = f"{name}  —  No DCIM folder"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, (path, dcim))
            drive_list.addItem(item)

        # Select first item with photos
        for i in range(drive_list.count()):
            item = drive_list.item(i)
            _, dcim = item.data(Qt.ItemDataRole.UserRole)
            if dcim:
                drive_list.setCurrentRow(i)
                break

        layout.addWidget(drive_list)

        # Collection/Farm dropdown
        collection_layout = QHBoxLayout()
        collection_layout.addWidget(QLabel("Collection/Farm:"))
        collection_combo = QComboBox()
        collection_combo.setEditable(True)
        collection_combo.setPlaceholderText("Select or type a collection name")
        # Load existing collections from database
        existing_collections = self.db.get_distinct_collections()
        collection_combo.addItems(existing_collections)
        collection_combo.setCurrentIndex(-1)  # Start with no selection
        collection_layout.addWidget(collection_combo)
        layout.addLayout(collection_layout)

        # Skip duplicates checkbox
        skip_dup_checkbox = QCheckBox("Skip duplicates (hash check)")
        layout.addWidget(skip_dup_checkbox)

        # Buttons
        btn_layout = QHBoxLayout()
        import_btn = QPushButton("Import")
        import_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(import_btn)
        layout.addLayout(btn_layout)

        cancel_btn.clicked.connect(dialog.reject)
        import_btn.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Get selected drive
        selected = drive_list.currentItem()
        if not selected:
            return

        drive_path, dcim_path = selected.data(Qt.ItemDataRole.UserRole)

        # Use DCIM path if available, otherwise use drive root
        import_path = Path(dcim_path) if dcim_path else Path(drive_path)

        # Find all JPG files
        jpg_files = self._find_image_files(import_path)
        if not jpg_files:
            QMessageBox.information(self, "No Images", f"No JPG/JPEG files found in {import_path}")
            return

        # Get selected collection
        selected_collection = collection_combo.currentText().strip()

        reply = QMessageBox.question(
            self,
            "Import Photos",
            f"Found {len(jpg_files)} photo(s) on '{selected.text().split('  —')[0]}'.\n\nImport them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Create progress dialog
        progress = QProgressDialog("Importing photos...", "Cancel", 0, len(jpg_files), self)
        progress.setWindowTitle("Import Progress")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._import_cancelled = False

        def update_progress(current, total, filename):
            if progress.wasCanceled():
                self._import_cancelled = True
                return
            progress.setValue(current)
            progress.setLabelText(f"Importing {current + 1} of {total}:\n{filename}")
            QApplication.processEvents()

        imported = self._import_files(jpg_files, skip_hash=skip_dup_checkbox.isChecked(), collection=selected_collection, progress_callback=update_progress)
        progress.setValue(len(jpg_files))
        progress.close()

        if self._import_cancelled:
            QMessageBox.information(self, "Import", f"Import cancelled. Imported {imported} photo(s) before cancellation.")
        else:
            QMessageBox.information(self, "Import Complete", f"Imported {imported} photo(s).")

        if imported:
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_collection_filter_options()  # Refresh collection dropdown
            self._populate_photo_list()
            if self.photos:
                self.index = 0
                self.load_photo()

    def setup_cuddelink_credentials(self):
        """Show dialog to set up CuddeLink credentials."""
        dlg = CuddeLinkCredentialsDialog(self)
        dlg.exec()

    def check_cuddelink_status(self):
        """Check and display CuddeLink server status."""
        # Show a waiting cursor
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = check_server_status()
        finally:
            QApplication.restoreOverrideCursor()

        # Build detailed message
        status = result["status"]
        message = result["message"]
        details = result.get("details", [])

        detail_lines = []
        for d in details:
            endpoint = d["endpoint"]
            ep_status = d["status"]
            code = d.get("code", "N/A")
            time_val = d.get("time")
            time_str = f"{time_val:.2f}s" if time_val else "N/A"

            if ep_status == "ok":
                detail_lines.append(f"  {endpoint}: OK ({time_str})")
            elif ep_status == "slow":
                detail_lines.append(f"  {endpoint}: Slow ({time_str})")
            elif ep_status == "down":
                detail_lines.append(f"  {endpoint}: Down (Error {code})")
            elif ep_status == "timeout":
                detail_lines.append(f"  {endpoint}: Timeout")
            elif ep_status == "unreachable":
                detail_lines.append(f"  {endpoint}: Unreachable")
            else:
                detail_lines.append(f"  {endpoint}: {ep_status} ({code})")

        full_message = f"{message}\n\nEndpoint Status:\n" + "\n".join(detail_lines)

        if status == "ok":
            QMessageBox.information(self, "CuddeLink Status", full_message)
        elif status == "slow":
            QMessageBox.warning(self, "CuddeLink Status", full_message)
        else:
            QMessageBox.critical(self, "CuddeLink Status", full_message)

    def download_cuddelink(self):
        """Download photos from CuddeLink."""
        settings = QSettings("TrailCam", "Trainer")
        email = settings.value("cuddelink_email", "")
        password = settings.value("cuddelink_password", "")

        # If no credentials saved, prompt to set them up
        if not email or not password:
            reply = QMessageBox.question(
                self, "CuddeLink",
                "No CuddeLink credentials found.\n\nWould you like to set them up now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.setup_cuddelink_credentials()
                # Re-check credentials after setup
                email = settings.value("cuddelink_email", "")
                password = settings.value("cuddelink_password", "")
                if not email or not password:
                    return
            else:
                return

        # Show date range picker dialog
        date_dialog = QDialog(self)
        date_dialog.setWindowTitle("CuddeLink Date Range")
        date_dialog.setMinimumWidth(300)
        date_layout = QVBoxLayout(date_dialog)

        date_layout.addWidget(QLabel("Select date range to download:"))

        # Start date - default to 1 day before newest photo (buffer for upload delay) or 30 days ago
        # First try saved last download date, then check newest photo in database
        last_download = settings.value("cuddelink_last_download", "")
        if not last_download:
            # Query database for most recent photo date
            try:
                cursor = self.db.conn.cursor()
                cursor.execute("SELECT MAX(date_taken) FROM photos WHERE date_taken IS NOT NULL")
                row = cursor.fetchone()
                if row and row[0]:
                    # date_taken format is "YYYY-MM-DD HH:MM:SS" or similar
                    dt = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
                    last_download = dt.date().isoformat()
            except Exception:
                pass

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("From:"))
        start_edit = QDateEdit()
        start_edit.setCalendarPopup(True)
        start_edit.setDisplayFormat("yyyy-MM-dd")
        if last_download:
            last_date = QDate.fromString(last_download, "yyyy-MM-dd")
            start_edit.setDate(last_date.addDays(-1))  # Go back 1 day for upload delay buffer
        else:
            start_edit.setDate(QDate.currentDate().addDays(-30))
        start_row.addWidget(start_edit)
        date_layout.addLayout(start_row)

        # End date
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("To:"))
        end_edit = QDateEdit()
        end_edit.setCalendarPopup(True)
        end_edit.setDisplayFormat("yyyy-MM-dd")
        end_edit.setDate(QDate.currentDate())
        end_row.addWidget(end_edit)
        date_layout.addLayout(end_row)

        # Quick select buttons
        quick_row = QHBoxLayout()
        quick_row.addWidget(QLabel("Quick:"))
        btn_7d = QPushButton("7 days")
        btn_30d = QPushButton("30 days")
        btn_90d = QPushButton("90 days")
        btn_all = QPushButton("All 2024+")

        def set_days(days):
            start_edit.setDate(QDate.currentDate().addDays(-days))
            end_edit.setDate(QDate.currentDate())

        btn_7d.clicked.connect(lambda: set_days(7))
        btn_30d.clicked.connect(lambda: set_days(30))
        btn_90d.clicked.connect(lambda: set_days(90))
        btn_all.clicked.connect(lambda: (start_edit.setDate(QDate(2024, 1, 1)), end_edit.setDate(QDate.currentDate())))

        quick_row.addWidget(btn_7d)
        quick_row.addWidget(btn_30d)
        quick_row.addWidget(btn_90d)
        quick_row.addWidget(btn_all)
        date_layout.addLayout(quick_row)

        # Collection/Farm dropdown
        collection_layout = QHBoxLayout()
        collection_layout.addWidget(QLabel("Collection/Farm:"))
        collection_combo = QComboBox()
        collection_combo.setEditable(True)
        collection_combo.setPlaceholderText("Select or type a collection name")
        # Load existing collections from database
        existing_collections = self.db.get_distinct_collections()
        collection_combo.addItems(existing_collections)
        # Default to "Brooke Farm" if it exists
        brooke_idx = collection_combo.findText("Brooke Farm")
        if brooke_idx >= 0:
            collection_combo.setCurrentIndex(brooke_idx)
        else:
            collection_combo.setCurrentIndex(-1)
        collection_layout.addWidget(collection_combo)
        date_layout.addLayout(collection_layout)

        # OK/Cancel buttons
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Download")
        ok_btn.setDefault(True)
        cancel_btn_date = QPushButton("Cancel")
        ok_btn.clicked.connect(date_dialog.accept)
        cancel_btn_date.clicked.connect(date_dialog.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn_date)
        date_layout.addLayout(btn_row)

        if date_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        start_date = start_edit.date().toString("yyyy-MM-dd")
        end_date = end_edit.date().toString("yyyy-MM-dd")
        selected_collection = collection_combo.currentText().strip()

        # Use a simple dialog instead of QProgressDialog to avoid auto-cancel issues
        self._cudde_dialog = QDialog(self)
        self._cudde_dialog.setWindowTitle("CuddeLink Download")
        self._cudde_dialog.setModal(True)
        self._cudde_dialog.setMinimumWidth(350)
        layout = QVBoxLayout(self._cudde_dialog)
        self._cudde_label = QLabel("Connecting to CuddeLink...")
        layout.addWidget(self._cudde_label)

        # Day progress bar (e.g., Day 3/7)
        self._cudde_day_progress = QProgressBar()
        self._cudde_day_progress.setFormat("Day %v of %m")
        self._cudde_day_progress.setValue(0)
        layout.addWidget(self._cudde_day_progress)

        # Download progress bar (bytes downloaded)
        self._cudde_download_label = QLabel("")
        layout.addWidget(self._cudde_download_label)
        self._cudde_download_progress = QProgressBar()
        self._cudde_download_progress.setFormat("%p%")
        self._cudde_download_progress.setValue(0)
        layout.addWidget(self._cudde_download_progress)

        cancel_btn = QPushButton("Cancel")
        layout.addWidget(cancel_btn)

        self._cudde_cancelled = False
        def on_cancel():
            self._cudde_cancelled = True
            self._cudde_label.setText("Cancelling...")
            cancel_btn.setEnabled(False)
        cancel_btn.clicked.connect(on_cancel)

        # Run download in background thread
        dest = Path(tempfile.gettempdir())

        class DownloadWorker(QThread):
            finished = pyqtSignal(list, str)  # files, error
            status = pyqtSignal(str)  # status update
            progress = pyqtSignal(int, int, str, str)  # current, total, stage, message

            def __init__(self, dest, email, password, start_date, end_date):
                super().__init__()
                self.dest = dest
                self.email = email
                self.password = password
                self.start_date = start_date
                self.end_date = end_date

            def run(self):
                try:
                    self.status.emit("Logging in...")

                    def progress_callback(current, total, stage, message):
                        self.progress.emit(current, total, stage, message)

                    files = download_new_photos(self.dest, user=self.email, password=self.password,
                                                start_date=self.start_date, end_date=self.end_date,
                                                progress_callback=progress_callback)
                    self.finished.emit(files, "")
                except Exception as e:
                    self.finished.emit([], str(e))

        def on_status(msg):
            self._cudde_label.setText(msg)

        def on_progress(current, total, stage, message):
            if stage == "login":
                self._cudde_label.setText(message)
                self._cudde_day_progress.setValue(0)
                self._cudde_download_progress.setValue(0)
            elif stage == "day":
                self._cudde_label.setText(message)
                self._cudde_day_progress.setMaximum(total)
                self._cudde_day_progress.setValue(current)
                # Reset download progress for new day
                self._cudde_download_progress.setValue(0)
                self._cudde_download_label.setText("")
            elif stage == "download":
                # Show download progress in MB
                mb_current = current / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                self._cudde_download_label.setText(f"Downloading: {mb_current:.1f} / {mb_total:.1f} MB")
                self._cudde_download_progress.setMaximum(total)
                self._cudde_download_progress.setValue(current)
            elif stage == "done":
                self._cudde_label.setText(message)
                self._cudde_day_progress.setValue(self._cudde_day_progress.maximum())
                self._cudde_download_progress.setValue(self._cudde_download_progress.maximum())

        def on_download_complete(files, error):
            self._cudde_dialog.close()
            temp_dir = dest / ".cuddelink_tmp"

            try:
                if self._cudde_cancelled:
                    return

                if error:
                    if "credentials" in error.lower() or "login" in error.lower() or "invalid" in error.lower() or "password" in error.lower():
                        reply = QMessageBox.question(
                            self, "CuddeLink Login Failed",
                            f"Login failed: {error}\n\nWould you like to update your credentials?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            self.setup_cuddelink_credentials()
                    else:
                        QMessageBox.warning(self, "CuddeLink", f"Download failed: {error}")
                    return

                if not files:
                    QMessageBox.information(self, "CuddeLink", "No new photos found to download.")
                    return

                # Filter out already-imported photos by original filename
                new_files = []
                skipped = 0
                for f in files:
                    if self.db.photo_exists_by_original_name(f.name):
                        skipped += 1
                    else:
                        new_files.append(f)

                if not new_files:
                    QMessageBox.information(self, "CuddeLink", f"All {skipped} photo(s) already imported.")
                    return

                # Scan for unique camera IDs and show mapping dialog
                camera_id_counts = {}
                for f in new_files:
                    cam_id = extract_cuddelink_camera_id(str(f))
                    if cam_id:
                        camera_id_counts[cam_id] = camera_id_counts.get(cam_id, 0) + 1

                # Show site mapping dialog if we found camera IDs
                site_mappings = {}  # camera_id -> site_name
                if camera_id_counts:
                    import json
                    site_mapping_path = Path.home() / ".trailcam" / "site_mappings.json"

                    def load_site_mappings():
                        saved = {}
                        if site_mapping_path.exists():
                            try:
                                with open(site_mapping_path, "r") as f:
                                    saved = json.load(f) or {}
                            except Exception:
                                saved = {}
                        else:
                            saved_str = settings.value("cuddelink_site_mappings", "")
                            if saved_str:
                                try:
                                    saved = json.loads(saved_str) or {}
                                except Exception:
                                    saved = {}
                                if saved:
                                    site_mapping_path.parent.mkdir(parents=True, exist_ok=True)
                                    tmp_path = site_mapping_path.with_suffix(".tmp")
                                    with open(tmp_path, "w") as f:
                                        json.dump(saved, f)
                                    os.replace(tmp_path, site_mapping_path)
                                    settings.remove("cuddelink_site_mappings")
                        return saved

                    def save_site_mappings(mappings):
                        site_mapping_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path = site_mapping_path.with_suffix(".tmp")
                        with open(tmp_path, "w") as f:
                            json.dump(mappings, f)
                        os.replace(tmp_path, site_mapping_path)

                    saved_mappings = load_site_mappings()

                    # Check if all cameras have saved mappings
                    all_mapped = all(cam_id in saved_mappings for cam_id in camera_id_counts)

                    if all_mapped:
                        # Show quick confirmation with option to change
                        summary = "Photos will be assigned to these sites:\n\n"
                        for cam_id, count in sorted(camera_id_counts.items()):
                            site = saved_mappings.get(cam_id, cam_id)
                            summary += f"  • {cam_id} → {site} ({count} photos)\n"
                        summary += "\nClick 'Import' to proceed or 'Change...' to modify."

                        confirm_dialog = QMessageBox(self)
                        confirm_dialog.setWindowTitle("Confirm Site Assignments")
                        confirm_dialog.setText(summary)
                        confirm_dialog.setIcon(QMessageBox.Icon.Question)
                        import_btn = confirm_dialog.addButton("Import", QMessageBox.ButtonRole.AcceptRole)
                        change_btn = confirm_dialog.addButton("Change...", QMessageBox.ButtonRole.ActionRole)
                        cancel_btn_confirm = confirm_dialog.addButton(QMessageBox.StandardButton.Cancel)

                        confirm_dialog.exec()
                        clicked = confirm_dialog.clickedButton()

                        if clicked == cancel_btn_confirm:
                            return
                        elif clicked == import_btn:
                            # Use saved mappings directly
                            for cam_id in camera_id_counts:
                                site_mappings[cam_id] = saved_mappings.get(cam_id, cam_id)
                            # Skip to import
                            imported = self._import_files(new_files, skip_hash=True, collection=selected_collection,
                                                         site_mappings=site_mappings)
                            msg = f"Imported {imported} new photo(s)."
                            if skipped > 0:
                                msg += f"\nSkipped {skipped} duplicate(s)."
                            QMessageBox.information(self, "CuddeLink", msg)
                            if imported:
                                settings.setValue("cuddelink_last_download", end_date)
                                self.photos = self._sorted_photos(self.db.get_all_photos())
                                self._populate_collection_filter_options()
                                self._populate_photo_list()
                            return
                        # Otherwise fall through to show edit dialog

                    map_dialog = QDialog(self)
                    map_dialog.setWindowTitle("Camera Site Mapping")
                    map_dialog.setMinimumWidth(450)
                    map_layout = QVBoxLayout(map_dialog)

                    map_layout.addWidget(QLabel(
                        f"Found {len(camera_id_counts)} camera(s) in {len(new_files)} photos.\n"
                        "Site names are remembered from previous downloads.\n"
                        "Change if camera was moved to a new location:"))

                    # Get existing sites for suggestions
                    existing_sites = sorted(set(
                        p.get('camera_location', '').strip()
                        for p in self.db.get_all_photos()
                        if p.get('camera_location', '').strip()
                    ))

                    # Create editable rows for each camera ID
                    site_edits = {}
                    scroll = QScrollArea()
                    scroll.setWidgetResizable(True)
                    scroll_widget = QWidget()
                    scroll_layout = QFormLayout(scroll_widget)

                    for cam_id, count in sorted(camera_id_counts.items()):
                        combo = QComboBox()
                        combo.setEditable(True)
                        combo.addItems(existing_sites)
                        # Use saved mapping if available, otherwise use camera ID
                        default_site = saved_mappings.get(cam_id, cam_id)
                        combo.setCurrentText(default_site)
                        combo.setMinimumWidth(200)
                        site_edits[cam_id] = combo
                        scroll_layout.addRow(f"{cam_id} ({count} photos):", combo)

                    scroll.setWidget(scroll_widget)
                    scroll.setMaximumHeight(300)
                    map_layout.addWidget(scroll)

                    # Buttons
                    btn_layout = QHBoxLayout()
                    ok_btn = QPushButton("Import")
                    ok_btn.setDefault(True)
                    cancel_btn = QPushButton("Cancel")
                    ok_btn.clicked.connect(map_dialog.accept)
                    cancel_btn.clicked.connect(map_dialog.reject)
                    btn_layout.addWidget(ok_btn)
                    btn_layout.addWidget(cancel_btn)
                    map_layout.addLayout(btn_layout)

                    if map_dialog.exec() != QDialog.DialogCode.Accepted:
                        # Clean up temp files
                        return

                    # Collect mappings
                    for cam_id, combo in site_edits.items():
                        site_name = combo.currentText().strip()
                        if site_name:
                            site_mappings[cam_id] = site_name

                    # Save mappings for future downloads (merge with existing)
                    all_mappings = saved_mappings.copy()
                    all_mappings.update(site_mappings)
                    save_site_mappings(all_mappings)

                imported = self._import_files(new_files, skip_hash=True, collection=selected_collection,
                                             site_mappings=site_mappings)
                msg = f"Imported {imported} new photo(s)."
                if skipped > 0:
                    msg += f"\nSkipped {skipped} duplicate(s)."
                QMessageBox.information(self, "CuddeLink", msg)
                if imported:
                    # Save the end date as last download date for next time
                    settings.setValue("cuddelink_last_download", end_date)
                    self.photos = self._sorted_photos(self.db.get_all_photos())
                    self._populate_collection_filter_options()  # Refresh collection dropdown
                    self._populate_photo_list()
                    if self.photos:
                        # Find newest photo by date_taken
                        newest_idx = 0
                        newest_date = ""
                        for i, p in enumerate(self.photos):
                            dt = p.get("date_taken") or ""
                            if dt > newest_date:
                                newest_date = dt
                                newest_idx = i
                        self.index = newest_idx
                        self.load_photo()
            finally:
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass

            # Check for missing verification photos
            self._check_missing_verifications(new_files, start_date, end_date, settings)

        self._cudde_worker = DownloadWorker(dest, email, password, start_date, end_date)
        self._cudde_worker.status.connect(on_status)
        self._cudde_worker.progress.connect(on_progress)
        self._cudde_worker.finished.connect(on_download_complete)
        self._cudde_worker.start()
        self._cudde_dialog.show()

    def _check_missing_verifications(self, downloaded_files, start_date: str, end_date: str, settings):
        """Check for cameras missing verification photos and alert user.

        Only alerts for cameras that:
        1. Had a verification photo since the previous download (minus 1 day buffer)
        2. Did NOT have a verification in the current download

        This ensures we alert at least once when a camera goes down, but
        avoids alerting for cameras that were already offline before the last download.

        Includes time buffer: if downloading today's photos before noon, don't alert
        about missing verifications (gives time for 9 AM photos to upload).
        """
        import json
        import os
        from datetime import datetime, timedelta

        MAX_SIZE_KB = 15  # Verification photos are typically 6-7 KB
        VERIFICATION_ALERT_HOUR = 12  # Don't alert until after noon

        # Check if we should skip alerting due to time buffer
        # If download includes today and it's before noon, skip the alert
        # (verification photos are taken at 9 AM, give time to upload)
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        skip_alert_for_time = (end_date == today_str and now.hour < VERIFICATION_ALERT_HOUR)

        # Load camera tracking data: {mac_address: {last_verification: date, site_name: str, snoozed: bool}}
        camera_data_str = settings.value("cuddelink_camera_tracking", "{}")
        try:
            camera_data = json.loads(camera_data_str)
        except:
            camera_data = {}

        # Get the previous download date (before this one was saved)
        # We use the start_date of current download as reference since last_download hasn't been updated yet
        prev_download_str = settings.value("cuddelink_last_download", "")

        # Scan downloaded files for verification photos and cameras
        cameras_in_download = {}  # mac -> {has_verification: bool, site_name: str}
        for f in downloaded_files:
            mac = extract_cuddelink_mac_address(str(f))
            cam_id = extract_cuddelink_camera_id(str(f))
            if not mac:
                continue

            if mac not in cameras_in_download:
                cameras_in_download[mac] = {"has_verification": False, "site_name": cam_id or mac}

            # Check if this is a verification photo (small file)
            try:
                size_kb = os.path.getsize(f) / 1024
                if size_kb < MAX_SIZE_KB:
                    cameras_in_download[mac]["has_verification"] = True
            except:
                pass

        # Update camera tracking with new verifications
        today = datetime.now().strftime("%Y-%m-%d")
        for mac, info in cameras_in_download.items():
            if mac not in camera_data:
                camera_data[mac] = {"last_verification": None, "site_name": info["site_name"], "snoozed": False}

            camera_data[mac]["site_name"] = info["site_name"]  # Update site name

            if info["has_verification"]:
                camera_data[mac]["last_verification"] = today
                camera_data[mac]["snoozed"] = False  # Camera is back, reset snooze

        # Determine cutoff date - one day before previous download
        # This ensures we catch cameras that went down after the last download
        if prev_download_str:
            try:
                prev_download = datetime.strptime(prev_download_str, "%Y-%m-%d")
                cutoff_date = (prev_download - timedelta(days=1)).strftime("%Y-%m-%d")
            except:
                cutoff_date = start_date  # Fallback to start of current download
        else:
            # First download ever - no previous reference, use start date
            cutoff_date = start_date

        # Check for missing verifications from recently-active cameras
        missing_cameras = []

        for mac, data in camera_data.items():
            # Skip if snoozed (user dismissed alert)
            if data.get("snoozed"):
                continue

            # Skip cameras that never had a verification
            last_verif = data.get("last_verification")
            if not last_verif:
                continue

            # Was this camera active since the last download?
            # (had verification on or after cutoff date)
            if last_verif < cutoff_date:
                continue  # Camera was already offline before last download

            # Did this camera have a verification in this download?
            if mac in cameras_in_download and cameras_in_download[mac]["has_verification"]:
                continue  # All good

            # Camera was active since last download but missing verification now
            missing_cameras.append({
                "mac": mac,
                "site_name": data.get("site_name", mac),
                "last_verification": last_verif
            })

        # Save updated tracking data
        settings.setValue("cuddelink_camera_tracking", json.dumps(camera_data))

        # Show alert if any cameras are missing verifications
        # But skip if it's before noon and download includes today (give time for 9 AM verification to upload)
        if missing_cameras and skip_alert_for_time:
            # Still before noon - don't alert yet, verifications may still be uploading
            return

        if missing_cameras:
            msg = "The following camera(s) may be down or disconnected:\n\n"
            for cam in missing_cameras:
                msg += f"  • {cam['site_name']} (last verification: {cam['last_verification']})\n"
            msg += "\nThese cameras had recent verifications but didn't send one today."

            alert = QMessageBox(self)
            alert.setWindowTitle("Camera Status Alert")
            alert.setText(msg)
            alert.setIcon(QMessageBox.Icon.Warning)
            alert.addButton(QMessageBox.StandardButton.Ok)
            snooze_btn = alert.addButton("Don't remind until back online", QMessageBox.ButtonRole.ActionRole)

            alert.exec()

            if alert.clickedButton() == snooze_btn:
                # Snooze alerts for these cameras
                for cam in missing_cameras:
                    if cam["mac"] in camera_data:
                        camera_data[cam["mac"]]["snoozed"] = True
                settings.setValue("cuddelink_camera_tracking", json.dumps(camera_data))

    # ─────────────────────────────────────────────────────────────────────
    # User Setup (Username & Hunting Club)
    # ─────────────────────────────────────────────────────────────────────

    def _show_user_setup_dialog(self):
        """Show dialog for username and hunting club selection."""
        dlg = UserSetupDialog(self)
        dlg.exec()

    def show_user_settings(self):
        """Show user settings dialog (can be called from menu)."""
        self._show_user_setup_dialog()

    # ─────────────────────────────────────────────────────────────────────
    # Supabase Cloud Sync
    # ─────────────────────────────────────────────────────────────────────

    def _maybe_show_login_dialog(self):
        """Show login dialog on startup if not authenticated."""
        client = self._get_supabase_client_silent()
        if not client or not client.is_configured():
            self._update_auth_status()
            return
        if client.is_authenticated:
            self._update_auth_status()
            return
        dlg = SupabaseAuthDialog(client, self)
        dlg.exec()
        self._update_auth_status()
        self._clear_cached_username()
        if dlg.result_mode == "continue":
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage(
                    "Continuing without login (anon key in use)", 8000
                )

    def _update_auth_status(self):
        """Update the status bar user indicator."""
        if not hasattr(self, 'user_status_label'):
            return
        client = self._get_supabase_client_silent()
        if client and client.is_authenticated:
            session = client.get_session_info()
            email = session.get("user_email")
            display = session.get("display_name")
            label = email or display or "Logged in"
            self.user_status_label.setText(f"User: {label}")
            self.user_status_label.setStyleSheet("color: green; padding: 0 10px;")
        else:
            self.user_status_label.setText("User: Not logged in")
            self.user_status_label.setStyleSheet("color: gray; padding: 0 10px;")

    def _sign_out(self):
        """Sign out of Supabase."""
        client = self._get_supabase_client_silent()
        if client:
            client.sign_out()
        self._update_auth_status()
        self._clear_cached_username()

    def setup_supabase_credentials(self):
        """Show dialog to set up Supabase credentials."""
        dlg = SupabaseCredentialsDialog(self)
        dlg.exec()

    def _get_supabase_client_silent(self):
        """Get Supabase client without showing dialogs (for background sync)."""
        url = None
        key = None

        # Try bundled cloud_config.json
        try:
            import json
            config_paths = [
                Path(__file__).parent.parent / "cloud_config.json",
                Path(__file__).parent / "cloud_config.json",
                Path.cwd() / "cloud_config.json",
            ]
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    if "supabase" in config:
                        url = config["supabase"].get("url", "")
                        key = config["supabase"].get("anon_key", "")
                        if url and key:
                            break
        except Exception:
            pass

        # Fall back to user settings
        if not url or not key:
            settings = QSettings("TrailCam", "Trainer")
            url = settings.value("supabase_url", "")
            key = settings.value("supabase_key", "")

        if not url or not key:
            return None

        try:
            from supabase_rest import create_client
            client = create_client(url, key)
            try:
                client.refresh_if_needed()
            except Exception:
                pass
            # Surface auth refresh failures so the user knows to re-login
            if getattr(client, '_auth_error', None):
                logger.warning(f"Auth refresh issue: {client._auth_error}")
                if hasattr(self, 'sync_status_label'):
                    self.sync_status_label.setText("Cloud: Auth expired")
                    self.sync_status_label.setStyleSheet("color: red;")
                client._auth_error = None  # Clear after surfacing
            return client
        except Exception:
            return None

    def _on_sync_status_changed(self, status: str):
        """Handle sync status changes for UI update."""
        if hasattr(self, 'sync_status_label'):
            status_text = {
                'idle': 'Synced',
                'pending': 'Pending...',
                'syncing': 'Syncing...',
                'offline': 'Offline'
            }.get(status, status)
            self.sync_status_label.setText(f"Cloud: {status_text}")
            # Update color based on status
            colors = {
                'idle': 'green',
                'pending': 'orange',
                'syncing': 'blue',
                'offline': 'gray'
            }
            color = colors.get(status, 'gray')
            self.sync_status_label.setStyleSheet(f"color: {color};")

    def _on_sync_completed(self, count: int):
        """Handle successful sync."""
        if count > 0:
            logger.info(f"Auto-sync completed: {count} items synced")

        # Run auto-archive after sync if enabled
        archived_count = self.run_auto_archive()
        if archived_count > 0:
            # Refresh the photo list to reflect archived photos
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()
            if self.index >= len(self.photos):
                self.index = max(0, len(self.photos) - 1)
            if self.photos:
                self.load_photo()

    def _on_sync_failed(self, error: str):
        """Handle sync failure — show visible error to user."""
        logger.warning(f"Auto-sync failed: {error}")
        # Update status bar to show error in red
        if hasattr(self, 'sync_status_label'):
            self.sync_status_label.setText("Cloud: Sync Failed")
            self.sync_status_label.setStyleSheet("color: red;")
        # Show a non-blocking status bar message
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(f"Cloud sync failed: {error}", 10000)

    def _get_r2_storage(self):
        """Get R2 storage instance."""
        try:
            from r2_storage import R2Storage
            return R2Storage()
        except Exception:
            return None

    def _on_r2_status_changed(self, status: str):
        """Handle R2 upload status changes."""
        if hasattr(self, 'r2_status_label'):
            self.r2_status_label.setText(status)
            # Update color based on status
            if 'Synced' in status:
                self.r2_status_label.setStyleSheet("color: green; padding: 0 10px;")
            elif 'pending' in status or 'Uploading' in status:
                self.r2_status_label.setStyleSheet("color: orange; padding: 0 10px;")
            elif 'failed' in status:
                self.r2_status_label.setStyleSheet("color: red; padding: 0 10px;")
            else:
                self.r2_status_label.setStyleSheet("color: gray; padding: 0 10px;")

    def _on_r2_upload_completed(self, uploaded: int, failed: int):
        """Handle R2 upload batch completion."""
        if uploaded > 0:
            logger.info(f"R2 upload completed: {uploaded} photos")

    def _get_supabase_client(self):
        """Get Supabase client, loading from bundled config or user settings."""
        url = None
        key = None

        # First try bundled cloud_config.json
        try:
            import json
            from pathlib import Path
            # Try multiple locations for bundled config
            config_paths = [
                Path(__file__).parent.parent / "cloud_config.json",  # Dev
                Path(__file__).parent / "cloud_config.json",  # PyInstaller
                Path.cwd() / "cloud_config.json",  # Current dir
            ]
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    if "supabase" in config:
                        url = config["supabase"].get("url", "")
                        key = config["supabase"].get("anon_key", "")
                        if url and key:
                            break
        except Exception as e:
            logger.warning(f"Failed to load bundled Supabase config: {e}")

        # Fall back to user settings if bundled config not found
        if not url or not key:
            settings = QSettings("TrailCam", "Trainer")
            url = settings.value("supabase_url", "")
            key = settings.value("supabase_key", "")

        if not url or not key:
            # Open the credentials setup dialog directly
            self.setup_supabase_credentials()
            # Check again after dialog closes
            settings = QSettings("TrailCam", "Trainer")
            url = settings.value("supabase_url", "")
            key = settings.value("supabase_key", "")
            if not url or not key:
                return None

        try:
            from supabase_rest import create_client
            client = create_client(url, key)
            try:
                client.refresh_if_needed()
            except Exception:
                pass
            if getattr(client, '_auth_error', None) or (client.get_session_info() and not client.is_authenticated):
                client._auth_error = None  # Clear after detecting
                self._maybe_show_login_dialog()
            return client
        except Exception as e:
            QMessageBox.warning(self, "Supabase", f"Failed to connect to Supabase:\n{str(e)}")
            return None

    def push_to_cloud(self):
        """Push local labels to Supabase cloud."""
        client = self._get_supabase_client()
        if not client:
            return

        reply = QMessageBox.question(
            self, "Push to Cloud",
            "This will upload all your labels to the cloud.\n\n"
            "Existing cloud data will be updated with your local changes.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Create progress dialog
        progress = QProgressDialog("Preparing to push...", None, 0, 7, self)
        progress.setWindowTitle("Push to Cloud")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        progress.raise_()
        QApplication.processEvents()

        def update_progress(step, total, message):
            progress.setValue(step)
            progress.setLabelText(message)
            QApplication.processEvents()

        try:
            counts = self.db.push_to_supabase(client, progress_callback=update_progress)
            progress.close()
            summary = (
                f"Pushed to cloud:\n\n"
                f"• Photos: {counts['photos']}\n"
                f"• Tags: {counts['tags']}\n"
                f"• Deer metadata: {counts['deer_metadata']}\n"
                f"• Additional deer: {counts['deer_additional']}\n"
                f"• Buck profiles: {counts['buck_profiles']}\n"
                f"• Season profiles: {counts['buck_profile_seasons']}\n"
                f"• Annotation boxes: {counts['annotation_boxes']}"
            )
            QMessageBox.information(self, "Push Complete", summary)
        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Push Failed", f"Error pushing to cloud:\n{str(e)}")

    def pull_from_cloud(self):
        """Pull labels from Supabase cloud to local database."""
        client = self._get_supabase_client()
        if not client:
            return

        reply = QMessageBox.question(
            self, "Pull from Cloud",
            "This will download labels from the cloud.\n\n"
            "Cloud data will be merged with your local labels.\n"
            "Only photos that exist locally will be updated.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Create progress dialog
        progress = QProgressDialog("Preparing to pull...", None, 0, 7, self)
        progress.setWindowTitle("Pull from Cloud")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        progress.raise_()
        QApplication.processEvents()

        def update_progress(step, total, message):
            progress.setValue(step)
            progress.setLabelText(message)
            QApplication.processEvents()

        try:
            counts = self.db.pull_from_supabase(client, progress_callback=update_progress)
            progress.close()

            # Run auto-archive after pull if enabled
            archived_count = self.run_auto_archive()
            archive_msg = ""
            if archived_count > 0:
                archive_msg = f"\n\n• Auto-archived: {archived_count} photos"

            summary = (
                f"Pulled from cloud:\n\n"
                f"• Photos updated: {counts['photos']}\n"
                f"• Tags: {counts['tags']}\n"
                f"• Deer metadata: {counts['deer_metadata']}\n"
                f"• Additional deer: {counts['deer_additional']}\n"
                f"• Buck profiles: {counts['buck_profiles']}\n"
                f"• Season profiles: {counts['buck_profile_seasons']}"
                f"{archive_msg}"
            )
            QMessageBox.information(self, "Pull Complete", summary)
            # Reload current photo to show updated data
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()
            if self.photos:
                self.load_photo()
        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Pull Failed", f"Error pulling from cloud:\n{str(e)}")

    def remove_duplicates(self):
        """Open duplicate removal dialog."""
        photos = self.db.get_all_photos()
        hash_map = {}
        for p in photos:
            fp = p.get("file_path")
            if not fp or not os.path.exists(fp):
                continue
            h = self._hash_file(Path(fp))
            if not h:
                continue
            hash_map.setdefault(h, []).append(p)
        duplicates = {h: lst for h, lst in hash_map.items() if len(lst) > 1}
        if not duplicates:
            QMessageBox.information(self, "Duplicates", "No duplicate photos found.")
            return
        dlg = DuplicateDialog(duplicates, self.db, parent=self)
        dlg.exec()
        # refresh list after potential deletions
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_photo_list()
        if self.photos:
            self.index = min(self.index, len(self.photos) - 1)
            self.load_photo()

    def toggle_global_enhance(self, checked: bool):
        """Placeholder: global enhance flag."""
        self.auto_enhance_all = checked

    def run_ai_suggestions(self):
        """Run AI suggestions on current view with options dialog.

        Shows a dialog to let user choose:
        - Which photos to process (without suggestions vs all unlabeled)
        - Which AI steps to run (detect boxes, species ID, deer heads, buck/doe)

        Runs on main thread with progress dialog for reliable UI feedback.
        """
        # Show options dialog
        dialog = AIOptionsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        options = dialog.get_options()

        # Check if any steps are selected
        if not any([options["detect_boxes"], options["species_id"],
                    options["deer_head_boxes"], options["buck_doe"]]):
            QMessageBox.information(self, "AI Suggestions", "No AI steps selected.")
            return

        # Check model availability for selected steps
        if options["detect_boxes"] or options["species_id"]:
            # Ensure SpeciesNet is initialized
            if not self.speciesnet_wrapper.is_available:
                if not user_config.is_speciesnet_initialized():
                    dlg = SpeciesNetDownloadDialog(self.speciesnet_wrapper, parent=self)
                    if dlg.exec() != QDialog.DialogCode.Accepted:
                        return
                    user_config.set_config_value("speciesnet_initialized", True)
                else:
                    self.speciesnet_wrapper.initialize()
                if not self.speciesnet_wrapper.is_available:
                    QMessageBox.warning(self, "AI Model Not Available",
                        f"SpeciesNet could not be loaded:\n{self.speciesnet_wrapper.error_message}")
                    return

        if options["buck_doe"] and (not self.suggester or not self.suggester.buckdoe_ready):
            QMessageBox.information(self, "AI Model Not Available",
                "Buck/doe classifier not loaded. Uncheck 'Identify buck vs doe' or add models/buckdoe.onnx")
            return

        # Filter photos based on scope
        # Always skip photos suggested as "Verification" - these are test photos
        if options["scope"] == "no_suggestions":
            # Only photos without any AI suggestions
            target_photos = [p for p in self.photos
                           if not self._photo_has_human_species(p)
                           and not p.get("suggested_tag")]
        else:
            # All unlabeled photos (including those with existing suggestions)
            # But skip Verification photos
            target_photos = [p for p in self.photos
                           if not self._photo_has_human_species(p)
                           and p.get("suggested_tag") != "Verification"]
            # Clear existing suggestions for these photos before re-running
            for p in target_photos:
                pid = p.get("id")
                if pid:
                    if options["species_id"]:
                        self.db.set_suggested_tag(pid, None, None)
                    if options["buck_doe"]:
                        self.db.set_suggested_sex(pid, None, None)

        total = len(target_photos)

        if total == 0:
            QMessageBox.information(self, "AI Suggestions",
                "No photos to process based on selected scope.")
            return

        # Check if already processing
        if self.ai_processing:
            QMessageBox.information(self, "AI Processing", "AI processing is already running.")
            return

        # Create progress dialog
        progress = QProgressDialog("Loading AI model...", "Cancel", 0, total, self)
        progress.setWindowTitle("AI Processing")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        # SpeciesNet handles detection + classification together

        # Set up queue mode for review after processing
        self.queue_mode = True
        self.queue_type = "ai_processing"
        self.queue_photo_ids = []
        self.queue_data = {}
        self.queue_reviewed = set()
        if self.photos and self.index < len(self.photos):
            self.queue_pre_filter_index = self.index

        self.ai_processing = True
        counts = {"detect": 0, "species": 0, "heads": 0, "sex": 0}

        # Process each photo on main thread with progress updates
        cancelled = False
        for i, p in enumerate(target_photos):
            # Check cancel status but don't let stray events trigger it
            if progress.wasCanceled():
                cancelled = True
                break

            progress.setValue(i)
            progress.setLabelText(f"Processing photo {i + 1} of {total}...")
            # Process events to update UI, but limit to avoid picking up stray cancel clicks
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

            try:
                pid = p.get("id")
                crop = None

                # Step 0: Check if this is a verification photo (small file < 15 KB)
                # Verification photos are test shots that shouldn't go through species ID
                file_path = p.get("file_path")
                if file_path and os.path.exists(file_path):
                    try:
                        size_kb = os.path.getsize(file_path) / 1024
                        if size_kb < 15:  # Verification photos are typically 6-7 KB
                            self.db.set_suggested_tag(pid, "Verification", 0.95)
                            counts["species"] += 1
                            # Add to queue for review
                            if pid not in self.queue_photo_ids:
                                self.queue_photo_ids.append(pid)
                                self.queue_data[pid] = {
                                    "species": "Verification",
                                    "conf": 0.95,
                                    "sex": None,
                                    "sex_conf": 0
                                }
                            continue  # Skip remaining steps for verification photos
                    except Exception:
                        pass  # If we can't check size, continue with normal processing

                # Steps 1+2: SpeciesNet detection + classification
                if (options["detect_boxes"] or options["species_id"]) and self.speciesnet_wrapper and self.speciesnet_wrapper.is_available:
                    sn_result = self.speciesnet_wrapper.detect_and_classify(file_path)
                    sn_boxes = sn_result.get("detections", [])
                    app_species = sn_result.get("app_species")
                    score = sn_result.get("prediction_score", 0)

                    # Store boxes if photo doesn't have any yet
                    if options["detect_boxes"] and sn_boxes:
                        has_existing = False
                        try:
                            has_existing = self.db.has_detection_boxes(pid)
                        except Exception:
                            pass
                        if not has_existing:
                            self.db.set_boxes(pid, sn_boxes)
                            counts["detect"] += 1

                    if options["species_id"]:
                        check_boxes = self.db.get_boxes(pid) if pid else sn_boxes
                        has_person = any(b.get("label") == "ai_person" for b in check_boxes)
                        has_vehicle = any(b.get("label") == "ai_vehicle" for b in check_boxes)
                        animal_boxes = [b for b in check_boxes if b.get("label") in ("ai_animal", "ai_subject", "subject")]

                        label = None
                        conf = None

                        if has_person:
                            label, conf = "Person", 0.95
                        elif has_vehicle:
                            label, conf = "Vehicle", 0.95
                        elif animal_boxes and app_species and app_species not in ("Empty",):
                            label, conf = app_species, score
                            for box in animal_boxes:
                                box_id = box.get("id")
                                if box_id:
                                    self.db.set_box_ai_suggestion(box_id, app_species, score)
                        elif not check_boxes:
                            label, conf = "Empty", 0.95
                        else:
                            label, conf = app_species or "Empty", score or 0.95

                        if label and pid:
                            self.db.set_suggested_tag(pid, label, conf)
                            counts["species"] += 1

                            # Add to queue for review
                            if pid not in self.queue_photo_ids:
                                self.queue_photo_ids.append(pid)
                                self.queue_data[pid] = {
                                    "species": label,
                                    "conf": conf,
                                    "sex": None,
                                    "sex_conf": 0
                                }

                            # Buck/doe for deer
                            if label == "Deer":
                                if options["buck_doe"] and self.suggester and self.suggester.buckdoe_ready:
                                    head_crop = self._best_head_crop_for_photo(p)
                                    if head_crop:
                                        sex_res = self.suggester.predict_sex(str(head_crop))
                                        if sex_res:
                                            sex_label, sex_conf = sex_res
                                            self.db.set_suggested_sex(pid, sex_label, sex_conf)
                                            counts["sex"] += 1
                                            if pid in self.queue_data:
                                                self.queue_data[pid]["sex"] = sex_label
                                                self.queue_data[pid]["sex_conf"] = sex_conf
                                        try:
                                            Path(head_crop).unlink(missing_ok=True)
                                        except Exception:
                                            pass

            except Exception as e:
                logger.warning(f"AI processing failed for photo {p.get('id')}: {e}")

        progress.setValue(total)
        progress.close()
        self.ai_processing = False

        # Reload photos from database to get updated suggestions
        self.photos = self._sorted_photos(self.db.get_all_photos())

        # Show results
        if self.queue_photo_ids:
            self.queue_title_label.setText(f"Species Review ({len(self.queue_photo_ids)})")
            self.queue_suggestion_label.setText(f"AI complete: {counts['species']} species, {counts['sex']} buck/doe")
            self.queue_progress_bar.hide()
            self.queue_panel.show()
            self._populate_photo_list()
            self._update_queue_ui()
            QMessageBox.information(self, "AI Complete",
                f"Processed {total} photos.\n"
                f"Species identified: {counts['species']}\n"
                f"Buck/doe identified: {counts['sex']}\n\n"
                f"Ready for review.")
        else:
            self.queue_mode = False
            self.queue_panel.hide()
            self._populate_photo_list()  # Still refresh the list
            QMessageBox.information(self, "AI Complete",
                f"Processed {total} photos but no species suggestions were made.")

    def run_ai_suggestions_all(self):
        """Run AI suggestions on all photos in DB (uses background thread).

        Automatically runs detection first if no boxes exist, then uses
        subject/head crops for better classification accuracy.
        """
        # Just call the background version - no need for blocking UI
        self.run_ai_suggestions_background()

    def run_sex_suggestions_on_deer(self):
        """Run buck/doe suggestions on deer boxes without sex predictions.

        Runs per-box: each deer box gets its own buck/doe prediction.
        Automatically runs detection first if no boxes exist.
        """
        if not self.suggester or not self.suggester.buckdoe_ready:
            QMessageBox.information(self, "Buck/Doe Model Not Available",
                "Buck/doe classifier not loaded.\nPlace buckdoe.onnx in the models/ folder.")
            return

        # Find photos with deer boxes that need sex prediction
        # Skip photos suggested as "Verification" - these are test photos
        all_photos = self.db.get_all_photos()
        photos_to_process = []

        for p in all_photos:
            pid = p.get("id")
            if not pid:
                continue
            # Skip verification photos
            if p.get("suggested_tag") == "Verification":
                continue
            boxes = self.db.get_boxes(pid)
            # Check if any deer box lacks sex prediction
            has_deer_without_sex = False
            for box in boxes:
                species = (box.get("species") or "").lower()
                ai_species = (box.get("ai_suggested_species") or "").lower()
                if (species == "deer" or ai_species == "deer") and not box.get("sex"):
                    has_deer_without_sex = True
                    break
            if has_deer_without_sex:
                photos_to_process.append(p)

        if not photos_to_process:
            QMessageBox.information(self, "Buck/Doe Suggestions",
                "No deer boxes without buck/doe predictions found.")
            return

        # Get detector for auto-detection if needed
        detector, names = self._get_detector_for_suggestions()

        total_boxes = 0
        progress = QProgressDialog(
            f"Running buck/doe predictions on {len(photos_to_process)} photos...",
            "Cancel", 0, len(photos_to_process), self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        for i, p in enumerate(photos_to_process):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(
                f"Processing photo {i + 1} of {len(photos_to_process)}..."
            )
            QApplication.processEvents()
            # Auto-detect boxes if none exist (enables head crops)
            self._ensure_detection_boxes(p, detector=detector, names=names)
            # Run per-box sex prediction
            box_count = self._predict_sex_for_deer_boxes(p)
            total_boxes += box_count

        progress.setValue(len(photos_to_process))

        msg = f"Suggested buck/doe for {total_boxes} deer box(es) across {len(photos_to_process)} photo(s)."
        msg += "\nUse 'Review Buck/Doe Suggestions' to verify."
        QMessageBox.information(self, "Buck/Doe Suggestions", msg)
        self._populate_photo_list()

    # ====== INTEGRATED QUEUE MODE ======

    def _enter_queue_mode(self, queue_type: str, photo_ids: list, data: dict = None, title: str = "Review Queue"):
        """Enter queue mode with the given photos.

        Args:
            queue_type: Type of queue ('species', 'sex', 'boxes', etc.)
            photo_ids: List of photo IDs to include in queue
            data: Optional dict mapping photo_id to extra data (e.g., suggestions)
            title: Title to display in queue panel
        """
        if not photo_ids:
            QMessageBox.information(self, title, "No items to review.")
            return

        # Remember current position
        if self.photos and self.index < len(self.photos):
            self.queue_pre_filter_index = self.index

        # Disable AI review mode if active — only one review system at a time
        if self.ai_review_mode:
            self.ai_review_mode = False
            self.ai_review_queue = []
            self.ai_reviewed_photos = set()
            self._advancing_review = False

        self.queue_mode = True
        self.queue_type = queue_type
        self.queue_photo_ids = list(photo_ids)
        self.queue_data = data or {}
        self.queue_reviewed = set()

        # Update queue panel
        self.queue_title_label.setText(title)
        self.queue_panel.show()

        # Refresh photo list to show only queue photos
        self._populate_photo_list()

        # Jump to first photo in queue
        if self.queue_photo_ids:
            first_pid = self.queue_photo_ids[0]
            for i, p in enumerate(self.photos):
                if p.get("id") == first_pid:
                    self.index = i
                    break
            self.load_photo()

        self._update_queue_ui()

    def _check_special_queue_menu(self):
        """Check for special one-time review queue files and add menu items if they exist."""
        import json

        # Check for Turkey queue
        turkey_file = os.path.expanduser("~/.trailcam/turkey_review_queue.json")
        if os.path.exists(turkey_file):
            try:
                with open(turkey_file, 'r') as f:
                    photo_ids = json.load(f)
                if photo_ids:
                    action = self.tools_menu.addAction(f"★ Review Turkey Boxes ({len(photo_ids)} photos)")
                    action.triggered.connect(lambda: self._run_special_queue("turkey", turkey_file))
                    self._special_queue_actions = getattr(self, '_special_queue_actions', [])
                    self._special_queue_actions.append(action)
            except Exception:
                pass

        # Check for Quail queue
        quail_file = os.path.expanduser("~/.trailcam/quail_review_queue.json")
        if os.path.exists(quail_file):
            try:
                with open(quail_file, 'r') as f:
                    photo_ids = json.load(f)
                if photo_ids:
                    action = self.tools_menu.addAction(f"★ Review Quail Boxes ({len(photo_ids)} photos)")
                    action.triggered.connect(lambda: self._run_special_queue("quail", quail_file))
                    self._special_queue_actions = getattr(self, '_special_queue_actions', [])
                    self._special_queue_actions.append(action)
            except Exception:
                pass

        # Check for Tightwad House comparison queue
        tightwad_file = os.path.expanduser("~/.trailcam/tightwad_review_queue.json")
        if os.path.exists(tightwad_file):
            try:
                with open(tightwad_file, 'r') as f:
                    items = json.load(f)
                if items:
                    action = self.tools_menu.addAction(f"★ Tightwad Comparison ({len(items)} photos)")
                    action.triggered.connect(lambda: self._run_tightwad_comparison_queue(tightwad_file))
                    self._special_queue_actions = getattr(self, '_special_queue_actions', [])
                    self._special_queue_actions.append(action)
            except Exception:
                pass

        # Check for Misclassified (Deer/Turkey confusion) queue
        misclassified_file = os.path.expanduser("~/.trailcam/misclassified_review_queue.json")
        if os.path.exists(misclassified_file):
            try:
                with open(misclassified_file, 'r') as f:
                    photo_ids = json.load(f)
                if photo_ids:
                    action = self.tools_menu.addAction(f"★ Review Misclassified ({len(photo_ids)} photos)")
                    action.triggered.connect(lambda: self._run_special_queue("misclassified", misclassified_file))
                    self._special_queue_actions = getattr(self, '_special_queue_actions', [])
                    self._special_queue_actions.append(action)
            except Exception:
                pass

        # Check for Verification review queue (uncertain species)
        verification_file = os.path.expanduser("~/.trailcam/verification_review_queue.json")
        if os.path.exists(verification_file):
            try:
                with open(verification_file, 'r') as f:
                    queue_data = json.load(f)
                if queue_data:
                    action = self.tools_menu.addAction(f"★ Review Verification ({len(queue_data)} items)")
                    action.triggered.connect(lambda: self._run_special_queue("verification", verification_file))
                    self._special_queue_actions = getattr(self, '_special_queue_actions', [])
                    self._special_queue_actions.append(action)
            except Exception:
                pass

    def _run_tightwad_comparison_queue(self, queue_file: str):
        """Run the Tightwad House comparison review queue."""
        import json

        if not os.path.exists(queue_file):
            QMessageBox.information(self, "Comparison Queue", "Queue file not found.")
            return

        with open(queue_file, 'r') as f:
            items = json.load(f)

        if not items:
            QMessageBox.information(self, "Comparison Queue", "No photos in queue.")
            os.remove(queue_file)
            return

        # Create comparison review dialog
        dlg = TightwadComparisonDialog(self, items, self.db, queue_file)
        dlg.photo_selected.connect(self._jump_to_photo)
        dlg.exec()

        # Refresh menu after dialog closes
        self._check_special_queue_menu()

    def _jump_to_photo(self, photo_id: int):
        """Jump to a specific photo by ID."""
        # First, ensure we can find the photo (may need to reset filters)
        found = False
        for i, p in enumerate(self.photos):
            if p.get("id") == photo_id:
                self.index = i
                found = True
                break

        if not found:
            # Photo not in current filter - try loading with all photos
            self.collection_filter_combo.blockSignals(True)
            self.collection_filter_combo.setCurrentText("All Collections")
            self.collection_filter_combo.blockSignals(False)
            self.archive_filter_combo.blockSignals(True)
            self.archive_filter_combo.setCurrentText("All Photos")
            self.archive_filter_combo.blockSignals(False)
            if hasattr(self.species_filter_combo, 'select_all'):
                self.species_filter_combo.select_all()
            else:
                self.species_filter_combo.blockSignals(True)
                self.species_filter_combo.setCurrentText("All Species")
                self.species_filter_combo.blockSignals(False)
            self._populate_photo_list()

            for i, p in enumerate(self.photos):
                if p.get("id") == photo_id:
                    self.index = i
                    found = True
                    break

        if found:
            self.load_photo()
            # Update list selection
            for row in range(self.photo_list_widget.count()):
                item = self.photo_list_widget.item(row)
                if item and item.data(Qt.ItemDataRole.UserRole) == self.index:
                    self.photo_list_widget.setCurrentItem(item)
                    break

    def _run_special_queue(self, queue_name: str, queue_file: str):
        """Run a special review queue."""
        import json

        if not os.path.exists(queue_file):
            QMessageBox.information(self, "Special Queue", "Queue file not found.")
            return

        with open(queue_file, 'r') as f:
            queue_data = json.load(f)

        if not queue_data:
            QMessageBox.information(self, "Special Queue", "No photos in queue.")
            os.remove(queue_file)
            return

        # Handle new format with {id, old_tag, new_suggestion} or {photo_id, ...} or old format with just IDs
        if isinstance(queue_data[0], dict):
            # Check which key is used for photo ID
            id_key = 'photo_id' if 'photo_id' in queue_data[0] else 'id'
            photo_ids = list(set(item[id_key] for item in queue_data))  # Unique photo IDs
            # Store comparison data for display during review
            self._queue_comparison_data = {item[id_key]: item for item in queue_data}
        else:
            photo_ids = queue_data
            self._queue_comparison_data = {}

        title = queue_name.title()

        # Enter queue mode directly (AI detection already done)
        self._enter_queue_mode(
            queue_type=f"{queue_name}_boxes",
            photo_ids=photo_ids,
            title=f"{title} Box Review ({len(photo_ids)} photos)"
        )

        # Connect exit to cleanup
        self._special_queue_cleanup = lambda: self._cleanup_special_queue(queue_file)

    def _cleanup_special_queue(self, queue_file: str):
        """Remove the special queue file after review is complete."""
        try:
            if os.path.exists(queue_file):
                os.remove(queue_file)
            # Remove menu items for this queue
            if hasattr(self, '_special_queue_actions'):
                for action in self._special_queue_actions[:]:
                    if queue_file.split('/')[-1].replace('_review_queue.json', '') in action.text().lower():
                        self.tools_menu.removeAction(action)
                        self._special_queue_actions.remove(action)
        except Exception:
            pass

    def _exit_queue_mode(self):
        """Exit queue mode and return to normal view."""
        if not self.queue_mode:
            return

        # Check for special queue cleanup
        if hasattr(self, '_special_queue_cleanup') and self._special_queue_cleanup:
            self._special_queue_cleanup()
            self._special_queue_cleanup = None

        # Remember current photo to stay on it
        current_photo_id = None
        if self.photos and self.index < len(self.photos):
            current_photo_id = self.photos[self.index].get("id")

        self.queue_mode = False
        self.queue_type = None
        self.queue_photo_ids = []
        self.queue_data = {}
        self.queue_reviewed = set()

        self.queue_panel.hide()

        # Refresh photo list
        self._populate_photo_list()

        # Try to stay on the same photo, or return to pre-queue position
        if current_photo_id:
            for i, p in enumerate(self.photos):
                if p.get("id") == current_photo_id:
                    self.index = i
                    self.load_photo()
                    # Select in list
                    for row in range(self.photo_list_widget.count()):
                        item = self.photo_list_widget.item(row)
                        if item.data(Qt.ItemDataRole.UserRole) == i:
                            self.photo_list_widget.setCurrentItem(item)
                            break
                    return

        # Fallback to pre-queue position
        if self.queue_pre_filter_index < len(self.photos):
            self.index = self.queue_pre_filter_index
            self.load_photo()

    def _update_queue_ui(self):
        """Update the queue panel UI with current state."""
        if not self.queue_mode:
            return

        # Count reviewed vs total
        total = len(self.queue_photo_ids)
        reviewed = len(self.queue_reviewed)
        remaining = total - reviewed
        self.queue_count_label.setText(f"{reviewed} / {total} reviewed ({remaining} remaining)")

        # Get current photo's queue data
        current_pid = None
        if self.photos and self.index < len(self.photos):
            current_pid = self.photos[self.index].get("id")

        if current_pid and current_pid in self.queue_data:
            data = self.queue_data[current_pid]
            if self.queue_type == "species":
                species = data.get("species", "?")
                conf = data.get("conf", 0)
                self.queue_suggestion_label.setText(f"Suggested: {species} ({conf:.0%})")
            elif self.queue_type == "sex":
                sex = data.get("sex", "?")
                conf = data.get("conf", 0)
                self.queue_suggestion_label.setText(f"Suggested: {sex} ({conf:.0%})")
            else:
                self.queue_suggestion_label.setText("")
        else:
            self.queue_suggestion_label.setText("")

    def _queue_accept(self):
        """Accept the current suggestion and move to next."""
        if not self.queue_mode:
            return

        current_pid = None
        if self.photos and self.index < len(self.photos):
            current_pid = self.photos[self.index].get("id")

        if current_pid:
            # Apply the suggestion based on queue type
            if self.queue_type == "species" and current_pid in self.queue_data:
                species = self.queue_data[current_pid].get("species")
                if species:
                    self.db.add_tag(current_pid, species)
                    self.db.set_suggested_tag(current_pid, None, None)
                    # Mark as reviewed
                    self.queue_reviewed.add(current_pid)
                    # Remove from queue
                    if current_pid in self.queue_photo_ids:
                        self.queue_photo_ids.remove(current_pid)

            elif self.queue_type == "sex" and current_pid in self.queue_data:
                sex = self.queue_data[current_pid].get("sex")
                if sex:
                    self.db.add_tag(current_pid, sex)
                    self.db.set_suggested_sex(current_pid, None, None)
                    self.queue_reviewed.add(current_pid)
                    if current_pid in self.queue_photo_ids:
                        self.queue_photo_ids.remove(current_pid)

            # Mark as reviewed (green highlight)
            self._mark_current_list_item_reviewed()

            # Refresh display
            self._populate_species_dropdown()
            self._update_recent_species_buttons()
            self.load_photo()

        # Move to next or exit if done
        self._queue_advance()

    def _queue_reject(self):
        """Reject AI suggestion and move to next photo."""
        if not self.queue_mode:
            return

        current_pid = None
        if self.photos and self.index < len(self.photos):
            current_pid = self.photos[self.index].get("id")

        if current_pid:
            # Clear the AI suggestion since user rejected it
            if self.queue_type == "species":
                self.db.set_suggested_tag(current_pid, None, None)
            elif self.queue_type == "sex":
                self.db.set_suggested_sex(current_pid, None, None)
            self.queue_reviewed.add(current_pid)
            self._mark_current_list_item_reviewed()

        self._queue_advance()

    def _mark_current_list_item_reviewed(self):
        """Quickly mark current list item green without rebuilding list."""
        current_item = self.photo_list_widget.currentItem()
        if current_item:
            current_item.setBackground(QColor(144, 238, 144))  # Light green

    def _queue_advance(self):
        """Move to next unreviewed photo in queue, or exit if done."""
        if not self.queue_mode or not self.queue_photo_ids:
            self._exit_queue_mode()
            return

        # Find current photo's position in queue
        current_pid = None
        if self.photos and self.index < len(self.photos):
            current_pid = self.photos[self.index].get("id")

        current_queue_idx = -1
        if current_pid and current_pid in self.queue_photo_ids:
            try:
                current_queue_idx = self.queue_photo_ids.index(current_pid)
            except ValueError:
                current_queue_idx = -1

        # Look for next unreviewed photo AFTER current position
        # Start from current+1, then wrap around if needed
        queue_len = len(self.queue_photo_ids)
        for offset in range(1, queue_len + 1):
            check_idx = (current_queue_idx + offset) % queue_len
            pid = self.queue_photo_ids[check_idx]
            if pid not in self.queue_reviewed:
                # Find this photo in the list
                for i, p in enumerate(self.photos):
                    if p.get("id") == pid:
                        self.index = i
                        self.load_photo()
                        # Select in list without rebuilding
                        for row in range(self.photo_list_widget.count()):
                            item = self.photo_list_widget.item(row)
                            idx = item.data(Qt.ItemDataRole.UserRole)
                            if idx is not None and 0 <= idx < len(self.photos) and self.photos[idx].get("id") == pid:
                                self.photo_list_widget.blockSignals(True)
                                self.photo_list_widget.setCurrentItem(item)
                                self.photo_list_widget.scrollToItem(item)
                                self.photo_list_widget.blockSignals(False)
                                break
                        self._update_queue_ui()
                        return

        # All done
        reviewed_count = len(self.queue_reviewed)
        self._exit_queue_mode()
        QMessageBox.information(self, "Queue Complete", f"Finished reviewing {reviewed_count} photo(s).")

    def review_label_suggestions(self):
        """Review and approve/reject pending label suggestions from club members."""
        from PyQt6.QtWidgets import QGraphicsView

        username = self._get_current_username()
        pending = self.db.get_pending_suggestions(owner_username=username)
        if not pending:
            QMessageBox.information(self, "Label Suggestions", "No pending label suggestions to review.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Review Label Suggestions ({len(pending)})")
        dlg.resize(1000, 650)
        layout = QVBoxLayout(dlg)

        # Main content: list on left, photo on right
        content_layout = QHBoxLayout()

        # Left side: list of suggestions
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel(f"Pending suggestions ({len(pending)}):"))
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setMaximumWidth(320)

        for item in pending:
            text = f"{item['tag_name']} — by {item['suggested_by']}"
            li = QListWidgetItem(text)
            li.setData(Qt.ItemDataRole.UserRole, item)
            list_widget.addItem(li)

        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)

        left_panel.addWidget(list_widget)
        content_layout.addLayout(left_panel)

        # Right side: photo preview + action buttons
        right_panel = QVBoxLayout()
        suggestion_label = QLabel("Suggestion: —")
        suggestion_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        right_panel.addWidget(suggestion_label)

        # Photo preview
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.setMinimumSize(600, 450)
        view.setStyleSheet("background: #222; border: 1px solid #444;")
        right_panel.addWidget(view, 1)

        # Action buttons
        btn_row = QHBoxLayout()
        approve_btn = QPushButton("Approve")
        approve_btn.setStyleSheet("background: #2d5a2d; color: white; padding: 8px 20px; font-weight: bold;")
        reject_btn = QPushButton("Reject")
        reject_btn.setStyleSheet("background: #5a2d2d; color: white; padding: 8px 20px; font-weight: bold;")
        skip_btn = QPushButton("Skip")
        skip_btn.setStyleSheet("padding: 8px 20px;")
        btn_row.addWidget(approve_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addWidget(skip_btn)
        btn_row.addStretch()
        right_panel.addLayout(btn_row)

        content_layout.addLayout(right_panel, 1)
        layout.addLayout(content_layout)

        # Status bar
        status_label = QLabel("")
        layout.addWidget(status_label)

        # Counter
        counts = {"approved": 0, "rejected": 0}

        def _update_preview():
            item = list_widget.currentItem()
            if not item:
                suggestion_label.setText("Suggestion: —")
                scene.clear()
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            suggestion_label.setText(
                f"Suggestion: \"{data['tag_name']}\" by {data['suggested_by']}")
            # Load photo
            scene.clear()
            path = data.get("file_path", "")
            if path and os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scene.addPixmap(pixmap)
                    view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        def _approve():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            username = self._get_current_username()
            self.db.approve_suggestion(data["id"], username)
            counts["approved"] += 1
            row = list_widget.row(item)
            list_widget.takeItem(row)
            status_label.setText(
                f"Approved: {counts['approved']}  Rejected: {counts['rejected']}  Remaining: {list_widget.count()}")
            if list_widget.count() == 0:
                QMessageBox.information(dlg, "Done", "All suggestions reviewed!")
                dlg.accept()

        def _reject():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            username = self._get_current_username()
            self.db.reject_suggestion(data["id"], username)
            counts["rejected"] += 1
            row = list_widget.row(item)
            list_widget.takeItem(row)
            status_label.setText(
                f"Approved: {counts['approved']}  Rejected: {counts['rejected']}  Remaining: {list_widget.count()}")
            if list_widget.count() == 0:
                QMessageBox.information(dlg, "Done", "All suggestions reviewed!")
                dlg.accept()

        def _skip():
            current = list_widget.currentRow()
            if current < list_widget.count() - 1:
                list_widget.setCurrentRow(current + 1)

        list_widget.currentItemChanged.connect(lambda: _update_preview())
        approve_btn.clicked.connect(_approve)
        reject_btn.clicked.connect(_reject)
        skip_btn.clicked.connect(_skip)

        _update_preview()
        dlg.exec()

        # Refresh main view after reviewing
        if counts["approved"] > 0 or counts["rejected"] > 0:
            self.load_photo()

    def review_species_suggestions_integrated(self):
        """Enter integrated queue mode for species review."""
        pending = self._gather_pending_species_suggestions()
        if not pending:
            QMessageBox.information(self, "Species Suggestions", "No pending species suggestions to review.")
            return

        # Build queue data
        photo_ids = [item["id"] for item in pending]
        queue_data = {}
        for item in pending:
            queue_data[item["id"]] = {
                "species": item.get("species"),
                "conf": item.get("conf", 0)
            }

        self._enter_queue_mode(
            queue_type="species",
            photo_ids=photo_ids,
            data=queue_data,
            title=f"Species Review ({len(pending)})"
        )

    def _review_species_batch_grid(self, pending):
        """Batch grid view for species review - 16 thumbnails at a time."""
        # Group by suggested species
        by_species = {}
        for item in pending:
            sp = item.get("species", "Unknown")
            if sp not in by_species:
                by_species[sp] = []
            by_species[sp].append(item)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Species Review ({len(pending)} photos)")
        dlg.resize(1200, 850)
        layout = QVBoxLayout(dlg)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Suggested:"))
        species_filter = QComboBox()
        species_filter.addItem(f"All ({len(pending)})")
        for sp, items in sorted(by_species.items(), key=lambda x: -len(x[1])):
            species_filter.addItem(f"{sp} ({len(items)})")
        species_filter.setMinimumWidth(200)
        top_row.addWidget(species_filter)
        top_row.addStretch()

        page_label = QLabel("Page 1 of 1")
        prev_btn = QPushButton("Prev")
        next_btn = QPushButton("Next")
        top_row.addWidget(prev_btn)
        top_row.addWidget(page_label)
        top_row.addWidget(next_btn)

        select_all_btn = QPushButton("Select All")
        clear_sel_btn = QPushButton("Clear Sel")
        top_row.addWidget(select_all_btn)
        top_row.addWidget(clear_sel_btn)
        layout.addLayout(top_row)

        GRID_COLS, GRID_ROWS, THUMB_SIZE = 4, 4, 200
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(5)

        thumb_widgets = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                cell = QWidget()
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(2, 2, 2, 2)
                cell_layout.setSpacing(2)
                thumb_label = QLabel()
                thumb_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
                thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                thumb_label.setStyleSheet("background-color: #333; border: 2px solid #555;")
                info_label = QLabel()
                info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                info_label.setStyleSheet("font-size: 10px; color: #aaa;")
                check = QCheckBox()
                cell_layout.addWidget(thumb_label)
                cell_layout.addWidget(info_label)
                cell_layout.addWidget(check, alignment=Qt.AlignmentFlag.AlignCenter)
                grid_layout.addWidget(cell, row, col)
                thumb_widgets.append({"label": thumb_label, "info": info_label, "check": check, "item": None})

        scroll = QScrollArea()
        scroll.setWidget(grid_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        tag_row = QHBoxLayout()
        tag_row.addWidget(QLabel("Apply to selected:"))
        accept_btn = QPushButton("Accept Suggestion")
        accept_btn.setStyleSheet("background-color: #66aa66; font-weight: bold;")
        tag_row.addWidget(accept_btn)
        for species in ["Deer", "Turkey", "Raccoon", "Squirrel", "Opossum", "Empty"]:
            btn = QPushButton(species)
            btn.setMinimumWidth(60)
            if species == "Deer":
                btn.setStyleSheet("background-color: #8B4513; color: white;")
            tag_row.addWidget(btn)
            btn.clicked.connect(lambda checked, s=species: apply_tag(s))
        tag_row.addStretch()
        skip_btn = QPushButton("Skip Selected")
        tag_row.addWidget(skip_btn)
        single_view_btn = QPushButton("Single View")
        single_view_btn.setStyleSheet("background-color: #555; color: white;")
        single_view_btn.setToolTip("Switch to single photo view")
        tag_row.addWidget(single_view_btn)
        close_btn = QPushButton("Close")
        tag_row.addWidget(close_btn)
        layout.addLayout(tag_row)

        switch_to_single = [False]

        current_items = []
        current_page = [0]
        ITEMS_PER_PAGE = GRID_COLS * GRID_ROWS

        def get_filtered():
            ft = species_filter.currentText()
            fr = ft.split(" (")[0] if " (" in ft else None
            return pending[:] if fr == "All" else [i for i in pending if i.get("species") == fr]

        def load_page():
            nonlocal current_items
            current_items = get_filtered()
            tp = max(1, (len(current_items) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
            current_page[0] = max(0, min(current_page[0], tp - 1))
            page_label.setText(f"Page {current_page[0] + 1} of {tp}")
            prev_btn.setEnabled(current_page[0] > 0)
            next_btn.setEnabled(current_page[0] < tp - 1)
            start = current_page[0] * ITEMS_PER_PAGE
            page_items = current_items[start:start + ITEMS_PER_PAGE]
            for i, tw in enumerate(thumb_widgets):
                if i < len(page_items):
                    item = page_items[i]
                    tw["item"] = item
                    tw["check"].setChecked(False)
                    tw["check"].setVisible(True)
                    tw["label"].setVisible(True)
                    tw["info"].setVisible(True)
                    tw["info"].setText(f"{item.get('species', '?')} ({int(item.get('conf', 0)*100)}%)")
                    path = item.get("path", "")
                    if path and os.path.exists(path):
                        px = QPixmap(path)
                        if not px.isNull():
                            tw["label"].setPixmap(px.scaled(THUMB_SIZE, THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
                        else:
                            tw["label"].setText("Error")
                    else:
                        tw["label"].setText("N/A")
                else:
                    tw["item"] = None
                    tw["check"].setVisible(False)
                    tw["label"].setVisible(False)
                    tw["info"].setVisible(False)
                    tw["label"].clear()
            dlg.setWindowTitle(f"Species Review ({len(current_items)} photos)")

        def get_selected():
            return [tw["item"] for tw in thumb_widgets if tw["item"] and tw["check"].isChecked()]

        def apply_tag(species):
            sel = get_selected()
            if not sel:
                QMessageBox.information(dlg, "No Selection", "Select photos first.")
                return
            boxes_labeled = 0
            for item in sel:
                pid = item["id"]
                # Apply species to ALL boxes on this photo
                boxes = self.db.get_boxes(pid)
                for box in boxes:
                    box_id = box.get("id")
                    if box_id:
                        self.db.set_box_species(box_id, species, 1.0)
                        boxes_labeled += 1
                # Also add tag to photo for consistency
                self.db.add_tag(pid, species)
                self.db.set_suggested_tag(pid, None, None)
                if item in pending:
                    pending.remove(item)
            load_page()
            if not pending:
                QMessageBox.information(dlg, "Done", f"All reviewed! ({boxes_labeled} boxes labeled)")
                dlg.close()

        def accept_suggestions():
            sel = get_selected()
            if not sel:
                QMessageBox.information(dlg, "No Selection", "Select photos first.")
                return
            boxes_labeled = 0
            for item in sel:
                pid = item["id"]
                species = item.get("species")
                if species:
                    # Apply species to ALL boxes on this photo
                    boxes = self.db.get_boxes(pid)
                    for box in boxes:
                        box_id = box.get("id")
                        if box_id:
                            self.db.set_box_species(box_id, species, 1.0)
                            boxes_labeled += 1
                    self.db.add_tag(pid, species)
                self.db.set_suggested_tag(pid, None, None)
                if item in pending:
                    pending.remove(item)
            load_page()
            if not pending:
                QMessageBox.information(dlg, "Done", f"All reviewed! ({boxes_labeled} boxes labeled)")
                dlg.close()

        def skip_selected():
            sel = get_selected()
            if not sel:
                return
            for item in sel:
                self.db.set_suggested_tag(item["id"], None, None)
                if item in pending:
                    pending.remove(item)
            load_page()
            if not pending:
                dlg.close()

        species_filter.currentIndexChanged.connect(lambda: (current_page.__setitem__(0, 0), load_page()))
        prev_btn.clicked.connect(lambda: (current_page.__setitem__(0, current_page[0] - 1), load_page()))
        next_btn.clicked.connect(lambda: (current_page.__setitem__(0, current_page[0] + 1), load_page()))
        select_all_btn.clicked.connect(lambda: [tw["check"].setChecked(True) for tw in thumb_widgets if tw["item"]])
        clear_sel_btn.clicked.connect(lambda: [tw["check"].setChecked(False) for tw in thumb_widgets])
        accept_btn.clicked.connect(accept_suggestions)
        skip_btn.clicked.connect(skip_selected)

        def switch_to_single_view():
            switch_to_single[0] = True
            dlg.accept()

        single_view_btn.clicked.connect(switch_to_single_view)
        close_btn.clicked.connect(dlg.close)
        for tw in thumb_widgets:
            tw["label"].mousePressEvent = lambda e, t=tw: t["check"].setChecked(not t["check"].isChecked()) if t["item"] else None
        load_page()
        dlg.exec()

        # After dialog closes, check if we should switch to single view
        if switch_to_single[0] and pending:
            self._review_species_single(pending)

    # ====== BACKGROUND AI PROCESSING ======

    def run_ai_suggestions_background(self):
        """Run AI suggestions in background thread with live queue updates."""
        if self.ai_processing:
            QMessageBox.information(self, "AI Processing", "AI processing is already running.")
            return

        # Ensure SpeciesNet is initialized (download models on first use)
        if not self.speciesnet_wrapper.is_available:
            if not user_config.is_speciesnet_initialized():
                dlg = SpeciesNetDownloadDialog(self.speciesnet_wrapper, parent=self)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                user_config.set_config_value("speciesnet_initialized", True)
            else:
                # Models downloaded previously, just need to load
                self.speciesnet_wrapper.initialize()
            if not self.speciesnet_wrapper.is_available:
                QMessageBox.warning(self, "AI Model Not Available",
                    f"SpeciesNet could not be loaded:\n{self.speciesnet_wrapper.error_message}")
                return

        # Get unlabeled photos (skip Verification photos)
        unlabeled_photos = [p for p in self.photos
                          if not self._photo_has_human_species(p)
                          and p.get("suggested_tag") != "Verification"]
        if not unlabeled_photos:
            QMessageBox.information(self, "AI Suggestions", "All photos in current view already have species labels.")
            return

        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Run AI Suggestions",
            f"Process {len(unlabeled_photos)} unlabeled photo(s) with AI?\n\n"
            "Photos will be added to the review queue as they're processed.\n"
            "You can continue working while AI runs in the background.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Enter queue mode immediately (empty queue, will fill as photos are processed)
        self.queue_mode = True
        self.queue_type = "species"
        self.queue_photo_ids = []
        self.queue_data = {}
        self.queue_reviewed = set()
        if self.photos and self.index < len(self.photos):
            self.queue_pre_filter_index = self.index

        self.queue_title_label.setText(f"AI Processing (0 / {len(unlabeled_photos)})")
        self.queue_suggestion_label.setText("Starting AI processing...")
        self.queue_progress_bar.setRange(0, len(unlabeled_photos))
        self.queue_progress_bar.setValue(0)
        self.queue_progress_bar.show()
        self.queue_panel.show()

        # Create and start worker thread
        self.ai_processing = True
        # Determine detector/classifier based on user settings
        det_choice = user_config.get_detector_choice()
        cls_choice = user_config.get_classifier_choice()

        detector_obj = None
        if det_choice == "megadetector":
            try:
                from ai_detection import MegaDetectorV6
                detector_obj = MegaDetectorV6()
                if not detector_obj.is_available:
                    detector_obj = None
            except Exception:
                detector_obj = None

        species_suggester = None
        if cls_choice == "custom" and self.suggester and self.suggester.ready:
            species_suggester = self.suggester

        ai_options = {
            "detect_boxes": True,
            "species_id": True,
            "deer_head_boxes": True,
            "buck_doe": True,
            "use_megadetector": detector_obj is not None,
            "use_custom_classifier": species_suggester is not None,
        }

        self.ai_worker = AIWorker(
            photos=unlabeled_photos,
            db=self.db,
            speciesnet_wrapper=self.speciesnet_wrapper,
            buckdoe_suggester=self.suggester,
            species_suggester=species_suggester,
            detector=detector_obj,
            parent=self,
            options=ai_options,
        )
        self.ai_worker.progress.connect(self._on_ai_progress)
        self.ai_worker.photo_processed.connect(self._on_ai_photo_processed)
        self.ai_worker.finished_all.connect(self._on_ai_finished)
        self.ai_worker.error.connect(self._on_ai_error)
        self.ai_worker.start()

    def _on_ai_progress(self, current: int, total: int, message: str):
        """Handle progress update from AI worker."""
        self.queue_progress_bar.setValue(current)
        self.queue_title_label.setText(f"AI Processing ({current} / {total})")
        remaining = len(self.queue_photo_ids) - len(self.queue_reviewed)
        self.queue_count_label.setText(f"{remaining} to review")

    def _on_ai_photo_processed(self, result: dict):
        """Handle a photo being processed by AI worker."""
        photo_id = result.get("id")
        species = result.get("species")
        conf = result.get("species_conf", 0)

        if photo_id and species:
            # Add to queue
            if photo_id not in self.queue_photo_ids:
                self.queue_photo_ids.append(photo_id)
                self.queue_data[photo_id] = {
                    "species": species,
                    "conf": conf,
                    "sex": result.get("sex"),
                    "sex_conf": result.get("sex_conf", 0)
                }

            # DON'T call _populate_photo_list() during processing - it's too expensive
            # and causes UI freeze. The list will be refreshed once at the end in _on_ai_finished

            # Update suggestion label (lightweight, always do this)
            remaining = len(self.queue_photo_ids) - len(self.queue_reviewed)
            self.queue_count_label.setText(f"{remaining} to review")

            # If this is the first photo, load it
            if len(self.queue_photo_ids) == 1:
                for i, p in enumerate(self.photos):
                    if p.get("id") == photo_id:
                        self.index = i
                        self.load_photo()
                        break

    def _on_ai_finished(self, counts):
        """Handle AI processing completion."""
        self.ai_processing = False
        self.ai_worker = None
        self.queue_progress_bar.hide()

        # Handle both old (int, int) and new (dict) formats for backwards compatibility
        if isinstance(counts, dict):
            species_count = counts.get("species", 0)
            sex_count = counts.get("sex", 0)
        else:
            # Old format: species_count, sex_count as positional
            species_count = counts
            sex_count = 0

        if self.queue_photo_ids:
            self.queue_title_label.setText(f"Species Review ({len(self.queue_photo_ids)})")
            self.queue_suggestion_label.setText(f"AI complete: {species_count} species, {sex_count} buck/doe")
            self._populate_photo_list()  # Final refresh to show all processed photos
            self._update_queue_ui()
        else:
            self._exit_queue_mode()
            QMessageBox.information(
                self,
                "AI Complete",
                f"Processed {species_count} species suggestion(s).\n"
                f"Suggested buck/doe for {sex_count} deer photo(s).\n\n"
                "No photos require review."
            )

    def _on_ai_finished_with_options(self, counts: dict):
        """Handle AI processing completion with options-aware message."""
        self.ai_processing = False
        self.ai_worker = None
        self.queue_progress_bar.hide()

        # Get options that were used
        options = getattr(self, '_ai_options', {})

        # Build result message based on what was requested
        msg_parts = []
        if options.get("detect_boxes"):
            msg_parts.append(f"Detected boxes on {counts.get('detect', 0)} photo(s)")
        if options.get("species_id"):
            msg_parts.append(f"Suggested species for {counts.get('species', 0)} photo(s)")
        if options.get("deer_head_boxes"):
            msg_parts.append(f"Added deer head boxes on {counts.get('heads', 0)} photo(s)")
        if options.get("buck_doe"):
            msg_parts.append(f"Suggested buck/doe for {counts.get('sex', 0)} deer photo(s)")

        msg = "\n".join(msg_parts) if msg_parts else "Processing complete."

        # Reload photos from database to get updated values
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_species_dropdown()
        self._populate_photo_list()

        if self.queue_photo_ids:
            self.queue_title_label.setText(f"Review ({len(self.queue_photo_ids)})")
            self.queue_suggestion_label.setText(msg)
            self._update_queue_ui()
        else:
            self._exit_queue_mode()
            QMessageBox.information(self, "AI Suggestions Complete", msg)

    def _on_ai_error(self, message: str):
        """Handle AI processing error."""
        self.ai_processing = False
        self.ai_worker = None
        self.queue_progress_bar.hide()
        QMessageBox.warning(self, "AI Error", f"AI processing failed:\n{message}")

    def stop_ai_processing(self):
        """Stop background AI processing."""
        if self.ai_worker and self.ai_processing:
            self.ai_worker.cancel()
            self.ai_processing = False
            self.queue_progress_bar.hide()
            self.queue_suggestion_label.setText("AI processing cancelled")

    def review_species_suggestions(self, pending=None):
        """Review and approve/reject pending species suggestions with zoomable photo preview."""
        from PyQt6.QtWidgets import QGraphicsView

        # Gather pending species suggestions if not passed
        if pending is None:
            pending = self._gather_pending_species_suggestions()
        if not pending:
            QMessageBox.information(self, "Species Suggestions", "No pending species suggestions to review.")
            return

        switch_to_grid = [False]  # Flag to switch to grid view after closing

        # Create review dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Review Species Suggestions ({len(pending)})")
        dlg.resize(1100, 750)
        layout = QVBoxLayout(dlg)

        # Main content: list on left, photo on right
        content_layout = QHBoxLayout()

        # Left side: list of suggestions
        left_panel = QVBoxLayout()

        # Sort controls
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        sort_combo = QComboBox()
        sort_combo.addItem("Species", "species")
        sort_combo.addItem("Confidence", "conf")
        sort_combo.addItem("Filename", "name")
        sort_row.addWidget(sort_combo)
        sort_row.addStretch()
        left_panel.addLayout(sort_row)

        left_panel.addWidget(QLabel("Pending suggestions:"))
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setMaximumWidth(300)

        def _populate_list(sort_key="species"):
            list_widget.clear()
            if sort_key == "species":
                sorted_pending = sorted(pending, key=lambda x: (x.get("species", ""), -(x.get("conf") or 0)))
            elif sort_key == "conf":
                sorted_pending = sorted(pending, key=lambda x: -(x.get("conf") or 0))
            else:
                sorted_pending = sorted(pending, key=lambda x: x.get("name", ""))
            for item in sorted_pending:
                pct = int(item["conf"] * 100) if item["conf"] and item["conf"] <= 1 else int(item["conf"] or 0)
                text = f"{item['species']} ({pct}%) - {item['name'][:25]}"
                li = QListWidgetItem(text)
                li.setData(Qt.ItemDataRole.UserRole, item)
                list_widget.addItem(li)
            if list_widget.count() > 0:
                list_widget.setCurrentRow(0)

        def _on_sort_changed():
            sort_key = sort_combo.currentData()
            _populate_list(sort_key)

        sort_combo.currentIndexChanged.connect(_on_sort_changed)
        _populate_list("species")  # Initial population

        left_panel.addWidget(list_widget)
        content_layout.addLayout(left_panel)

        # Right side: zoomable photo preview
        right_panel = QVBoxLayout()
        suggestion_label = QLabel("Suggestion: —")
        suggestion_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        right_panel.addWidget(suggestion_label)

        # Zoom controls
        zoom_row = QHBoxLayout()
        zoom_label = QLabel("Zoom:")
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(30)
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedWidth(30)
        zoom_fit_btn = QPushButton("Fit")
        zoom_100_btn = QPushButton("100%")
        zoom_row.addWidget(zoom_label)
        zoom_row.addWidget(zoom_out_btn)
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(zoom_fit_btn)
        zoom_row.addWidget(zoom_100_btn)
        zoom_row.addStretch()
        right_panel.addLayout(zoom_row)

        # Graphics view for zoomable image
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.setMinimumSize(700, 550)
        view.setStyleSheet("background: #222; border: 1px solid #444;")
        view.setDragMode(QGraphicsView.DragMode.NoDrag)  # Start with no drag; enabled when zoomed in
        view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        right_panel.addWidget(view, 1)
        content_layout.addLayout(right_panel, 1)

        layout.addLayout(content_layout, 1)

        # Species quick buttons - get all species from database
        species_container = QWidget()
        species_layout = QVBoxLayout(species_container)
        species_layout.setContentsMargins(0, 0, 0, 0)
        species_layout.setSpacing(4)
        species_btn_row1 = QHBoxLayout()
        species_btn_row2 = QHBoxLayout()
        species_btn_row3 = QHBoxLayout()
        species_buttons = {}

        # Get all species used in database
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT t.tag_name FROM tags t
            JOIN annotation_boxes b ON t.photo_id = b.photo_id
            WHERE t.tag_name NOT IN ('Buck', 'Doe', 'Other')
              AND t.deleted_at IS NULL
        """)
        db_species = set(row[0] for row in cursor.fetchall())

        # Birds to lump (except Turkey)
        BIRD_SPECIES = {"Turkey Buzzard", "Flicker", "Quail"}
        if any(b in db_species for b in BIRD_SPECIES):
            db_species -= BIRD_SPECIES
            db_species.add("Other Bird")

        # Sort by priority
        priority = ["Deer", "Empty", "Squirrel", "Turkey", "Raccoon", "Opossum", "Rabbit",
                    "Coyote", "Fox", "Person", "Bobcat", "House Cat", "Other Bird"]
        all_species = [s for s in priority if s in db_species]
        all_species += sorted(db_species - set(priority))

        for i, sp in enumerate(all_species):
            btn = QPushButton(sp)
            btn.setStyleSheet("padding: 6px 12px;")
            species_buttons[sp] = btn
            if i < 7:
                species_btn_row1.addWidget(btn)
            elif i < 14:
                species_btn_row2.addWidget(btn)
            else:
                species_btn_row3.addWidget(btn)

        # "Other (+)" button for adding new species
        other_btn = QPushButton("Other (+)")
        other_btn.setStyleSheet("padding: 6px 12px; background: #444;")
        other_btn.setToolTip("Add a new species")
        species_btn_row2.addWidget(other_btn)

        species_btn_row1.addStretch()
        species_btn_row2.addStretch()
        species_btn_row3.addStretch()
        species_layout.addLayout(species_btn_row1)
        species_layout.addLayout(species_btn_row2)
        species_layout.addLayout(species_btn_row3)
        layout.addWidget(species_container)

        # Action buttons
        btn_row = QHBoxLayout()
        accept_btn = QPushButton("Accept Suggestion (A)")
        accept_btn.setStyleSheet("background: #264; padding: 8px 16px;")
        reject_btn = QPushButton("Reject (R)")
        reject_btn.setStyleSheet("padding: 8px 16px;")
        skip_btn = QPushButton("Skip (S)")
        # Multi-species checkbox - when checked, adding a tag doesn't advance
        multi_checkbox = QCheckBox("Multi-species")
        multi_checkbox.setToolTip("Check to add multiple species to the same photo")
        next_btn = QPushButton("Next (N)")
        next_btn.setToolTip("Move to next photo (use when Multi-species is checked)")
        grid_view_btn = QPushButton("Grid View")
        grid_view_btn.setStyleSheet("background-color: #555; color: white;")
        grid_view_btn.setToolTip("Switch to batch grid view (16 at a time)")
        close_btn = QPushButton("Close")
        btn_row.addWidget(accept_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addWidget(skip_btn)
        btn_row.addWidget(multi_checkbox)
        btn_row.addWidget(next_btn)
        btn_row.addStretch()
        btn_row.addWidget(grid_view_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        current_pixmap = [None]

        def _update_preview():
            item = list_widget.currentItem()
            if not item:
                scene.clear()
                suggestion_label.setText("Suggestion: —")
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            path = data.get("path")
            photo_id = data.get("id")
            species = data.get("species", "")
            conf = data.get("conf", 0)
            pct = int(conf * 100) if conf and conf <= 1 else int(conf or 0)
            suggestion_label.setText(f"AI Suggestion: {species} ({pct}% confidence)")
            if path and os.path.exists(path):
                # Load and scale down to prevent memory exhaustion
                full_pixmap = QPixmap(path)
                if not full_pixmap.isNull():
                    # Scale to max 1600px on longest side for preview
                    max_preview_size = 1600
                    orig_w, orig_h = full_pixmap.width(), full_pixmap.height()
                    if orig_w > max_preview_size or orig_h > max_preview_size:
                        pixmap = full_pixmap.scaled(
                            max_preview_size, max_preview_size,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                    else:
                        pixmap = full_pixmap
                    del full_pixmap  # Free original immediately

                    scene.clear()
                    scene.addPixmap(pixmap)
                    scene.setSceneRect(pixmap.rect().toRectF())
                    current_pixmap[0] = pixmap

                    # Draw boxes if available (skip boxes entirely in bottom 5% - timestamp area)
                    if photo_id:
                        boxes = self.db.get_boxes(photo_id)
                        # Use original dimensions for normalized coords, then scale to pixmap
                        scale_x = pixmap.width() / orig_w
                        scale_y = pixmap.height() / orig_h
                        for b in boxes:
                            # Skip boxes entirely in bottom 5% (timestamp area)
                            if b["y1"] >= 0.95:
                                continue
                            x1 = b["x1"] * orig_w * scale_x
                            y1 = b["y1"] * orig_h * scale_y
                            w = (b["x2"] - b["x1"]) * orig_w * scale_x
                            h = (b["y2"] - b["y1"]) * orig_h * scale_y
                            rect = QRectF(x1, y1, w, h)
                            lbl = b.get("label", "")
                            if str(lbl) == "ai_deer_head" or str(lbl) == "deer_head":
                                pen = QPen(Qt.GlobalColor.magenta)
                            elif str(lbl).startswith("ai_"):
                                pen = QPen(Qt.GlobalColor.yellow)
                            else:
                                pen = QPen(Qt.GlobalColor.green)
                            pen.setWidth(4)  # Thicker lines
                            scene.addRect(rect, pen)

                    _zoom_fit()
                else:
                    scene.clear()
                    current_pixmap[0] = None
            else:
                scene.clear()
                current_pixmap[0] = None

        min_scale = [0.1]  # Will be updated to fit scale
        max_scale = 5.0

        def _get_current_scale():
            return view.transform().m11()

        def _zoom_in():
            current = _get_current_scale()
            if current < max_scale:
                view.scale(1.25, 1.25)

        def _zoom_out():
            current = _get_current_scale()
            # Don't zoom out past the fit scale
            if current > min_scale[0] * 1.1:
                view.scale(0.8, 0.8)

        def _zoom_fit():
            if current_pixmap[0] and scene.sceneRect().width() > 0:
                view.resetTransform()
                # Use fitInView for proper scaling with aspect ratio preserved
                view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                # Store fit scale as minimum (with small padding)
                min_scale[0] = _get_current_scale() * 0.95
                view.scale(0.95, 0.95)

        def _zoom_100():
            view.resetTransform()

        reviewed_items = set()  # Track reviewed item row indices
        reviewed_photo_ids = set()  # Track reviewed photo IDs
        added_tags = {}  # Track tags added per photo: {photo_id: [tags]}

        def _mark_reviewed(item, action_text: str):
            """Mark item as reviewed with green highlight and update text."""
            row = list_widget.row(item)
            reviewed_items.add(row)
            data = item.data(Qt.ItemDataRole.UserRole)
            original_name = data.get("name", "")[:25]
            item.setText(f"✓ {action_text} - {original_name}")
            item.setBackground(QColor(144, 238, 144))  # Light green
            # Update window title with remaining count
            remaining = list_widget.count() - len(reviewed_items)
            dlg.setWindowTitle(f"Review Species Suggestions ({remaining} remaining)")

        def _next_unreviewed():
            """Move to next unreviewed item."""
            current = list_widget.currentRow()
            total = list_widget.count()

            # Look forward first
            for i in range(current + 1, total):
                item = list_widget.item(i)
                if item:
                    data = item.data(Qt.ItemDataRole.UserRole)
                    pid = data.get("id") if data else None
                    if pid and pid not in reviewed_photo_ids and i not in reviewed_items:
                        list_widget.setCurrentRow(i)
                        _update_preview()
                        return

            # Then look from beginning
            for i in range(0, current):
                item = list_widget.item(i)
                if item:
                    data = item.data(Qt.ItemDataRole.UserRole)
                    pid = data.get("id") if data else None
                    if pid and pid not in reviewed_photo_ids and i not in reviewed_items:
                        list_widget.setCurrentRow(i)
                        _update_preview()
                        return

            # All reviewed - stay on current
            _update_preview()

        def _accept_as(species_tag: str):
            row = list_widget.currentRow()
            if row < 0:
                return
            item = list_widget.item(row)
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data["id"]
            ai_suggested = data.get("species", "")

            # Initialize tracking for this photo
            if pid not in added_tags:
                added_tags[pid] = []

            # Toggle behavior: if already added, remove it
            if species_tag in added_tags[pid]:
                added_tags[pid].remove(species_tag)
                self.db.remove_tag(pid, species_tag)
                # Update display
                if added_tags[pid]:
                    tags_str = " + ".join(added_tags[pid])
                    _mark_reviewed(item, tags_str)
                else:
                    # No tags left - unmark as reviewed
                    reviewed_items.discard(row)
                    reviewed_photo_ids.discard(pid)
                    original_name = data.get("name", "")[:25]
                    pct = int((data.get("conf") or 0) * 100)
                    item.setText(f"{data.get('species', '')} ({pct}%) - {original_name}")
                    item.setBackground(QColor(255, 255, 255))  # White
                    remaining = list_widget.count() - len(reviewed_items)
                    dlg.setWindowTitle(f"Review Species Suggestions ({remaining} remaining)")
                return  # Don't advance after toggle-off

            # Log if user corrected the AI (accepted different label)
            if ai_suggested and ai_suggested != species_tag:
                self.db.log_ai_rejection(
                    photo_id=pid,
                    suggestion_type="species",
                    ai_suggested=ai_suggested,
                    correct_label=species_tag,
                    model_version=get_model_version("species")
                )

            # Add tag
            self.db.add_tag(pid, species_tag)
            added_tags[pid].append(species_tag)

            # Handle boxes based on species
            if species_tag in ("Verification", "Empty"):
                # Verification/Empty photos should have no detection boxes - delete them
                self.db.set_boxes(pid, [])
            else:
                # For other species, update all boxes to have this species
                # This prevents _persist_boxes from overwriting the tag with box species
                boxes = self.db.get_boxes(pid)
                for box in boxes:
                    if box.get("label") and "head" in box.get("label", "").lower():
                        continue  # Skip head boxes
                    self.db.set_box_species(box["id"], species_tag, None)  # Confirmed, not suggestion

            # Clear suggestion from photos table
            self.db.set_suggested_tag(pid, None, None)

            # Mark as reviewed
            reviewed_photo_ids.add(pid)
            tags_str = " + ".join(added_tags[pid])
            _mark_reviewed(item, tags_str)

            # Advance logic
            if not multi_checkbox.isChecked():
                _next_unreviewed()
            # In multi-mode, stay on current (no action needed)

        def _accept_suggestion():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            species = data.get("species", "")
            if species:
                _accept_as(species)

        def _reject():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data["id"]
            ai_suggested = data.get("suggest", "")
            # Log rejection for future model training
            if ai_suggested:
                self.db.log_ai_rejection(
                    photo_id=pid,
                    suggestion_type="species",
                    ai_suggested=ai_suggested,
                    correct_label=None,  # Unknown - user just rejected
                    model_version=get_model_version("species")
                )
            self.db.set_suggested_tag(pid, None, None)
            # Mark as reviewed (green) instead of removing
            _mark_reviewed(item, "REJECTED")
            _next_unreviewed()

        def _skip():
            row = list_widget.currentRow()
            if row < list_widget.count() - 1:
                list_widget.setCurrentRow(row + 1)
                _update_preview()

        def _on_multi_toggled(checked):
            """Update UI when multi-species mode is toggled."""
            if checked:
                next_btn.setStyleSheet("background: #46a; padding: 8px 16px; font-weight: bold;")
                next_btn.setText("Next (N) - Click when done")
            else:
                next_btn.setStyleSheet("padding: 8px 16px;")
                next_btn.setText("Next (N)")

        def _manual_next():
            """User clicked Next - turn off multi-mode and advance."""
            multi_checkbox.setChecked(False)
            _next_unreviewed()

        # Connect signals
        list_widget.currentItemChanged.connect(_update_preview)
        zoom_in_btn.clicked.connect(_zoom_in)
        zoom_out_btn.clicked.connect(_zoom_out)
        zoom_fit_btn.clicked.connect(_zoom_fit)
        zoom_100_btn.clicked.connect(_zoom_100)
        accept_btn.clicked.connect(_accept_suggestion)
        reject_btn.clicked.connect(_reject)
        skip_btn.clicked.connect(_skip)
        next_btn.clicked.connect(_manual_next)
        close_btn.clicked.connect(dlg.accept)
        multi_checkbox.toggled.connect(_on_multi_toggled)

        def _switch_to_grid():
            switch_to_grid[0] = True
            dlg.accept()
        grid_view_btn.clicked.connect(_switch_to_grid)

        # Connect species quick buttons
        for sp, btn in species_buttons.items():
            btn.clicked.connect(lambda checked, s=sp: _accept_as(s))

        # "Other" button - just applies "Other" tag (custom species disabled)
        def _apply_other():
            _accept_as("Other")
        other_btn.clicked.connect(_apply_other)

        # Keyboard shortcuts
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("A"), dlg).activated.connect(_accept_suggestion)
        QShortcut(QKeySequence("R"), dlg).activated.connect(_reject)
        QShortcut(QKeySequence("S"), dlg).activated.connect(_skip)
        QShortcut(QKeySequence("N"), dlg).activated.connect(_manual_next)
        QShortcut(QKeySequence("M"), dlg).activated.connect(lambda: multi_checkbox.setChecked(not multi_checkbox.isChecked()))
        QShortcut(QKeySequence("D"), dlg).activated.connect(lambda: _accept_as("Deer"))
        QShortcut(QKeySequence("T"), dlg).activated.connect(lambda: _accept_as("Turkey"))
        QShortcut(QKeySequence("E"), dlg).activated.connect(lambda: _accept_as("Empty"))
        QShortcut(QKeySequence("O"), dlg).activated.connect(lambda: _accept_as("Other"))
        QShortcut(QKeySequence("P"), dlg).activated.connect(lambda: _accept_as("Person"))
        QShortcut(QKeySequence("V"), dlg).activated.connect(lambda: _accept_as("Vehicle"))
        QShortcut(QKeySequence("+"), dlg).activated.connect(_zoom_in)
        QShortcut(QKeySequence("="), dlg).activated.connect(_zoom_in)
        QShortcut(QKeySequence("-"), dlg).activated.connect(_zoom_out)
        QShortcut(QKeySequence("F"), dlg).activated.connect(_zoom_fit)

        is_fit_mode = [True]  # Track fit vs 100% state for double-click toggle

        def _update_drag_mode():
            """Enable drag mode when zoomed in past fit."""
            if _get_current_scale() > min_scale[0] * 1.05:
                view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            else:
                view.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Mouse wheel: Ctrl+scroll to zoom, regular scroll to pan
        original_wheel = view.wheelEvent
        def _wheel_event(event):
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Ctrl+scroll = zoom
                current = _get_current_scale()
                if event.angleDelta().y() > 0:
                    if current < max_scale:
                        view.scale(1.15, 1.15)
                        is_fit_mode[0] = False
                        _update_drag_mode()
                else:
                    if current > min_scale[0] * 1.05:
                        view.scale(1/1.15, 1/1.15)
                        _update_drag_mode()
                event.accept()
            else:
                # Regular scroll = pan (default behavior)
                QGraphicsView.wheelEvent(view, event)
        view.wheelEvent = _wheel_event

        # Double-click to toggle fit/100%
        def _double_click(event):
            if is_fit_mode[0]:
                _zoom_100()
                is_fit_mode[0] = False
            else:
                _zoom_fit()
                is_fit_mode[0] = True
            _update_drag_mode()
        view.mouseDoubleClickEvent = _double_click

        # Enable gesture events for pinch-to-zoom on trackpad
        view.grabGesture(Qt.GestureType.PinchGesture)
        view.viewport().grabGesture(Qt.GestureType.PinchGesture)

        # Use event filter for reliable gesture handling
        from PyQt6.QtCore import QObject, QEvent
        class GestureFilter(QObject):
            GESTURE_TYPE = QEvent.Type.Gesture  # Class attribute for scope
            def eventFilter(self, obj, event):
                if event.type() == self.GESTURE_TYPE:
                    pinch = event.gesture(Qt.GestureType.PinchGesture)
                    if pinch and pinch.state() == Qt.GestureState.GestureUpdated:
                        current = _get_current_scale()
                        if pinch.scaleFactor() > 1.0:
                            if current < max_scale:
                                view.scale(1.15, 1.15)
                                is_fit_mode[0] = False
                                _update_drag_mode()
                        else:
                            if current > min_scale[0] * 1.05:
                                view.scale(1/1.15, 1/1.15)
                                _update_drag_mode()
                        event.accept()
                        return True
                return False

        gesture_filter = GestureFilter(view)
        view.installEventFilter(gesture_filter)
        view.viewport().installEventFilter(gesture_filter)

        # Select first item and delay initial zoom fit
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
            _update_preview()
            # Delay zoom fit until dialog is shown and properly sized
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(200, _zoom_fit)

        dlg.exec()
        self._populate_photo_list()

        # After dialog closes, check if we should switch to grid view
        if switch_to_grid[0] and pending:
            self._review_species_batch_grid(pending)

    def _review_species_single(self, pending):
        """Single-photo species review (wrapper for toggle)."""
        self.review_species_suggestions(pending)

    def _gather_pending_species_suggestions(self) -> list:
        """Return list of photos with pending species suggestions.

        Excludes photos that:
        - Already have a species tag
        - Have a box with confirmed species (user-labeled, not AI suggestion)
        - Have a box with confirmed sex (Buck/Doe) - implies deer already identified
        - Have a deer_id set - implies deer already identified
        """
        species_set = self._species_set()
        pending = []
        try:
            cursor = self.db.conn.cursor()
            # Get photos with suggestions, excluding those already confirmed
            cursor.execute("""
                SELECT p.id, p.original_name, p.file_path, p.suggested_tag, p.suggested_confidence
                FROM photos p
                WHERE p.suggested_tag IS NOT NULL AND p.suggested_tag != ''
                  AND NOT EXISTS (
                      SELECT 1 FROM annotation_boxes b
                      WHERE b.photo_id = p.id
                        AND b.deleted_at IS NULL
                        AND b.sex IN ('Buck', 'Doe')
                        AND (b.sex_conf IS NULL OR b.sex_conf = 0)
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM deer_metadata dm
                      WHERE dm.photo_id = p.id
                        AND dm.deer_id IS NOT NULL AND dm.deer_id != ''
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM annotation_boxes b
                      WHERE b.photo_id = p.id
                        AND b.deleted_at IS NULL
                        AND b.species IS NOT NULL AND b.species != '' AND b.species != 'Unknown'
                        AND b.label NOT LIKE '%head%'
                        AND (b.species_conf IS NULL OR b.species_conf = 0)
                  )
                ORDER BY p.suggested_tag ASC, p.suggested_confidence DESC
            """)
            for row in cursor.fetchall():
                pid, name, path, species, conf = row
                # Check if already has a species tag
                tags = self.db.get_tags(pid)
                has_species = any(t in species_set for t in tags)
                if not has_species:
                    pending.append({
                        "id": pid,
                        "name": name or "",
                        "path": path,
                        "species": species,
                        "conf": conf or 0
                    })
        except Exception as e:
            logger.warning(f"Failed to gather species suggestions: {e}")
        return pending

    def review_unlabeled_with_boxes(self):
        """Review photos that have AI boxes but no species label or suggestion."""
        from PyQt6.QtWidgets import QGraphicsView

        # Gather photos with boxes but no species label/suggestion
        pending = self._gather_unlabeled_with_boxes()
        if not pending:
            QMessageBox.information(self, "No Photos", "No unlabeled photos with AI boxes found.")
            return

        # Create review dialog (similar to species review)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Label Photos with Boxes ({len(pending)})")
        dlg.resize(1100, 750)
        layout = QVBoxLayout(dlg)

        # Main content: list on left, photo on right
        content_layout = QHBoxLayout()

        # Left side: list of photos
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel(f"Photos with boxes, no label ({len(pending)}):"))
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setMaximumWidth(300)

        for item in pending:
            text = f"{item['name'][:30]}"
            li = QListWidgetItem(text)
            li.setData(Qt.ItemDataRole.UserRole, item)
            list_widget.addItem(li)
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)

        left_panel.addWidget(list_widget)
        content_layout.addLayout(left_panel)

        # Right side: zoomable photo preview
        right_panel = QVBoxLayout()
        info_label = QLabel("Select a species to label this photo")
        info_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        right_panel.addWidget(info_label)

        # Zoom controls
        zoom_row = QHBoxLayout()
        zoom_label = QLabel("Zoom:")
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(30)
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedWidth(30)
        zoom_fit_btn = QPushButton("Fit")
        zoom_100_btn = QPushButton("100%")
        zoom_row.addWidget(zoom_label)
        zoom_row.addWidget(zoom_out_btn)
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(zoom_fit_btn)
        zoom_row.addWidget(zoom_100_btn)
        zoom_row.addStretch()
        right_panel.addLayout(zoom_row)

        # Graphics view for zoomable image with box interaction
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.setMinimumSize(700, 550)
        view.setStyleSheet("background: #222; border: 1px solid #444;")
        view.setDragMode(QGraphicsView.DragMode.NoDrag)
        view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        right_panel.addWidget(view, 1)

        # Box interaction state
        box_items = []  # List of (QGraphicsRectItem, box_data_dict)
        selected_box = [None]  # Currently selected box item
        draw_mode = [False]  # Whether we're in draw mode
        draw_start = [None]  # Start point for drawing
        temp_rect = [None]  # Temporary rectangle while drawing
        content_layout.addLayout(right_panel, 1)

        layout.addLayout(content_layout, 1)

        # Species quick buttons - get all species from database
        species_container = QWidget()
        species_layout = QVBoxLayout(species_container)
        species_layout.setContentsMargins(0, 0, 0, 0)
        species_layout.setSpacing(4)
        species_btn_row1 = QHBoxLayout()
        species_btn_row2 = QHBoxLayout()
        species_btn_row3 = QHBoxLayout()  # More species row
        species_buttons = {}
        custom_species = []  # Track custom species added this session

        # Get all species used in database (on photos with boxes)
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT t.tag_name FROM tags t
            JOIN annotation_boxes b ON t.photo_id = b.photo_id
            WHERE t.tag_name NOT IN ('Buck', 'Doe', 'Empty', 'Other')
              AND t.deleted_at IS NULL
        """)
        db_species = set(row[0] for row in cursor.fetchall())

        # Birds to lump (except Turkey)
        BIRD_SPECIES = {"Turkey Buzzard", "Flicker", "Quail"}
        if any(b in db_species for b in BIRD_SPECIES):
            db_species -= BIRD_SPECIES
            db_species.add("Other Bird")

        # Sort by priority: common species first, then alphabetical
        priority = ["Deer", "Squirrel", "Turkey", "Raccoon", "Opossum", "Rabbit",
                    "Coyote", "Fox", "Person", "Bobcat", "House Cat", "Other Bird"]
        all_species = [s for s in priority if s in db_species]
        all_species += sorted(db_species - set(priority))

        for i, sp in enumerate(all_species):
            btn = QPushButton(sp)
            btn.setStyleSheet("padding: 6px 12px;")
            species_buttons[sp] = btn
            if i < 7:
                species_btn_row1.addWidget(btn)
            elif i < 14:
                species_btn_row2.addWidget(btn)
            else:
                species_btn_row3.addWidget(btn)

        # "Other (+)" button for adding new species
        other_btn = QPushButton("Other (+)")
        other_btn.setStyleSheet("padding: 6px 12px; background: #444;")
        other_btn.setToolTip("Add a new species")
        species_btn_row2.addWidget(other_btn)

        # "Empty" button for no subject
        empty_btn = QPushButton("Empty")
        empty_btn.setStyleSheet("padding: 6px 12px; background: #333;")
        species_buttons["Empty"] = empty_btn
        species_btn_row2.addWidget(empty_btn)

        species_btn_row1.addStretch()
        species_btn_row2.addStretch()
        species_btn_row3.addStretch()
        species_layout.addLayout(species_btn_row1)
        species_layout.addLayout(species_btn_row2)
        species_layout.addLayout(species_btn_row3)
        layout.addWidget(species_container)

        # Action buttons
        btn_row = QHBoxLayout()
        draw_box_btn = QPushButton("Draw Box (B)")
        draw_box_btn.setCheckable(True)
        draw_box_btn.setStyleSheet("padding: 6px 12px;")
        draw_box_btn.setToolTip("Toggle draw mode to add new boxes")
        delete_selected_btn = QPushButton("Delete Selected (X)")
        delete_selected_btn.setStyleSheet("background: #633; padding: 6px 12px;")
        delete_selected_btn.setToolTip("Delete the currently selected box")
        delete_selected_btn.setEnabled(False)
        accept_boxes_btn = QPushButton("Accept Boxes (A)")
        accept_boxes_btn.setStyleSheet("background: #264; padding: 6px 12px;")
        accept_boxes_btn.setToolTip("Keep boxes as-is and move to next photo")
        delete_boxes_btn = QPushButton("Delete All (D)")
        delete_boxes_btn.setStyleSheet("background: #622; padding: 6px 12px;")
        delete_boxes_btn.setToolTip("Remove all boxes from this photo")
        skip_btn = QPushButton("Skip (S)")
        multi_checkbox = QCheckBox("Multi-species")
        multi_checkbox.setToolTip("Check to add multiple species to the same photo")
        next_btn = QPushButton("Next (N)")
        close_btn = QPushButton("Close")
        btn_row.addWidget(draw_box_btn)
        btn_row.addWidget(delete_selected_btn)
        btn_row.addWidget(accept_boxes_btn)
        btn_row.addWidget(delete_boxes_btn)
        btn_row.addWidget(skip_btn)
        btn_row.addWidget(multi_checkbox)
        btn_row.addWidget(next_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        current_pixmap = [None]
        reviewed_ids = set()

        def _update_preview():
            item = list_widget.currentItem()
            if not item:
                scene.clear()
                info_label.setText("No photo selected")
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            path = data.get("path")
            photo_id = data.get("id")
            info_label.setText(f"Photo: {data.get('name', '')}")

            # Mark reviewed items green
            if photo_id in reviewed_ids:
                item.setBackground(QColor(40, 80, 40))

            if path and os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scene.clear()
                    scene.addPixmap(pixmap)
                    scene.setSceneRect(pixmap.rect().toRectF())
                    current_pixmap[0] = pixmap

                    # Draw boxes (skip boxes entirely in bottom 5% - timestamp area)
                    box_items.clear()
                    selected_box[0] = None
                    delete_selected_btn.setEnabled(False)
                    if photo_id:
                        boxes = self.db.get_boxes(photo_id)
                        w = pixmap.width()
                        h = pixmap.height()
                        for b in boxes:
                            # Skip boxes entirely in bottom 5% (timestamp area)
                            if b["y1"] >= 0.95:
                                continue
                            rect = QRectF(b["x1"] * w, b["y1"] * h, (b["x2"] - b["x1"]) * w, (b["y2"] - b["y1"]) * h)
                            lbl = b.get("label", "")
                            if str(lbl) == "ai_deer_head" or str(lbl) == "deer_head":
                                pen = QPen(Qt.GlobalColor.magenta)
                            elif str(lbl).startswith("ai_"):
                                pen = QPen(Qt.GlobalColor.yellow)
                            else:
                                pen = QPen(Qt.GlobalColor.green)
                            pen.setWidth(4)
                            rect_item = scene.addRect(rect, pen)
                            rect_item.setData(0, b)  # Store box data in item
                            box_items.append((rect_item, b))

                    _zoom_fit()
                else:
                    scene.clear()
                    current_pixmap[0] = None
            else:
                scene.clear()
                current_pixmap[0] = None

        min_scale = [0.1]
        max_scale = 5.0

        def _get_current_scale():
            return view.transform().m11()

        def _zoom_in():
            current = _get_current_scale()
            if current < max_scale:
                view.scale(1.25, 1.25)

        def _zoom_out():
            current = _get_current_scale()
            if current > min_scale[0] * 1.1:
                view.scale(0.8, 0.8)

        def _zoom_fit():
            if current_pixmap[0] and scene.sceneRect().width() > 0:
                view.resetTransform()
                view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                min_scale[0] = _get_current_scale() * 0.95

        def _zoom_100():
            view.resetTransform()

        zoom_in_btn.clicked.connect(_zoom_in)
        zoom_out_btn.clicked.connect(_zoom_out)
        zoom_fit_btn.clicked.connect(_zoom_fit)
        zoom_100_btn.clicked.connect(_zoom_100)

        def _select_box(box_item):
            """Select a box and highlight it."""
            # Deselect previous
            if selected_box[0]:
                old_item, old_data = selected_box[0]
                lbl = old_data.get("label", "")
                if str(lbl) == "ai_deer_head" or str(lbl) == "deer_head":
                    pen = QPen(Qt.GlobalColor.magenta)
                elif str(lbl).startswith("ai_"):
                    pen = QPen(Qt.GlobalColor.yellow)
                else:
                    pen = QPen(Qt.GlobalColor.green)
                pen.setWidth(4)
                old_item.setPen(pen)
            # Select new
            if box_item:
                rect_item, box_data = box_item
                pen = QPen(Qt.GlobalColor.cyan)
                pen.setWidth(6)
                rect_item.setPen(pen)
                selected_box[0] = box_item
                delete_selected_btn.setEnabled(True)
            else:
                selected_box[0] = None
                delete_selected_btn.setEnabled(False)

        def _delete_selected():
            """Delete the currently selected box."""
            if not selected_box[0]:
                return
            rect_item, box_data = selected_box[0]
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            photo_id = data.get("id")
            if photo_id:
                # Get current boxes and remove the selected one
                current_boxes = self.db.get_boxes(photo_id)
                # Find and remove matching box
                new_boxes = []
                for b in current_boxes:
                    if (abs(b["x1"] - box_data["x1"]) < 0.001 and
                        abs(b["y1"] - box_data["y1"]) < 0.001 and
                        abs(b["x2"] - box_data["x2"]) < 0.001 and
                        abs(b["y2"] - box_data["y2"]) < 0.001):
                        continue  # Skip this box (delete it)
                    new_boxes.append(b)
                self.db.set_boxes(photo_id, new_boxes)
                # Remove from scene and list
                scene.removeItem(rect_item)
                box_items.remove(selected_box[0])
                selected_box[0] = None
                delete_selected_btn.setEnabled(False)

        def _toggle_draw_mode(checked):
            """Toggle draw mode for creating new boxes."""
            draw_mode[0] = checked
            if checked:
                view.setCursor(Qt.CursorShape.CrossCursor)
                draw_box_btn.setStyleSheet("background: #363; padding: 6px 12px;")
            else:
                view.setCursor(Qt.CursorShape.ArrowCursor)
                draw_box_btn.setStyleSheet("padding: 6px 12px;")

        def _on_mouse_press(event):
            """Handle mouse press for box selection or drawing."""
            if event.button() != Qt.MouseButton.LeftButton:
                return
            scene_pos = view.mapToScene(event.pos())
            # Always check if clicking on a box first (for selection)
            clicked_box = None
            for rect_item, box_data in box_items:
                if rect_item.contains(scene_pos):
                    clicked_box = (rect_item, box_data)
                    break
            if clicked_box:
                # Select the clicked box
                _select_box(clicked_box)
            elif draw_mode[0]:
                # Start drawing a new box (only in empty space)
                draw_start[0] = scene_pos
                temp_rect[0] = None
            else:
                # Clicked empty space, deselect
                _select_box(None)

        def _on_mouse_move(event):
            """Handle mouse move for drawing."""
            if draw_mode[0] and draw_start[0]:
                scene_pos = view.mapToScene(event.pos())
                # Remove old temp rect
                if temp_rect[0]:
                    scene.removeItem(temp_rect[0])
                # Draw new temp rect
                x1 = min(draw_start[0].x(), scene_pos.x())
                y1 = min(draw_start[0].y(), scene_pos.y())
                x2 = max(draw_start[0].x(), scene_pos.x())
                y2 = max(draw_start[0].y(), scene_pos.y())
                pen = QPen(Qt.GlobalColor.white)
                pen.setWidth(2)
                pen.setStyle(Qt.PenStyle.DashLine)
                temp_rect[0] = scene.addRect(QRectF(x1, y1, x2-x1, y2-y1), pen)

        def _on_mouse_release(event):
            """Handle mouse release for completing box drawing."""
            if draw_mode[0] and draw_start[0] and event.button() == Qt.MouseButton.LeftButton:
                scene_pos = view.mapToScene(event.pos())
                # Remove temp rect
                if temp_rect[0]:
                    scene.removeItem(temp_rect[0])
                    temp_rect[0] = None
                # Create the box if it's big enough
                if current_pixmap[0]:
                    w = current_pixmap[0].width()
                    h = current_pixmap[0].height()
                    x1 = min(draw_start[0].x(), scene_pos.x()) / w
                    y1 = min(draw_start[0].y(), scene_pos.y()) / h
                    x2 = max(draw_start[0].x(), scene_pos.x()) / w
                    y2 = max(draw_start[0].y(), scene_pos.y()) / h
                    # Clamp to image bounds
                    x1 = max(0, min(1, x1))
                    y1 = max(0, min(1, y1))
                    x2 = max(0, min(1, x2))
                    y2 = max(0, min(1, y2))
                    # Check minimum size (at least 2% of image)
                    if (x2 - x1) > 0.02 and (y2 - y1) > 0.02:
                        item = list_widget.currentItem()
                        if item:
                            data = item.data(Qt.ItemDataRole.UserRole)
                            photo_id = data.get("id")
                            if photo_id:
                                # Add new box to database
                                new_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": "manual"}
                                current_boxes = self.db.get_boxes(photo_id)
                                current_boxes.append(new_box)
                                self.db.set_boxes(photo_id, current_boxes)
                                # Add to scene
                                rect = QRectF(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h)
                                pen = QPen(Qt.GlobalColor.green)
                                pen.setWidth(4)
                                rect_item = scene.addRect(rect, pen)
                                rect_item.setData(0, new_box)
                                box_items.append((rect_item, new_box))
                draw_start[0] = None

        # Install event filter for mouse events
        view.viewport().installEventFilter(dlg)
        original_event_filter = dlg.eventFilter if hasattr(dlg, 'eventFilter') else None
        def custom_event_filter(obj, event):
            if obj == view.viewport():
                if event.type() == event.Type.MouseButtonPress:
                    _on_mouse_press(event)
                elif event.type() == event.Type.MouseMove:
                    _on_mouse_move(event)
                elif event.type() == event.Type.MouseButtonRelease:
                    _on_mouse_release(event)
            if original_event_filter:
                return original_event_filter(obj, event)
            return False
        dlg.eventFilter = custom_event_filter

        def _advance():
            current_row = list_widget.currentRow()
            if current_row < list_widget.count() - 1:
                list_widget.setCurrentRow(current_row + 1)

        def _label_species(species_name):
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            photo_id = data.get("id")
            if photo_id:
                self.db.add_tag(photo_id, species_name)
                # If labeling as Empty, clear all AI boxes (they were false positives)
                if species_name == "Empty":
                    self.db.set_boxes(photo_id, [])
                    # Clear box items from display
                    box_items.clear()
                    selected_box[0] = None
                    delete_selected_btn.setEnabled(False)
                    _update_preview()
                reviewed_ids.add(photo_id)
                item.setBackground(QColor(40, 80, 40))
                # Update item text to show labeled species
                name = data.get("name", "")[:20]
                current_text = item.text()
                # Check if already has a species label appended
                if " → " in current_text:
                    base_name = current_text.split(" → ")[0]
                    item.setText(f"{base_name} → {species_name}")
                else:
                    item.setText(f"{name} → {species_name}")
                remaining = list_widget.count() - len(reviewed_ids)
                dlg.setWindowTitle(f"Label Photos with Boxes ({remaining} remaining)")
                if not multi_checkbox.isChecked():
                    _advance()

        def _accept_boxes():
            """Accept current boxes and advance to next photo."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            photo_id = data.get("id")
            if photo_id:
                reviewed_ids.add(photo_id)
                item.setBackground(QColor(40, 80, 40))
                remaining = list_widget.count() - len(reviewed_ids)
                dlg.setWindowTitle(f"Label Photos with Boxes ({remaining} remaining)")
            _advance()

        def _delete_boxes():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            photo_id = data.get("id")
            if photo_id:
                # Delete all boxes for this photo
                self.db.set_boxes(photo_id, [])
                # Auto-label as Empty since no subjects detected
                self.db.add_tag(photo_id, "Empty")
                reviewed_ids.add(photo_id)
                item.setBackground(QColor(40, 80, 40))
                # Update item text to show Empty label
                name = data.get("name", "")[:20]
                item.setText(f"{name} → Empty")
                remaining = list_widget.count() - len(reviewed_ids)
                dlg.setWindowTitle(f"Label Photos with Boxes ({remaining} remaining)")
                # Redraw preview without boxes
                _update_preview()
            _advance()

        def _apply_other():
            """Apply 'Other' species tag (custom species disabled)."""
            _label_species("Other")

        for sp, btn in species_buttons.items():
            btn.clicked.connect(lambda checked, s=sp: _label_species(s))

        other_btn.clicked.connect(_apply_other)
        draw_box_btn.clicked.connect(_toggle_draw_mode)
        delete_selected_btn.clicked.connect(_delete_selected)
        accept_boxes_btn.clicked.connect(_accept_boxes)
        delete_boxes_btn.clicked.connect(_delete_boxes)
        skip_btn.clicked.connect(_advance)
        next_btn.clicked.connect(_advance)
        close_btn.clicked.connect(dlg.close)

        list_widget.currentRowChanged.connect(lambda _: _update_preview())
        _update_preview()

        # Keyboard shortcuts
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("B"), dlg, lambda: draw_box_btn.click())
        QShortcut(QKeySequence("X"), dlg, _delete_selected)
        QShortcut(QKeySequence("Delete"), dlg, _delete_selected)
        QShortcut(QKeySequence("A"), dlg, _accept_boxes)
        QShortcut(QKeySequence("D"), dlg, _delete_boxes)
        QShortcut(QKeySequence("S"), dlg, _advance)
        QShortcut(QKeySequence("N"), dlg, _advance)

        dlg.exec()

        # Refresh main view
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_species_dropdown()
        self._populate_photo_list()

    def _gather_unlabeled_with_boxes(self) -> list:
        """Return photos that have boxes but no species label and no suggestion."""
        # Combine all known species from SPECIES_OPTIONS, VALID_SPECIES, and custom species
        species_set = self._species_set() | VALID_SPECIES
        pending = []
        try:
            cursor = self.db.conn.cursor()
            # Get photos that have any boxes (not just AI boxes)
            cursor.execute("""
                SELECT DISTINCT p.id, p.original_name, p.file_path, p.suggested_tag
                FROM photos p
                INNER JOIN annotation_boxes ab ON p.id = ab.photo_id
                ORDER BY p.date_taken DESC
            """)
            for row in cursor.fetchall():
                pid, name, path, suggested = row
                # Skip if has suggestion
                if suggested and suggested.strip():
                    continue
                # Check if already has a species tag
                tags = self.db.get_tags(pid)
                has_species = any(t in species_set for t in tags)
                if has_species:
                    continue
                # Check if photo has any boxes outside timestamp area (bottom 5%)
                boxes = self.db.get_boxes(pid)
                has_visible_boxes = any(b.get("y1", 0) < 0.95 for b in boxes)
                if not has_visible_boxes:
                    continue  # Skip photos where all boxes are in timestamp area
                pending.append({
                    "id": pid,
                    "name": name or "",
                    "path": path
                })
        except Exception as e:
            logger.warning(f"Failed to gather unlabeled photos with boxes: {e}")
        return pending

    def review_sex_suggestions(self):
        """Review and approve/reject pending buck/doe suggestions with zoomable photo preview."""
        from PyQt6.QtWidgets import QGraphicsView
        # Gather pending sex suggestions
        pending = self._gather_pending_sex_suggestions()
        if not pending:
            QMessageBox.information(self, "Buck/Doe Suggestions", "No pending buck/doe suggestions to review.")
            return
        # Create review dialog with zoomable photo preview
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Review Buck/Doe Suggestions ({len(pending)})")
        dlg.resize(1100, 750)
        layout = QVBoxLayout(dlg)

        # Main content: list on left, photo on right
        content_layout = QHBoxLayout()

        # Left side: list of suggestions with filters
        left_panel = QVBoxLayout()

        # Filter controls
        filter_label = QLabel("Filters:")
        filter_label.setStyleSheet("font-weight: bold;")
        left_panel.addWidget(filter_label)

        # Suggestion filter (All/Buck/Doe)
        filter_row1 = QHBoxLayout()
        filter_row1.addWidget(QLabel("Type:"))
        sex_filter = QComboBox()
        sex_filter.addItems(["All", "Buck", "Doe"])
        sex_filter.setMaximumWidth(100)
        filter_row1.addWidget(sex_filter)
        filter_row1.addStretch()
        left_panel.addLayout(filter_row1)

        # Confidence filter
        filter_row2 = QHBoxLayout()
        filter_row2.addWidget(QLabel("Conf:"))
        conf_filter = QComboBox()
        conf_filter.addItems(["All", "≥90%", "≥80%", "≥70%", "<70%"])
        conf_filter.setMaximumWidth(100)
        filter_row2.addWidget(conf_filter)
        filter_row2.addStretch()
        left_panel.addLayout(filter_row2)

        # Date filter
        filter_row3 = QHBoxLayout()
        filter_row3.addWidget(QLabel("Date:"))
        date_filter = QComboBox()
        # Get unique dates from pending items
        unique_dates = sorted(set(item.get("date", "")[:10] for item in pending if item.get("date")), reverse=True)
        date_filter.addItem("All")
        for d in unique_dates[:30]:  # Limit to 30 most recent dates
            if d:
                date_filter.addItem(d)
        date_filter.setMaximumWidth(120)
        filter_row3.addWidget(date_filter)
        filter_row3.addStretch()
        left_panel.addLayout(filter_row3)

        # Collection filter
        filter_row4 = QHBoxLayout()
        filter_row4.addWidget(QLabel("Collection:"))
        collection_filter = QComboBox()
        # Get unique collections from pending items
        unique_collections = sorted(set(item.get("collection", "") for item in pending if item.get("collection")))
        collection_filter.addItem("All")
        for c in unique_collections:
            if c:
                collection_filter.addItem(c)
        collection_filter.setMaximumWidth(180)
        filter_row4.addWidget(collection_filter)
        filter_row4.addStretch()
        left_panel.addLayout(filter_row4)

        left_panel.addWidget(QLabel("Pending suggestions:"))
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setMaximumWidth(280)

        all_pending = pending  # Store original list for filtering

        def _populate_list():
            """Populate list based on current filters, sorted alphabetically by name."""
            list_widget.clear()
            sex_val = sex_filter.currentText()
            conf_val = conf_filter.currentText()
            date_val = date_filter.currentText()
            collection_val = collection_filter.currentText()

            # Collect filtered items first
            filtered_items = []
            for item in all_pending:
                # Apply sex filter
                if sex_val != "All" and item["sex"].lower() != sex_val.lower():
                    continue
                # Apply confidence filter
                conf_pct = item["conf"] * 100 if item["conf"] <= 1 else item["conf"]
                if conf_val == "≥90%" and conf_pct < 90:
                    continue
                elif conf_val == "≥80%" and conf_pct < 80:
                    continue
                elif conf_val == "≥70%" and conf_pct < 70:
                    continue
                elif conf_val == "<70%" and conf_pct >= 70:
                    continue
                # Apply date filter
                if date_val != "All":
                    item_date = (item.get("date") or "")[:10]
                    if item_date != date_val:
                        continue
                # Apply collection filter
                if collection_val != "All":
                    item_collection = item.get("collection") or ""
                    if item_collection != collection_val:
                        continue
                filtered_items.append((item, conf_pct))

            # Sort by sex (Buck, Doe, Unknown) then by confidence descending
            sex_order = {"buck": 0, "doe": 1, "unknown": 2}
            filtered_items.sort(key=lambda x: (
                sex_order.get(x[0].get("sex", "").lower(), 3),  # Sex order
                -x[1]  # Confidence descending (negative for reverse)
            ))

            # Add to list widget
            for item, conf_pct in filtered_items:
                pct = int(conf_pct)
                text = f"{item['sex'].title()} ({pct}%) - {item['name'][:22]}"
                li = QListWidgetItem(text)
                li.setData(Qt.ItemDataRole.UserRole, item)
                list_widget.addItem(li)

            # Update title with filtered count
            dlg.setWindowTitle(f"Review Buck/Doe Suggestions ({list_widget.count()} shown, {len(all_pending)} total)")

        # Connect filters
        sex_filter.currentIndexChanged.connect(_populate_list)
        conf_filter.currentIndexChanged.connect(_populate_list)
        date_filter.currentIndexChanged.connect(_populate_list)
        collection_filter.currentIndexChanged.connect(_populate_list)

        # Initial population
        _populate_list()

        left_panel.addWidget(list_widget)
        content_layout.addLayout(left_panel)

        # Right side: zoomable photo preview using QGraphicsView
        right_panel = QVBoxLayout()
        suggestion_label = QLabel("Suggestion: —")
        suggestion_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        right_panel.addWidget(suggestion_label)

        # Zoom controls
        zoom_row = QHBoxLayout()
        zoom_label = QLabel("Zoom:")
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(30)
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedWidth(30)
        zoom_fit_btn = QPushButton("Fit")
        zoom_100_btn = QPushButton("100%")
        zoom_row.addWidget(zoom_label)
        zoom_row.addWidget(zoom_out_btn)
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(zoom_fit_btn)
        zoom_row.addWidget(zoom_100_btn)
        zoom_row.addStretch()
        right_panel.addLayout(zoom_row)

        # Graphics view for zoomable image
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.setMinimumSize(700, 550)
        view.setStyleSheet("background: #222; border: 1px solid #444;")
        view.setDragMode(QGraphicsView.DragMode.NoDrag)  # Start with no drag; enabled when zoomed in
        view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        right_panel.addWidget(view, 1)
        content_layout.addLayout(right_panel, 1)

        layout.addLayout(content_layout, 1)

        # Buttons
        btn_row = QHBoxLayout()
        accept_buck_btn = QPushButton("Buck (B)")
        accept_buck_btn.setStyleSheet("background: #264; padding: 8px 16px;")
        accept_doe_btn = QPushButton("Doe (D)")
        accept_doe_btn.setStyleSheet("background: #462; padding: 8px 16px;")
        unknown_btn = QPushButton("Unknown (U)")
        unknown_btn.setStyleSheet("background: #444; padding: 8px 16px;")
        reject_btn = QPushButton("Reject (R)")
        reject_btn.setStyleSheet("padding: 8px 16px;")
        skip_btn = QPushButton("Skip (S)")
        props_btn = QPushButton("Properties (P)")
        props_btn.setStyleSheet("padding: 8px 16px;")
        close_btn = QPushButton("Close")
        btn_row.addWidget(accept_buck_btn)
        btn_row.addWidget(accept_doe_btn)
        btn_row.addWidget(unknown_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addWidget(skip_btn)
        btn_row.addWidget(props_btn)
        btn_row.addStretch()

        # Species selector for non-deer specimens
        btn_row.addWidget(QLabel("Not Deer:"))
        species_combo = QComboBox()
        species_combo.addItems(["", "Turkey", "Coyote", "Raccoon", "Squirrel", "Bobcat",
                               "Opossum", "Rabbit", "Fox", "Person", "Vehicle", "Empty", "Other"])
        species_combo.setMaximumWidth(100)
        species_combo.setToolTip("Select species if this is not a deer")
        btn_row.addWidget(species_combo)

        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Image state for memory management
        current_pixmap = [None]  # Current displayed pixmap
        current_path = [None]    # Path to current image (for full-res loading)
        current_orig_size = [0, 0]  # Original image dimensions
        is_full_res = [False]    # Whether full-res is currently loaded
        current_boxes_data = [None]  # Cached box data for redrawing
        current_target_box = [None]  # Target box for highlighting

        def _cleanup_memory():
            """Explicitly clean up pixmap memory."""
            if current_pixmap[0] is not None:
                current_pixmap[0] = None
            scene.clear()
            import gc
            gc.collect()

        def _load_image(path, full_res=False):
            """Load image, optionally at full resolution."""
            if not path or not os.path.exists(path):
                return None, 0, 0

            pixmap = QPixmap(path)
            if pixmap.isNull():
                return None, 0, 0

            orig_w, orig_h = pixmap.width(), pixmap.height()

            if not full_res:
                # Scale to max 1200px for fast preview
                max_preview_size = 1200
                if orig_w > max_preview_size or orig_h > max_preview_size:
                    scaled = pixmap.scaled(
                        max_preview_size, max_preview_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation
                    )
                    del pixmap
                    return scaled, orig_w, orig_h

            return pixmap, orig_w, orig_h

        def _draw_boxes_on_scene(pixmap, orig_w, orig_h, boxes, target_box):
            """Draw detection boxes on the scene."""
            if not boxes or not pixmap:
                return
            scale_x = pixmap.width() / orig_w
            scale_y = pixmap.height() / orig_h
            target_id = target_box.get("id") if target_box else None

            for box in boxes:
                lbl = box.get("label", "")
                if "head" in str(lbl).lower():
                    continue
                x1 = box.get("x1", 0) * orig_w * scale_x
                y1 = box.get("y1", 0) * orig_h * scale_y
                x2 = box.get("x2", 0) * orig_w * scale_x
                y2 = box.get("y2", 0) * orig_h * scale_y
                rect = QRectF(x1, y1, x2 - x1, y2 - y1)

                # Match by ID first, fallback to coordinates (IDs change when boxes are updated)
                is_target = False
                if target_box:
                    if target_id and box.get("id") == target_id:
                        is_target = True
                    elif (abs(box.get("x1", 0) - target_box.get("x1", -1)) < 0.01 and
                          abs(box.get("y1", 0) - target_box.get("y1", -1)) < 0.01 and
                          abs(box.get("x2", 0) - target_box.get("x2", -1)) < 0.01 and
                          abs(box.get("y2", 0) - target_box.get("y2", -1)) < 0.01):
                        is_target = True
                if is_target:
                    pen = QPen(Qt.GlobalColor.yellow)
                    pen.setWidth(4)
                else:
                    pen = QPen(Qt.GlobalColor.gray)
                    pen.setWidth(2)
                scene.addRect(rect, pen)

        def _update_preview():
            """Load preview image and draw boxes."""
            # Clean up previous image memory
            _cleanup_memory()

            item = list_widget.currentItem()
            if not item:
                suggestion_label.setText("Suggestion: —")
                return

            data = item.data(Qt.ItemDataRole.UserRole)
            path = data.get("path")
            pid = data.get("photo_id")
            target_box = data.get("box")
            sex = data.get("sex", "").title()
            conf = data.get("conf", 0)
            pct = int(conf * 100) if conf <= 1 else int(conf)
            suggestion_label.setText(f"AI Suggestion: {sex} ({pct}% confidence)")

            # Store state for potential full-res loading later
            current_path[0] = path
            current_target_box[0] = target_box
            is_full_res[0] = False

            # Load scaled preview
            pixmap, orig_w, orig_h = _load_image(path, full_res=False)
            if pixmap:
                current_pixmap[0] = pixmap
                current_orig_size[0] = orig_w
                current_orig_size[1] = orig_h

                scene.addPixmap(pixmap)
                scene.setSceneRect(pixmap.rect().toRectF())

                # Get and cache boxes
                if pid:
                    boxes = self.db.get_boxes(pid)
                    current_boxes_data[0] = boxes
                    _draw_boxes_on_scene(pixmap, orig_w, orig_h, boxes, target_box)

                _zoom_fit()

        def _load_full_res():
            """Load full resolution image when user zooms in."""
            if is_full_res[0] or not current_path[0]:
                return

            path = current_path[0]
            pixmap, orig_w, orig_h = _load_image(path, full_res=True)
            if pixmap:
                # Clear and reload at full res
                scene.clear()
                current_pixmap[0] = pixmap
                is_full_res[0] = True

                scene.addPixmap(pixmap)
                scene.setSceneRect(pixmap.rect().toRectF())

                # Redraw boxes
                if current_boxes_data[0]:
                    _draw_boxes_on_scene(pixmap, orig_w, orig_h,
                                        current_boxes_data[0], current_target_box[0])

        min_scale = [0.1]  # Will be updated to fit scale
        max_scale = 5.0

        def _get_current_scale():
            return view.transform().m11()

        def _zoom_in():
            current = _get_current_scale()
            if current < max_scale:
                view.scale(1.25, 1.25)
                # Load full-res when zooming past 1.5x fit scale for detail
                if not is_full_res[0] and current > min_scale[0] * 1.5:
                    _load_full_res()

        def _zoom_out():
            current = _get_current_scale()
            # Don't zoom out past the fit scale
            if current > min_scale[0] * 1.1:
                view.scale(0.8, 0.8)

        def _zoom_fit():
            if current_pixmap[0] and scene.sceneRect().width() > 0:
                view.resetTransform()
                # Use fitInView for proper scaling with aspect ratio preserved
                view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                # Store fit scale as minimum (with small padding)
                min_scale[0] = _get_current_scale() * 0.95
                view.scale(0.95, 0.95)

        def _zoom_100():
            """Zoom to 100% and load full-res for detail inspection."""
            view.resetTransform()
            if not is_full_res[0]:
                _load_full_res()

        reviewed_rows = set()  # Track reviewed list rows (per-box)
        reviewed_data = {}  # Track what action was taken: {pid: action}
        navigate_to_photo = None  # Store photo to navigate to after dialog closes

        def _get_box_key(data):
            """Get unique key for a box."""
            box = data.get("box", {})
            return (data.get("photo_id"), box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2"))

        def _mark_reviewed(item, action_text: str):
            """Mark item as reviewed with green highlight and update text."""
            row = list_widget.row(item)
            reviewed_rows.add(row)
            data = item.data(Qt.ItemDataRole.UserRole)
            original_name = data.get("name", "")[:22]
            item.setText(f"✓ {action_text} - {original_name}")
            item.setBackground(QColor(144, 238, 144))  # Light green
            # Update window title with remaining count
            remaining = list_widget.count() - len(reviewed_rows)
            dlg.setWindowTitle(f"Review Buck/Doe Suggestions ({remaining} remaining)")

        def _next_unreviewed():
            """Move to next unreviewed item."""
            current = list_widget.currentRow()
            # Look forward first
            for i in range(current + 1, list_widget.count()):
                if i not in reviewed_rows:
                    list_widget.setCurrentRow(i)
                    _update_preview()
                    return
            # Then look from beginning
            for i in range(0, current):
                if i not in reviewed_rows:
                    list_widget.setCurrentRow(i)
                    _update_preview()
                    return
            # All reviewed - stay on current
            _update_preview()

        def _accept_as(sex_tag: str):
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data.get("photo_id")
            box = data.get("box")
            ai_suggested = data.get("sex", "")
            # Only allow sex labels on deer boxes
            box_species = (box.get("species") or "").lower() if box else ""
            if box_species and box_species != "deer":
                _mark_reviewed(item, "SKIPPED (not deer)")
                _next_unreviewed()
                return
            # Log if user corrected the AI (accepted different label)
            if ai_suggested and ai_suggested != sex_tag:
                self.db.log_ai_rejection(
                    photo_id=pid,
                    suggestion_type="sex",
                    ai_suggested=ai_suggested,
                    correct_label=sex_tag,
                    model_version=get_model_version("buckdoe")
                )
            # Update the box's sex field and set species to Deer
            if box and pid:
                box_id = box.get("id")
                if not box_id:
                    all_boxes = self.db.get_boxes(pid)
                    for b in all_boxes:
                        if (abs(b.get("x1", 0) - box.get("x1", -1)) < 0.01 and
                            abs(b.get("y1", 0) - box.get("y1", -1)) < 0.01 and
                            abs(b.get("x2", 0) - box.get("x2", -1)) < 0.01 and
                            abs(b.get("y2", 0) - box.get("y2", -1)) < 0.01):
                            box_id = b.get("id")
                            break
                if box_id:
                    self.db.update_box_fields(
                        box_id,
                        sex=sex_tag,
                        sex_conf=None,
                        species="Deer",
                    )
                # Add Deer tag if not present
                tags = set(self.db.get_tags(pid))
                if "Deer" not in tags:
                    self.db.add_tag(pid, "Deer")
                # Also add Buck/Doe tag
                if sex_tag in ("Buck", "Doe") and sex_tag not in tags:
                    self.db.add_tag(pid, sex_tag)
                # Clear species suggestion only if all boxes have been reviewed
                try:
                    remaining = self.db.get_boxes(pid)
                    all_reviewed = True
                    for b in remaining:
                        if b.get("species", "").lower() == "deer" and (b.get("sex_conf") is not None):
                            all_reviewed = False
                            break
                    if all_reviewed:
                        self.db.set_suggested_tag(pid, None, None)
                except Exception:
                    pass
            # Track for cleanup
            reviewed_data[pid] = sex_tag
            _mark_reviewed(item, sex_tag)
            _next_unreviewed()

        def _reject():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data.get("photo_id")
            box = data.get("box")
            ai_suggested = data.get("sex", "")
            # Log rejection for future model training
            if ai_suggested and pid:
                self.db.log_ai_rejection(
                    photo_id=pid,
                    suggestion_type="sex",
                    ai_suggested=ai_suggested,
                    correct_label=None,
                    model_version=get_model_version("buckdoe")
                )
            # Clear the box's sex suggestion (mark as reviewed/rejected)
            if box and pid:
                box_id = box.get("id")
                if not box_id:
                    all_boxes = self.db.get_boxes(pid)
                    for b in all_boxes:
                        if (abs(b.get("x1", 0) - box.get("x1", -1)) < 0.01 and
                            abs(b.get("y1", 0) - box.get("y1", -1)) < 0.01 and
                            abs(b.get("x2", 0) - box.get("x2", -1)) < 0.01 and
                            abs(b.get("y2", 0) - box.get("y2", -1)) < 0.01):
                            box_id = b.get("id")
                            break
                if box_id:
                    self.db.update_box_fields(
                        box_id,
                        sex=None,
                        sex_conf=None,
                    )
            reviewed_data[pid] = "REJECTED"
            _mark_reviewed(item, "REJECTED")
            _next_unreviewed()

        def _skip():
            row = list_widget.currentRow()
            if row < list_widget.count() - 1:
                list_widget.setCurrentRow(row + 1)
                _update_preview()

        def _unknown():
            """Mark sex as unknown - still a deer, just can't determine buck/doe."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data.get("photo_id")
            box = data.get("box")
            # Set sex to Unknown and species to Deer
            if box and pid:
                box_id = box.get("id")
                if not box_id:
                    all_boxes = self.db.get_boxes(pid)
                    for b in all_boxes:
                        if (abs(b.get("x1", 0) - box.get("x1", -1)) < 0.01 and
                            abs(b.get("y1", 0) - box.get("y1", -1)) < 0.01 and
                            abs(b.get("x2", 0) - box.get("x2", -1)) < 0.01 and
                            abs(b.get("y2", 0) - box.get("y2", -1)) < 0.01):
                            box_id = b.get("id")
                            break
                if box_id:
                    self.db.update_box_fields(
                        box_id,
                        sex="Unknown",
                        sex_conf=None,
                        species="Deer",
                    )
                # Add Deer tag if not present
                tags = set(self.db.get_tags(pid))
                if "Deer" not in tags:
                    self.db.add_tag(pid, "Deer")
                # Clear species suggestion only if all boxes have been reviewed
                try:
                    remaining = self.db.get_boxes(pid)
                    all_reviewed = True
                    for b in remaining:
                        if b.get("species", "").lower() == "deer" and (b.get("sex_conf") is not None):
                            all_reviewed = False
                            break
                    if all_reviewed:
                        self.db.set_suggested_tag(pid, None, None)
                except Exception:
                    pass
            reviewed_data[pid] = "UNKNOWN"
            _mark_reviewed(item, "Deer (Unknown)")
            _next_unreviewed()

        def _set_species(species: str):
            """Set species for non-deer specimens in the sex review queue."""
            if not species:
                return
            item = list_widget.currentItem()
            if not item:
                species_combo.setCurrentIndex(0)  # Reset dropdown
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data.get("photo_id")
            box = data.get("box")
            if box and pid:
                # Update the box's species and clear sex suggestion
                box_id = box.get("id")
                all_boxes = self.db.get_boxes(pid)
                for b in all_boxes:
                    # Match by ID first, fallback to coordinates (IDs change when boxes are updated)
                    matched = False
                    if box_id and b.get("id") == box_id:
                        matched = True
                    elif (abs(b.get("x1", 0) - box.get("x1", -1)) < 0.01 and
                          abs(b.get("y1", 0) - box.get("y1", -1)) < 0.01 and
                          abs(b.get("x2", 0) - box.get("x2", -1)) < 0.01 and
                          abs(b.get("y2", 0) - box.get("y2", -1)) < 0.01):
                        matched = True
                    if matched:
                        b["species"] = species
                        b["sex"] = None
                        b["sex_conf"] = None
                        break  # Only update the specific box
                self.db.set_boxes(pid, all_boxes)
                # Add species tag to photo
                self.db.add_tag(pid, species)
                # Clear species suggestion since we've confirmed
                self.db.set_suggested_tag(pid, None, None)
            reviewed_data[pid] = species
            _mark_reviewed(item, species)
            species_combo.setCurrentIndex(0)  # Reset dropdown
            _next_unreviewed()

        def _open_properties():
            """Navigate to photo in main window and close review dialog."""
            nonlocal navigate_to_photo
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            # Store target photo info for navigation after dialog closes
            navigate_to_photo = {
                "photo_id": data.get("photo_id"),
                "collection": data.get("collection", "")
            }
            # Close the dialog - navigation happens after dlg.exec() returns
            dlg.accept()

        def _cleanup_reviewed():
            """Clear suggestions from DB for all reviewed items."""
            for pid, action in reviewed_data.items():
                self.db.set_suggested_sex(pid, "", None)

        # Connect cleanup to dialog finished signal (handles X button, Close button, etc.)
        dlg.finished.connect(_cleanup_reviewed)

        # Connect signals
        list_widget.currentItemChanged.connect(_update_preview)
        zoom_in_btn.clicked.connect(_zoom_in)
        zoom_out_btn.clicked.connect(_zoom_out)
        zoom_fit_btn.clicked.connect(_zoom_fit)
        zoom_100_btn.clicked.connect(_zoom_100)
        accept_buck_btn.clicked.connect(lambda: _accept_as("Buck"))
        accept_doe_btn.clicked.connect(lambda: _accept_as("Doe"))
        unknown_btn.clicked.connect(_unknown)
        reject_btn.clicked.connect(_reject)
        skip_btn.clicked.connect(_skip)
        props_btn.clicked.connect(_open_properties)
        species_combo.currentTextChanged.connect(_set_species)
        close_btn.clicked.connect(dlg.accept)

        # Keyboard shortcuts
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("B"), dlg).activated.connect(lambda: _accept_as("Buck"))
        QShortcut(QKeySequence("D"), dlg).activated.connect(lambda: _accept_as("Doe"))
        QShortcut(QKeySequence("U"), dlg).activated.connect(_unknown)
        QShortcut(QKeySequence("R"), dlg).activated.connect(_reject)
        QShortcut(QKeySequence("S"), dlg).activated.connect(_skip)
        QShortcut(QKeySequence("P"), dlg).activated.connect(_open_properties)
        QShortcut(QKeySequence("+"), dlg).activated.connect(_zoom_in)
        QShortcut(QKeySequence("="), dlg).activated.connect(_zoom_in)
        QShortcut(QKeySequence("-"), dlg).activated.connect(_zoom_out)
        QShortcut(QKeySequence("F"), dlg).activated.connect(_zoom_fit)

        is_fit_mode = [True]  # Track fit vs 100% state for double-click toggle

        def _update_drag_mode():
            """Enable drag mode when zoomed in past fit."""
            if _get_current_scale() > min_scale[0] * 1.05:
                view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            else:
                view.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Mouse wheel: Ctrl+scroll to zoom, regular scroll to pan
        def _wheel_event(event):
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Ctrl+scroll = zoom
                current = _get_current_scale()
                if event.angleDelta().y() > 0:
                    if current < max_scale:
                        view.scale(1.15, 1.15)
                        is_fit_mode[0] = False
                        _update_drag_mode()
                else:
                    if current > min_scale[0] * 1.05:
                        view.scale(1/1.15, 1/1.15)
                        _update_drag_mode()
                event.accept()
            else:
                # Regular scroll = pan (default behavior)
                QGraphicsView.wheelEvent(view, event)
        view.wheelEvent = _wheel_event

        # Double-click to toggle fit/100%
        def _double_click(event):
            if is_fit_mode[0]:
                _zoom_100()
                is_fit_mode[0] = False
            else:
                _zoom_fit()
                is_fit_mode[0] = True
            _update_drag_mode()
        view.mouseDoubleClickEvent = _double_click

        # Enable gesture events for pinch-to-zoom on trackpad
        view.grabGesture(Qt.GestureType.PinchGesture)
        view.viewport().grabGesture(Qt.GestureType.PinchGesture)

        # Use event filter for reliable gesture handling
        from PyQt6.QtCore import QObject, QEvent
        class GestureFilter(QObject):
            GESTURE_TYPE = QEvent.Type.Gesture  # Class attribute for scope
            def eventFilter(self, obj, event):
                if event.type() == self.GESTURE_TYPE:
                    pinch = event.gesture(Qt.GestureType.PinchGesture)
                    if pinch and pinch.state() == Qt.GestureState.GestureUpdated:
                        current = _get_current_scale()
                        if pinch.scaleFactor() > 1.0:
                            if current < max_scale:
                                view.scale(1.15, 1.15)
                                is_fit_mode[0] = False
                                _update_drag_mode()
                        else:
                            if current > min_scale[0] * 1.05:
                                view.scale(1/1.15, 1/1.15)
                                _update_drag_mode()
                        event.accept()
                        return True
                return False

        gesture_filter = GestureFilter(view)
        view.installEventFilter(gesture_filter)
        view.viewport().installEventFilter(gesture_filter)

        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
            _update_preview()
            # Delay zoom fit until dialog is shown and properly sized
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(200, _zoom_fit)
        dlg.exec()

        # Handle navigation to specific photo if Properties was clicked
        if navigate_to_photo:
            target_pid = navigate_to_photo.get("photo_id")
            self._navigate_to_photo_by_id(target_pid)
        else:
            # Default: just refresh normally
            self._populate_photo_list()
            if self.photos and 0 <= self.index < len(self.photos):
                self.load_photo()

    def review_mislabeled_species(self):
        """Review boxes that may be mislabeled (AI disagrees with current label)."""
        queue_file = os.path.expanduser("~/.trailcam/species_mislabel_review.json")
        if not os.path.exists(queue_file):
            QMessageBox.information(self, "Review Queue", "No mislabel review queue found.")
            return

        with open(queue_file, 'r') as f:
            queue_data = json.load(f)

        items = queue_data.get('items', [])
        if not items:
            QMessageBox.information(self, "Review Queue", "No items in queue.")
            os.remove(queue_file)
            return

        # Create review dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Review Mislabeled Species ({len(items)} items)")
        dlg.resize(1100, 750)
        layout = QVBoxLayout(dlg)

        # Info label
        info_label = QLabel(f"These boxes are labeled as 'Deer' but AI suggests a different species.\n"
                           f"Accept to change the species, or Reject to keep as Deer.")
        info_label.setStyleSheet("padding: 5px; background: #333; border-radius: 4px;")
        layout.addWidget(info_label)

        # Main content
        content_layout = QHBoxLayout()

        # Left: list
        left_panel = QVBoxLayout()

        # Filter by suggested species
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        species_filter = QComboBox()
        unique_species = sorted(set(item['suggested_label'] for item in items))
        species_filter.addItem("All")
        for s in unique_species:
            species_filter.addItem(s)
        species_filter.setMaximumWidth(120)
        filter_row.addWidget(species_filter)
        filter_row.addStretch()
        left_panel.addLayout(filter_row)

        left_panel.addWidget(QLabel("Items to review:"))
        list_widget = QListWidget()
        list_widget.setMaximumWidth(300)

        all_items = items[:]

        def _populate_list():
            list_widget.clear()
            filter_val = species_filter.currentText()
            for item in all_items:
                if filter_val != "All" and item['suggested_label'] != filter_val:
                    continue
                conf_pct = int(item['confidence'] * 100)
                text = f"Deer → {item['suggested_label']} ({conf_pct}%)"
                li = QListWidgetItem(text)
                li.setData(Qt.ItemDataRole.UserRole, item)
                list_widget.addItem(li)
            dlg.setWindowTitle(f"Review Mislabeled Species ({list_widget.count()} shown)")

        species_filter.currentIndexChanged.connect(_populate_list)
        _populate_list()

        left_panel.addWidget(list_widget)
        content_layout.addLayout(left_panel)

        # Right: photo preview
        right_panel = QVBoxLayout()
        suggestion_label = QLabel("Select an item to preview")
        suggestion_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        right_panel.addWidget(suggestion_label)

        # Graphics view for image
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.setMinimumSize(700, 550)
        view.setStyleSheet("background: #222; border: 1px solid #444;")
        right_panel.addWidget(view, 1)
        content_layout.addLayout(right_panel, 1)

        layout.addLayout(content_layout, 1)

        # Buttons
        btn_row = QHBoxLayout()
        accept_btn = QPushButton("Accept Suggestion (A)")
        accept_btn.setStyleSheet("background: #264; padding: 8px 16px;")
        reject_btn = QPushButton("Keep as Deer (K)")
        reject_btn.setStyleSheet("background: #642; padding: 8px 16px;")
        skip_btn = QPushButton("Skip (S)")
        props_btn = QPushButton("Go to Photo (G)")
        close_btn = QPushButton("Close")
        btn_row.addWidget(accept_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addWidget(skip_btn)
        btn_row.addWidget(props_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        def _update_preview():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            suggestion_label.setText(f"Current: Deer → Suggested: {data['suggested_label']} ({int(data['confidence']*100)}%)")

            file_path = data.get('file_path', '')
            if os.path.exists(file_path):
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scene.clear()
                    scene.addPixmap(pixmap)
                    view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        def _accept():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            # Update the box species in database
            self.db.conn.execute(
                "UPDATE annotation_boxes SET species = ? WHERE id = ?",
                (data['suggested_label'], data['box_id'])
            )
            self.db.conn.commit()
            # Remove from list
            all_items.remove(data)
            _populate_list()
            _save_queue()
            # Select next
            if list_widget.count() > 0:
                list_widget.setCurrentRow(0)
                _update_preview()

        def _reject():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            # Just remove from review queue (keep as Deer)
            all_items.remove(data)
            _populate_list()
            _save_queue()
            if list_widget.count() > 0:
                list_widget.setCurrentRow(0)
                _update_preview()

        def _skip():
            row = list_widget.currentRow()
            if row < list_widget.count() - 1:
                list_widget.setCurrentRow(row + 1)
            _update_preview()

        def _go_to_photo():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            photo_id = data.get('photo_id')
            if photo_id:
                dlg.close()
                self._navigate_to_photo(photo_id)

        def _save_queue():
            queue_data['items'] = all_items
            queue_data['total'] = len(all_items)
            if all_items:
                with open(queue_file, 'w') as f:
                    json.dump(queue_data, f, indent=2)
            else:
                if os.path.exists(queue_file):
                    os.remove(queue_file)
                QMessageBox.information(dlg, "Review Complete", "All items reviewed!")
                dlg.close()

        def _on_key(event):
            key = event.key()
            if key == Qt.Key.Key_A:
                _accept()
            elif key == Qt.Key.Key_K:
                _reject()
            elif key == Qt.Key.Key_S:
                _skip()
            elif key == Qt.Key.Key_G:
                _go_to_photo()
            else:
                QDialog.keyPressEvent(dlg, event)

        dlg.keyPressEvent = _on_key

        list_widget.currentItemChanged.connect(_update_preview)
        accept_btn.clicked.connect(_accept)
        reject_btn.clicked.connect(_reject)
        skip_btn.clicked.connect(_skip)
        props_btn.clicked.connect(_go_to_photo)
        close_btn.clicked.connect(dlg.close)

        # Select first item
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
            _update_preview()

        dlg.exec()

    def _gather_pending_sex_suggestions(self) -> list:
        """Return list of boxes with pending buck/doe suggestions.

        Optimized: Uses single SQL query instead of N+1 queries.
        Queries box-level suggestions (sex_conf > 0 means AI suggestion).
        Excludes photos that already have a deer_id set (implies buck already identified).
        """
        pending = []
        try:
            cursor = self.db.conn.cursor()

            # Auto-clear sex data on non-deer boxes (cleanup old incorrect data)
            cursor.execute("""
                UPDATE annotation_boxes
                SET sex = NULL, sex_conf = NULL
                WHERE (sex IS NOT NULL AND sex != '')
                  AND (
                      (species IS NOT NULL AND species != '' AND LOWER(species) != 'deer')
                      OR photo_id IN (
                          SELECT photo_id FROM tags
                          WHERE (tag_name IN ('Turkey', 'Coyote', 'Raccoon', 'Squirrel', 'Bird', 'Other Bird', 'Person', 'Vehicle', 'Empty', 'Verification')
                             OR tag_name LIKE '%Bird%')
                            AND deleted_at IS NULL
                      )
                  )
            """)
            if cursor.rowcount > 0:
                self.db.conn.commit()
                logger.info(f"Auto-cleared sex data from {cursor.rowcount} non-deer boxes")
            # Query boxes with sex suggestions (sex_conf > 0 = AI suggestion)
            # sex_conf = NULL or 0 means user confirmed, so exclude those
            cursor.execute("""
                SELECT
                    b.id as box_id,
                    b.photo_id,
                    b.x1, b.y1, b.x2, b.y2,
                    b.label, b.species, b.sex, b.sex_conf,
                    p.original_name, p.file_path, p.date_taken, p.import_date, p.collection
                FROM annotation_boxes b
                JOIN photos p ON b.photo_id = p.id
                LEFT JOIN deer_metadata dm ON dm.photo_id = p.id
                WHERE b.deleted_at IS NULL
                  AND b.sex IS NOT NULL
                  AND b.sex != ''
                  AND b.sex_conf IS NOT NULL
                  AND b.sex_conf > 0
                  AND (b.label IS NULL OR b.label NOT LIKE '%head%')
                  AND (dm.deer_id IS NULL OR dm.deer_id = '')
                  AND (b.species IS NULL OR b.species = '' OR LOWER(b.species) = 'deer')
                  AND NOT EXISTS (
                      SELECT 1 FROM tags t
                      WHERE t.photo_id = p.id
                      AND t.deleted_at IS NULL
                      AND (t.tag_name IN ('Turkey', 'Coyote', 'Raccoon', 'Squirrel', 'Bird', 'Other Bird', 'Person', 'Vehicle', 'Empty', 'Verification')
                           OR t.tag_name LIKE '%Bird%')
                  )
                ORDER BY b.sex_conf DESC
            """)

            for row in cursor.fetchall():
                box = {
                    "id": row[0],
                    "x1": row[2], "y1": row[3], "x2": row[4], "y2": row[5],
                    "label": row[6], "species": row[7], "sex": row[8], "sex_conf": row[9]
                }
                pending.append({
                    "photo_id": row[1],
                    "box_idx": 0,
                    "box": box,
                    "name": row[10] or Path(row[11] or "").name,
                    "sex": row[8].title() if row[8] else "",  # Capitalize buck -> Buck
                    "conf": row[9],
                    "path": row[11],
                    "date": row[12] or row[13] or "",
                    "collection": row[14] or "",
                })
        except Exception as e:
            logger.warning(f"Failed to gather pending sex suggestions: {e}")
        return pending

    def run_ai_boxes(self):
        """Run detector to propose subject boxes (AI-labeled). Prefer custom ONNX, fall back to torchvision."""
        if not self.current_pixmap:
            return
        photo = self._current_photo()
        if not photo:
            return
        # Skip if already fully human-labeled (species + boxes)
        if self._photo_has_human_species(photo) and self._photo_has_human_boxes(photo.get("id")):
            QMessageBox.information(self, "AI Boxes", "Skipped: photo already has human species and boxes.")
            return
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            QMessageBox.information(self, "AI Boxes", "Image file missing.")
            return
        try:
            new_boxes = []
            model_path = self._get_det_model_path()
            detector = None
            names = None
            if model_path:
                try:
                    from ultralytics import YOLO  # uses onnxruntime/torch under the hood
                    if not hasattr(self, "_yolo_detector") or getattr(self, "_yolo_detector_path", None) != model_path:
                        self._yolo_detector = YOLO(str(model_path))
                        self._yolo_detector_path = model_path
                    detector = self._yolo_detector
                    names = getattr(detector, "names", {}) if hasattr(detector, "names") else None
                except Exception:
                    detector = None
                    names = None
            new_boxes = self._detect_boxes_for_path(path, detector=detector, names=names, conf_thresh=0.25)
            if not new_boxes:
                QMessageBox.information(self, "AI Boxes", "No boxes detected.")
                return
            self.current_boxes = [b for b in self.current_boxes if not str(b.get("label", "")).startswith("ai_")]
            self.current_boxes.extend(new_boxes)
            self._draw_boxes()
            self._persist_boxes()
            QMessageBox.information(self, "AI Boxes", f"Added {len(new_boxes)} AI box(es).")
            return
        except Exception:
            pass
        # Fallback to torchvision detector if ONNX fails or is missing
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision import transforms
        except Exception:
            QMessageBox.information(self, "AI Boxes", "Detector not available (install torch/torchvision).")
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
            model.eval()
            tf = transforms.Compose([transforms.ToTensor()])
            import PIL.Image as PILImage
            img = PILImage.open(path).convert("RGB")
            inp = tf(img).to(device)
            with torch.no_grad():
                out = model([inp])[0]
            keep = [i for i, s in enumerate(out["scores"]) if float(s) >= 0.6]
            if not keep:
                QMessageBox.information(self, "AI Boxes", "No boxes detected.")
                return
            w, h = img.size
            new_boxes = []
            for i in keep:
                box = out["boxes"][i].cpu().numpy()
                x1, y1, x2, y2 = box
                x1 = max(0.0, min(w, x1)); x2 = max(0.0, min(w, x2))
                y1 = max(0.0, min(h, y1)); y2 = max(0.0, min(h, y2))
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                new_boxes.append({
                    "label": "ai_subject",
                    "x1": x1 / w,
                    "y1": y1 / h,
                    "x2": x2 / w,
                    "y2": y2 / h,
                })
            if not new_boxes:
                QMessageBox.information(self, "AI Boxes", "No boxes detected.")
                return
            self.current_boxes = [b for b in self.current_boxes if not str(b.get("label", "")).startswith("ai_")]
            self.current_boxes.extend(new_boxes)
            self._draw_boxes()
            self._persist_boxes()
            QMessageBox.information(self, "AI Boxes", f"Added {len(new_boxes)} AI box(es).")
        except Exception as exc:
            QMessageBox.warning(self, "AI Boxes", f"Detector failed: {exc}")

    def rerun_ai_current_photo(self):
        """Re-run all AI (detection + classification) on the current photo, ignoring existing labels."""
        if not self.photos or self.index >= len(self.photos):
            QMessageBox.information(self, "Re-run AI", "No photo selected.")
            return

        photo = self.photos[self.index]
        pid = photo.get("id")
        path = photo.get("file_path")

        if not path or not os.path.exists(path):
            QMessageBox.information(self, "Re-run AI", "Image file not found.")
            return

        # Update status and disable UI during processing
        self.statusBar().showMessage("Running AI detection (this may take a moment)...")
        self.setEnabled(False)
        QApplication.processEvents()

        results = []

        try:
            # Step 1: Clear existing AI boxes AND suggested tag
            self.current_boxes = [b for b in self.current_boxes if not str(b.get("label", "")).startswith("ai_")]
            self.db.set_suggested_tag(pid, None, None)
            photo["suggested_tag"] = None
            photo["suggested_confidence"] = None

            # Step 2: Run MegaDetector
            new_boxes = self._detect_boxes_megadetector(path, conf_thresh=0.2)
            if new_boxes:
                self.current_boxes.extend(new_boxes)
                results.append(f"Detected {len(new_boxes)} subject(s)")
            else:
                # Only suggest Empty if there are NO boxes at all
                existing_boxes = self.db.get_boxes(pid)
                if not existing_boxes and not self.current_boxes:
                    results.append("No subjects detected - suggesting Empty")
                    self.db.set_suggested_tag(pid, "Empty", 1.0)
                    photo["suggested_tag"] = "Empty"
                    photo["suggested_confidence"] = 1.0
                else:
                    results.append("No new subjects detected (existing boxes preserved)")

            # Step 3: Run species classifier on detected boxes
            if hasattr(self, "ai_suggester") and self.ai_suggester:
                animal_boxes = [b for b in self.current_boxes if str(b.get("label", "")).startswith("ai_animal")]
                if animal_boxes:
                    import PIL.Image as PILImage
                    img = PILImage.open(path).convert("RGB")
                    w, h = img.size

                    species_results = []
                    first_suggestion = None
                    for box in animal_boxes:
                        box_id = box.get("id")
                        x1 = int(box["x1"] * w)
                        y1 = int(box["y1"] * h)
                        x2 = int(box["x2"] * w)
                        y2 = int(box["y2"] * h)
                        pixel_area = (x2 - x1) * (y2 - y1)
                        crop = img.crop((x1, y1, x2, y2))

                        result = self.ai_suggester.predict(crop, pixel_area=pixel_area)
                        if result:
                            label, conf = result
                            species_results.append(f"{label} ({int(conf*100)}%)")
                            box["ai_suggested_species"] = label
                            # Store suggestion at box level
                            if box_id:
                                self.db.set_box_ai_suggestion(box_id, label, conf)
                            if first_suggestion is None:
                                first_suggestion = (label, conf)

                    if species_results:
                        results.append(f"Species: {', '.join(species_results)}")
                        # Also set photo-level suggestion (backwards compatibility)
                        if first_suggestion:
                            self.db.set_suggested_tag(pid, first_suggestion[0], first_suggestion[1])
                            photo["suggested_tag"] = first_suggestion[0]
                            photo["suggested_confidence"] = first_suggestion[1]

            # Step 4: Persist and update UI
            self._persist_boxes()
            self._draw_boxes()
            self._update_box_tab_bar()
            self._update_suggestion_display(photo)
            self._populate_photo_list()

            result_msg = "\n".join(results) if results else "AI processing complete"

        except Exception as e:
            result_msg = f"AI processing failed: {e}"
            logger.error(f"rerun_ai_current_photo error: {e}")

        finally:
            # Re-enable UI
            self.setEnabled(True)

        # Show results
        self.statusBar().showMessage(result_msg, 5000)
        QMessageBox.information(self, "Re-run AI", result_msg)

    def rerun_ai_on_selection(self):
        """Re-run AI on selected photos (or current photo if none selected)."""
        # Get selected photos
        selected_items = self.photo_list_widget.selectedItems()

        if not selected_items:
            # No selection - use current photo
            if not self.photos or self.index >= len(self.photos):
                QMessageBox.information(self, "Re-run AI", "No photo selected.")
                return
            selected_photos = [self.photos[self.index]]
        else:
            # Get photos from selection
            selected_photos = []
            for item in selected_items:
                idx = item.data(Qt.ItemDataRole.UserRole)
                if idx is not None and 0 <= idx < len(self.photos):
                    selected_photos.append(self.photos[idx])

        if not selected_photos:
            QMessageBox.information(self, "Re-run AI", "No photos selected.")
            return

        # Confirm if multiple
        if len(selected_photos) > 1:
            reply = QMessageBox.question(
                self, "Re-run AI",
                f"Run AI detection and classification on {len(selected_photos)} selected photos?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Progress dialog for multiple photos
        progress = None
        if len(selected_photos) > 1:
            progress = QProgressDialog("Running AI on selected photos...", "Cancel", 0, len(selected_photos), self)
            progress.setWindowTitle("Re-run AI")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

        processed = 0
        detected_count = 0
        cancelled = False

        for i, photo in enumerate(selected_photos):
            if progress:
                if progress.wasCanceled():
                    cancelled = True
                    break
                progress.setValue(i)
                progress.setLabelText(f"Processing {i+1}/{len(selected_photos)}...")
                QCoreApplication.processEvents()

            pid = photo.get("id")
            path = photo.get("file_path")

            if not path or not os.path.exists(path):
                continue

            # Clear existing AI suggestions
            try:
                self.db.set_suggested_tag(pid, None, None)
            except:
                pass

            # Run MegaDetector
            try:
                print(f"[AI DEBUG] Running MegaDetector on: {path}")
                new_boxes = self._detect_boxes_megadetector(path, conf_thresh=0.2)
                print(f"[AI DEBUG] Detected {len(new_boxes) if new_boxes else 0} boxes")
                if new_boxes:
                    # Save boxes to database (get existing non-AI boxes first, then add new AI boxes)
                    existing_boxes = self.db.get_boxes(pid)
                    non_ai_boxes = [b for b in existing_boxes if not str(b.get("label", "")).startswith("ai_")]
                    all_boxes = non_ai_boxes + new_boxes
                    self.db.set_boxes(pid, all_boxes)
                    detected_count += len(new_boxes)

                    # Run species classifier on each box
                    if hasattr(self, "ai_suggester") and self.ai_suggester:
                        import PIL.Image as PILImage
                        img = PILImage.open(path).convert("RGB")
                        w, h = img.size

                        # Re-fetch boxes to get their IDs from database
                        saved_boxes = self.db.get_boxes(pid)
                        ai_boxes = [b for b in saved_boxes if str(b.get("label", "")).startswith("ai_animal")]

                        first_suggestion = None
                        for box in ai_boxes:
                            box_id = box.get("id")
                            if not box_id:
                                continue
                            x1 = int(box["x1"] * w)
                            y1 = int(box["y1"] * h)
                            x2 = int(box["x2"] * w)
                            y2 = int(box["y2"] * h)
                            pixel_area = (x2 - x1) * (y2 - y1)
                            crop = img.crop((x1, y1, x2, y2))
                            result = self.ai_suggester.predict(crop, pixel_area=pixel_area)
                            if result:
                                label, conf = result
                                # Store suggestion at box level
                                self.db.set_box_ai_suggestion(box_id, label, conf)
                                if first_suggestion is None:
                                    first_suggestion = (label, conf)

                        # Also set photo-level suggestion (backwards compatibility)
                        if first_suggestion:
                            self.db.set_suggested_tag(pid, first_suggestion[0], first_suggestion[1])
                else:
                    # No detections - suggest Empty
                    self.db.set_suggested_tag(pid, "Empty", 1.0)

                processed += 1
            except Exception as e:
                print(f"[AI] Error processing {path}: {e}")

        if progress:
            progress.close()

        # Refresh UI
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_photo_list()
        self.load_photo()

        # Show results
        if cancelled:
            msg = f"Cancelled. Processed {processed} of {len(selected_photos)} photos."
        else:
            msg = f"Processed {processed} photo(s).\nDetected {detected_count} subject(s)."
        QMessageBox.information(self, "Re-run AI", msg)

    def detect_boxes_for_tagged_photos(self):
        """Run MegaDetector on photos that have species tags but no detection boxes."""
        # Valid animal species
        ANIMAL_SPECIES = {
            'Armadillo', 'Bobcat', 'Chipmunk', 'Coyote', 'Deer', 'Dog', 'Fox',
            'Ground Hog', 'House Cat', 'Opossum', 'Other', 'Other Bird', 'Otter',
            'Quail', 'Rabbit', 'Raccoon', 'Skunk', 'Squirrel', 'Turkey', 'Turkey Buzzard',
            'Flicker'
        }

        # Find photos with animal species tags but no boxes
        photos = self.db.get_all_photos(include_archived=True)
        needs_detection = []

        for p in photos:
            pid = p['id']
            tags = set(self.db.get_tags(pid))
            boxes = self.db.get_boxes(pid)

            animal_tags = tags & ANIMAL_SPECIES
            if animal_tags and not boxes:
                needs_detection.append(p)

        if not needs_detection:
            QMessageBox.information(self, "Detect Boxes",
                "All tagged photos already have detection boxes.")
            return

        # Confirm with user
        reply = QMessageBox.question(
            self, "Detect Boxes",
            f"Found {len(needs_detection)} photos with species tags but no detection boxes.\n\n"
            f"Run MegaDetector on these photos to add boxes?\n"
            f"(This may take a while)\n\n"
            f"A backup will be created before starting.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Create backup before batch operation
        self.statusBar().showMessage("Creating backup before batch operation...")
        QApplication.processEvents()
        if not self.db.backup_before_batch_operation():
            reply = QMessageBox.warning(
                self, "Backup Failed",
                "Could not create backup. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Progress dialog
        progress = QProgressDialog("Running MegaDetector...", "Cancel", 0, len(needs_detection), self)
        progress.setWindowTitle("Detecting Boxes")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        processed = 0
        detected_count = 0
        errors = 0
        cancelled = False

        # Use transaction for batch writes
        try:
            self.db.begin_transaction()

            for i, photo in enumerate(needs_detection):
                if progress.wasCanceled():
                    cancelled = True
                    break

                progress.setValue(i)
                progress.setLabelText(f"Processing {i+1}/{len(needs_detection)}...")
                QApplication.processEvents()

                pid = photo.get("id")
                path = photo.get("file_path")

                if not path or not os.path.exists(path):
                    errors += 1
                    continue

                try:
                    new_boxes = self._detect_boxes_megadetector(path, conf_thresh=0.2)
                    if new_boxes:
                        self.db.set_boxes(pid, new_boxes)
                        detected_count += len(new_boxes)
                    processed += 1

                    # Commit every 50 photos to avoid losing too much work
                    if processed % 50 == 0:
                        self.db.commit_transaction()
                        self.db.begin_transaction()

                except Exception as e:
                    print(f"[AI] Error detecting boxes for {path}: {e}")
                    errors += 1

            # Final commit
            self.db.commit_transaction()

        except Exception as e:
            logger.error(f"Batch operation failed: {e}")
            self.db.rollback_transaction()
            QMessageBox.critical(self, "Error", f"Batch operation failed: {e}\n\nChanges have been rolled back.")
            progress.close()
            return

        progress.close()

        # Refresh UI
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_photo_list()
        self.load_photo()

        # Show results
        msg = f"Processed {processed} photos.\n"
        msg += f"Added {detected_count} detection boxes.\n"
        if errors:
            msg += f"Errors: {errors}"
        QMessageBox.information(self, "Detect Boxes", msg)

    def run_ai_boxes_all(self):
        """Run detector across all photos in the database."""
        progress = None
        try:
            targets = list(self.db.get_all_photos())
            if not targets:
                QMessageBox.information(self, "AI Boxes", "No photos to process.")
                return
            progress = QProgressDialog("Running AI detect on all photos...", "Cancel", 0, len(targets), self)
            progress.setWindowTitle("AI Detect All")
            progress.setMinimumDuration(0)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            progress.setValue(0)
            progress.show()
            QCoreApplication.processEvents()
            model_path = self._get_det_model_path()
            detector = None
            names = None
            if model_path:
                try:
                    from ultralytics import YOLO
                    detector = YOLO(str(model_path))
                    names = getattr(detector, "names", {}) if hasattr(detector, "names") else None
                except Exception as exc:
                    if not getattr(self, "_detector_warned", False):
                        QMessageBox.warning(
                            self,
                            "AI Boxes",
                            f"Failed to load detector ({model_path}): {exc}\n"
                            "Falling back to torchvision. If this is an ONNX opset error, re-export with opset_version=21."
                        )
                        self._detector_warned = True
                    detector = None
                    names = None
            added_total = 0
            per_photo = []
            failures = 0
            empty_detections = 0
            queued = len(targets)
            first_error = None
            for idx, p in enumerate(targets, start=1):
                if progress:
                    progress.setValue(idx)
                    QCoreApplication.processEvents()
                    if progress.wasCanceled():
                        break
                path = p.get("file_path")
                if not path or not os.path.exists(path):
                    failures += 1
                    if first_error is None:
                        first_error = f"Missing file: {path}"
                    continue
                if self._photo_has_human_species(p) and self._photo_has_human_boxes(p.get("id")):
                    continue
                try:
                    new_boxes = self._detect_boxes_for_path(path, detector=detector, names=names, conf_thresh=0.25)
                    if not new_boxes:
                        per_photo.append((p.get("id"), 0))
                        empty_detections += 1
                        continue
                    # merge with existing boxes for that photo
                    existing = self.db.get_boxes(p["id"]) if hasattr(self.db, "get_boxes") else []
                    merged = [b for b in existing if not str(b.get("label", "")).startswith("ai_")]
                    merged.extend(new_boxes)
                    self.db.set_boxes(p["id"], merged)
                    added_total += len(new_boxes)
                    per_photo.append((p.get("id"), len(new_boxes)))
                except Exception as exc:
                    failures += 1
                    if first_error is None:
                        first_error = str(exc)
                    continue
            if progress:
                progress.setValue(len(targets))
                QCoreApplication.processEvents()
            # Refresh current view
            self.photos = self._sorted_photos(self.db.get_all_photos())
            if self.index >= len(self.photos):
                self.index = max(0, len(self.photos) - 1)
            self.load_photo()
            processed = len({pid for pid, _ in per_photo})
            msg = (
                f"Queued {queued} photo(s).\n"
                f"Added {added_total} AI box(es) across {processed} photo(s).\n"
                f"(Confidence >= 0.25; small boxes <6px skipped)"
            )
            if empty_detections:
                msg += f"\nNo detections in {empty_detections} photo(s)."
            if failures:
                msg += f"\nSkipped {failures} photo(s) due to missing files/errors."
                if first_error:
                    msg += f"\nFirst error: {first_error}"
            QMessageBox.information(self, "AI Boxes", msg)
        except Exception as exc:
            QMessageBox.warning(self, "AI Boxes", f"Bulk detect failed: {exc}")
        finally:
            # Reset so a future attempt will show the warning again if needed
            self._detector_warned = False
            if progress:
                progress.reset()

    def _photo_has_human_species(self, photo: dict) -> bool:
        """Check if photo has a human-applied species (tags intersect species set)."""
        if not photo:
            return False
        species_set = self._species_set()
        try:
            tags = set(self.db.get_tags(photo["id"]))
        except Exception:
            return False
        return any(t in species_set for t in tags)

    def _photo_has_human_boxes(self, pid: Optional[int]) -> bool:
        """Check if photo has any non-AI boxes."""
        if not pid:
            return False
        try:
            boxes = self.db.get_boxes(pid)
        except Exception:
            return False
        return any(not str(b.get("label", "")).startswith("ai_") for b in boxes)

    def _get_det_model_path(self):
        """Return preferred detector model path if present (ONNX preferred, PT fallback)."""
        candidates = [
            Path("runs/detect/train/weights/best.pt"),
            Path("runs/detect/exp/weights/best.pt"),
            Path("models/detector.pt"),
            Path("runs/detect/train/weights/best.onnx"),
            Path("runs/detect/exp/weights/best.onnx"),
            Path("models/detector.onnx"),
            Path("yolov8n.pt"),  # bundled coco-pretrain for suggestions only
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def accept_ai_boxes(self):
        """Review AI boxes individually; keep checked ones, convert to human labels, drop the rest."""
        ai_boxes = [(i, b) for i, b in enumerate(self.current_boxes) if str(b.get("label", "")).startswith("ai_")]
        if not ai_boxes:
            QMessageBox.information(self, "AI Boxes", "No AI boxes to review.")
            return
        keep_indices = {i for i, _ in ai_boxes}  # one-click accept all AI
        new_boxes = []
        for i, b in enumerate(self.current_boxes):
            if str(b.get("label", "")).startswith("ai_"):
                if i in keep_indices:
                    lbl = b.get("label", "")
                    b["label"] = "deer_head" if "head" in str(lbl).lower() else "subject"
                    new_boxes.append(b)
                continue
            new_boxes.append(b)
        self.current_boxes = new_boxes
        self._draw_boxes()
        self._persist_boxes()
        if not self._photo_has_ai_boxes(self.current_boxes):
            if self.ai_review_mode:
                self._advance_ai_review()
            else:
                self.next_photo()

    def accept_all_ai_boxes(self):
        """Keep all AI boxes, convert to manual labels."""
        changed = False
        for b in self.current_boxes:
            if str(b.get("label", "")).startswith("ai_"):
                b["label"] = "deer_head" if "head" in str(b.get("label", "")).lower() else "subject"
                changed = True
        if changed:
            self._draw_boxes()
            self._persist_boxes()
            QMessageBox.information(self, "AI Boxes", "Accepted all AI boxes.")
            if not self._photo_has_ai_boxes(self.current_boxes):
                if self.ai_review_mode:
                    self._advance_ai_review()
                else:
                    self.next_photo()

    def edit_boxes(self):
        """Dialog to tweak or delete any box (AI or manual)."""
        if not self.current_boxes:
            QMessageBox.information(self, "Edit Boxes", "No boxes to edit.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Boxes")
        layout = QVBoxLayout(dlg)
        rows = []
        for i, b in enumerate(self.current_boxes):
            row = QHBoxLayout()
            label = QLabel(f"{i+1}: {b.get('label','')}")
            for lbl in (label,):
                lbl.setMinimumWidth(90)
            spin_x1 = QDoubleSpinBox(); spin_x1.setRange(0, 1); spin_x1.setSingleStep(0.01); spin_x1.setDecimals(4); spin_x1.setValue(float(b.get("x1", 0)))
            spin_y1 = QDoubleSpinBox(); spin_y1.setRange(0, 1); spin_y1.setSingleStep(0.01); spin_y1.setDecimals(4); spin_y1.setValue(float(b.get("y1", 0)))
            spin_x2 = QDoubleSpinBox(); spin_x2.setRange(0, 1); spin_x2.setSingleStep(0.01); spin_x2.setDecimals(4); spin_x2.setValue(float(b.get("x2", 1)))
            spin_y2 = QDoubleSpinBox(); spin_y2.setRange(0, 1); spin_y2.setSingleStep(0.01); spin_y2.setDecimals(4); spin_y2.setValue(float(b.get("y2", 1)))
            del_btn = QToolButton(); del_btn.setText("✕"); del_btn.setToolTip("Remove this box")
            row.addWidget(label)
            row.addWidget(QLabel("x1")); row.addWidget(spin_x1)
            row.addWidget(QLabel("y1")); row.addWidget(spin_y1)
            row.addWidget(QLabel("x2")); row.addWidget(spin_x2)
            row.addWidget(QLabel("y2")); row.addWidget(spin_y2)
            row.addWidget(del_btn)
            row_widget = QWidget()
            row_widget.setLayout(row)
            layout.addWidget(row_widget)
            rows.append({"idx": i, "x1": spin_x1, "y1": spin_y1, "x2": spin_x2, "y2": spin_y2, "del_btn": del_btn})
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        # Track deletes
        to_delete = set()
        def make_del_handler(box_idx):
            return lambda: (to_delete.add(box_idx), None)
        for r in rows:
            r["del_btn"].clicked.connect(make_del_handler(r["idx"]))

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_boxes = []
        for r in rows:
            idx = r["idx"]
            if idx in to_delete:
                continue
            x1 = min(max(r["x1"].value(), 0.0), 1.0)
            y1 = min(max(r["y1"].value(), 0.0), 1.0)
            x2 = min(max(r["x2"].value(), 0.0), 1.0)
            y2 = min(max(r["y2"].value(), 0.0), 1.0)
            if x2 - x1 < 0.001 or y2 - y1 < 0.001:
                continue
            b = dict(self.current_boxes[idx])
            b["x1"], b["y1"], b["x2"], b["y2"] = x1, y1, x2, y2
            new_boxes.append(b)
        self.current_boxes = new_boxes
        self._draw_boxes()
        self._persist_boxes()

    @staticmethod
    def _find_image_files(folder: Path):
        """Find all JPG/JPEG files in a folder (recursive)."""
        exts = {".jpg", ".jpeg"}
        skip_folders = {'.cuddelink_tmp', '.thumbnails'}
        return [p for p in folder.rglob("*")
                if p.suffix.lower() in exts
                and not any(skip in str(p) for skip in skip_folders)]

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

    def _buck_exists(self, deer_id: str) -> bool:
        """Check if buck_id exists in any metadata tables."""
        if not deer_id:
            return False
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT 1 FROM deer_metadata WHERE deer_id = ? LIMIT 1", (deer_id,))
        if cursor.fetchone():
            return True
        cursor.execute("SELECT 1 FROM deer_additional WHERE deer_id = ? LIMIT 1", (deer_id,))
        return cursor.fetchone() is not None

    def on_deer_id_changed(self, text: str):
        """Handle deer ID edits: default species/sex and autosave when length is reasonable."""
        deer_id = text.strip()
        if deer_id:
            # Block signals to prevent queue advance when auto-setting species
            self.species_combo.blockSignals(True)
            if not self.species_combo.currentText().strip():
                self.species_combo.setCurrentText("Deer")
            self.species_combo.blockSignals(False)
            if self._get_sex() == "Unknown":
                self._set_sex("Buck")
        else:
            # If cleared back to unknown, default species to Deer
            self.species_combo.blockSignals(True)
            if self._get_sex() == "Unknown" and not self.species_combo.currentText().strip():
                self.species_combo.setCurrentText("Deer")
            self.species_combo.blockSignals(False)
        # Autosave when user has typed at least 3 chars (avoid saving partial IDs), OR when field is cleared
        if len(deer_id) >= 3 or deer_id == "":
            self.schedule_save()

    def _populate_species_dropdown(self):
        """Fill species dropdown with fixed alphabetized species list."""
        self.species_combo.blockSignals(True)
        current = self.species_combo.currentText()
        self.species_combo.clear()
        # Use fixed species list only (already alphabetized in SPECIES_OPTIONS)
        for s in SPECIES_OPTIONS:
            self.species_combo.addItem(s)
        if current:
            self.species_combo.setCurrentText(current)
        self.species_combo.blockSignals(False)
        self._update_recent_species_buttons()
        self._update_recent_suggest_buttons()

    def _recent_species(self, limit: int = 10) -> List[str]:
        """Return up to `limit` most recently used species.

        Prioritizes species applied during this session, then falls back to
        species by photo date_taken for variety.
        """
        from datetime import datetime

        # Start with session-applied species (most recently applied first)
        result = list(self._session_recent_species[:limit])

        # If we need more, fill from database by photo date
        if len(result) < limit:
            species_dates = {}
            species_set = set(SPECIES_OPTIONS)
            try:
                species_set.update([s for s in self.db.list_custom_species() if s])
            except Exception:
                pass

            def parse_dt(val: str):
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(val, fmt)
                    except Exception:
                        continue
                try:
                    parsed = datetime.fromisoformat(val)
                    # Strip timezone info to avoid naive vs aware comparison errors
                    if parsed.tzinfo is not None:
                        parsed = parsed.replace(tzinfo=None)
                    return parsed
                except Exception:
                    return None

            for photo in self.db.get_all_photos():
                dt = parse_dt(photo.get("date_taken") or "") or datetime.min
                tags = set(self.db.get_tags(photo["id"]))
                for t in tags:
                    if t in species_set and t.lower() not in SEX_TAGS:
                        if t not in species_dates or dt > species_dates[t]:
                            species_dates[t] = dt

            # Add species not already in result
            ordered = sorted(species_dates.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            for species, _ in ordered:
                if species not in result:
                    result.append(species)
                    if len(result) >= limit:
                        break

        return result[:limit]

    def _update_recent_species_buttons(self):
        recents = self._recent_species(limit=10)
        # Fallback to defaults if nothing labeled yet
        if not recents:
            recents = [s for s in SPECIES_OPTIONS if s]  # Skip empty string
        # Sort alphabetically
        recents = sorted([s for s in recents if s])
        for idx, btn in enumerate(self.recent_species_btns):
            if idx < len(recents):
                btn.setText(recents[idx])
                btn.setEnabled(True)
            else:
                btn.setText("")
                btn.setEnabled(False)

    def _recent_suggested_species(self, limit: int = 10) -> List[str]:
        # mirror recent used species for suggestion quick buttons (applied labels only)
        return self._recent_species(limit=limit)

    def _update_recent_suggest_buttons(self):
        if not hasattr(self, "recent_suggest_btns"):
            return
        recents = self._recent_suggested_species(limit=5)
        # Sort alphabetically
        recents = sorted([s for s in recents if s])
        for idx, btn in enumerate(self.recent_suggest_btns):
            if idx < len(recents):
                btn.setText(recents[idx])
                btn.setEnabled(True)
            else:
                btn.setText("")
                btn.setEnabled(False)

    def _on_quick_suggest_clicked(self):
        btn = self.sender()
        if not btn:
            return
        label = btn.text().strip()
        if not label:
            return
        self.species_combo.setCurrentText(label)
        self.schedule_save()

    def _on_quick_species_clicked(self):
        btn = self.sender()
        if not btn or not btn.text():
            return
        species = btn.text()

        # Update the current box's species directly
        if hasattr(self, "current_boxes") and self.current_boxes:
            if self.current_box_index < len(self.current_boxes):
                self.current_boxes[self.current_box_index]["species"] = species
                self._update_box_tab_name(self.current_box_index)

        # Update the combo (without triggering _on_species_changed again)
        self.species_combo.blockSignals(True)
        self.species_combo.setCurrentText(species)
        self.species_combo.blockSignals(False)

        # Save immediately
        self.save_current()

        # In queue mode, check if all boxes labeled and advance
        if self.queue_mode and not self._loading_photo_data:
            # Don't auto-advance if "Multiple species" is checked
            if hasattr(self, 'queue_multi_species') and self.queue_multi_species.isChecked():
                return
            # Only advance when ALL boxes are labeled
            if self._all_boxes_labeled():
                current_pid = None
                if self.photos and self.index < len(self.photos):
                    current_pid = self.photos[self.index].get("id")
                if current_pid:
                    if current_pid in self.queue_reviewed:
                        return  # Already reviewed
                    self.queue_reviewed.add(current_pid)
                    self.db.set_suggested_tag(current_pid, None, None)
                    self._mark_current_list_item_reviewed()
                self._queue_advance()

    def _on_other_species_clicked(self):
        """Apply 'Other' species tag (custom species addition disabled)."""
        self.species_combo.setCurrentText("Other")
        self.save_current()

    def _update_current_species_label(self, species_list: List[str]):
        """Update the label showing all current species tags."""
        if not species_list:
            self.current_species_label.setText("")
        elif len(species_list) == 1:
            self.current_species_label.setText(f"Tagged: {species_list[0]}")
        else:
            self.current_species_label.setText(f"Tagged: {', '.join(species_list)}")

    def _add_species_tag(self):
        """Add the selected species as an additional tag (multi-species support)."""
        photo = self._current_photo()
        if not photo:
            return
        species = self.species_combo.currentText().strip()
        if not species or species.lower() in SEX_TAGS:
            return
        pid = photo["id"]
        # Get current tags and species
        tags = self.db.get_tags(pid)
        species_options = set(SPECIES_OPTIONS)
        try:
            species_options.update(self.db.list_custom_species())
        except Exception:
            pass
        current_species = [t for t in tags if t in species_options]
        # Add if not already present
        if species not in current_species:
            self.db.add_tag(pid, species)
            current_species.append(species)
            self._update_current_species_label(current_species)
            # Note: Custom species addition disabled - using fixed species list
            self._update_recent_species_buttons()
            # Refresh photo data
            photo.update(self.db.get_photo_by_id(pid) or {})
            self._update_photo_list_item(self.index)

    def _bump_recent_buck(self, deer_id: Optional[str]):
        if not deer_id:
            return
        if len(deer_id.strip()) < 2:
            return
        # Only add to recent if this buck actually exists in metadata
        # (prevents partial typing from polluting the quick buttons)
        if not self._buck_exists(deer_id.strip()):
            return
        try:
            self.db.bump_recent_buck(deer_id)
        except Exception:
            pass
        self._update_recent_buck_buttons()

    def _update_recent_buck_buttons(self):
        try:
            recents_raw = self.db.list_recent_bucks(12)
        except Exception:
            recents_raw = []
        recents = []
        for did in recents_raw:
            if self._buck_exists(did):
                recents.append(did)
        # Fill buttons in order; if gaps appear, leave them blank until new IDs arrive.
        max_chars = 15  # Truncate display text to fit button width
        for idx, btn in enumerate(self.quick_buck_btns):
            if idx < len(recents):
                deer_id = recents[idx]
                # Store full name for click handler, show truncated for display
                btn.setProperty("full_deer_id", deer_id)
                if len(deer_id) > max_chars:
                    btn.setText(deer_id[:max_chars] + "...")
                else:
                    btn.setText(deer_id)
                btn.setToolTip(deer_id)  # Show full name on hover
                btn.setEnabled(True)
            else:
                btn.setText("")
                btn.setEnabled(False)

    def _label_priority_score(self, photo: dict) -> float:
        """Lower score = higher priority for labeling."""
        pid = photo.get("id")
        score = 0.0
        try:
            tags = set(self.db.get_tags(pid))
        except Exception:
            tags = set()
        # Encourage unlabeled
        if not tags:
            score -= 10.0
        # Species present?
        species_set = set(SPECIES_OPTIONS)
        try:
            species_set.update(self.db.list_custom_species())
        except Exception:
            pass
        has_species = any(t in species_set for t in tags)
        if not has_species:
            score -= 5.0
        # Buck metadata missing?
        try:
            deer = self.db.get_deer_metadata(pid)
            if not deer.get("deer_id"):
                score -= 2.0
        except Exception:
            pass
        # AI suggestion uncertainty (closer to 0.5 = higher priority)
        sugg = photo.get("suggested_tag")
        conf = photo.get("suggested_confidence")
        if sugg and conf is not None:
            score += abs(conf - 0.5) * 10.0
        else:
            score -= 1.0  # no suggestion -> some priority
        # Older photos lower priority by default
        dt = photo.get("date_taken") or ""
        score += 0.001 * hash(dt)
        return score

    def prioritize_for_labeling(self):
        """Reorder photo list to surface highest-value items to label."""
        if not self.photos:
            return
        # Use DB fresh to include current saved state
        photos = self.db.get_all_photos()
        scored = []
        for p in photos:
            scored.append((self._label_priority_score(p), p))
        scored.sort(key=lambda x: x[0])
        self.photos = [p for _, p in scored]
        self._populate_photo_list()
        if self.photos:
            self.index = 0
            self.load_photo()

    def _on_quick_buck_clicked(self):
        btn = self.sender()
        if not btn or not btn.text():
            return
        # Get full deer_id from property (not truncated text)
        deer_id = btn.property("full_deer_id") or btn.text()
        self.deer_id_edit.setCurrentText(deer_id)
        # Autosave will fire via signals; ensure selection propagates
        self._apply_buck_profile_to_ui()
        # Immediately register this ID as recent so buttons refresh without waiting for save
        self._bump_recent_buck(deer_id)

    def merge_buck_ids_dialog(self):
        """Merge one buck ID into another/new ID with confirmation and note."""
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
            affected = self.db.merge_deer_ids(src, tgt)
            if new_id != tgt:
                self.db.merge_deer_ids(tgt, new_id)
            self._note_original_name(src, new_id)
            QMessageBox.information(dlg, "Merge", f"Merged {src} into {new_id}. Photos updated: {affected}")
            self._populate_deer_id_dropdown()
            dlg.accept()

        ok_btn.clicked.connect(do_merge)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()

    def open_buck_profile(self):
        """Show a profile dialog with seasons and full photo history."""
        deer_id = self.deer_id_edit.currentText().strip()
        if not deer_id:
            QMessageBox.information(self, "Buck Profile", "Set a Deer ID first.")
            return
        summaries = self.db.get_buck_season_summaries(deer_id)
        encounters = self.db.get_buck_encounters(deer_id, window_minutes=30)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Buck Profile: {deer_id}")
        layout = QVBoxLayout(dlg)
        info = QTextBrowser()
        lines = []
        lines.append("<b>Season Summaries</b>")
        prev = None
        for s in summaries:
            label = self.db.format_season_label(s["season_year"]) if s.get("season_year") else "Unknown season"
            lines.append(f"{label} ({s.get('photo_count',0)} photos)")
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

        # Encounter tree (collapsible)
        enc_label = QLabel("Encounters (grouped by camera within 30 minutes)")
        layout.addWidget(enc_label)
        enc_tree = QTreeWidget()
        enc_tree.setHeaderLabels(["Encounter / Photos"])
        enc_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        # Build a quick lookup for photo details
        photo_lookup = {}
        for p in self.db.get_all_photos():
            photo_lookup[p["id"]] = p
        for enc in encounters:
            cam = enc.get("camera_location") or "Unknown camera"
            start = enc.get("start") or "?"
            end = enc.get("end") or "?"
            count = enc.get("count") or 0
            title = f"{cam} | {start} – {end} | {count} photo(s)"
            root = QTreeWidgetItem([title])
            root.setData(0, Qt.ItemDataRole.UserRole, enc.get("photo_ids") or [])
            for pid in enc.get("photo_ids") or []:
                p = photo_lookup.get(pid, {})
                fname = os.path.basename(p.get("file_path", "")) if p else ""
                dt = p.get("date_taken") or ""
                child = QTreeWidgetItem([f"{dt} | {fname}"])
                child.setData(0, Qt.ItemDataRole.UserRole, pid)
                root.addChild(child)
            root.setExpanded(False)
            enc_tree.addTopLevelItem(root)
        layout.addWidget(enc_tree)

        btns = QHBoxLayout()
        open_btn = QPushButton("Open Selection")
        compare_btn = QPushButton("Compare Selection")
        change_id_btn = QPushButton("Change ID...")
        close = QPushButton("Close")
        btns.addWidget(open_btn)
        btns.addWidget(compare_btn)
        btns.addWidget(change_id_btn)
        btns.addStretch()
        btns.addWidget(close)
        layout.addLayout(btns)

        def open_selected():
            sel = enc_tree.selectedItems()
            if not sel:
                return
            # prefer first child if selected, else encounter's first photo
            item = sel[0]
            data = item.data(0, Qt.ItemDataRole.UserRole)
            pid = None
            if isinstance(data, list) and data:
                pid = data[0]
            elif isinstance(data, int):
                pid = data
            if pid:
                self._select_photo_by_id(pid)
                dlg.accept()

        def compare_selected():
            sel = enc_tree.selectedItems()
            if not sel:
                QMessageBox.information(dlg, "Compare", "Select encounters or photos to compare.")
                return
            photo_ids = []
            for item in sel:
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(data, list):
                    for pid in data:
                        if pid not in photo_ids:
                            photo_ids.append(pid)
                elif isinstance(data, int):
                    if data not in photo_ids:
                        photo_ids.append(data)
                if len(photo_ids) >= 4:
                    break
            if len(photo_ids) < 2:
                QMessageBox.information(dlg, "Compare", "Select at least two photos (or an encounter with multiple photos).")
                return
            dlg.accept()
            cdlg = CompareWindow(photo_ids=photo_ids[:4], db=self.db, parent=self)
            cdlg.exec()

        def change_deer_id():
            # Get available seasons for this deer
            seasons = [s.get("season_year") for s in summaries if s.get("season_year")]

            # Create change ID dialog
            change_dlg = QDialog(dlg)
            change_dlg.setWindowTitle(f"Change Deer ID: {deer_id}")
            change_dlg.setMinimumWidth(400)
            clayout = QVBoxLayout(change_dlg)

            clayout.addWidget(QLabel(f"Current ID: <b>{deer_id}</b>"))

            clayout.addWidget(QLabel("New ID:"))
            new_id_edit = QLineEdit()
            new_id_edit.setPlaceholderText("Enter new deer ID...")
            clayout.addWidget(new_id_edit)

            clayout.addSpacing(10)
            clayout.addWidget(QLabel("Apply to:"))

            scope_group = QButtonGroup(change_dlg)
            all_radio = QRadioButton(f"All photos with this deer ID")
            all_radio.setChecked(True)
            scope_group.addButton(all_radio)
            clayout.addWidget(all_radio)

            season_radio = QRadioButton("Only selected seasons:")
            scope_group.addButton(season_radio)
            clayout.addWidget(season_radio)

            # Season checkboxes
            season_checks = {}
            season_widget = QWidget()
            season_layout = QVBoxLayout(season_widget)
            season_layout.setContentsMargins(20, 0, 0, 0)
            for sy in sorted(seasons, reverse=True):
                label = self.db.format_season_label(sy)
                cb = QCheckBox(label)
                cb.setChecked(True)
                season_checks[sy] = cb
                season_layout.addWidget(cb)
            clayout.addWidget(season_widget)
            season_widget.setEnabled(False)

            def toggle_seasons(checked):
                season_widget.setEnabled(season_radio.isChecked())

            season_radio.toggled.connect(toggle_seasons)
            all_radio.toggled.connect(toggle_seasons)

            clayout.addSpacing(10)

            cbtn_layout = QHBoxLayout()
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(change_dlg.reject)
            apply_btn = QPushButton("Apply Change")
            apply_btn.clicked.connect(change_dlg.accept)
            cbtn_layout.addStretch()
            cbtn_layout.addWidget(cancel_btn)
            cbtn_layout.addWidget(apply_btn)
            clayout.addLayout(cbtn_layout)

            if change_dlg.exec() != QDialog.DialogCode.Accepted:
                return

            new_id = new_id_edit.text().strip()
            if not new_id:
                QMessageBox.warning(dlg, "Change ID", "Please enter a new deer ID.")
                return

            if new_id == deer_id:
                QMessageBox.warning(dlg, "Change ID", "New ID is the same as the current ID.")
                return

            # Confirm
            if all_radio.isChecked():
                msg = f"Change deer ID from '{deer_id}' to '{new_id}' for ALL photos?"
                selected_seasons = None
            else:
                selected_seasons = [sy for sy, cb in season_checks.items() if cb.isChecked()]
                if not selected_seasons:
                    QMessageBox.warning(dlg, "Change ID", "Please select at least one season.")
                    return
                season_labels = [self.db.format_season_label(sy) for sy in selected_seasons]
                msg = f"Change deer ID from '{deer_id}' to '{new_id}' for seasons:\n{', '.join(season_labels)}?"

            if QMessageBox.question(dlg, "Confirm Change", msg,
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                    ) != QMessageBox.StandardButton.Yes:
                return

            # Apply the change
            affected = self.db.rename_deer_id(deer_id, new_id, selected_seasons)
            QMessageBox.information(dlg, "Change ID", f"Changed {affected} photo(s) from '{deer_id}' to '{new_id}'.")

            # Refresh UI
            self._populate_deer_id_dropdown()
            self.deer_id_edit.setCurrentText(new_id)
            dlg.accept()  # Close profile dialog

        open_btn.clicked.connect(open_selected)
        compare_btn.clicked.connect(compare_selected)
        change_id_btn.clicked.connect(change_deer_id)
        close.clicked.connect(dlg.accept)
        dlg.resize(700, 600)
        dlg.exec()

    def review_site_suggestions(self):
        """Review and approve/reject pending camera location suggestions."""
        import struct
        # Gather pending site suggestions
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT p.id, p.file_path, p.thumbnail_path, p.suggested_site_id,
                   p.suggested_site_confidence, s.name as suggested_name
            FROM photos p
            JOIN sites s ON p.suggested_site_id = s.id
            WHERE p.suggested_site_id IS NOT NULL
            ORDER BY s.name, p.suggested_site_confidence DESC
        """)
        pending = []
        for row in cursor.fetchall():
            item = dict(row)
            # Convert bytes confidence to float if needed
            conf = item.get("suggested_site_confidence")
            if isinstance(conf, bytes):
                try:
                    conf = struct.unpack('f', conf[:4])[0]
                except:
                    conf = 0.5
            item["suggested_site_confidence"] = conf or 0
            pending.append(item)

        if not pending:
            QMessageBox.information(self, "Site Suggestions", "No pending site suggestions to review.")
            return

        # Create review dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Review Site Suggestions ({len(pending)})")
        dlg.resize(1000, 700)
        layout = QVBoxLayout(dlg)

        # Main content: list on left, photo on right
        content_layout = QHBoxLayout()

        # Left side: list with location filter
        left_panel = QVBoxLayout()

        # Location filter
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Location:"))
        location_filter = QComboBox()
        location_filter.addItem("All")
        locations = sorted(set(item["suggested_name"] for item in pending))
        for loc in locations:
            count = sum(1 for item in pending if item["suggested_name"] == loc)
            location_filter.addItem(f"{loc} ({count})")
        location_filter.setMinimumWidth(150)
        filter_row.addWidget(location_filter)
        filter_row.addStretch()
        left_panel.addLayout(filter_row)

        left_panel.addWidget(QLabel("Pending suggestions:"))
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setMinimumWidth(250)

        all_pending = pending[:]  # Store original list
        reviewed_rows = set()  # Track reviewed row indices

        def _populate_list():
            """Populate list based on current filter."""
            list_widget.clear()
            reviewed_rows.clear()
            loc_text = location_filter.currentText()
            loc_filter = loc_text.split(" (")[0] if " (" in loc_text else loc_text

            for item in all_pending:
                if loc_filter != "All" and item["suggested_name"] != loc_filter:
                    continue

                conf_pct = int((item["suggested_site_confidence"] or 0) * 100)
                path = item.get("file_path") or item.get("thumbnail_path") or ""
                fname = os.path.basename(path) if path else f"Photo {item['id']}"

                # Check if already reviewed
                if item.get("_reviewed"):
                    display = f"✓ {item.get('_action', 'Reviewed')} - {fname}"
                    li = QListWidgetItem(display)
                    li.setBackground(QColor(144, 238, 144))  # Light green
                    reviewed_rows.add(list_widget.count())
                else:
                    display = f"{item['suggested_name']} ({conf_pct}%) - {fname}"
                    li = QListWidgetItem(display)

                li.setData(Qt.ItemDataRole.UserRole, item)
                list_widget.addItem(li)

            # Update title with remaining count
            reviewed_count = sum(1 for item in all_pending if item.get("_reviewed"))
            remaining = len(all_pending) - reviewed_count
            dlg.setWindowTitle(f"Review Site Suggestions ({remaining} remaining)")

        _populate_list()
        left_panel.addWidget(list_widget, 1)
        content_layout.addLayout(left_panel)

        # Right side: photo preview
        right_panel = QVBoxLayout()
        info_label = QLabel("Select a suggestion to preview")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_panel.addWidget(info_label)

        photo_label = QLabel()
        photo_label.setMinimumSize(500, 400)
        photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        photo_label.setStyleSheet("background-color: #222; border: 1px solid #555;")
        right_panel.addWidget(photo_label, 1)

        content_layout.addLayout(right_panel, 1)
        layout.addLayout(content_layout)

        def _update_preview():
            """Update photo preview when selection changes."""
            item = list_widget.currentItem()
            if not item:
                photo_label.clear()
                info_label.setText("Select a suggestion to preview")
                return

            data = item.data(Qt.ItemDataRole.UserRole)
            conf_pct = int((data["suggested_site_confidence"] or 0) * 100)
            info_label.setText(f"Suggested: {data['suggested_name']} ({conf_pct}% confidence)")

            path = data.get("file_path") or data.get("thumbnail_path")
            if path and os.path.exists(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    scaled = pix.scaled(photo_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
                    photo_label.setPixmap(scaled)
                else:
                    photo_label.setText("Failed to load image")
            else:
                photo_label.setText("Image not found")

        list_widget.currentItemChanged.connect(_update_preview)
        location_filter.currentIndexChanged.connect(_populate_list)

        # Buttons
        btn_layout = QHBoxLayout()

        approve_btn = QPushButton("✓ Approve")
        approve_btn.setStyleSheet("background-color: #2a5; color: white; font-weight: bold; padding: 8px 16px;")
        reject_btn = QPushButton("✗ Reject")
        reject_btn.setStyleSheet("background-color: #a33; color: white; font-weight: bold; padding: 8px 16px;")
        approve_all_btn = QPushButton("Approve All Shown")
        reject_all_btn = QPushButton("Reject All Shown")
        close_btn = QPushButton("Close")

        btn_layout.addWidget(approve_btn)
        btn_layout.addWidget(reject_btn)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(approve_all_btn)
        btn_layout.addWidget(reject_all_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        def _mark_reviewed(item, action: str, location: str):
            """Mark item as reviewed with green highlight."""
            data = item.data(Qt.ItemDataRole.UserRole)
            data["_reviewed"] = True
            data["_action"] = f"{action}: {location}"
            row = list_widget.row(item)
            reviewed_rows.add(row)

            # Update display
            path = data.get("file_path") or data.get("thumbnail_path") or ""
            fname = os.path.basename(path) if path else f"Photo {data['id']}"
            item.setText(f"✓ {action}: {location} - {fname}")
            item.setBackground(QColor(144, 238, 144))  # Light green

            # Update title
            reviewed_count = sum(1 for p in all_pending if p.get("_reviewed"))
            remaining = len(all_pending) - reviewed_count
            dlg.setWindowTitle(f"Review Site Suggestions ({remaining} remaining)")

        def _next_unreviewed():
            """Move to next unreviewed item."""
            current = list_widget.currentRow()
            # Look forward first
            for i in range(current + 1, list_widget.count()):
                if i not in reviewed_rows:
                    list_widget.setCurrentRow(i)
                    _update_preview()
                    return
            # Then look from beginning
            for i in range(0, current):
                if i not in reviewed_rows:
                    list_widget.setCurrentRow(i)
                    _update_preview()
                    return

        def _approve_current():
            """Approve the current suggestion."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            if data.get("_reviewed"):
                _next_unreviewed()
                return

            pid = data["id"]
            location = data["suggested_name"]

            # Set camera_location and clear suggestion
            self.db.update_photo_attributes(photo_id=pid, camera_location=location,
                                           key_characteristics=None)
            cursor = self.db.conn.cursor()
            cursor.execute("UPDATE photos SET suggested_site_id = NULL, suggested_site_confidence = NULL WHERE id = ?", (pid,))
            self.db.conn.commit()

            _mark_reviewed(item, "Approved", location)
            _next_unreviewed()

        def _reject_current():
            """Reject the current suggestion."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            if data.get("_reviewed"):
                _next_unreviewed()
                return

            pid = data["id"]
            location = data["suggested_name"]

            # Clear suggestion
            cursor = self.db.conn.cursor()
            cursor.execute("UPDATE photos SET suggested_site_id = NULL, suggested_site_confidence = NULL WHERE id = ?", (pid,))
            self.db.conn.commit()

            _mark_reviewed(item, "Rejected", location)
            _next_unreviewed()

        def _approve_all_shown():
            """Approve all currently shown unreviewed suggestions."""
            count = 0
            cursor = self.db.conn.cursor()
            for i in range(list_widget.count()):
                if i in reviewed_rows:
                    continue
                item = list_widget.item(i)
                data = item.data(Qt.ItemDataRole.UserRole)
                if data.get("_reviewed"):
                    continue

                pid = data["id"]
                location = data["suggested_name"]

                self.db.update_photo_attributes(photo_id=pid, camera_location=location,
                                               key_characteristics=None)
                cursor.execute("UPDATE photos SET suggested_site_id = NULL, suggested_site_confidence = NULL WHERE id = ?", (pid,))
                _mark_reviewed(item, "Approved", location)
                count += 1

            self.db.conn.commit()
            QMessageBox.information(dlg, "Approved", f"Approved {count} suggestions.")

        def _reject_all_shown():
            """Reject all currently shown unreviewed suggestions."""
            count = 0
            cursor = self.db.conn.cursor()
            for i in range(list_widget.count()):
                if i in reviewed_rows:
                    continue
                item = list_widget.item(i)
                data = item.data(Qt.ItemDataRole.UserRole)
                if data.get("_reviewed"):
                    continue

                pid = data["id"]
                location = data["suggested_name"]
                cursor.execute("UPDATE photos SET suggested_site_id = NULL, suggested_site_confidence = NULL WHERE id = ?", (pid,))
                _mark_reviewed(item, "Rejected", location)
                count += 1

            self.db.conn.commit()
            QMessageBox.information(dlg, "Rejected", f"Rejected {count} suggestions.")

        approve_btn.clicked.connect(_approve_current)
        reject_btn.clicked.connect(_reject_current)
        approve_all_btn.clicked.connect(_approve_all_shown)
        reject_all_btn.clicked.connect(_reject_all_shown)
        close_btn.clicked.connect(dlg.accept)

        # Select first item
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
            _update_preview()

        dlg.exec()

        # Refresh the main UI
        self._populate_camera_locations()
        self._populate_site_filter_options()
        self._populate_photo_list()

    # ─────────────────────────────────────────────────────────────────────
    # Mislabel Detection
    # ─────────────────────────────────────────────────────────────────────

    def find_potential_mislabels(self):
        """Find photos where AI strongly disagrees with human label."""
        if not self.suggester or not self.suggester.ready:
            QMessageBox.warning(self, "AI Not Available", "Species AI model is not loaded.")
            return

        # Get all labeled photos
        species_set = self._species_set()
        all_photos = self.db.get_all_photos()
        labeled_photos = []
        for p in all_photos:
            tags = set(self.db.get_tags(p["id"]))
            species_tags = [t for t in tags if t in species_set and t not in ("Empty", "Unknown", "Verification")]
            if species_tags:
                labeled_photos.append((p, species_tags))

        if not labeled_photos:
            QMessageBox.information(self, "Mislabel Check", "No labeled photos to check.")
            return

        # Progress dialog
        progress = QProgressDialog("Checking for potential mislabels...", "Cancel", 0, len(labeled_photos), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        disagreements = []
        MIN_CONFIDENCE = 0.70  # Only flag if AI is at least 70% confident

        for i, (photo, human_labels) in enumerate(labeled_photos):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            QApplication.processEvents()

            file_path = photo.get("file_path")
            if not file_path or not os.path.exists(file_path):
                continue

            try:
                result = self.suggester.predict(file_path)
                if not result:
                    continue
                ai_label, ai_conf = result

                # Check if AI disagrees with human labels
                if ai_label and ai_label not in human_labels and ai_conf >= MIN_CONFIDENCE:
                    disagreements.append({
                        "photo": photo,
                        "human_labels": human_labels,
                        "ai_label": ai_label,
                        "ai_confidence": ai_conf
                    })
            except Exception as e:
                logger.warning(f"Error checking {file_path}: {e}")
                continue

        progress.setValue(len(labeled_photos))

        if not disagreements:
            QMessageBox.information(self, "Mislabel Check",
                f"Checked {len(labeled_photos)} labeled photos.\n\n"
                "No potential mislabels found - AI agrees with all labels!")
            return

        # Sort by current label (group similar mislabels together)
        disagreements.sort(key=lambda x: (x["human_labels"][0] if x["human_labels"] else "", -x["ai_confidence"]))

        # Create review dialog
        self._show_mislabel_review_dialog(disagreements)

    def _show_mislabel_review_dialog(self, disagreements: list):
        """Show dialog to review potential mislabels."""
        # Load previously reviewed photo IDs
        reviewed_file = os.path.expanduser("~/.trailcam/mislabel_reviewed.json")
        previously_reviewed = set()
        try:
            if os.path.exists(reviewed_file):
                with open(reviewed_file, 'r') as f:
                    data = json.load(f)
                    previously_reviewed = set(data.get('photo_ids', []))
                    print(f"[Mislabel] Loaded {len(previously_reviewed)} previously reviewed photos")
        except Exception:
            pass

        # Filter out already-reviewed photos
        original_count = len(disagreements)
        disagreements = [d for d in disagreements if d['photo']['id'] not in previously_reviewed]
        if original_count != len(disagreements):
            print(f"[Mislabel] Filtered {original_count - len(disagreements)} already-reviewed photos")

        if not disagreements:
            QMessageBox.information(self, "Mislabel Check",
                f"No new potential mislabels found.\n\n"
                f"({len(previously_reviewed)} photos were previously reviewed)")
            return

        # Save queue to file for reference
        queue_backup_file = os.path.expanduser("~/.trailcam/mislabel_review_backup.json")
        try:
            backup_data = {
                'saved_at': datetime.now().isoformat(),
                'total': len(disagreements),
                'items': []
            }
            for item in disagreements:
                backup_data['items'].append({
                    'photo_id': item['photo']['id'],
                    'file_path': item['photo'].get('file_path', ''),
                    'human_labels': item['human_labels'],
                    'ai_label': item['ai_label'],
                    'ai_confidence': float(item['ai_confidence'])
                })
            with open(queue_backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            print(f"[Mislabel] Saved {len(disagreements)} items to {queue_backup_file}")
        except Exception as e:
            print(f"[Mislabel] ERROR saving backup: {e}")

        # Track reviewed photos in this session
        session_reviewed = set()

        def _save_reviewed():
            """Save all reviewed photo IDs to file."""
            all_reviewed = previously_reviewed | session_reviewed
            try:
                with open(reviewed_file, 'w') as f:
                    json.dump({
                        'updated_at': datetime.now().isoformat(),
                        'photo_ids': list(all_reviewed)
                    }, f, indent=2)
                print(f"[Mislabel] Saved {len(all_reviewed)} reviewed photos to {reviewed_file}")
            except Exception as e:
                print(f"[Mislabel] ERROR saving reviewed: {e}")

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Potential Mislabels ({len(disagreements)})")
        dlg.resize(1400, 900)
        layout = QVBoxLayout(dlg)

        # Instructions
        info = QLabel(
            "Photos where AI strongly disagrees with the current label. "
            "Review each and keep, change, or clear the label."
        )
        info.setStyleSheet("color: #aaa; margin-bottom: 10px;")
        layout.addWidget(info)

        # Main content
        content_layout = QHBoxLayout()

        # Left: filter and list
        left_panel = QVBoxLayout()

        # Filter by current label
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter by label:"))
        label_filter = QComboBox()
        label_filter.addItem("All")
        # Get unique labels
        unique_labels = sorted(set(item["human_labels"][0] for item in disagreements if item["human_labels"]))
        for lbl in unique_labels:
            count = sum(1 for item in disagreements if item["human_labels"] and item["human_labels"][0] == lbl)
            label_filter.addItem(f"{lbl} ({count})")
        label_filter.setMinimumWidth(180)
        filter_row.addWidget(label_filter)
        filter_row.addStretch()
        left_panel.addLayout(filter_row)

        left_panel.addWidget(QLabel("Potential mislabels (sorted by current label):"))
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setMinimumWidth(380)

        all_items = disagreements[:]

        def _populate_list():
            list_widget.clear()
            filter_text = label_filter.currentText()
            filter_label = filter_text.split(" (")[0] if " (" in filter_text else filter_text

            for item in all_items:
                human = item["human_labels"][0] if item["human_labels"] else ""
                if filter_label != "All" and human != filter_label:
                    continue

                pid = item["photo"]["id"]
                ai_pct = int(item["ai_confidence"] * 100)
                human_str = ", ".join(item["human_labels"])
                ai = item["ai_label"]
                fname = os.path.basename(item["photo"].get("file_path", ""))

                if pid in session_reviewed:
                    display = f"✓ {fname}"
                    li = QListWidgetItem(display)
                    li.setBackground(QColor(144, 238, 144))
                else:
                    display = f"[{human_str}] → AI: {ai} ({ai_pct}%) - {fname}"
                    li = QListWidgetItem(display)

                li.setData(Qt.ItemDataRole.UserRole, item)
                list_widget.addItem(li)

            remaining = len([item for item in all_items if item["photo"]["id"] not in session_reviewed])
            dlg.setWindowTitle(f"Potential Mislabels ({remaining} remaining)")

        _populate_list()
        label_filter.currentIndexChanged.connect(lambda: _populate_list())
        left_panel.addWidget(list_widget, 1)
        content_layout.addLayout(left_panel)

        # Right: full-size photo preview
        right_panel = QVBoxLayout()
        preview_label = QLabel("Select a photo to preview")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setMinimumSize(800, 600)
        preview_label.setStyleSheet("background: #222; border: 1px solid #444;")
        right_panel.addWidget(preview_label, 1)

        # Info labels
        info_layout = QHBoxLayout()
        current_label_display = QLabel("Current label: —")
        current_label_display.setStyleSheet("font-size: 16px; font-weight: bold;")
        ai_label_display = QLabel("AI suggests: —")
        ai_label_display.setStyleSheet("font-size: 16px; color: #ff9966;")
        info_layout.addWidget(current_label_display)
        info_layout.addStretch()
        info_layout.addWidget(ai_label_display)
        right_panel.addLayout(info_layout)

        content_layout.addLayout(right_panel, 1)
        layout.addLayout(content_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        keep_btn = QPushButton("Keep Current Label (K)")
        keep_btn.setStyleSheet("background: #446; padding: 10px 20px; font-size: 14px;")
        change_btn = QPushButton("Change to AI Suggestion (A)")
        change_btn.setStyleSheet("background: #664; padding: 10px 20px; font-size: 14px;")
        clear_btn = QPushButton("Clear Label (C)")
        clear_btn.setStyleSheet("background: #644; padding: 10px 20px; font-size: 14px;")
        skip_btn = QPushButton("Skip (S)")
        skip_btn.setStyleSheet("padding: 10px 20px; font-size: 14px;")
        btn_layout.addWidget(keep_btn)
        btn_layout.addWidget(change_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(skip_btn)
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("padding: 10px 20px; font-size: 14px;")
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        def _update_preview():
            item = list_widget.currentItem()
            if not item:
                preview_label.clear()
                preview_label.setText("Select a photo")
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            photo = data["photo"]

            # Load FULL SIZE image (not thumbnail)
            path = photo.get("file_path")
            if path and os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    # Scale to fit preview area while maintaining aspect ratio
                    scaled = pixmap.scaled(preview_label.width() - 10, preview_label.height() - 10,
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
                    # Draw annotation boxes on preview
                    photo_id = photo.get("id")
                    if photo_id:
                        try:
                            boxes = self.db.get_annotation_boxes(photo_id)
                            if boxes:
                                from PyQt6.QtGui import QPainter, QPen, QFont
                                # Create a copy to paint on
                                painted = QPixmap(scaled.size())
                                painted.fill(Qt.GlobalColor.transparent)
                                painter = QPainter(painted)
                                painter.drawPixmap(0, 0, scaled)
                                pen = QPen(Qt.GlobalColor.yellow)
                                pen.setWidth(3)
                                painter.setPen(pen)
                                font = QFont()
                                font.setPointSize(12)
                                font.setBold(True)
                                painter.setFont(font)
                                # Calculate scale factors
                                orig_w, orig_h = pixmap.width(), pixmap.height()
                                scale_x = scaled.width() / orig_w if orig_w > 0 else 1
                                scale_y = scaled.height() / orig_h if orig_h > 0 else 1
                                for box in boxes:
                                    x1 = int(box.get('x1', 0) * orig_w * scale_x)
                                    y1 = int(box.get('y1', 0) * orig_h * scale_y)
                                    x2 = int(box.get('x2', 0) * orig_w * scale_x)
                                    y2 = int(box.get('y2', 0) * orig_h * scale_y)
                                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                                    species = box.get('species', '')
                                    if species:
                                        painter.drawText(x1 + 3, y1 + 16, species)
                                painter.end()
                                scaled = painted
                        except Exception as e:
                            print(f"[Mislabel] Failed to draw boxes: {e}")
                            import traceback
                            traceback.print_exc()
                    preview_label.setPixmap(scaled)
            else:
                preview_label.setText("Image not found")

            # Update info
            human = ", ".join(data["human_labels"])
            ai = data["ai_label"]
            ai_pct = int(data["ai_confidence"] * 100)
            current_label_display.setText(f"Current label: {human}")
            ai_label_display.setText(f"AI suggests: {ai} ({ai_pct}% confident)")

        list_widget.currentRowChanged.connect(lambda: _update_preview())

        def _next_unreviewed():
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                data = item.data(Qt.ItemDataRole.UserRole)
                if data["photo"]["id"] not in session_reviewed:
                    list_widget.setCurrentRow(i)
                    return
            # All done
            remaining = len(all_items) - len(reviewed_set)
            if remaining == 0:
                QMessageBox.information(dlg, "Complete", "All potential mislabels reviewed!")

        def _keep_current():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data["photo"]["id"]
            # Log the confirmation to audit trail
            tags = self.db.get_tags(pid)
            if tags:
                self.db.log_tag_confirmation(pid, tags)
            session_reviewed.add(pid)
            print(f"[Mislabel] Added {pid} to session_reviewed, now {len(session_reviewed)} items")
            # Save immediately
            all_reviewed = previously_reviewed | session_reviewed
            try:
                with open(reviewed_file, 'w') as f:
                    json.dump({'updated_at': datetime.now().isoformat(), 'photo_ids': list(all_reviewed)}, f)
                print(f"[Mislabel] Saved {len(all_reviewed)} to {reviewed_file}")
            except Exception as e:
                print(f"[Mislabel] SAVE ERROR: {e}")
            _populate_list()
            _next_unreviewed()

        def _change_to_ai():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data["photo"]["id"]
            ai_suggestion = data["ai_label"]

            # Update tags: remove old species, add AI suggestion
            current_tags = self.db.get_tags(pid)
            species_set = self._species_set()
            new_tags = [t for t in current_tags if t not in species_set]
            new_tags.append(ai_suggestion)
            self.db.update_photo_tags(pid, new_tags)

            session_reviewed.add(pid)
            # Save immediately
            all_reviewed = previously_reviewed | session_reviewed
            try:
                with open(reviewed_file, 'w') as f:
                    json.dump({'updated_at': datetime.now().isoformat(), 'photo_ids': list(all_reviewed)}, f)
                print(f"[Mislabel] Saved {len(all_reviewed)} to {reviewed_file}")
            except Exception as e:
                print(f"[Mislabel] SAVE ERROR: {e}")
            _populate_list()
            _next_unreviewed()

        def _clear_label():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data["photo"]["id"]

            # Remove all species tags, leaving only non-species tags
            current_tags = self.db.get_tags(pid)
            species_set = self._species_set()
            new_tags = [t for t in current_tags if t not in species_set]
            self.db.update_photo_tags(pid, new_tags)

            session_reviewed.add(pid)
            # Save immediately
            all_reviewed = previously_reviewed | session_reviewed
            try:
                with open(reviewed_file, 'w') as f:
                    json.dump({'updated_at': datetime.now().isoformat(), 'photo_ids': list(all_reviewed)}, f)
                print(f"[Mislabel] Saved {len(all_reviewed)} to {reviewed_file}")
            except Exception as e:
                print(f"[Mislabel] SAVE ERROR: {e}")
            _populate_list()
            _next_unreviewed()

        def _skip():
            item = list_widget.currentItem()
            if not item:
                return
            current_row = list_widget.currentRow()
            if current_row < list_widget.count() - 1:
                list_widget.setCurrentRow(current_row + 1)

        keep_btn.clicked.connect(_keep_current)
        change_btn.clicked.connect(_change_to_ai)
        clear_btn.clicked.connect(_clear_label)
        skip_btn.clicked.connect(_skip)
        close_btn.clicked.connect(dlg.accept)

        # Keyboard shortcuts
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("K"), dlg).activated.connect(_keep_current)
        QShortcut(QKeySequence("A"), dlg).activated.connect(_change_to_ai)
        QShortcut(QKeySequence("C"), dlg).activated.connect(_clear_label)
        QShortcut(QKeySequence("S"), dlg).activated.connect(_skip)

        # Select first
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
            _update_preview()

        dlg.exec()

        # Refresh main UI
        self._populate_photo_list()

    # ─────────────────────────────────────────────────────────────────────
    # Claude Review Queue
    # ─────────────────────────────────────────────────────────────────────

    def _update_claude_queue_menu(self):
        """Update the Claude review queue menu item with count."""
        if not hasattr(self, 'claude_review_action'):
            return
        count = self.db.get_claude_review_count()
        if count > 0:
            self.claude_review_action.setText(f"Claude Review Queue ({count})...")
        else:
            self.claude_review_action.setText("Claude Review Queue...")

    def review_claude_queue(self):
        """Review photos flagged by Claude - single photo view with zoomable preview."""
        from PyQt6.QtWidgets import QGraphicsView

        queue = self.db.get_claude_review_queue()
        if not queue:
            QMessageBox.information(self, "Claude Review", "No photos in the Claude review queue.")
            return

        # Enrich queue items with tags and file info
        for item in queue:
            photo_id = item.get("photo_id")
            if photo_id:
                item["_tags"] = self.db.get_tags(photo_id)
            else:
                item["_tags"] = []

        # Create review dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Claude Review Queue ({len(queue)} photos)")
        dlg.resize(1100, 750)
        layout = QVBoxLayout(dlg)

        # Main content: list on left, photo on right
        content_layout = QHBoxLayout()

        # Left side: list of items
        left_panel = QVBoxLayout()

        # Filter by reason
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        reason_filter = QComboBox()
        by_reason = {}
        for item in queue:
            reason = item.get("reason", "Unknown")
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(item)
        reason_filter.addItem(f"All ({len(queue)})", "All")
        for reason, items in sorted(by_reason.items()):
            reason_filter.addItem(f"{reason} ({len(items)})", reason)
        filter_row.addWidget(reason_filter)
        filter_row.addStretch()
        left_panel.addLayout(filter_row)

        left_panel.addWidget(QLabel("Pending review:"))
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setMaximumWidth(320)

        def _populate_list(filter_reason="All"):
            list_widget.clear()
            # Filter items first
            filtered_items = []
            for item in queue:
                if filter_reason != "All" and item.get("reason") != filter_reason:
                    continue
                filtered_items.append(item)
            # Sort by current label (tags)
            filtered_items.sort(key=lambda x: ", ".join(x.get("_tags", [])) or "zzz")
            # Add to list
            for item in filtered_items:
                tags = item.get("_tags", [])
                tag_text = ", ".join(tags) if tags else "(no label)"
                reason = item.get("reason", "")
                name = os.path.basename(item.get("file_path", ""))[:20]
                text = f"{tag_text} → {reason[:20]} - {name}"
                li = QListWidgetItem(text)
                li.setData(Qt.ItemDataRole.UserRole, item)
                list_widget.addItem(li)
            if list_widget.count() > 0:
                list_widget.setCurrentRow(0)

        def _on_filter_changed():
            filter_reason = reason_filter.currentData()
            _populate_list(filter_reason)
            _update_preview()

        reason_filter.currentIndexChanged.connect(_on_filter_changed)
        _populate_list("All")

        left_panel.addWidget(list_widget)
        content_layout.addLayout(left_panel)

        # Right side: photo preview with labels
        right_panel = QVBoxLayout()

        # Current label display (prominent)
        current_label_display = QLabel("Current Label: —")
        current_label_display.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffcc00; padding: 5px; background: #333;")
        right_panel.addWidget(current_label_display)

        # AI suggestion display
        suggestion_label = QLabel("AI Suggests: —")
        suggestion_label.setStyleSheet("font-size: 14px; color: #88ccff; padding: 5px;")
        right_panel.addWidget(suggestion_label)

        # Zoom controls
        zoom_row = QHBoxLayout()
        zoom_label = QLabel("Zoom:")
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(30)
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedWidth(30)
        zoom_fit_btn = QPushButton("Fit")
        zoom_100_btn = QPushButton("100%")
        zoom_row.addWidget(zoom_label)
        zoom_row.addWidget(zoom_out_btn)
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(zoom_fit_btn)
        zoom_row.addWidget(zoom_100_btn)
        zoom_row.addStretch()
        right_panel.addLayout(zoom_row)

        # Graphics view for zoomable image
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.setMinimumSize(700, 500)
        view.setStyleSheet("background: #222; border: 1px solid #444;")
        view.setDragMode(QGraphicsView.DragMode.NoDrag)
        view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        right_panel.addWidget(view, 1)
        content_layout.addLayout(right_panel, 1)

        layout.addLayout(content_layout, 1)

        # Species buttons row
        species_container = QWidget()
        species_layout = QVBoxLayout(species_container)
        species_layout.setContentsMargins(0, 0, 0, 0)
        species_layout.setSpacing(4)
        species_btn_row = QHBoxLayout()

        common_species = ["Deer", "Turkey", "Raccoon", "Squirrel", "Opossum", "Rabbit", "Empty", "Coyote", "Fox", "Person"]
        for sp in common_species:
            btn = QPushButton(sp)
            btn.setStyleSheet("padding: 6px 12px;")
            if sp == "Deer":
                btn.setStyleSheet("padding: 6px 12px; background: #8B4513; color: white;")
            elif sp == "Empty":
                btn.setStyleSheet("padding: 6px 12px; background: #666;")
            species_btn_row.addWidget(btn)
            btn.clicked.connect(lambda checked, s=sp: _apply_species(s))

        species_btn_row.addStretch()
        species_layout.addLayout(species_btn_row)
        layout.addWidget(species_container)

        # Action buttons
        btn_row = QHBoxLayout()
        keep_btn = QPushButton("Keep Current Label (K)")
        keep_btn.setStyleSheet("background: #4488cc; padding: 8px 16px; font-weight: bold;")
        keep_btn.setToolTip("Current label is correct - mark as reviewed")
        clear_btn = QPushButton("Clear Label (C)")
        clear_btn.setStyleSheet("background: #cc6644; padding: 8px 16px;")
        skip_btn = QPushButton("Skip (S)")
        skip_btn.setStyleSheet("padding: 8px 16px;")
        props_btn = QPushButton("Properties (P)")
        props_btn.setStyleSheet("padding: 8px 16px;")
        props_btn.setToolTip("Open this photo in the main window")
        btn_row.addWidget(keep_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(skip_btn)
        btn_row.addWidget(props_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        current_pixmap = [None]
        min_scale = [0.1]
        max_scale = 5.0
        reviewed_items = set()
        navigate_to_photo = [None]  # Store photo to navigate to after dialog closes

        def _get_current_scale():
            return view.transform().m11()

        def _zoom_in():
            current = _get_current_scale()
            if current < max_scale:
                view.scale(1.25, 1.25)

        def _zoom_out():
            current = _get_current_scale()
            if current > min_scale[0] * 1.1:
                view.scale(0.8, 0.8)

        def _zoom_fit():
            if current_pixmap[0] and scene.sceneRect().width() > 0:
                view.resetTransform()
                view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                min_scale[0] = _get_current_scale() * 0.95
                view.scale(0.95, 0.95)

        def _zoom_100():
            view.resetTransform()

        def _update_preview():
            item = list_widget.currentItem()
            if not item:
                scene.clear()
                current_label_display.setText("Current Label: —")
                suggestion_label.setText("AI Suggests: —")
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            path = data.get("file_path")
            photo_id = data.get("photo_id")
            tags = data.get("_tags", [])
            reason = data.get("reason", "")

            # Display current label
            if tags:
                current_label_display.setText(f"Current Label: {', '.join(tags)}")
                current_label_display.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffcc00; padding: 5px; background: #333;")
            else:
                current_label_display.setText("Current Label: (none)")
                current_label_display.setStyleSheet("font-size: 18px; font-weight: bold; color: #888; padding: 5px; background: #333;")

            # Display AI suggestion from reason
            suggestion_label.setText(f"AI Suggests: {reason}" if reason else "AI Suggests: —")

            if path and os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scene.clear()
                    scene.addPixmap(pixmap)
                    scene.setSceneRect(pixmap.rect().toRectF())
                    current_pixmap[0] = pixmap

                    # Draw detection boxes
                    if photo_id:
                        boxes = self.db.get_boxes(photo_id)
                        w = pixmap.width()
                        h = pixmap.height()
                        for b in boxes:
                            if b["y1"] >= 0.95:
                                continue
                            rect = QRectF(b["x1"] * w, b["y1"] * h, (b["x2"] - b["x1"]) * w, (b["y2"] - b["y1"]) * h)
                            lbl = b.get("label", "")
                            if str(lbl) == "ai_deer_head":
                                pen = QPen(Qt.GlobalColor.magenta)
                            elif str(lbl).startswith("ai_"):
                                pen = QPen(Qt.GlobalColor.yellow)
                            elif str(lbl) == "deer_head":
                                pen = QPen(Qt.GlobalColor.red)
                            else:
                                pen = QPen(Qt.GlobalColor.green)
                            pen.setWidth(4)
                            scene.addRect(rect, pen)

                    _zoom_fit()
                else:
                    scene.clear()
                    current_pixmap[0] = None
            else:
                scene.clear()
                current_pixmap[0] = None

        def _mark_reviewed(item, action_text: str):
            """Mark item as reviewed with visual feedback."""
            row = list_widget.row(item)
            reviewed_items.add(row)
            data = item.data(Qt.ItemDataRole.UserRole)
            name = os.path.basename(data.get("file_path", ""))[:20]
            item.setText(f"✓ {action_text} - {name}")
            item.setBackground(QColor(144, 238, 144))
            remaining = list_widget.count() - len(reviewed_items)
            dlg.setWindowTitle(f"Claude Review Queue ({remaining} remaining)")

        def _next_unreviewed():
            """Move to next unreviewed item."""
            current = list_widget.currentRow()
            total = list_widget.count()
            for i in range(current + 1, total):
                if i not in reviewed_items:
                    list_widget.setCurrentRow(i)
                    _update_preview()
                    return
            for i in range(0, current):
                if i not in reviewed_items:
                    list_widget.setCurrentRow(i)
                    _update_preview()
                    return
            # All done
            QMessageBox.information(dlg, "Done", "All photos reviewed!")
            dlg.close()

        def _apply_species(species):
            """Apply a species label and advance."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data.get("photo_id")
            if not pid:
                return
            # Remove old tags and add new one
            old_tags = data.get("_tags", [])
            for tag in old_tags:
                self.db.remove_tag(pid, tag)
            self.db.add_tag(pid, species)
            self.db.set_suggested_tag(pid, None, None)
            self.db.mark_claude_reviewed(pid)
            _mark_reviewed(item, species)
            self._update_claude_queue_menu()
            _next_unreviewed()

        def _keep_label():
            """Keep current label and advance."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data.get("photo_id")
            tags = data.get("_tags", [])
            if not pid:
                return
            # Log the confirmation to audit trail
            if tags:
                self.db.log_tag_confirmation(pid, tags)
            self.db.set_suggested_tag(pid, None, None)
            self.db.mark_claude_reviewed(pid)
            tag_text = ", ".join(tags) if tags else "kept"
            _mark_reviewed(item, tag_text)
            self._update_claude_queue_menu()
            _next_unreviewed()

        def _clear_label():
            """Clear all labels and advance."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data.get("photo_id")
            tags = data.get("_tags", [])
            if not pid:
                return
            for tag in tags:
                self.db.remove_tag(pid, tag)
            self.db.set_suggested_tag(pid, None, None)
            self.db.mark_claude_reviewed(pid)
            _mark_reviewed(item, "cleared")
            self._update_claude_queue_menu()
            _next_unreviewed()

        def _skip():
            """Skip to next without marking reviewed."""
            current = list_widget.currentRow()
            total = list_widget.count()
            if current < total - 1:
                list_widget.setCurrentRow(current + 1)
                _update_preview()

        def _open_properties():
            """Navigate to photo in main window and close review dialog."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            navigate_to_photo[0] = data.get("photo_id")
            dlg.accept()

        # Keyboard shortcuts
        def _key_handler(event):
            key = event.key()
            if key == Qt.Key.Key_K:
                _keep_label()
            elif key == Qt.Key.Key_C:
                _clear_label()
            elif key == Qt.Key.Key_S:
                _skip()
            elif key == Qt.Key.Key_P:
                _open_properties()
            elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
                _zoom_in()
            elif key == Qt.Key.Key_Minus:
                _zoom_out()
            else:
                QDialog.keyPressEvent(dlg, event)
        dlg.keyPressEvent = _key_handler

        # Connect signals
        list_widget.currentItemChanged.connect(lambda: _update_preview())
        zoom_in_btn.clicked.connect(_zoom_in)
        zoom_out_btn.clicked.connect(_zoom_out)
        zoom_fit_btn.clicked.connect(_zoom_fit)
        zoom_100_btn.clicked.connect(_zoom_100)
        keep_btn.clicked.connect(_keep_label)
        clear_btn.clicked.connect(_clear_label)
        skip_btn.clicked.connect(_skip)
        props_btn.clicked.connect(_open_properties)
        close_btn.clicked.connect(dlg.close)

        _update_preview()
        dlg.exec()

        # Handle navigation to specific photo if Properties was clicked
        if navigate_to_photo[0]:
            target_pid = navigate_to_photo[0]
            # Use the dedicated navigation method
            self._navigate_to_photo_by_id(target_pid)

    def open_stamp_reader(self):
        """Open the stamp reader for pattern-based OCR."""
        reader = StampReader(self.db, self)
        reader.exec()

    def open_buck_profiles_list(self):
        """List all buck profiles and open one."""
        ids = sorted({d for d in self._all_deer_ids() if d})
        if not ids:
            QMessageBox.information(self, "Buck Profiles", "No buck profiles found.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Buck Profiles")
        layout = QVBoxLayout(dlg)
        search_box = QLineEdit()
        search_box.setPlaceholderText("Search...")
        layout.addWidget(search_box)
        list_widget = QListWidget()
        for did in ids:
            item = QListWidgetItem(did)
            list_widget.addItem(item)
        layout.addWidget(list_widget)
        btns = QHBoxLayout()
        open_btn = QPushButton("Open Profile")
        close_btn = QPushButton("Close")
        btns.addWidget(open_btn)
        btns.addStretch()
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        def open_selected():
            sel = list_widget.selectedItems()
            if not sel:
                return
            self.deer_id_edit.setCurrentText(sel[0].text())
            self.open_buck_profile()
            dlg.accept()

        def apply_filter(text: str):
            txt = text.lower().strip()
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                visible = txt in item.text().lower()
                item.setHidden(not visible)

        search_box.textChanged.connect(apply_filter)
        open_btn.clicked.connect(open_selected)
        close_btn.clicked.connect(dlg.reject)
        list_widget.itemDoubleClicked.connect(lambda _: open_selected())
        dlg.resize(300, 400)
        dlg.exec()

    def _sync_zoom_slider(self, scale: float):
        percent = int(scale * 100)
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(percent)
        self.zoom_slider.blockSignals(False)

    def _populate_deer_id_dropdown(self):
        """Fill deer ID dropdown from existing IDs."""
        try:
            ids = set()
            for photo in self.db.get_all_photos():
                meta = self.db.get_deer_metadata(photo["id"])
                if meta.get("deer_id"):
                    ids.add(meta["deer_id"])
                for add in self.db.get_additional_deer(photo["id"]):
                    if add.get("deer_id"):
                        ids.add(add["deer_id"])
            sorted_ids = sorted(ids)
        except Exception:
            sorted_ids = []
        current = self.deer_id_edit.currentText() if isinstance(self.deer_id_edit, QComboBox) else ""
        self.deer_id_edit.blockSignals(True)
        self.deer_id_edit.clear()
        self.deer_id_edit.addItem("")
        for did in sorted_ids:
            self.deer_id_edit.addItem(did)
        if current:
            self.deer_id_edit.setCurrentText(current)
        self.deer_id_edit.blockSignals(False)

    # ========== Verification Photo Detection ==========

    def detect_verification_photos(self):
        """Detect verification/test photos based on small file size and no text overlay."""
        import os

        # Get all photos without species tags
        all_photos = self.db.get_all_photos()
        untagged = []
        for p in all_photos:
            tags = self.db.get_tags(p['id'])
            if not tags:  # No species tags
                untagged.append(p)

        if not untagged:
            QMessageBox.information(self, "Verification Detection",
                "No untagged photos found.")
            return

        # Confirmation dialog
        reply = QMessageBox.question(self, "Verification Detection",
            f"This will scan {len(untagged)} untagged photos for verification images.\n\n"
            "Verification photos are identified by:\n"
            "• Very small file size (< 15 KB)\n"
            "• No camera text overlay\n\n"
            "Photos detected as verification will get 'Verification' suggested.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Progress dialog
        progress = QProgressDialog("Detecting verification photos...", "Cancel", 0, len(untagged), self)
        progress.setWindowTitle("Verification Detection")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QCoreApplication.processEvents()

        # Detection criteria
        MAX_SIZE_KB = 15  # Verification photos are typically 6-7 KB

        detected = 0
        checked = 0

        for i, photo in enumerate(untagged):
            if progress.wasCanceled():
                break

            progress.setValue(i + 1)
            progress.setLabelText(f"Checking {i+1}/{len(untagged)}... Found: {detected}")
            QCoreApplication.processEvents()

            path = photo.get('file_path')
            if not path or not os.path.exists(path):
                continue

            checked += 1

            # Check file size
            try:
                size_kb = os.path.getsize(path) / 1024
            except:
                continue

            if size_kb >= MAX_SIZE_KB:
                continue  # Too large to be verification

            # Small file - likely verification photo
            # Set AI suggestion to "Verification"
            self.db.set_suggested_tag(photo['id'], "Verification", 0.95)
            detected += 1

        progress.close()

        # Results
        QMessageBox.information(self, "Verification Detection",
            f"Detection complete!\n\n"
            f"Checked: {checked} photos\n"
            f"Verification photos found: {detected}\n\n"
            "Use 'Review Species Suggestions' to review and confirm.")

        # Refresh
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_photo_list()

    # ========== Site Clustering ==========

    def run_site_clustering(self):
        """Auto-detect camera sites using OCR text overlay detection."""
        try:
            from site_detector import SiteDetector, OCR_AVAILABLE
        except ImportError as e:
            QMessageBox.warning(self, "Site Detection", f"Failed to load site detector: {e}")
            return

        # Show collection selection dialog
        all_photos = self.db.get_all_photos()
        collections = sorted(set(p.get('collection') for p in all_photos if p.get('collection')))

        # Create selection dialog
        select_dlg = QDialog(self)
        select_dlg.setWindowTitle("Auto-Detect Sites")
        select_dlg.setMinimumWidth(400)
        layout = QVBoxLayout(select_dlg)

        layout.addWidget(QLabel("Select collection to detect sites within:"))
        layout.addWidget(QLabel("(Sites will only be matched to locations found in this collection)"))

        collection_combo = QComboBox()
        collection_combo.addItem("All Collections", None)
        for coll in collections:
            # Count photos in collection
            count = len([p for p in all_photos if p.get('collection') == coll])
            collection_combo.addItem(f"{coll} ({count} photos)", coll)

        # Default to current filter if set
        if hasattr(self, "collection_filter_combo"):
            current = self.collection_filter_combo.currentData()
            if current:
                idx = collection_combo.findData(current)
                if idx >= 0:
                    collection_combo.setCurrentIndex(idx)

        layout.addWidget(collection_combo)
        layout.addSpacing(10)

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(select_dlg.reject)
        ok_btn = QPushButton("Continue")
        ok_btn.clicked.connect(select_dlg.accept)
        ok_btn.setDefault(True)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        if select_dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected_collection = collection_combo.currentData()

        # Filter by selected collection
        if selected_collection:
            all_photos = [p for p in all_photos if p.get('collection') == selected_collection]
            collection_msg = f" in collection '{selected_collection}'"
        else:
            collection_msg = ""

        labeled = [p for p in all_photos if (p.get('camera_location') or '').strip()]
        unlabeled = [p for p in all_photos if not (p.get('camera_location') or '').strip()]

        if not labeled:
            QMessageBox.warning(self, "Site Detection",
                f"No labeled photos found{collection_msg}.\n\n"
                "Please label some photos with camera locations first\n"
                "(use the 'Camera Location' field in the photo info panel).")
            return

        # Get unique locations from the filtered photos
        locations = set(p['camera_location'].strip() for p in labeled)

        info_msg = (
            f"Ready to auto-detect sites for {len(unlabeled)} unlabeled photos{collection_msg}.\n\n"
            f"Using {len(labeled)} labeled photos from {len(locations)} sites as reference:\n"
            f"  {', '.join(sorted(locations))}\n\n"
            "Detection methods:\n"
            "1. OCR - reads site name from camera text overlay (most accurate)\n"
            "2. Visual - matches scene appearance (fallback)\n\n"
            "Continue?"
        )
        if QMessageBox.question(self, "Site Detection", info_msg,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                ) != QMessageBox.StandardButton.Yes:
            return

        # Run detection with progress dialog
        progress = QProgressDialog("Detecting sites...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Auto-Detect Sites")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.show()
        QCoreApplication.processEvents()

        cancelled = False

        def progress_cb(current, total, msg):
            nonlocal cancelled
            if progress.wasCanceled():
                cancelled = True
                return
            if total > 0:
                progress.setMaximum(total)
                progress.setValue(current)
            progress.setLabelText(msg)
            QCoreApplication.processEvents()

        try:
            # Create detector
            progress.setLabelText("Loading OCR detection...")
            QCoreApplication.processEvents()

            if not OCR_AVAILABLE:
                progress.close()
                QMessageBox.warning(self, "Site Detection",
                    "OCR not available.\n"
                    "Install pytesseract for site detection.")
                return

            detector = SiteDetector()
            if not detector.ready:
                progress.close()
                QMessageBox.warning(self, "Site Detection",
                    "OCR detector not ready.")
                return

            # Process unlabeled photos
            ocr_count = 0
            failed_count = 0
            by_site = {}

            progress.setMaximum(len(unlabeled))
            for i, photo in enumerate(unlabeled):
                if progress.wasCanceled():
                    cancelled = True
                    break

                path = photo.get('file_path')
                if not path:
                    failed_count += 1
                    continue

                # Update progress every photo to keep UI responsive
                progress.setValue(i + 1)
                progress.setLabelText(f"Processing {i+1}/{len(unlabeled)}... Detected: {ocr_count}")
                QCoreApplication.processEvents()

                result = detector.detect_site(path)

                if result:
                    site_name, confidence = result

                    # Save suggestion to database
                    site = self.db.get_site_by_name(site_name)
                    if site:
                        self.db.set_photo_site_suggestion(photo['id'], site['id'], confidence)
                    else:
                        site_id = self.db.create_site(site_name, confirmed=False)
                        self.db.set_photo_site_suggestion(photo['id'], site_id, confidence)

                    by_site[site_name] = by_site.get(site_name, 0) + 1
                    ocr_count += 1
                else:
                    failed_count += 1

            progress.close()

            if cancelled:
                QMessageBox.information(self, "Site Detection", "Detection cancelled.")
                return

            # Show results
            msg = (f"Site detection complete!\n\n"
                   f"Detected via OCR: {ocr_count}\n"
                   f"Could not detect: {failed_count}\n\n")

            if by_site:
                msg += "By site:\n"
                for site, count in sorted(by_site.items()):
                    msg += f"  {site}: {count}\n"
                msg += "\n"

            msg += "Go to Tools → Manage Sites to review suggestions."
            QMessageBox.information(self, "Site Detection", msg)

            # Refresh UI
            self._populate_site_filter_options()
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()

        except Exception as e:
            progress.close()
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Site Detection", f"Detection failed: {e}")

    def manage_sites(self):
        """Open dialog to view/rename/confirm/reject sites."""
        sites = self.db.get_all_sites()
        if not sites:
            QMessageBox.information(self, "Manage Sites",
                "No sites found. Run 'Auto-Detect Sites' first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Manage Sites")
        dlg.setMinimumSize(700, 450)
        layout = QVBoxLayout(dlg)

        # Site list
        site_list = QListWidget()
        site_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        def refresh_site_list():
            site_list.clear()
            sites = self.db.get_all_sites()
            for site in sites:
                confirmed = site.get("confirmed", 1)
                count = site.get("photo_count", 0)
                suggested = site.get("suggested_count", 0)
                if confirmed:
                    label = f"✓ {site['name']} ({count} photos)"
                else:
                    label = f"⭐ {site['name']} ({suggested} suggested photos)"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, site["id"])
                item.setData(Qt.ItemDataRole.UserRole + 1, confirmed)  # Store confirmed status
                site_list.addItem(item)

        refresh_site_list()
        layout.addWidget(QLabel("Sites (✓ = confirmed, ⭐ = suggested):"))
        layout.addWidget(site_list)

        # Actions row
        btn_row = QHBoxLayout()
        confirm_btn = QPushButton("✓ Confirm Site")
        confirm_btn.setStyleSheet("background-color: #90EE90;")
        reject_btn = QPushButton("✗ Reject Suggestion")
        reject_btn.setStyleSheet("background-color: #FFB6C1;")
        rename_btn = QPushButton("Rename")
        view_btn = QPushButton("View Photos")
        delete_btn = QPushButton("Delete Site")
        btn_row.addWidget(confirm_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addWidget(rename_btn)
        btn_row.addWidget(view_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        def confirm_site_suggestion():
            item = site_list.currentItem()
            if not item:
                return
            site_id = item.data(Qt.ItemDataRole.UserRole)
            confirmed = item.data(Qt.ItemDataRole.UserRole + 1)
            if confirmed:
                QMessageBox.information(dlg, "Already Confirmed",
                    "This site is already confirmed.")
                return
            site = self.db.get_site(site_id)
            if not site:
                return
            reply = QMessageBox.question(
                dlg, "Confirm Site",
                f"Confirm '{site['name']}' as a real site?\n\nAll suggested photos will be assigned to this site.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.db.confirm_site(site_id)
                refresh_site_list()
                self._populate_site_filter_options()
                self.photos = self._sorted_photos(self.db.get_all_photos())
                self._populate_photo_list()

        def reject_site_suggestion():
            item = site_list.currentItem()
            if not item:
                return
            site_id = item.data(Qt.ItemDataRole.UserRole)
            confirmed = item.data(Qt.ItemDataRole.UserRole + 1)
            if confirmed:
                QMessageBox.information(dlg, "Cannot Reject",
                    "This site is already confirmed. Use 'Delete' to remove it.")
                return
            site = self.db.get_site(site_id)
            if not site:
                return
            reply = QMessageBox.question(
                dlg, "Reject Suggestion",
                f"Reject '{site['name']}'?\n\nSuggested photos will be marked as unassigned.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.db.reject_site_suggestion(site_id)
                refresh_site_list()
                self._populate_site_filter_options()
                self.photos = self._sorted_photos(self.db.get_all_photos())
                self._populate_photo_list()

        def rename_site():
            item = site_list.currentItem()
            if not item:
                return
            site_id = item.data(Qt.ItemDataRole.UserRole)
            site = self.db.get_site(site_id)
            if not site:
                return
            new_name, ok = QInputDialog.getText(
                dlg, "Rename Site", "New name:",
                text=site["name"]
            )
            if ok and new_name.strip():
                self.db.update_site(site_id, name=new_name.strip())
                refresh_site_list()
                self._populate_site_filter_options()

        def view_photos():
            item = site_list.currentItem()
            if not item:
                return
            site_id = item.data(Qt.ItemDataRole.UserRole)
            dlg.accept()
            # Set site filter and refresh
            idx = self.site_filter_combo.findData(site_id)
            if idx >= 0:
                self.site_filter_combo.setCurrentIndex(idx)

        def delete_site():
            item = site_list.currentItem()
            if not item:
                return
            site_id = item.data(Qt.ItemDataRole.UserRole)
            site = self.db.get_site(site_id)
            if not site:
                return
            confirm = QMessageBox.question(
                dlg, "Delete Site",
                f"Delete '{site['name']}'?\n\nPhotos will be marked as unassigned.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self.db.delete_site(site_id)
                refresh_site_list()
                self._populate_site_filter_options()

        confirm_btn.clicked.connect(confirm_site_suggestion)
        reject_btn.clicked.connect(reject_site_suggestion)
        rename_btn.clicked.connect(rename_site)
        view_btn.clicked.connect(view_photos)
        delete_btn.clicked.connect(delete_site)

        dlg.exec()

    def manage_cameras(self):
        """Open dialog to view/create/rename/delete cameras."""
        username = self._get_current_username()

        dlg = QDialog(self)
        dlg.setWindowTitle("Manage Cameras")
        dlg.setMinimumSize(700, 450)
        layout = QVBoxLayout(dlg)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Name", "Owner", "Verified", "Permissions"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        def refresh_table():
            cameras = self.db.get_all_cameras()
            table.setRowCount(len(cameras))
            for row, cam in enumerate(cameras):
                name_item = QTableWidgetItem(cam.get("name", ""))
                owner_item = QTableWidgetItem(cam.get("owner", ""))
                verified_item = QTableWidgetItem("Yes" if cam.get("verified") else "No")
                perm_count = self.db.get_camera_permission_count(cam.get("id"))
                perm_item = QTableWidgetItem(str(perm_count))
                for col, item in enumerate([name_item, owner_item, verified_item, perm_item]):
                    item.setData(Qt.ItemDataRole.UserRole, cam)
                    table.setItem(row, col, item)
            table.resizeColumnsToContents()

        refresh_table()
        layout.addWidget(table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Camera")
        edit_btn = QPushButton("Edit")
        delete_btn = QPushButton("Delete")
        close_btn = QPushButton("Close")
        btn_row.addWidget(add_btn)
        btn_row.addWidget(edit_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        def _get_selected_camera():
            row = table.currentRow()
            if row < 0:
                return None
            item = table.item(row, 0)
            if not item:
                return None
            return item.data(Qt.ItemDataRole.UserRole)

        def _add_camera():
            name, ok = QInputDialog.getText(dlg, "Add Camera", "Camera name:")
            if ok and name.strip():
                self.db.create_camera(name.strip(), username)
                refresh_table()

        def _edit_camera():
            cam = _get_selected_camera()
            if not cam:
                return
            if cam.get("owner") != username:
                QMessageBox.information(dlg, "Not Allowed", "Only the camera owner can edit.")
                return
            new_name, ok = QInputDialog.getText(
                dlg, "Rename Camera", "New name:", text=cam.get("name", "")
            )
            if ok and new_name.strip():
                self.db.rename_camera(cam["id"], new_name.strip())
                refresh_table()

        def _delete_camera():
            cam = _get_selected_camera()
            if not cam:
                return
            if cam.get("owner") != username:
                QMessageBox.information(dlg, "Not Allowed", "Only the camera owner can delete.")
                return
            confirm = QMessageBox.question(
                dlg, "Delete Camera",
                f"Delete '{cam.get('name', '')}'?\n\nThis will remove permissions and pending suggestions for this camera.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self.db.delete_camera(cam["id"])
                refresh_table()

        add_btn.clicked.connect(_add_camera)
        edit_btn.clicked.connect(_edit_camera)
        delete_btn.clicked.connect(_delete_camera)
        close_btn.clicked.connect(dlg.accept)

        dlg.exec()

    # ====== PHOTO STORAGE LOCATIONS ======

    def _manage_photo_locations(self):
        """Manage folders where photos are stored (no import/copy needed)."""
        from PyQt6.QtWidgets import QDialog, QListWidget, QFileDialog

        settings = QSettings("TrailCam", "Trainer")
        locations = settings.value("photo_locations", []) or []
        if isinstance(locations, str):
            locations = [locations] if locations else []

        dlg = QDialog(self)
        dlg.setWindowTitle("Photo Storage Locations")
        dlg.setMinimumSize(500, 400)
        layout = QVBoxLayout(dlg)

        # Instructions
        info = QLabel(
            "Add folders containing your trail camera photos.\n"
            "Photos will be added to the database without copying.\n"
            "The app will scan these folders for new photos."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # List of locations
        list_widget = QListWidget()
        for loc in locations:
            list_widget.addItem(loc)
        layout.addWidget(list_widget)

        # Buttons
        btn_layout = QHBoxLayout()

        add_btn = QPushButton("Add Folder...")
        remove_btn = QPushButton("Remove Selected")
        scan_btn = QPushButton("Scan for New Photos")

        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(scan_btn)
        layout.addLayout(btn_layout)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        def add_folder():
            folder = QFileDialog.getExistingDirectory(
                dlg, "Select Photo Folder", str(Path.home())
            )
            if folder and folder not in locations:
                locations.append(folder)
                list_widget.addItem(folder)
                settings.setValue("photo_locations", locations)

        def remove_selected():
            current = list_widget.currentItem()
            if current:
                path = current.text()
                if path in locations:
                    locations.remove(path)
                    settings.setValue("photo_locations", locations)
                list_widget.takeItem(list_widget.row(current))

        def scan_folders():
            if not locations:
                QMessageBox.information(dlg, "Scan", "No folders configured. Add a folder first.")
                return

            # Scan all configured folders for photos
            from PyQt6.QtWidgets import QProgressDialog

            progress = QProgressDialog("Scanning folders...", "Cancel", 0, 0, dlg)
            progress.setWindowTitle("Scanning for Photos")
            progress.setMinimumDuration(0)
            progress.show()

            added_count = 0
            skipped_count = 0
            photo_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'}

            for folder in locations:
                if progress.wasCanceled():
                    break

                folder_path = Path(folder)
                if not folder_path.exists():
                    continue

                progress.setLabelText(f"Scanning: {folder}")
                QApplication.processEvents()

                # Find all image files
                for file_path in folder_path.rglob("*"):
                    if progress.wasCanceled():
                        break

                    # Skip temp folders (CuddeLink downloads, thumbnails, etc.)
                    if '.cuddelink_tmp' in str(file_path) or '.thumbnails' in str(file_path):
                        continue

                    if file_path.suffix.lower() in photo_extensions:
                        # Check if already in database by path
                        existing = self.db.get_photo_by_path(str(file_path))
                        if existing:
                            skipped_count += 1
                            continue

                        # Compute hash and check for duplicate content
                        file_hash = self._hash_file(file_path)
                        if file_hash and self.db.photo_exists_by_hash(file_hash):
                            skipped_count += 1
                            continue

                        # Add to database without copying
                        try:
                            from image_processor import extract_exif_data

                            date_taken, camera_model = extract_exif_data(str(file_path))
                            photo_id = self.db.add_photo(
                                file_path=str(file_path),
                                original_name=file_path.name,
                                date_taken=date_taken,
                                camera_model=camera_model,
                                thumbnail_path=None,
                                file_hash=file_hash
                            )
                            if photo_id:
                                added_count += 1

                                # Update progress every 10 photos
                                if added_count % 10 == 0:
                                    progress.setLabelText(f"Added {added_count} photos...")
                                    QApplication.processEvents()

                        except Exception as e:
                            logger.warning(f"Failed to add {file_path}: {e}")

            progress.close()

            # Refresh photo list and collection dropdown
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_collection_filter_options()  # Refresh collection dropdown
            self._populate_photo_list()
            if self.photos:
                self.index = 0
                self.load_photo()

            QMessageBox.information(
                dlg,
                "Scan Complete",
                f"Added {added_count} new photos.\n"
                f"Skipped {skipped_count} already in database."
            )

        add_btn.clicked.connect(add_folder)
        remove_btn.clicked.connect(remove_selected)
        scan_btn.clicked.connect(scan_folders)

        dlg.exec()

    # ====== UPDATE & ABOUT ======

    def _check_for_updates(self):
        """Check GitHub for software updates."""
        try:
            # Show checking message
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            update_info = updater.check_for_updates()

            if update_info is None:
                QApplication.restoreOverrideCursor()
                QMessageBox.information(
                    self,
                    "Check for Updates",
                    f"You're running the latest version ({__version__})."
                )
                return

            # Check for delta update possibility
            delta_info = updater.check_delta_update(update_info)
            QApplication.restoreOverrideCursor()

            # New version available
            version = update_info.get("version", "unknown")
            notes = update_info.get("release_notes", "")
            download_url = update_info.get("download_url")
            html_url = update_info.get("html_url", "")

            msg = f"A new version is available!\n\n"
            msg += f"Current version: {__version__}\n"
            msg += f"New version: {version}\n"

            # Show delta update info if available
            if delta_info:
                download_size = delta_info.get("download_size", 0)
                file_count = delta_info.get("download_count", 0)
                size_mb = download_size / 1024 / 1024
                msg += f"\nDelta update: {file_count} files changed ({size_mb:.1f} MB)\n"
                update_info["delta_info"] = delta_info

            if notes:
                # Truncate long release notes
                if len(notes) > 500:
                    notes = notes[:500] + "..."
                msg += f"\nRelease notes:\n{notes}\n"

            if download_url:
                reply = QMessageBox.question(
                    self,
                    "Update Available",
                    msg + "\nWould you like to download and install the update?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self._download_and_install_update(update_info)
            else:
                # No direct download - open release page
                msg += f"\nVisit the release page to download:\n{html_url}"
                QMessageBox.information(self, "Update Available", msg)
                if html_url:
                    import webbrowser
                    webbrowser.open(html_url)

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self,
                "Update Check Failed",
                f"Could not check for updates:\n{str(e)}"
            )

    def _download_and_install_update(self, update_info: dict):
        """Download and install the update (using delta if available)."""
        download_url = update_info.get("download_url")
        if not download_url:
            QMessageBox.warning(self, "Update Error", "No download URL available.")
            return

        delta_info = update_info.get("delta_info")
        is_delta = delta_info is not None

        try:
            # Create progress dialog
            from PyQt6.QtWidgets import QProgressDialog

            label = "Downloading delta update..." if is_delta else "Downloading update..."
            progress = QProgressDialog(label, "Cancel", 0, 100, self)
            progress.setWindowTitle("Downloading Update")
            progress.setMinimumDuration(0)
            progress.setValue(0)

            def update_progress(downloaded, total):
                if total > 0:
                    pct = int(downloaded / total * 100)
                    progress.setValue(pct)
                    progress.setLabelText(f"Downloading... {downloaded // 1024} / {total // 1024} KB")
                QApplication.processEvents()

            # Download (delta or full)
            if is_delta:
                extract_dir = updater.download_delta_update(download_url, delta_info, update_progress)
            else:
                update_file = updater.download_update(download_url, update_progress)
            progress.close()

            # Build install message
            if is_delta:
                file_count = delta_info.get("download_count", 0)
                install_msg = f"Download complete. Install {file_count} changed files now?\n\n"
            else:
                install_msg = "Download complete. Install now?\n\n"
            install_msg += "The application will restart after installation."

            # Confirm install
            reply = QMessageBox.question(
                self,
                "Install Update",
                install_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Install (delta or full)
                if is_delta:
                    success = updater.install_delta_update(extract_dir, delta_info)
                else:
                    success = updater.install_update(update_file)

                if success:
                    QMessageBox.information(
                        self,
                        "Update",
                        "Update is being installed. The application will restart."
                    )
                    QApplication.quit()
                else:
                    QMessageBox.warning(
                        self,
                        "Update Error",
                        "Failed to install update. Please try again or install manually."
                    )

        except Exception as e:
            QMessageBox.warning(
                self,
                "Update Error",
                f"Failed to download/install update:\n{str(e)}"
            )

    # === CLOUD SYNC METHODS ===

    def show_cloud_status(self):
        """Show cloud storage status dialog."""
        try:
            from r2_storage import R2Storage

            storage = R2Storage()
            username = user_config.get_username() or "not set"

            if not storage.is_configured():
                QMessageBox.information(
                    self,
                    "Cloud Status",
                    "Cloud storage (Cloudflare R2) is not configured.\n\n"
                    "See PLAN.md for setup instructions."
                )
                return

            stats = storage.get_bucket_stats()

            if "error" in stats:
                QMessageBox.warning(
                    self,
                    "Cloud Status",
                    f"Error connecting to cloud:\n{stats['error']}"
                )
                return

            # Count files in shared structure
            # Photos and thumbnails are now stored in shared paths, not per-user
            status_text = f"""
<h3>Cloud Storage Status</h3>
<p><b>Bucket:</b> trailcam-photos</p>
<hr>
<p><b>Total in bucket:</b></p>
<ul>
<li>Objects: {stats.get('object_count', 0)}</li>
<li>Size: {stats.get('total_size_mb', 0):.1f} MB</li>
</ul>
<hr>
<p><b>Free tier:</b> 10 GB storage, unlimited downloads</p>
"""
            msg = QMessageBox(self)
            msg.setWindowTitle("Cloud Status")
            msg.setTextFormat(Qt.TextFormat.RichText)
            msg.setText(status_text)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.exec()

        except ImportError:
            QMessageBox.warning(
                self,
                "Cloud Status",
                "Cloud storage module not found.\n"
                "Make sure r2_storage.py exists and boto3 is installed."
            )
        except Exception as e:
            QMessageBox.warning(self, "Cloud Status", f"Error: {str(e)}")

    def upload_to_cloud(self, thumbnails_only: bool = True):
        """Upload photos to cloud storage."""
        try:
            from r2_storage import R2Storage

            storage = R2Storage()
            if not storage.is_configured():
                QMessageBox.warning(
                    self,
                    "Cloud Upload",
                    "Cloud storage is not configured.\n"
                    "See PLAN.md for setup instructions."
                )
                return

            username = user_config.get_username()
            if not username:
                QMessageBox.warning(
                    self,
                    "Cloud Upload",
                    "Username not set. Please set your username first."
                )
                return

            # Get photos (excludes archived by default)
            photos = self.db.get_all_photos()
            if not photos:
                QMessageBox.information(self, "Cloud Upload", "No photos to upload.")
                return

            # Confirm
            mode = "thumbnails only" if thumbnails_only else "full photos + thumbnails"
            reply = QMessageBox.question(
                self,
                "Cloud Upload",
                f"Upload {len(photos)} photos ({mode}) as '{username}'?\n\n"
                "This may take a while on slow connections.\n"
                "You can cancel by closing the progress dialog.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            # Progress dialog
            from PyQt6.QtWidgets import QProgressDialog
            progress = QProgressDialog(
                "Uploading to cloud...", "Cancel", 0, len(photos), self
            )
            progress.setWindowTitle("Cloud Upload")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()

            uploaded = 0
            skipped = 0
            errors = 0

            for i, photo in enumerate(photos):
                if progress.wasCanceled():
                    break

                file_hash = photo.get('file_hash')
                if not file_hash:
                    errors += 1
                    continue

                progress.setValue(i)
                progress.setLabelText(f"Uploading {i+1}/{len(photos)}...")
                QApplication.processEvents()

                # Upload thumbnail using actual path from database
                # Shared structure: thumbnails/{hash}_thumb.jpg (no username prefix)
                thumb_path_str = photo.get('thumbnail_path')
                if thumb_path_str:
                    thumb_path = Path(thumb_path_str)
                    if thumb_path.exists():
                        r2_key = f"thumbnails/{file_hash}_thumb.jpg"
                        if not storage.check_exists(r2_key):
                            if storage.upload_file(thumb_path, r2_key):
                                uploaded += 1
                            else:
                                errors += 1
                        else:
                            skipped += 1

                # Upload full photo if requested
                # Shared structure: photos/{hash}.jpg (no username prefix)
                if not thumbnails_only:
                    photo_path = Path(photo['file_path'])
                    if photo_path.exists():
                        r2_key = f"photos/{file_hash}.jpg"
                        if not storage.check_exists(r2_key):
                            if storage.upload_file(photo_path, r2_key):
                                uploaded += 1
                            else:
                                errors += 1
                        else:
                            skipped += 1

            progress.close()

            QMessageBox.information(
                self,
                "Cloud Upload Complete",
                f"Upload finished!\n\n"
                f"Uploaded: {uploaded}\n"
                f"Skipped (already exists): {skipped}\n"
                f"Errors: {errors}"
            )

        except ImportError:
            QMessageBox.warning(
                self,
                "Cloud Upload",
                "Cloud storage module not found.\n"
                "Make sure r2_storage.py exists and boto3 is installed."
            )
        except Exception as e:
            QMessageBox.warning(self, "Cloud Upload", f"Error: {str(e)}")

    def change_username(self):
        """Change the username."""
        current = user_config.get_username() or ""
        new_username, ok = QInputDialog.getText(
            self,
            "Change Username",
            "Enter new username:",
            QLineEdit.EchoMode.Normal,
            current
        )
        if ok and new_username and len(new_username.strip()) >= 2:
            user_config.set_username(new_username.strip())
            QMessageBox.information(
                self,
                "Username Changed",
                f"Username changed to: {new_username.strip()}\n\n"
                "Note: Photos already uploaded will remain under the old username."
            )

    def show_cloud_admin(self):
        """Show admin panel with all users and their storage."""
        try:
            from r2_storage import R2Storage

            storage = R2Storage()
            if not storage.is_configured():
                QMessageBox.warning(
                    self,
                    "Cloud Admin",
                    "Cloud storage is not configured.\n"
                    "See PLAN.md for setup instructions."
                )
                return

            # Show loading message
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()

            # Get all objects
            objects = []
            try:
                paginator = storage.client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=storage.bucket_name):
                    for obj in page.get('Contents', []):
                        objects.append(obj)
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "Cloud Admin", f"Error: {str(e)}")
                return

            QApplication.restoreOverrideCursor()

            # Analyze by user
            users = {}
            total_size = 0

            for obj in objects:
                key = obj['Key']
                size = obj['Size']
                total_size += size

                parts = key.split('/')
                if len(parts) >= 2 and parts[0] == 'users':
                    username = parts[1]
                    if username not in users:
                        users[username] = {'photos': 0, 'thumbnails': 0, 'size': 0}
                    users[username]['size'] += size
                    if '/photos/' in key:
                        users[username]['photos'] += 1
                    elif '/thumbnails/' in key:
                        users[username]['thumbnails'] += 1

            # Format size helper
            def fmt_size(b):
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if b < 1024:
                        return f"{b:.1f} {unit}"
                    b /= 1024
                return f"{b:.1f} TB"

            # Build HTML table
            if not users:
                user_rows = "<tr><td colspan='4' style='text-align:center; color:#888;'>No users yet</td></tr>"
            else:
                user_rows = ""
                for username in sorted(users.keys()):
                    data = users[username]
                    user_rows += f"""
                    <tr>
                        <td style='padding:8px;'><b>{username}</b></td>
                        <td style='padding:8px; text-align:right;'>{data['photos']}</td>
                        <td style='padding:8px; text-align:right;'>{data['thumbnails']}</td>
                        <td style='padding:8px; text-align:right;'>{fmt_size(data['size'])}</td>
                    </tr>
                    """

            free_remaining = 10 * 1024 * 1024 * 1024 - total_size  # 10GB free tier

            admin_html = f"""
            <h2>Cloud Admin Panel</h2>
            <hr>
            <h3>Storage Summary</h3>
            <p><b>Total objects:</b> {len(objects)}</p>
            <p><b>Total size:</b> {fmt_size(total_size)}</p>
            <p><b>Free tier remaining:</b> {fmt_size(free_remaining)}</p>
            <hr>
            <h3>Users ({len(users)})</h3>
            <table border='0' cellspacing='0' style='width:100%;'>
                <tr style='background:#333;'>
                    <th style='padding:8px; text-align:left;'>Username</th>
                    <th style='padding:8px; text-align:right;'>Photos</th>
                    <th style='padding:8px; text-align:right;'>Thumbnails</th>
                    <th style='padding:8px; text-align:right;'>Size</th>
                </tr>
                {user_rows}
            </table>
            <hr>
            <p style='color:#888; font-size:0.9em;'>Free tier: 10 GB storage, unlimited downloads</p>
            """

            # Show in dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Cloud Admin")
            dialog.setMinimumSize(500, 400)

            layout = QVBoxLayout(dialog)

            from PyQt6.QtWidgets import QTextBrowser
            browser = QTextBrowser()
            browser.setHtml(admin_html)
            browser.setOpenExternalLinks(True)
            layout.addWidget(browser)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)

            dialog.exec()

        except ImportError:
            QMessageBox.warning(
                self,
                "Cloud Admin",
                "Cloud storage module not found.\n"
                "Make sure r2_storage.py exists and boto3 is installed."
            )
        except Exception as e:
            QMessageBox.warning(self, "Cloud Admin", f"Error: {str(e)}")

    def _show_about(self):
        """Show about dialog."""
        about_text = f"""
<h2>Trail Camera Software</h2>
<p><b>Version:</b> {__version__}</p>
<p>A desktop application for organizing trail camera photos with AI-assisted wildlife labeling, deer tracking, and custom model training.</p>
<p><b>Features:</b></p>
<ul>
<li>AI-powered species identification</li>
<li>Buck/doe classification</li>
<li>Individual deer tracking</li>
<li>Antler point counting</li>
<li>Cloud sync via Supabase</li>
</ul>
<p><b>GitHub:</b> <a href="https://github.com/cabbp3/Trail-Camera-Software">github.com/cabbp3/Trail-Camera-Software</a></p>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle("About Trail Camera Software")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(about_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()


def _add_user_to_registry(username: str):
    """Add a new user to the user registry if not already present."""
    import json
    registry_path = Path.home() / '.trailcam' / 'user_registry.json'

    try:
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
        else:
            registry = {'users': []}

        # Check if user already exists
        existing = [u for u in registry['users'] if u.get('username') == username]
        if not existing:
            registry['users'].append({
                'username': username,
                'is_admin': False,
                'status': 'active'
            })
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Added new user to registry: {username}")
    except Exception as e:
        logger.warning(f"Failed to update user registry: {e}")


def main():
    app = QApplication(sys.argv)
    # Look for icon in standard locations
    parent_dir = Path(__file__).resolve().parent.parent
    icon_candidates = [
        parent_dir / "icon.png",
        parent_dir / "app_icon.png",
        parent_dir / "ChatGPT Image Dec 5, 2025, 07_07_24 PM.png",  # Legacy fallback
    ]
    for icon_path in icon_candidates:
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            break
    app.setStyleSheet(APP_STYLE)

    # Check for username on first launch
    username = user_config.get_username()
    if not username:
        username, ok = QInputDialog.getText(
            None,
            "Welcome to Trail Camera Software",
            "Please enter your name.\n"
            "This identifies your photos when syncing to the cloud.",
            QLineEdit.EchoMode.Normal,
            ""
        )
        if ok and username and len(username.strip()) >= 2:
            username = username.strip()
            user_config.set_username(username)
            _add_user_to_registry(username)
        else:
            # User cancelled or invalid - use default
            user_config.set_username("user")
            username = "user"

    win = TrainerWindow()
    # Show username in window title
    win.setWindowTitle(f"TrailCam - {username}")
    win.resize(1280, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
