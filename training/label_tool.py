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
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal, QCoreApplication, QSettings, QPoint, QRect, QSize, QThread, QDate
from PyQt6.QtGui import QPixmap, QIcon, QAction, QPen, QShortcut, QKeySequence, QColor
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
    QLayout,
    QWidgetItem,
    QDateEdit,
)
import PyQt6  # used for plugin path detection
import sysconfig

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database import TrailCamDatabase  # noqa: E402
from preview_window import ImageGraphicsView  # reuse zoom/pan behavior


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
from image_processor import import_photo, create_thumbnail
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


SPECIES_OPTIONS = ["", "Coyote", "Deer", "Empty", "Person", "Raccoon", "Turkey", "Unknown", "Vehicle"]
SEX_TAGS = {"buck", "doe"}

# Master list of valid species labels - ONLY these can be written to labels.txt
# Includes both simplified model categories AND detailed species for manual tagging
VALID_SPECIES = {
    "Bobcat", "Coyote", "Deer", "Empty", "Opossum", "Other", "Other Bird",
    "Other Mammal", "Person", "Quail", "Rabbit", "Raccoon", "Squirrel",
    "Turkey", "Unknown", "Vehicle"
}
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
    photo_processed = pyqtSignal(dict)  # {id, species, species_conf, sex, sex_conf}
    finished_all = pyqtSignal(int, int)  # species_count, sex_count
    error = pyqtSignal(str)  # error message

    def __init__(self, photos: list, db, suggester, detector_getter, parent=None):
        super().__init__(parent)
        self.photos = photos
        self.db = db
        self.suggester = suggester
        self._get_detector = detector_getter
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """Run AI processing in background thread."""
        total = len(self.photos)
        species_count = 0
        sex_count = 0

        # Get detector (may need to load model)
        try:
            detector, names = self._get_detector()
        except Exception as e:
            self.error.emit(f"Failed to load detector: {e}")
            return

        for i, p in enumerate(self.photos):
            if self._cancelled:
                break

            # Emit progress
            self.progress.emit(i, total, f"Processing {i + 1} of {total}...")

            result = {
                "id": p.get("id"),
                "species": None,
                "species_conf": 0,
                "sex": None,
                "sex_conf": 0
            }

            try:
                # Run detection if no boxes exist
                self._ensure_detection_boxes(p, detector, names)

                # Check detection boxes and auto-classify based on MegaDetector
                boxes = self.db.get_boxes(p["id"]) if p.get("id") else []
                has_person = any(b.get("label") == "ai_person" for b in boxes)
                has_vehicle = any(b.get("label") == "ai_vehicle" for b in boxes)
                has_animal = any(b.get("label") in ("ai_animal", "ai_subject", "subject") for b in boxes)

                label = None
                conf = None

                if has_person:
                    # MegaDetector found person - auto-classify as Person
                    label, conf = "Person", 0.95
                elif has_vehicle:
                    # MegaDetector found vehicle - auto-classify as Vehicle
                    label, conf = "Vehicle", 0.95
                elif has_animal:
                    # Run classifier on animal crop
                    crop = self._best_crop_for_photo(p)
                    path = str(crop) if crop else p.get("file_path")
                    res = self.suggester.predict(path)
                    if res:
                        label, conf = res
                        # If classifier says Empty but detector found animals, use Unknown
                        # Also convert "Other" to "Unknown" - Other is for manual entry only
                        if label in ("Empty", "Other"):
                            label = "Unknown"
                            conf = 0.5
                    if crop:
                        try:
                            Path(crop).unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    # No detections at all - suggest Empty
                    label, conf = "Empty", 0.95

                if label:
                    self.db.set_suggested_tag(p["id"], label, conf)
                    result["species"] = label
                    result["species_conf"] = conf
                    species_count += 1

                    # If Deer, also suggest buck/doe
                    if label == "Deer":
                                self._add_deer_head_boxes(p, detector, names)
                                if self.suggester.buckdoe_ready:
                                    head_crop = self._best_head_crop_for_photo(p)
                                    if head_crop:
                                        sex_res = self.suggester.predict_sex(str(head_crop))
                                        if sex_res:
                                            sex_label, sex_conf = sex_res
                                            self.db.set_suggested_sex(p["id"], sex_label, sex_conf)
                                            result["sex"] = sex_label
                                            result["sex_conf"] = sex_conf
                                            sex_count += 1
                                        try:
                                            Path(head_crop).unlink(missing_ok=True)
                                        except Exception:
                                            pass

                    if crop:
                        try:
                            Path(crop).unlink(missing_ok=True)
                        except Exception:
                            pass

            except Exception as e:
                logger.warning(f"AI processing failed for photo {p.get('id')}: {e}")

            # Emit result for this photo
            if result["species"]:
                self.photo_processed.emit(result)

        self.finished_all.emit(species_count, sex_count)

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
        """Return a temp file path of the best crop or None."""
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
            return None
        path = photo.get("file_path")
        if not path or not os.path.exists(path):
            return None
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            x1 = int(chosen["x1"] * w); x2 = int(chosen["x2"] * w)
            y1 = int(chosen["y1"] * h); y2 = int(chosen["y2"] * h)
            x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
            if x2 - x1 < 5 or y2 - y1 < 5:
                return None
            tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
            img.crop((x1, y1, x2, y2)).save(tmp, "JPEG", quality=90)
            return tmp
        except Exception:
            return None

    def _best_head_crop_for_photo(self, photo: dict):
        """Return a temp file path of deer_head crop or None."""
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
        chosen = None
        for b in boxes:
            if b.get("label") == "deer_head":
                chosen = b
                break
        if chosen is None:
            for b in boxes:
                if b.get("label") == "ai_deer_head":
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


class TrainerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrailCam Trainer")
        icon_path = Path(__file__).resolve().parent.parent / "ChatGPT Image Dec 5, 2025, 07_07_24 PM.png"
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            self.setWindowIcon(icon)
        self.db = TrailCamDatabase()
        self._write_species_labels_file()
        self.suggester = CombinedSuggester()
        self.auto_enhance_all = True
        self.photos = self._sorted_photos(self.db.get_all_photos())
        # Start at most recent photo (photos are sorted oldest to newest)
        self.index = len(self.photos) - 1 if self.photos else 0
        self.in_review_mode = False
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_current)
        self._known_hashes = None  # lazy-populated MD5 set for duplicate skip

        self._save_pending = False
        self.current_boxes = []
        self.box_items = []
        self.box_mode = None
        self.boxes_hidden = False
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

        # Session-based recently applied species (for quick buttons)
        self._session_recent_species = []  # Species applied this session, most recent first

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

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(10, 12, 10, 12)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
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
        self.apply_suggest_btn.setText("Apply suggestion")
        self.apply_suggest_btn.clicked.connect(self._apply_suggestion)
        # Quick buttons reuse the same recent species buttons
        suggest_row = QVBoxLayout()
        suggest_row.setContentsMargins(0, 0, 0, 0)
        suggest_row.setSpacing(4)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)
        top_row.addWidget(self.suggest_label)
        top_row.addWidget(self.apply_suggest_btn)
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
        quick_container = QWidget()
        quick_container.setLayout(quick_grid)
        form.addRow("", quick_container)
        # Box tools (split into two rows to avoid crowding)
        box_row_top = QHBoxLayout()
        box_row_bottom = QHBoxLayout()
        self.box_subject_btn = QPushButton("Box: Subject")
        self.box_subject_btn.setCheckable(True)
        self.box_subject_btn.clicked.connect(lambda: self._set_box_mode("subject"))
        self.box_head_btn = QPushButton("Box: Deer Head")
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
        self.box_review_ai_btn = QToolButton()
        self.box_review_ai_btn.setText("Review AI Queue")
        self.box_review_ai_btn.clicked.connect(self.start_ai_review)
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
        box_row_bottom.addWidget(self.box_review_ai_btn)
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

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_current)
        self.save_next_btn = QPushButton("Save & Next →")
        self.save_next_btn.clicked.connect(self.save_and_next)
        self.prev_btn = QPushButton("← Prev")
        self.prev_btn.clicked.connect(self.prev_photo)
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.next_photo)
        self.export_btn = QPushButton("Export training CSVs")
        self.export_btn.clicked.connect(self.export_csvs)
        self.compare_btn = QPushButton("Compare Selected")
        self.compare_btn.clicked.connect(self.compare_selected)
        self.priority_btn = QPushButton("Label Next (priority)")
        self.priority_btn.clicked.connect(self.prioritize_for_labeling)
        self.suggest_review_btn = QPushButton("Suggest & Review")
        self.suggest_review_btn.clicked.connect(self.suggest_and_review)
        self.review_queue_btn = QPushButton("Review Queue")
        self.review_queue_btn.clicked.connect(self.enter_review_queue)
        self.exit_review_btn = QToolButton()
        self.exit_review_btn.setText("Exit Review")
        self.exit_review_btn.clicked.connect(self.exit_review_modes)
        self.retrain_btn = QPushButton("Retrain Model")
        self.retrain_btn.clicked.connect(self.retrain_model)
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_photos)
        self.clear_sel_btn = QPushButton("Clear Selection")
        self.clear_sel_btn.clicked.connect(self.clear_selection)
        self.bulk_species_btn = QPushButton("Set Species on Selected")
        self.bulk_species_btn.clicked.connect(self.apply_species_to_selected)
        self.details_toggle = QToolButton()
        self.details_toggle.setText("Hide Details")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setChecked(False)
        self.details_toggle.toggled.connect(self._toggle_details_panel)

        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addStretch()
        nav.addWidget(self.save_btn)
        nav.addWidget(self.save_next_btn)
        nav.addWidget(self.export_btn)
        nav.addWidget(self.compare_btn)
        nav.addWidget(self.priority_btn)
        nav.addWidget(self.suggest_review_btn)
        nav.addWidget(self.review_queue_btn)
        nav.addWidget(self.exit_review_btn)
        nav.addWidget(self.retrain_btn)
        nav.addWidget(self.select_all_btn)
        nav.addWidget(self.clear_sel_btn)
        nav.addWidget(self.bulk_species_btn)
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
        # Species filter
        self.species_filter_combo = QComboBox()
        self.species_filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.species_filter_combo.view().setMinimumWidth(200)
        self.species_filter_combo.currentIndexChanged.connect(self._populate_photo_list)
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
        self._populate_species_filter_options()
        self._populate_sex_filter_options()
        self._populate_deer_id_filter_options()
        self._populate_year_filter_options()
        self._populate_collection_filter_options()

        # Filter layout - two rows
        filter_layout = QVBoxLayout()
        filter_layout.setContentsMargins(2, 2, 2, 2)
        filter_layout.setSpacing(4)

        filter_row1 = QHBoxLayout()
        filter_row1.setSpacing(4)
        filter_row1.addWidget(QLabel("Species:"))
        filter_row1.addWidget(self.species_filter_combo, 1)
        filter_row1.addWidget(QLabel("Sex:"))
        filter_row1.addWidget(self.sex_filter_combo, 1)

        filter_row2 = QHBoxLayout()
        filter_row2.setSpacing(4)
        filter_row2.addWidget(QLabel("Deer ID:"))
        filter_row2.addWidget(self.deer_id_filter_combo, 1)
        filter_row2.addWidget(QLabel("Location:"))
        filter_row2.addWidget(self.site_filter_combo, 1)

        filter_row3 = QHBoxLayout()
        filter_row3.setSpacing(4)
        filter_row3.addWidget(QLabel("Year:"))
        filter_row3.addWidget(self.year_filter_combo, 1)
        filter_row3.addWidget(QLabel("Collection:"))
        filter_row3.addWidget(self.collection_filter_combo, 1)
        filter_row3.addWidget(QLabel("Sort:"))
        filter_row3.addWidget(self.sort_combo, 1)

        filter_layout.addLayout(filter_row1)
        filter_layout.addLayout(filter_row2)
        filter_layout.addLayout(filter_row3)

        filter_row_container = QWidget()
        filter_row_container.setLayout(filter_layout)
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
        left_layout.addWidget(filter_row_container)
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
        export_labels_action = file_menu.addAction("Export Labels to Excel...")
        export_labels_action.triggered.connect(self.export_labels_to_excel)
        import_labels_action = file_menu.addAction("Import Labels from Excel...")
        import_labels_action.triggered.connect(self.import_labels_from_excel)
        file_menu.addSeparator()
        push_cloud_action = file_menu.addAction("Push to Cloud...")
        push_cloud_action.triggered.connect(self.push_to_cloud)
        pull_cloud_action = file_menu.addAction("Pull from Cloud...")
        pull_cloud_action.triggered.connect(self.pull_from_cloud)
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
        ai_all_action = self.tools_menu.addAction("Suggest Tags (AI) — All Photos")
        ai_all_action.triggered.connect(self.run_ai_suggestions_all)
        ai_bg_action = self.tools_menu.addAction("Suggest Tags (AI) — Background + Live Queue")
        ai_bg_action.triggered.connect(self.run_ai_suggestions_background)
        self.tools_menu.addSeparator()
        sex_suggest_action = self.tools_menu.addAction("Suggest Buck/Doe (AI) on Deer Photos...")
        sex_suggest_action.triggered.connect(self.run_sex_suggestions_on_deer)

        # === REVIEW QUEUES (grouped together) ===
        self.tools_menu.addSeparator()
        review_label = self.tools_menu.addAction("── Review Queues ──")
        review_label.setEnabled(False)

        box_review_action = self.tools_menu.addAction("Review AI Boxes (Subject/Head)...")
        box_review_action.triggered.connect(self.start_ai_review)

        species_review_action = self.tools_menu.addAction("Review Species Suggestions...")
        species_review_action.triggered.connect(self.review_species_suggestions_integrated)

        unlabeled_boxes_action = self.tools_menu.addAction("Label Photos with Boxes (No Suggestion)...")
        unlabeled_boxes_action.triggered.connect(self.review_unlabeled_with_boxes)

        sex_review_action = self.tools_menu.addAction("Review Buck/Doe Suggestions...")
        sex_review_action.triggered.connect(self.review_sex_suggestions)

        site_review_action = self.tools_menu.addAction("Review Site Suggestions...")
        site_review_action.triggered.connect(self.review_site_suggestions)

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

        # Store AI-related actions for simple mode hiding
        self.advanced_menu_actions = [
            clear_ai_action, ai_action, ai_all_action, sex_suggest_action,
            review_label, box_review_action, species_review_action,
            sex_review_action, site_review_action, profiles_action,
            site_label, auto_site_action, manage_sites_action
        ]

        menubar.addMenu(self.tools_menu)
        settings_menu = QMenu("Settings", self)
        # Simple/Advanced mode toggle
        self.simple_mode = False
        self.simple_mode_action = settings_menu.addAction("Simple Mode")
        self.simple_mode_action.setCheckable(True)
        self.simple_mode_action.setChecked(False)
        self.simple_mode_action.toggled.connect(self._toggle_simple_mode)
        settings_menu.addSeparator()
        self.enhance_toggle_action = settings_menu.addAction("Auto Enhance All")
        self.enhance_toggle_action.setCheckable(True)
        self.enhance_toggle_action.setChecked(True)
        self.enhance_toggle_action.toggled.connect(self.toggle_global_enhance)
        settings_menu.addSeparator()
        supabase_setup_action = settings_menu.addAction("Setup Supabase Cloud...")
        supabase_setup_action.triggered.connect(self.setup_supabase_credentials)
        reset_sync_prefs_action = settings_menu.addAction("Reset Cloud Sync Preferences...")
        reset_sync_prefs_action.triggered.connect(self._reset_cloud_sync_preferences)
        menubar.addMenu(settings_menu)
        layout.setMenuBar(menubar)
        layout.addWidget(split)
        layout.addLayout(nav)
        self.setCentralWidget(container)

        if not self.photos:
            QMessageBox.information(self, "Trainer", "No photos found in the database.")
        else:
            self.load_photo()
        self._populate_photo_list()

        # Autosave on edits
        self.species_combo.currentIndexChanged.connect(self._on_species_changed)
        self.species_combo.editTextChanged.connect(self._on_species_changed)
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

        # Restore simple mode
        simple_mode = settings.value("simple_mode", False, type=bool)
        if simple_mode:
            self.simple_mode_action.setChecked(True)

    def _save_settings(self):
        """Save window settings."""
        settings = QSettings("TrailCam", "Trainer")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("maximized", self.isMaximized())
        settings.setValue("simple_mode", self.simple_mode)

    def showEvent(self, event):
        """Handle window show - prompt for cloud sync and CuddeLink on first show."""
        super().showEvent(event)
        # Only run once on first show
        if not hasattr(self, '_shown_once'):
            self._shown_once = True
            # Use a timer to run after the window is fully shown
            QTimer.singleShot(500, self._run_startup_prompts)

    def _run_startup_prompts(self):
        """Run all startup prompts in sequence."""
        self._check_cloud_pull_on_startup()
        # Check CuddeLink after cloud sync
        QTimer.singleShot(100, self._check_cuddelink_on_startup)

    def _check_cloud_pull_on_startup(self):
        """Check if we should pull from cloud on startup."""
        settings = QSettings("TrailCam", "Trainer")

        # Check if we have Supabase credentials
        url = settings.value("supabase_url", "")
        key = settings.value("supabase_key", "")
        if not url or not key:
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
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Cloud Sync", f"Failed to pull from cloud:\n{str(e)}")

    def closeEvent(self, event):
        """Save settings and optionally push to cloud before closing."""
        self._save_settings()

        # Check if we should push to cloud
        settings = QSettings("TrailCam", "Trainer")
        url = settings.value("supabase_url", "")
        key = settings.value("supabase_key", "")

        if url and key:
            always_push = settings.value("cloud_always_push_on_close", "")

            if always_push == "yes":
                self._do_cloud_push_silent()
            elif always_push == "no":
                pass  # User chose to never push
            else:
                # Ask the user
                if self._prompt_cloud_push():
                    self._do_cloud_push_silent()

        event.accept()
        QApplication.instance().quit()

    def _prompt_cloud_push(self):
        """Show dialog asking whether to push to cloud. Returns True if user chose yes."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Cloud Sync")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        label = QLabel("Would you like to save your labels to the cloud before closing?")
        label.setWordWrap(True)
        layout.addWidget(label)

        remember_check = QCheckBox("Remember my choice")
        layout.addWidget(remember_check)

        btn_layout = QHBoxLayout()
        yes_btn = QPushButton("Yes, Save to Cloud")
        no_btn = QPushButton("No, Just Close")
        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(no_btn)
        layout.addLayout(btn_layout)

        result = {"choice": False}

        def on_yes():
            result["choice"] = True
            if remember_check.isChecked():
                settings = QSettings("TrailCam", "Trainer")
                settings.setValue("cloud_always_push_on_close", "yes")
            dialog.accept()

        def on_no():
            result["choice"] = False
            if remember_check.isChecked():
                settings = QSettings("TrailCam", "Trainer")
                settings.setValue("cloud_always_push_on_close", "no")
            dialog.accept()

        yes_btn.clicked.connect(on_yes)
        no_btn.clicked.connect(on_no)

        dialog.exec()

        return result["choice"]

    def _do_cloud_push_silent(self):
        """Push to cloud without showing the full dialog."""
        try:
            client = self._get_supabase_client()
            if not client:
                return

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()

            self.db.push_to_supabase(client)

            QApplication.restoreOverrideCursor()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            # Don't show error on close, just log it
            print(f"Cloud push failed: {e}")

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
        self.tags_edit.setText(", ".join(tags))

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

        # species: use stored tags only (suggestions require user approval)
        species_options = set(SPECIES_OPTIONS)
        try:
            species_options.update(self.db.list_custom_species())
        except Exception as e:
            logger.debug(f"Failed to list custom species: {e}")
        # Collect ALL species tags for this photo
        current_species = [t for t in tags if t in species_options]
        # Show first species in combo, all in label
        species = current_species[0] if current_species else ""
        self.species_combo.setCurrentText(species)
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
                self.species_combo.setCurrentText("Deer")
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

        self._loading_photo_data = False  # Done loading, allow auto-advance

    def save_current(self):
        if not self.photos:
            return
        photo = self.photos[self.index]
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
        # Always keep existing species and add the new one (multi-species support)
        existing_species = [t for t in tags if t in species_set]
        # Only remove species tags if explicitly replacing (not adding)
        # Add current species and sex (only if valid - known species or length >= 3)
        if species and species not in tags:
            # Only save if it's a known species OR at least 3 characters (avoid partial typing)
            if species in species_set or len(species) >= 3:
                tags.append(species)
        if sex.lower() in ("buck", "doe") and sex not in tags:
            tags.append(sex)
        self.db.update_photo_tags(pid, tags)
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
            print(f"Box save failed: {exc}")
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
        if species:
            # Official tag applied; clear suggestion
            self.db.set_suggested_tag(pid, None, None)
            # Bump to session recent species list
            if species in self._session_recent_species:
                self._session_recent_species.remove(species)
            self._session_recent_species.insert(0, species)
            self._session_recent_species = self._session_recent_species[:20]  # Keep max 20
            # persist custom species (only if length >= 3 to avoid partial typing)
            if species not in SPECIES_OPTIONS and len(species) >= 3:
                self.db.add_custom_species(species)
                # refresh dropdown with saved customs
                self._populate_species_dropdown()
        # Refresh camera location dropdown if new location was added
        cam_loc = self.camera_combo.currentText().strip()
        if cam_loc and self.camera_combo.findText(cam_loc) == -1:
            self._populate_camera_locations()
        self._update_recent_species_buttons()
        # refresh in-memory photo metadata
        self.photos[self.index].update(self.db.get_photo_by_id(pid) or {})
        self._bump_recent_buck(deer_id)
        self._update_recent_buck_buttons()
        self._update_photo_list_item(self.index)
        # If in review mode and resolved, remove from queue and advance
        if self.in_review_mode and self._current_photo_resolved(pid):
            self.photo_list_widget.blockSignals(True)
            item = self.photo_list_widget.item(self.index)
            if item:
                self.photo_list_widget.takeItem(self.index)
            self.photo_list_widget.blockSignals(False)
            # Remove photo from list at current index before adjusting
            try:
                if 0 <= self.index < len(self.photos):
                    self.photos.pop(self.index)
            except Exception as e:
                logger.warning(f"Failed to remove photo from list at index {self.index}: {e}")
            # Adjust index if we were at the end
            if self.index >= len(self.photos):
                self.index = max(0, len(self.photos) - 1)
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
            self._write_species_labels_file()
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
        # Save immediately instead of debouncing to ensure tag changes are saved
        self.save_timer.stop()
        self.save_current()

        # In queue mode, auto-advance when user selects a species (unless multi-species checked)
        if self.queue_mode and not self._loading_photo_data:
            # Don't auto-advance if "Multiple species" is checked
            if hasattr(self, 'queue_multi_species') and self.queue_multi_species.isChecked():
                return
            species = self.species_combo.currentText().strip()
            if species:  # Only advance if a species was actually selected
                current_pid = None
                if self.photos and self.index < len(self.photos):
                    current_pid = self.photos[self.index].get("id")
                if current_pid:
                    # Mark as reviewed for green highlighting
                    self.queue_reviewed.add(current_pid)
                    # Clear AI suggestion since user made a decision
                    self.db.set_suggested_tag(current_pid, None, None)
                    self._mark_current_list_item_reviewed()
                self._queue_advance()

    def _get_filtered_indices(self) -> List[int]:
        """Get list of photo indices that pass current filters."""
        return [idx for idx, _ in self._filtered_photos()]

    def prev_photo(self):
        if not self.photos:
            return
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

    # --- Detection helpers ---
    def _get_megadetector(self):
        """Load and cache MegaDetector model."""
        if hasattr(self, "_megadetector") and self._megadetector is not None:
            return self._megadetector
        try:
            from megadetector.detection import run_detector
            self._megadetector = run_detector.load_detector('MDV5A')
            return self._megadetector
        except Exception as e:
            logger.warning(f"Failed to load MegaDetector: {e}")
            return None

    def _detect_boxes_megadetector(self, path: str, conf_thresh: float = 0.2):
        """Detect animals/people/vehicles using MegaDetector."""
        model = self._get_megadetector()
        if model is None:
            return []
        try:
            from megadetector.visualization import visualization_utils as vis_utils
            image = vis_utils.load_image(path)
            result = model.generate_detections_one_image(image)
            detections = [d for d in result.get('detections', []) if d['conf'] >= conf_thresh]
            boxes = []
            for det in detections:
                # MegaDetector format: [x, y, width, height] normalized 0-1
                # category: 1=animal, 2=person, 3=vehicle
                x, y, w, h = det['bbox']
                category = det['category']
                conf = det['conf']
                label_map = {1: 'ai_animal', 2: 'ai_person', 3: 'ai_vehicle'}
                label = label_map.get(int(category), 'ai_subject')
                boxes.append({
                    'label': label,
                    'x1': x,
                    'y1': y,
                    'x2': x + w,
                    'y2': y + h,
                    'confidence': conf
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
            elif str(lbl).startswith("ai_"):
                pen = QPen(Qt.GlobalColor.yellow)
            elif lbl == "deer_head":
                pen = QPen(Qt.GlobalColor.red)
            else:
                pen = QPen(Qt.GlobalColor.green)
            pen.setWidth(5)
            item = DraggableBoxItem(idx, rect, pen, _on_change)
            self.scene.addItem(item)
            self.box_items.append(item)

    def _persist_boxes(self):
        """Persist boxes immediately for the current photo."""
        if not self.photos or self.index >= len(self.photos):
            return
        pid = self.photos[self.index].get("id")
        if not pid:
            return
        try:
            self.db.set_boxes(pid, self.current_boxes)
        except Exception as exc:
            print(f"Box save failed: {exc}")

    def _delete_box_by_idx(self, idx: int):
        """Remove a box by index (used by Delete key)."""
        if idx < 0 or idx >= len(self.current_boxes):
            return
        del self.current_boxes[idx]
        self._draw_boxes()
        self._persist_boxes()
        if self.ai_review_mode and not self._photo_has_ai_boxes(self.current_boxes):
            self._advance_ai_review()

    def exit_review_modes(self):
        """Leave any review mode and show full library."""
        self.in_review_mode = False
        self.ai_review_mode = False
        self.ai_review_queue = []
        self._advancing_review = False
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
            print(f"[AI] Wrote {len(labels)} valid species to labels.txt")
        except Exception as exc:
            print(f"[AI] Failed to write labels.txt: {exc}")

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

    def export_labels_to_excel(self):
        """Export all photo labels to an Excel file for sharing between computers.

        Exports: filename, date_taken, camera_model, species, sex (buck/doe),
        deer_id, age_class, camera_location, notes.
        """
        import pandas as pd
        from pathlib import Path as P

        if not self.photos:
            QMessageBox.warning(self, "Export", "No photos to export.")
            return

        # Ask for save location
        default_name = "trailcam_labels.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Labels to Excel", default_name,
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if not file_path:
            return

        # Collect data
        rows = []
        for photo in self.photos:
            pid = photo["id"]
            path = P(photo.get("file_path", ""))
            tags = self.db.get_tags(pid)
            deer_meta = self.db.get_deer_metadata(pid) or {}

            # Extract species (non-sex tags)
            species_tags = [t for t in tags if t.lower() not in ("buck", "doe", "unknown")]
            species = ", ".join(species_tags) if species_tags else ""

            # Extract sex
            sex = ""
            if "Buck" in tags:
                sex = "Buck"
            elif "Doe" in tags:
                sex = "Doe"

            rows.append({
                "filename": path.name,
                "date_taken": photo.get("date_taken", ""),
                "camera_model": photo.get("camera_model", ""),
                "species": species,
                "sex": sex,
                "deer_id": deer_meta.get("deer_id", ""),
                "age_class": deer_meta.get("age_class", ""),
                "camera_location": photo.get("camera_location", ""),
                "notes": photo.get("notes", ""),
                # Include full path as reference (won't be used for matching)
                "full_path": str(path),
            })

        df = pd.DataFrame(rows)

        try:
            df.to_excel(file_path, index=False, engine="openpyxl")
            labeled_count = len([r for r in rows if r["species"] or r["sex"] or r["deer_id"]])
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(rows)} photos to:\n{file_path}\n\n"
                f"{labeled_count} photos have labels."
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    def import_labels_from_excel(self):
        """Import labels from an Excel file, matching by filename.

        Matches photos by filename and applies labels. If a photo already has
        labels, you can choose to skip or overwrite.
        """
        import pandas as pd
        from pathlib import Path as P

        # Ask for file to import
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Labels from Excel", "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if not file_path:
            return

        try:
            df = pd.read_excel(file_path, engine="openpyxl")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to read Excel file:\n{e}")
            return

        # Check required columns
        required = ["filename"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            QMessageBox.warning(
                self, "Import Error",
                f"Excel file is missing required columns: {missing}\n\n"
                "Required: filename\n"
                "Optional: species, sex, deer_id, age_class, camera_location, notes"
            )
            return

        # Build filename -> photo mapping
        photo_by_filename = {}
        for photo in self.photos:
            fname = P(photo.get("file_path", "")).name
            if fname:
                photo_by_filename[fname] = photo

        # Process imports
        matched = 0
        updated = 0
        skipped = 0

        progress = QProgressDialog("Importing labels...", "Cancel", 0, len(df), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        for idx, row in df.iterrows():
            if progress.wasCanceled():
                break
            progress.setValue(idx)

            filename = str(row.get("filename", "")).strip()
            if not filename or filename not in photo_by_filename:
                skipped += 1
                continue

            photo = photo_by_filename[filename]
            pid = photo["id"]
            matched += 1

            # Get values from Excel (handle NaN)
            species = str(row.get("species", "")) if pd.notna(row.get("species")) else ""
            sex = str(row.get("sex", "")) if pd.notna(row.get("sex")) else ""
            deer_id = str(row.get("deer_id", "")) if pd.notna(row.get("deer_id")) else ""
            age_class = str(row.get("age_class", "")) if pd.notna(row.get("age_class")) else ""
            location = str(row.get("camera_location", "")) if pd.notna(row.get("camera_location")) else ""
            notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""

            # Build tag list
            tags = []
            if species:
                # Handle comma-separated species
                for s in species.split(","):
                    s = s.strip()
                    if s:
                        tags.append(s)
            if sex and sex.lower() in ("buck", "doe"):
                tags.append(sex.title())

            # Apply tags
            if tags:
                self.db.set_tags(pid, tags)

            # Apply deer metadata
            if deer_id or age_class:
                self.db.set_deer_metadata(pid, deer_id.strip(), age_class.strip())

            # Apply location and notes
            cursor = self.db.conn.cursor()
            if location or notes:
                cursor.execute(
                    "UPDATE photos SET camera_location = COALESCE(?, camera_location), "
                    "notes = COALESCE(?, notes) WHERE id = ?",
                    (location or None, notes or None, pid)
                )
            self.db.conn.commit()

            updated += 1

        progress.setValue(len(df))

        # Refresh UI
        self.photos = self._sorted_photos(self.db.get_all_photos())
        self._populate_photo_list()
        if self.photos:
            self.load_photo()

        QMessageBox.information(
            self, "Import Complete",
            f"Imported labels from Excel:\n\n"
            f"Rows in file: {len(df)}\n"
            f"Matched photos: {matched}\n"
            f"Updated: {updated}\n"
            f"Skipped (no match): {skipped}"
        )

    @staticmethod
    def _sorted_photos(photos: list) -> list:
        """Sort photos by date_taken (ascending), fallback to file_path."""
        from datetime import datetime
        def key(p):
            dt = p.get("date_taken") or ""
            try:
                parsed = datetime.fromisoformat(dt)
            except Exception:
                parsed = None
            return (parsed or datetime.min, p.get("file_path") or "")
        return sorted(photos, key=key)

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
                dt = str(exif_data[key]).replace(":", "-", 2).replace(" ", " ")
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

    def _toggle_simple_mode(self, enabled: bool):
        """Toggle between Simple Mode and Advanced Mode.

        Simple Mode hides AI features, training tools, and complex annotation options.
        Shows only: photo browsing, basic tagging, species, buck/doe, deer ID, location, notes.
        """
        self.simple_mode = enabled
        show_advanced = not enabled

        # Update window title
        mode_str = " (Simple Mode)" if enabled else ""
        self.setWindowTitle(f"TrailCam Trainer{mode_str}")

        # --- Hide/show UI elements ---

        # Box annotation tools
        self.box_container_top.setVisible(show_advanced)
        self.box_container_bottom.setVisible(show_advanced)

        # AI suggestion rows
        self.suggest_row_widget.setVisible(show_advanced)
        self.sex_suggest_label.setVisible(show_advanced)
        self.apply_sex_suggest_btn.setVisible(show_advanced)

        # Antler details toggle (hide the toggle, container follows)
        self.antler_toggle.setVisible(show_advanced)
        if enabled:
            self.antler_container.setVisible(False)

        # Tags row (comma-separated tags)
        self.tags_edit.setVisible(show_advanced)

        # Key characteristics - keep visible in simple mode

        # Bulk operations container (merge, bulk buck, profile buttons)
        self.bulk_container.setVisible(show_advanced)

        # Additional buck toggle and container
        self.add_buck_toggle.setVisible(show_advanced)
        if enabled:
            self.add_buck_container.setVisible(False)

        # Navigation buttons (hide advanced ones)
        self.export_btn.setVisible(show_advanced)
        self.compare_btn.setVisible(show_advanced)
        self.priority_btn.setVisible(show_advanced)
        self.suggest_review_btn.setVisible(show_advanced)
        self.review_queue_btn.setVisible(show_advanced)
        self.exit_review_btn.setVisible(show_advanced)
        self.retrain_btn.setVisible(show_advanced)
        self.select_all_btn.setVisible(show_advanced)
        self.clear_sel_btn.setVisible(show_advanced)
        self.bulk_species_btn.setVisible(show_advanced)

        # Suggestion filter in left panel
        self.suggest_filter_combo.setVisible(show_advanced)

        # Menu items
        for action in self.advanced_menu_actions:
            action.setVisible(show_advanced)

    def _update_suggestion_display(self, photo: dict):
        """Show AI suggestion with confidence, require explicit apply."""
        tag = photo.get("suggested_tag") or ""
        conf = photo.get("suggested_confidence")
        self.current_suggested_tag = tag
        self.current_suggested_conf = conf
        if tag:
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
        self.species_combo.setCurrentText(tag)
        self.schedule_save()

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
            for photo in self.db.get_all_photos():
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
            for photo in self.db.get_all_photos():
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
            for photo in self.db.get_all_photos():
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
            suggestions = sorted({p.get("suggested_tag") for p in self.db.get_all_photos() if p.get("suggested_tag")})
        except Exception:
            suggestions = []
        for s in suggestions:
            if s:
                self.suggest_filter_combo.addItem(f"Suggested: {s}", s)
        idx = self.suggest_filter_combo.findData(current)
        self.suggest_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.suggest_filter_combo.blockSignals(False)

    def _populate_species_filter_options(self):
        """Fill the species filter combo with counts."""
        if not hasattr(self, "species_filter_combo"):
            return
        current = self.species_filter_combo.currentData()
        self.species_filter_combo.blockSignals(True)
        self.species_filter_combo.clear()
        self.species_filter_combo.addItem("All Species", "")
        try:
            # Count photos by species
            species_counts = {}
            unlabeled_count = 0
            for photo in self.db.get_all_photos():
                tags = set(self.db.get_tags(photo["id"]))
                species_tags = tags & VALID_SPECIES
                if not species_tags:
                    unlabeled_count += 1
                for sp in species_tags:
                    species_counts[sp] = species_counts.get(sp, 0) + 1
            self.species_filter_combo.addItem(f"Unlabeled ({unlabeled_count})", "__unlabeled__")
            # Add all species alphabetically
            for sp in sorted(species_counts.keys()):
                self.species_filter_combo.addItem(f"{sp} ({species_counts[sp]})", sp)
        except Exception:
            self.species_filter_combo.addItem("Unlabeled", "__unlabeled__")
            self.species_filter_combo.addItem("Deer", "Deer")
        idx = self.species_filter_combo.findData(current)
        self.species_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.species_filter_combo.blockSignals(False)

    def _populate_sex_filter_options(self):
        """Fill the sex filter combo with counts."""
        if not hasattr(self, "sex_filter_combo"):
            return
        current = self.sex_filter_combo.currentData()
        self.sex_filter_combo.blockSignals(True)
        self.sex_filter_combo.clear()
        self.sex_filter_combo.addItem("All", "")
        try:
            buck_count = 0
            doe_count = 0
            unknown_count = 0
            for photo in self.db.get_all_photos():
                tags = set(self.db.get_tags(photo["id"]))
                if "Buck" in tags:
                    buck_count += 1
                elif "Doe" in tags:
                    doe_count += 1
                elif "Unknown" in tags:
                    unknown_count += 1
            self.sex_filter_combo.addItem(f"Buck ({buck_count})", "Buck")
            self.sex_filter_combo.addItem(f"Doe ({doe_count})", "Doe")
            self.sex_filter_combo.addItem(f"Unknown ({unknown_count})", "Unknown")
        except Exception:
            self.sex_filter_combo.addItem("Buck", "Buck")
            self.sex_filter_combo.addItem("Doe", "Doe")
            self.sex_filter_combo.addItem("Unknown", "Unknown")
        idx = self.sex_filter_combo.findData(current)
        self.sex_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.sex_filter_combo.blockSignals(False)

    def _populate_deer_id_filter_options(self):
        """Fill the deer ID filter combo with existing deer IDs and counts."""
        if not hasattr(self, "deer_id_filter_combo"):
            return
        current = self.deer_id_filter_combo.currentData()
        self.deer_id_filter_combo.blockSignals(True)
        self.deer_id_filter_combo.clear()
        self.deer_id_filter_combo.addItem("All Deer IDs", "")
        try:
            id_counts = {}
            has_id_count = 0
            no_id_count = 0
            for photo in self.db.get_all_photos():
                meta = self.db.get_deer_metadata(photo["id"])
                deer_id = meta.get("deer_id")
                if deer_id:
                    has_id_count += 1
                    id_counts[deer_id] = id_counts.get(deer_id, 0) + 1
                else:
                    no_id_count += 1
            self.deer_id_filter_combo.addItem(f"Has ID ({has_id_count})", "__has_id__")
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
        """Fill the site filter combo with camera_location values."""
        if not hasattr(self, "site_filter_combo"):
            return
        current = self.site_filter_combo.currentData()
        self.site_filter_combo.blockSignals(True)
        self.site_filter_combo.clear()
        self.site_filter_combo.addItem("All locations", None)
        self.site_filter_combo.addItem("Unassigned", "__unassigned__")
        try:
            # Count photos by camera_location
            location_counts = {}
            for photo in self.db.get_all_photos():
                loc = photo.get("camera_location")
                if loc and loc.strip():
                    loc = loc.strip()
                    location_counts[loc] = location_counts.get(loc, 0) + 1
            # Add sorted locations with counts
            for loc in sorted(location_counts.keys()):
                count = location_counts[loc]
                label = f"{loc} ({count})"
                self.site_filter_combo.addItem(label, loc)
        except Exception:
            pass
        idx = self.site_filter_combo.findData(current)
        self.site_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.site_filter_combo.blockSignals(False)

    def _populate_year_filter_options(self):
        """Fill the year filter combo with antler years (May-April, displayed as YYYY-YYYY)."""
        if not hasattr(self, "year_filter_combo"):
            return
        current = self.year_filter_combo.currentData()
        self.year_filter_combo.blockSignals(True)
        self.year_filter_combo.clear()
        self.year_filter_combo.addItem("All Years", None)
        try:
            # Count photos by antler year
            year_counts = {}
            for photo in self.db.get_all_photos():
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
        """Fill the collection/farm filter combo."""
        if not hasattr(self, "collection_filter_combo"):
            return
        current = self.collection_filter_combo.currentData()
        self.collection_filter_combo.blockSignals(True)
        self.collection_filter_combo.clear()
        self.collection_filter_combo.addItem("All Collections", None)
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT collection, COUNT(*) as cnt
                FROM photos
                WHERE collection IS NOT NULL AND collection != ''
                GROUP BY collection
                ORDER BY cnt DESC
            """)
            for row in cursor.fetchall():
                coll, count = row
                self.collection_filter_combo.addItem(f"{coll} ({count})", coll)
            # Add unassigned option
            cursor.execute("SELECT COUNT(*) FROM photos WHERE collection IS NULL OR collection = ''")
            unassigned = cursor.fetchone()[0]
            if unassigned > 0:
                self.collection_filter_combo.addItem(f"Unassigned ({unassigned})", "__unassigned__")
        except Exception:
            pass
        idx = self.collection_filter_combo.findData(current)
        self.collection_filter_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.collection_filter_combo.blockSignals(False)

    def _filtered_photos(self):
        """Apply all filters to in-memory photo list."""
        result = list(enumerate(self.photos))

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

        # Apply species filter
        if hasattr(self, "species_filter_combo"):
            species_flt = self.species_filter_combo.currentData()
            if species_flt:
                filtered = []
                for idx, p in result:
                    tags = set(self.db.get_tags(p["id"]))
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
                    tags = set(self.db.get_tags(p["id"]))
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
                    meta = self.db.get_deer_metadata(p["id"])
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
                    tags = self.db.get_tags(p["id"])
                    species_tags = set(tags) & VALID_SPECIES
                    return sorted(species_tags)[0] if species_tags else "zzz"
                result.sort(key=lambda x: (get_species(x[1]), x[1].get("date_taken") or ""))
            elif sort_key == "deer_id":
                def get_deer_id(p):
                    meta = self.db.get_deer_metadata(p["id"])
                    return meta.get("deer_id") or "zzz"
                result.sort(key=lambda x: (get_deer_id(x[1]), x[1].get("date_taken") or ""))

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
            if idx is None:
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
        photo = self.photos[self.index]
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
        self._populate_suggest_filter_options()
        self._populate_site_filter_options()
        self._populate_year_filter_options()
        self.photo_list_widget.blockSignals(True)
        self.photo_list_widget.clear()
        filtered = self._filtered_photos()
        filtered_indices = [idx for idx, _ in filtered]

        # If current photo not in filtered set, jump to first filtered photo
        jumped = False
        if filtered_indices and self.index not in filtered_indices:
            self.index = filtered_indices[0]
            jumped = True

        target_item = None
        for row, (idx, p) in enumerate(filtered):
            display = self._build_photo_label(idx)
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.photo_list_widget.addItem(item)
            if idx == self.index:
                target_item = item
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
            if idx is None:
                continue
            pid = self.photos[idx].get("id")
            if pid:
                photo_ids.append(pid)
        if len(photo_ids) < 2:
            QMessageBox.information(self, "Compare", "Need at least two valid photos.")
            return
        dlg = CompareWindow(photo_ids=photo_ids, db=self.db, parent=self)
        dlg.exec()

    def select_all_photos(self):
        """Select all photos in the list."""
        self.photo_list_widget.blockSignals(True)
        self.photo_list_widget.selectAll()
        self.photo_list_widget.blockSignals(False)
        self.on_photo_selection_changed()

    def clear_selection(self):
        """Clear list selection."""
        self.photo_list_widget.clearSelection()

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
        self.ai_review_mode = True
        self.ai_review_queue = queue
        self.ai_reviewed_photos = set()  # Clear reviewed tracking
        self._advancing_review = False  # Reset guard
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
        for item in selected:
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is None:
                continue
            pid = self.photos[idx]["id"]
            tags = set(self.db.get_tags(pid))
            tags.add(species)
            self.db.update_photo_tags(pid, list(tags))
            self.db.set_suggested_tag(pid, species, None)
        QMessageBox.information(self, "Bulk Set", f"Assigned species '{species}' to {len(selected)} photo(s).")
        self._populate_photo_list()

    def _build_photo_label(self, idx: int) -> str:
        """Build list label with markers for species (*) and buck (^)"""
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
        has_species = False
        has_buck = False
        verified = False
        auto_suggest = False
        species_labels = set(SPECIES_OPTIONS)
        try:
            species_labels.update(self.db.list_custom_species())
        except Exception:
            pass
        try:
            tags = set(self.db.get_tags(pid))
            if p.get("suggested_tag"):
                auto_suggest = True
            if p.get("suggested_tag"):
                tags.add(p.get("suggested_tag"))
            has_species = any(t in species_labels for t in tags if t)
            deer = self.db.get_deer_metadata(pid)
            has_buck = bool(deer.get("deer_id"))
            boxes = self.db.get_boxes(pid)
            verified = bool(boxes)
        except Exception:
            pass
        suffix = ""
        if has_species:
            suffix += "*"
        if has_buck:
            suffix += "^"
        if verified:
            suffix += "V"
        elif auto_suggest:
            suffix += "A"

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

        display = f"{label} {suffix}{queue_suffix}".strip()
        return display

    def _update_photo_list_item(self, idx: int):
        """Update a single list item label without resetting selection."""
        for i in range(self.photo_list_widget.count()):
            item = self.photo_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == idx:
                item.setText(self._build_photo_label(idx))
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

    def _import_files(self, files: List[Path], skip_hash: bool = True, collection: str = "", progress_callback=None) -> int:
        """Copy files into the library, add to DB, and build thumbnails.

        Args:
            files: List of file paths to import
            skip_hash: If True, skip files that match existing hashes
            collection: Collection/farm name to assign to imported photos
            progress_callback: Optional callable(current, total, filename) for progress updates.
                               If it returns False, import is cancelled.
        """
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

            file_hash = self._hash_file(file_path) if skip_hash else None
            if skip_hash and file_hash and file_hash in self._known_hashes:
                continue
            try:
                dest_path, original_name, date_taken, camera_model = import_photo(str(file_path))
            except Exception as exc:
                print(f"Import failed for {file_path}: {exc}")
                continue
            if self.db.get_photo_id(dest_path):
                # Already imported.
                continue
            thumb_path = create_thumbnail(dest_path)
            try:
                self.db.add_photo(dest_path, original_name, date_taken or "", camera_model or "", thumb_path, collection=collection)
                if skip_hash and file_hash:
                    self._known_hashes.add(file_hash)
                imported += 1
            except Exception as exc:
                print(f"DB insert failed for {dest_path}: {exc}")
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
            print(f"Hash failed for {path}: {exc}")
            return None

    def _load_known_hashes(self) -> set:
        """Compute hashes for existing photos once to skip duplicates upfront."""
        hashes = set()
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

    def _best_crop_for_photo(self, photo: dict) -> Optional[Path]:
        """Return a temp file path of the best crop (deer_head > ai_animal > subject) or None."""
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
        """Return a temp file path of deer_head crop for buck/doe classification, or None."""
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
        # Only use deer_head boxes (human-labeled preferred)
        chosen = None
        for b in boxes:
            if b.get("label") == "deer_head":
                chosen = b
                break
        # Fall back to ai_deer_head if no human-labeled head
        if chosen is None:
            for b in boxes:
                if b.get("label") == "ai_deer_head":
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
            self._populate_photo_list()
            if self.photos:
                self.index = 0
                self.load_photo()

    def setup_cuddelink_credentials(self):
        """Show dialog to set up CuddeLink credentials."""
        settings = QSettings("TrailCam", "Trainer")
        current_email = settings.value("cuddelink_email", "")

        dialog = QDialog(self)
        dialog.setWindowTitle("CuddeLink Credentials")
        dialog.setMinimumWidth(350)
        layout = QVBoxLayout(dialog)

        # Instructions
        info_label = QLabel("Enter your CuddeLink account credentials.\nThese will be saved for future downloads.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Email field
        email_layout = QHBoxLayout()
        email_layout.addWidget(QLabel("Email:"))
        email_edit = QLineEdit()
        email_edit.setText(current_email)
        email_edit.setPlaceholderText("your@email.com")
        email_layout.addWidget(email_edit)
        layout.addLayout(email_layout)

        # Password field
        pass_layout = QHBoxLayout()
        pass_layout.addWidget(QLabel("Password:"))
        pass_edit = QLineEdit()
        pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        pass_edit.setPlaceholderText("Enter password")
        pass_layout.addWidget(pass_edit)
        layout.addLayout(pass_layout)

        # Note about storage
        note_label = QLabel("Note: Password is stored locally on this computer.")
        note_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note_label)

        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        test_btn = QPushButton("Test Connection")
        btn_layout.addWidget(test_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        def save_credentials():
            email = email_edit.text().strip()
            password = pass_edit.text()
            if not email or not password:
                QMessageBox.warning(dialog, "CuddeLink", "Please enter both email and password.")
                return
            settings.setValue("cuddelink_email", email)
            settings.setValue("cuddelink_password", password)
            QMessageBox.information(dialog, "CuddeLink", "Credentials saved successfully!")
            dialog.accept()

        def test_connection():
            email = email_edit.text().strip()
            password = pass_edit.text()
            if not email or not password:
                QMessageBox.warning(dialog, "CuddeLink", "Please enter both email and password.")
                return
            # Try to login
            try:
                import requests
                import re
                session = requests.Session()
                session.headers.update({
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                })
                # Get login page
                login_url = "https://camp.cuddeback.com/Identity/Account/Login"
                resp = session.get(login_url, timeout=20)
                resp.raise_for_status()
                # Extract token
                token = None
                m = re.search(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', resp.text)
                if m:
                    token = m.group(1)
                # Login
                payload = {
                    "Input.Email": email,
                    "Input.Password": password,
                    "Input.RememberMe": "false",
                }
                if token:
                    payload["__RequestVerificationToken"] = token
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "https://camp.cuddeback.com",
                    "Referer": login_url,
                }
                post = session.post(login_url, data=payload, headers=headers, timeout=20, allow_redirects=True)
                # Check for invalid login message
                if "Invalid login" in post.text or ("invalid" in post.text.lower() and "login" in post.url.lower()):
                    QMessageBox.warning(dialog, "CuddeLink", "Invalid email or password.")
                    return
                # Check if we're logged in by accessing photos page
                photos_resp = session.get("https://camp.cuddeback.com/photos", timeout=20)
                if "login" in photos_resp.url.lower() or photos_resp.status_code >= 400:
                    QMessageBox.warning(dialog, "CuddeLink", "Login failed. Please check your credentials.")
                else:
                    QMessageBox.information(dialog, "CuddeLink", "Connection successful! Credentials are valid.")
            except Exception as e:
                QMessageBox.warning(dialog, "CuddeLink", f"Connection failed: {str(e)}")

        save_btn.clicked.connect(save_credentials)
        cancel_btn.clicked.connect(dialog.reject)
        test_btn.clicked.connect(test_connection)

        dialog.exec()

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
                    last_download = row[0][:10]  # Take just the date part
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
        self._cudde_dialog.setMinimumWidth(300)
        layout = QVBoxLayout(self._cudde_dialog)
        self._cudde_label = QLabel("Connecting to CuddeLink...")
        layout.addWidget(self._cudde_label)
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
                    files = download_new_photos(self.dest, user=self.email, password=self.password,
                                                start_date=self.start_date, end_date=self.end_date)
                    self.finished.emit(files, "")
                except Exception as e:
                    self.finished.emit([], str(e))

        def on_status(msg):
            self._cudde_label.setText(msg)

        def on_download_complete(files, error):
            self._cudde_dialog.close()

            if self._cudde_cancelled:
                return

            if error:
                if "credentials" in error.lower() or "login" in error.lower() or "invalid" in error.lower():
                    QMessageBox.warning(self, "CuddeLink", f"Login failed. Please check your credentials in\nFile → Setup CuddeLink Credentials.\n\nError: {error}")
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

            imported = self._import_files(new_files, skip_hash=True, collection=selected_collection)
            # Clean up extracted files
            temp_dir = dest / ".cuddelink_tmp"
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

            msg = f"Imported {imported} new photo(s)."
            if skipped > 0:
                msg += f"\nSkipped {skipped} duplicate(s)."
            QMessageBox.information(self, "CuddeLink", msg)
            if imported:
                # Save the end date as last download date for next time
                settings.setValue("cuddelink_last_download", end_date)
                self.photos = self._sorted_photos(self.db.get_all_photos())
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

        self._cudde_worker = DownloadWorker(dest, email, password, start_date, end_date)
        self._cudde_worker.status.connect(on_status)
        self._cudde_worker.finished.connect(on_download_complete)
        self._cudde_worker.start()
        self._cudde_dialog.show()

    # ─────────────────────────────────────────────────────────────────────
    # Supabase Cloud Sync
    # ─────────────────────────────────────────────────────────────────────

    def setup_supabase_credentials(self):
        """Show dialog to set up Supabase credentials."""
        settings = QSettings("TrailCam", "Trainer")
        current_url = settings.value("supabase_url", "")
        current_key = settings.value("supabase_key", "")

        dialog = QDialog(self)
        dialog.setWindowTitle("Supabase Cloud Setup")
        dialog.setMinimumWidth(450)
        layout = QVBoxLayout(dialog)

        # Instructions
        info_label = QLabel(
            "Enter your Supabase project credentials.\n"
            "Find these in your Supabase dashboard under Project Settings → API."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # URL field
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("Project URL:"))
        url_edit = QLineEdit()
        url_edit.setText(current_url)
        url_edit.setPlaceholderText("https://xxxxx.supabase.co")
        url_layout.addWidget(url_edit)
        layout.addLayout(url_layout)

        # API Key field
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("Anon Key:"))
        key_edit = QLineEdit()
        key_edit.setText(current_key)
        key_edit.setPlaceholderText("eyJhbGciOiJIUzI1NiIs...")
        key_layout.addWidget(key_edit)
        layout.addLayout(key_layout)

        # Note
        note_label = QLabel("Note: These credentials are stored locally on this computer.")
        note_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note_label)

        # Buttons
        btn_layout = QHBoxLayout()
        test_btn = QPushButton("Test Connection")
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(test_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        def test_connection():
            url = url_edit.text().strip()
            key = key_edit.text().strip()
            if not url or not key:
                QMessageBox.warning(dialog, "Supabase", "Please enter both URL and API key.")
                return

            # Show a "testing..." message
            test_btn.setEnabled(False)
            test_btn.setText("Testing...")
            QApplication.processEvents()

            try:
                from supabase_rest import create_client
                # Create client and test connection
                client = create_client(url, key)
                if client.test_connection():
                    QMessageBox.information(dialog, "Supabase", "Connection successful!")
                else:
                    raise Exception("Could not reach Supabase API")
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    error_msg = "Connection timed out. Check your internet connection."
                QMessageBox.warning(dialog, "Supabase", f"Connection failed:\n{error_msg}")
            finally:
                test_btn.setEnabled(True)
                test_btn.setText("Test Connection")

        def save_credentials():
            url = url_edit.text().strip()
            key = key_edit.text().strip()
            if not url or not key:
                QMessageBox.warning(dialog, "Supabase", "Please enter both URL and API key.")
                return
            settings.setValue("supabase_url", url)
            settings.setValue("supabase_key", key)
            QMessageBox.information(dialog, "Supabase", "Credentials saved successfully!")
            dialog.accept()

        test_btn.clicked.connect(test_connection)
        save_btn.clicked.connect(save_credentials)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def _get_supabase_client(self):
        """Get Supabase client, prompting for credentials if needed."""
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
            return create_client(url, key)
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

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            counts = self.db.push_to_supabase(client)
            QApplication.restoreOverrideCursor()
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
            QApplication.restoreOverrideCursor()
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

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            counts = self.db.pull_from_supabase(client)
            QApplication.restoreOverrideCursor()
            summary = (
                f"Pulled from cloud:\n\n"
                f"• Photos updated: {counts['photos']}\n"
                f"• Tags: {counts['tags']}\n"
                f"• Deer metadata: {counts['deer_metadata']}\n"
                f"• Additional deer: {counts['deer_additional']}\n"
                f"• Buck profiles: {counts['buck_profiles']}\n"
                f"• Season profiles: {counts['buck_profile_seasons']}"
            )
            QMessageBox.information(self, "Pull Complete", summary)
            # Reload current photo to show updated data
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()
            if self.photos:
                self.load_photo()
        except Exception as e:
            QApplication.restoreOverrideCursor()
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
        """Run AI suggestions on current view (all photos in list).

        Automatically runs detection first if no boxes exist, then uses
        subject/head crops for better classification accuracy.
        """
        if not self.suggester or not self.suggester.ready:
            QMessageBox.information(self, "AI Model Not Available", "AI model not loaded.")
            return
        # Filter to only unlabeled photos
        unlabeled_photos = [p for p in self.photos if not self._photo_has_human_species(p)]
        total = len(unlabeled_photos)

        if total == 0:
            QMessageBox.information(self, "AI Suggestions", "All photos in current view already have species labels.")
            return

        # Get detector for auto-detection if needed
        detector, names = self._get_detector_for_suggestions()
        species_count = 0
        sex_count = 0
        detect_count = 0

        # Create progress dialog
        progress = QProgressDialog("Running AI suggestions...", "Cancel", 0, total, self)
        progress.setWindowTitle("AI Suggestions")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        was_cancelled = False
        for i, p in enumerate(unlabeled_photos):
            # Update progress
            progress.setValue(i)
            progress.setLabelText(f"Processing photo {i + 1} of {total}...\nSpecies: {species_count} | Buck/Doe: {sex_count}")
            QApplication.processEvents()

            if progress.wasCanceled():
                was_cancelled = True
                break
            # Auto-detect boxes if none exist (enables head crops for buck/doe)
            self._ensure_detection_boxes(p, detector=detector, names=names)

            # Check detection boxes and auto-classify based on MegaDetector
            boxes = self.db.get_boxes(p["id"]) if p.get("id") else []
            has_person = any(b.get("label") == "ai_person" for b in boxes)
            has_vehicle = any(b.get("label") == "ai_vehicle" for b in boxes)
            has_animal = any(b.get("label") in ("ai_animal", "ai_subject", "subject") for b in boxes)

            label = None
            conf = None
            crop = None

            if has_person:
                # MegaDetector found person - auto-classify as Person
                label, conf = "Person", 0.95
            elif has_vehicle:
                # MegaDetector found vehicle - auto-classify as Vehicle
                label, conf = "Vehicle", 0.95
            elif has_animal:
                # Run classifier on animal crop
                detect_count += 1
                crop = self._best_crop_for_photo(p)
                path = str(crop) if crop else p.get("file_path")
                res = self.suggester.predict(path)
                if res:
                    label, conf = res
                    # If classifier says Empty but detector found animals, use Unknown
                    if label in ("Empty", "Other"):
                        label = "Unknown"
                        conf = 0.5
            else:
                # No detections at all - suggest Empty
                label, conf = "Empty", 0.95

            if label:
                self.db.set_suggested_tag(p["id"], label, conf)
                species_count += 1
                # If Deer, add deer head boxes and suggest buck/doe
                if label == "Deer":
                    self._add_deer_head_boxes(p, detector=detector, names=names)
                    if self.suggester.buckdoe_ready:
                        head_crop = self._best_head_crop_for_photo(p)
                        if head_crop:
                            sex_res = self.suggester.predict_sex(str(head_crop))
                            if sex_res:
                                sex_label, sex_conf = sex_res
                                self.db.set_suggested_sex(p["id"], sex_label, sex_conf)
                                sex_count += 1
                            try:
                                Path(head_crop).unlink(missing_ok=True)
                            except Exception:
                                pass
            if crop:
                try:
                    Path(crop).unlink(missing_ok=True)
                except Exception:
                    pass

        progress.setValue(total)
        progress.close()

        msg = f"Suggested species for {species_count} photo(s)."
        if sex_count > 0:
            msg += f"\nSuggested buck/doe for {sex_count} deer photo(s) (using head crops)."
        if was_cancelled:
            msg = f"Cancelled. " + msg
        QMessageBox.information(self, "AI Suggestions", msg)
        self._populate_species_dropdown()
        self._populate_photo_list()

    def run_ai_suggestions_all(self):
        """Run AI suggestions on all photos in DB (uses background thread).

        Automatically runs detection first if no boxes exist, then uses
        subject/head crops for better classification accuracy.
        """
        # Just call the background version - no need for blocking UI
        self.run_ai_suggestions_background()

    def run_sex_suggestions_on_deer(self):
        """Run buck/doe suggestions on photos already tagged as Deer.

        Automatically runs detection first if no boxes exist, then uses
        deer_head crops for best accuracy on buck/doe classification.
        """
        if not self.suggester or not self.suggester.buckdoe_ready:
            QMessageBox.information(self, "Buck/Doe Model Not Available",
                "Buck/doe classifier not loaded.\nPlace buckdoe.onnx in the models/ folder.")
            return
        # Find photos tagged as Deer but without buck/doe tags
        all_photos = self.db.get_all_photos()
        deer_photos = []
        for p in all_photos:
            tags = set(t.lower() for t in self.db.get_tags(p["id"]))
            is_deer = "deer" in tags
            has_sex = "buck" in tags or "doe" in tags
            # Check if already has suggestion
            has_sex_suggestion = bool(p.get("suggested_sex"))
            if is_deer and not has_sex and not has_sex_suggestion:
                deer_photos.append(p)
        if not deer_photos:
            QMessageBox.information(self, "Buck/Doe Suggestions",
                "No deer photos without buck/doe labels found.")
            return
        # Get detector for auto-detection if needed
        detector, names = self._get_detector_for_suggestions()
        count = 0
        count_fallback = 0
        for p in deer_photos:
            # Auto-detect boxes if none exist (enables head crops)
            self._ensure_detection_boxes(p, detector=detector, names=names)
            # Try deer_head crop first (more accurate)
            head_crop = self._best_head_crop_for_photo(p)
            if head_crop:
                sex_res = self.suggester.predict_sex(str(head_crop))
                if sex_res:
                    sex_label, sex_conf = sex_res
                    self.db.set_suggested_sex(p["id"], sex_label, sex_conf)
                    count += 1
                try:
                    Path(head_crop).unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                # Fallback: use full photo or subject crop (less accurate but better than nothing)
                crop = self._best_crop_for_photo(p)
                path = str(crop) if crop else p.get("file_path")
                if path and os.path.exists(path):
                    sex_res = self.suggester.predict_sex(path)
                    if sex_res:
                        sex_label, sex_conf = sex_res
                        # Lower confidence for fallback predictions
                        adjusted_conf = sex_conf * 0.7  # Reduce confidence since not using head crop
                        self.db.set_suggested_sex(p["id"], sex_label, adjusted_conf)
                        count += 1
                        count_fallback += 1
                if crop:
                    try:
                        Path(crop).unlink(missing_ok=True)
                    except Exception:
                        pass
        msg = f"Suggested buck/doe for {count} deer photo(s)."
        if count_fallback > 0:
            msg += f"\n({count_fallback} used full photo - may be less accurate)"
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

    def _exit_queue_mode(self):
        """Exit queue mode and return to normal view."""
        if not self.queue_mode:
            return

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
                            if idx is not None and self.photos[idx].get("id") == pid:
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

    def review_species_suggestions_integrated(self):
        """Review species suggestions using integrated queue mode."""
        pending = self._gather_pending_species_suggestions()
        if not pending:
            QMessageBox.information(self, "Species Suggestions", "No pending species suggestions to review.")
            return

        # Build queue data
        photo_ids = [p["id"] for p in pending]
        data = {p["id"]: {"species": p["species"], "conf": p["conf"]} for p in pending}

        self._enter_queue_mode(
            queue_type="species",
            photo_ids=photo_ids,
            data=data,
            title=f"Species Review ({len(pending)})"
        )

    # ====== BACKGROUND AI PROCESSING ======

    def run_ai_suggestions_background(self):
        """Run AI suggestions in background thread with live queue updates."""
        if self.ai_processing:
            QMessageBox.information(self, "AI Processing", "AI processing is already running.")
            return

        if not self.suggester or not self.suggester.ready:
            QMessageBox.information(self, "AI Model Not Available", "AI model not loaded.")
            return

        # Get unlabeled photos
        unlabeled_photos = [p for p in self.photos if not self._photo_has_human_species(p)]
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
        self.ai_worker = AIWorker(
            photos=unlabeled_photos,
            db=self.db,
            suggester=self.suggester,
            detector_getter=self._get_detector_for_suggestions,
            parent=self
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

            # Refresh photo list if in queue mode
            if self.queue_mode:
                self._populate_photo_list()

            # Update suggestion label
            remaining = len(self.queue_photo_ids) - len(self.queue_reviewed)
            self.queue_count_label.setText(f"{remaining} to review")

            # If this is the first photo, load it
            if len(self.queue_photo_ids) == 1:
                for i, p in enumerate(self.photos):
                    if p.get("id") == photo_id:
                        self.index = i
                        self.load_photo()
                        break

    def _on_ai_finished(self, species_count: int, sex_count: int):
        """Handle AI processing completion."""
        self.ai_processing = False
        self.ai_worker = None
        self.queue_progress_bar.hide()

        if self.queue_photo_ids:
            self.queue_title_label.setText(f"Species Review ({len(self.queue_photo_ids)})")
            self.queue_suggestion_label.setText(f"AI complete: {species_count} species, {sex_count} buck/doe")
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

    def review_species_suggestions(self):
        """Review and approve/reject pending species suggestions with zoomable photo preview."""
        from PyQt6.QtWidgets import QGraphicsView

        # Gather pending species suggestions
        pending = self._gather_pending_species_suggestions()
        if not pending:
            QMessageBox.information(self, "Species Suggestions", "No pending species suggestions to review.")
            return

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
        close_btn = QPushButton("Close")
        btn_row.addWidget(accept_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addWidget(skip_btn)
        btn_row.addWidget(multi_checkbox)
        btn_row.addWidget(next_btn)
        btn_row.addStretch()
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
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scene.clear()
                    scene.addPixmap(pixmap)
                    scene.setSceneRect(pixmap.rect().toRectF())
                    current_pixmap[0] = pixmap

                    # Draw boxes if available (skip boxes entirely in bottom 5% - timestamp area)
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

        # Connect species quick buttons
        for sp, btn in species_buttons.items():
            btn.clicked.connect(lambda checked, s=sp: _accept_as(s))

        # "Other (+)" handler to add new species
        def _add_custom_species():
            from PyQt6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(dlg, "Add New Species", "Enter species name:")
            if ok and name.strip():
                species_name = name.strip()
                # Save custom species to database
                if species_name not in SPECIES_OPTIONS and species_name not in VALID_SPECIES:
                    self.db.add_custom_species(species_name)
                _accept_as(species_name)
        other_btn.clicked.connect(_add_custom_species)

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
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Gesture:
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

    def _gather_pending_species_suggestions(self) -> list:
        """Return list of photos with pending species suggestions."""
        species_set = self._species_set()
        pending = []
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT id, original_name, file_path, suggested_tag, suggested_confidence
                FROM photos
                WHERE suggested_tag IS NOT NULL AND suggested_tag != ''
                ORDER BY suggested_tag ASC, suggested_confidence DESC
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

        def _add_custom_species():
            """Prompt for new species name, label photo, and add quick button."""
            from PyQt6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(dlg, "Add New Species", "Enter species name:")
            if ok and name.strip():
                species_name = name.strip()
                # Save custom species to database so it's recognized by filters
                if species_name not in SPECIES_OPTIONS and species_name not in VALID_SPECIES:
                    self.db.add_custom_species(species_name)
                # Label the current photo
                _label_species(species_name)
                # Add button if not already added
                if species_name not in custom_species and species_name not in species_buttons:
                    custom_species.append(species_name)
                    new_btn = QPushButton(species_name)
                    new_btn.setStyleSheet("padding: 6px 12px; background: #363;")
                    new_btn.clicked.connect(lambda checked, s=species_name: _label_species(s))
                    # Insert before the stretch
                    species_btn_row3.insertWidget(species_btn_row3.count() - 1, new_btn)

        for sp, btn in species_buttons.items():
            btn.clicked.connect(lambda checked, s=sp: _label_species(s))

        other_btn.clicked.connect(_add_custom_species)
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

        left_panel.addWidget(QLabel("Pending suggestions:"))
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setMaximumWidth(280)

        all_pending = pending  # Store original list for filtering

        def _populate_list():
            """Populate list based on current filters."""
            list_widget.clear()
            sex_val = sex_filter.currentText()
            conf_val = conf_filter.currentText()
            date_val = date_filter.currentText()

            for item in all_pending:
                # Skip if already reviewed
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
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        current_pixmap = [None]  # Store current pixmap for zoom operations

        def _update_preview():
            item = list_widget.currentItem()
            if not item:
                scene.clear()
                suggestion_label.setText("Suggestion: —")
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            path = data.get("path")
            sex = data.get("sex", "").title()
            conf = data.get("conf", 0)
            pct = int(conf * 100) if conf <= 1 else int(conf)
            suggestion_label.setText(f"AI Suggestion: {sex} ({pct}% confidence)")
            if path and os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scene.clear()
                    scene.addPixmap(pixmap)
                    scene.setSceneRect(pixmap.rect().toRectF())
                    current_pixmap[0] = pixmap
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

        reviewed_ids = set()  # Track reviewed photo IDs
        reviewed_data = {}  # Track what action was taken: {pid: action}

        def _mark_reviewed(item, action_text: str):
            """Mark item as reviewed with green highlight and update text."""
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data["id"]
            reviewed_ids.add(pid)
            original_name = data.get("name", "")[:22]
            item.setText(f"✓ {action_text} - {original_name}")
            item.setBackground(QColor(144, 238, 144))  # Light green
            # Update window title with remaining count
            remaining = sum(1 for i in range(list_widget.count())
                          if list_widget.item(i).data(Qt.ItemDataRole.UserRole)["id"] not in reviewed_ids)
            dlg.setWindowTitle(f"Review Buck/Doe Suggestions ({remaining} remaining)")

        def _next_unreviewed():
            """Move to next unreviewed item."""
            current = list_widget.currentRow()
            # Look forward first
            for i in range(current + 1, list_widget.count()):
                item_data = list_widget.item(i).data(Qt.ItemDataRole.UserRole)
                if item_data["id"] not in reviewed_ids:
                    list_widget.setCurrentRow(i)
                    _update_preview()
                    return
            # Then look from beginning
            for i in range(0, current):
                item_data = list_widget.item(i).data(Qt.ItemDataRole.UserRole)
                if item_data["id"] not in reviewed_ids:
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
            pid = data["id"]
            ai_suggested = data.get("sex", "")
            # Log if user corrected the AI (accepted different label)
            if ai_suggested and ai_suggested != sex_tag:
                self.db.log_ai_rejection(
                    photo_id=pid,
                    suggestion_type="sex",
                    ai_suggested=ai_suggested,
                    correct_label=sex_tag,  # What user said was correct
                    model_version=get_model_version("buckdoe")
                )
            # Add tag
            tags = set(self.db.get_tags(pid))
            if sex_tag not in tags:
                self.db.add_tag(pid, sex_tag)
            # Track for clearing suggestion on close
            reviewed_data[pid] = sex_tag
            # Mark as reviewed (green) instead of removing
            _mark_reviewed(item, sex_tag)
            _next_unreviewed()

        def _reject():
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data["id"]
            ai_suggested = data.get("sex", "")
            # Log rejection for future model training
            if ai_suggested:
                self.db.log_ai_rejection(
                    photo_id=pid,
                    suggestion_type="sex",
                    ai_suggested=ai_suggested,
                    correct_label=None,  # Unknown - user just rejected
                    model_version=get_model_version("buckdoe")
                )
            # Track for clearing suggestion on close
            reviewed_data[pid] = "REJECTED"
            # Mark as reviewed (green) instead of removing
            _mark_reviewed(item, "REJECTED")
            _next_unreviewed()

        def _skip():
            row = list_widget.currentRow()
            if row < list_widget.count() - 1:
                list_widget.setCurrentRow(row + 1)
                _update_preview()

        def _unknown():
            """Mark as unknown - clear suggestion without adding buck/doe tag."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data["id"]
            # Track for clearing suggestion on close
            reviewed_data[pid] = "UNKNOWN"
            # Mark as reviewed (green)
            _mark_reviewed(item, "UNKNOWN")
            _next_unreviewed()

        def _open_properties():
            """Navigate to photo in main window and close review dialog."""
            item = list_widget.currentItem()
            if not item:
                return
            data = item.data(Qt.ItemDataRole.UserRole)
            pid = data["id"]
            # Find photo index in main window
            for i, photo in enumerate(self.photos):
                if photo["id"] == pid:
                    self.index = i
                    break
            # Close the dialog - cleanup will run via finished signal
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
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Gesture:
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
        self._populate_photo_list()
        if self.photos and 0 <= self.index < len(self.photos):
            self.load_photo()

    def _gather_pending_sex_suggestions(self) -> list:
        """Return list of photos with pending buck/doe suggestions."""
        # Non-deer species tags - if photo has any of these, skip buck/doe suggestion
        non_deer_species = {"rabbit", "turkey", "coyote", "raccoon", "squirrel",
                           "opossum", "bobcat", "quail", "person", "vehicle",
                           "empty", "other", "other mammal", "other bird"}
        pending = []
        for p in self.db.get_all_photos():
            pid = p["id"]
            tags = set(t.lower() for t in self.db.get_tags(pid))
            has_sex = "buck" in tags or "doe" in tags
            # Skip if already has non-deer species tag
            has_non_deer = bool(tags & non_deer_species)
            sugg_sex = p.get("suggested_sex") or ""
            sugg_conf = p.get("suggested_sex_confidence")
            if sugg_sex and not has_sex and not has_non_deer:
                pending.append({
                    "id": pid,
                    "name": p.get("original_name") or Path(p.get("file_path", "")).name,
                    "sex": sugg_sex,
                    "conf": sugg_conf or 0,
                    "path": p.get("file_path"),
                    "date": p.get("taken_date") or p.get("import_date") or "",
                })
        pending.sort(key=lambda x: x["conf"], reverse=True)
        return pending

    def run_ai_boxes(self):
        """Run detector to propose subject boxes (AI-labeled). Prefer custom ONNX, fall back to torchvision."""
        if not self.current_pixmap or not self.photos:
            return
        photo = self.photos[self.index]
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
        return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

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
        """Fill species dropdown with defaults + actually used custom species (from applied tags only)."""
        self.species_combo.blockSignals(True)
        current = self.species_combo.currentText()
        self.species_combo.clear()
        # defaults always available
        for s in SPECIES_OPTIONS:
            self.species_combo.addItem(s)
        # gather used species from applied tags AND custom_species table
        used = set()
        try:
            # Include all custom species
            for s in self.db.list_custom_species():
                if s and s.lower() not in SEX_TAGS:
                    used.add(s)
            # Also scan tags for any species-like tags not in custom_species
            # (catches species that were saved but not registered)
            all_tags = self.db.get_all_distinct_tags()
            skip_tags = SEX_TAGS | {"buck", "doe", "favorite"}
            for t in all_tags:
                if t and t.lower() not in skip_tags and t not in SPECIES_OPTIONS:
                    used.add(t)
        except Exception:
            pass
        for s in sorted(used):
            if s not in SPECIES_OPTIONS:
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
                    return datetime.fromisoformat(val)
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
        self.species_combo.setCurrentText(species)
        # save_current() now handles multi-species mode automatically
        self.schedule_save()

    def _on_other_species_clicked(self):
        """Open dialog to type a custom species name."""
        from PyQt6.QtWidgets import QInputDialog
        species, ok = QInputDialog.getText(
            self, "Custom Species", "Enter species name:",
            text=""
        )
        if ok and species and len(species.strip()) >= 3:
            species = species.strip()
            self.species_combo.setCurrentText(species)
            # Add to custom species if not already known
            if species not in SPECIES_OPTIONS:
                self.db.add_custom_species(species)
                self._populate_species_dropdown()
            # save_current() now handles multi-species mode automatically
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
        if not self.photos:
            return
        species = self.species_combo.currentText().strip()
        if not species or species.lower() in SEX_TAGS:
            return
        photo = self.photos[self.index]
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
            # Persist custom species if needed (only if length >= 3)
            if species not in SPECIES_OPTIONS and len(species) >= 3:
                self.db.add_custom_species(species)
                self._populate_species_dropdown()
            self._update_recent_species_buttons()
            # Refresh photo data
            self.photos[self.index].update(self.db.get_photo_by_id(pid) or {})
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
        close = QPushButton("Close")
        btns.addWidget(open_btn)
        btns.addWidget(compare_btn)
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

        open_btn.clicked.connect(open_selected)
        compare_btn.clicked.connect(compare_selected)
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

    # ========== Site Clustering ==========

    def run_site_clustering(self):
        """Auto-detect camera sites using image similarity clustering."""
        try:
            from site_clustering import SiteClusterer, run_site_clustering, suggest_cluster_parameters
        except ImportError as e:
            QMessageBox.warning(self, "Site Clustering", f"Failed to load clustering module: {e}")
            return

        # Check if there are already sites assigned
        existing_sites = self.db.get_all_sites()
        unassigned = self.db.get_unassigned_photo_count()
        total_photos = len(self.db.get_all_photos())

        if existing_sites:
            msg = (f"Found {len(existing_sites)} existing site(s) with {total_photos - unassigned} "
                   f"photos assigned.\n\n{unassigned} photos are unassigned.\n\n"
                   "What would you like to do?")
            btn = QMessageBox.question(
                self, "Site Clustering",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            btn_text = {
                QMessageBox.StandardButton.Yes: "Cluster unassigned only",
                QMessageBox.StandardButton.No: "Re-cluster all photos (clears existing sites)",
            }
            if btn == QMessageBox.StandardButton.Cancel:
                return
            if btn == QMessageBox.StandardButton.No:
                # Clear existing sites
                confirm = QMessageBox.question(
                    self, "Clear Sites",
                    "This will delete all existing sites and re-cluster all photos. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return
                for site in existing_sites:
                    self.db.delete_site(site["id"])
                self.db.clear_all_embeddings()

        # Get suggested parameters
        params = suggest_cluster_parameters(self.db)
        info_msg = (
            f"Ready to analyze {params['photo_count']} photos.\n\n"
            "This will:\n"
            "1. Extract visual features from each photo (may take a few minutes)\n"
            "2. Cluster similar backgrounds together\n"
            "3. Create site SUGGESTIONS for you to review\n\n"
            "Continue?"
        )
        if QMessageBox.question(self, "Site Clustering", info_msg,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                ) != QMessageBox.StandardButton.Yes:
            return

        # Run clustering with progress dialog
        progress = QProgressDialog("Analyzing photos...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Auto-Detect Sites")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.show()
        QCoreApplication.processEvents()

        def progress_cb(current, total, msg):
            if total > 0:
                progress.setMaximum(total)
                progress.setValue(current)
            progress.setLabelText(msg)
            QCoreApplication.processEvents()
            return not progress.wasCanceled()

        try:
            clusterer = SiteClusterer()
            if not clusterer.ready:
                progress.close()
                QMessageBox.warning(self, "Site Clustering",
                    "Clustering model not available. Check torch installation.")
                return

            result = run_site_clustering(
                self.db,
                clusterer=clusterer,
                eps=params["eps"],
                min_samples=params["min_samples"],
                confidence_threshold=params.get("confidence_threshold", 0.7),
                incremental=True,
                progress_callback=progress_cb
            )
            progress.close()

            if "error" in result:
                QMessageBox.warning(self, "Site Clustering", result["error"])
                return

            # Build detailed results message
            matched_existing = result.get('photos_matched_existing', 0)
            low_conf = result.get('low_confidence_count', 0)
            msg = (f"Site clustering complete!\n\n"
                   f"⭐ Site suggestions created: {result['sites_created']}\n"
                   f"Photos suggested (high confidence): {result['photos_assigned']}\n")
            if matched_existing > 0:
                msg += f"Matched to existing sites: {matched_existing}\n"
            if low_conf > 0:
                msg += f"Low confidence (kept unassigned): {low_conf}\n"
            msg += f"Unassigned (need more data): {result['noise_count']}\n\n"
            msg += "Go to Tools → Manage Sites to review and confirm suggestions.\n"
            msg += "Run again later to assign more photos as confidence grows."
            QMessageBox.information(self, "Site Clustering", msg)

            # Refresh UI
            self._populate_site_filter_options()
            self.photos = self._sorted_photos(self.db.get_all_photos())
            self._populate_photo_list()

        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Site Clustering", f"Clustering failed: {e}")

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
    win = TrainerWindow()
    win.resize(1280, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
