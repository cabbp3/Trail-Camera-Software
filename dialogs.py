"""
Extracted dialog classes from training/label_tool.py.

These are standalone dialogs that don't depend on TrainerWindow state.
"""
import json
import os

from PyQt6.QtCore import Qt, pyqtSignal, QSettings
from PyQt6.QtGui import QPixmap, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

class AIOptionsDialog(QDialog):
    """Dialog for selecting AI processing options."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Processing Options")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # === Photo Scope Section ===
        scope_label = QLabel("<b>Which photos to process:</b>")
        layout.addWidget(scope_label)

        self.scope_group = QButtonGroup(self)

        self.scope_no_suggestions = QCheckBox("Photos without any AI suggestions")
        self.scope_no_suggestions.setChecked(True)
        self.scope_group.addButton(self.scope_no_suggestions, 0)
        layout.addWidget(self.scope_no_suggestions)

        self.scope_all_unlabeled = QCheckBox("All unlabeled photos (re-run existing suggestions)")
        self.scope_group.addButton(self.scope_all_unlabeled, 1)
        layout.addWidget(self.scope_all_unlabeled)

        # Make them mutually exclusive like radio buttons
        self.scope_no_suggestions.toggled.connect(lambda checked: self.scope_all_unlabeled.setChecked(not checked) if checked else None)
        self.scope_all_unlabeled.toggled.connect(lambda checked: self.scope_no_suggestions.setChecked(not checked) if checked else None)

        layout.addSpacing(15)

        # === AI Steps Section ===
        steps_label = QLabel("<b>AI steps to run:</b>")
        layout.addWidget(steps_label)

        self.step_detect_boxes = QCheckBox("Detect subject boxes (MegaDetector)")
        self.step_detect_boxes.setChecked(True)
        self.step_detect_boxes.setToolTip("Run MegaDetector to find animals/people/vehicles in photos")
        layout.addWidget(self.step_detect_boxes)

        self.step_species_id = QCheckBox("Identify species")
        self.step_species_id.setChecked(True)
        self.step_species_id.setToolTip("Classify detected animals by species (Deer, Turkey, Raccoon, etc.)")
        layout.addWidget(self.step_species_id)

        self.step_deer_head_boxes = QCheckBox("Detect deer head boxes")
        self.step_deer_head_boxes.setChecked(True)
        self.step_deer_head_boxes.setToolTip("Find deer heads in Deer photos for buck/doe classification")
        layout.addWidget(self.step_deer_head_boxes)

        self.step_buck_doe = QCheckBox("Identify buck vs doe")
        self.step_buck_doe.setChecked(True)
        self.step_buck_doe.setToolTip("Classify deer as Buck or Doe using head crops")
        layout.addWidget(self.step_buck_doe)

        layout.addSpacing(15)

        # === Buttons ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_options(self) -> dict:
        """Return the selected options."""
        return {
            "scope": "no_suggestions" if self.scope_no_suggestions.isChecked() else "all_unlabeled",
            "detect_boxes": self.step_detect_boxes.isChecked(),
            "species_id": self.step_species_id.isChecked(),
            "deer_head_boxes": self.step_deer_head_boxes.isChecked(),
            "buck_doe": self.step_buck_doe.isChecked(),
        }


class TightwadComparisonDialog(QDialog):
    """Dialog for reviewing Tightwad House photos where AI disagrees with existing labels."""
    photo_selected = pyqtSignal(int)  # Emits photo_id when user wants to view in main app

    def __init__(self, parent, items: list, db, queue_file: str):
        super().__init__(parent)
        self.items = items
        self.db = db
        self.queue_file = queue_file
        self.current_index = 0

        self.setWindowTitle(f"Tightwad Comparison Review ({len(items)} photos)")
        self.resize(900, 700)

        layout = QVBoxLayout(self)

        # Header with progress
        header = QHBoxLayout()
        self.progress_label = QLabel(f"1 / {len(items)}")
        self.progress_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        header.addWidget(self.progress_label)
        header.addStretch()

        skip_btn = QPushButton("Skip")
        skip_btn.clicked.connect(self._skip)
        header.addWidget(skip_btn)

        layout.addLayout(header)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("background-color: #222;")
        layout.addWidget(self.image_label, 1)

        # Comparison info
        info_frame = QFrame()
        info_frame.setStyleSheet("QFrame { background-color: #333; border-radius: 6px; padding: 10px; }")
        info_layout = QHBoxLayout(info_frame)

        self.old_label = QLabel()
        self.old_label.setStyleSheet("font-size: 16px; color: #f88;")
        self.ai_label = QLabel()
        self.ai_label.setStyleSheet("font-size: 16px; color: #8f8;")

        info_layout.addWidget(QLabel("Old Label:"))
        info_layout.addWidget(self.old_label)
        info_layout.addStretch()
        info_layout.addWidget(QLabel("AI Suggestion:"))
        info_layout.addWidget(self.ai_label)

        layout.addWidget(info_frame)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        self.accept_ai_btn = QPushButton("Accept AI (A)")
        self.accept_ai_btn.setStyleSheet("background-color: #4a4; color: white; font-weight: bold; font-size: 14px; padding: 10px 20px;")
        self.accept_ai_btn.clicked.connect(self._accept_ai)

        self.keep_old_btn = QPushButton("Keep Old (K)")
        self.keep_old_btn.setStyleSheet("background-color: #44a; color: white; font-weight: bold; font-size: 14px; padding: 10px 20px;")
        self.keep_old_btn.clicked.connect(self._keep_old)

        self.clear_btn = QPushButton("Clear (C)")
        self.clear_btn.setStyleSheet("background-color: #a44; color: white; font-weight: bold; font-size: 14px; padding: 10px 20px;")
        self.clear_btn.clicked.connect(self._clear_label)

        self.view_btn = QPushButton("View in App (V)")
        self.view_btn.setStyleSheet("background-color: #666; color: white; font-size: 14px; padding: 10px 20px;")
        self.view_btn.clicked.connect(self._view_in_app)

        btn_layout.addWidget(self.accept_ai_btn)
        btn_layout.addWidget(self.keep_old_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.view_btn)

        layout.addLayout(btn_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(items))
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Keyboard shortcuts
        QShortcut(QKeySequence("A"), self, self._accept_ai)
        QShortcut(QKeySequence("K"), self, self._keep_old)
        QShortcut(QKeySequence("C"), self, self._clear_label)
        QShortcut(QKeySequence("V"), self, self._view_in_app)
        QShortcut(QKeySequence("Space"), self, self._skip)
        QShortcut(QKeySequence("Right"), self, self._skip)
        QShortcut(QKeySequence("Left"), self, self._prev)

        self._load_current()

    def _load_current(self):
        """Load the current photo for review."""
        if self.current_index >= len(self.items):
            self._finish()
            return

        item = self.items[self.current_index]
        self.progress_label.setText(f"{self.current_index + 1} / {len(self.items)}")
        self.progress_bar.setValue(self.current_index)

        # Load image
        file_path = item.get('file_path', '')
        if os.path.exists(file_path):
            pixmap = QPixmap(file_path)
            scaled = pixmap.scaled(
                self.image_label.width() - 20,
                self.image_label.height() - 20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.setText("Image not found")

        # Update labels
        old_label = item.get('existing_label', 'Unknown')
        ai_label = item.get('ai_suggestion', 'Unknown')
        confidence = item.get('ai_confidence', 0)

        self.old_label.setText(f"<b>{old_label}</b>")
        self.ai_label.setText(f"<b>{ai_label}</b> ({confidence:.0%})")

    def _accept_ai(self):
        """Accept the AI suggestion - replace old label with AI label."""
        if self.current_index >= len(self.items):
            return
        item = self.items[self.current_index]
        photo_id = item['photo_id']
        ai_species = item['ai_suggestion']

        # Route through DB methods for proper audit logging
        self.db.update_photo_tags(photo_id, [ai_species])
        self.db.set_suggested_tag(photo_id, None, None)

        self._remove_and_next()

    def _keep_old(self):
        """Keep the old label - just clear the AI suggestion."""
        if self.current_index >= len(self.items):
            return
        item = self.items[self.current_index]
        photo_id = item['photo_id']

        # Clear suggestion, keep existing tags
        self.db.set_suggested_tag(photo_id, None, None)

        self._remove_and_next()

    def _clear_label(self):
        """Clear both labels - user will manually label."""
        if self.current_index >= len(self.items):
            return
        item = self.items[self.current_index]
        photo_id = item['photo_id']

        # Route through DB methods for proper audit logging
        self.db.update_photo_tags(photo_id, [])
        self.db.set_suggested_tag(photo_id, None, None)

        self._remove_and_next()

    def _view_in_app(self):
        """Emit signal to view this photo in the main app."""
        if self.current_index >= len(self.items):
            return
        item = self.items[self.current_index]
        self.photo_selected.emit(item['photo_id'])

    def _skip(self):
        """Skip to next without making changes."""
        self.current_index += 1
        self._load_current()

    def _prev(self):
        """Go back to previous photo."""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current()

    def _remove_and_next(self):
        """Remove current item from queue and move to next."""
        if self.current_index < len(self.items):
            del self.items[self.current_index]
            self._save_queue()
        # Don't increment index since we removed current item
        self._load_current()

    def _save_queue(self):
        """Save updated queue to file."""
        with open(self.queue_file, 'w') as f:
            json.dump(self.items, f, indent=2)

    def _finish(self):
        """Called when all items are reviewed."""
        QMessageBox.information(self, "Complete", "All photos in this queue have been reviewed!")
        # Remove empty queue file
        if os.path.exists(self.queue_file) and not self.items:
            os.remove(self.queue_file)
        self.accept()


class UserSetupDialog(QDialog):
    """Dialog for username and hunting club selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from user_config import (
            get_username, set_username, get_hunting_clubs, set_hunting_clubs,
            get_available_clubs, add_club, is_admin, set_admin
        )

        self.setWindowTitle("Welcome to Trail Camera Organizer")
        self.setMinimumWidth(400)
        self.setModal(True)
        layout = QVBoxLayout(self)

        # Welcome message
        welcome = QLabel("Welcome! Please enter your name and select your club(s).")
        welcome.setWordWrap(True)
        welcome.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(welcome)

        # Username field
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Your Name:"))
        name_edit = QLineEdit()
        name_edit.setText(get_username() or "")
        name_edit.setPlaceholderText("Enter your name")
        name_layout.addWidget(name_edit)
        layout.addLayout(name_layout)

        # Club selection with checkboxes (multi-select)
        club_label = QLabel("Select your club(s):")
        layout.addWidget(club_label)

        clubs = get_available_clubs()
        current_clubs = get_hunting_clubs()
        club_checkboxes = {}

        club_group = QWidget()
        club_layout = QVBoxLayout(club_group)
        club_layout.setContentsMargins(20, 0, 0, 0)

        for club in clubs:
            cb = QCheckBox(club)
            cb.setChecked(club in current_clubs)
            club_checkboxes[club] = cb
            club_layout.addWidget(cb)

        layout.addWidget(club_group)

        # Add new club field
        new_club_layout = QHBoxLayout()
        new_club_layout.addWidget(QLabel("Add new club:"))
        new_club_edit = QLineEdit()
        new_club_edit.setPlaceholderText("Type new club name and press Enter")
        new_club_layout.addWidget(new_club_edit)
        layout.addLayout(new_club_layout)

        def add_new_club():
            new_club = new_club_edit.text().strip()
            if new_club and new_club not in club_checkboxes:
                cb = QCheckBox(new_club)
                cb.setChecked(True)
                club_checkboxes[new_club] = cb
                club_layout.addWidget(cb)
                new_club_edit.clear()

        new_club_edit.returnPressed.connect(add_new_club)

        # Admin checkbox (hidden by default, shown with secret key)
        admin_check = QCheckBox("Admin mode (view all clubs)")
        admin_check.setChecked(is_admin())
        admin_check.setVisible(False)  # Hidden until secret key
        layout.addWidget(admin_check)

        # Info label
        info = QLabel("Select one or more clubs to see their photos.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 11px; margin-top: 10px;")
        layout.addWidget(info)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("Continue")
        save_btn.setDefault(True)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        # Secret: Ctrl+Shift+A to show admin option
        def keyPressEvent(event):
            if (event.modifiers() == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
                and event.key() == Qt.Key.Key_A):
                admin_check.setVisible(True)
            QDialog.keyPressEvent(self, event)
        self.keyPressEvent = keyPressEvent

        def save_and_close():
            name = name_edit.text().strip()

            if not name or len(name) < 2:
                QMessageBox.warning(self, "Setup", "Please enter your name (at least 2 characters).")
                return

            # Get selected clubs
            selected_clubs = [club for club, cb in club_checkboxes.items() if cb.isChecked()]

            if not selected_clubs and not admin_check.isChecked():
                QMessageBox.warning(self, "Setup", "Please select at least one club (or enable admin mode).")
                return

            set_username(name)
            set_hunting_clubs(selected_clubs)
            set_admin(admin_check.isChecked())

            # Add any new clubs to the available list
            for club in selected_clubs:
                if club not in clubs:
                    add_club(club)

            self.accept()

        save_btn.clicked.connect(save_and_close)


class SupabaseCredentialsDialog(QDialog):
    """Dialog to set up Supabase credentials."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self._settings = settings or QSettings("TrailCam", "Trainer")
        current_url = self._settings.value("supabase_url", "")
        current_key = self._settings.value("supabase_key", "")

        self.setWindowTitle("Supabase Cloud Setup")
        self.setMinimumWidth(450)
        layout = QVBoxLayout(self)

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
        self.url_edit = QLineEdit()
        self.url_edit.setText(current_url)
        self.url_edit.setPlaceholderText("https://xxxxx.supabase.co")
        url_layout.addWidget(self.url_edit)
        layout.addLayout(url_layout)

        # API Key field
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("Anon Key:"))
        self.key_edit = QLineEdit()
        self.key_edit.setText(current_key)
        self.key_edit.setPlaceholderText("eyJhbGciOiJIUzI1NiIs...")
        key_layout.addWidget(self.key_edit)
        layout.addLayout(key_layout)

        # Note
        note_label = QLabel("Note: These credentials are stored locally on this computer.")
        note_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.test_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        self.test_btn.clicked.connect(self._test_connection)
        save_btn.clicked.connect(self._save_credentials)
        cancel_btn.clicked.connect(self.reject)

    def _test_connection(self):
        url = self.url_edit.text().strip()
        key = self.key_edit.text().strip()
        if not url or not key:
            QMessageBox.warning(self, "Supabase", "Please enter both URL and API key.")
            return

        # Show a "testing..." message
        self.test_btn.setEnabled(False)
        self.test_btn.setText("Testing...")
        QApplication.processEvents()

        try:
            from supabase_rest import create_client
            # Create client and test connection
            client = create_client(url, key)
            if client.test_connection():
                QMessageBox.information(self, "Supabase", "Connection successful!")
            else:
                raise Exception("Could not reach Supabase API")
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                error_msg = "Connection timed out. Check your internet connection."
            QMessageBox.warning(self, "Supabase", f"Connection failed:\n{error_msg}")
        finally:
            self.test_btn.setEnabled(True)
            self.test_btn.setText("Test Connection")

    def _save_credentials(self):
        url = self.url_edit.text().strip()
        key = self.key_edit.text().strip()
        if not url or not key:
            QMessageBox.warning(self, "Supabase", "Please enter both URL and API key.")
            return
        self._settings.setValue("supabase_url", url)
        self._settings.setValue("supabase_key", key)
        QMessageBox.information(self, "Supabase", "Credentials saved successfully!")
        self.accept()


class CuddeLinkCredentialsDialog(QDialog):
    """Dialog to set up CuddeLink credentials."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self._settings = settings or QSettings("TrailCam", "Trainer")
        current_email = self._settings.value("cuddelink_email", "")

        self.setWindowTitle("CuddeLink Credentials")
        self.setMinimumWidth(350)
        layout = QVBoxLayout(self)

        # Instructions
        info_label = QLabel("Enter your CuddeLink account credentials.\nThese will be saved for future downloads.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Email field
        email_layout = QHBoxLayout()
        email_layout.addWidget(QLabel("Email:"))
        self.email_edit = QLineEdit()
        self.email_edit.setText(current_email)
        self.email_edit.setPlaceholderText("your@email.com")
        email_layout.addWidget(self.email_edit)
        layout.addLayout(email_layout)

        # Password field
        pass_layout = QHBoxLayout()
        pass_layout.addWidget(QLabel("Password:"))
        self.pass_edit = QLineEdit()
        self.pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.pass_edit.setPlaceholderText("Enter password")
        pass_layout.addWidget(self.pass_edit)
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

        save_btn.clicked.connect(self._save_credentials)
        cancel_btn.clicked.connect(self.reject)
        test_btn.clicked.connect(self._test_connection)

    def _save_credentials(self):
        email = self.email_edit.text().strip()
        password = self.pass_edit.text()
        if not email or not password:
            QMessageBox.warning(self, "CuddeLink", "Please enter both email and password.")
            return
        self._settings.setValue("cuddelink_email", email)
        self._settings.setValue("cuddelink_password", password)
        QMessageBox.information(self, "CuddeLink", "Credentials saved successfully!")
        self.accept()

    def _test_connection(self):
        email = self.email_edit.text().strip()
        password = self.pass_edit.text()
        if not email or not password:
            QMessageBox.warning(self, "CuddeLink", "Please enter both email and password.")
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
                QMessageBox.warning(self, "CuddeLink", "Invalid email or password.")
                return
            # Check if we're logged in by accessing photos page
            photos_resp = session.get("https://camp.cuddeback.com/photos", timeout=20)
            if "login" in photos_resp.url.lower() or photos_resp.status_code >= 400:
                QMessageBox.warning(self, "CuddeLink", "Login failed. Please check your credentials.")
            else:
                QMessageBox.information(self, "CuddeLink", "Connection successful! Credentials are valid.")
        except Exception as e:
            QMessageBox.warning(self, "CuddeLink", f"Connection failed: {str(e)}")
