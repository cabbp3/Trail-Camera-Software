# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role & Communication Style

You are a **senior software engineer** specializing in building useful, not-complicated software. When starting a session:

1. **Read relevant files** to gain context about what's being worked on
2. **Explain progress and updates at a high level** - the user is a non-technical founder
3. **Avoid jargon** - use plain English when possible
4. **For any actions the user needs to take** (setting up databases, getting API keys, configuring services, etc.), provide **clear step-by-step instructions** so they can complete them independently

Keep solutions simple and focused. Prioritize working software over perfect architecture.

---

## Product Vision

This software is intended to eventually become a **marketable product** for hunters and wildlife enthusiasts who manage trail cameras. Keep this in mind when making design decisions:

- **User experience matters** - features should be intuitive for non-technical users
- **Reliability is critical** - data loss or corruption is unacceptable
- **Flexibility for different setups** - users have varying camera brands, naming conventions, and workflows
- **Not all features may ship** - some functionality (like training pipelines) may remain internal tools

### Multi-Platform Goal

The long-term vision is availability across **all major platforms**:
- **Desktop:** Windows, macOS (standalone apps, no Python/C++ install required)
- **Mobile:** Android, iPhone
- **Web:** Browser-based access

These don't need to be the same app, but should share data via cloud sync (Supabase).

### Technical Decisions to Support Multi-Platform

When making architecture decisions, keep these principles in mind:

1. **Avoid dependencies that require compilation** - The `supabase` Python package requires C++ Build Tools on Windows. Consider replacing with direct REST API calls using `requests` to eliminate this barrier.

2. **Keep business logic separate from UI** - Database operations, API calls, and AI inference should be in standalone modules that could be reused or ported.

3. **Use Supabase as the central data layer** - All platforms can sync through the same Supabase backend.

4. **Consider web-first for new features** - Features that don't need local file access (viewing labels, buck profiles, reports) could be web-based and work everywhere.

---

## Project Overview

Trail Camera Photo Organizer - A PyQt6 desktop application for organizing trail camera photos with AI-assisted wildlife labeling, deer tracking, and custom model training.

## Commands

### Run the Application (macOS)
```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
source .venv/bin/activate
python main.py
```

### Run the Application (Windows)
```cmd
cd C:\Users\mbroo\OneDrive\Desktop\Trail Camera Software V 1.0
venv\Scripts\activate
python main.py
```

**Windows Status (Dec 25, 2024):**
- App runs and all core features work
- Supabase cloud sync FIXED - replaced supabase package with REST API (supabase_rest.py)
- No C++ Build Tools needed anymore
- **IMPORTANT:** Use Python 3.11 (not 3.14) - newer versions lack package support
- McAfee/antivirus may block the .exe - add to exclusions or restore from quarantine

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Build Windows Executable
```cmd
# On Windows - double-click BUILD.bat (recommended), or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller TrailCamOrganizer_Windows.spec --clean
# Output: dist\TrailCamOrganizer\TrailCamOrganizer.exe
```

**Build files included:**
- `BUILD.bat` - One-click build script (creates venv, installs deps, builds exe)
- `RUN.bat` - Launch the built app
- `Create Desktop Shortcut.vbs` - Creates Windows desktop shortcut
- `icon.ico` - Windows app icon (already created)

**Requirements:**
- Python 3.11 (NOT 3.14 - too new, packages don't support it yet)
- Check "Add Python to PATH" during Python installation

**Antivirus Note:** McAfee and other antivirus may quarantine the .exe. Add the folder to exclusions or restore from quarantine after build.

### Build macOS Standalone App
```bash
# On macOS:
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
source .venv/bin/activate
pip3 install pyinstaller
python3 -m PyInstaller TrailCamOrganizer_macOS.spec --clean
# Output: dist/TrailCamOrganizer.app (489MB standalone)
```

**Note:** First-time launch may show "unidentified developer" warning. Right-click → Open to bypass.

### Training Models (requires additional dependencies)
```bash
# Install training dependencies
pip install ultralytics torch torchvision timm pytorch-metric-learning pyyaml onnx onnxruntime

# Train detector
python training/train_detector.py --config training/configs/detector.yaml

# Train classifier
python training/train_classifier.py --config training/configs/classifier.yaml

# Train re-ID model
python training/train_reid.py --config training/configs/reid.yaml

# Export models to app
python training/export_to_app.py --detector outputs/detector.onnx --labels outputs/detector_labels.txt
```

## Architecture

### Entry Point
- `main.py` → Launches `TrainerWindow` from `training/label_tool.py` (the primary UI)

### Core Modules
- **training/label_tool.py** (3500+ lines) - Primary UI for photo labeling with species/deer ID/antler tagging
- **database.py** - SQLite operations; schema includes `photos`, `tags`, `deer_metadata`, `deer_additional`, `camera_info`, `annotation_history` tables
- **preview_window.py** - Full-size image viewer with zoom/pan and annotation UI
- **main_window.py** - Secondary/original photo browser GUI (5x5 thumbnail grid)

### AI/ML Components
- **ai_detection.py** - MegaDetector v5 wrapper for animal detection
- **ai_suggester.py** - ONNX species classifier (trained on user's photos)
- **models/** - Pre-trained ONNX models (species.onnx, buckdoe.onnx, labels.txt)

### Utilities
- **image_processor.py** - EXIF extraction, file import, thumbnail creation
- **cuddelink_downloader.py** - CuddeLink camera cloud photo scraper
- **duplicate_dialog.py** - Find duplicates by MD5 hash
- **compare_window.py** - Side-by-side photo comparison (up to 4)

### Training Pipeline (training/)
- **train_detector.py** - YOLO-based species/empty detector
- **train_classifier.py** - Image classifier for species or buck/doe
- **train_reid.py** - Metric-learning for individual deer identification
- **train_antler_heads.py** - Multi-head antler count predictor
- **export_to_app.py** - Copy ONNX exports to models/ folder

## Data Locations

- **Photo Library**: `~/TrailCamLibrary/YYYY/MM/` (macOS/Linux) or `C:\TrailCamLibrary\` (Windows)
- **Database**: `~/.trailcam/trailcam.db`
- **Thumbnails**: `~/TrailCamLibrary/.thumbnails/`

## Key Patterns

- Long operations (imports, hash calculation) run in QThread to avoid UI blocking
- PyQt6 signals/slots for component communication
- ONNX models for species/buck-doe classification (no external API dependencies)
- Database uses SQLite with platform-specific paths

---

## Current State (Jan 1, 2026)

**GitHub:** https://github.com/cabbp3/Trail-Camera-Software

**AI Pipeline:**
1. MegaDetector detects animals → `ai_animal` boxes
2. ONNX species classifier runs on cropped boxes (per-box classification)
3. If Deer → deer head detection → buck/doe classifier

**Key Features Working:**
- Windows & macOS standalone builds (PyInstaller)
- Supabase cloud sync (REST API, no C++ deps)
- CuddeLink photo downloads with retry logic
- Integrated queue mode for reviewing AI suggestions
- Background AI processing with live updates
- Per-box tabbed interface for labeling multiple detections
- Box labels on image showing species/suggestions

**Known Issues:**
- `training/label_tool.py` is 9000+ lines (could be split)

**Supabase:** Working via REST API (`supabase_rest.py`). Run `supabase_setup.sql` once to create tables.

---

## Recent Session (Jan 1, 2026)

### Work Completed This Session

**Per-Box Labeling UI:**
- Restructured photo info pane with QTabWidget - each detection box gets its own tab
- Box labels on image show "Box 1: Species" or "Box 1: Species?" for suggestions
- "Apply to All" applies species/sex/age to all boxes (except deer ID which must be unique)

**Bug Fixes:**
1. **Species typing glitch** - Typing species name was advancing to next photo after each letter
   - Fixed by connecting `editTextChanged` to `schedule_save` (debounced) instead of `_on_species_changed`
   - Also filtered single-letter species from dropdown

2. **Review queue losing suggestions** - Clicking on photo and clicking out was clearing suggestions
   - Added `_original_saved_species` tracking when loading photo
   - Only clear suggestion when a NEW species tag is actually added (not already saved)

3. **Queue not advancing when accepting suggestion** - Selecting species didn't push to next photo
   - Added `activated` signal connection to species combo (fires when explicitly selecting item)
   - Rebuilt photo list after removing resolved photo (refreshes UserRole indices)
   - Added guard against double-advance when multiple signals fire

### Key Code Changes (training/label_tool.py)

**Box Tab Bar** (around line 1059):
```python
self.box_tab_bar = QTabWidget()
self.box_tab_bar.currentChanged.connect(self._on_box_tab_switched)
```

**Species Signal Connections** (around line 1641):
```python
self.species_combo.currentIndexChanged.connect(self._on_species_changed)
self.species_combo.activated.connect(self._on_species_changed)  # For explicit selection
self.species_combo.editTextChanged.connect(self.schedule_save)  # Debounced for typing
```

**Original Species Tracking** (in load_photo, around line 2131):
```python
self._original_saved_species = set(current_species)
```

**New Species Check** (in save_current, around line 2273):
```python
original_species = getattr(self, "_original_saved_species", set())
is_new_species = species and species not in original_species
if is_new_species:
    self.db.set_suggested_tag(pid, None, None)  # Only clear if NEW
```

### Testing Notes
- Review queue advancement: Select species from dropdown, should advance to next unreviewed photo
- Box labels: Should show "Box 1: Deer" (confirmed) or "Box 1: Deer?" (suggestion)
- Suggestions preserved: Click between photos without selecting species, suggestions should remain

---

## Windows Build Status (Dec 25, 2024)

**In Progress:** Building standalone Windows .exe

**Completed:**
- Created BUILD.bat, RUN.bat, Create Desktop Shortcut.vbs
- Created icon.ico from icon.png
- Fixed database migrations (added missing `collection`, `left_ab_points_max`, `right_ab_points_max` columns)
- Confirmed Python 3.11 works (3.14 is too new)
- Dependencies install successfully

**Remaining:**
- Complete Windows build after McAfee exclusion is set up
- Test the standalone .exe on Windows
- Consider code signing for future distribution (avoids antivirus issues)
- Set up GitHub Actions for automated builds (optional)
