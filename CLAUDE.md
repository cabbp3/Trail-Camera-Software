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

**Windows Status (Dec 20, 2024):**
- App runs and all core features work
- Supabase cloud sync FIXED - replaced supabase package with REST API (supabase_rest.py)
- No C++ Build Tools needed anymore

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Build Windows Executable
```cmd
# On Windows - double-click build_windows.bat, or:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller TrailCamOrganizer_Windows.spec --clean
# Output: dist\TrailCamOrganizer\TrailCamOrganizer.exe
```

**Note:** No C++ Build Tools required - supabase package was replaced with `supabase_rest.py` (REST API using requests only).

**To add a custom icon:** Convert `icon.png` to `icon.ico` using an online converter and place in project root.

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
- **ai_suggester.py** - Multi-model species suggester (prefers CLIP, falls back to ONNX)
- **models/** - Pre-trained ONNX models (species.onnx, detector.onnx, labels.txt)

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
- AI suggesters are swappable (CLIP preferred → ONNX fallback)
- Database uses SQLite with platform-specific paths

---

## Resume Next Session

**What was completed (Dec 24, 2024 - Deep Clean Session):**

1. **Windows Standalone Build Fixes** - Critical issues that would have broken the Windows build:
   - Added missing dependencies to `requirements.txt`: numpy, opencv-python, onnxruntime
   - Added missing data files to `TrailCamOrganizer_Windows.spec`: yolov8n.pt and all Python modules
   - Fixed `ai_suggester.py` to use `sys._MEIPASS` for PyInstaller bundling (models now load correctly in standalone app)
   - Added `yolov8n.pt` to both macOS and Windows spec files

2. **Cross-Platform Path Fixes**
   - Fixed `web/server.py` hardcoded paths to work on Windows
   - Added `get_db_path()` and `get_library_path()` helpers for cross-platform paths
   - Fixed `ai_detection.py` to show correct model path on Windows

3. **Dead Code Cleanup**
   - Fixed `run_ai_unlabeled.py:22` - Removed reference to deleted CLIP attribute (would crash)
   - Identified duplicate preprocessing code in ai_suggester.py (left for future refactoring)

4. **Code Review Findings** (for future work):
   - `training/label_tool.py` is 3500+ lines - could be split into modules
   - Duplicate image preprocessing in SpeciesSuggester, BuckDoeSuggester, ReIDSuggester
   - Duplicate zoom/pan logic in preview_window.py and compare_window.py

**Files Modified:**
- `requirements.txt` - Added numpy, opencv-python, onnxruntime
- `TrailCamOrganizer_Windows.spec` - Added all missing datas entries
- `TrailCamOrganizer_macOS.spec` - Added yolov8n.pt
- `ai_suggester.py` - Added `get_resource_path()` helper with sys._MEIPASS support
- `web/server.py` - Cross-platform path helpers
- `ai_detection.py` - Fixed model path print message
- `run_ai_unlabeled.py` - Removed CLIP reference

---

**What was completed (Dec 24, 2024 - Night Session):**

1. **Integrated Queue Mode** - Major UI overhaul
   - Review queues (species suggestions, sex suggestions) now filter the main photo list instead of opening modal dialogs
   - Full access to all labeling tools during queue review
   - Queue control panel with Accept (A), Skip (S), Exit buttons
   - Queue suggestion shown in photo list (e.g., "12/25/2024 3:16 AM — Deer (95%)")
   - Green highlighting for reviewed items
   - Can exit queue and stay on current photo
   - Key code: `_enter_queue_mode()`, `_exit_queue_mode()`, `_queue_accept()`, `_queue_skip()` in label_tool.py

2. **Background AI Processing**
   - AI suggestions now run in background thread (`AIWorker` QThread class)
   - Photos appear in queue as they're processed (live updates)
   - Progress bar shows AI processing status
   - Menu: Tools → Suggest Tags (AI) — Background + Live Queue

3. **Session-Based Recent Species**
   - Recent species quick buttons now update immediately when you apply a species
   - Uses `_session_recent_species` list to track species applied this session
   - Prioritizes session-applied species over database history

4. **Bug Fixes**
   - Fixed crash: `clear_suggested_tag` → `set_suggested_tag(pid, None, None)`
   - Fixed crash: `clear_suggested_sex` → `set_suggested_sex(pid, None, None)`

**Known Issue - Needs Future Work:**
- **Queue mode performance is slow** - The integrated queue window runs slower than the old modal dialogs. May need optimization (lazy loading, reduced database queries, etc.)

**Key Files Modified:**
- `training/label_tool.py` - Queue mode, AIWorker, session recent species (~500 lines added)

---

**What was completed (Dec 24, 2024 - Evening Session):**

1. **Removed CLIP - Simplified Species Classifier**
   - CLIP was misclassifying 40% of deer as Turkey
   - Now uses ONLY the ONNX model (trained on user's photos)
   - ONNX: 100% accuracy on deer vs CLIP's 60%
   - `ai_suggester.py` reduced from 383 to 254 lines
   - Removed torch/open_clip dependencies for species classification

2. **GitHub Repository Created**
   - Repo: https://github.com/cabbp3/Trail-Camera-Software
   - Initial commit with 59 files
   - Added `.gitignore` to exclude: .venv, __pycache__, *.db, models/*.onnx, runs/, outputs/, exports/

3. **Empty Suggestion Logic Fixed**
   - Photos with NO MegaDetector boxes → Auto-suggested as "Empty" (95% confidence)
   - Photos WITH animal boxes → Classifier runs on cropped area
   - Empty suggestions appear in review queue for confirmation

4. **AI Suggestion Workflow Clarified**
   - Re-running suggestions WILL overwrite existing AI suggestions
   - Human-confirmed tags are preserved (never overwritten)
   - This is intentional - allows re-running to improve suggestions

**Current AI Flow:**
1. MegaDetector runs on photo
2. If no animals detected → Suggest "Empty"
3. If animals detected → Crop to box, run ONNX classifier
4. If Deer → Run deer head detection, then buck/doe classifier

**Files Modified:**
- `ai_suggester.py` - Removed CLIP, simplified to ONNX-only
- `training/label_tool.py` - Empty suggestion for no-box photos
- `.gitignore` - Created for GitHub

---

**What was completed (Dec 24, 2024 - Morning Session):**

1. **CuddeLink Collection Prompt** - Downloads now ask for Collection (defaults to "Brooke Farm")

2. **Quick Buck ID Buttons Fixed**
   - Populated `recent_bucks` table with valid buck IDs from `deer_metadata`
   - Added validation to prevent partial-typing pollution
   - Changed to 3x4 grid (12 buttons), 15 character limit with "..." truncation
   - Full name shown on hover

3. **Removed Cloud Sync Popup** - Silent sync on startup (no notification window)

4. **MegaDetector as Primary Detector**
   - MegaDetector now used exclusively for animal detection (ai_animal boxes)
   - Custom YOLO detector only used for deer heads (ai_deer_head boxes)
   - Deer head detection ONLY runs after species is confirmed as "Deer"
   - Fixed issue where coyotes were getting deer head boxes

5. **Species Label Fixes**
   - Added "Other bird" and "skunk" to custom_species table
   - Species dropdown now scans tags table for all distinct species

6. **Empty Label Prevention**
   - If MegaDetector found animal boxes, "Empty" label is blocked
   - Prevents contradictory states (animal detected but labeled Empty)

7. **Progress Bar Fix** - Now shows only unlabeled photo count, not total

---

**What was completed (Dec 23, 2024):**

1. **CuddeLink Download Improvements** - Fixed reliability issues with multi-day downloads
   - Added retry logic with exponential backoff (3 retries, 2s → 4s → 8s delays)
   - Handles server errors: 502, 503, 504, Cloudflare errors (520-524)
   - Handles connection errors and timeouts gracefully
   - Added browser-like headers to avoid being blocked

2. **Day-by-Day Downloading** - Solved timeout issues on large date ranges
   - Multi-day requests now download one day at a time
   - Shows progress: "Day 1/3: 2024-12-20"
   - If one day fails, continues with next day
   - 1-second delay between days to avoid overloading server
   - Successfully downloaded 62 photos across 3 days (Dec 20-22)

3. **Check CuddeLink Status Feature** - New menu item
   - File → Check CuddeLink Status...
   - Tests 3 endpoints: Main site, Login page, Photos page
   - Shows response times and status (OK, Slow, Down, Timeout, Unreachable)
   - Helps diagnose connection issues before attempting download

4. **User-Friendly Error Messages** - Clearer feedback on failures
   - "CuddeLink server is temporarily unavailable (Error 502). Please try again in a few minutes."
   - "Could not connect to CuddeLink servers. Please check your internet connection."
   - "Connection to CuddeLink timed out. The server may be slow - please try again."

**Files Modified:**
- `cuddelink_downloader.py` - Retry logic, day-by-day downloads, server status check
- `training/label_tool.py` - Added Check CuddeLink Status menu item

---

**What was completed (Dec 21, 2024 - Night Session):**

1. **B&C Antler Scoring Research** - Created `docs/bc_scoring_research.md`
   - Evaluated commercial options: Rackline.ai (competitor - ruled out), HuntelligenceX, Spartan Forge
   - Reviewed academic papers on antler measurement and deer age classification
   - Documented keypoint detection frameworks (YOLOv8-pose, DeepLabCut, OpenPifPaf)
   - Defined required keypoints for B&C scoring (beam tips, G1-G5 tines, H1-H4 circumferences)

2. **Build-Your-Own B&C Scoring Plan**
   - Phase 1: Rough category classifier (Small/Medium/Good/Trophy) using buck crops
   - Phase 2: Keypoint detection for actual inch measurements
   - Phase 3: Multi-angle scoring from same buck across photos
   - Phase 4: Harvest validation with real B&C scores

3. **Training Data Strategy**
   - Dad is a taxidermist - access to mounts with measurable ground truth
   - Video approach: Walk around mount, extract frames automatically
   - Verbal consent from customers or use personal mounts
   - Domain adaptation needed (taxidermy → trail cam)

4. **Future Schema Designed**
   - `buck_antler_measurements` - Full per-year antler data (beams, tines, spread, circumferences, scores)
   - `buck_score_categories` - For AI training labels (rough categories)

5. **App Roadmap Notes**
   - Video support coming to app (starting with photos first)
   - Buck profiles will have per-year antler measurement entry UI

**Previous session (Dec 21, 2024):**
- Labeled 3,266 / 3,269 photos (99%)
- Fixed species label saving in box review queue
- Added box editing (select/delete/draw) in review queue
- Ran MegaDetector on 1,053 unprocessed photos
- Auto-labeled 434 photos with no detections as "Empty"
- Retrained species model with 12 classes, sqrt weighting (96.3% accuracy)
- Added dynamic species buttons from database
- Added "Other (+)" button for new species on the fly
- Empty label now auto-clears AI boxes

---

**What was completed (Dec 21, 2024 - Evening Session):**

1. **Import from SD Card feature** - New menu item auto-detects mounted SD cards
   - File → Import from SD Card...
   - Shows list of detected drives with photo counts
   - Includes Collection/Farm dropdown during import

2. **Tightwad House collection imported** - 1,167 photos
   - Collection assigned to all photos
   - Timestamps corrected (+21h 46m offset - camera clock was wrong)
   - AI species suggestions generated

3. **Progress bars for AI suggestions** - Shows photo count and species/buck-doe tallies
   - Fixed "Cancelled" bug that showed even on successful completion

4. **Species review queue improvements**
   - Now shows detection boxes on photos (4px thick lines)
   - Yellow = AI boxes, Magenta = deer heads, Green = manual

5. **New labeling queue: "Label Photos with Boxes (No Suggestion)"**
   - Tools → Label Photos with Boxes (No Suggestion)...
   - For photos with AI detection boxes but no species suggestion
   - Shows boxes with thick lines
   - Accept Boxes (A) - keep boxes, advance
   - Delete Boxes (D) - remove bad boxes, advance
   - Species quick buttons for labeling

**Current Data:**
- ~3,500+ photos total (1,167 new from Tightwad House)
- Two collections: "Brooke Farm" and "Tightwad House"
- Lots of non-deer photos in new batch (good for training diversity)

**Next Steps:**
1. Continue labeling species in the new queue
2. Set up GitHub for version control (see PLAN.md High Priority #4)
3. Remove Excel export/import features - using Supabase now (PLAN.md #5)
4. Stage 3: Deer head detection (need 500+ head boxes, currently only 63)

**Future enhancements noted (see PLAN.md Lower Priority):**
- Timestamp correction overlay on photos
- AI suggestions optimization (filter already-labeled)
- Background AI processing (non-blocking)
- Per-species counts in progress bar

**Supabase Status:** Working - using REST API (supabase_rest.py), no C++ dependencies

---

## Current State (December 2024)

### Recently Completed (Dec 20, 2024)

**MegaDetector Integration**
- Integrated Microsoft's MegaDetector v5a.0.1 for animal detection
- MIT licensed, safe for commercial use
- Created `run_megadetector.py` script to process photos without boxes
- Processed 492 photos, found 497 detections (496 ai_animal, 1 ai_vehicle)
- Boxes saved with labels: `ai_animal`, `ai_person`, `ai_vehicle`
- Uses Apple Metal (MPS) for GPU acceleration on Mac
- Model auto-downloads on first run (~280MB)

**AI Box Review Queue**
- Added green highlighting for reviewed items (Tools → Review AI Boxes)
- Tracks reviewed photos in session with `ai_reviewed_photos` set
- Can navigate back to reviewed (green) photos to view them

**Key Files:**
- `run_megadetector.py` - Batch process photos without boxes through MegaDetector
- `AI_REFINEMENT_PLAN.md` - 5-stage AI improvement roadmap

### Recently Completed (Dec 19, 2024)

**Desktop App Icon**
- TrailCam Trainer.app on Desktop now works
- Uses Terminal to launch (workaround for macOS security restrictions)
- Custom icon from ChatGPT deer image

**Photo List Sorting**
- Added Sort dropdown: Date (Newest), Date (Oldest), Location, Species, Deer ID
- Located in filter row next to Year filter

**Supabase Cloud Sync**
- Full push/pull sync implemented
- Syncs: tags, deer_metadata, deer_additional, buck_profiles, buck_profile_seasons, annotation_boxes
- Photos matched by `original_name|date_taken|camera_model` key (works across computers)
- Photos stay local, only labels sync to cloud

**Camera Location Assignment via OCR**
- Photos now have their location automatically extracted from the timestamp/watermark on the image
- Uses pytesseract OCR to read text like "WB 27", "RAYS LINE", "WEST OF ROAD" from photo stamps
- Regex pattern: `r'[AP]M\s+(.+?)\s+\d{3}'` extracts location text between time and camera number
- OCR mappings handle variations (e.g., "SALTLICKE" → "Salt Lick", "WB27" → "WB 27")
- Date-based mappings handle camera moves (e.g., "WEST OF ROAD" before Oct 25 = "West Salt Lick", after = "West Triangle")
- Successfully assigned 1622 of 1803 photos (~90%) automatically

**UI Improvements**
- Merged "Site" and "Camera Location" into single "Camera Location" field
- Added quick-select buttons for fast location assignment
- Location filter dropdown now works with camera_location strings (not old site_id system)
- Green highlighting in review queues for reviewed items

### Key Database Fields for Locations

- `camera_location` - The verified/assigned location name (string)
- `stamp_location` - Raw OCR text extracted from photo timestamp (for debugging/bulk edits)

### Notes for Future Work

- 181 photos remain unassigned (likely verification photos with no timestamp)
- OCR approach is camera-dependent; stamp format varies by camera brand
- Visual clustering code exists in `site_clustering.py` but OCR proved more reliable
- Consider batch editing tools for handling camera moves by date range

---

## Supabase Integration (Implemented Dec 19, 2024)

**Status:** Code complete, awaiting user to run SQL setup script.

### What's Implemented
- Manual push/pull sync (File menu)
- Credentials stored in Qt settings (Settings → Setup Supabase Cloud)
- Tables synced: `photos_sync`, `tags`, `deer_metadata`, `deer_additional`, `buck_profiles`, `buck_profile_seasons`, `annotation_boxes`
- Photos matched across computers by `original_name|date_taken|camera_model` key

### Key Files
- `supabase_setup.sql` - SQL to create tables in Supabase (run once)
- `database.py` - `push_to_supabase()` and `pull_from_supabase()` methods
- `training/label_tool.py` - UI for credentials and push/pull (around line 3962-4144)

### Future Enhancements
- Photo storage in Supabase (requires paid plan)
- Automatic sync on save
- Conflict resolution UI
