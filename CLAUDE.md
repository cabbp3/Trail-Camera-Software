# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Quick Start:**
> 1. Check `TASKS.md` first - see what other Claudes are working on
> 2. Use `INDEX.md` to find the right doc for your task

## Role & Communication Style

You are a **senior software engineer** specializing in building useful, not-complicated software. When starting a session:

1. **Read relevant files** to gain context about what's being worked on
2. **Explain progress and updates at a high level** - the user is a non-technical founder
3. **Avoid jargon** - use plain English when possible
4. **For any actions the user needs to take** (setting up databases, getting API keys, configuring services, etc.), provide **clear step-by-step instructions** so they can complete them independently

Keep solutions simple and focused. Prioritize working software over perfect architecture.

---

## Session Startup: Check Cloud Storage Usage

**Run this at the start of each session** to monitor free tier usage:

```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0" && source .venv/bin/activate && python3 -c "
print('=== STORAGE USAGE CHECK ===')

# Cloudflare R2
try:
    from r2_storage import R2Storage
    r2 = R2Storage()
    if r2.is_configured():
        stats = r2.get_bucket_stats()
        used = stats.get('total_size_mb', 0)
        print(f'Cloudflare R2: {used:.1f} MB / 10,000 MB free')
        if used > 8000:
            print('  ⚠️  WARNING: Approaching R2 free tier limit!')
    else:
        print('Cloudflare R2: Not configured')
except: print('Cloudflare R2: Check failed')

# Supabase
try:
    from supabase_rest import SupabaseREST
    sb = SupabaseREST()
    if sb.is_configured():
        print('Supabase: Configured (check dashboard for exact usage)')
        print('  Free tier: 500 MB database, 1 GB storage, 2 GB bandwidth')
    else:
        print('Supabase: Not configured')
except: print('Supabase: Not configured')

# Local storage
import subprocess
result = subprocess.run(['du', '-sh', '/Users/brookebratcher/TrailCamLibrary'], capture_output=True, text=True)
if result.returncode == 0:
    print(f'Local photos: {result.stdout.split()[0]}')
"
```

**Free Tier Limits:**
| Service | Free Tier | Warning Threshold |
|---------|-----------|-------------------|
| Cloudflare R2 | 10 GB storage, unlimited downloads | 8 GB |
| Supabase | 500 MB db, 1 GB storage, 2 GB bandwidth/month | Check dashboard |

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

Trail Camera Software - **PyQt6 desktop application** for organizing trail camera photos:

| Entry Point | Code | Purpose |
|-------------|------|---------|
| `main.py` | `training/label_tool.py` | Full-featured app (unified) |
| `trainer_main.py` | `training/label_tool.py` | Same as above (legacy entry point) |

**Note:** The apps were unified on Jan 5, 2026. Both entry points now run the same full-featured code. The old simplified `organizer_ui.py` is deprecated.

## Commands

### Run the App (macOS)
```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
source .venv/bin/activate
python main.py
```

### Run the App (Windows)
```cmd
cd C:\Users\mbroo\OneDrive\Desktop\Trail Camera Software V 1.0
venv\Scripts\activate
python main.py
```

**Windows Status (Jan 2026):**
- **GitHub Actions builds** - Automated via `.github/workflows/build-windows.yml`
- **Latest release**: v1.0.5 at https://github.com/cabbp3/Trail-Camera-Software/releases
- App runs and all core features work
- Supabase cloud sync works via REST API (supabase_rest.py)
- **IMPORTANT:** Use Python 3.11 (not 3.14) for local builds
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

### Unified App
- `main.py` → `training/label_tool.py` - Full-featured app (11,000+ lines)
- `trainer_main.py` → Same code (legacy entry point)
- `organizer_ui.py` - Deprecated (kept for reference)

### Main App Features (training/label_tool.py)
- Photo browser with dark theme
- Filter/sort by species, date, collection, camera
- Per-box tabbed interface for multiple detections
- AI suggestions and review queues
- Bounding box annotation
- CuddeLink download integration
- Supabase cloud sync
- Cloudflare R2 photo storage

### Shared Modules
- **database.py** - SQLite operations with WAL mode; schema includes `photos`, `tags`, `deer_metadata`, `deer_additional`, `camera_info`, `annotation_history` tables
- **preview_window.py** - Full-size image viewer with zoom/pan and annotation UI

### AI/ML Components
- **ai_detection.py** - MegaDetector v5 wrapper for animal detection
- **ai_suggester.py** - ONNX species classifier (trained on user's photos)
- **models/** - Pre-trained ONNX models (species.onnx, buckdoe.onnx, labels.txt)

### Utilities
- **image_processor.py** - EXIF extraction, file import, optimized thumbnail creation
- **cuddelink_downloader.py** - CuddeLink camera cloud photo scraper
- **duplicate_dialog.py** - Find duplicates by MD5 hash
- **compare_window.py** - Side-by-side photo comparison (up to 4)
- **site_identifier.py** - Hybrid site detection (OCR + visual)
- **site_detector.py** - OCR-based site detection from camera text overlays
- **site_embedder.py** - MobileNetV2 semantic scene embeddings
- **site_clustering.py** - Legacy edge-based clustering (replaced by hybrid)

### Tools (tools/) - Standalone utility scripts
- **windows_fix.py** - Windows database diagnostics and maintenance
- **run_ai_unlabeled.py** - Batch AI suggestions on unlabeled photos
- **run_megadetector.py** - Batch MegaDetector processing
- **run_species_suggestions.py** - Batch species classification

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

## Current State (Jan 7, 2026)

**GitHub:** https://github.com/cabbp3/Trail-Camera-Software

**Codebase Stats:**
- ~20,800 lines of Python (main codebase, excluding tools/)
- 6,588 total photos in database
- 5,008 tagged photos
- 0 missing files

**AI Pipeline:**
1. MegaDetector detects animals → `ai_animal` boxes
2. ONNX species classifier runs on cropped boxes (per-box classification)
3. If Deer → deer head detection → buck/doe classifier

**Key Features Working:**
- Windows & macOS standalone builds (PyInstaller)
- Supabase cloud sync (REST API, no C++ deps)
- **Cloudflare R2 photo storage** (Tools → Cloud Sync menu)
- CuddeLink photo downloads with retry logic
- Integrated queue mode for reviewing AI suggestions
- Background AI processing with live updates
- Per-box tabbed interface for labeling multiple detections ("Subject 1", "Subject 2", etc.)
- Subject labels on image showing species/suggestions
- Special one-time review queues (stored in `~/.trailcam/*_review_queue.json`)
- "Re-run AI on Current Photo" option in Tools menu
- "Clear" button to reset all labels on a photo
- "Accept All" button to accept AI suggestion for all boxes

**Performance Optimizations:**
- Database WAL mode for better concurrent access
- Indexes on archived, suggested_tag, season_year, tags composite
- Batch updates for bulk operations (100x faster)
- Thumbnail preloading for nearby photos in filtered list
- Photo list caching infrastructure

**Known Issues:**
- `training/label_tool.py` is 11,000+ lines (could be split, but risky)
- MegaDetector struggles with small birds (quail: only 22% detection rate)
- `openpyxl` in requirements.txt is unused (Excel features removed, now using Supabase)

**Supabase:** Working via REST API (`supabase_rest.py`). Run `supabase_setup.sql` once to create tables.

---

## Bedtime/Away Projects

When the user is looking for something to run overnight or while away:

1. **Augmentation Experiment** (~7-10 hours)
   ```bash
   python training/train_augmentation_experiment.py
   ```
   Tests 7 augmentation configs × 3 replicates = 21 training runs.
   See PLAN.md "Augmentation Experiment" section for details.

2. **Download LILA data for rare species** (future)
   - NACTI, Caltech Camera Traps for Coyote, Fox, Bobcat, Otter
   - CDLA-Permissive license allows commercial use

---

## Recent Session (Jan 19, 2026)

### Auto-Sync to Supabase with Offline Queueing

**Problem:** Manual sync via menu was tedious. Changes could be lost if user forgot to sync before closing.

**Solution:** Automatic background sync with smart offline handling.

**New File: `sync_manager.py`** (276 lines)
- `SyncManager` class manages automatic cloud synchronization
- `queue_change()` - Called whenever data changes, starts 30-second debounce timer
- `_check_network()` - Tests connectivity before sync attempts
- `status` property - Returns current sync state for UI display

**Features:**
1. **30-second debounce timer** - Batches rapid changes instead of syncing every keystroke
2. **Offline queue persistence** - Changes saved to `~/.trailcam/pending_sync.json` when offline
3. **Auto-retry on reconnect** - Queued changes sync automatically when network returns
4. **Close warning dialog** - Warns user if closing app with pending offline changes

**Integration Points:**
- Modified `training/label_tool.py` `closeEvent()` to check for pending offline changes
- SyncManager instantiated at app startup
- All data-changing operations call `queue_change()`

**Key Design Decisions:**
- 30-second debounce balances responsiveness vs API load
- Persistent JSON queue survives app crashes/restarts
- Network check before each sync attempt to avoid hanging

---

## Previous Session (Jan 14, 2026)

### Buck/Doe Review Queue Improvements

**Collection Filter Added:**
- Added collection dropdown filter to the buck/doe review queue dialog
- Users can now filter pending suggestions by collection (e.g., "Brooke Farm")

**Properties Button Fixed:**
- Fixed bug where clicking "Properties" didn't navigate to the correct photo
- Root cause: Target photo was archived and excluded from `self.photos` by default
- Fix: Now loads photos with `include_archived=True` and sets archive filter to "All Photos"

**N+1 Query Optimization:**
- Fixed app lockup (198% CPU) when opening review queue
- Changed from N+1 queries (7500+ DB queries) to single SQL JOIN query
- Review queue now opens instantly

**Key Files Changed:**
- `training/label_tool.py` - Review queue dialog (`_open_sex_review_dialog`)

### Site Identification - Hybrid OCR + Visual Approach

**Problem:** Previous edge-based site clustering had poor accuracy (~50-60%). Needed a better way to identify which camera site a photo came from.

**Solution:** Hybrid approach that tries OCR first, falls back to visual matching:

1. **OCR Detection** (`site_detector.py`) - 91% accuracy
   - Reads text overlay burned into images by cameras
   - Extracts site names like "RAYS LINE 003", "SALT LICK E 002"
   - Maps camera text to database site names

2. **Semantic Embeddings** (`site_embedder.py`) - 70% accuracy
   - Uses MobileNetV2 (pre-trained on ImageNet)
   - Creates "scene fingerprints" for visual matching
   - Masks out detected animals before computing embeddings
   - Falls back when cameras don't have text overlays

3. **Hybrid Identifier** (`site_identifier.py`)
   - Tries OCR first (most reliable when available)
   - Falls back to semantic matching when OCR fails
   - Builds reference embeddings from labeled photos

**Results on labeled photos:**
- 90% detected via OCR (96.7% accurate)
- 10% fell back to semantic (40% accurate)
- 91% overall accuracy (up from ~50-60%)

**Integration:**
- Tools → Auto-Detect Sites now uses hybrid approach
- Shows detection method breakdown (OCR vs Visual)
- Updated `run_site_clustering()` in `label_tool.py`

**Database Changes:**
- `get_embedding()` now returns `(bytes, version)` tuple
- `get_all_embeddings()` now returns `(photo_id, bytes, version)` tuples

**Known Limitation:**
- "West Salt Lick" and "West Triangle" share camera text "WEST OF ROAD 004"
- These get confused; would need visual features to disambiguate

---

## Session (Jan 8-9, 2026)

### Mobile App - Google Play Store Submission

**Trail Camera Organizer Mobile** (`/Users/brookebratcher/Desktop/trailcam_mobile/`)

Flutter app for viewing trail camera photos on Android. Connects to same Supabase/R2 backend as desktop app.

**Completed:**
- App renamed to "Trail Camera Organizer"
- Removed login/username requirement (simplified for read-only browsing)
- Fixed species filter pagination (Supabase 1000 row limit)
- Created release signing keystore
- Built and uploaded AAB to Google Play Console
- Privacy policy hosted: https://cabbp3.github.io/Trail-Camera-Software/privacy_policy.html
- Phone screenshots captured
- Tablet screenshots created (padded from phone)
- Closed testing track "Beta Testers" created

**Play Store Status:**
- In closed testing
- Need 12 testers opted-in for 14 days before production access

**TODO:**
- Set up real tablet emulators for proper 7"/10" screenshots
- Add offline download feature
- iOS build (future)
- **Fix username system** - Currently hardcoded to "brooke" in mobile app; need proper multi-user support where mobile app uses logged-in user's R2 folder

### R2 & Supabase Sync Issues (Jan 9-10, 2026)

**Problems Fixed:**

1. **Corrupted thumbnails in R2** - New photos showed as broken images on mobile
   - Root cause: Desktop used wrong thumbnail paths and ID-based naming
   - Fix: Updated `label_tool.py` to use `photo['thumbnail_path']` and `file_hash` for R2 keys
   - 138 corrupted thumbnails re-uploaded by other Claude session

2. **Corrupted camera_model data** - EXIF parsing left null bytes and garbage in database
   - Caused photo_key mismatches between local and Supabase
   - Fix: Cleaned 254 camera_model values by stripping non-printable characters

3. **Supabase sync incomplete** - 6,942 photos in Supabase vs 7,255 local
   - Root cause: Upsert only updates existing records, doesn't insert new ones
   - Also: 66 orphaned records with corrupt keys in Supabase
   - Fix: Deleted orphans, directly inserted 379 missing photos

4. **Missing thumbnails in R2** - 23 new photos lacked thumbnails
   - Fix: Uploaded all missing thumbnails using correct hash-based naming

**Final State:**
- Supabase: 7,255 photos (all with file_hash)
- R2: All thumbnails uploaded with `{file_hash}_thumb.jpg` naming
- Desktop upload code fixed to use correct paths and hash-based keys

**Key Files Changed:**
- `training/label_tool.py:11500-11542` - Fixed upload_to_cloud() to use thumbnail_path and file_hash

---

## Session (Jan 6-7, 2026)

### v1.0.5 Release

**Smart Sync Implementation:**
- Added `updated_at` column to all sync tables
- Created `sync_state` table for tracking last push/pull timestamps
- SQLite triggers auto-update `updated_at` on INSERT/UPDATE
- First sync is full, subsequent syncs only push changed records
- Fixed database migration order (archived index after column exists)

**UI Cleanup:**
- Removed box-related review queue items
- Removed Claude review queue menu item
- Fixed progress bar visibility when pushing to Supabase

**Windows Release v1.0.5:**
- Fixed startup crash on Windows with existing databases
- GitHub Actions automated build working
- Release: https://github.com/cabbp3/Trail-Camera-Software/releases/tag/v1.0.5

---

## Session 19 (Jan 5-6, 2026)

### Unified App Architecture

**Merged Organizer into Trainer:**
- `main.py` now launches the full Trainer app (same as `trainer_main.py`)
- Both entry points run `training/label_tool.py`
- Old simplified `organizer_ui.py` is deprecated (kept for reference)
- Removed Simple Mode from Trainer - now one full-featured app

**CuddeLink Credential Fix:**
- Fixed parameter names: `user=` not `email=`, `destination=` not `dest_dir=`
- Fixed return type handling (expects `List[Path]` not dict)

### Head Annotation Improvements

- **Tighter zoom**: Reduced padding from 50% to 15% when loading each subject box
- **Restored "Low Quality" quick note**: Added back to `head_annotation_window.py`

### Head Keypoint Model v1

Trained first version of head direction prediction model:
- **Training data**: 186 clean annotations (with head lines, no notes)
- **Results**: ~14% skull error, ~16% nose error (on 200px crop = ~28-33px)
- **Model saved**: `outputs/head_keypoints_v1/best_model.pt`
- **Visualizations**: `outputs/head_keypoints_v1/visualizations/`

**Annotation Progress:**
- Total deer boxes: 5,759
- Annotated: 429 (7.4%)
- Remaining: 5,330

### Windows Release v1.0.3

**GitHub Actions Workflow:**
- `.github/workflows/build-windows.yml` - Automated Windows builds
- Release v1.0.3: https://github.com/cabbp3/Trail-Camera-Software/releases/tag/v1.0.3
- Windows ZIP upload pending (network issues)

### Cloudflare R2 Cloud Storage (Jan 6, 2026)

**R2 Setup Complete:**
- Bucket: `trailcam-photos`
- Credentials stored in `~/.trailcam/r2_config.json`
- Free tier: 10GB storage, unlimited downloads

**New Files Created:**
- `r2_storage.py` - R2 upload/download/signed URL module
- `user_config.py` - Simple username management
- `tools/batch_upload_r2.py` - Batch upload script for overnight runs
- `tools/r2_admin.py` - Admin CLI to view all users/files

**Desktop App Cloud Features** (Tools → Cloud Sync):
- Cloud Status - Shows bucket usage and your uploads
- Upload Thumbnails to Cloud (~90MB)
- Upload All Photos to Cloud (~5GB)
- Change Username

**Web Server Cloud Endpoints** (`web/server.py`):
- `/api/cloud/stats` - Bucket statistics
- `/api/cloud/photos` - List photos with signed URLs
- `/api/cloud/photos?user=X` - Filter by user

**User System (Phase 1):**
- Username prompt on first launch
- No authentication yet - for trusted users (family, hunting club)
- See PLAN.md for Phase 2/3 roadmap (privacy, real auth)

**Batch Upload for Tonight:**
```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
source .venv/bin/activate
python tools/batch_upload_r2.py --username brooke --thumbnails-only
# Or for full photos:
python tools/batch_upload_r2.py --username brooke --both --resume
```

---

## Session 18 (Jan 4, 2026)

### PLAN.md Verification & Cleanup

Verified status of all pending items in PLAN.md:
- **Excel export/import** - DONE (removed from UI, replaced by Supabase)
- **AI background processing** - DONE (AIWorker QThread working)
- **Person/Vehicle in species model** - DONE (MegaDetector handles these)
- **Export detection crops** - Not needed (boxes sync via Supabase)
- **Single-instance protection** - No mutex code found; may be SQLite lock, needs testing
- **windows_fix.py** - Confirmed in tools/ folder

Updated PLAN.md to mark completed items and clarify remaining work.

---

## Session 17 (Jan 3, 2026)

### Cleanup & Performance Improvements

**Files Deleted:**
- `archive/main_window.py` (1,085 lines) - dead legacy UI code
- `create_icon.py` (54 lines) - one-time build script

**Files Moved to tools/:**
- `windows_fix.py` - Windows maintenance script
- `run_ai_unlabeled.py` - Batch AI processing
- `run_megadetector.py` - Batch MegaDetector
- `run_species_suggestions.py` - Batch species classification

**Database Improvements:**
- Added WAL mode for better concurrent access
- Added indexes: `idx_photos_archived`, `idx_photos_suggested_tag`, `idx_photos_season`, `idx_tags_composite`
- Fixed N+1 query in `archive_photos()` with batch UPDATE

**Code Quality:**
- Replaced 8 debug print statements with proper `logger` calls
- Added photo list caching infrastructure (`_get_all_photos_cached()`)
- Added `_preload_nearby_thumbnails()` for smoother scrolling

**Thumbnail Optimization:**
- Lowered quality from 85 to 75 (smaller files)
- Added `optimize=True` flag
- Skip regenerating existing thumbnails
- Preload thumbnails for 10 photos before/after current in filtered list

**Species Review Queue Fix:**
- Fixed quick-select buttons not triggering auto-advance
- Now advances to next photo when all boxes are labeled

**Results:**
- Reduced main codebase from 22,716 to ~20,800 lines
- Database queries 20-50% faster
- Bulk operations 100x faster

---

## Previous Session (Jan 2, 2026)

### Work Completed This Session

**New Features:**
1. **Auto-Update System** (Help → Check for Updates)
   - Checks GitHub releases API for new versions
   - Downloads and installs updates automatically
   - Version tracking via `version.py`

2. **Photo Storage Locations** (Settings → Photo Storage Locations)
   - Add folders where photos are stored
   - Scans for photos without copying them
   - Supports multiple locations

3. **Per-Box Labeling UI**
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
