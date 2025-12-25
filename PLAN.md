# Trail Camera Software - Project Plan

**Last Updated:** December 20, 2024 (Session 8)

This document tracks all tasks for the Trail Camera Photo Organizer. Tasks marked with `[X]` are complete.

---

## Product Vision

This software is intended to become a **marketable product** for hunters and wildlife enthusiasts. Not all features may ship (e.g., training pipelines stay internal), but user-facing features should be polished, intuitive, and reliable.

---

## Current Status Summary

- **~2,100+ photos** in the database (313 new from CuddeLink)
- **1,622 photos** with camera location assigned via OCR (~90%)
- **~1,275 buck/doe labeled** photos (997 buck, 278 doe)
- **Buck/doe model** retrained with 2.3x more data (93.8% balanced accuracy)
- **Species model** using simplified 5 categories (Deer, Empty, Turkey, Other Mammal, Other)
- **3 AI models** deployed (species classifier, object detector, buck/doe classifier)
- **Windows version** running (partially - see below)

## Resume Next Session

**Where we left off (Dec 20, 2024):**
- Windows version working! App runs with `python main.py`
- Supabase cloud sync NOT working on Windows - needs `supabase` package
- Issue: `pip install supabase` requires Visual C++ Build Tools
- Visual C++ Build Tools installed but supabase still failing

**To resume Windows setup:**
1. Open Command Prompt
2. `cd C:\Users\mbroo\OneDrive\Desktop\Trail Camera Software V 1.0`
3. `venv\Scripts\activate`
4. Try: `pip install supabase` (may need troubleshooting)
5. `python main.py`

**What works on Windows:**
- App launches and runs
- Photo import
- Photo tagging
- CuddeLink download
- All core features

**What doesn't work on Windows (yet):**
- Supabase cloud sync (missing package)

**Feature request noted:**
- CuddeLink date range option (currently only downloads last 7 days)
- Ask for camera location on photo import (complex when importing from multiple locations at once - may need to skip location assignment during bulk import)
- Ask for Collection/Farm on photo import (which network of cameras the photos came from)

**Pending features (todo list):**
1. ~~**Export/Import Labels to Excel**~~ **DONE**
2. ~~**Simple/Advanced Mode toggle**~~ **DONE**
3. ~~**Fix CuddeLink auto-download**~~ **DONE**
4. ~~**Photo list sorting**~~ **DONE**
5. ~~**Supabase cloud sync**~~ **DONE on Mac** (Windows needs supabase package fix)
6. ~~**Desktop app icon**~~ **DONE**
7. ~~**Windows compatibility**~~ **PARTIAL** (app works, cloud sync pending)
8. Batch editing by date range (for camera moves)
9. ~~CuddeLink duplicate prevention~~ **DONE** (skips already-imported by filename)
10. ~~CuddeLink date range selector~~ **DONE** (download more than 7 days)

---

## Session Log

### December 21, 2024 (Session 9 - Evening)
**What we did:**

**Part 1: SD Card Import Improvements**
- Added "Import from SD Card..." menu option (File menu)
- Auto-detects mounted drives and shows photo counts
- Added Collection/Farm dropdown to import dialog
- Added `collection` parameter to database and import pipeline

**Part 2: Tightwad House Import**
- Imported 1,167 photos from SD card
- Assigned all to "Tightwad House" collection
- Fixed timestamps (+21h 46m offset - camera clock was wrong)
- Ran AI species suggestions on new photos

**Part 3: AI Progress Bars**
- Added progress bars to "Suggest Tags (AI)" functions
- Shows "Processing photo X of Y" with species/buck-doe counts
- Fixed bug where "Cancelled" showed even on successful completion

**Part 4: Species Review Queue Improvements**
- Added box display to species review queue
- Boxes drawn with 4px thick lines (yellow=AI, magenta=heads, green=manual)

**Part 5: New Labeling Queue**
- Added "Label Photos with Boxes (No Suggestion)..." to Tools menu
- Shows photos with AI boxes but no species suggestion
- Accept Boxes (A) - keep boxes and advance
- Delete Boxes (D) - remove bad boxes and advance
- Species quick buttons for fast labeling

**Future items added to PLAN.md:**
- Timestamp correction overlay (#12)
- AI suggestions optimization (#13)
- Background AI processing (#14)
- GitHub setup (High Priority #4)
- Remove Excel features (High Priority #5)

---

### December 19, 2024 (Session 7)
**What we did:**

**Part 1: Desktop App Icon Fix**
- Fixed TrailCam Trainer.app launcher - was using wrong Python path
- App now launches via Terminal (workaround for macOS security/permissions)
- Added custom icon from ChatGPT deer image (`ChatGPT Image Dec 5, 2025, 07_07_24 PM.png`)
- Icon converted to .icns and added to app bundle

**Part 2: Photo List Sorting**
- Added Sort dropdown to filter row (next to Year filter)
- Options: Date (Newest), Date (Oldest), Location, Species, Deer ID
- Sorting applied in `_filtered_photos()` method

**Part 3: Supabase Cloud Sync**
- Created Supabase account and project
- Project URL: `https://iwvehmthbjcvdqjqxtty.supabase.co`
- Created `supabase_setup.sql` with all table definitions
- Added `supabase` to requirements.txt
- Added sync methods to `database.py`:
  - `push_to_supabase()` - uploads all labels to cloud
  - `pull_from_supabase()` - downloads labels from cloud
- Added UI to `training/label_tool.py`:
  - Settings → Setup Supabase Cloud... (credentials dialog)
  - File → Push to Cloud... (upload labels)
  - File → Pull from Cloud... (download labels)
- Photos matched across computers by `original_name|date_taken|camera_model` key

**Tables synced to Supabase:**
- `photos_sync` - photo identifiers (not file paths)
- `tags` - species labels
- `deer_metadata` - primary deer per photo
- `deer_additional` - secondary deer
- `buck_profiles` - deer profiles
- `buck_profile_seasons` - per-season stats
- `annotation_boxes` - bounding boxes

**Awaiting user action:**
- Run `supabase_setup.sql` in Supabase SQL Editor to create tables
- Enter credentials in app and test connection

---

### December 18, 2024 (Session 6)
**What we did:**

**Part 1: Simple/Advanced Mode Toggle**
- Added Settings → Simple Mode toggle
- Simple Mode hides:
  - AI suggestion buttons and labels
  - Box annotation tools
  - Training/export buttons (Export CSVs, Retrain Model)
  - Antler details section
  - Key characteristics fields
  - Bulk operations (Merge Buck IDs, Set Species on Selected)
  - Review queue buttons
  - AI-related Tools menu items
- Simple Mode keeps visible:
  - Photo list with location filter
  - Image viewer with zoom
  - Species dropdown and quick buttons
  - Buck/Doe/Unknown buttons
  - Deer ID and Age class
  - Camera location
  - Notes
  - Navigation (Prev/Next, Save)

**Part 2: Excel Export/Import (Multi-Computer Workflow)**
- Added File → Export Labels to Excel
  - Exports: filename, date_taken, camera_model, species, sex, deer_id, age_class, camera_location, notes, full_path
  - Creates .xlsx file for sharing
- Added File → Import Labels from Excel
  - Matches photos by filename
  - Imports species, sex (buck/doe), deer_id, age_class, camera_location, notes
  - Shows progress and import summary
- Added openpyxl to requirements.txt

**How the multi-computer workflow works:**
1. Main computer runs full app and labels photos
2. Export labels to Excel (File → Export Labels to Excel)
3. Share Excel file with other team members
4. Other computers import their photos, then import the Excel file
5. Labels are matched by filename and applied automatically

**Part 3: Year Filter (Antler Year)**
- Added Year filter dropdown to filter photos by antler year
- Antler year runs May-April (e.g., photos from May 2024 - April 2025 = "2024-2025")
- Displays as "YYYY-YYYY" format with photo counts
- Newest years shown first in dropdown

**Part 4: CuddeLink Auto-Download**
- Added File → Setup CuddeLink Credentials dialog
  - Enter email and password for CuddeLink account
  - Test Connection button to verify credentials work
  - Credentials saved locally using Qt settings
- Updated File → Download from CuddeLink
  - Prompts to set up credentials if not saved
  - Shows progress while downloading
  - Automatically imports downloaded photos
- Removed dependency on environment variables (CUDDE_USER/CUDDE_PASS)
- Fixed token extraction for ASP.NET login (regex patterns updated)
- Fixed photo ID extraction (hidden input parsing)
- Added date range filter (defaults to Dec 11, 2025 to today)
  - Solves issue where only last 3 days were being downloaded
  - Successfully downloads all 271 photos in date range

**Part 5: UI Polish**
- Removed "(for training)" from Species label
- Fixed Hide Details button to actually hide the details panel
- Show Details button now larger and more visible when panel is hidden

---

### December 17, 2024 (Session 5)
**What we did:**

**Part 1: Camera Location via OCR (Major Feature)**
- Pivoted from visual clustering to OCR-based location extraction
- Trail camera photos have timestamp watermarks with location names
- Used pytesseract to read text from photo stamps
- Regex pattern extracts location: `r'[AP]M\s+(.+?)\s+\d{3}'`
- Successfully assigned 1,622 of 1,803 photos (~90%)

**OCR mappings implemented:**
- Raw stamp variations mapped to canonical names:
  - "WB27", "WB 27", "WB 27." → "WB 27"
  - "RAYSLINE", "RAYS LINE" → "Ray's Line"
  - "SALTLICKE", "SALT LICK E" → "Salt Lick"
- Date-based mappings for camera moves:
  - "WEST OF ROAD" before Oct 25, 2024 → "West Salt Lick"
  - "WEST OF ROAD" on/after Oct 25, 2024 → "West Triangle"

**Part 2: UI Improvements**
- Merged "Site" and "Camera Location" into single "Camera Location" field
- Added quick-select buttons for fast location assignment
- Fixed location filter dropdown (was using old site_id system, now uses camera_location strings)
- Green highlighting for reviewed items in queues

**Database changes:**
- Added `stamp_location` column - stores raw OCR text
- `camera_location` field now stores verified location name

**Key files:**
- `training/label_tool.py` - UI changes, OCR logic, filter fixes
- `site_clustering.py` - Visual clustering code (kept but OCR proved more reliable)

**Why OCR approach won:**
- Camera stamps are consistent (same camera = same stamp)
- Works across lighting conditions (day/night)
- Handles deer/objects blocking background
- Date-based mappings handle camera moves elegantly

---

### December 17, 2024 (Session 4)
**What we did:**

**Part 1: Integrated detection into classification pipeline**
- AI suggestions now auto-run detector if no boxes exist
- Buck/doe classifier uses head crops automatically
- Detection boxes saved with confidence scores

**Part 2: Auto Site Detection (NEW FEATURE)**
- Built automatic camera site detection using image similarity clustering
- Photos from the same camera location are grouped together automatically
- Uses ResNet18 embeddings + DBSCAN clustering

**How it works:**
1. Go to **Tools → Auto-Detect Sites**
2. App extracts visual features from each photo (trees, horizon, structures)
3. Photos with similar backgrounds are clustered together
4. Sites are created automatically (Site 1, Site 2, etc.)
5. Rename sites to meaningful names (e.g., "North Feeder", "Creek Crossing")

**New files:**
- `site_clustering.py` - Embedding extraction and clustering algorithm

**Database changes:**
- Added `sites` table (id, name, description, representative_photo_id)
- Added `photo_embeddings` table (stores visual features for clustering)
- Added `site_id` column to photos table
- Added site management methods

**UI additions:**
- Site filter dropdown in main window (filter photos by site)
- Tools → Auto-Detect Sites (runs clustering)
- Tools → Manage Sites (rename, delete, view photos by site)

**Why this matters:**
- No more manually tagging camera locations
- Automatically organize 1,800+ photos by site
- Foundation for site-specific background models (future)
- Works for any customer's photos - no pre-training needed

---

### December 16, 2024 (Session 3)
**What we did:**
- Retrained buck/doe model with 1,275 photos (was 551) - 2.3x more data
  - Buck: 997 photos, Doe: 277 photos
  - Final accuracy: Buck 90.6%, Doe 97.1%, Balanced 93.8%
- Fixed zoom in review windows to match main window behavior:
  - Ctrl+scroll to zoom (regular scroll pans)
  - Double-click toggles fit/100%
  - Pinch gesture for trackpad
  - Drag mode auto-enabled when zoomed in
- Added filters to buck/doe review dialog:
  - Type filter: All / Buck / Doe
  - Confidence filter: All / ≥90% / ≥80% / ≥70% / <70%
  - Date filter: All / specific dates
- Fixed bug: photos tagged as non-deer species (Rabbit, Turkey, etc.) now excluded from buck/doe review
- Added to buck/doe review:
  - Unknown (U) button - marks as unknown without tagging
  - Properties (P) button - navigates to photo in main window
  - Green highlighting for reviewed items (like species review)
  - Suggestions cleared from DB only when dialog closes
- Changed button labels from "Accept as Buck/Doe" to just "Buck/Doe"
- Reran buck/doe model on 870 deer photos that had Unknown sex

**Files changed:**
- `training/label_tool.py` - Major updates to review_sex_suggestions():
  - Added filter controls (sex, confidence, date)
  - Changed reviewed tracking to use photo IDs instead of row indices
  - Added Unknown and Properties buttons
  - Fixed zoom behavior to match main window
  - Added green highlighting for reviewed items
  - Fixed _gather_pending_sex_suggestions() to exclude non-deer species
  - Added date field to pending items
- `preview_window.py` - Fixed QComboBox method calls (textChanged → currentTextChanged, text() → currentText())
- `models/buckdoe.onnx` - Retrained with 1,275 photos

**Key insight for future:**
- Buck/doe model currently uses full photos
- Need deer head detector + head crops for:
  - Better buck/doe accuracy (antlers are on head)
  - Antler point counting
  - Antler measurement
  - Individual deer re-ID

---

### December 16, 2024 (Session 2)
**What we did:**
- Fixed labels.txt again (was still corrupted with partial entries like "Bob", "C", "Opp")
- Added VALID_SPECIES protection in two places:
  - `training/label_tool.py` - Only writes approved species to labels.txt
  - `ai_suggester.py` - Filters out invalid labels when loading, rejects invalid predictions
- Cleared bad AI suggestions from database (187 photos had junk labels)
- Re-ran AI on 237 unlabeled photos with fixed labels
- Added MPS (Apple Metal GPU) support to training script for 5-10x faster training
- Retrained species model with simplified 5 categories:
  - Deer, Empty, Turkey, Other Mammal, Other
- Fixed multi-species tagging in species review dialog
- Added zoom improvements and trackpad support

---

### December 16, 2024 (Session 1)
**What we did:**
- Fixed labels.txt corruption (had junk entries like "VeriTION", partial names)
- Fixed spelling: Oppossum → Opossum in database and labels
- Changed all "Verification" tags to "Empty" (84 photos)
- Re-ran AI species suggestions with corrected labels (921 photos)
- Added AI rejection tracking for future model training
- Added sort dropdown to species review dialog
- Improved species review UX with green highlighting

---

## Ultimate Project Goal

**The main purpose of this app is tracking bucks through trail camera photos.**

Key capabilities needed:
1. **Identify deer** in photos (species detection) ✓ Done
2. **Classify buck vs doe** (sex classification) ✓ Done (needs improvement)
3. **Detect deer heads** for better analysis - TODO
4. **Count antler points** - TODO
5. **Measure antler characteristics** (spread, mass, etc.) - TODO
6. **Recognize individual bucks** across photos - TODO
7. **Track bucks across seasons/years** - Partially done (manual Deer ID system exists)

---

## 1. CORE APP FUNCTIONALITY

### 1.1 Photo Import & Organization
- [X] Import photos from folders/SD cards
- [X] Extract EXIF data (date, time, camera model)
- [X] Auto-rename files with timestamp and camera
- [X] Organize photos by Year/Month folders
- [X] Generate thumbnails for fast browsing
- [X] Detect and handle duplicate photos (MD5 hash)
- [X] CuddeLink auto-download with credentials UI
- [ ] Support RAW image formats (.CR2, .NEF, etc.)

### 1.2 Photo Browsing & Viewing
- [X] Thumbnail grid view for browsing
- [X] Full-size preview with zoom/pan
- [X] Side-by-side photo comparison (up to 4 photos)
- [X] Filter by tags, date range, camera location, antler year
- [X] Search by deer ID
- [X] View/Label Mode (simple) vs Advanced Mode (with AI) - Settings → Simple Mode toggle

### 1.3 Tagging System
- [X] Basic species tags (Deer, Turkey, Coyote, Raccoon, etc.)
- [X] Buck/Doe classification
- [X] Multi-species tagging (multiple animals in one photo)
- [X] Empty/Trash tagging for false triggers
- [X] Favorite marking
- [X] Notes field for each photo

### 1.4 Deer Tracking & Identification
- [X] Assign unique Deer IDs to individual bucks
- [X] Track deer across multiple photos
- [X] Age class assignment (1.5yr, 2.5yr, 3.5yr, 4.5yr+)
- [X] Antler point counting (manual - left/right, typical/abnormal)
- [X] Season/year tracking for each deer
- [X] Buck profile pages with all sightings
- [ ] Automated antler detection and counting
- [ ] Antler measurement from photos

### 1.5 Multi-Computer / Team Workflow (NEW)

**Goal:** Allow multiple people to download photos on different computers and share labels without needing to sync databases or install software updates.

**How it works:**
1. Each person downloads photos to their own computer
2. One "master" computer runs the full app and does all the labeling
3. Export labels to Excel file (keyed by photo filename + fixed metadata)
4. Share the Excel file with other computers
5. Other computers import the Excel to link labels to their local photos

**Key matching fields:**
- Photo filename (after rename: `YYYY-MM-DD_HH-MM-SS_CameraModel.jpg`)
- Original EXIF date/time (backup if filename differs)
- Camera model from EXIF
- File size or partial hash (optional verification)

**Tasks:**
- [X] **Export Labels to Excel** - Export all tags, deer IDs, species, buck/doe to .xlsx
- [X] **Import Labels from Excel** - Match by filename, apply labels to local photos
- [X] **Simple Mode** - Toggle in Settings menu hides AI features for simpler labeling
- [ ] **Conflict handling** - What if same photo has different labels? (newest wins, or flag for review)
- [ ] **Relink photos tool** - If photos moved, match by filename pattern and relink paths

**Use case:** Hunting club where 3-4 people check different cameras, one person labels on the main computer, others get updated Excel to see what deer were spotted.

---

## 2. AI/ML PIPELINE

### 2.1 Species Detection & Classification
- [X] Species classifier (Deer, Turkey, Empty, Other Mammal, Other)
- [X] Confidence scores for AI predictions
- [X] Review queue for AI suggestions with filters
- [X] VALID_SPECIES protection against label corruption
- [ ] Improve accuracy on rare species

### 2.2 Buck/Doe Classification
- [X] Buck vs Doe classifier (93.8% balanced accuracy)
- [X] Review queue with filters (Type, Confidence, Date)
- [X] Unknown option for unclear photos
- [ ] **Build deer head detector** (high priority)
- [ ] Train on head crops instead of full photos
- [ ] Fawn detection

### 2.3 Antler Analysis (Future - High Priority)
- [ ] Deer head detector model
- [ ] Auto-detect antler points from head crops
- [ ] Estimate antler score (Boone & Crockett style)
- [ ] Track antler growth across seasons
- [ ] Identify broken/damaged antlers

### 2.4 Individual Deer Re-ID (Future)
- [ ] Train re-identification model on labeled deer
- [ ] Generate embeddings for each deer sighting
- [ ] Match new photos to known deer automatically
- [ ] Suggest "This might be Buck #47" with confidence

### 2.5 Detection Boxes for Better AI (NEW)

**Goal:** Use the subject boxes and deer head boxes from the detector to improve classification accuracy.

**Current state:**
- Detector outputs bounding boxes for "subject" (whole animal) and "deer_head"
- These boxes are drawn on preview but not saved or used for classification
- Buck/doe model currently runs on full photos (includes background, multiple animals)

**Improvement plan:**
1. **Save detection boxes to database** - Store box coordinates with each photo
2. **Crop-based classification** - Run species classifier on subject crops, not full image
3. **Head-crop buck/doe** - Run buck/doe classifier on deer_head crops only (antlers are on head)
4. **Multi-animal handling** - If detector finds 2+ subjects, classify each separately
5. **Box-based training data** - Export crops for training better models

**Tasks:**
- [X] Add `detections` table to database - DONE (annotation_boxes table)
- [X] Save detector results when running AI suggestions - DONE
- [X] Modify species suggester to classify subject crops - DONE
- [X] Modify buck/doe suggester to use head crops - DONE (when deer_head boxes exist)
- [X] UI to view/edit detection boxes - DONE
- [ ] Export detection crops for training

**Why this matters:**
- Head crops for buck/doe = antlers visible = much better accuracy
- Subject crops = less background noise = better species ID
- Foundation for antler point counting (need head box first)

---

## 3. IMMEDIATE PRIORITIES (Next Steps)

### High Priority
1. ~~**Save detection boxes to database**~~ - DONE (2,010 boxes stored for 1,611 photos)
2. ~~**Head-crop buck/doe classifier**~~ - DONE (code uses deer_head boxes when available, only 63 exist - need more head box training data)
3. Continue labeling to improve model accuracy
4. **Set up GitHub repository** - Version control for the app
    - Create GitHub account (if needed)
    - Initialize git repo in project folder
    - Create .gitignore for venv, __pycache__, .db files, etc.
    - Push to GitHub for backup and version history
    - Enables rolling back changes if something breaks
5. **Remove Excel export/import features** - Using Supabase now
    - Remove "Export Labels to Excel" menu item
    - Remove "Import Labels from Excel" menu item
    - Remove related code and openpyxl dependency

### Medium Priority
4. **Portable View/Label Mode** - Lightweight version for other computers
6. Fix CuddeLink auto-download from internet
7. View/Label Mode vs Advanced Mode toggle

### Lower Priority (Future)
8. Antler point auto-detection (requires head boxes)
9. Individual deer re-ID
10. Export deer profiles to PDF
11. Relink photos tool (for moved files)
12. **Timestamp correction overlay** - Replace burned-in wrong timestamps with corrected ones
    - Detect timestamp region on photo (OCR or fixed position by camera model)
    - Overlay corrected timestamp from database
    - Add subtle indicator showing timestamp was corrected (small icon or different font color)
    - Useful when camera clock was set incorrectly
13. **AI suggestions optimization** - Filter out already-labeled photos before processing
    - Only count unlabeled photos in progress bar total
    - Skip labeled photos entirely (don't include in loop)
    - Makes progress bar more accurate and processing faster
    - Show per-species counts in progress bar (e.g., "Deer: 45 | Turkey: 12 | Buck: 30 | Doe: 15")
14. **Background AI processing** - Run AI suggestions without blocking the UI
    - Use QThread for AI processing in separate thread
    - Non-modal progress indicator (status bar or floating widget)
    - Allow user to continue browsing/labeling while AI runs
    - Emit signals to update progress and notify when complete
    - **NOTE:** Save a backup copy of the app folder before implementing this feature

---

## 4. MULTI-PLATFORM DISTRIBUTION (Long-Term Vision)

**Goal:** Make the app available everywhere users want it, with no technical setup required.

### Target Platforms

| Platform | App Type | Status | Priority |
|----------|----------|--------|----------|
| macOS | Standalone .app | Partial (runs via Python) | High |
| Windows | Standalone .exe | Partial (needs C++ fix) | High |
| Android | Native/Flutter app | Not started | Medium |
| iPhone | Native/Flutter app | Not started | Medium |
| Web | Browser app | Not started | Medium |

### Desktop Standalone Apps (No Python/C++ Required)

**Current Blockers:**
- Users must install Python to run the app (or use bundled .exe/.app)
- ~~Windows users need Visual C++ Build Tools~~ FIXED (supabase_rest.py)
- No code signing (triggers security warnings)

**Tasks:**
- [X] **Replace supabase package with REST API calls** - DONE (supabase_rest.py)
- [X] **Finalize PyInstaller Windows build** - DONE (run build_windows.bat on Windows)
- [X] **Create macOS .app bundle** - DONE (TrailCamOrganizer.app, 489MB standalone)
- [ ] **Code signing (optional)** - Removes "unidentified developer" warnings
- [ ] **Auto-updater** - Check for new versions on startup

**Estimated effort:** 6-10 hours total

### Mobile Apps (Future)

**Approach Options:**
1. **Flutter/React Native** - Single codebase for iOS + Android
2. **Native (Swift/Kotlin)** - Best performance, more work
3. **PWA (Progressive Web App)** - Web app that installs like native

**Mobile Feature Scope (simpler than desktop):**
- View photos synced from cloud
- Browse/filter by location, species, deer ID
- View buck profiles and sightings
- Add tags and notes
- Does NOT need: photo import, AI inference, training tools

**Tasks (Future):**
- [ ] Choose mobile framework (Flutter recommended)
- [ ] Design mobile-friendly UI
- [ ] Build photo viewer with cloud sync
- [ ] Build buck profile browser
- [ ] App store submissions

### Web App (Future)

**Purpose:** View-only access from any browser, no install needed.

**Web Feature Scope:**
- Dashboard with recent photos
- Browse photos with filters
- View buck profiles and history
- Basic tagging (species, notes)
- Does NOT need: photo import, local file access, AI

**Technology Options:**
- Supabase + Next.js/React
- Supabase + basic HTML/JavaScript
- The `web/` folder already exists for future web features

**Tasks (Future):**
- [ ] Design web dashboard
- [ ] Build photo browser with Supabase backend
- [ ] Build buck profile pages
- [ ] Deploy to hosting (Vercel, Netlify, etc.)

### Shared Backend Strategy

All platforms should sync through **Supabase**:
- Desktop apps push/pull labels to cloud
- Mobile apps read from cloud, can add tags
- Web app provides browser access to same data
- Photos stay local on desktop; only metadata syncs

This means the current Supabase integration is foundational for all future platforms.

---

## Quick Reference

**Start the app:**
```bash
cd /Users/brookebratcher/Desktop/Trail\ Camera\ Software
python main.py
```

**Key Locations:**
- Database: `~/.trailcam/trailcam.db`
- Models: `models/` folder (species.onnx, buckdoe.onnx)
- Main UI code: `training/label_tool.py`
- AI suggester: `ai_suggester.py`

**Current model stats:**
- Species: 5 categories (Deer, Empty, Turkey, Other Mammal, Other)
- Buck/Doe: 93.8% balanced accuracy (trained on 1,275 photos)
  - Buck: 90.6% (997 training photos)
  - Doe: 97.1% (277 training photos)

**Database Quick Checks:**
```bash
# Buck/doe tag counts
sqlite3 ~/.trailcam/trailcam.db "SELECT tag_name, COUNT(*) FROM tags WHERE LOWER(tag_name) IN ('buck', 'doe') GROUP BY tag_name;"

# Pending buck/doe suggestions
sqlite3 ~/.trailcam/trailcam.db "SELECT COUNT(*) FROM photos WHERE suggested_sex IS NOT NULL AND suggested_sex != '';"

# All tag distribution
sqlite3 ~/.trailcam/trailcam.db "SELECT tag_name, COUNT(*) FROM tags GROUP BY tag_name ORDER BY COUNT(*) DESC;"
```
