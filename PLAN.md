# Trail Camera Software - Project Plan

**Last Updated:** December 26, 2024 (Session 12)

---

## Session 12 Progress (Dec 26, 2024)

**Problem Identified:** Windows computer couldn't sync labels from Supabase because CuddeLink generates unique filenames per download session (like `2025-12-24T21_58_40.934128-1.jpeg`). Mac and Windows had different filenames for the same photos, so the sync key (`original_name|date_taken|camera_model`) didn't match.

**Solution Implemented:** Added `file_hash` (MD5) column for content-based matching.

**Completed:**
- [X] Added `file_hash` column to local database schema
- [X] Added `calculate_missing_hashes()` method to database.py
- [X] Updated `push_to_supabase()` to include file_hash
- [X] Updated `pull_from_supabase()` to use file_hash as fallback matching
- [X] Updated `_get_photo_by_key()` to try file_hash when photo_key doesn't match
- [X] Calculated MD5 hashes for all 4,422 Mac photos
- [X] Created `windows_fix.py` script for Windows cleanup and hash-based sync
- [X] Created `supabase_add_file_hash.sql` migration script

**Still TODO (resume here):**
- [ ] Run `supabase_add_file_hash.sql` in Supabase dashboard to add file_hash columns
- [ ] Push from Mac to Supabase (will include file hashes)
- [ ] On Windows: Run `python windows_fix.py` to calculate hashes and pull labels

**Supabase SQL to run:**
```sql
ALTER TABLE photos_sync ADD COLUMN IF NOT EXISTS file_hash TEXT;
ALTER TABLE tags ADD COLUMN IF NOT EXISTS file_hash TEXT;
ALTER TABLE deer_metadata ADD COLUMN IF NOT EXISTS file_hash TEXT;
ALTER TABLE deer_additional ADD COLUMN IF NOT EXISTS file_hash TEXT;
ALTER TABLE annotation_boxes ADD COLUMN IF NOT EXISTS file_hash TEXT;
CREATE INDEX IF NOT EXISTS idx_photos_sync_file_hash ON photos_sync(file_hash);
CREATE INDEX IF NOT EXISTS idx_tags_file_hash ON tags(file_hash);
```

**After running SQL, push from Mac:**
```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
source .venv/bin/activate
python -c "
from database import TrailCamDatabase
import supabase_rest
db = TrailCamDatabase()
client = supabase_rest.create_client(
    'https://iwvehmthbjcvdqjqxtty.supabase.co',
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml3dmVobXRoYmpjdmRxanF4dHR5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjYyMDI0NDQsImV4cCI6MjA4MTc3ODQ0NH0._z6WAfUBP_Qda0IcjTS_LEI_J7r147BrmSib3dyneLE'
)
db.push_to_supabase(client)
"
```

**On Windows:**
```cmd
cd "C:\Users\mbroo\OneDrive\Desktop\Trail Camera Software V 1.0"
python windows_fix.py
```

---

## Timestamp Corrections Needed

**216 photos at WB 27 need date correction:**
- Currently set to: July 1-8, 2024
- Original (wrong) dates: January 1-8, 2022
- These came from a camera with incorrect clock settings
- Need to determine actual dates and apply correction

To find these photos:
```sql
SELECT * FROM photos WHERE date_taken LIKE '2024-07%' AND camera_location = 'WB 27';
```

This document tracks all tasks for the Trail Camera Photo Organizer. Tasks marked with `[X]` are complete.

---

## Product Vision

This software is intended to become a **marketable product** for hunters and wildlife enthusiasts. Not all features may ship (e.g., training pipelines stay internal), but user-facing features should be polished, intuitive, and reliable.

---

## Current Status Summary

- **4,422 photos** in the database (all with file hashes calculated)
- **macOS** standalone build working
- **Windows** standalone build working (run windows_fix.py for sync)
- **Supabase cloud sync** working via REST API with hash-based matching
- **Species model:** 12 classes, 96.3% accuracy
- **Buck/doe model:** 93.8% balanced accuracy
- **GitHub:** https://github.com/cabbp3/Trail-Camera-Software

---

## Session 11 Progress (Dec 25, 2024)

**Windows Standalone Build:**
- [X] Created BUILD.bat, RUN.bat, Create Desktop Shortcut.vbs
- [X] Created icon.ico for Windows
- [X] Fixed Python version issue (must use 3.11, not 3.14)
- [X] Fixed missing database columns (`collection`, `left_ab_points_max`, `right_ab_points_max`)
- [ ] Complete build (McAfee blocking .exe - needs folder exclusion)
- [ ] Test standalone .exe on Windows

**Future Improvements:**
- [ ] Code signing certificate (prevents antivirus false positives)
- [ ] GitHub Actions for automated Windows/macOS builds

---

## Pending Tasks

- [ ] Complete Windows standalone build
- [ ] Batch editing by date range (for camera moves)
- [ ] Queue mode performance optimization
- [ ] Deer head detection training (need more head boxes)

---

## Key Technical Notes

### Camera Location via OCR
- Photos have timestamp watermarks with location names
- Regex: `r'[AP]M\s+(.+?)\s+\d{3}'` extracts location text
- OCR mappings for variations: "WB27" → "WB 27", "SALTLICKE" → "Salt Lick"
- Date-based mappings for camera moves: "WEST OF ROAD" before Oct 25 = "West Salt Lick", after = "West Triangle"

### Database Schema Additions
- `stamp_location` - Raw OCR text from photo timestamp
- `camera_location` - Verified location name
- `sites` table - Site clustering results
- `annotation_boxes` - Detection boxes from AI

### Supabase Tables
- `photos_sync`, `tags`, `deer_metadata`, `deer_additional`
- `buck_profiles`, `buck_profile_seasons`, `annotation_boxes`
- Photos matched by `original_name|date_taken|camera_model` key

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
14. **[X] Background AI processing** - Run AI suggestions without blocking the UI
    - IMPLEMENTED Dec 24, 2024 - `AIWorker` QThread class in label_tool.py
    - Menu: Tools → Suggest Tags (AI) — Background + Live Queue
    - Photos appear in queue as they're processed
    - Progress bar shows AI processing status
15. **Integrated queue mode performance** - Queue mode runs slow, needs optimization
    - Current implementation re-filters and rebuilds photo list on each change
    - Possible fixes: lazy loading, reduced database queries, caching
    - May need to profile to find bottlenecks

---

## 4. MULTI-PLATFORM DISTRIBUTION (Long-Term Vision)

**Goal:** Make the app available everywhere users want it, with no technical setup required.

### Target Platforms

| Platform | App Type | Status | Priority |
|----------|----------|--------|----------|
| macOS | Standalone .app | Done | High |
| Windows | Standalone .exe | Done | High |
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
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
source .venv/bin/activate
python main.py
```

**Key Locations:**
- Database: `~/.trailcam/trailcam.db`
- Models: `models/` folder (species.onnx, buckdoe.onnx)
- Main UI code: `training/label_tool.py`
- AI suggester: `ai_suggester.py`

**Current model stats:**
- Species: 12 classes, 96.3% accuracy
- Buck/Doe: 93.8% balanced accuracy

**Database Quick Checks:**
```bash
# Buck/doe tag counts
sqlite3 ~/.trailcam/trailcam.db "SELECT tag_name, COUNT(*) FROM tags WHERE LOWER(tag_name) IN ('buck', 'doe') GROUP BY tag_name;"

# Pending buck/doe suggestions
sqlite3 ~/.trailcam/trailcam.db "SELECT COUNT(*) FROM photos WHERE suggested_sex IS NOT NULL AND suggested_sex != '';"

# All tag distribution
sqlite3 ~/.trailcam/trailcam.db "SELECT tag_name, COUNT(*) FROM tags GROUP BY tag_name ORDER BY COUNT(*) DESC;"
```
