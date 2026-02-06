# TASKS - Active Work

## Instructions for Claude

1. **Starting work:** Add your task below with a random 4-character ID (e.g., `A7X2`)
2. **While working:** Update status as needed
3. **When done:** Delete your task entry
4. **Check first:** Make sure another Claude isn't already working on the same thing

---

## Active Tasks

| ID | Task | Status | Started |
|----|------|--------|---------|
| T4R2 | Species classifier training - paused at epoch 11/50 (93.8% best) | Paused | Jan 23 |

*(Completed tasks moved to Recently Completed section)*


---

## Overnight Task Lists (Jan 20, 2026)

Three Claudes can work in parallel on these non-overlapping tasks.

### Claude 1 - Desktop Auto-Pull Feature (HIGHEST PRIORITY)

**Goal:** When desktop app opens, check for new photos in cloud and offer to download them.

**Files you OWN (only you touch these):**
- `training/label_tool.py` - UI changes
- `supabase_rest.py` - Query for new photos
- `r2_storage.py` - Download photos/thumbnails
- `database.py` - Insert new photo records

**Tasks:**
1. Add `get_cloud_only_photos()` to `supabase_rest.py` - query photos_sync for file_hashes not in local DB
2. Add `download_photo()` and `download_thumbnail()` to `r2_storage.py` if not exists
3. On app startup in `label_tool.py`:
   - Call Supabase to check for new photos
   - If found, show dialog: "X new photos found in cloud. Download?"
   - If user confirms, download thumbnails (and optionally full photos) from R2
   - Show progress bar during download
   - Insert new records into local database
4. Test the full flow

**DO NOT TOUCH:** Mobile app, utility files, site_*.py files

---

### Claude 2 - Mobile App Refactoring

**Goal:** Split the monolithic supabase_service.dart into smaller, focused services.

**Files you OWN (only you touch these):**
- Everything in `/Users/brookebratcher/Desktop/trailcam_mobile/`

**Tasks:**
1. Create `lib/services/photo_service.dart` - extract photo-related methods
2. Create `lib/services/deer_service.dart` - extract deer metadata/profile methods
3. Create `lib/services/sync_service.dart` - extract sync-related methods
4. Update imports in all screens that use supabase_service.dart
5. Run `flutter analyze` to check for errors
6. Test the app still works: `flutter run`

**DO NOT TOUCH:** Desktop app files

---

### Claude 3 - Desktop Cleanup (Isolated Files)

**Goal:** Clean up utility files without touching core app files.

**Files you OWN (only you touch these):**
- `site_clustering.py` - DELETE (dead code)
- `requirements.txt` - cleanup
- `image_processor.py` - logging
- `cuddelink_downloader.py` - logging
- `duplicate_dialog.py` - logging
- `compare_window.py` - logging
- `preview_window.py` - logging

**Tasks:**
1. Delete `site_clustering.py` (568 lines of dead code, replaced by site_identifier.py)
2. Edit `requirements.txt`: remove `openpyxl`, add `pytesseract`
3. In each utility file listed above:
   - Add `import logging` and `logger = logging.getLogger(__name__)` if missing
   - Replace `print()` statements with `logger.info()`, `logger.warning()`, `logger.error()`
   - Keep user-facing print() in CLI scripts
4. Run the app to verify nothing broke

**DO NOT TOUCH:** label_tool.py, database.py, supabase_rest.py, r2_storage.py, mobile app, ai_*.py, site_detector.py, site_embedder.py, site_identifier.py

---

## Code Audit Findings (Jan 20, 2026)

Comprehensive audit across desktop and mobile apps. Items below are prioritized for future work.

### HIGHEST Priority - Feature

| Issue | Location | Status |
|-------|----------|--------|
| ~~Desktop auto-pull from cloud~~ | `training/label_tool.py`, `supabase_rest.py`, `r2_storage.py`, `database.py` | **DONE** - [K7P2] Implemented Jan 20 |

### HIGH Priority - Architecture

| Issue | Location | Effort |
|-------|----------|--------|
| Monolithic god class | `training/label_tool.py` (14,776 lines, 326+ methods) | Split into: UIComponents, PhotoManager, AIManager, SyncManager, DialogManager. ~2-3 days. |
| Silent exception swallowing | 51 instances of bare `except Exception:` with pass/print | Add proper logging, re-raise where appropriate. ~4 hours. |
| N+1 queries in sync | `database.py` sync operations, `supabase_rest.py` | Batch queries, use JOINs. ~2-3 hours. |

### MEDIUM Priority - Performance & Cleanup

| Issue | Location | Effort |
|-------|----------|--------|
| Missing indexes | `annotation_boxes.photo_id`, `photos.file_hash` | Add indexes. ~30 min. |
| Dead code | `site_clustering.py` (568 lines) | Replaced by `site_identifier.py`. Delete or deprecate. ~15 min. |
| Duplicated dialog patterns | 29 similar dialogs in `label_tool.py` (~800 lines) | Extract `DialogFactory` class. ~2 hours. |
| Mobile service file | `trailcam_mobile/lib/services/supabase_service.dart` (1,363 lines) | Split into PhotoService, DeerService, SyncService. ~1 day. |

### LOW Priority - Code Quality

| Issue | Location | Notes |
|-------|----------|-------|
| Hardcoded paths/values | Various files | Extract to config. |
| Inconsistent logging | Mix of print() and logger | Standardize on logger. |
| `openpyxl` unused | `requirements.txt` | Remove dependency. |
| `pytesseract` missing | `requirements.txt` | Add for OCR features. |

### Positive Findings

- Good database schema with proper foreign keys and constraints
- Thread safety with QThread for long operations
- Effective use of SQLite WAL mode
- Clean ONNX model integration
- Good separation of AI inference from UI

---

## Recently Completed

- **(Feb 2) CRITICAL: Fixed labels being lost** - Three bugs causing label loss: (1) `pull_from_supabase()` in database.py was overwriting local species/sex with NULL from cloud - fixed with COALESCE; (2) Buck/doe queue coordinate matching failed after `set_boxes()` reassigned IDs - fixed fallback logic in 4 functions; (3) Box labels weren't creating photo tags - synced 2,300+ missing Deer tags and 3,000+ Buck/Doe tags. Labels now persist through sync.
- (Feb 2) Tagged all 288 photos at exactly 9:00 AM as "Verification" (camera test shots)
- (Jan 26) Database duplicate cleanup - migrated 125 deer_metadata + 3 boxes + 1 tag, deleted 171 duplicate photos and 85 orphaned `.cuddelink_tmp` records. Fixed imports to skip temp folders (`_find_image_files()`, `scan_folders()`). Freed 14 MB.
- (Jan 26) Cloud sync SQL fix - `since_clause()` was returning tuple but used directly in f-strings at 4 locations in `database.py`. Fixed to properly unpack `(clause, params)`.
- (Jan 26) Mobile app landing page - added bottom navigation (Photos, Account, Settings), updated splash_screen.dart to navigate to HomeScreen
- (Jan 26) Individual buck identification roadmap - created `docs/individual_buck_identification.md` documenting multi-feature discrimination approach, annotation strategy, data model, ML architecture for long-term individual deer tracking
- (Jan 26) Cloud sync fixes - fixed incremental pull (added .gt() method to supabase_rest.py), fixed timezone format (+00:00 suffix), added deletion sync, added full photo download option on startup
- (Jan 26) [B7X4] Box-level species suggestions architecture fix - AI pipeline was storing suggestions at photo level instead of per-box. Fixed AIWorker, rerun_ai_current_photo, and rerun_ai_on_selection to store `ai_suggested_species` on each annotation_box. Added `db.set_box_ai_suggestion()` method. Re-ran AI on 3,575 boxes without labels. Added missing indexes (idx_boxes_photo_id, idx_boxes_label, idx_photos_file_hash).
- (Jan 26) [B7X4] Verification photo detection - small files (<50KB) taken at exactly 9:00:00 AM are camera test shots. Auto-labeled 266 as "Verification" and deleted their subject boxes. Non-9AM small files kept as regular photos with species suggestions.
- (Jan 22) [D8K3] Desktop cleanup + database audit - deleted site_clustering.py (568 lines dead code), updated requirements.txt (removed openpyxl, added pytesseract), converted print() to logger in 4 utility files. Database audit: integrity checks passed, cleaned 38 orphaned records via WAL checkpoint, verified audit log matches DB (99.88% accuracy, 11 unlogged tags from Supabase sync).
- (Jan 22) [P2K7] Pixel area confidence scaling - trained species model v5.0 (92.8% test accuracy), added pixel area logging during training, wired up confidence scaling in ai_suggester.py and label_tool.py. Small detection boxes now get reduced confidence scores.
- (Jan 22) [X7MD] Fixed MegaDetector disappearing - fixed broken venv activate script (folder was renamed), moved model to persistent `~/.trailcam/md_v5a.0.1.pt` instead of temp folder, added error tracking with `is_available` and `error_message` properties.
- (Jan 21) [C3H0] Auto-archive feature - Settings → Auto-Archive in desktop (Settings menu) and mobile (gear icon). Archives photos after sync that don't match selected species (Bucks Only or custom). Favorites and unlabeled photos always kept.
- (Jan 20) [M4X9] Mobile app refactoring - split supabase_service.dart (1,363 lines) into deer_service.dart (280 lines) and label_service.dart (340 lines), now 655 lines
- (Jan 20) [K7P2] Desktop auto-pull from cloud - checks for new photos in Supabase on startup, prompts user to download thumbnails from R2 with progress bar, menu option under Tools > Cloud Sync
- (Jan 20) [C3H0] Fixed annotation_boxes sync in database.py
- (Jan 20) Comprehensive code audit across desktop and mobile apps - no changes made, findings documented above
- (Jan 19) Implemented auto-sync to Supabase with offline queueing - created `sync_manager.py` with 30-second debounce timer, persistent offline queue at `~/.trailcam/pending_sync.json`, close warning for pending changes
- (Jan 18) Fixed `pull_from_supabase()` to include annotation_boxes - was missing entirely, now syncs box coordinates, species, sex, and head annotations
- (Jan 14) [W8KP] Fixed Properties button in buck/doe review queue - now correctly navigates to photo (even if archived) by resetting filters and refreshing photo list
- (Jan 14) [Q3KJ] Added collection filter to buck/doe review queue - filter by collection when reviewing pending suggestions
- (Jan 14) [P7XM] Fixed mobile app "!" errors: uploaded 47 missing thumbnails to R2, deleted 130 duplicate Supabase records, fixed mobile compile error (finalHashes scope bug from M3QP changes)
- (Jan 14) [K2VN] Reinstalled MegaDetector (was missing from venv). Fixed WAL database corruption. Explained corruption cause: forceful app termination corrupts WAL.
- (Jan 14) [K8VN] Site identification improvements: Created `site_embedder.py` (semantic MobileNetV2 - 70%), `site_detector.py` (OCR - 91%), and `site_identifier.py` (hybrid). Integrated into Tools → Auto-Detect Sites menu. Uses OCR first, falls back to visual matching.
- (Jan 13) [M3QP] Mobile app fixes: removed 200 hash limit, fixed deer ID N+1 queries, added URL cache limit, added HTTP timeouts
- (Jan 13) [R4WJ] Archive sync to mobile - verified already implemented (filter + UI toggle)
- (Jan 13) [7X9K] Fixed app lockup (WAL corruption) and sync failure (4 duplicate photos deleted)
- (Jan 13) Fixed sex filter bug in mobile app - was filtering in-memory after pagination
- (Jan 13) Reorganized documentation files, created INDEX.md
