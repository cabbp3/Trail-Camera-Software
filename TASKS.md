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
| (none) | | | |

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

| Issue | Location | Effort |
|-------|----------|--------|
| Desktop auto-pull from cloud | `training/label_tool.py`, `supabase_rest.py`, `r2_storage.py` | On app open: query Supabase for photos not in local DB, prompt user to download, show progress bar while pulling thumbnails/photos from R2. ~1 day. |

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
