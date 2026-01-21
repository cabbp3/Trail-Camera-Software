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

## Code Audit Findings (Jan 20, 2026)

Comprehensive audit across desktop and mobile apps. Items below are prioritized for future work.

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
