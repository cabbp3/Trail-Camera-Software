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

## Recently Completed

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
