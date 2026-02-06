# HANDOFF - Session Notes for Next Claude

**Last Updated:** Feb 2, 2026
**Last Deep Dive:** Feb 2, 2026

---

## CRITICAL FIX (Feb 2, 2026) - Labels Were Being Lost

### Problem
User was labeling photos but labels kept disappearing. "We keep losing labels."

### Root Causes Found & Fixed

1. **Cloud sync overwriting local labels with NULL** (`database.py:3223-3235`)
   - `pull_from_supabase()` was updating local annotation_boxes with cloud values
   - Cloud had NULL for species/sex, which overwrote local labels
   - **Fix:** Changed UPDATE to use `COALESCE(?, local_value)` - only updates if cloud has data

2. **Box coordinate matching failing** (`label_tool.py` - 4 functions)
   - When accepting buck/doe suggestions, `set_boxes()` deletes and re-inserts all boxes with NEW IDs
   - Code tried to match by ID first, but old ID no longer existed
   - Coordinate fallback only ran if `box_id is None`, which it wasn't
   - **Fix:** Changed to try ID first, then ALWAYS try coordinate matching as fallback
   - Functions fixed: `_accept_as`, `_reject`, `_unknown`, `_set_species`

3. **Tags not being created from box labels**
   - Boxes had `species='Deer'` but photos had no Deer tag
   - ~2,300+ photos affected
   - **Fix:** Synced all missing tags (Deer, Buck, Doe) from box labels to tags table

### Verification
After fixes:
- Local database: 0 photos with Deer box but missing Deer tag
- Cloud sync no longer overwrites local labels
- Buck/doe queue properly saves accept/reject decisions

---

## Current System State

### Data Counts (as of Feb 2, 2026)

| Location | Photos | Tags | Annotation Boxes | Deer Metadata |
|----------|--------|------|------------------|---------------|
| Local SQLite | 8,521 | 12,322 | 12,216 | ~7,000 |
| Supabase | ~8,200 | ~12,000 | synced | synced |
| R2 Storage | 8,500+ photos + thumbs | - | - | - |

### Storage Usage
- **R2:** ~5.2 GB / 10 GB free tier
- **Local photos:** ~5.5 GB at `~/TrailCamLibrary/`
- **Local database:** ~12 MB at `~/.trailcam/trailcam.db`

### What's Working
- Desktop app runs fine
- **Labels now persist through cloud sync** (fixed Feb 2)
- CuddeLink GitHub Action downloading 3x daily (6:15 AM, 11:15 AM, 8:15 PM Central)
- R2 storage healthy - all photos have thumbnails
- R2 download from cloud working
- Mobile app connects to Supabase/R2 with correct endpoints
- 100% of local photos have file_hash
- Collections: Brooke Farm, Tightwad House, East Side Hunting Club

### Tag Distribution (Local - Feb 2, 2026)
| Species | Count |
|---------|-------|
| Deer | 5,576 |
| Buck | 3,041 |
| Doe | 1,512 |
| Empty | 712 |
| Squirrel | 393 |
| Verification | 288 |
| Turkey | 227 |
| Raccoon | 156 |
| Opossum | 136 |

---

## Known Issues (Priority Order)

### 1. ~~FIXED: Duplicate Photos in Supabase~~
- ✅ **Cleaned:** Deleted 1,751 duplicate records
- ✅ **Root cause fixed:** Changed upsert conflict key from `photo_key` to `file_hash`
- ✅ **Unique constraint added:** `ALTER TABLE photos_sync ADD CONSTRAINT photos_sync_file_hash_unique UNIQUE (file_hash);`

### 2. LOW: Annotation Boxes Not in Supabase
- **Status:** Not a priority per user (Jan 26)
- **Problem:** 11,815 boxes locally, 0 in Supabase
- **Impact:** Mobile app can't show detection boxes (acceptable for now)

### 3. LOW: Tags Sync Gap
- **Problem:** Minor differences between local and Supabase tags
- **Impact:** Minimal - most tags are synced

---

## Recent Fixes Applied This Session

### 1. supabase_rest.py Config Fix
- **File:** `supabase_rest.py` line 253
- **Problem:** `get_client()` looked for `key` but config has `anon_key`
- **Fix:** Now checks both: `config.get("key") or config.get("anon_key")`
- **Impact:** Supabase client now initializes correctly from `cloud_config.json`

### 2. annotation_boxes Sync Bug Fix (C3H0)
- **File:** `database.py` lines 2513-2534, 2562
- **Problem:** All 7,961 annotation boxes had `updated_at` BEFORE `last_push_at`
  - Incremental sync only pushes boxes where `updated_at > last_push_at`
  - All boxes failed this check → 0 boxes synced to Supabase
- **Root cause:** Early syncs set `last_push_at` when <100 boxes existed (skipped by safety check). Later boxes were added but had older timestamps.
- **Fix:** Added check for empty cloud table. If cloud `annotation_boxes` is empty but local has >100, force full sync (skip `since_clause` filter)
- **Impact:** Next push to Supabase will sync all 7,961 boxes to cloud. Mobile app will then show detection boxes and ML species.

### 3. Duplicate Photos Cleanup & Prevention (C3H0)
- **Problem:** 1,751 duplicate records in `photos_sync` (same file_hash, different photo_key)
- **Root cause:** Upsert used `photo_key` for conflict resolution, but CuddeLink and desktop create different photo_key formats
- **Cleanup:** Deleted 1,751 duplicate records, keeping most recent `updated_at` for each file_hash
- **Prevention fixes:**
  - `database.py`: Changed all upserts to use `file_hash` instead of `photo_key` as conflict key
  - `scripts/cuddelink_to_r2.py`: Added `?on_conflict=file_hash` to Supabase insert
- **Manual step required:** Run this SQL in Supabase Dashboard to add unique constraint:
  ```sql
  ALTER TABLE photos_sync ADD CONSTRAINT photos_sync_file_hash_key UNIQUE (file_hash);
  ```
- **Final state:** 8,026 unique photos in Supabase (no duplicates)

### 4. R2 Download Crash Fix (R2F9)
- **Problem:** App crashed when downloading photos from R2
- **Root cause:** Worker threads were calling `self.db.conn.execute()` directly, bypassing the database lock
- **Fixes applied:**
  - Added thread-safe `update_date_taken()` method to `database.py`
  - Changed workers to use `db.update_date_taken()` instead of direct SQL
  - Added comprehensive try/except error handling around worker `run()` methods
- **Files changed:** `database.py`, `training/label_tool.py`

### 5. Deep Dive Security & Stability Fixes (S3C7)
- **Problem:** Comprehensive audit revealed multiple security and stability issues
- **Fixes applied:**
  1. **Thread safety** - Added `with self._lock:` to 8+ database methods: `_log_tag_change`, `verify_audit_log`, `remove_tag`, `get_tags`, `get_all_distinct_tags`, `set_deer_metadata`, etc.
  2. **SQL injection** - Changed `since_clause()` from string interpolation to parameterized queries with `?` placeholders
  3. **Socket leak** - Added `sock.close()` after network check in `sync_manager.py`
  4. **Path traversal** - Added `.resolve()` validation in `web/server.py` to prevent `../` attacks
  5. **Command injection** - Removed `shell=True` from `subprocess.Popen()` in `updater.py`
  6. **Resource leaks** - Added context managers (`with Image.open()`) in `ai_detection.py` and `site_detector.py`
- **Files changed:** `database.py`, `sync_manager.py`, `web/server.py`, `updater.py`, `ai_detection.py`, `site_detector.py`
- **Remaining issues documented:** 125+ silent exception handlers, training scripts with unguarded Image.open()

### 6. MegaDetector Disappearing Fix (X7MD)
- **Problem:** MegaDetector would silently stop working due to multiple issues
- **Root causes found:**
  1. Broken venv activate script - folder was renamed from "Trail Camera Software" to "Trail Camera Software V 1.0" but venv paths weren't updated
  2. Model stored in macOS temp folder (`/var/folders/...`) which gets cleared on restart
  3. Silent failures - all errors returned empty list instead of raising exceptions
- **Fixes applied:**
  - Fixed `.venv/bin/activate` VIRTUAL_ENV path
  - Updated `ai_detection.py` and `label_tool.py` to use persistent model at `~/.trailcam/md_v5a.0.1.pt`
  - Added `is_available` and `error_message` properties for better error tracking
  - Copied model to persistent location (won't disappear on restart)
- **Model files now at:**
  - `~/.trailcam/md_v5a.0.0.pt` (267.8 MB) - older version
  - `~/.trailcam/md_v5a.0.1.pt` (280.8 MB) - newer version, preferred

---

## Architecture Overview

### Data Flow
```
CuddeLink Cameras
       ↓
GitHub Action (3x daily) ──→ R2 (photos + thumbnails)
       ↓                          ↓
   Supabase ←──── sync ────→ Desktop App
       ↓                          ↓
  Mobile App              Local SQLite DB
```

### Sync Tables (Supabase)
- `photos_sync` - Photo metadata (date, camera, collection, file_hash)
- `tags` - Species labels (file_hash + tag_name)
- `annotation_boxes` - Detection boxes (currently empty!)
- `deer_metadata` - Deer ID assignments

### Key Files
| File | Purpose |
|------|---------|
| `training/label_tool.py` | Main desktop app (~13,500 lines) |
| `sync_manager.py` | Auto-sync with 30s debounce, offline queue |
| `supabase_rest.py` | Supabase REST client (no C++ deps) |
| `r2_storage.py` | Cloudflare R2 client |
| `scripts/cuddelink_to_r2.py` | GitHub Action sync script |
| `.github/workflows/cuddelink-sync.yml` | Scheduled CuddeLink downloads |

### Config Files
- `cloud_config.json` - R2 and Supabase credentials (in app folder)
- `~/.trailcam/r2_config.json` - User R2 config (takes priority)
- `~/.trailcam/pending_sync.json` - Offline sync queue

---

## Quick Diagnostic Commands

```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
source .venv/bin/activate

# Check R2 storage usage
python3 -c "
from r2_storage import R2Storage
r2 = R2Storage()
print(f'R2: {r2.get_bucket_stats().get(\"total_size_mb\", 0):.0f} MB / 10,000 MB')
"

# Check local database counts
python3 -c "
import sqlite3
from pathlib import Path
conn = sqlite3.connect(str(Path.home() / '.trailcam/trailcam.db'))
c = conn.cursor()
for table in ['photos', 'tags', 'annotation_boxes', 'deer_metadata']:
    c.execute(f'SELECT COUNT(*) FROM {table}')
    print(f'{table}: {c.fetchone()[0]:,}')
"

# Check Supabase counts
python3 -c "
import json
from supabase_rest import SupabaseRestClient
with open('cloud_config.json') as f:
    cfg = json.load(f)['supabase']
client = SupabaseRestClient(cfg['url'], cfg['anon_key'])
for table in ['photos_sync', 'tags', 'annotation_boxes']:
    r = client.table(table).select('id').execute(fetch_all=True)
    print(f'{table}: {len(r.data):,}')
"

# Check for pending offline sync
cat ~/.trailcam/pending_sync.json
```

---

## Mobile App Notes

- **Location:** `/Users/brookebratcher/Desktop/trailcam_mobile/`
- **Config:** `lib/config/constants.dart` has Supabase URL and anon_key
- **Status:** In Google Play closed testing (needs 12 testers for 14 days)
- **Known limitation:** Reads from `annotation_boxes` for species but table is empty

---

## Notes for Next Session

1. Check `TASKS.md` before starting - register your task with 4-char ID
2. The duplicate cleanup and annotation_boxes sync are highest impact fixes
3. Don't forget to update this HANDOFF.md when you find/fix issues
4. Other Claudes may be working - coordinate via TASKS.md
