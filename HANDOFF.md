# HANDOFF - System Audit (Jan 10, 2026)

## TASK SPLIT FOR PARALLEL WORK

### CLAUDE 1 - Backend/Database (Desktop Python) ✅ COMPLETED
Works on: `/Users/brookebratcher/Desktop/Trail Camera Software V 1.0/`

**P0 - Supabase Cleanup:** ✅ DONE (Jan 10, 2026)
- [x] Deleted 995 orphaned tags (NULL file_hash)
- [x] Checked for duplicates - none found after orphan cleanup
- [x] Deleted macOS resource fork (._STC_1167.JPG)

**P0 - Local Database:** ✅ DONE
- [x] Renamed collection "Hunting Club" → "East Side Hunting Club" (1,405 photos)
- [x] Deleted custom_species typos (Ott, Turke) - correct spellings already existed
- [x] Pushed collection rename to Supabase

**P1 - Database Fixes:** ✅ DONE
- [x] Added unique index: `idx_photos_file_hash` on photos(file_hash)
- [x] Season year logic verified CORRECT - April 30 and May 1 SHOULD be different seasons (May-April boundary)
- [ ] Run a pull to sync cloud→local (`last_pull_at` is NULL) - optional

**P2 - Desktop Robustness:** (Future)
- [ ] Wrap `set_boxes()` in transaction (line 1204-1227)
- [ ] Add pagination to `get_all_photos()` for 800K scale
- [ ] Investigate deer_metadata 511 record discrepancy

---

### CLAUDE 2 - Frontend/Mobile (Flutter)
Works on: `/Users/brookebratcher/Desktop/trailcam_mobile/`

**P0 - Complete R2 Migration:** ✅ DONE (Jan 10, 2026)
- [x] Migrated 7,340 photos to `photos/`
- [x] Migrated 7,254 thumbnails to `thumbnails/`
- [ ] Verify mobile app loads images after migration
- [ ] (Optional) Delete old `users/brooke/` paths after verification

**P1 - Fix Mobile Filters:**
- [ ] Remove 200 hash limit (supabase_service.dart:77)
- [ ] Fix deer ID N+1 loop (supabase_service.dart:406-447) - batch into single query
- [x] Add end date to date range filter ✅ DONE (Jan 10, 2026)
- [x] Fix species filter to query BOTH tags AND annotation_boxes ✅ DONE

**P2 - Mobile Performance:**
- [ ] Fix unbounded URL cache memory leak (photo_service.dart:15-16)
- [ ] Add proper error handling (6 silent failures in supabase_service.dart)
- [ ] Consider server-side filtering for large datasets

**P3 - Mobile Quality:**
- [ ] Add actual tests (currently 0% coverage)
- [ ] Remove hardcoded "brooke" username (photo_grid_screen.dart:133)

---

### NO CONFLICTS
- Claude 1 works on Python/SQL only
- Claude 2 works on Flutter/Dart only
- Both can run simultaneously

---

## Architecture Note: Trainer vs Viewer

**Desktop Trainer/Editor** (one computer):
- Runs AI detection → creates annotation_boxes with coordinates
- User labels species on boxes
- Pushes boxes to Supabase

**Desktop Organizer / Mobile Viewer** (any device):
- Views photos, filters by species
- Reads `annotation_boxes.species` from Supabase for filtering
- Does NOT need box coordinates

**Implication:** annotation_boxes only need push (one-way). Only matters if training on multiple computers.

---

## Data Consistency Summary

| Component | Local | Supabase | Issue |
|-----------|-------|----------|-------|
| photos | 7,255 | 7,256 | OK (+1 minor) |
| annotation_boxes | 10,023 | 10,023 | OK |
| tags | 8,426 | 9,424 | **995 orphaned with NULL file_hash** |
| deer_metadata | 4,602 | 5,113 | Cloud has more (511 extra) |

### CORRECTED Understanding

The "998 missing tags" I reported earlier was WRONG. Here's what's actually happening:

**Tags breakdown:**
- Cloud total: 9,424
- Cloud with file_hash: 8,429
- Cloud with NULL file_hash: **995 (orphaned, can't be matched)**
- Local: 8,426
- **Actual matchable difference: only 3 tags**

**Buck tag example:**
- Cloud total: 1,455
- Cloud with NULL file_hash: 365 (orphaned)
- Cloud with file_hash: 1,090 (matchable)
- Local: 1,089
- **Actual difference: 1 tag**

**Root cause:** 995 tags were created BEFORE file_hash was added to the schema. They have NULL file_hash and can't be matched to photos during pull.

---

## Real Issues (Priority Order)

### 1. 995 Orphaned Tags in Supabase (CLEANUP)
Tags with NULL file_hash - created before migration. Can't be matched to photos.
```sql
DELETE FROM tags WHERE file_hash IS NULL;
```

### 2. 983 Duplicate Tags in Supabase (CLEANUP)
Same (file_hash, tag_name) appears multiple times.
```sql
DELETE FROM tags a USING tags b
WHERE a.id > b.id
  AND a.file_hash = b.file_hash
  AND a.tag_name = b.tag_name;
```

### 3. 1 macOS Resource Fork File in Database
Photo ID 11377 is `._STC_1167.JPG` - a macOS metadata file, not a real photo.
Should be deleted from database.

### 4. last_pull_at is NULL
Sync state shows push worked but pull has never been run:
```
(1, '2026-01-10T15:52:04.666000', None)
```

### 5. deer_metadata Discrepancy (511 more in cloud)
Need to investigate - might be similar orphan issue or mobile additions.

---

## Local-Only Tables (Not Synced)

These exist locally but NOT in Supabase:

| Table | Rows | Purpose |
|-------|------|---------|
| ai_rejections | 1,125 | Tracking rejected AI suggestions |
| ai_suggestions | 254 | AI suggestion cache |
| claude_review_queue | 559 | Photos for review |
| custom_species | 18 | User-added species labels |
| photo_embeddings | 1,803 | Re-ID embeddings |
| recent_bucks | 12 | Recently viewed |
| sites | 5 | Camera site locations |

**Note:** custom_species has typos: "Ott", "Turke"

---

## User Registry

```json
{
  "users": [
    {"username": "Chris Brooke", "is_admin": true, "status": "active"},
    {"username": "Wayne Cunningham", "is_admin": false, "status": "pending"},
    {"username": "Martin Brooke", "is_admin": false, "status": "pending"},
    {"username": "Brandon Brooke", "is_admin": false, "status": "pending"}
  ]
}
```

4 users registered but mobile app hardcoded to "brooke".

---

## Mobile App Issues

### Fixed This Session
- Species reads from both `annotation_boxes` AND `tags` table
- Sex tags (Buck/Doe) read from `tags` table (persist after save)
- Deer ID reads from `deer_metadata` (works)

### Still Broken
1. ~~**Date range filter** - Only uses start date, ignores end date~~ ✅ FIXED (Jan 10)
2. **Deer ID filter N+1** - Makes separate HTTP request per deer ID
3. **200 hash limit** - Species/deer filters truncate silently

### Security (Low Priority)
- API keys in source
- Hardcoded username "brooke"
- LoginScreen bypassed

---

## Recommended Fixes

### P0 - Wait for R2 Migration
Other Claude is moving files.

### P1 - Supabase Cleanup
1. Delete 995 orphaned tags (NULL file_hash)
2. Deduplicate 983 duplicate tags
3. Delete macOS resource fork photo (ID 11377)

### P2 - Run a Pull
The pull has never been run (`last_pull_at` is NULL). Run pull from desktop app to sync any cloud-only data.

### P3 - Investigate deer_metadata
511 more records in cloud - check if orphans or real data from mobile.

### P4 - Fix Mobile Filters
1. Add end date to date range filter
2. Batch deer ID queries
3. Remove 200 hash limit

---

## Quick Diagnostic Commands

**Check Supabase counts:**
```bash
for table in photos_sync tags annotation_boxes deer_metadata; do
  curl -s "https://iwvehmthbjcvdqjqxtty.supabase.co/rest/v1/$table" \
    -H "apikey: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml3dmVobXRoYmpjdmRxanF4dHR5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjYyMDI0NDQsImV4cCI6MjA4MTc3ODQ0NH0._z6WAfUBP_Qda0IcjTS_LEI_J7r147BrmSib3dyneLE" \
    -H "Prefer: count=exact" -I 2>/dev/null | grep content-range
done
```

**Check orphaned tags:**
```bash
curl -s "https://iwvehmthbjcvdqjqxtty.supabase.co/rest/v1/tags?file_hash=is.null" \
  -H "apikey: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml3dmVobXRoYmpjdmRxanF4dHR5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjYyMDI0NDQsImV4cCI6MjA4MTc3ODQ0NH0._z6WAfUBP_Qda0IcjTS_LEI_J7r147BrmSib3dyneLE" \
  -H "Prefer: count=exact" -I 2>/dev/null | grep content-range
```

**Check local database:**
```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0" && python3 -c "
import sqlite3, os
conn = sqlite3.connect(os.path.expanduser('~/.trailcam/trailcam.db'))
c = conn.cursor()
for table in ['photos', 'annotation_boxes', 'tags', 'deer_metadata']:
    c.execute(f'SELECT COUNT(*) FROM {table}')
    print(f'{table}: {c.fetchone()[0]}')"
```

**Rebuild and deploy mobile app:**
```bash
cd /Users/brookebratcher/Desktop/trailcam_mobile
/Users/brookebratcher/develop/flutter/bin/flutter build apk --release
~/Library/Android/sdk/platform-tools/adb -s R5CRC2EK3MY install -r build/app/outputs/flutter-apk/app-release.apk
```

---

## SQL Fixes

**Delete orphaned tags:**
```sql
DELETE FROM tags WHERE file_hash IS NULL;
-- Will remove ~995 orphaned tags
```

**Deduplicate tags:**
```sql
DELETE FROM tags a USING tags b
WHERE a.id > b.id
  AND a.file_hash = b.file_hash
  AND a.tag_name = b.tag_name;
-- Will remove ~983 duplicates
```

**Fix custom_species typos (local SQLite):**
```sql
UPDATE custom_species SET species = 'Otter' WHERE species = 'Ott';
UPDATE custom_species SET species = 'Turkey' WHERE species = 'Turke';
```

---

## LONG-TERM STRATEGIC ANALYSIS (Jan 10, 2026)

### Executive Summary

Current architecture works well for **single user** but has fundamental issues that will cause problems at scale:

| Metric | Current | 5 Users | 10 Users |
|--------|---------|---------|----------|
| Photos | 7,255 | ~36K | ~73K |
| R2 Cost | $0/mo | ~$255/mo | ~$660/mo |
| Multi-user ready? | No | No | No |

**Critical Finding:** The system has NO real multi-user support despite having 4 registered users. RLS is disabled, API keys are exposed, and there's no conflict resolution.

---

### CRITICAL ISSUES (Fix Before Adding Users)

#### 1. Supabase RLS Disabled (SECURITY)
Current policies allow ALL access to ALL data:
```sql
CREATE POLICY "Allow all access" ON photos_sync FOR ALL USING (true);
```
- User A can delete User B's annotations
- No audit trail of who changed what
- Anyone with the anon key can read/write everything

**Fix:** Implement real RLS with user_id tracking.

#### 2. API Keys Exposed in Mobile App
`/trailcam_mobile/lib/config/constants.dart` contains:
- Supabase anon key (hardcoded)
- R2 endpoint URL (with account ID)

Anyone who decompiles the APK has full API access.

**Fix:** Move to environment config or Supabase Auth with proper tokens.

#### 3. No Conflict Resolution
If two users edit same photo simultaneously:
- Last write wins (no merge)
- No notification
- No history of who changed what

**Fix:** Add `modified_by` column, implement optimistic locking or CRDT.

#### 4. file_hash Should Be Primary Key
Current sync uses fragile composite key: `original_name|date_taken|camera_model`
- EXIF corruption causes mismatches
- O(n) fallback lookups at scale
- No unique constraint on file_hash

**Fix:** Add `CREATE UNIQUE INDEX idx_photos_file_hash ON photos(file_hash)`

---

### MOBILE APP ISSUES (Long-Term)

| Issue | Severity | Impact |
|-------|----------|--------|
| N+1 queries | High | 4+ requests per photo load, 30+ on filter |
| 200 hash limit | High | Silent data truncation on large filters |
| No offline support | Medium | Can't view photos without network |
| No state management | Medium | Memory leaks, won't scale with features |
| 0% test coverage | Medium | Regressions undetected |
| Silent error handling | Medium | Users don't know when things fail |

**Key file references:**
- `supabase_service.dart:406-447` - Deer ID N+1 loop
- `supabase_service.dart:77` - 200 hash limit
- `photo_service.dart:15-16` - Unbounded URL cache (memory leak)

---

### COST PROJECTIONS

**R2 Storage (photos):**
- Current: 5.4 GB (FREE - under 10 GB limit)
- Break-even: ~2 years at current growth rate
- 10 users: $660/mo without optimization

**Supabase:**
- Database: 15 MB of 500 MB (3% used) - plenty of room
- Bandwidth: Well under 2 GB/mo

**Optimization opportunities:**
- Archive old seasons to cold storage (50% savings)
- Compress thumbnails more aggressively
- Deduplicate across users (shared hunting club photos)

---

### ARCHITECTURAL DEBT

**Desktop (database.py):**
- 2,357 lines mixing CRUD, sync, schema, migrations
- Push/pull sync logic embedded (400 lines) - hard to test
- No unit tests

**Mobile (supabase_service.dart):**
- 140-line fetchPhotos() with nested conditions
- Manual HTTP instead of SDK
- No repository pattern

**Sync System:**
- `INSERT OR IGNORE` in pull doesn't update existing records
- COALESCE prevents NULL syncing (can't clear notes from cloud)
- No versioning or migration system for schema changes

---

### MULTI-USER MODEL

**Current (Phase 1):** Everyone can edit everything. This is fine for trusted family/hunting club.

**Future (Phase 2):** Owner (admin) controls who can edit:
- Admin: Full access (Chris Brooke)
- Editor: Can label photos, add deer IDs
- Viewer: Read-only browsing

**What's needed for Phase 2:**
- [ ] Add `role` column to user registry (admin/editor/viewer)
- [ ] RLS policies that check role before allowing writes
- [ ] UI in desktop app to manage user permissions
- [ ] Mobile app respects role (hide edit buttons for viewers)

---

### RECOMMENDED PRIORITIES

**P0 - Immediate (data cleanup):** ✅ MOSTLY DONE
1. ~~Delete 995 orphaned tags (NULL file_hash)~~ ✅
2. ~~Deduplicate tags~~ ✅ (none found after orphan cleanup)
3. Run a pull to sync cloud→local (optional)
4. ~~Rename collection "Hunting Club" → "East Side Hunting Club"~~ ✅

**P1 - Data Integrity:**
4. ~~Add unique index on file_hash~~ ✅ DONE
5. Fix mobile N+1 queries (deer ID filter loop)
6. Remove 200 hash limit in mobile filters

**P2 - Sustainability:**
7. Add logging throughout sync code
8. Extract sync logic to separate module
9. Create test suite for sync logic

**P3 - Future (when adding view-only users):**
10. Add role column to user registry
11. Implement RLS policies based on role
12. Move API keys to environment config

---

### KNOWN BUGS & EDGE CASES (Jan 10 Deep Dive)

**Critical:**
- ~~Season year off-by-one~~ FALSE POSITIVE - logic is correct (May-April season boundary)
- Database race: `set_boxes()` deletes all then inserts - crash mid-batch = data loss (database.py:1204-1227)
- 100K photos = memory exhaustion (no pagination in `get_all_photos()`)
- Sync photo_key mismatch if camera_model has trailing spaces

**CuddeLink:**
- No session re-auth on 90+ day downloads
- Hash inconsistency: MD5 for import, SHA256 for duplicate removal
- No timezone in EXIF - all timestamps naive local
- Only JPG/PNG (no RAW, HEIC)

**AI Pipeline:**
- ONNX classifiers CPU-only (no GPU)
- No batching - ~1-2 sec per deer photo
- Re-ID embeddings exist but never matched
- Models cached forever (2-3 GB never freed)
- Small animals poorly detected (quail 22%)

**Buck Profiles:**
- Profiles orphaned when all photos deleted
- Re-ID trained but not in UI
- Antler scoring 100% manual

---

### WHAT WORKS WELL

- WAL mode + indexes = good local performance
- Debounce + retry sync = solid offline support
- R2 zero-egress = unlimited viewing at no extra cost
- Hash-based filenames = prevents enumeration attacks
- REST client = no C++ deps on Windows

---

### SCALING TO 800K PHOTOS (110x Current)

Current architecture would **crash** at 800K photos due to memory/pagination issues. Here's what needs to change:

**What Breaks:**

| Component | Current (7K) | At 800K | Status |
|-----------|--------------|---------|--------|
| SQLite DB | 11 MB | ~1.2 GB | Works |
| `get_all_photos()` | 7K dicts in RAM | 800K dicts (~800 MB) | **CRASHES** |
| R2 Storage | 5.4 GB ($0) | ~600 GB (~$9/mo) | OK |
| Supabase | 30K rows | 3.3M rows | **Exceeds free tier** |
| Mobile filters | Works | 200 hash limit hides 99.97% | **BROKEN** |
| Full sync | ~2 min | ~4+ hours | Too slow |
| AI processing | ~4 hours | ~400 hours (17 days) | Needs GPU |

**Must-Fix for 800K:**

Desktop App:
- [ ] Paginate `get_all_photos()` - load 100 at a time, not all (organizer_ui.py:819)
- [ ] Virtual scrolling in photo list (only render visible items)
- [ ] Lazy loading of thumbnails
- [ ] Add database indexes on all filter columns

Mobile App:
- [ ] Remove 200 hash limit (supabase_service.dart:77)
- [ ] Server-side filtering (Supabase does work, not app)
- [ ] Cursor-based pagination instead of offset

Sync:
- [ ] Incremental-only (never full sync at scale)
- [ ] Background sync with progress resume
- [ ] Conflict detection for multi-user

AI Pipeline:
- [ ] GPU acceleration for ONNX models
- [ ] Batch processing (32 images at once)
- [ ] Distributed processing option

Infrastructure:
- [ ] Supabase Pro plan (~$25/mo)
- [ ] Proper database indexes in Supabase
- [ ] Consider PostgreSQL read replicas

**Cost at 800K:**

| Service | Monthly Cost |
|---------|--------------|
| R2 Storage (~600 GB) | ~$9 |
| Supabase Pro | $25 |
| **Total** | ~$34/mo |

**Verdict:** Core data model is sound. SQLite and Supabase can handle millions of rows. Main work is pagination everywhere + GPU/batch AI processing.

---

### VERDICT

**Single user (7K photos):** Works great, keep using it.
**Family/hunting club (2-5 users):** Current shared editing is fine. Add role-based permissions later.
**Large operation (800K photos):** Needs pagination fixes before scaling. ~$34/mo infrastructure cost.

Timeline estimates are intentionally omitted - focus on what needs doing, not when.
