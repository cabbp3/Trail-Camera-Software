# HANDOFF - System Audit (Jan 12, 2026)

## RECENT CHANGES (Jan 12, 2026 - Latest)

### Per-Box Buck/Doe Predictions (Jan 12, 2026)

**What Changed:**
- Buck/doe predictions now run per-box instead of per-photo
- Added `sex` and `sex_conf` columns to `annotation_boxes` table
- Box labels now show "Subject 1: Deer (Buck)" format
- Migrated 2,414 existing photo-level labels to per-box system

**Model Retrain:**
- Retrained buck/doe model (84% accuracy)
- Training data: 1,157 Buck / 341 Doe photos (3.4:1 imbalance)
- Ran predictions on all 7,575 deer boxes

**Issue Found:**
- Model heavily biased toward Buck (97.7% of predictions)
- Need more Doe training data to balance
- Consider: confidence threshold, head-only crops, or antler detection approach

**Database Changes:**
```sql
-- Added to annotation_boxes
ALTER TABLE annotation_boxes ADD COLUMN sex TEXT;
ALTER TABLE annotation_boxes ADD COLUMN sex_conf REAL;
```

---

## STRATEGIC PRIORITIES (Jan 12, 2026)

### What Matters Most

| Priority | Area | Why |
|----------|------|-----|
| 1 | **Fix Buck/Doe AI** | Core feature, currently broken (97% Buck bias) |
| 2 | **Get Android Live** | Need 12 testers for 14 days, then production |
| 3 | **iOS App** | Flutter makes this easy, unlocks half mobile market |
| 4 | **Deer Re-ID** | High value for tracking individual bucks |
| 5 | **Code Quality** | 11,000 line file is risky to maintain |

### Buck/Doe Model Options

1. **More Doe data** - Label more does, use LILA dataset
2. **Head-only crops** - May improve accuracy (antlers more visible)
3. **Antler detection** - Binary "has visible antlers?" might work better
4. **Confidence threshold** - Show "Unknown" for low-confidence predictions

---

## MODULAR ARCHITECTURE PLAN

### Current Problem
`training/label_tool.py` is 11,000+ lines in one file. Hard to:
- Find specific code
- Make safe changes
- Test individual components
- Have multiple developers

### Proposed Structure

```
training/
├── label_tool.py          (2,000 lines) - Main window, orchestration
├── ui/
│   ├── photo_browser.py   - Photo list, thumbnails, navigation
│   ├── photo_viewer.py    - Image display, box drawing, zoom
│   ├── info_panel.py      - Species/sex/deer ID controls
│   ├── filter_panel.py    - Filter/sort controls
│   └── dialogs/
│       ├── buck_profile.py
│       ├── compare_window.py
│       ├── cloud_sync.py
│       └── review_queue.py
├── services/
│   ├── ai_pipeline.py     - Run detection → species → buck/doe
│   ├── photo_service.py   - Load, crop, cache photos
│   └── sync_service.py    - Supabase/R2 operations
└── models/
    └── photo.py           - Data classes
```

### Migration Approach
1. Extract one module at a time (start with AI pipeline - most independent)
2. Keep imports working from main file
3. Test thoroughly after each extraction
4. Estimated: 2-3 focused sessions

### Benefits
- Easier to find code ("buck/doe logic? → ai_pipeline.py")
- Safer changes (editing dialogs won't break photo display)
- Testable (can test AI pipeline without launching UI)
- Multiple developers can work without conflicts

### Risks
- Breaking existing features (mitigate: incremental, test each step)
- PyQt signal/slot complexity (mitigate: keep connections in main window)

---

## DATABASE SCALING PLAN

### Key Insight: Users Only See Their Data

At commercial scale, any given user only accesses a small subset of the total database:
- Total database: 1,000,000+ photos across all users
- Single user's view: 5,000 - 20,000 photos (their hunting club)
- Queries are always scoped to user's collection(s)

**This means:**
- The "million photo query" problem doesn't exist
- RLS (Row Level Security) naturally filters to user's data
- Primary index should be on `collection_id` or `user_id`
- Each user's experience stays fast regardless of total DB size

### Current State
- ~7,500 photos, queries are fast
- SQLite with WAL mode and indexes
- Works well for single user

### Target Scale
- 100,000 - 1,000,000+ photos (total across all users)
- Any single user sees 5,000 - 20,000 photos max
- Buck photos are highest priority within user's view
- Empty/other species can be slower

### Archive Feature (Already Implemented)

**Purpose:** Hide photos users don't want to see again (empty, blurry, duplicates, uninteresting species).

**How it works:**
- `archived` column in photos table (0=active, 1=archived)
- Index: `idx_photos_archived`
- Default queries exclude archived at SQL level
- UI filter: All / Active (default) / Archived

**Query optimization:**
```sql
-- All queries filter archived by default
WHERE (p.archived IS NULL OR p.archived = 0)
```

**Scaling impact:**
- If user archives 80% of photos (empty, squirrels, etc.), queries only touch 20%
- Combined with collection filtering, a user with 20K photos might query only 2-4K active photos
- This is why complex optimizations aren't needed - the data naturally shrinks

**Model training:** Archived photos ARE included in training - archive status is ignored for AI model training so labeled data isn't lost.

**Recommended workflow for users:**
1. Import photos from trail camera
2. Run AI detection
3. Review and archive empty/uninteresting photos
4. Work with remaining photos (mostly deer/wildlife of interest)

### TODO: Sync Archive Status to Mobile

**Current state:** `archived` is local-only, not synced to Supabase. Mobile shows all photos.

**Planned changes:**

1. **Supabase schema** - Add archived column:
```sql
ALTER TABLE photos_sync ADD COLUMN archived BOOLEAN DEFAULT FALSE;
```

2. **Desktop sync** - Include archived in push (database.py ~line 1988):
```python
photos_data.append({
    ...
    "archived": bool(photo.get("archived")),  # Add this
})
```

3. **Mobile default** - Filter out archived photos:
```dart
// supabase_service.dart - add to fetchPhotos query
.eq('archived', false)  // Default: hide archived
```

4. **Mobile settings** - Add toggle to retrieve archived:
```dart
// settings_screen.dart
SwitchListTile(
  title: Text('Show Archived Photos'),
  subtitle: Text('Include photos marked as archived'),
  value: showArchived,
  onChanged: (val) => setState(() => showArchived = val),
)
```

**Benefit:** Archived photos don't sync to mobile by default (saves bandwidth/storage), but users can access them on demand via settings.

### Primary Optimization: Collection-Based Indexing

```sql
-- Most queries will filter by collection first
CREATE INDEX idx_photos_collection ON photos(collection);
CREATE INDEX idx_photos_collection_season ON photos(collection, season_year);
CREATE INDEX idx_annotation_boxes_collection ON annotation_boxes(photo_id, species);
```

With RLS in Supabase:
```sql
-- Users only see photos from their collections
CREATE POLICY "Users see their collections" ON photos_sync
FOR SELECT USING (
    collection IN (SELECT collection FROM user_collections WHERE user_id = auth.uid())
);
```

### Secondary Optimization: Denormalized Flags

Add columns directly to `photos` table:
```sql
ALTER TABLE photos ADD COLUMN is_buck BOOLEAN DEFAULT FALSE;
ALTER TABLE photos ADD COLUMN is_deer BOOLEAN DEFAULT FALSE;
ALTER TABLE photos ADD COLUMN priority INTEGER DEFAULT 0;
-- priority: 0=empty, 1=other species, 2=deer, 3=buck

CREATE INDEX idx_photos_is_buck ON photos(is_buck) WHERE is_buck = 1;
CREATE INDEX idx_photos_is_deer ON photos(is_deer) WHERE is_deer = 1;
CREATE INDEX idx_photos_priority ON photos(priority);
CREATE INDEX idx_photos_buck_season ON photos(is_buck, season_year) WHERE is_buck = 1;
```

**Why This Helps:**
- Avoids JOIN to tags table for common filters
- Partial indexes only store buck/deer rows (smaller, faster)
- Composite index for "all bucks from 2025" queries
- SQLite handles 1M+ rows easily with proper indexes

**Trigger to Keep in Sync:**
```sql
CREATE TRIGGER update_photo_flags AFTER INSERT ON tags
BEGIN
    UPDATE photos SET
        is_buck = CASE WHEN NEW.tag_name = 'Buck' THEN 1 ELSE is_buck END,
        is_deer = CASE WHEN NEW.tag_name IN ('Buck', 'Doe', 'Deer') THEN 1 ELSE is_deer END,
        priority = CASE
            WHEN NEW.tag_name = 'Buck' THEN 3
            WHEN NEW.tag_name IN ('Doe', 'Deer') THEN MAX(priority, 2)
            WHEN NEW.tag_name != 'Empty' THEN MAX(priority, 1)
            ELSE priority
        END
    WHERE id = NEW.photo_id;
END;
```

### What We Don't Need
- PostgreSQL (SQLite is fine for single-user desktop)
- Partitioning (SQLite doesn't support, not needed anyway)
- NoSQL (structured data, relational queries = SQL is right)
- Archive table (just use priority column)

### Pagination (Required at Scale)
All list queries must use LIMIT/OFFSET:
```python
# Bad - loads all 100K photos into memory
photos = db.get_all_photos()

# Good - loads 100 at a time
photos = db.get_photos(limit=100, offset=page * 100)
```

---

## RECENT CHANGES (Jan 11, 2026)

### Deep Review & Cleanup (Jan 11, 2026)

**R2 Storage Cleanup - COMPLETED**
- Deleted 14,594 orphaned files from old `users/brooke/` path structure
- Reclaimed 5.35 GB of storage
- R2 now at 5.22 GB (was 10.7 GB - over free tier limit!)
- Current structure: `photos/` (7,365 files) + `thumbnails/` (7,279 files)

**Data Counts (Updated Jan 11, 2026)**
| Component | Local | Supabase | Status |
|-----------|-------|----------|--------|
| photos | 7,280 | 7,279 | ✅ OK (±1) |
| annotation_boxes | 10,226 | 10,023 | ⚠️ Local +203 (push needed) |
| tags | 8,715 | 8,723 | ✅ OK |
| deer_metadata | 4,864 | 5,378 | ⚠️ Cloud +514 orphans |

**Collections**
| Collection | Photos |
|------------|--------|
| Brooke Farm | 4,677 |
| East Side Hunting Club | 1,405 |
| Tightwad House | 1,198 |

---

## RECENT CHANGES (Jan 10, 2026)

### Mobile App - Photo Swipe Navigation
- Swipe left/right in photo viewer to navigate between photos
- Shows "X of Y" counter in title bar
- Uses PageView widget for smooth transitions

### Mobile App - Photo Download to Gallery
- Tap download button to save photo to device gallery
- Creates "Trail Camera" album
- Shows progress spinner while downloading
- Handles Android permissions (READ_MEDIA_IMAGES for Android 13+)

### ⚠️ GOOGLE PLAY DATA SAFETY UPDATE REQUIRED
**Before releasing to production, update Google Play Console Data Safety declarations:**

1. **Data Sharing** - Must disclose:
   - Photos are viewable/downloadable by other users in the same collection/club
   - Photo EXIF data (location, date, camera) is shared with club members

2. **New Permissions** - App now uses:
   - `READ_MEDIA_IMAGES` (Android 13+)
   - `WRITE_EXTERNAL_STORAGE` (Android 10 and below)

3. **Privacy Policy** - Should mention:
   - Photos uploaded are visible to collection/club members
   - Members can download each other's photos to their device

### Fixed Species List (Admin-Only)
- Species list is now fixed and alphabetized in SPECIES_OPTIONS
- 25 species: Armadillo, Bobcat, Chipmunk, Coyote, Deer, Dog, Empty, Flicker, Fox, Ground Hog, House Cat, Opossum, Other, Other Bird, Otter, Person, Quail, Rabbit, Raccoon, Skunk, Squirrel, Turkey, Turkey Buzzard, Unknown, Vehicle
- Custom species addition disabled - "Other (+)" now just applies "Other" tag
- VALID_SPECIES synced in both label_tool.py and ai_suggester.py

### Contextual Filter Dropdowns
- Filter dropdowns now show only options that exist in the current filter context
- Example: When filtering by "East Side Hunting Club", species dropdown only shows species found at ESHC
- Affects: Species, Sex, Deer ID, Site, Year, Collection filters
- Added `_get_context_filtered_photos(exclude_filter)` helper method
- All filter options refresh when any filter changes

---

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

**P0 - Complete R2 Migration:** ✅ FULLY COMPLETED (Jan 11, 2026)
- [x] Migrated 7,365 photos to `photos/`
- [x] Migrated 7,279 thumbnails to `thumbnails/`
- [x] Deleted 14,594 orphaned files from old `users/brooke/` paths
- [x] R2 storage reduced from 10.7 GB to 5.22 GB (reclaimed 5.35 GB)

**P1 - Fix Mobile Filters:**
- [x] Remove 200 hash limit (supabase_service.dart:77) ✅ DONE (Jan 13) - uses chunked queries
- [x] Fix deer ID N+1 loop (supabase_service.dart:406-447) ✅ DONE (Jan 13) - batch query with in.()
- [x] Add end date to date range filter ✅ DONE (Jan 10, 2026)
- [x] Fix species filter to query BOTH tags AND annotation_boxes ✅ DONE

**P2 - Mobile Performance:**
- [x] Fix unbounded URL cache memory leak (photo_service.dart:15-16) ✅ DONE (Jan 13) - max 1000 entries with LRU eviction
- [x] Add HTTP timeouts to all API calls ✅ DONE (Jan 13) - 30 second timeout on all HTTP calls
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

## Data Consistency Summary (Updated Jan 11, 2026)

| Component | Local | Supabase | Status |
|-----------|-------|----------|--------|
| photos | 7,280 | 7,279 | ✅ OK (±1) |
| annotation_boxes | 10,226 | 10,023 | ⚠️ Local +203 (push needed) |
| tags | 8,715 | 8,723 | ✅ OK (orphans cleaned) |
| deer_metadata | 4,864 | 5,378 | ⚠️ Cloud +514 orphans |

**R2 Storage (Updated Jan 11, 2026)**
| Path | Files | Size |
|------|-------|------|
| photos/ | 7,365 | 5.15 GB |
| thumbnails/ | 7,279 | 77 MB |
| **Total** | **14,644** | **5.22 GB** |

Free tier limit: 10 GB ✅

---

## Code Review Findings (Jan 11, 2026)

A comprehensive code review identified the following issues across desktop and mobile apps.

### CRITICAL (Fix Soon)

| Issue | App | Location | Description |
|-------|-----|----------|-------------|
| Missing annotation_boxes pull | Desktop | `database.py:2163-2288` | `push_to_supabase()` syncs boxes but `pull_from_supabase()` never pulls them back. One-way sync = data loss between computers. |
| Silent sync failures | Desktop | `database.py:1940-1945` | `batch_upsert()` doesn't check if Supabase returned an error. User sees "Pushed: 0 photos" and thinks it worked. |
| Exposed API key | Mobile | `constants.dart:1-5` | Supabase anon key hardcoded. Anyone can extract from APK. Mitigated by RLS but should rotate if repo goes public. |

### HIGH PRIORITY

| Issue | App | Location | Description |
|-------|-----|----------|-------------|
| Photo key pipe bug | Desktop | `database.py:1862-1874` | Photo keys use `\|` delimiter. If filename/camera contains `\|`, sync silently breaks. |
| ~~N+1 query pattern~~ | Mobile | `supabase_service.dart` | ~~Fetching photos makes 4-7 separate API calls~~ **FIXED Jan 13** - deer ID uses batch query |
| Silent error handling | Mobile | `supabase_service.dart` (15+ places) | Errors caught and ignored. User has no idea why data isn't loading. |
| ~~No HTTP timeouts~~ | Mobile | `supabase_service.dart` | ~~All API calls have no timeout~~ **FIXED Jan 13** - 30s timeout on all calls |
| ~~Sex filter in-memory bug~~ | Mobile | `supabase_service.dart:156-162` | ~~Sex filter applied after pagination, not at query level~~ **FIXED Jan 13, 2026** |

### MEDIUM PRIORITY

| Issue | App | Location | Description |
|-------|-----|----------|-------------|
| R2 check-exists overhead | Desktop | `label_tool.py:11792-11806` | Every upload does HEAD before PUT. 100 photos = 200 API calls. |
| COALESCE ignores deletions | Desktop | `database.py:2191-2193` | Pull uses `COALESCE(cloud, local)`. Deleting notes on Computer A won't clear on Computer B. |
| Hardcoded username | Mobile | `photo_grid_screen.dart:140,177` | R2 paths hardcoded to 'brooke'. Won't work for other users. |
| No deer ID validation | Mobile | `photo_view_screen.dart:314-350` | Users can enter anything as Buck ID. No length/char limits. |
| Demo mode silent fallback | Mobile | `photo_grid_screen.dart` | If no photos found, shows demo data without warning. |

### LOW PRIORITY

| Issue | App | Location | Description |
|-------|-----|----------|-------------|
| Unused dependencies | Mobile | `pubspec.yaml` | `crypto` package imported but never used. |
| Unused LoginScreen | Mobile | `login_screen.dart` | File exists but is never used in navigation. |
| N+1 in add_many_to_claude_review | Desktop | `database.py:2307-2317` | Individual queries per photo instead of batch. |

### Top 3 Fixes Recommended

1. **Add annotation_boxes to `pull_from_supabase()`** - Actively causing data loss
2. **Add error checking after `batch_upsert()`** - Check `response.error` and raise exception
3. **Add logging throughout mobile app** - Replace silent `catch(e) {}` with proper error logging

---

## Deep Review Findings (Jan 11, 2026)

### Security Issues

**1. Supabase RLS Policies Are Wide Open**
```sql
-- supabase_setup.sql - ALL tables have this:
CREATE POLICY "Allow all access" ON photos_sync FOR ALL USING (true);
```
- Anyone with the anon key can read/write/delete ANY data
- Acceptable for personal/family use, NOT for production multi-user
- **Action:** No change needed for current use case

**2. Debug Scripts with API Keys (CLEANUP NEEDED)**
Three Python scripts in mobile folder contain hardcoded Supabase keys:
- `/Users/brookebratcher/Desktop/trailcam_mobile/detailed_analysis.py`
- `/Users/brookebratcher/Desktop/trailcam_mobile/check_photo_9731.py`
- `/Users/brookebratcher/Desktop/trailcam_mobile/diagnostic_check.py`
- **Action:** Delete these files before any git commits

### Technical Debt

**3. Python 3.9 Deprecation**
- Boto3 will drop Python 3.9 support on April 29, 2026
- **Action:** Upgrade to Python 3.10+ before April 2026

**4. Mobile App - 200 Hash Limit (supabase_service.dart:80)**
```dart
final hashList = speciesFilterHashes.take(200).toList();
```
- Species/deer ID filters truncate results beyond 200 photos
- **Action:** Remove limit or implement server-side filtering

**5. Mobile App - Unbounded URL Cache (photo_service.dart:15)**
```dart
final Map<String, _CachedUrl> _urlCache = {};
```
- No memory limit, potential leak on long sessions
- Has expiration but no max size
- **Action:** Add LRU eviction or max cache size

**6. Desktop - set_boxes() Not in Transaction (database.py:1202-1227)**
- DELETE then INSERT without transaction wrapper
- Crash between operations = data loss
- **Action:** Wrap in BEGIN...COMMIT

### What's Working Well

- ✅ Database indexes - comprehensive coverage
- ✅ Sync logic - no race conditions, proper batching
- ✅ Credential storage - cloud_config.json in .gitignore
- ✅ Android keystore - key.properties in .gitignore
- ✅ Dependencies - no critical vulnerabilities

---

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

## Real Issues (Priority Order) - Updated Jan 11, 2026

### ✅ COMPLETED
- [x] 995 Orphaned Tags - CLEANED (Jan 10)
- [x] 983 Duplicate Tags - CLEANED (none found after orphan cleanup)
- [x] macOS Resource Fork - CLEANED (._STC_1167.JPG)
- [x] R2 Orphaned Files - CLEANED (14,594 files, 5.35 GB reclaimed)

### Remaining Issues

**1. deer_metadata Discrepancy (514 orphans in cloud)**
- Cloud has 5,378 records, Local has 4,864
- 514 orphaned records that don't match local photos
- **Action:** Run cleanup query or investigate source

**2. annotation_boxes Not Synced**
- Local has 10,226, Cloud has 10,023
- 203 new boxes haven't been pushed
- **Action:** Run a push from desktop app

**3. last_pull_at is NULL**
- Pull has never been run (push-only sync so far)
- **Action:** Optional - only needed if editing from multiple machines

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

**Note:** custom_species typos (Ott, Turke) were cleaned - table now has 16 valid species.

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
2. ~~**Deer ID filter N+1** - Makes separate HTTP request per deer ID~~ ✅ FIXED (Jan 13)
3. ~~**200 hash limit** - Species/deer filters truncate silently~~ ✅ FIXED (Jan 13)

### Security (Low Priority)
- API keys in source
- Hardcoded username "brooke"
- LoginScreen bypassed

---

## Recommended Next Steps (Updated Jan 11, 2026)

### ✅ Completed
- [x] R2 Migration - All photos/thumbnails in new structure
- [x] R2 Cleanup - Orphaned files deleted, 5.35 GB reclaimed
- [x] Supabase Cleanup - Orphaned tags, duplicates, resource fork deleted
- [x] Collection rename - "Hunting Club" → "East Side Hunting Club"
- [x] Date range filter - Now uses both start AND end date

### P1 - Mobile App Fixes
1. ~~Remove 200 hash limit (supabase_service.dart:80)~~ ✅ FIXED Jan 13
2. ~~Add URL cache size limit (photo_service.dart:15)~~ ✅ FIXED Jan 13
3. Delete debug scripts (3 Python files with API keys)

### P2 - Data Sync
1. Push 203 new annotation_boxes to Supabase
2. Clean 514 orphaned deer_metadata in cloud
3. Run a pull to sync cloud→local (optional)

### P3 - Code Quality
1. Wrap set_boxes() in transaction (database.py:1202)
2. Upgrade to Python 3.10+ before April 2026

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
| ~~N+1 queries~~ | ~~High~~ | ~~4+ requests per photo load~~ **FIXED Jan 13** |
| ~~200 hash limit~~ | ~~High~~ | ~~Silent data truncation~~ **FIXED Jan 13** |
| ~~No HTTP timeouts~~ | ~~High~~ | ~~App hangs on slow networks~~ **FIXED Jan 13** |
| ~~URL cache memory leak~~ | ~~Medium~~ | ~~Unbounded cache~~ **FIXED Jan 13** |
| No offline support | Medium | Can't view photos without network |
| No state management | Medium | Memory leaks, won't scale with features |
| 0% test coverage | Medium | Regressions undetected |
| Silent error handling | Medium | Users don't know when things fail |

**Key file references (FIXED Jan 13):**
- ~~`supabase_service.dart:406-447` - Deer ID N+1 loop~~ - Now uses batch query
- ~~`supabase_service.dart:77` - 200 hash limit~~ - Now uses chunked queries
- ~~`photo_service.dart:15-16` - Unbounded URL cache~~ - Now has 1000 entry limit with LRU eviction

---

### COST PROJECTIONS (Updated Jan 11, 2026)

**R2 Storage (photos):**
- Current: 5.22 GB (FREE - well under 10 GB limit)
- After cleanup: Reclaimed 5.35 GB by deleting orphaned files
- Break-even: ~4+ years at current growth rate
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
5. ~~Fix mobile N+1 queries (deer ID filter loop)~~ ✅ FIXED Jan 13
6. ~~Remove 200 hash limit in mobile filters~~ ✅ FIXED Jan 13

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
| Mobile filters | Works | Chunked queries handle large sets | **FIXED Jan 13** |
| Full sync | ~2 min | ~4+ hours | Too slow |
| AI processing | ~4 hours | ~400 hours (17 days) | Needs GPU |

**Must-Fix for 800K:**

Desktop App:
- [ ] Paginate `get_all_photos()` - load 100 at a time, not all (organizer_ui.py:819)
- [ ] Virtual scrolling in photo list (only render visible items)
- [ ] Lazy loading of thumbnails
- [ ] Add database indexes on all filter columns

Mobile App:
- [x] Remove 200 hash limit (supabase_service.dart:77) ✅ FIXED Jan 13
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
