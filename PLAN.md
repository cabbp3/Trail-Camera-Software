# Trail Camera Software - Project Plan

**Last Updated:** January 13, 2026

---

## Product Vision

This software is intended to become a **marketable product** for hunters and wildlife enthusiasts. Not all features may ship (e.g., training pipelines stay internal), but user-facing features should be polished, intuitive, and reliable.

**Ultimate Goal:** Track bucks through trail camera photos.

Key capabilities:
1. **Identify deer** in photos (species detection) - Done
2. **Classify buck vs doe** (sex classification) - Done (needs improvement)
3. **Detect deer heads** for better analysis - In progress
4. **Count antler points** - Future
5. **Measure antler characteristics** - Future
6. **Recognize individual bucks** across photos - Partial (manual Deer ID system)
7. **Track bucks across seasons/years** - Done

---

## Current Status Summary

- **~7,500 photos** in the database
- **macOS** standalone build working
- **Windows** standalone build working (v1.0.5 on GitHub)
- **Android** mobile app in closed testing on Google Play
- **Supabase cloud sync** working via REST API with **auto-sync** (30-sec debounce, offline queueing)
- **Cloudflare R2** photo storage working
- **Species model:** 12 classes, 97.0% accuracy (v3.0)
- **Buck/doe model:** 84% accuracy (needs more Doe data)
- **GitHub:** https://github.com/cabbp3/Trail-Camera-Software

---

## Mobile App Status

**Trail Camera Organizer Mobile** - Flutter app for Android

| Task | Status |
|------|--------|
| Flutter project setup | Done |
| Mobile-friendly UI | Done |
| Connect to Supabase | Done |
| Connect to R2 (signed URLs) | Done |
| Photo viewer with filters | Done |
| Species/Sex/Deer ID editing | Done |
| Custom deer ID creation | Done |
| Buck profile creation | Done |
| Google Play closed testing | Done |
| Need 12 testers for 14 days | In Progress |
| iOS build | Future |

---

## Pending Tasks

### High Priority
- [ ] Fix buck/doe model bias (97% predicts Buck) - need more Doe training data
- [ ] Get Android app to production (need testers)
- [x] Add annotation_boxes to `pull_from_supabase()` (data loss risk) - Fixed Jan 18

### Medium Priority
- [ ] Build iOS app (Flutter makes this straightforward)
- [x] Fix mobile N+1 query patterns (Jan 13)
- [x] Remove 200 hash limit in mobile filters (Jan 13)
- [x] Add HTTP timeouts to mobile API calls (Jan 13)
- [x] Sync archive status to mobile

### Lower Priority
- [ ] **Split/merge deer IDs in app** - Allow splitting one deer ID into two (e.g., when a deer was misidentified across seasons). Needs UI to select which photos go to which ID. Options: by date range, by season, by manual selection. Complex UX to design.
- [ ] Batch editing by date range (for camera moves)
- [ ] Queue mode performance optimization
- [ ] Deer head detection training (need 500+ head boxes)
- [ ] Video support (MP4, AVI)
- [ ] Code signing certificates (removes security warnings)
- [ ] Auto-updater for desktop app

---

## Multi-Platform Distribution

| Platform | App Type | Status |
|----------|----------|--------|
| macOS | Standalone .app | Done |
| Windows | Standalone .exe | Done (v1.0.5) |
| Android | Flutter app | Closed Testing |
| iPhone | Flutter app | Future |
| Web | Browser app | Future |

---

## Key Technical Notes

### Database
- SQLite with WAL mode
- Schema: `photos`, `tags`, `deer_metadata`, `annotation_boxes`, `buck_profiles`
- Sync via Supabase REST API

### Photo Storage
- Cloudflare R2 bucket: `trailcam-photos`
- Structure: `photos/{file_hash}.jpg`, `thumbnails/{file_hash}_thumb.jpg`
- Free tier: 10GB storage, unlimited downloads

### Supabase Tables
- `photos_sync`, `tags`, `deer_metadata`, `deer_additional`
- `buck_profiles`, `buck_profile_seasons`, `annotation_boxes`

---

## Documentation Guide

| File | Purpose |
|------|---------|
| `TASKS.md` | **Active work handoff between Claudes** |
| `INDEX.md` | Quick topic index for all docs |
| `PLAN.md` | Product roadmap (this file) |
| `CLAUDE.md` | Developer context, commands |
| `TECHNICAL_AUDIT.md` | System audit, technical debt, scaling |
| `AI_REFINEMENT_PLAN.md` | AI model development roadmap |
| `README.md` | Public-facing project overview |

---

## Known Issues

### Timestamp Corrections Needed
**216 photos at WB 27 need date correction:**
- Currently set to: July 1-8, 2024
- Original (wrong) dates: January 1-8, 2022
- Camera had incorrect clock settings

```sql
SELECT * FROM photos WHERE date_taken LIKE '2024-07%' AND camera_location = 'WB 27';
```

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
- Models: `models/` folder
- Main UI code: `training/label_tool.py`

**Build mobile APK:**
```bash
cd /Users/brookebratcher/Desktop/trailcam_mobile
/Users/brookebratcher/develop/flutter/bin/flutter build apk --release
```

---

## Mobile App TODO

### Offline Mode
- Cache photos locally for viewing without internet
- Download photos on demand or in bulk
- Sync labels/changes when back online

### Upload Photos from Phone
- Allow taking photos with phone camera or selecting from gallery
- Upload directly to R2 with metadata
- Useful for photos not from trail cameras (scouting, food plots, etc.)

### Desktop Auto-Sync from Cloud
- Merge "Pull labels" and "Pull photos" into single auto-sync on app open
- Use timestamps to detect what changed (only sync new/modified records)
- Pull new photo records from Supabase that don't exist locally
- Download thumbnails from R2 for cloud-only photos
- Should be fast/silent if nothing changed
