# Documentation Index

Quick reference to find information across all documentation files.

---

## File Overview

| File | Purpose | When to Read |
|------|---------|--------------|
| `TASKS.md` | **Active work handoff between Claudes** | **First - see what's in progress** |
| `PLAN.md` | Product roadmap, pending tasks | Planning what to work on |
| `CLAUDE.md` | Developer commands, architecture | Starting a coding session |
| `TECHNICAL_AUDIT.md` | System audit, tech debt, bugs | Debugging, scaling, cleanup |
| `AI_REFINEMENT_PLAN.md` | AI model pipeline | Working on ML models |
| `README.md` | Public project overview | New users, GitHub visitors |

---

## Topic Index

### Apps & Platforms
- Desktop app (Python/PyQt6): `CLAUDE.md` → "Project Overview"
- Mobile app (Flutter): `TECHNICAL_AUDIT.md` → "Mobile App Issues"
- Multi-platform status: `PLAN.md` → "Multi-Platform Distribution"
- Build commands: `CLAUDE.md` → "Commands"

### Cloud & Sync
- Supabase setup: `CLAUDE.md` → "Supabase"
- R2 storage setup: `CLAUDE.md` → "Cloudflare R2 Cloud Storage"
- Sync architecture: `TECHNICAL_AUDIT.md` → "Architecture Note"
- Data consistency: `TECHNICAL_AUDIT.md` → "Data Consistency Summary"

### Database
- Schema & tables: `CLAUDE.md` → "Database Schema Additions"
- Scaling plan: `TECHNICAL_AUDIT.md` → "Database Scaling Plan"
- Quick SQL checks: `TECHNICAL_AUDIT.md` → "Quick Diagnostic Commands"

### AI & Models
- Pipeline overview: `AI_REFINEMENT_PLAN.md` → "AI Pipeline Order of Operations"
- Species classifier: `AI_REFINEMENT_PLAN.md` → "Stage 2"
- Buck/doe classifier: `AI_REFINEMENT_PLAN.md` → "Stage 4"
- Deer head detection: `AI_REFINEMENT_PLAN.md` → "Stage 3"
- **Site identification**: `CLAUDE.md` → "Site Identification - Hybrid OCR + Visual Approach"
- Augmentation experiment: `AI_REFINEMENT_PLAN.md` → "Augmentation Experiment"
- Future AI ideas: `AI_REFINEMENT_PLAN.md` → "Future AI Ideas"

### Bugs & Technical Debt
- Code review findings: `TECHNICAL_AUDIT.md` → "Code Review Findings"
- Security issues: `TECHNICAL_AUDIT.md` → "Security Issues"
- Mobile app issues: `TECHNICAL_AUDIT.md` → "Mobile App Issues"
- Known bugs: `TECHNICAL_AUDIT.md` → "Known Bugs & Edge Cases"

### Tasks & Planning
- Pending tasks: `PLAN.md` → "Pending Tasks"
- Strategic priorities: `TECHNICAL_AUDIT.md` → "Strategic Priorities"
- Task split (parallel work): `TECHNICAL_AUDIT.md` → "Task Split for Parallel Work"

### Users & Permissions
- User registry: `TECHNICAL_AUDIT.md` → "User Registry"
- Multi-user model: `TECHNICAL_AUDIT.md` → "Multi-User Model"
- RLS policies: `TECHNICAL_AUDIT.md` → "Supabase RLS Policies"

### Session History
- Recent changes: `TECHNICAL_AUDIT.md` → "Recent Changes"
- Session notes: `CLAUDE.md` → "Session" sections at bottom

---

## Quick Commands

**Start desktop app:**
```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
source .venv/bin/activate
python main.py
```

**Build & deploy mobile:**
```bash
cd /Users/brookebratcher/Desktop/trailcam_mobile
/Users/brookebratcher/develop/flutter/bin/flutter build apk --release
~/Library/Android/sdk/platform-tools/adb install -r build/app/outputs/flutter-apk/app-release.apk
```

**Check Supabase counts:**
```bash
for table in photos_sync tags annotation_boxes deer_metadata; do
  curl -s "https://iwvehmthbjcvdqjqxtty.supabase.co/rest/v1/$table" \
    -H "apikey: YOUR_KEY" -H "Prefer: count=exact" -I 2>/dev/null | grep content-range
done
```

---

## File Sizes (for context budgeting)

| File | Lines | Notes |
|------|-------|-------|
| `TASKS.md` | ~25 | **Read first** - active work handoff |
| `INDEX.md` | ~100 | Topic index |
| `PLAN.md` | ~160 | Product roadmap |
| `README.md` | ~80 | Public-facing only |
| `AI_REFINEMENT_PLAN.md` | ~400 | AI model details |
| `CLAUDE.md` | ~900 | Dev context & commands |
| `TECHNICAL_AUDIT.md` | ~1000 | Deep technical audit |
