# Trail Camera Software

Cross-platform desktop application for organizing trail camera photos, with a focus on deer tracking and identification. Also includes a Flutter mobile app for viewing photos on Android.

## Apps

| App | Platform | Purpose |
|-----|----------|---------|
| **Desktop** | macOS, Windows | Full-featured photo organizer with AI labeling |
| **Mobile** | Android | View and filter photos from anywhere |

## Features

- **Automatic Import**: Import photos from SD cards or folders with automatic organization
- **EXIF Extraction**: Automatically extracts date/time and camera model from photos
- **Smart Organization**: Photos are organized by Year/Month in your library folder
- **AI Detection**: MegaDetector v5 for animal detection, species classifier, buck/doe classifier
- **Tagging System**: Tag photos as Buck, Doe, Deer, Turkey, Coyote, or 25+ other species
- **Deer Tracking**: Track individual bucks with custom IDs across seasons
- **Search & Filter**: Filter by tags, deer ID, camera location, date range, collection
- **Cloud Sync**: Supabase for metadata, Cloudflare R2 for photo storage
- **CuddeLink Integration**: Download photos directly from CuddeLink cameras

## Requirements

- Python 3.9+ (3.11 recommended for Windows)
- PyQt6
- See `requirements.txt` for full dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Desktop App

```bash
# macOS/Linux
python main.py

# Windows
python main.py
```

### Mobile App

See `/trailcam_mobile/` folder for Flutter app source.

## Data Locations

- **Photo Library**: `~/TrailCamLibrary/Year/Month/` (macOS) or `C:\TrailCamLibrary\` (Windows)
- **Database**: `~/.trailcam/trailcam.db`
- **Thumbnails**: `~/TrailCamLibrary/.thumbnails/`

## Cloud Services

- **Supabase**: Metadata sync (photos, tags, deer IDs)
- **Cloudflare R2**: Photo and thumbnail storage (zero egress fees)

## Documentation

- `CLAUDE.md` - Developer context and commands
- `HANDOFF.md` - System audit and technical debt
- `AI_REFINEMENT_PLAN.md` - AI model development roadmap
- `PLAN.md` - Product roadmap and session history

## GitHub

https://github.com/cabbp3/Trail-Camera-Software

## Current Stats

- ~7,500 photos in database
- Species model: 12 classes, 97% accuracy
- Buck/doe model: 84% accuracy (needs more Doe training data)
