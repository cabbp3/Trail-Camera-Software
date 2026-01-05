# Trail Camera Software

Two cross-platform Python desktop applications for organizing trail camera photos, with a focus on deer tracking and identification.

## Two Apps

| App | Command | Purpose |
|-----|---------|---------|
| **Organizer** | `python main.py` | Simplified photo browser for end users |
| **Trainer** | `python trainer_main.py` | Advanced labeling tool for AI training (internal) |

Both apps share the same database and AI models.

## Features

- **Automatic Import**: Import photos from SD cards or folders with automatic organization
- **EXIF Extraction**: Automatically extracts date/time and camera model from photos
- **Smart Organization**: Photos are organized by Year/Month in your library folder
- **Tagging System**: Tag photos as Buck, Doe, Fawn, Turkey, Coyote, or Trash
- **Deer Tracking**: Track individual deer with custom IDs and age classes
- **Search & Filter**: Search by tags, deer ID, or date range
- **Thumbnail Grid**: Browse photos in a clean 5×5 grid layout
- **Full Preview**: Click any thumbnail to view full-size image with tagging options

## Requirements

- Python 3.8 or higher
- PyQt6
- Pillow (PIL)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. **Import Photos**:
   - Click `File → Import Folder...` (or press `Ctrl+I`)
   - Select the folder containing your trail camera photos (e.g., SD card)
   - The app will automatically:
     - Copy all JPG files to your library folder
     - Extract EXIF data (date/time, camera model)
     - Rename files as `YYYY-MM-DD_HH-MM-SS_CameraModel.jpg`
     - Organize by Year/Month

3. **Library Location**:
   - **macOS**: `~/TrailCamLibrary/Year/Month/`
   - **Windows**: `C:\TrailCamLibrary\Year\Month\`

4. **Tagging Photos**:
   - Click any thumbnail to open the preview window
   - Click tag buttons (Buck, Doe, Fawn, etc.) to tag the photo
   - Enter a Deer ID (e.g., "Bucky-2025")
   - Select an age class from the dropdown

5. **Searching**:
   - Use the search bar at the top to filter by:
     - Tag type
     - Deer ID
     - Date range

## Keyboard Shortcuts

- `Ctrl+I`: Import Folder
- `Ctrl+Q`: Quit Application

## Database

All metadata is stored in a SQLite database located at:
- **macOS/Linux**: `~/.trailcam/trailcam.db`
- **Windows**: `%USERPROFILE%\.trailcam\trailcam.db`

The database stores:
- Photo metadata (path, date, camera model)
- Tags (permanently saved)
- Deer IDs and age classes

## Notes

- The app only imports JPG files (case-insensitive)
- Duplicate filenames are automatically handled with a counter
- Thumbnails are cached in `.thumbnails` folder within the library
- If EXIF date is missing, file modification time is used




