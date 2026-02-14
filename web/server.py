#!/usr/bin/env python3
"""
Trail Camera Web Server

Serves the web frontend and provides API endpoints for:
- Photo files from TrailCamLibrary
- CuddeLink downloads
- Database sync
"""

import http.server
import socketserver
import os
import sys
import json
import urllib.parse
import threading
from pathlib import Path

PORT = 8080
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(WEB_DIR)

# Add project to path for imports
sys.path.insert(0, PROJECT_DIR)

# Download state (simple in-memory tracking)
download_state = {
    "status": "idle",  # idle, logging_in, fetching, downloading, extracting, complete, error
    "message": "",
    "progress": 0,
    "photos_found": 0,
    "photos_downloaded": 0,
    "error": None
}
download_lock = threading.Lock()


def update_download_state(**kwargs):
    with download_lock:
        download_state.update(kwargs)


def get_download_state():
    with download_lock:
        return dict(download_state)


def run_cuddelink_download(email, password, date_from, date_to):
    """Run CuddeLink download in background thread."""
    try:
        update_download_state(status="logging_in", message="Logging in to CuddeLink...", progress=10)

        # Import the downloader
        from cuddelink_downloader import (
            _login, _apply_date_filter, _fetch_photo_ids,
            _request_download_guid, _download_zip, _extract_images
        )
        import requests

        session = requests.Session()

        # Login
        _login(session, email, password)
        update_download_state(status="fetching", message="Fetching photo list...", progress=25)

        # Apply date filter
        _apply_date_filter(session, date_from, date_to)
        update_download_state(progress=35)

        # Get photo IDs
        photo_ids = _fetch_photo_ids(session)
        if not photo_ids:
            update_download_state(status="complete", message="No photos found in date range", progress=100, photos_found=0)
            return

        update_download_state(
            status="downloading",
            message=f"Downloading {len(photo_ids)} photos...",
            progress=50,
            photos_found=len(photo_ids)
        )

        # Request download
        guid = _request_download_guid(session, photo_ids)
        update_download_state(progress=65)

        # Download zip
        zip_path = _download_zip(session, guid)
        update_download_state(status="extracting", message="Extracting photos...", progress=70)

        # Extract to temp folder first
        temp_dir = get_library_path() / ".cuddelink_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        extracted = _extract_images(zip_path, temp_dir)

        # Clean up zip
        try:
            zip_path.unlink()
        except:
            pass

        if not extracted:
            update_download_state(status="complete", message="No photos extracted", progress=100, photos_downloaded=0)
            return

        # Now import photos properly (organize into YYYY/MM and add to database)
        update_download_state(status="importing", message=f"Importing {len(extracted)} photos...", progress=75)

        from image_processor import import_photo, create_thumbnail
        from database import TrailCamDatabase

        db = TrailCamDatabase()
        imported_count = 0
        skipped_count = 0

        for i, photo_path in enumerate(extracted):
            try:
                original_name = photo_path.name

                # Check if this photo was already imported (by original name)
                cursor = db.conn.cursor()
                cursor.execute("SELECT id FROM photos WHERE original_name = ?", (original_name,))
                if cursor.fetchone():
                    # Already exists, skip it
                    skipped_count += 1
                    try:
                        photo_path.unlink()
                    except:
                        pass
                    continue

                # Import photo (moves to YYYY/MM folder)
                new_path, original_name, date_taken, camera_model = import_photo(str(photo_path))

                # Create thumbnail
                thumb_path = create_thumbnail(new_path)

                # Add to database
                db.add_photo(
                    file_path=new_path,
                    original_name=original_name,
                    date_taken=date_taken,
                    camera_model=camera_model,
                    thumbnail_path=thumb_path
                )

                imported_count += 1

                # Delete the temp file (import_photo copies, so delete original)
                try:
                    photo_path.unlink()
                except:
                    pass

                # Update progress
                progress = 75 + int((i + 1) / len(extracted) * 20)
                update_download_state(
                    progress=progress,
                    message=f"Importing {i + 1}/{len(extracted)} photos..."
                )

            except Exception as e:
                print(f"Error importing {photo_path}: {e}")

        db.close()

        # Clean up temp dir
        try:
            temp_dir.rmdir()
        except:
            pass

        # Refresh photos.json
        update_download_state(status="refreshing", message="Refreshing photo list...", progress=95)
        refresh_photos_json()

        # Build completion message
        if skipped_count > 0:
            msg = f"Imported {imported_count} new photos ({skipped_count} duplicates skipped)"
        else:
            msg = f"Imported {imported_count} photos"

        update_download_state(
            status="complete",
            message=msg,
            progress=100,
            photos_downloaded=imported_count
        )

    except Exception as e:
        update_download_state(status="error", message=str(e), error=str(e))


def get_db_path():
    """Get database path - cross-platform."""
    home = Path.home()
    if sys.platform == 'win32':
        db_dir = home / 'AppData' / 'Roaming' / 'TrailCam'
    else:
        db_dir = home / '.trailcam'
    return str(db_dir / 'trailcam.db')


def get_library_path():
    """Get photo library path - cross-platform."""
    if sys.platform == 'win32':
        return Path("C:/TrailCamLibrary")
    return Path.home() / "TrailCamLibrary"


def refresh_photos_json():
    """Re-export photos from database to JSON."""
    import sqlite3
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.id, p.file_path, p.original_name, p.date_taken, p.camera_model,
               p.thumbnail_path, p.camera_location, p.suggested_tag, p.notes,
               GROUP_CONCAT(t.tag_name) as tags
        FROM photos p
        LEFT JOIN tags t ON p.id = t.photo_id
        GROUP BY p.id
        ORDER BY p.date_taken DESC
    """)

    photos = [dict(row) for row in cursor.fetchall()]
    conn.close()

    with open(os.path.join(WEB_DIR, 'photos.json'), 'w') as f:
        json.dump(photos, f)


def get_r2_storage():
    """Get R2 storage instance if configured."""
    try:
        from r2_storage import R2Storage
        storage = R2Storage()
        if storage.is_configured():
            return storage
    except ImportError:
        pass
    return None


def get_cloud_stats():
    """Get cloud storage statistics."""
    storage = get_r2_storage()
    if not storage:
        return {"error": "R2 not configured"}

    try:
        # List all objects
        objects = []
        paginator = storage.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=storage.bucket_name):
            for obj in page.get('Contents', []):
                objects.append(obj)

        # Analyze by user
        users = {}
        total_size = 0

        for obj in objects:
            key = obj['Key']
            size = obj['Size']
            total_size += size

            parts = key.split('/')
            if len(parts) >= 2 and parts[0] == 'users':
                username = parts[1]
                if username not in users:
                    users[username] = {'photos': 0, 'thumbnails': 0, 'size': 0}
                users[username]['size'] += size
                if '/photos/' in key:
                    users[username]['photos'] += 1
                elif '/thumbnails/' in key:
                    users[username]['thumbnails'] += 1

        return {
            "total_objects": len(objects),
            "total_size": total_size,
            "total_photos": sum(u['photos'] for u in users.values()),
            "users": list(users.keys()),
            "user_stats": users
        }
    except Exception as e:
        return {"error": str(e)}


def get_cloud_photos(username=None):
    """Get list of photos from cloud storage.

    Note: username parameter is kept for API compatibility but ignored.
    All photos use shared structure: thumbnails/{hash}_thumb.jpg
    """
    storage = get_r2_storage()
    if not storage:
        return []

    try:
        photos = []
        # Shared structure: thumbnails are at root level, not per-user
        prefix = "thumbnails/"

        paginator = storage.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=storage.bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('_thumb.jpg'):
                    # Extract file hash from thumbnail key (thumbnails/{hash}_thumb.jpg)
                    file_hash = key.replace('thumbnails/', '').replace('_thumb.jpg', '')

                    # Generate signed URLs
                    thumb_url = storage.get_signed_url(key, expires_in=3600)
                    photo_key = f"photos/{file_hash}.jpg"
                    photo_url = storage.get_signed_url(photo_key, expires_in=3600)

                    photos.append({
                        'id': file_hash,
                        'user': 'shared',  # Shared structure, no per-user folders
                        'thumbnail_url': thumb_url,
                        'photo_url': photo_url
                    })

        return photos
    except Exception as e:
        print(f"Error getting cloud photos: {e}")
        return []


class TrailCamHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        # API: Cloud stats
        if path == '/api/cloud/stats':
            self.send_json_response(get_cloud_stats())
            return

        # API: Cloud photos
        if path == '/api/cloud/photos':
            username = query.get('user', [None])[0]
            photos = get_cloud_photos(username)
            self.send_json_response(photos)
            return

        # API: Download status
        if path == '/api/cuddelink/status':
            self.send_json_response(get_download_state())
            return

        # Serve photos.json with no-cache headers
        if path == '/photos.json':
            json_path = os.path.join(WEB_DIR, 'photos.json')
            if os.path.exists(json_path):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()
                with open(json_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, 'photos.json not found')
            return

        # Serve photos from TrailCamLibrary (with path traversal protection)
        if path.startswith('/photos/'):
            base_path = get_library_path()
            requested_path = (base_path / path[8:].lstrip('/')).resolve()
            # Ensure the resolved path is within the library directory
            if str(requested_path).startswith(str(base_path.resolve())):
                self.serve_file(str(requested_path), 'image/jpeg')
            else:
                self.send_error(403, 'Access denied')
            return

        # Serve thumbnails (with path traversal protection)
        if path.startswith('/thumbnails/'):
            base_path = get_library_path() / '.thumbnails'
            requested_path = (base_path / path[12:].lstrip('/')).resolve()
            # Ensure the resolved path is within the thumbnails directory
            if str(requested_path).startswith(str(base_path.resolve())):
                self.serve_file(str(requested_path), 'image/jpeg')
            else:
                self.send_error(403, 'Access denied')
            return

        # Default handler
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        # API: Start CuddeLink download
        if path == '/api/cuddelink/download':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')

            try:
                data = json.loads(body)
                email = data.get('email', '')
                password = data.get('password', '')
                date_from = data.get('dateFrom', '')
                date_to = data.get('dateTo', '')

                if not email or not password:
                    self.send_json_response({'error': 'Email and password required'}, status=400)
                    return

                # Reset state and start download in background
                update_download_state(
                    status="starting",
                    message="Starting download...",
                    progress=0,
                    photos_found=0,
                    photos_downloaded=0,
                    error=None
                )

                thread = threading.Thread(
                    target=run_cuddelink_download,
                    args=(email, password, date_from, date_to)
                )
                thread.daemon = True
                thread.start()

                self.send_json_response({'success': True, 'message': 'Download started'})

            except json.JSONDecodeError:
                self.send_json_response({'error': 'Invalid JSON'}, status=400)
            except Exception as e:
                self.send_json_response({'error': str(e)}, status=500)
            return

        # API: Refresh photos JSON
        if path == '/api/refresh':
            try:
                refresh_photos_json()
                self.send_json_response({'success': True})
            except Exception as e:
                self.send_json_response({'error': str(e)}, status=500)
            return

        self.send_error(404, 'Not Found')

    def serve_file(self, file_path, content_type):
        if os.path.exists(file_path):
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Cache-Control', 'max-age=86400')
            self.end_headers()
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, f'File not found: {file_path}')

    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        # Quieter logging
        if '404' in str(args) or '500' in str(args):
            super().log_message(format, *args)


if __name__ == '__main__':
    # Allow socket reuse
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", PORT), TrailCamHandler) as httpd:
        print(f"TrailCam Web Server")
        print(f"-------------------")
        print(f"URL: http://localhost:{PORT}")
        print(f"Web files: {WEB_DIR}")
        print(f"Photos: {get_library_path()}")
        print(f"\nPress Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
