"""
R2UploadQueue - Background upload of photos to Cloudflare R2.

Manages a persistent queue of photos to upload:
- Background thread for non-blocking uploads
- Persistent queue survives app restart
- Deduplication (checks if file exists before upload)
- Progress tracking and status updates
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal

logger = logging.getLogger(__name__)


class R2UploadWorker(QThread):
    """Background worker thread for R2 uploads."""

    progress = pyqtSignal(int, int)  # current, total
    upload_complete = pyqtSignal(str, bool)  # file_hash, success
    all_complete = pyqtSignal(int, int)  # uploaded, failed

    def __init__(self, queue: List[dict], storage):
        super().__init__()
        self.queue = queue
        self.storage = storage
        self._cancelled = False

    def run(self):
        """Process the upload queue."""
        uploaded = 0
        failed = 0
        total = len(self.queue)

        for i, item in enumerate(self.queue):
            if self._cancelled:
                break

            file_hash = item.get('file_hash')
            file_path = Path(item.get('file_path', ''))
            thumb_path = Path(item.get('thumbnail_path', ''))

            success = True

            # Upload thumbnail (shared structure - no username)
            if thumb_path and thumb_path.exists():
                thumb_key = f"thumbnails/{file_hash}_thumb.jpg"
                if not self.storage.check_exists(thumb_key):
                    if not self.storage.upload_file(thumb_path, thumb_key):
                        success = False
                        logger.error(f"Failed to upload thumbnail: {thumb_key}")

            # Upload full photo (shared structure - no username)
            if file_path and file_path.exists():
                photo_key = f"photos/{file_hash}.jpg"
                if not self.storage.check_exists(photo_key):
                    if not self.storage.upload_file(file_path, photo_key):
                        success = False
                        logger.error(f"Failed to upload photo: {photo_key}")

            if success:
                uploaded += 1
            else:
                failed += 1

            self.progress.emit(i + 1, total)
            self.upload_complete.emit(file_hash, success)

        self.all_complete.emit(uploaded, failed)

    def cancel(self):
        """Cancel the upload operation."""
        self._cancelled = True


class R2UploadManager(QObject):
    """Manages the R2 upload queue and coordinates background uploads."""

    # Signals for UI updates
    upload_started = pyqtSignal()
    upload_progress = pyqtSignal(int, int)  # current, total
    upload_completed = pyqtSignal(int, int)  # uploaded, failed
    status_changed = pyqtSignal(str)  # status message

    QUEUE_FILE = Path.home() / ".trailcam" / "r2_upload_queue.json"

    def __init__(self, get_storage: Callable):
        """
        Initialize the R2UploadManager.

        Args:
            get_storage: Callable that returns an R2Storage instance
        """
        super().__init__()
        self._get_storage = get_storage
        self._queue: List[dict] = []
        self._worker: Optional[R2UploadWorker] = None
        self._uploading = False

        # Load any pending uploads from disk
        self._load_queue()

        # Auto-start if we have pending uploads
        if self._queue:
            logger.info(f"Found {len(self._queue)} pending R2 uploads")

    @property
    def is_uploading(self) -> bool:
        """Check if uploads are in progress."""
        return self._uploading

    @property
    def pending_count(self) -> int:
        """Get number of pending uploads."""
        return len(self._queue)

    def queue_photo(self, photo_id: int, file_hash: str,
                    file_path: Path, thumbnail_path: Optional[Path] = None):
        """
        Add a photo to the upload queue.

        Args:
            photo_id: Database ID of the photo
            file_hash: MD5 hash of the file
            file_path: Path to the full photo
            thumbnail_path: Path to the thumbnail (optional)
        """
        if not file_hash:
            logger.warning(f"Cannot queue photo {photo_id}: no file hash")
            return

        # Check if already in queue
        for item in self._queue:
            if item.get('file_hash') == file_hash:
                return  # Already queued

        item = {
            'photo_id': photo_id,
            'file_hash': file_hash,
            'file_path': str(file_path) if file_path else None,
            'thumbnail_path': str(thumbnail_path) if thumbnail_path else None,
            'queued_at': datetime.now().isoformat(),
            'retry_count': 0
        }
        self._queue.append(item)
        self._save_queue()
        self.status_changed.emit(f"R2: {len(self._queue)} pending")

    def start_background_upload(self):
        """Start processing the queue in background thread."""
        if self._uploading or not self._queue:
            return

        storage = self._get_storage()
        if not storage or not storage.is_configured():
            logger.warning("R2 not configured, skipping upload")
            return

        self._uploading = True
        self.upload_started.emit()
        self.status_changed.emit(f"R2: Uploading...")

        # Create a copy of the queue for the worker
        queue_copy = list(self._queue)

        self._worker = R2UploadWorker(queue_copy, storage)
        self._worker.progress.connect(self._on_progress)
        self._worker.upload_complete.connect(self._on_item_complete)
        self._worker.all_complete.connect(self._on_all_complete)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int):
        """Handle progress updates."""
        self.upload_progress.emit(current, total)
        self.status_changed.emit(f"R2: {current}/{total}")

    def _on_item_complete(self, file_hash: str, success: bool):
        """Handle individual upload completion."""
        if success:
            # Remove from queue
            self._queue = [q for q in self._queue if q.get('file_hash') != file_hash]
            self._save_queue()

    def _on_all_complete(self, uploaded: int, failed: int):
        """Handle all uploads complete."""
        self._uploading = False
        self.upload_completed.emit(uploaded, failed)

        if failed > 0:
            self.status_changed.emit(f"R2: {failed} failed")
        elif self._queue:
            self.status_changed.emit(f"R2: {len(self._queue)} pending")
        else:
            self.status_changed.emit("R2: Synced")

        logger.info(f"R2 upload batch complete: {uploaded} uploaded, {failed} failed")

    def _on_worker_finished(self):
        """Clean up worker thread."""
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _save_queue(self):
        """Persist queue to disk."""
        try:
            self.QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'pending': self._queue,
                'updated_at': datetime.now().isoformat()
            }
            self.QUEUE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save R2 upload queue: {e}")

    def _load_queue(self):
        """Load queue from disk."""
        try:
            if self.QUEUE_FILE.exists():
                data = json.loads(self.QUEUE_FILE.read_text())
                self._queue = data.get('pending', [])
        except Exception as e:
            logger.error(f"Failed to load R2 upload queue: {e}")
            self._queue = []

    def cancel(self):
        """Cancel any pending upload operations."""
        if self._worker:
            self._worker.cancel()
            self._worker.wait(2000)
        self._uploading = False

    def get_status(self) -> dict:
        """Get current upload status."""
        return {
            'pending': len(self._queue),
            'uploading': self._uploading,
            'queue': self._queue[:10]  # First 10 items for display
        }
