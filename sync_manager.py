"""
SyncManager - Automatic synchronization to Supabase with offline support.

Manages debounced syncing of label changes to the cloud:
- 30 second debounce after last change
- Queues changes when offline
- Retries with exponential backoff
- Status indicator updates
"""

import json
import logging
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, QThread

logger = logging.getLogger(__name__)


class SyncWorker(QThread):
    """Background worker thread for sync operations."""

    sync_completed = pyqtSignal(dict)  # counts dict
    sync_failed = pyqtSignal(str)  # error message

    def __init__(self, db, supabase_client):
        super().__init__()
        self.db = db
        self.supabase_client = supabase_client
        self._cancelled = False

    def run(self):
        """Execute the sync operation."""
        try:
            # Use existing push_to_supabase method
            counts = self.db.push_to_supabase(
                self.supabase_client,
                progress_callback=None  # Silent sync
            )
            if not self._cancelled:
                self.sync_completed.emit(counts)
        except Exception as e:
            if not self._cancelled:
                self.sync_failed.emit(str(e))

    def cancel(self):
        """Cancel the sync operation."""
        self._cancelled = True


class SyncManager(QObject):
    """Manages automatic synchronization to Supabase with offline support."""

    # Signals for UI updates
    sync_started = pyqtSignal()
    sync_completed = pyqtSignal(int)  # total items synced
    sync_failed = pyqtSignal(str)  # error message
    status_changed = pyqtSignal(str)  # 'idle', 'pending', 'syncing', 'offline'

    # Configuration
    DEBOUNCE_DELAY = 30000  # 30 seconds in milliseconds
    PERIODIC_SYNC_INTERVAL = 120000  # 2 minutes - regular sync to catch all changes
    QUEUE_FILE = Path.home() / ".trailcam" / "pending_sync.json"
    MAX_RETRIES = 5
    RETRY_DELAYS = [30000, 60000, 120000, 300000, 600000]  # 30s, 1m, 2m, 5m, 10m

    def __init__(self, db, get_supabase_client: Callable):
        """
        Initialize the SyncManager.

        Args:
            db: Database instance with push_to_supabase method
            get_supabase_client: Callable that returns the Supabase client
        """
        super().__init__()
        self.db = db
        self._get_supabase_client = get_supabase_client

        # State
        self._status = 'idle'
        self._pending = False
        self._retry_count = 0
        self._worker: Optional[SyncWorker] = None

        # Debounce timer
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._on_debounce_timeout)

        # Retry timer
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.timeout.connect(self._attempt_sync)

        # Network check timer (periodic when offline)
        self._network_timer = QTimer(self)
        self._network_timer.timeout.connect(self._check_and_sync)

        # Periodic sync timer - DISABLED for now due to potential threading issues
        # self._periodic_timer = QTimer(self)
        # self._periodic_timer.timeout.connect(self._periodic_sync)
        # self._periodic_timer.start(self.PERIODIC_SYNC_INTERVAL)
        # logger.info(f"Periodic sync enabled: every {self.PERIODIC_SYNC_INTERVAL // 1000} seconds")

        # Load any pending state from previous session
        self._load_queue()

        # If we have pending changes from last session, schedule sync
        if self._pending:
            self._start_debounce_timer()

    @property
    def status(self) -> str:
        """Current sync status."""
        return self._status

    def _set_status(self, status: str):
        """Update status and emit signal."""
        if status != self._status:
            self._status = status
            self.status_changed.emit(status)
            logger.debug(f"Sync status changed to: {status}")

    def queue_change(self):
        """
        Queue a label change for syncing.

        Call this after any local save that should sync to cloud.
        Starts/resets the 30-second debounce timer.
        """
        self._pending = True
        self._save_queue()
        self._start_debounce_timer()
        self._set_status('pending')

    def _start_debounce_timer(self):
        """Start or reset the debounce timer."""
        self._debounce_timer.stop()
        self._debounce_timer.start(self.DEBOUNCE_DELAY)

    def _on_debounce_timeout(self):
        """Called when debounce timer expires."""
        self._attempt_sync()

    def _periodic_sync(self):
        """Called every 2 minutes to ensure data is regularly pushed.

        This catches any changes that weren't properly queued via queue_change().
        Uses incremental sync based on updated_at timestamps.
        """
        # Don't run if already syncing or offline
        if self._status == 'syncing':
            return

        # Always mark pending to ensure we push any missed changes
        self._pending = True
        logger.debug("Periodic sync triggered")
        self._attempt_sync()

    def _check_network(self) -> bool:
        """Check if we can reach the Supabase server."""
        try:
            # Quick socket check to Supabase
            sock = socket.create_connection(("iwvehmthbjcvdqjqxtty.supabase.co", 443), timeout=5)
            sock.close()  # Close socket to prevent resource leak
            return True
        except (socket.timeout, socket.error, OSError):
            return False

    def _attempt_sync(self):
        """Attempt to sync to Supabase."""
        if not self._pending:
            self._set_status('idle')
            return

        # Check network first
        if not self._check_network():
            self._set_status('offline')
            self._schedule_retry()
            return

        # Get Supabase client
        try:
            client = self._get_supabase_client()
            if not client or not client.is_configured():
                logger.warning("Supabase not configured, skipping sync")
                self._set_status('offline')
                return
        except Exception as e:
            logger.error(f"Failed to get Supabase client: {e}")
            self._set_status('offline')
            self._schedule_retry()
            return

        # Start sync in background thread
        self._set_status('syncing')
        self.sync_started.emit()

        self._worker = SyncWorker(self.db, client)
        self._worker.sync_completed.connect(self._on_sync_completed)
        self._worker.sync_failed.connect(self._on_sync_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_sync_completed(self, counts: dict):
        """Handle successful sync."""
        self._pending = False
        self._retry_count = 0
        self._save_queue()

        # Calculate total items synced
        total = sum(counts.values()) if counts else 0

        self._set_status('idle')
        self.sync_completed.emit(total)
        logger.info(f"Sync completed: {counts}")

    def _on_sync_failed(self, error: str):
        """Handle sync failure."""
        logger.error(f"Sync failed: {error}")
        self.sync_failed.emit(error)
        self._schedule_retry()

    def _on_worker_finished(self):
        """Clean up worker thread."""
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _schedule_retry(self):
        """Schedule a retry with exponential backoff."""
        if self._retry_count < self.MAX_RETRIES:
            delay = self.RETRY_DELAYS[min(self._retry_count, len(self.RETRY_DELAYS) - 1)]
            self._retry_timer.start(delay)
            self._retry_count += 1
            logger.info(f"Scheduling retry {self._retry_count} in {delay/1000}s")
        else:
            # Start network monitoring for reconnection
            self._set_status('offline')
            self._network_timer.start(60000)  # Check every minute
            logger.warning("Max retries reached, waiting for network")

    def _check_and_sync(self):
        """Check network and sync if available (called by network timer)."""
        if self._check_network():
            self._network_timer.stop()
            self._retry_count = 0
            self._attempt_sync()

    def _save_queue(self):
        """Persist pending state to disk."""
        try:
            self.QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'pending': self._pending,
                'last_change': datetime.now().isoformat(),
                'retry_count': self._retry_count
            }
            self.QUEUE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save sync queue: {e}")

    def _load_queue(self):
        """Load pending state from disk."""
        try:
            if self.QUEUE_FILE.exists():
                data = json.loads(self.QUEUE_FILE.read_text())
                self._pending = data.get('pending', False)
                self._retry_count = data.get('retry_count', 0)
        except Exception as e:
            logger.error(f"Failed to load sync queue: {e}")
            self._pending = False

    def force_sync(self):
        """Force an immediate sync (ignores debounce)."""
        self._debounce_timer.stop()
        self._retry_timer.stop()
        self._pending = True
        self._retry_count = 0
        self._attempt_sync()

    def cancel(self):
        """Cancel any pending sync operations."""
        self._debounce_timer.stop()
        self._retry_timer.stop()
        self._network_timer.stop()
        if self._worker:
            self._worker.cancel()
            self._worker.wait(1000)

    def get_time_until_sync(self) -> int:
        """Get milliseconds until next sync (0 if not pending)."""
        if self._debounce_timer.isActive():
            return self._debounce_timer.remainingTime()
        return 0
