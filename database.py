"""
Database module for managing trail camera photo metadata and tags.
"""
import sqlite3
import os
import time
import threading
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class TrailCamDatabase:
    """Manages SQLite database for trail camera photos."""

    def __init__(self, db_path: str = None):
        """Initialize database connection.

        Args:
            db_path: Path to database file. If None, uses default location.
        """
        if db_path is None:
            # Default to user's home directory
            home = Path.home()
            db_dir = home / ".trailcam"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "trailcam.db")

        self.db_path = db_path
        self.db_dir = Path(db_path).parent
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        # Check for crash on previous run and repair if needed
        self._check_and_repair_after_crash()

        self.conn = self._connect_with_retry(db_path)
        self.conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access and performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign key enforcement (SQLite has it off by default)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_database()

        # Set crash flag - will be removed on clean exit
        self._set_crash_flag()

        # Audit log connection (lazy loaded)
        self._audit_conn = None

        # Create daily backup if not done today (protects against corruption)
        self.check_daily_backup()

    def _connect_with_retry(self, db_path: str, max_retries: int = 5) -> sqlite3.Connection:
        """Connect to SQLite with exponential backoff on 'database is locked'."""
        delay = 0.1
        for attempt in range(max_retries):
            try:
                return sqlite3.connect(db_path, check_same_thread=False)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

    def _get_audit_conn(self):
        """Get connection to audit log database (lazy loaded)."""
        if self._audit_conn is None:
            audit_path = self.db_dir / "audit_log.db"
            self._audit_conn = sqlite3.connect(str(audit_path), check_same_thread=False)
            self._audit_conn.execute("""
                CREATE TABLE IF NOT EXISTS tag_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    action TEXT NOT NULL,
                    photo_id INTEGER NOT NULL,
                    file_path TEXT,
                    file_hash TEXT,
                    tag_name TEXT NOT NULL
                )
            """)
            self._audit_conn.commit()
        return self._audit_conn

    def _log_tag_change(self, action: str, photo_id: int, tag_name: str):
        """Log a tag change to the audit database. Thread-safe."""
        try:
            # Get file_path and file_hash for this photo
            with self._lock:
                cursor = self.conn.cursor()
                cursor.execute("SELECT file_path, file_hash FROM photos WHERE id = ?", (photo_id,))
                row = cursor.fetchone()
                file_path = row['file_path'] if row else None
                file_hash = row['file_hash'] if row else None

            audit = self._get_audit_conn()
            audit.execute(
                "INSERT INTO tag_log (action, photo_id, file_path, file_hash, tag_name) VALUES (?, ?, ?, ?, ?)",
                (action, photo_id, file_path, file_hash, tag_name)
            )
            audit.commit()
        except Exception as e:
            logger.warning(f"Failed to log tag change: {e}")

    def log_tag_confirmation(self, photo_id: int, tags: list):
        """Log that existing tags were confirmed/kept during review."""
        for tag in tags:
            self._log_tag_change('confirm', photo_id, tag)

    def verify_audit_log(self) -> Dict:
        """Compare audit log against main database to find discrepancies.

        Returns a dict with:
            - missing_in_db: Tags in audit log (net adds) but not in main DB
            - extra_in_db: Tags in main DB but not accounted for in audit log
            - total_audit_entries: Number of entries in audit log
        """
        audit = self._get_audit_conn()
        cursor = audit.cursor()

        # Get net state from audit log (adds/baseline minus removes per photo/tag)
        cursor.execute("""
            SELECT file_hash, tag_name,
                   SUM(CASE WHEN action IN ('add', 'baseline') THEN 1 ELSE -1 END) as net
            FROM tag_log
            WHERE file_hash IS NOT NULL
            GROUP BY file_hash, tag_name
            HAVING net > 0
        """)
        audit_tags = {(row[0], row[1]) for row in cursor.fetchall()}

        # Get current state from main DB (thread-safe)
        with self._lock:
            main_cursor = self.conn.cursor()
            main_cursor.execute("""
                SELECT p.file_hash, t.tag_name
                FROM tags t
                JOIN photos p ON t.photo_id = p.id
                WHERE p.file_hash IS NOT NULL
                  AND t.deleted_at IS NULL
            """)
            db_tags = {(row[0], row[1]) for row in main_cursor.fetchall()}

        cursor.execute("SELECT COUNT(*) FROM tag_log")
        total_entries = cursor.fetchone()[0]

        return {
            'missing_in_db': audit_tags - db_tags,
            'extra_in_db': db_tags - audit_tags,
            'total_audit_entries': total_entries
        }

    def _get_crash_flag_path(self) -> Path:
        """Get path to crash detection flag file."""
        return self.db_dir / ".running"

    def _set_crash_flag(self):
        """Set flag indicating app is running (for crash detection)."""
        try:
            flag_path = self._get_crash_flag_path()
            flag_path.write_text(f"pid:{os.getpid()}\nstarted:{datetime.now().isoformat()}")
        except Exception as e:
            logger.warning(f"Could not set crash flag: {e}")

    def _clear_crash_flag(self):
        """Clear crash flag on clean exit."""
        try:
            flag_path = self._get_crash_flag_path()
            if flag_path.exists():
                flag_path.unlink()
        except Exception as e:
            logger.warning(f"Could not clear crash flag: {e}")

    def _check_and_repair_after_crash(self):
        """Check if previous session crashed and repair database if needed."""
        import shutil
        flag_path = self._get_crash_flag_path()

        if not flag_path.exists():
            return  # Clean shutdown last time

        # Check if the PID in the flag file is still alive
        try:
            content = flag_path.read_text()
            for line in content.splitlines():
                if line.startswith("pid:"):
                    old_pid = int(line.split(":", 1)[1].strip())
                    try:
                        os.kill(old_pid, 0)  # Signal 0 = check if alive
                        # PID is alive — another instance is running, not a crash
                        logger.info(f"Another instance (PID {old_pid}) is still running, skipping crash repair")
                        return
                    except (OSError, ProcessLookupError):
                        break  # PID is dead — treat as crash
        except Exception:
            pass  # Can't read flag — assume crash

        logger.warning("Detected unclean shutdown - checking database integrity...")
        print("Detected unclean shutdown - checking database integrity...")

        db_path = Path(self.db_path)
        wal_path = db_path.with_suffix('.db-wal')
        shm_path = db_path.with_suffix('.db-shm')

        try:
            # Step 1: Try to checkpoint WAL to merge pending writes
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()

            # Step 2: Clean up WAL/SHM files after checkpoint
            # These can cause issues if left over from a crash
            for f in [wal_path, shm_path]:
                if f.exists():
                    try:
                        f.unlink()
                        logger.info(f"Removed stale {f.name}")
                    except Exception as e:
                        logger.warning(f"Could not remove {f.name}: {e}")

            # Step 3: Check integrity
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            conn.close()

            if result != "ok":
                logger.warning(f"Database integrity issues: {result[:200]}...")
                print("Database corruption detected - attempting recovery...")

                # Back up the corrupted database first
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                corrupt_backup = db_path.parent / f"trailcam_corrupted_{timestamp}.db"
                shutil.copy2(self.db_path, corrupt_backup)
                logger.info(f"Backed up corrupted database to {corrupt_backup}")
                print(f"Backed up corrupted DB to: {corrupt_backup.name}")

                # Try VACUUM INTO to create a clean copy
                recovered_path = db_path.parent / "trailcam_recovered.db"
                try:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.execute(f"VACUUM INTO '{recovered_path}'")
                    conn.close()

                    # Verify the recovered database
                    test_conn = sqlite3.connect(str(recovered_path), check_same_thread=False)
                    test_cursor = test_conn.cursor()
                    test_cursor.execute("PRAGMA integrity_check")
                    test_result = test_cursor.fetchone()[0]
                    test_cursor.execute("SELECT COUNT(*) FROM photos")
                    photo_count = test_cursor.fetchone()[0]
                    test_conn.close()

                    if test_result == "ok" and photo_count > 0:
                        # Replace corrupted DB with recovered one
                        db_path.unlink()
                        recovered_path.rename(db_path)
                        logger.info(f"Database recovered successfully ({photo_count} photos)")
                        print(f"Database recovered! ({photo_count} photos intact)")
                    else:
                        recovered_path.unlink()
                        raise Exception(f"Recovery failed: integrity={test_result}, photos={photo_count}")

                except Exception as vacuum_error:
                    logger.error(f"VACUUM recovery failed: {vacuum_error}")
                    print(f"VACUUM recovery failed: {vacuum_error}")

                    # Try to find most recent good backup
                    backup_dir = db_path.parent / "backups"
                    if backup_dir.exists():
                        backups = sorted(backup_dir.glob("trailcam_*.db"),
                                        key=lambda p: p.stat().st_mtime, reverse=True)
                        for backup in backups:
                            try:
                                test_conn = sqlite3.connect(str(backup), check_same_thread=False)
                                test_cursor = test_conn.cursor()
                                test_cursor.execute("PRAGMA integrity_check")
                                if test_cursor.fetchone()[0] == "ok":
                                    test_cursor.execute("SELECT COUNT(*) FROM photos")
                                    count = test_cursor.fetchone()[0]
                                    test_conn.close()

                                    print(f"Found good backup: {backup.name} ({count} photos)")
                                    logger.info(f"Restoring from backup: {backup.name}")
                                    db_path.unlink()
                                    shutil.copy2(backup, db_path)
                                    print(f"Restored from backup! ({count} photos)")
                                    break
                                test_conn.close()
                            except Exception:
                                continue
                        else:
                            print("WARNING: No valid backup found. Database may be corrupted.")
                            logger.error("No valid backup found for restoration")
                    else:
                        print("WARNING: No backups exist. Consider restoring manually.")
            else:
                logger.info("Database integrity OK after crash recovery")
                print("Database integrity verified - OK")

            # Clear the old crash flag (we'll set a new one after init)
            flag_path.unlink()

        except Exception as e:
            logger.error(f"Error during crash recovery: {e}")
            print(f"Error during recovery: {e}")
            # Still try to clear the flag to avoid infinite loop
            try:
                flag_path.unlink()
            except Exception:
                pass

    def close(self):
        """Close database connection cleanly."""
        try:
            # Close audit log connection if open
            if self._audit_conn:
                self._audit_conn.close()
                self._audit_conn = None
            # Checkpoint WAL before closing
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.conn.close()
            # Clear crash flag on clean exit
            self._clear_crash_flag()
            logger.info("Database closed cleanly")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

    def checkpoint_wal(self) -> bool:
        """Checkpoint WAL to main database file.

        This should be called periodically (e.g., every 5 minutes) to reduce
        data loss risk in case of crash. The WAL checkpoint merges pending
        writes from the write-ahead log into the main database file.

        Returns:
            True if checkpoint succeeded, False otherwise
        """
        try:
            with self._lock:
                result = self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                # PASSIVE checkpoint doesn't block - returns (busy, log, checkpointed)
                if result:
                    logger.debug(f"WAL checkpoint: log={result[1]}, checkpointed={result[2]}")
                return True
        except Exception as e:
            logger.warning(f"WAL checkpoint failed: {e}")
            return False

    # --- Backup Methods ---

    def get_backup_dir(self) -> Path:
        """Get the backup directory path, creating it if needed."""
        backup_dir = self.db_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        return backup_dir

    def create_backup(self, reason: str = "manual") -> Optional[Path]:
        """Create a backup of the database.

        Args:
            reason: Reason for backup (e.g., 'manual', 'batch_operation', 'daily')

        Returns:
            Path to backup file, or None if backup failed
        """
        import shutil

        try:
            # Checkpoint WAL first to ensure all data is in main db file
            with self._lock:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            backup_dir = self.get_backup_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"trailcam_{reason}_{timestamp}.db"
            backup_path = backup_dir / backup_name

            # Copy the database file
            shutil.copy2(self.db_path, backup_path)

            logger.info(f"Database backup created: {backup_path}")
            print(f"[Backup] Created: {backup_name}")

            # Clean up old backups (keep last 30 per reason)
            self._cleanup_old_backups(reason, keep=30)

            return backup_path

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            print(f"[Backup] Failed: {e}")
            return None

    def _cleanup_old_backups(self, reason: str, keep: int = 30):
        """Remove old backups, keeping only the most recent ones."""
        try:
            backup_dir = self.get_backup_dir()
            pattern = f"trailcam_{reason}_*.db"
            backups = sorted(backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove backups beyond the keep limit
            for old_backup in backups[keep:]:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup.name}")
        except Exception as e:
            logger.warning(f"Error cleaning up old backups: {e}")

    def backup_before_batch_operation(self) -> bool:
        """Create a backup before a batch operation.

        Returns:
            True if backup was successful, False otherwise
        """
        backup_path = self.create_backup(reason="pre_batch")
        return backup_path is not None

    def check_daily_backup(self) -> bool:
        """Create a daily backup if one hasn't been made today.

        Returns:
            True if a new backup was created or today's backup exists
        """
        try:
            backup_dir = self.get_backup_dir()
            today = datetime.now().strftime("%Y%m%d")

            # Check if today's backup already exists
            today_pattern = f"trailcam_daily_{today}*.db"
            existing = list(backup_dir.glob(today_pattern))

            if existing:
                logger.info(f"Daily backup already exists: {existing[0].name}")
                return True

            # Checkpoint WAL before backup to capture recent writes
            self.checkpoint_wal()
            # Create today's backup
            backup_path = self.create_backup(reason="daily")
            return backup_path is not None

        except Exception as e:
            logger.error(f"Error checking daily backup: {e}")
            return False

    def get_latest_backup(self) -> Optional[Path]:
        """Get the most recent backup file."""
        try:
            backup_dir = self.get_backup_dir()
            backups = sorted(backup_dir.glob("trailcam_*.db"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            return backups[0] if backups else None
        except Exception as e:
            logger.error(f"Error finding latest backup: {e}")
            return None

    # --- Transaction Methods ---

    def begin_transaction(self):
        """Begin an explicit transaction for batch operations."""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            logger.debug("Transaction started")

    def commit_transaction(self):
        """Commit the current transaction."""
        with self._lock:
            self.conn.commit()
            logger.debug("Transaction committed")

    def rollback_transaction(self):
        """Rollback the current transaction."""
        with self._lock:
            self.conn.rollback()
            logger.debug("Transaction rolled back")

    def _execute_with_lock(self, func):
        """Execute a database operation with thread safety."""
        with self._lock:
            return func()
    
    def _init_database(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        
        # Photos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                original_name TEXT,
                date_taken TEXT,
                camera_model TEXT,
                import_date TEXT,
                thumbnail_path TEXT,
                favorite INTEGER DEFAULT 0,
                notes TEXT DEFAULT '',
                season_year INTEGER,
                camera_location TEXT,
                key_characteristics TEXT,
                suggested_tag TEXT,
                suggested_confidence REAL,
                suggested_sex TEXT,
                suggested_sex_confidence REAL,
                site_id INTEGER,
                suggested_site_id INTEGER,
                suggested_site_confidence REAL,
                stamp_location TEXT,
                collection TEXT DEFAULT '',
                file_hash TEXT
            )
        """)
        
        # Tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                tag_name TEXT NOT NULL,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                UNIQUE(photo_id, tag_name)
            )
        """)
        
        # Deer metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deer_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
            deer_id TEXT,
            age_class TEXT,
            left_points_min INTEGER,
            right_points_min INTEGER,
            left_points_uncertain INTEGER DEFAULT 0,
            right_points_uncertain INTEGER DEFAULT 0,
            left_ab_points_min INTEGER,
            right_ab_points_min INTEGER,
            left_ab_points_uncertain INTEGER DEFAULT 0,
            right_ab_points_uncertain INTEGER DEFAULT 0,
            left_points_max INTEGER,
            right_points_max INTEGER,
            abnormal_points_min INTEGER,
            abnormal_points_max INTEGER,
            broken_antler_side TEXT,
            broken_antler_note TEXT,
            FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                UNIQUE(photo_id)
            )
        """)

        # Additional deer per photo
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deer_additional (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
            deer_id TEXT NOT NULL,
            age_class TEXT,
            left_points_min INTEGER,
            right_points_min INTEGER,
            left_points_uncertain INTEGER DEFAULT 0,
            right_points_uncertain INTEGER DEFAULT 0,
            left_ab_points_min INTEGER,
            right_ab_points_min INTEGER,
            left_ab_points_uncertain INTEGER DEFAULT 0,
            right_ab_points_uncertain INTEGER DEFAULT 0,
            left_points_max INTEGER,
            right_points_max INTEGER,
            abnormal_points_min INTEGER,
            abnormal_points_max INTEGER,
            FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
            UNIQUE(photo_id, deer_id)
        )
        """)

        # Buck profiles (per deer, per season)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS buck_profiles (
                deer_id TEXT PRIMARY KEY,
                display_name TEXT,
                notes TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS buck_profile_seasons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deer_id TEXT NOT NULL,
                season_year INTEGER NOT NULL,
                camera_locations TEXT,
                key_characteristics TEXT,
                left_points_min INTEGER,
                left_points_uncertain INTEGER DEFAULT 0,
                right_points_min INTEGER,
                right_points_uncertain INTEGER DEFAULT 0,
                left_ab_points_min INTEGER,
                left_ab_points_uncertain INTEGER DEFAULT 0,
                right_ab_points_min INTEGER,
                right_ab_points_uncertain INTEGER DEFAULT 0,
                abnormal_points_min INTEGER,
                abnormal_points_max INTEGER,
                broken_antler_side TEXT,
                broken_antler_note TEXT,
                UNIQUE(deer_id, season_year)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_date ON photos(date_taken)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_photo ON tags(photo_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(tag_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_deer_photo ON deer_metadata(photo_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_deer_id ON deer_metadata(deer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_deer_additional_photo ON deer_additional(photo_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_deer_additional_id ON deer_additional(deer_id)")
        # Additional indexes for common query patterns (columns that exist in CREATE TABLE)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_suggested_tag ON photos(suggested_tag)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_season ON photos(season_year)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_composite ON tags(photo_id, tag_name)")
        # Annotation boxes: store relative coords (0-1), label, and optional confidence
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotation_boxes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                confidence REAL,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
            )
        """)

        # Sites table for auto-detected camera locations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                representative_photo_id INTEGER,
                photo_count INTEGER DEFAULT 0,
                created_at TEXT,
                confirmed INTEGER DEFAULT 1,
                FOREIGN KEY (representative_photo_id) REFERENCES photos(id)
            )
        """)

        # Photo embeddings for site clustering (stored as blob for efficiency)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photo_embeddings (
                photo_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_version TEXT,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
            )
        """)

        # Claude review queue - photos flagged by Claude for manual review
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claude_review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                reason TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TEXT,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_claude_review_photo ON claude_review_queue(photo_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_claude_review_pending ON claude_review_queue(reviewed_at)")

        self.conn.commit()
        self._ensure_photo_columns()
        self._ensure_site_columns()
        self._ensure_deer_columns()
        self._ensure_box_columns()
        self._ensure_sync_tracking()
        self._migrate_deer_metadata_to_boxes()
        self._ensure_roles_schema()

    def _ensure_photo_columns(self):
        """Ensure newer photo columns exist for backward compatibility."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(photos)")
        cols = {row[1] for row in cursor.fetchall()}

        def add_column(name: str, ddl: str, default=None):
            if name not in cols:
                cursor.execute(f"ALTER TABLE photos ADD COLUMN {ddl}")
                if default is not None:
                    cursor.execute(f"UPDATE photos SET {name} = ?", (default,))

        add_column("favorite", "favorite INTEGER DEFAULT 0", 0)
        add_column("notes", "notes TEXT DEFAULT ''", "")
        add_column("season_year", "season_year INTEGER", None)
        add_column("camera_location", "camera_location TEXT", "")
        add_column("key_characteristics", "key_characteristics TEXT", "")
        add_column("suggested_tag", "suggested_tag TEXT", "")
        add_column("suggested_confidence", "suggested_confidence REAL", None)
        add_column("suggested_sex", "suggested_sex TEXT", "")
        add_column("suggested_sex_confidence", "suggested_sex_confidence REAL", None)
        add_column("site_id", "site_id INTEGER", None)  # Auto-detected site
        add_column("collection", "collection TEXT", "")  # Photo collection grouping
        add_column("file_hash", "file_hash TEXT", None)  # MD5 hash for cross-computer sync
        add_column("archived", "archived INTEGER DEFAULT 0", 0)  # Hide from default view
        add_column("owner", "owner TEXT", "")  # Photo owner for cloud sync
        # Create index on archived column (must be after column exists)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_archived ON photos(archived)")
        # Custom species list table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_species (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        # Recent buck IDs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recent_bucks (
                deer_id TEXT PRIMARY KEY,
                updated_at INTEGER NOT NULL
            )
        """)
        self.conn.commit()
        self._backfill_season_years()

    def _ensure_site_columns(self):
        """Ensure newer site columns exist for backward compatibility."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(sites)")
        cols = {row[1] for row in cursor.fetchall()}
        if "confirmed" not in cols:
            cursor.execute("ALTER TABLE sites ADD COLUMN confirmed INTEGER DEFAULT 1")
            cursor.execute("UPDATE sites SET confirmed = 1 WHERE confirmed IS NULL")
        self.conn.commit()

    def _ensure_deer_columns(self):
        """Ensure deer metadata tables have newer columns."""
        cursor = self.conn.cursor()

        def add_column(table: str, name: str, ddl: str):
            cursor.execute(f"PRAGMA table_info({table})")
            cols = {row[1] for row in cursor.fetchall()}
            if name not in cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

        for table in ["deer_metadata", "deer_additional"]:
            add_column(table, "left_points_min", "left_points_min INTEGER")
            add_column(table, "left_points_max", "left_points_max INTEGER")
            add_column(table, "right_points_min", "right_points_min INTEGER")
            add_column(table, "right_points_max", "right_points_max INTEGER")
            add_column(table, "left_points_uncertain", "left_points_uncertain INTEGER DEFAULT 0")
            add_column(table, "right_points_uncertain", "right_points_uncertain INTEGER DEFAULT 0")
            add_column(table, "left_ab_points_min", "left_ab_points_min INTEGER")
            add_column(table, "left_ab_points_max", "left_ab_points_max INTEGER")
            add_column(table, "right_ab_points_min", "right_ab_points_min INTEGER")
            add_column(table, "right_ab_points_max", "right_ab_points_max INTEGER")
            add_column(table, "left_ab_points_uncertain", "left_ab_points_uncertain INTEGER DEFAULT 0")
            add_column(table, "right_ab_points_uncertain", "right_ab_points_uncertain INTEGER DEFAULT 0")
            add_column(table, "abnormal_points_min", "abnormal_points_min INTEGER")
            add_column(table, "abnormal_points_max", "abnormal_points_max INTEGER")
            if table == "deer_additional":
                add_column(table, "age_class", "age_class TEXT")
            add_column(table, "broken_antler_side", "broken_antler_side TEXT")
            add_column(table, "broken_antler_note", "broken_antler_note TEXT")
        # Box-level migration tracking columns on deer_metadata
        add_column("deer_metadata", "deer_id_source", "deer_id_source TEXT")
        add_column("deer_metadata", "primary_box_id", "primary_box_id INTEGER")
        add_column("deer_metadata", "migrated_at", "migrated_at TEXT")
        # Buck profile tables: ensure exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS buck_profiles (
                deer_id TEXT PRIMARY KEY,
                display_name TEXT,
                notes TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS buck_profile_seasons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deer_id TEXT NOT NULL,
                season_year INTEGER NOT NULL,
                camera_locations TEXT,
                key_characteristics TEXT,
                left_points_min INTEGER,
                left_points_uncertain INTEGER DEFAULT 0,
                right_points_min INTEGER,
                right_points_uncertain INTEGER DEFAULT 0,
                left_ab_points_min INTEGER,
                left_ab_points_uncertain INTEGER DEFAULT 0,
                right_ab_points_min INTEGER,
                right_ab_points_uncertain INTEGER DEFAULT 0,
                abnormal_points_min INTEGER,
                abnormal_points_max INTEGER,
                broken_antler_side TEXT,
                broken_antler_note TEXT,
                UNIQUE(deer_id, season_year)
            )
        """)
        # AI rejection tracking for model improvement
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_rejections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                suggestion_type TEXT NOT NULL,
                ai_suggested TEXT NOT NULL,
                correct_label TEXT,
                model_version TEXT,
                rejected_at TEXT NOT NULL,
                FOREIGN KEY (photo_id) REFERENCES photos(id)
            )
        """)
        self.conn.commit()

    def _ensure_box_columns(self):
        """Ensure annotation_boxes table has newer columns."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(annotation_boxes)")
        cols = {row[1] for row in cursor.fetchall()}
        if "confidence" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN confidence REAL")
        # Head direction line columns (for deer pose annotation)
        if "head_x1" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN head_x1 REAL")
        if "head_y1" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN head_y1 REAL")
        if "head_x2" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN head_x2 REAL")
        if "head_y2" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN head_y2 REAL")
        if "head_notes" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN head_notes TEXT")
        # Per-box sex (buck/doe) prediction columns
        if "sex" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN sex TEXT")
        if "sex_conf" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN sex_conf REAL")
        # Per-box AI species suggestion (from SpeciesNet or ONNX classifier)
        if "ai_suggested_species" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN ai_suggested_species TEXT")
        # Per-box confirmed species and confidence
        if "species" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN species TEXT")
        if "species_conf" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN species_conf REAL")
        # Stable sync ID for multi-device conflict resolution
        if "sync_id" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN sync_id TEXT")
            import uuid as _uuid
            cursor.execute("SELECT id FROM annotation_boxes WHERE sync_id IS NULL")
            for row in cursor.fetchall():
                cursor.execute("UPDATE annotation_boxes SET sync_id = ? WHERE id = ?",
                              (str(_uuid.uuid4()), row[0]))
        # Tombstone support for soft-delete sync
        if "deleted_at" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN deleted_at TEXT")
        # Per-box deer metadata columns (deer_id is now box-level, not photo-level)
        if "deer_id" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN deer_id TEXT")
        if "age_class" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN age_class TEXT")
        if "left_points_min" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN left_points_min INTEGER")
        if "right_points_min" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN right_points_min INTEGER")
        if "left_points_uncertain" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN left_points_uncertain INTEGER DEFAULT 0")
        if "right_points_uncertain" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN right_points_uncertain INTEGER DEFAULT 0")
        if "left_ab_points_min" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN left_ab_points_min INTEGER")
        if "right_ab_points_min" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN right_ab_points_min INTEGER")
        if "left_ab_points_uncertain" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN left_ab_points_uncertain INTEGER DEFAULT 0")
        if "right_ab_points_uncertain" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN right_ab_points_uncertain INTEGER DEFAULT 0")
        if "abnormal_points_min" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN abnormal_points_min INTEGER")
        if "abnormal_points_max" not in cols:
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN abnormal_points_max INTEGER")
        # One-time migration: reset old "Unknown" sex defaults to NULL (now means "Unexamined")
        # Before this change, "Unknown" was the auto-checked default — not a deliberate user choice.
        # Only runs once: after migration, no boxes should have sex="Unknown" unless user explicitly sets it.
        cursor.execute("PRAGMA table_info(annotation_boxes)")
        box_cols = {row[1] for row in cursor.fetchall()}
        if "sex_migrated_v2" not in box_cols:
            count = cursor.execute("SELECT COUNT(*) FROM annotation_boxes WHERE sex = 'Unknown'").fetchone()[0]
            if count > 0:
                cursor.execute("UPDATE annotation_boxes SET sex = NULL WHERE sex = 'Unknown'")
                logger.info(f"Migration: reset {count} old 'Unknown' sex defaults to Unexamined (NULL)")
            cursor.execute("ALTER TABLE annotation_boxes ADD COLUMN sex_migrated_v2 INTEGER DEFAULT 1")
        self.conn.commit()

    def _ensure_sync_tracking(self):
        """Add updated_at columns and sync_state table for smart sync."""
        cursor = self.conn.cursor()

        # Add updated_at to tables that need sync tracking
        sync_tables = ['photos', 'tags', 'deer_metadata', 'deer_additional',
                       'buck_profiles', 'buck_profile_seasons', 'annotation_boxes']

        for table in sync_tables:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = {row[1] for row in cursor.fetchall()}
            if "updated_at" not in cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN updated_at TEXT")
                # Set existing records to now so they sync on first push
                cursor.execute(f"UPDATE {table} SET updated_at = datetime('now') WHERE updated_at IS NULL")

        # Create sync_state table to track last sync time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_push_at TEXT,
                last_pull_at TEXT
            )
        """)

        # Initialize sync_state if empty
        cursor.execute("INSERT OR IGNORE INTO sync_state (id) VALUES (1)")

        # Tombstone support for tags (soft-delete for multi-device sync)
        cursor.execute("PRAGMA table_info(tags)")
        tag_cols = {row[1] for row in cursor.fetchall()}
        if "deleted_at" not in tag_cols:
            cursor.execute("ALTER TABLE tags ADD COLUMN deleted_at TEXT")

        # Add schema_version for migration tracking
        cursor.execute("PRAGMA table_info(sync_state)")
        ss_cols = {row[1] for row in cursor.fetchall()}
        if "schema_version" not in ss_cols:
            cursor.execute("ALTER TABLE sync_state ADD COLUMN schema_version INTEGER DEFAULT 1")

        # Create indexes for updated_at columns
        for table in sync_tables:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_updated_at ON {table}(updated_at)")

        # Indexes for tombstone and sync_id columns
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_deleted_at ON tags(deleted_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotation_boxes_sync_id ON annotation_boxes(sync_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotation_boxes_deleted_at ON annotation_boxes(deleted_at)")

        # Create triggers to auto-update updated_at on INSERT and UPDATE
        for table in sync_tables:
            # Trigger for INSERT
            cursor.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_insert_updated_at
                AFTER INSERT ON {table}
                FOR EACH ROW
                WHEN NEW.updated_at IS NULL
                BEGIN
                    UPDATE {table} SET updated_at = datetime('now') WHERE rowid = NEW.rowid;
                END
            """)
            # Trigger for UPDATE
            cursor.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_update_updated_at
                AFTER UPDATE ON {table}
                FOR EACH ROW
                WHEN NEW.updated_at = OLD.updated_at OR NEW.updated_at IS NULL
                BEGIN
                    UPDATE {table} SET updated_at = datetime('now') WHERE rowid = NEW.rowid;
                END
            """)

        self.conn.commit()

    def _migrate_deer_metadata_to_boxes(self):
        """One-time migration: copy deer_metadata to annotation_boxes for single-box photos."""
        import json
        from pathlib import Path

        cursor = self.conn.cursor()

        # Check if migration already done
        cursor.execute("PRAGMA table_info(sync_state)")
        ss_cols = {row[1] for row in cursor.fetchall()}
        if "deer_box_migration_done" not in ss_cols:
            cursor.execute("ALTER TABLE sync_state ADD COLUMN deer_box_migration_done INTEGER DEFAULT 0")
            self.conn.commit()

        cursor.execute("SELECT deer_box_migration_done FROM sync_state WHERE id = 1")
        row = cursor.fetchone()
        if row and row[0]:
            return  # Already migrated

        # Get all photos with deer_metadata that has a deer_id
        cursor.execute("""
            SELECT dm.photo_id, dm.deer_id, dm.age_class,
                   dm.left_points_min, dm.right_points_min,
                   dm.left_points_uncertain, dm.right_points_uncertain,
                   dm.left_ab_points_min, dm.right_ab_points_min,
                   dm.left_ab_points_uncertain, dm.right_ab_points_uncertain,
                   dm.abnormal_points_min, dm.abnormal_points_max
            FROM deer_metadata dm
            WHERE dm.deer_id IS NOT NULL AND dm.deer_id != ''
              AND (dm.migrated_at IS NULL)
        """)
        deer_rows = cursor.fetchall()

        if not deer_rows:
            cursor.execute("UPDATE sync_state SET deer_box_migration_done = 1 WHERE id = 1")
            self.conn.commit()
            return

        review_queue = []
        migrated_count = 0

        for dr in deer_rows:
            photo_id = dr[0]
            deer_id = dr[1]

            # Check for deer_additional records
            cursor.execute("SELECT COUNT(*) FROM deer_additional WHERE photo_id = ?", (photo_id,))
            additional_count = cursor.fetchone()[0]

            # Get non-head subject boxes
            cursor.execute("""
                SELECT id FROM annotation_boxes
                WHERE photo_id = ? AND deleted_at IS NULL
                  AND (label NOT LIKE '%head%' AND label NOT LIKE '%deer_head%')
            """, (photo_id,))
            subject_boxes = cursor.fetchall()

            if len(subject_boxes) == 1 and additional_count == 0:
                # Single box, no additional deer — safe to migrate
                box_id = subject_boxes[0][0]
                # Check if box already has a deer_id (idempotency)
                cursor.execute("SELECT deer_id FROM annotation_boxes WHERE id = ?", (box_id,))
                existing = cursor.fetchone()
                if existing and existing[0]:
                    # Already has deer_id, skip
                    pass
                else:
                    cursor.execute("""
                        UPDATE annotation_boxes SET
                            deer_id=?, age_class=?,
                            left_points_min=?, right_points_min=?,
                            left_points_uncertain=?, right_points_uncertain=?,
                            left_ab_points_min=?, right_ab_points_min=?,
                            left_ab_points_uncertain=?, right_ab_points_uncertain=?,
                            abnormal_points_min=?, abnormal_points_max=?
                        WHERE id = ?
                    """, (dr[1], dr[2], dr[3], dr[4], dr[5], dr[6],
                          dr[7], dr[8], dr[9], dr[10], dr[11], dr[12], box_id))
                # Mark deer_metadata as migrated
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("""
                    UPDATE deer_metadata SET deer_id_source='box', primary_box_id=?, migrated_at=?
                    WHERE photo_id = ?
                """, (box_id, now, photo_id))
                migrated_count += 1
            else:
                # Multiple boxes, additional deer, or no boxes — flag for review
                reason = []
                if len(subject_boxes) != 1:
                    reason.append(f"{len(subject_boxes)} subject boxes")
                if additional_count > 0:
                    reason.append(f"{additional_count} deer_additional records")
                review_queue.append({
                    "photo_id": photo_id,
                    "deer_id": deer_id,
                    "reason": ", ".join(reason),
                })

            # Commit in batches of 100
            if migrated_count % 100 == 0:
                self.conn.commit()

        # Final commit
        self.conn.commit()

        # Write review queue if any
        if review_queue:
            queue_path = Path.home() / ".trailcam" / "deer_id_migration_queue.json"
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(json.dumps(review_queue, indent=2))
            logger.info(f"Deer ID migration: {migrated_count} migrated, {len(review_queue)} need review ({queue_path})")
        else:
            logger.info(f"Deer ID migration: {migrated_count} migrated, 0 need review")

        # Mark done
        cursor.execute("UPDATE sync_state SET deer_box_migration_done = 1 WHERE id = 1")
        self.conn.commit()

    def _ensure_roles_schema(self):
        """Add cameras, permissions, clubs, shares, and label_suggestions tables.

        Supports the camera-owner-based permissions model where:
        - Camera owner = photo owner
        - Owner can share cameras to clubs
        - Owner can delegate co-owner rights
        - Members can suggest labels; only owners approve
        """
        import uuid
        cursor = self.conn.cursor()

        # --- New tables (all idempotent) ---

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                owner TEXT NOT NULL,
                verified INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_permissions (
                id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                granted_by TEXT,
                granted_at TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (camera_id) REFERENCES cameras(id),
                UNIQUE(camera_id, username)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clubs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_by TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS club_memberships (
                id TEXT PRIMARY KEY,
                club_id TEXT NOT NULL,
                username TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (club_id) REFERENCES clubs(id),
                UNIQUE(club_id, username)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_club_shares (
                id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                club_id TEXT NOT NULL,
                shared_by TEXT,
                visibility TEXT DEFAULT 'full',
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (camera_id) REFERENCES cameras(id),
                FOREIGN KEY (club_id) REFERENCES clubs(id),
                UNIQUE(camera_id, club_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS label_suggestions (
                id TEXT PRIMARY KEY,
                photo_id INTEGER NOT NULL,
                file_hash TEXT,
                tag_name TEXT NOT NULL,
                suggested_by TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                reviewed_by TEXT,
                reviewed_at TEXT,
                camera_id TEXT,
                created_at TEXT,
                updated_at TEXT,
                deleted_at TEXT,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                FOREIGN KEY (camera_id) REFERENCES cameras(id),
                UNIQUE(photo_id, tag_name, suggested_by)
            )
        """)

        # --- Add camera_id to photos table ---
        cursor.execute("PRAGMA table_info(photos)")
        photo_cols = {row[1] for row in cursor.fetchall()}
        if "camera_id" not in photo_cols:
            cursor.execute("ALTER TABLE photos ADD COLUMN camera_id TEXT")

        # --- Add created_by to tags table ---
        cursor.execute("PRAGMA table_info(tags)")
        tag_cols = {row[1] for row in cursor.fetchall()}
        if "created_by" not in tag_cols:
            cursor.execute("ALTER TABLE tags ADD COLUMN created_by TEXT")

        # --- Indexes on foreign keys and lookup columns ---
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_camera_id ON photos(camera_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_camera_permissions_camera ON camera_permissions(camera_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_camera_permissions_user ON camera_permissions(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_club_memberships_club ON club_memberships(club_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_club_memberships_user ON club_memberships(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_camera_club_shares_camera ON camera_club_shares(camera_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_camera_club_shares_club ON camera_club_shares(club_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_label_suggestions_photo ON label_suggestions(photo_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_label_suggestions_camera ON label_suggestions(camera_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_label_suggestions_status ON label_suggestions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_label_suggestions_deleted ON label_suggestions(deleted_at)")

        # --- Triggers for updated_at on new sync-able tables ---
        new_sync_tables = ['cameras', 'camera_permissions', 'clubs',
                           'club_memberships', 'camera_club_shares', 'label_suggestions']
        for table in new_sync_tables:
            cursor.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_insert_updated_at
                AFTER INSERT ON {table}
                FOR EACH ROW
                WHEN NEW.updated_at IS NULL
                BEGIN
                    UPDATE {table} SET updated_at = datetime('now') WHERE rowid = NEW.rowid;
                END
            """)
            cursor.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_update_updated_at
                AFTER UPDATE ON {table}
                FOR EACH ROW
                WHEN NEW.updated_at = OLD.updated_at OR NEW.updated_at IS NULL
                BEGIN
                    UPDATE {table} SET updated_at = datetime('now') WHERE rowid = NEW.rowid;
                END
            """)

        # --- Data migration: auto-create cameras from existing sites ---
        cursor.execute("SELECT COUNT(*) FROM cameras")
        camera_count = cursor.fetchone()[0]

        if camera_count == 0:
            # First time: create cameras from sites table
            from user_config import get_username
            current_user = get_username() or "unknown"
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            # Get all sites
            cursor.execute("SELECT id, name FROM sites")
            sites = cursor.fetchall()

            # Map site_id -> camera_id for photo assignment
            site_to_camera = {}

            for site_id, site_name in sites:
                camera_id = str(uuid.uuid4())
                camera_name = f"Unverified: {site_name}" if site_name else "Unverified: Unknown"
                cursor.execute(
                    "INSERT INTO cameras (id, name, owner, verified, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?)",
                    (camera_id, camera_name, current_user, now, now))
                # Owner permission
                perm_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO camera_permissions (id, camera_id, username, role, granted_by, granted_at, created_at, updated_at) VALUES (?, ?, ?, 'owner', ?, ?, ?, ?)",
                    (perm_id, camera_id, current_user, current_user, now, now, now))
                site_to_camera[site_id] = camera_id

            # Create a fallback camera for photos with no site_id
            fallback_camera_id = None
            cursor.execute("SELECT COUNT(*) FROM photos WHERE site_id IS NULL OR site_id = 0")
            no_site_count = cursor.fetchone()[0]
            if no_site_count > 0:
                fallback_camera_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO cameras (id, name, owner, verified, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?)",
                    (fallback_camera_id, "Unverified: Unknown", current_user, now, now))
                perm_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO camera_permissions (id, camera_id, username, role, granted_by, granted_at, created_at, updated_at) VALUES (?, ?, ?, 'owner', ?, ?, ?, ?)",
                    (perm_id, fallback_camera_id, current_user, current_user, now, now, now))

            # Assign photos to cameras based on site_id
            for site_id, camera_id in site_to_camera.items():
                cursor.execute("UPDATE photos SET camera_id = ? WHERE site_id = ?", (camera_id, site_id))

            # Assign photos with no site to fallback
            if fallback_camera_id:
                cursor.execute("UPDATE photos SET camera_id = ? WHERE camera_id IS NULL", (fallback_camera_id,))

            migrated = sum(1 for _ in site_to_camera.values()) + (1 if fallback_camera_id else 0)
            if migrated > 0:
                logger.info(f"Roles migration: created {migrated} cameras from {len(sites)} sites, assigned all photos")

        self.conn.commit()

    def _normalize_ts(self, ts):
        """Normalize timestamp for comparison: 'YYYY-MM-DD HH:MM:SS' format.

        Handles both SQLite (YYYY-MM-DD HH:MM:SS) and Supabase
        (YYYY-MM-DDTHH:MM:SS.ssssss+00:00) formats.
        """
        if not ts:
            return ""
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        s = s.replace('T', ' ')
        s = s.split('.')[0]
        s = re.sub(r'([+-]\d{2}:\d{2})$', '', s)
        return s

    def purge_old_tombstones(self, days=30):
        """Hard-delete tombstoned tags and boxes older than `days` days.

        Call after a successful sync to prevent tombstone buildup.
        """
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM tags WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,))
            tags_purged = cursor.rowcount
            cursor.execute("DELETE FROM annotation_boxes WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,))
            boxes_purged = cursor.rowcount
            self.conn.commit()
        if tags_purged or boxes_purged:
            logger.info(f"Tombstone GC: purged {tags_purged} tags, {boxes_purged} boxes older than {days} days")

    def get_last_sync_time(self, sync_type='push') -> Optional[str]:
        """Get the last sync timestamp."""
        cursor = self.conn.cursor()
        col = 'last_push_at' if sync_type == 'push' else 'last_pull_at'
        cursor.execute(f"SELECT {col} FROM sync_state WHERE id = 1")
        row = cursor.fetchone()
        return row[0] if row else None

    def set_last_sync_time(self, sync_type='push', timestamp=None):
        """Set the last sync timestamp."""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        cursor = self.conn.cursor()
        col = 'last_push_at' if sync_type == 'push' else 'last_pull_at'
        cursor.execute(f"UPDATE sync_state SET {col} = ? WHERE id = 1", (timestamp,))
        self.conn.commit()

    def mark_record_updated(self, table: str, record_id: int):
        """Mark a record as updated for sync tracking."""
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE {table} SET updated_at = datetime('now') WHERE id = ?", (record_id,))
        self.conn.commit()

    @staticmethod
    def compute_season_year(date_str: Optional[str]) -> Optional[int]:
        """Map calendar date to hunting season year (May–Apr)."""
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str)
        except Exception:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                return None
        if dt.month >= 5:
            return dt.year
        return dt.year - 1

    @staticmethod
    def format_season_label(season_year: Optional[int]) -> str:
        """Return display label like '2025-2026' for a given season start year."""
        if season_year is None:
            return ""
        return f"{season_year}-{season_year + 1}"

    def _backfill_season_years(self):
        """Populate season_year for rows that don't have it yet."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, date_taken FROM photos WHERE season_year IS NULL OR season_year = ''")
        rows = cursor.fetchall()
        updated = []
        for row in rows:
            season = self.compute_season_year(row["date_taken"])
            if season is not None:
                updated.append((season, row["id"]))
        if updated:
            cursor.executemany("UPDATE photos SET season_year = ? WHERE id = ?", updated)
            self.conn.commit()
    
    def add_photo(self, file_path: str, original_name: str, date_taken: str,
                  camera_model: str, thumbnail_path: str = None,
                  favorite: int = 0, notes: str = "", season_year: Optional[int] = None,
                  camera_location: str = "", key_characteristics: str = "",
                  suggested_tag: str = "", suggested_confidence: Optional[float] = None,
                  collection: str = "", file_hash: str = None) -> Optional[int]:
        """Add a photo to the database.

        Args:
            file_hash: MD5 hash of the file. If provided and a photo with same hash exists,
                      the insert will be skipped and None returned.

        Returns:
            Photo ID, or None if photo already exists (duplicate hash)
        """
        # Check for duplicate by file_hash (prevents importing same photo twice)
        if file_hash and self.photo_exists_by_hash(file_hash):
            return None

        if season_year is None:
            season_year = self.compute_season_year(date_taken)
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO photos
                (file_path, original_name, date_taken, camera_model, import_date, thumbnail_path, favorite, notes, season_year, camera_location, key_characteristics, suggested_tag, suggested_confidence, collection, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_path, original_name, date_taken, camera_model,
                  datetime.now().isoformat(), thumbnail_path, favorite, notes, season_year, camera_location, key_characteristics, suggested_tag, suggested_confidence, collection, file_hash))
            photo_id = cursor.lastrowid
            self.conn.commit()
            return photo_id
    
    def get_photo_id(self, file_path: str) -> Optional[int]:
        """Get photo ID by file path."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM photos WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        return row['id'] if row else None
    
    def get_photo_by_id(self, photo_id: int) -> Optional[Dict]:
        """Get photo information by ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, file_path, original_name, date_taken, camera_model, thumbnail_path, favorite, notes, season_year, camera_location, key_characteristics, suggested_tag, suggested_confidence, suggested_sex, suggested_sex_confidence
            FROM photos WHERE id = ?
        """, (photo_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def photo_exists_by_original_name(self, original_name: str) -> bool:
        """Check if a photo with this original filename already exists."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM photos WHERE original_name = ? LIMIT 1", (original_name,))
        return cursor.fetchone() is not None

    def photo_exists_by_hash(self, file_hash: str) -> bool:
        """Check if a photo with this file_hash already exists.

        This prevents importing duplicate photos even if they have different filenames.
        Thread-safe for use from worker threads.
        """
        if not file_hash:
            return False
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM photos WHERE file_hash = ? LIMIT 1", (file_hash,))
            return cursor.fetchone() is not None

    def get_photo_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get photo by its file hash. Thread-safe."""
        if not file_hash:
            return None
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM photos WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_file_hashes(self) -> set:
        """Get all file_hashes from local database. Thread-safe.

        Returns:
            Set of file_hash strings
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_hash FROM photos WHERE file_hash IS NOT NULL AND file_hash != ''")
            return {row['file_hash'] for row in cursor.fetchall()}

    def add_cloud_photo(self, cloud_record: Dict) -> Optional[int]:
        """Add a photo record from cloud data (for photos that only exist in cloud).

        The file_path will be set to 'cloud://{file_hash}' to indicate it's cloud-only.
        The thumbnail will be downloaded separately.

        Args:
            cloud_record: Photo record from Supabase photos_sync table

        Returns:
            Photo ID if inserted, None if already exists
        """
        file_hash = cloud_record.get("file_hash")
        if not file_hash:
            return None

        # Check if already exists
        if self.photo_exists_by_hash(file_hash):
            return None

        # Use cloud:// prefix to indicate this photo only exists in cloud
        file_path = f"cloud://{file_hash}"

        # Extract fields from cloud record
        original_name = cloud_record.get("original_name", "")
        date_taken = cloud_record.get("date_taken", "")
        camera_model = cloud_record.get("camera_model", "")
        camera_location = cloud_record.get("camera_location", "")
        favorite = cloud_record.get("favorite", 0)
        notes = cloud_record.get("notes", "")
        collection = cloud_record.get("collection", "")

        # Compute season year
        season_year = self.compute_season_year(date_taken) if date_taken else None

        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO photos
                (file_path, original_name, date_taken, camera_model, import_date,
                 thumbnail_path, favorite, notes, season_year, camera_location,
                 collection, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_path, original_name, date_taken, camera_model,
                  datetime.now().isoformat(), None, favorite, notes,
                  season_year, camera_location, collection, file_hash))
            photo_id = cursor.lastrowid
            self.conn.commit()
            return photo_id

    def update_thumbnail_path(self, photo_id: int, thumbnail_path: str):
        """Update the thumbnail_path for a photo."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET thumbnail_path = ? WHERE id = ?",
                          (thumbnail_path, photo_id))
            self.conn.commit()

    def get_photos_by_ids(self, photo_ids: List[int]) -> List[Dict]:
        """Get multiple photos by ID in one query. Thread-safe."""
        if not photo_ids:
            return []
        with self._lock:
            placeholders = ",".join(["?"] * len(photo_ids))
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT * FROM photos WHERE id IN ({placeholders})", photo_ids)
            return [dict(row) for row in cursor.fetchall()]

    def update_file_path(self, photo_id: int, file_path: str):
        """Update the file_path for a photo (used after downloading from cloud)."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET file_path = ? WHERE id = ?",
                          (file_path, photo_id))
            self.conn.commit()

    def update_date_taken(self, photo_id: int, date_taken: str):
        """Update the date_taken for a photo (thread-safe for worker threads)."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET date_taken = ? WHERE id = ?",
                          (date_taken, photo_id))
            self.conn.commit()

    def get_cloud_only_photos(self) -> List[Dict]:
        """Get all photos that only exist in cloud (file_path starts with 'cloud://').
        Thread-safe.

        Returns:
            List of photo dicts
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM photos WHERE file_path LIKE 'cloud://%'")
            return [dict(row) for row in cursor.fetchall()]

    def get_photo_path(self, photo_id: int) -> Optional[str]:
        """Get file path for a photo ID. Thread-safe."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path FROM photos WHERE id = ?", (photo_id,))
            row = cursor.fetchone()
        return row['file_path'] if row else None

    def get_photo_by_path(self, file_path: str) -> Optional[Dict]:
        """Get photo information by file path."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, file_path, original_name, date_taken, camera_model, thumbnail_path, favorite, notes, season_year, camera_location, key_characteristics, suggested_tag, suggested_confidence, suggested_sex, suggested_sex_confidence
            FROM photos WHERE file_path = ?
        """, (file_path,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def set_camera_model(self, photo_id: int, camera_model: str):
        """Update camera_model for a photo."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET camera_model = ? WHERE id = ?", (camera_model or "", photo_id))
            self.conn.commit()

    def set_date_taken(self, photo_id: int, date_taken: Optional[str]):
        """Update date_taken for a photo."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET date_taken = ? WHERE id = ?", (date_taken, photo_id))
            self.conn.commit()
    
    def add_tag(self, photo_id: int, tag_name: str):
        """Add a tag to a photo. Un-tombstones if previously soft-deleted."""
        with self._lock:
            cursor = self.conn.cursor()
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            # Check if tombstoned version exists — revive it
            cursor.execute("SELECT id FROM tags WHERE photo_id = ? AND tag_name = ? AND deleted_at IS NOT NULL",
                          (photo_id, tag_name))
            tombstoned = cursor.fetchone()
            if tombstoned:
                cursor.execute("UPDATE tags SET deleted_at = NULL, updated_at = ? WHERE id = ?",
                              (now, tombstoned[0]))
                self.conn.commit()
                self._log_tag_change('add', photo_id, tag_name)
                return
            try:
                cursor.execute("INSERT INTO tags (photo_id, tag_name, updated_at) VALUES (?, ?, ?)",
                             (photo_id, tag_name, now))
                self.conn.commit()
                self._log_tag_change('add', photo_id, tag_name)
            except sqlite3.IntegrityError:
                # Tag already exists and is active, ignore
                pass
    
    def update_photo_tags(self, photo_id: int, tags: List[str]):
        """Replace all tags for a photo with the provided list.

        Uses differential soft-delete for multi-device sync safety:
        removed tags get tombstoned, new tags are inserted or un-tombstoned.

        If tags include 'Review' or 'Verification', automatically sets that
        as the photo-level suggested_tag for visibility in filters/reports.
        """
        with self._lock:
            cursor = self.conn.cursor()
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            # Get current active (non-tombstoned) tags
            cursor.execute("SELECT id, tag_name FROM tags WHERE photo_id = ? AND deleted_at IS NULL", (photo_id,))
            active = {row[1]: row[0] for row in cursor.fetchall()}

            # Deduplicate incoming tags
            unique_tags = list(dict.fromkeys(tags))
            new_tags = set(unique_tags)
            old_tags = set(active.keys())

            # Soft-delete removed tags
            for tag in old_tags - new_tags:
                cursor.execute("UPDATE tags SET deleted_at = ?, updated_at = ? WHERE id = ?",
                              (now, now, active[tag]))

            # Un-tombstone or insert added tags
            for tag in new_tags - old_tags:
                # Check if there's a tombstoned version we can revive
                cursor.execute("SELECT id FROM tags WHERE photo_id = ? AND tag_name = ? AND deleted_at IS NOT NULL",
                              (photo_id, tag))
                tombstoned = cursor.fetchone()
                if tombstoned:
                    cursor.execute("UPDATE tags SET deleted_at = NULL, updated_at = ? WHERE id = ?",
                                  (now, tombstoned[0]))
                else:
                    cursor.execute("INSERT INTO tags (photo_id, tag_name, updated_at) VALUES (?, ?, ?)",
                                  (photo_id, tag, now))

            # Auto-sync verification tags to photo level for visibility
            review_tags = {'Review', 'Verification'}
            found_review = next((t for t in unique_tags if t in review_tags), None)
            if found_review:
                cursor.execute("""
                    UPDATE photos SET suggested_tag = ?, suggested_confidence = NULL
                    WHERE id = ?
                """, (found_review, photo_id))

            self.conn.commit()

            # Log tag changes to audit trail
            for tag in old_tags - new_tags:
                self._log_tag_change('remove', photo_id, tag)
            for tag in new_tags - old_tags:
                self._log_tag_change('add', photo_id, tag)
    
    def get_photo_tags(self, photo_id: int) -> List[str]:
        """Alias for get_tags; returns all tags for a photo."""
        return self.get_tags(photo_id)
    
    def remove_tag(self, photo_id: int, tag_name: str):
        """Remove a tag from a photo (soft-delete for sync). Thread-safe."""
        with self._lock:
            cursor = self.conn.cursor()
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "UPDATE tags SET deleted_at = ?, updated_at = ? WHERE photo_id = ? AND tag_name = ? AND deleted_at IS NULL",
                (now, now, photo_id, tag_name))
            self.conn.commit()
        # Log to audit trail
        self._log_tag_change('remove', photo_id, tag_name)
    
    def get_tags(self, photo_id: int) -> List[str]:
        """Get all active (non-tombstoned) tags for a photo. Thread-safe."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT tag_name FROM tags WHERE photo_id = ? AND deleted_at IS NULL ORDER BY tag_name",
                          (photo_id,))
            return [row['tag_name'] for row in cursor.fetchall()]

    def get_all_tags_batch(self, photo_ids: List[int]) -> Dict[int, List[str]]:
        """Get tags for many photos in one query. Thread-safe."""
        if not photo_ids:
            return {}
        with self._lock:
            cursor = self.conn.cursor()
            placeholders = ",".join(["?"] * len(photo_ids))
            cursor.execute(
                f"SELECT photo_id, tag_name FROM tags "
                f"WHERE photo_id IN ({placeholders}) AND deleted_at IS NULL",
                photo_ids
            )
            tags_map: Dict[int, List[str]] = {}
            for row in cursor.fetchall():
                pid = row["photo_id"]
                tags_map.setdefault(pid, []).append(row["tag_name"])
            return tags_map

    def get_deer_sex_batch(self, photo_ids: List[int]) -> Dict[int, str]:
        """Get the confirmed sex for deer boxes across many photos in one query.

        Returns photo_id -> sex mapping. For photos with multiple deer boxes,
        prefers definitive sex (Buck/Doe) over Unknown/empty.
        Only includes deer boxes (species='deer' or unspecified ai_animal boxes).
        """
        if not photo_ids:
            return {}
        with self._lock:
            cursor = self.conn.cursor()
            placeholders = ",".join(["?"] * len(photo_ids))
            cursor.execute(
                f"SELECT photo_id, sex FROM annotation_boxes "
                f"WHERE photo_id IN ({placeholders}) "
                f"AND (deleted_at IS NULL) "
                f"AND (LOWER(species) = 'deer' OR (species IS NULL AND label = 'ai_animal')) "
                f"AND sex IS NOT NULL AND sex != ''",
                photo_ids
            )
            result: Dict[int, str] = {}
            for row in cursor.fetchall():
                pid = row["photo_id"]
                sex = row["sex"]
                # Prefer definitive (Buck/Doe) over Unknown
                if pid not in result or sex in ("Buck", "Doe"):
                    result[pid] = sex
            return result

    def get_all_distinct_tags(self) -> List[str]:
        """Get all distinct tag names across all photos. Thread-safe."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT tag_name FROM tags WHERE deleted_at IS NULL ORDER BY tag_name")
            return [row['tag_name'] for row in cursor.fetchall()]

    def set_deer_metadata(self, photo_id: int, deer_id: str = None, age_class: str = None,
                          left_points_min: Optional[int] = None, right_points_min: Optional[int] = None,
                          left_points_uncertain: bool = False, right_points_uncertain: bool = False,
                          left_ab_points_min: Optional[int] = None, right_ab_points_min: Optional[int] = None,
                          left_ab_points_uncertain: bool = False, right_ab_points_uncertain: bool = False,
                          abnormal_points_min: Optional[int] = None, abnormal_points_max: Optional[int] = None,
                          broken_antler_side: Optional[str] = None, broken_antler_note: Optional[str] = None):
        """Set deer metadata for a photo. Thread-safe."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO deer_metadata (photo_id, deer_id, age_class,
                    left_points_min, left_points_max, right_points_min, right_points_max,
                    left_points_uncertain, right_points_uncertain,
                    left_ab_points_min, right_ab_points_min,
                    left_ab_points_uncertain, right_ab_points_uncertain,
                    abnormal_points_min, abnormal_points_max,
                    broken_antler_side, broken_antler_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (photo_id, deer_id, age_class, left_points_min, None, right_points_min, None,
                  1 if left_points_uncertain else 0, 1 if right_points_uncertain else 0,
                  left_ab_points_min, right_ab_points_min,
                  1 if left_ab_points_uncertain else 0, 1 if right_ab_points_uncertain else 0,
                  abnormal_points_min, abnormal_points_max, broken_antler_side, broken_antler_note))
            self.conn.commit()

    # Additional deer helpers
    def add_additional_deer(self, photo_id: int, deer_id: str, age_class: str = None,
                            left_points_min: Optional[int] = None, right_points_min: Optional[int] = None,
                            left_points_uncertain: bool = False, right_points_uncertain: bool = False,
                            left_ab_points_min: Optional[int] = None, right_ab_points_min: Optional[int] = None,
                            left_ab_points_uncertain: bool = False, right_ab_points_uncertain: bool = False,
                            abnormal_points_min: Optional[int] = None, abnormal_points_max: Optional[int] = None):
        """Add another deer entry to a photo."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO deer_additional (photo_id, deer_id, age_class,
                    left_points_min, left_points_max, right_points_min, right_points_max,
                    left_points_uncertain, right_points_uncertain,
                    left_ab_points_min, right_ab_points_min,
                    left_ab_points_uncertain, right_ab_points_uncertain,
                    abnormal_points_min, abnormal_points_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (photo_id, deer_id, age_class, left_points_min, None, right_points_min, None,
                  1 if left_points_uncertain else 0, 1 if right_points_uncertain else 0,
                  left_ab_points_min, right_ab_points_min,
                  1 if left_ab_points_uncertain else 0, 1 if right_ab_points_uncertain else 0,
                  abnormal_points_min, abnormal_points_max))
            self.conn.commit()

    def remove_additional_deer(self, photo_id: int, deer_id: str):
        """Remove an additional deer entry."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM deer_additional WHERE photo_id = ? AND deer_id = ?", (photo_id, deer_id))
        self.conn.commit()

    def get_additional_deer(self, photo_id: int) -> List[Dict[str, Optional[str]]]:
        """Get list of additional deer for a photo."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT deer_id, age_class, left_points_min, right_points_min,
                   left_points_uncertain, right_points_uncertain,
                   left_ab_points_min, right_ab_points_min,
                   left_ab_points_uncertain, right_ab_points_uncertain,
                   abnormal_points_min, abnormal_points_max
            FROM deer_additional WHERE photo_id = ? ORDER BY deer_id
        """, (photo_id,))
        rows = []
        for row in cursor.fetchall():
            rows.append({
                "deer_id": row["deer_id"],
                "age_class": row["age_class"],
                "left_points_min": row["left_points_min"],
                "right_points_min": row["right_points_min"],
                "left_points_uncertain": bool(row["left_points_uncertain"]) if row["left_points_uncertain"] is not None else False,
                "right_points_uncertain": bool(row["right_points_uncertain"]) if row["right_points_uncertain"] is not None else False,
                "left_ab_points_min": row["left_ab_points_min"],
                "right_ab_points_min": row["right_ab_points_min"],
                "left_ab_points_uncertain": bool(row["left_ab_points_uncertain"]) if row["left_ab_points_uncertain"] is not None else False,
                "right_ab_points_uncertain": bool(row["right_ab_points_uncertain"]) if row["right_ab_points_uncertain"] is not None else False,
                "abnormal_points_min": row["abnormal_points_min"],
                "abnormal_points_max": row["abnormal_points_max"],
            })
        return rows

    # Buck profiles
    def get_buck_profile(self, deer_id: str, season_year: Optional[int]) -> Dict:
        """Return buck profile for a given deer and season."""
        if not deer_id or season_year is None:
            return {}
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT deer_id, season_year, camera_locations, key_characteristics,
                   left_points_min, left_points_uncertain,
                   right_points_min, right_points_uncertain,
                   left_ab_points_min, left_ab_points_uncertain,
                   right_ab_points_min, right_ab_points_uncertain,
                   abnormal_points_min, abnormal_points_max,
                   broken_antler_side, broken_antler_note
            FROM buck_profile_seasons
            WHERE deer_id = ? AND season_year = ?
        """, (deer_id, season_year))
        row = cursor.fetchone()
        if not row:
            return {}
        return dict(row)

    def get_buck_season_summaries(self, deer_id: str) -> List[Dict]:
        """
        Return per-season summaries for a buck: season_year, photo_count, key characteristics, and antler stats.
        Stats are aggregated from deer_metadata joined to photos for that buck_id.
        """
        if not deer_id:
            return []
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.season_year,
                   COUNT(*) as photo_count,
                   MAX(dm.left_points_min) as left_points_min,
                   MAX(dm.right_points_min) as right_points_min,
                   MAX(dm.left_ab_points_min) as left_ab_points_min,
                   MAX(dm.right_ab_points_min) as right_ab_points_min,
                   MAX(dm.abnormal_points_min) as abnormal_points_min,
                   MAX(dm.abnormal_points_max) as abnormal_points_max,
                   GROUP_CONCAT(DISTINCT p.camera_location) as cameras,
                   GROUP_CONCAT(DISTINCT p.key_characteristics) as key_chars
            FROM photos p
            JOIN deer_metadata dm ON dm.photo_id = p.id
            WHERE dm.deer_id = ?
            GROUP BY p.season_year
            ORDER BY p.season_year
        """, (deer_id,))
        rows = []
        for row in cursor.fetchall():
            rows.append({
                "season_year": row["season_year"],
                "photo_count": row["photo_count"],
                "left_points_min": row["left_points_min"],
                "right_points_min": row["right_points_min"],
                "left_ab_points_min": row["left_ab_points_min"],
                "right_ab_points_min": row["right_ab_points_min"],
                "abnormal_points_min": row["abnormal_points_min"],
                "abnormal_points_max": row["abnormal_points_max"],
                "camera_locations": row["cameras"] or "",
                "key_characteristics": row["key_chars"] or "",
            })
        return rows

    def get_buck_photo_history(self, deer_id: str) -> List[Dict]:
        """Return all photos for a buck across years with date/location."""
        if not deer_id:
            return []
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.id, p.file_path, p.date_taken, p.camera_location, p.season_year
            FROM photos p
            JOIN deer_metadata dm ON dm.photo_id = p.id
            WHERE dm.deer_id = ?
            ORDER BY p.date_taken ASC
        """, (deer_id,))
        rows = []
        for row in cursor.fetchall():
            rows.append(dict(row))
        return rows

    def get_buck_encounters(self, deer_id: str, window_minutes: int = 30) -> List[Dict]:
        """Group photos into encounters by camera and time proximity."""
        if not deer_id:
            return []
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.id, p.file_path, p.date_taken, p.camera_location, p.season_year
            FROM photos p
            JOIN deer_metadata dm ON dm.photo_id = p.id
            WHERE dm.deer_id = ?
              AND p.date_taken IS NOT NULL
            ORDER BY p.camera_location, p.date_taken ASC
        """, (deer_id,))
        rows = cursor.fetchall()
        encounters = []
        current = None
        from datetime import datetime, timedelta

        def parse_dt(val: str):
            if not val:
                return None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(val, fmt)
                except Exception:
                    continue
            try:
                return datetime.fromisoformat(val)
            except Exception:
                return None

        for row in rows:
            dt = parse_dt(row["date_taken"])
            if dt is None:
                continue
            cam = row["camera_location"] or ""
            if current and current["camera_location"] == cam and dt - current["end_dt"] <= timedelta(minutes=window_minutes):
                current["end_dt"] = dt
                current["photo_ids"].append(row["id"])
            else:
                if current:
                    encounters.append(current)
                current = {
                    "camera_location": cam,
                    "start_dt": dt,
                    "end_dt": dt,
                    "photo_ids": [row["id"]],
                }
        if current:
            encounters.append(current)

        # Convert to display-friendly dicts
        formatted = []
        for enc in encounters:
            formatted.append({
                "camera_location": enc["camera_location"],
                "start": enc["start_dt"].isoformat(sep=" "),
                "end": enc["end_dt"].isoformat(sep=" "),
                "count": len(enc["photo_ids"]),
                "photo_ids": enc["photo_ids"],
            })
        return formatted

    def update_buck_profile_from_photo(self, deer_id: Optional[str], season_year: Optional[int], photo_row: Dict, deer_meta: Dict):
        """Upsert buck season profile using data from a photo + deer metadata."""
        if not deer_id or season_year is None:
            return
        # Merge tags and camera locations
        cam_loc = (photo_row.get("camera_location") or "").strip()
        key_chars = (photo_row.get("key_characteristics") or "").strip()
        existing = self.get_buck_profile(deer_id, season_year)
        def merge_tags(old: str, new: str) -> str:
            vals = []
            for src in [old or "", new or ""]:
                for tok in src.replace("\n", ",").split(","):
                    t = tok.strip()
                    if t:
                        vals.append(t)
            seen = set()
            out = []
            for t in vals:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return ", ".join(out)
        merged_cam = merge_tags(existing.get("camera_locations") if existing else "", cam_loc)
        merged_chars = merge_tags(existing.get("key_characteristics") if existing else "", key_chars)
        payload = (
            deer_id,
            season_year,
            merged_cam,
            merged_chars,
            deer_meta.get("left_points_min"),
            1 if deer_meta.get("left_points_uncertain") else 0,
            deer_meta.get("right_points_min"),
            1 if deer_meta.get("right_points_uncertain") else 0,
            deer_meta.get("left_ab_points_min"),
            1 if deer_meta.get("left_ab_points_uncertain") else 0,
            deer_meta.get("right_ab_points_min"),
            1 if deer_meta.get("right_ab_points_uncertain") else 0,
            deer_meta.get("abnormal_points_min"),
            deer_meta.get("abnormal_points_max"),
            deer_meta.get("broken_antler_side"),
            deer_meta.get("broken_antler_note"),
        )
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO buck_profile_seasons (
                deer_id, season_year, camera_locations, key_characteristics,
                left_points_min, left_points_uncertain,
                right_points_min, right_points_uncertain,
                left_ab_points_min, left_ab_points_uncertain,
                right_ab_points_min, right_ab_points_uncertain,
                abnormal_points_min, abnormal_points_max,
                broken_antler_side, broken_antler_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(deer_id, season_year) DO UPDATE SET
                camera_locations=excluded.camera_locations,
                key_characteristics=excluded.key_characteristics,
                left_points_min=excluded.left_points_min,
                left_points_uncertain=excluded.left_points_uncertain,
                right_points_min=excluded.right_points_min,
                right_points_uncertain=excluded.right_points_uncertain,
                left_ab_points_min=excluded.left_ab_points_min,
                left_ab_points_uncertain=excluded.left_ab_points_uncertain,
                right_ab_points_min=excluded.right_ab_points_min,
                right_ab_points_uncertain=excluded.right_ab_points_uncertain,
                abnormal_points_min=excluded.abnormal_points_min,
                abnormal_points_max=excluded.abnormal_points_max,
                broken_antler_side=excluded.broken_antler_side,
                broken_antler_note=excluded.broken_antler_note
        """, payload)
        self.conn.commit()

    def apply_profile_to_season_photos(self, deer_id: str, season_year: Optional[int]):
        """Apply buck season profile to all photos of that buck in the same season."""
        if not deer_id or season_year is None:
            return 0
        profile = self.get_buck_profile(deer_id, season_year)
        if not profile:
            return 0
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.id FROM photos p
            JOIN deer_metadata dm ON dm.photo_id = p.id
            WHERE dm.deer_id = ? AND p.season_year = ?
        """, (deer_id, season_year))
        ids = [r[0] for r in cursor.fetchall()]
        for pid in ids:
            self.set_deer_metadata(
                photo_id=pid,
                deer_id=deer_id,
                age_class=None,
                left_points_min=profile.get("left_points_min"),
                right_points_min=profile.get("right_points_min"),
                left_points_uncertain=bool(profile.get("left_points_uncertain")),
                right_points_uncertain=bool(profile.get("right_points_uncertain")),
                left_ab_points_min=profile.get("left_ab_points_min"),
                right_ab_points_min=profile.get("right_ab_points_min"),
                left_ab_points_uncertain=bool(profile.get("left_ab_points_uncertain")),
                right_ab_points_uncertain=bool(profile.get("right_ab_points_uncertain")),
                abnormal_points_min=profile.get("abnormal_points_min"),
                abnormal_points_max=profile.get("abnormal_points_max"),
                broken_antler_side=profile.get("broken_antler_side"),
                broken_antler_note=profile.get("broken_antler_note"),
            )
            if profile.get("key_characteristics"):
                photo = self.get_photo_by_id(pid) or {}
                self.update_photo_attributes(
                    photo_id=pid,
                    camera_location=photo.get("camera_location") or "",
                    key_characteristics=profile.get("key_characteristics") or "",
                )
        return len(ids)

    def merge_deer_ids(self, source_id: str, target_id: str) -> int:
        """Merge source deer_id into target across metadata and profiles. Returns affected photo count."""
        if not source_id or not target_id or source_id == target_id:
            return 0
        cursor = self.conn.cursor()
        # Update primary metadata
        cursor.execute("UPDATE deer_metadata SET deer_id = ? WHERE deer_id = ?", (target_id, source_id))
        affected = cursor.rowcount
        # Update additional deer; handle conflicts by removing duplicates
        cursor.execute("""
            DELETE FROM deer_additional
            WHERE deer_id = ?
              AND EXISTS (SELECT 1 FROM deer_additional d2 WHERE d2.photo_id = deer_additional.photo_id AND d2.deer_id = ?)
        """, (source_id, target_id))
        cursor.execute("UPDATE deer_additional SET deer_id = ? WHERE deer_id = ?", (target_id, source_id))
        # Update box-level deer IDs
        cursor.execute("UPDATE annotation_boxes SET deer_id = ? WHERE deer_id = ? AND deleted_at IS NULL",
                       (target_id, source_id))
        # Move profiles
        cursor.execute("UPDATE OR IGNORE buck_profiles SET deer_id = ? WHERE deer_id = ?", (target_id, source_id))
        cursor.execute("UPDATE OR IGNORE buck_profile_seasons SET deer_id = ? WHERE deer_id = ?", (target_id, source_id))
        cursor.execute("DELETE FROM buck_profiles WHERE deer_id = ?", (source_id,))
        cursor.execute("DELETE FROM buck_profile_seasons WHERE deer_id = ?", (source_id,))
        self.conn.commit()
        return affected

    def rename_deer_id(self, old_id: str, new_id: str, season_years: List[int] = None) -> int:
        """Rename a deer ID, optionally only for specific seasons.

        Args:
            old_id: Current deer ID
            new_id: New deer ID
            season_years: If provided, only rename for photos in these seasons.
                         If None, rename all photos with this deer ID.

        Returns:
            Number of photos affected
        """
        if not old_id or not new_id or old_id == new_id:
            return 0

        cursor = self.conn.cursor()
        affected = 0

        if season_years:
            # Get photo IDs for the specified seasons
            placeholders = ','.join('?' * len(season_years))
            cursor.execute(f"""
                SELECT DISTINCT dm.photo_id FROM deer_metadata dm
                JOIN photos p ON dm.photo_id = p.id
                WHERE dm.deer_id = ? AND p.season_year IN ({placeholders})
            """, [old_id] + season_years)
            photo_ids = [row[0] for row in cursor.fetchall()]

            if photo_ids:
                placeholders = ','.join('?' * len(photo_ids))
                cursor.execute(f"""
                    UPDATE deer_metadata SET deer_id = ?
                    WHERE deer_id = ? AND photo_id IN ({placeholders})
                """, [new_id, old_id] + photo_ids)
                affected = cursor.rowcount

                # Update additional deer too
                cursor.execute(f"""
                    UPDATE deer_additional SET deer_id = ?
                    WHERE deer_id = ? AND photo_id IN ({placeholders})
                """, [new_id, old_id] + photo_ids)

                # Update box-level deer IDs for these photos
                cursor.execute(f"""
                    UPDATE annotation_boxes SET deer_id = ?
                    WHERE deer_id = ? AND photo_id IN ({placeholders}) AND deleted_at IS NULL
                """, [new_id, old_id] + photo_ids)

            # Update buck_profile_seasons for specified seasons
            for sy in season_years:
                cursor.execute("""
                    UPDATE OR IGNORE buck_profile_seasons SET deer_id = ?
                    WHERE deer_id = ? AND season_year = ?
                """, (new_id, old_id, sy))
                cursor.execute("""
                    DELETE FROM buck_profile_seasons
                    WHERE deer_id = ? AND season_year = ?
                """, (old_id, sy))
        else:
            # Rename all - use merge logic
            affected = self.merge_deer_ids(old_id, new_id)

        self.conn.commit()
        return affected

    def set_additional_deer(self, photo_id: int, deer_entries: List[Dict[str, Optional[Union[str, int, bool]]]]):
        """Replace additional deer list with provided entries."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM deer_additional WHERE photo_id = ?", (photo_id,))
        if deer_entries:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO deer_additional
                (photo_id, deer_id, age_class, left_points_min, left_points_max, right_points_min, right_points_max,
                 left_points_uncertain, right_points_uncertain,
                 left_ab_points_min, right_ab_points_min,
                 left_ab_points_uncertain, right_ab_points_uncertain,
                 abnormal_points_min, abnormal_points_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        photo_id,
                        entry.get("deer_id"),
                        entry.get("age_class"),
                        entry.get("left_points_min"),
                        entry.get("left_points_max"),
                        entry.get("right_points_min"),
                        entry.get("right_points_max"),
                        1 if entry.get("left_points_uncertain") else 0,
                        1 if entry.get("right_points_uncertain") else 0,
                        entry.get("left_ab_points_min"),
                        entry.get("right_ab_points_min"),
                        1 if entry.get("left_ab_points_uncertain") else 0,
                        1 if entry.get("right_ab_points_uncertain") else 0,
                        entry.get("abnormal_points_min"),
                        entry.get("abnormal_points_max"),
                    )
                    for entry in deer_entries
                    if entry.get("deer_id")
                ]
            )
        self.conn.commit()

    def set_favorite(self, photo_id: int, is_favorite: bool):
        """Set favorite flag for a photo."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET favorite = ? WHERE id = ?", (1 if is_favorite else 0, photo_id))
            self.conn.commit()

    def is_favorite(self, photo_id: int) -> bool:
        """Check if photo is marked favorite."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT favorite FROM photos WHERE id = ?", (photo_id,))
        row = cursor.fetchone()
        return bool(row['favorite']) if row else False

    def set_notes(self, photo_id: int, notes: str):
        """Set notes text for a photo."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET notes = ? WHERE id = ?", (notes or "", photo_id))
            self.conn.commit()

    def get_notes(self, photo_id: int) -> str:
        """Get notes text for a photo."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT notes FROM photos WHERE id = ?", (photo_id,))
        row = cursor.fetchone()
        return row['notes'] if row else ""
    
    def update_photo_attributes(self, photo_id: int, camera_location: str = "", key_characteristics: str = ""):
        """Update camera location and key characteristics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE photos SET camera_location = ?, key_characteristics = ? WHERE id = ?
        """, (camera_location, key_characteristics, photo_id))
        self.conn.commit()
    
    def set_suggested_tag(self, photo_id: int, tag: str, confidence: Optional[float]):
        """Store AI-suggested tag and confidence."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE photos SET suggested_tag = ?, suggested_confidence = ? WHERE id = ?
            """, (tag, confidence, photo_id))
            self.conn.commit()

    def set_suggested_sex(self, photo_id: int, sex: str, confidence: Optional[float]):
        """Store AI-suggested sex (buck/doe) and confidence."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE photos SET suggested_sex = ?, suggested_sex_confidence = ? WHERE id = ?
            """, (sex, confidence, photo_id))
            self.conn.commit()

    def log_ai_rejection(self, photo_id: int, suggestion_type: str, ai_suggested: str,
                         correct_label: Optional[str] = None, model_version: Optional[str] = None):
        """Log a rejected AI suggestion for future model training.

        Args:
            photo_id: The photo ID
            suggestion_type: 'species' or 'sex'
            ai_suggested: What the AI predicted
            correct_label: What the correct label is (if known)
            model_version: Version of the model that made the prediction
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ai_rejections (photo_id, suggestion_type, ai_suggested, correct_label, model_version, rejected_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (photo_id, suggestion_type, ai_suggested, correct_label, model_version))
        self.conn.commit()

    def get_ai_rejections(self, suggestion_type: Optional[str] = None) -> List[dict]:
        """Get all logged AI rejections, optionally filtered by type."""
        cursor = self.conn.cursor()
        if suggestion_type:
            cursor.execute("""
                SELECT r.*, p.file_path FROM ai_rejections r
                JOIN photos p ON r.photo_id = p.id
                WHERE r.suggestion_type = ?
                ORDER BY r.rejected_at DESC
            """, (suggestion_type,))
        else:
            cursor.execute("""
                SELECT r.*, p.file_path FROM ai_rejections r
                JOIN photos p ON r.photo_id = p.id
                ORDER BY r.rejected_at DESC
            """)
        return [dict(row) for row in cursor.fetchall()]

    def propagate_deer_id_to_empty(self, deer_id: str, exclude_photo_id: Optional[int] = None) -> List[int]:
        """Set deer_id on photos that do not yet have one. Returns list of updated photo IDs."""
        if not deer_id:
            return []
        cursor = self.conn.cursor()
        params = []
        exclude_clause = ""
        if exclude_photo_id is not None:
            exclude_clause = "AND p.id != ?"
            params.append(exclude_photo_id)

        cursor.execute(f"""
            SELECT p.id FROM photos p
            LEFT JOIN deer_metadata d ON p.id = d.photo_id
            WHERE (d.deer_id IS NULL OR d.deer_id = '') {exclude_clause}
        """, params)
        rows = [r[0] for r in cursor.fetchall()]
        updated_ids: List[int] = []
        for pid in rows:
            self.set_deer_metadata(photo_id=pid, deer_id=deer_id)
            updated_ids.append(pid)
        return updated_ids

    def clear_deer_id(self, photo_ids: List[int]):
        """Clear deer_id for given photo IDs."""
        if not photo_ids:
            return
        cursor = self.conn.cursor()
        cursor.executemany("UPDATE deer_metadata SET deer_id = NULL WHERE photo_id = ?", [(pid,) for pid in photo_ids])
        self.conn.commit()

    def set_season_year(self, photo_id: int, season_year: Optional[int]):
        """Set the antler season year (May–Apr mapping)."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE photos SET season_year = ? WHERE id = ?", (season_year, photo_id))
        self.conn.commit()

    def get_season_year(self, photo_id: int) -> Optional[int]:
        """Get the antler season year for a photo."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT season_year FROM photos WHERE id = ?", (photo_id,))
        row = cursor.fetchone()
        return row['season_year'] if row else None

    # Custom species helpers
    def list_custom_species(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM custom_species ORDER BY name")
        return [r[0] for r in cursor.fetchall()]

    def add_custom_species(self, name: str):
        if not name:
            return
        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT OR IGNORE INTO custom_species (name) VALUES (?)", (name,))
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Failed to add custom species '{name}': {e}")

    def bump_recent_buck(self, deer_id: str):
        """Remember a buck ID with a timestamp (queue semantics, stable order on reuse)."""
        if not deer_id:
            return
        ts = int(time.time())
        try:
            cursor = self.conn.cursor()
            # If already present, keep its original slot.
            cursor.execute("SELECT 1 FROM recent_bucks WHERE deer_id = ?", (deer_id,))
            if cursor.fetchone():
                return
            cursor.execute("INSERT OR IGNORE INTO recent_bucks (deer_id, updated_at) VALUES (?, ?)", (deer_id, ts))
            # Keep only the latest 12 insertions (for the 3x4 quick buck button grid)
            cursor.execute(
                "DELETE FROM recent_bucks WHERE deer_id NOT IN ("
                "SELECT deer_id FROM recent_bucks ORDER BY updated_at DESC LIMIT 12)"
            )
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Failed to bump recent buck '{deer_id}': {e}")

    def list_recent_bucks(self, limit: int = 5) -> List[str]:
        cursor = self.conn.cursor()
        # Return in insertion order among the kept set (oldest of the kept first, newest last)
        cursor.execute(
            "SELECT deer_id FROM (SELECT deer_id, updated_at FROM recent_bucks "
            "ORDER BY updated_at DESC LIMIT ?) ORDER BY updated_at ASC",
            (limit,),
        )
        return [row[0] for row in cursor.fetchall()]

    # Annotation boxes
    def set_boxes(self, photo_id: int, boxes: List[Dict[str, float]]):
        """Replace boxes for a photo. Uses differential update with soft-delete for sync safety.

        Boxes with an 'id' matching an existing active box are updated in-place.
        Boxes without a matching id are inserted with a new sync_id.
        Existing active boxes not present in the incoming list are soft-deleted.
        """
        import uuid as _uuid
        with self._lock:
            cursor = self.conn.cursor()
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            # Get existing active (non-tombstoned) boxes
            cursor.execute(
                "SELECT id, sync_id, label, x1, y1, x2, y2 FROM annotation_boxes "
                "WHERE photo_id = ? AND deleted_at IS NULL",
                (photo_id,),
            )
            existing_rows = cursor.fetchall()
            existing = {row[0]: row[1] for row in existing_rows}
            existing_ai = []
            for row in existing_rows:
                label = row[2] or ""
                if label.startswith("ai_"):
                    existing_ai.append({
                        "id": row[0],
                        "label": label,
                        "x1": row[3],
                        "y1": row[4],
                        "x2": row[5],
                        "y2": row[6],
                    })

            # Deduplicate incoming boxes by (label, rounded coordinates)
            seen = set()
            unique_boxes = []
            for b in boxes:
                key = (b.get("label", ""),
                       round(float(b["x1"]), 2), round(float(b["y1"]), 2),
                       round(float(b["x2"]), 2), round(float(b["y2"]), 2))
                if key not in seen:
                    seen.add(key)
                    unique_boxes.append(b)

            def _iou(box_a, box_b):
                ax1, ay1, ax2, ay2 = box_a["x1"], box_a["y1"], box_a["x2"], box_a["y2"]
                bx1, by1, bx2, by2 = box_b["x1"], box_b["y1"], box_b["x2"], box_b["y2"]
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    return 0.0
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
                b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
                union_area = a_area + b_area - inter_area
                return inter_area / union_area if union_area > 0 else 0.0

            incoming_ids = set()
            for b in unique_boxes:
                box_id = b.get("id")
                conf = b.get("confidence")
                species = b.get("species", "")
                species_conf = b.get("species_conf")
                sex = b.get("sex", "")
                sex_conf = b.get("sex_conf")
                ai_suggested = b.get("ai_suggested_species", "")
                label = b.get("label", "")

                # Extract deer metadata fields
                deer_id = b.get("deer_id", "")
                age_class = b.get("age_class", "")
                lp_min = b.get("left_points_min")
                rp_min = b.get("right_points_min")
                lp_unc = 1 if b.get("left_points_uncertain") else 0
                rp_unc = 1 if b.get("right_points_uncertain") else 0
                lab_min = b.get("left_ab_points_min")
                rab_min = b.get("right_ab_points_min")
                lab_unc = 1 if b.get("left_ab_points_uncertain") else 0
                rab_unc = 1 if b.get("right_ab_points_uncertain") else 0
                ab_min = b.get("abnormal_points_min")
                ab_max = b.get("abnormal_points_max")

                if box_id and box_id in existing:
                    # UPDATE existing box (preserves sync_id)
                    incoming_ids.add(box_id)
                    cursor.execute("""UPDATE annotation_boxes SET
                        label=?, x1=?, y1=?, x2=?, y2=?, confidence=?,
                        species=?, species_conf=?, sex=?, sex_conf=?,
                        head_x1=?, head_y1=?, head_x2=?, head_y2=?, head_notes=?,
                        ai_suggested_species=?,
                        deer_id=?, age_class=?,
                        left_points_min=?, right_points_min=?,
                        left_points_uncertain=?, right_points_uncertain=?,
                        left_ab_points_min=?, right_ab_points_min=?,
                        left_ab_points_uncertain=?, right_ab_points_uncertain=?,
                        abnormal_points_min=?, abnormal_points_max=?
                        WHERE id = ?""",
                        (label, float(b["x1"]), float(b["y1"]),
                         float(b["x2"]), float(b["y2"]),
                         float(conf) if conf is not None else None,
                         species if species else None,
                         float(species_conf) if species_conf is not None else None,
                         sex if sex else None,
                         float(sex_conf) if sex_conf is not None else None,
                         b.get("head_x1"), b.get("head_y1"),
                         b.get("head_x2"), b.get("head_y2"),
                         b.get("head_notes") or None,
                         ai_suggested if ai_suggested else None,
                         deer_id if deer_id else None,
                         age_class if age_class else None,
                         lp_min, rp_min, lp_unc, rp_unc,
                         lab_min, rab_min, lab_unc, rab_unc,
                         ab_min, ab_max,
                         box_id))
                else:
                    # Deduplicate AI boxes against existing AI boxes with high IoU (preserve labeled boxes)
                    if label.startswith("ai_"):
                        new_box = {
                            "x1": float(b["x1"]),
                            "y1": float(b["y1"]),
                            "x2": float(b["x2"]),
                            "y2": float(b["y2"]),
                        }
                        matched_id = None
                        for e in existing_ai:
                            if e["label"] != label:
                                continue
                            if _iou(new_box, e) > 0.8:
                                matched_id = e["id"]
                                break
                        if matched_id:
                            incoming_ids.add(matched_id)
                            continue
                    # INSERT new box with sync_id
                    sync_id = b.get("sync_id") or str(_uuid.uuid4())
                    cursor.execute("""INSERT INTO annotation_boxes
                        (photo_id, label, x1, y1, x2, y2, confidence, species, species_conf,
                         sex, sex_conf, head_x1, head_y1, head_x2, head_y2, head_notes,
                         ai_suggested_species, sync_id,
                         deer_id, age_class,
                         left_points_min, right_points_min,
                         left_points_uncertain, right_points_uncertain,
                         left_ab_points_min, right_ab_points_min,
                         left_ab_points_uncertain, right_ab_points_uncertain,
                         abnormal_points_min, abnormal_points_max)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (photo_id, label, float(b["x1"]), float(b["y1"]),
                         float(b["x2"]), float(b["y2"]),
                         float(conf) if conf is not None else None,
                         species if species else None,
                         float(species_conf) if species_conf is not None else None,
                         sex if sex else None,
                         float(sex_conf) if sex_conf is not None else None,
                         b.get("head_x1"), b.get("head_y1"),
                         b.get("head_x2"), b.get("head_y2"),
                         b.get("head_notes") or None,
                         ai_suggested if ai_suggested else None,
                         sync_id,
                         deer_id if deer_id else None,
                         age_class if age_class else None,
                         lp_min, rp_min, lp_unc, rp_unc,
                         lab_min, rab_min, lab_unc, rab_unc,
                         ab_min, ab_max))

            # Soft-delete boxes not in incoming set
            for box_id in existing:
                if box_id not in incoming_ids:
                    cursor.execute("UPDATE annotation_boxes SET deleted_at = ?, updated_at = ? WHERE id = ?",
                                  (now, now, box_id))

            self.conn.commit()

    def get_boxes(self, photo_id: int) -> List[Dict[str, float]]:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, label, x1, y1, x2, y2, confidence, species, species_conf,
                       head_x1, head_y1, head_x2, head_y2, head_notes, sex, sex_conf,
                       ai_suggested_species, sync_id,
                       deer_id, age_class,
                       left_points_min, right_points_min,
                       left_points_uncertain, right_points_uncertain,
                       left_ab_points_min, right_ab_points_min,
                       left_ab_points_uncertain, right_ab_points_uncertain,
                       abnormal_points_min, abnormal_points_max
                FROM annotation_boxes WHERE photo_id = ? AND deleted_at IS NULL
            """, (photo_id,))
            out = []
            for row in cursor.fetchall():
                box = {"id": row[0], "label": row[1], "x1": row[2], "y1": row[3], "x2": row[4], "y2": row[5]}
                if row[6] is not None:
                    box["confidence"] = row[6]
                if row[7] is not None:
                    box["species"] = row[7]
                if row[8] is not None:
                    box["species_conf"] = row[8]
                # Head direction line
                if row[9] is not None and row[10] is not None:
                    box["head_x1"] = row[9]
                    box["head_y1"] = row[10]
                    box["head_x2"] = row[11]
                    box["head_y2"] = row[12]
                if row[13]:
                    box["head_notes"] = row[13]
                # Per-box sex (buck/doe)
                if row[14]:
                    box["sex"] = row[14]
                if row[15] is not None:
                    box["sex_conf"] = row[15]
                if row[16]:
                    box["ai_suggested_species"] = row[16]
                if row[17]:
                    box["sync_id"] = row[17]
                # Per-box deer metadata
                if row[18]:
                    box["deer_id"] = row[18]
                if row[19]:
                    box["age_class"] = row[19]
                if row[20] is not None:
                    box["left_points_min"] = row[20]
                if row[21] is not None:
                    box["right_points_min"] = row[21]
                box["left_points_uncertain"] = bool(row[22]) if row[22] is not None else False
                box["right_points_uncertain"] = bool(row[23]) if row[23] is not None else False
                if row[24] is not None:
                    box["left_ab_points_min"] = row[24]
                if row[25] is not None:
                    box["right_ab_points_min"] = row[25]
                box["left_ab_points_uncertain"] = bool(row[26]) if row[26] is not None else False
                box["right_ab_points_uncertain"] = bool(row[27]) if row[27] is not None else False
                if row[28] is not None:
                    box["abnormal_points_min"] = row[28]
                if row[29] is not None:
                    box["abnormal_points_max"] = row[29]
                out.append(box)
            return out

    def update_box_fields(self, box_id: int, **fields) -> None:
        """Update specific fields on a single annotation box by id."""
        if not box_id:
            return
        allowed = {
            "label", "x1", "y1", "x2", "y2", "confidence",
            "species", "species_conf", "sex", "sex_conf",
            "head_x1", "head_y1", "head_x2", "head_y2", "head_notes",
            "ai_suggested_species",
            "deer_id", "age_class",
            "left_points_min", "right_points_min",
            "left_points_uncertain", "right_points_uncertain",
            "left_ab_points_min", "right_ab_points_min",
            "left_ab_points_uncertain", "right_ab_points_uncertain",
            "abnormal_points_min", "abnormal_points_max",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys()) + ", updated_at = ?"
        values = list(updates.values()) + [now, box_id]
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(f"UPDATE annotation_boxes SET {set_clause} WHERE id = ?", values)
            self.conn.commit()

    def set_box_species(self, box_id: int, species: str, confidence: float = None):
        """Set species classification for a specific box."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE annotation_boxes SET species = ?, species_conf = ? WHERE id = ?",
                (species, confidence, box_id)
            )
            self.conn.commit()

    def set_box_sex(self, box_id: int, sex: str, confidence: float = None):
        """Set sex (buck/doe) classification for a specific box."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE annotation_boxes SET sex = ?, sex_conf = ? WHERE id = ?",
                (sex, confidence, box_id)
            )
            self.conn.commit()

    def set_box_ai_suggestion(self, box_id: int, species: str, confidence: float = None):
        """Set AI-suggested species for a specific box (separate from user label)."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE annotation_boxes SET ai_suggested_species = ? WHERE id = ?",
                (species, box_id)
            )
            self.conn.commit()

    def set_box_head_line(self, box_id: int, x1: float, y1: float, x2: float, y2: float, notes: str = None):
        """Set head direction line for a specific box.

        Args:
            box_id: The annotation box ID
            x1, y1: Top of skull position (relative 0-1)
            x2, y2: Nose tip position (relative 0-1)
            notes: Optional notes about this annotation
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE annotation_boxes SET head_x1 = ?, head_y1 = ?, head_x2 = ?, head_y2 = ?, head_notes = ? WHERE id = ?",
                (x1, y1, x2, y2, notes, box_id)
            )
            self.conn.commit()

    def clear_box_head_line(self, box_id: int):
        """Clear head direction line for a specific box."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE annotation_boxes SET head_x1 = NULL, head_y1 = NULL, head_x2 = NULL, head_y2 = NULL, head_notes = NULL WHERE id = ?",
                (box_id,)
            )
            self.conn.commit()

    def get_deer_boxes_for_head_annotation(self) -> List[Dict]:
        """Get all deer boxes that need head direction annotation.

        Returns boxes where species is 'Deer' and head line is not set.
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT ab.id, ab.photo_id, p.file_path, ab.x1, ab.y1, ab.x2, ab.y2,
                       ab.head_x1, ab.head_y1, ab.head_x2, ab.head_y2, ab.head_notes
                FROM annotation_boxes ab
                JOIN photos p ON ab.photo_id = p.id
                JOIN tags t ON p.id = t.photo_id
                WHERE t.tag_name = 'Deer'
                  AND t.deleted_at IS NULL
                  AND ab.label IN ('subject', 'ai_animal')
                ORDER BY p.date_taken DESC
            """)
            out = []
            for row in cursor.fetchall():
                out.append({
                    "box_id": row[0],
                    "photo_id": row[1],
                    "file_path": row[2],
                    "x1": row[3], "y1": row[4], "x2": row[5], "y2": row[6],
                    "head_x1": row[7], "head_y1": row[8], "head_x2": row[9], "head_y2": row[10],
                    "head_notes": row[11]
                })
            return out

    def has_detection_boxes(self, photo_id: int) -> bool:
        """Check if photo has any detection boxes (AI or human-labeled)."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM annotation_boxes WHERE photo_id = ? AND deleted_at IS NULL", (photo_id,))
            return cursor.fetchone()[0] > 0

    def get_seasons(self) -> List[int]:
        """Return distinct antler season years present in the library, newest first."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT season_year FROM photos WHERE season_year IS NOT NULL ORDER BY season_year DESC")
        rows = cursor.fetchall()
        return [row['season_year'] for row in rows if row['season_year'] is not None]
    
    def get_suggested_tags(self) -> List[str]:
        """Return distinct suggested tags present in the library."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT suggested_tag FROM photos WHERE suggested_tag IS NOT NULL AND suggested_tag != ''")
        rows = cursor.fetchall()
        return [row['suggested_tag'] for row in rows if row['suggested_tag']]
    
    def get_deer_metadata(self, photo_id: int) -> Dict[str, Optional[Union[str, int]]]:
        """Get deer metadata for a photo."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT deer_id, age_class, left_points_min, right_points_min,
                   left_points_uncertain, right_points_uncertain,
                   left_ab_points_min, right_ab_points_min,
                   left_ab_points_uncertain, right_ab_points_uncertain,
                   abnormal_points_min, abnormal_points_max, broken_antler_side, broken_antler_note
            FROM deer_metadata WHERE photo_id = ?
        """, (photo_id,))
        row = cursor.fetchone()
        if row:
            return {
                'deer_id': row['deer_id'],
                'age_class': row['age_class'],
                'left_points_min': row['left_points_min'],
                'left_points_uncertain': bool(row['left_points_uncertain']) if row['left_points_uncertain'] is not None else False,
                'right_points_min': row['right_points_min'],
                'right_points_uncertain': bool(row['right_points_uncertain']) if row['right_points_uncertain'] is not None else False,
                'left_ab_points_min': row['left_ab_points_min'],
                'right_ab_points_min': row['right_ab_points_min'],
                'left_ab_points_uncertain': bool(row['left_ab_points_uncertain']) if row['left_ab_points_uncertain'] is not None else False,
                'right_ab_points_uncertain': bool(row['right_ab_points_uncertain']) if row['right_ab_points_uncertain'] is not None else False,
                'abnormal_points_min': row['abnormal_points_min'],
                'abnormal_points_max': row['abnormal_points_max'],
                'broken_antler_side': row['broken_antler_side'],
                'broken_antler_note': row['broken_antler_note'],
            }
        return {
            'deer_id': None,
            'age_class': None,
            'left_points_min': None,
            'right_points_min': None,
            'left_points_uncertain': False,
            'right_points_uncertain': False,
            'left_ab_points_min': None,
            'right_ab_points_min': None,
            'left_ab_points_uncertain': False,
            'right_ab_points_uncertain': False,
            'abnormal_points_min': None,
            'abnormal_points_max': None,
            'broken_antler_side': None,
            'broken_antler_note': None,
        }

    def get_all_deer_metadata_batch(self, photo_ids: List[int]) -> Dict[int, Dict]:
        """Get deer metadata for many photos in one query. Thread-safe."""
        if not photo_ids:
            return {}
        with self._lock:
            cursor = self.conn.cursor()
            placeholders = ",".join(["?"] * len(photo_ids))
            cursor.execute(f"""
                SELECT photo_id, deer_id, age_class, left_points_min, right_points_min,
                       left_points_uncertain, right_points_uncertain,
                       left_ab_points_min, right_ab_points_min,
                       left_ab_points_uncertain, right_ab_points_uncertain,
                       abnormal_points_min, abnormal_points_max, broken_antler_side, broken_antler_note
                FROM deer_metadata WHERE photo_id IN ({placeholders})
            """, photo_ids)
            out: Dict[int, Dict] = {}
            for row in cursor.fetchall():
                out[row["photo_id"]] = {
                    'deer_id': row['deer_id'],
                    'age_class': row['age_class'],
                    'left_points_min': row['left_points_min'],
                    'left_points_uncertain': bool(row['left_points_uncertain']) if row['left_points_uncertain'] is not None else False,
                    'right_points_min': row['right_points_min'],
                    'right_points_uncertain': bool(row['right_points_uncertain']) if row['right_points_uncertain'] is not None else False,
                    'left_ab_points_min': row['left_ab_points_min'],
                    'right_ab_points_min': row['right_ab_points_min'],
                    'left_ab_points_uncertain': bool(row['left_ab_points_uncertain']) if row['left_ab_points_uncertain'] is not None else False,
                    'right_ab_points_uncertain': bool(row['right_ab_points_uncertain']) if row['right_ab_points_uncertain'] is not None else False,
                    'abnormal_points_min': row['abnormal_points_min'],
                    'abnormal_points_max': row['abnormal_points_max'],
                    'broken_antler_side': row['broken_antler_side'],
                    'broken_antler_note': row['broken_antler_note'],
                }
            return out
    
    def get_nearby_photos(self, camera_location: Optional[str], date_taken: Optional[str], window_minutes: int = 5) -> List[Dict]:
        """Return photos from the same camera_location taken within +/- window_minutes of date_taken."""
        if not camera_location or not date_taken:
            return []
        try:
            base_dt = datetime.fromisoformat(date_taken)
        except Exception:
            try:
                base_dt = datetime.strptime(date_taken, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return []
        start = base_dt.timestamp() - window_minutes * 60
        end = base_dt.timestamp() + window_minutes * 60
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, file_path, date_taken, camera_location FROM photos
            WHERE camera_location = ?
              AND date_taken IS NOT NULL
            """,
            (camera_location,),
        )
        rows = cursor.fetchall()
        nearby: List[Dict] = []
        for row in rows:
            try:
                dt = datetime.fromisoformat(row["date_taken"])
            except Exception:
                try:
                    dt = datetime.strptime(row["date_taken"], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
            ts = dt.timestamp()
            if start <= ts <= end:
                nearby.append(dict(row))
        return nearby
    
    def search_photos(self, tag: Union[str, List[str]] = None, deer_id: str = None,
                     age_class: Union[str, List[str]] = None, season_year: Union[int, List[int], None] = None,
                     date_from: str = None, date_to: str = None, camera_location: Optional[str] = None,
                     suggested_tag: Union[str, List[str], None] = None,
                     include_archived: bool = False, archived_only: bool = False) -> List[Dict]:
        """Search photos by various criteria.

        Args:
            include_archived: If True, include archived photos in results
            archived_only: If True, only return archived photos

        Returns:
            List of photo dictionaries
        """
        cursor = self.conn.cursor()
        query = """
            SELECT DISTINCT p.id, p.file_path, p.original_name, p.date_taken,
                   p.camera_model, p.thumbnail_path, p.favorite, p.notes, p.season_year, p.camera_location, p.key_characteristics, p.suggested_tag, p.suggested_confidence, p.suggested_sex, p.suggested_sex_confidence, p.site_id, p.suggested_site_id, p.suggested_site_confidence, p.collection, p.archived
            FROM photos p
        """
        conditions = []
        params = []

        # Archive filtering
        if archived_only:
            conditions.append("p.archived = 1")
        elif not include_archived:
            conditions.append("(p.archived IS NULL OR p.archived = 0)")
        
        if tag:
            query += " INNER JOIN tags t ON p.id = t.photo_id"
            conditions.append("t.deleted_at IS NULL")
            if isinstance(tag, list):
                placeholders = ",".join(["?"] * len(tag))
                conditions.append(f"t.tag_name IN ({placeholders})")
                params.extend(tag)
            else:
                conditions.append("t.tag_name = ?")
                params.append(tag)

        deer_filter_needed = deer_id or age_class
        if deer_filter_needed:
            query += " LEFT JOIN deer_metadata d ON p.id = d.photo_id"
        if deer_id:
            query += " LEFT JOIN deer_additional da ON p.id = da.photo_id"

        if deer_id:
            if isinstance(deer_id, list):
                placeholders = ",".join(["?"] * len(deer_id))
                conditions.append(f"(d.deer_id IN ({placeholders}) OR da.deer_id IN ({placeholders}))")
                params.extend(deer_id)
                params.extend(deer_id)
            else:
                conditions.append("(d.deer_id = ? OR da.deer_id = ?)")
                params.append(deer_id)
                params.append(deer_id)

        if age_class:
            if isinstance(age_class, list):
                placeholders = ",".join(["?"] * len(age_class))
                conditions.append(f"d.age_class IN ({placeholders})")
                params.extend(age_class)
            else:
                conditions.append("d.age_class = ?")
                params.append(age_class)
        
        if date_from:
            conditions.append("p.date_taken >= ?")
            params.append(date_from)
        
        if date_to:
            conditions.append("p.date_taken <= ?")
            params.append(date_to)

        if season_year is not None:
            if isinstance(season_year, list):
                placeholders = ",".join(["?"] * len(season_year))
                conditions.append(f"p.season_year IN ({placeholders})")
                params.extend(season_year)
            else:
                conditions.append("p.season_year = ?")
                params.append(season_year)

        if camera_location:
            conditions.append("p.camera_location LIKE ?")
            params.append(f"%{camera_location}%")

        if suggested_tag:
            if isinstance(suggested_tag, list):
                placeholders = ",".join(["?"] * len(suggested_tag))
                conditions.append(f"p.suggested_tag IN ({placeholders})")
                params.extend(suggested_tag)
            else:
                conditions.append("p.suggested_tag = ?")
                params.append(suggested_tag)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY p.date_taken DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_all_photos(self, include_archived: bool = True) -> List[Dict]:
        """Get all photos (including archived by default for UI filtering)."""
        return self.search_photos(include_archived=include_archived)

    def get_distinct_collections(self) -> List[str]:
        """Get all distinct collection names from the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT collection FROM photos
            WHERE collection IS NOT NULL AND collection != ''
            ORDER BY collection
        """)
        return [row[0] for row in cursor.fetchall()]

    def remove_photo(self, photo_id: int):
        """Remove a photo from the database.
        
        Note: CASCADE will automatically remove associated tags and deer_metadata.
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
        self.conn.commit()
    
    def delete_photo(self, photo_id: int):
        """Alias for remove_photo for API compatibility."""
        self.remove_photo(photo_id)

    def archive_photo(self, photo_id: int):
        """Archive a photo (hide from default view but don't delete)."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE photos SET archived = 1 WHERE id = ?", (photo_id,))
        self.conn.commit()

    def unarchive_photo(self, photo_id: int):
        """Unarchive a photo (restore to default view)."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE photos SET archived = 0 WHERE id = ?", (photo_id,))
        self.conn.commit()

    def archive_photos(self, photo_ids: List[int]):
        """Archive multiple photos using batch update. Favorites are protected."""
        if not photo_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ",".join(["?"] * len(photo_ids))
        # Skip favorites - they are protected from archiving
        cursor.execute(f"UPDATE photos SET archived = 1 WHERE id IN ({placeholders}) AND (favorite IS NULL OR favorite = 0)", photo_ids)
        self.conn.commit()

    def toggle_favorite(self, photo_id: int) -> bool:
        """Toggle favorite status for a photo. Returns new favorite status."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT favorite FROM photos WHERE id = ?", (photo_id,))
        row = cursor.fetchone()
        current = row[0] if row and row[0] else 0
        new_value = 0 if current else 1
        cursor.execute("UPDATE photos SET favorite = ? WHERE id = ?", (new_value, photo_id))
        self.conn.commit()
        return bool(new_value)

    def set_favorite(self, photo_id: int, favorite: bool):
        """Set favorite status for a photo."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET favorite = ? WHERE id = ?", (1 if favorite else 0, photo_id))
            self.conn.commit()

    def get_archived_count(self) -> int:
        """Get count of archived photos."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos WHERE archived = 1")
        return cursor.fetchone()[0]

    def get_photos_for_auto_archive(self, keep_species: List[str]) -> List[int]:
        """Get photo IDs that should be auto-archived based on species criteria.

        Returns photos that:
        - Are NOT archived
        - Are NOT favorite
        - Have at least one tag (not unlabeled)
        - Have NO tags in the keep_species list

        Args:
            keep_species: List of species/tags to keep (not archive)

        Returns:
            List of photo IDs to archive
        """
        if not keep_species:
            return []

        cursor = self.conn.cursor()

        # Build placeholders for the keep_species list
        placeholders = ",".join(["?" for _ in keep_species])

        # Find photos that:
        # 1. Have tags (not unlabeled)
        # 2. None of their tags are in keep_species
        # 3. Not archived
        # 4. Not favorite
        query = f"""
            SELECT DISTINCT p.id
            FROM photos p
            INNER JOIN tags t ON t.photo_id = p.id
            WHERE p.archived = 0
              AND t.deleted_at IS NULL
              AND (p.favorite IS NULL OR p.favorite = 0)
              AND p.id NOT IN (
                  SELECT DISTINCT photo_id
                  FROM tags
                  WHERE tag_name IN ({placeholders})
                    AND deleted_at IS NULL
              )
        """

        cursor.execute(query, keep_species)
        return [row[0] for row in cursor.fetchall()]

    def get_all_species_tags(self) -> List[str]:
        """Get all unique species tags from the database, sorted by frequency."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT tag_name, COUNT(*) as cnt
            FROM tags
            WHERE tag_name NOT IN ('Buck', 'Doe', 'Unknown')
              AND deleted_at IS NULL
            GROUP BY tag_name
            ORDER BY cnt DESC
        """)
        return [row[0] for row in cursor.fetchall()]

    # ========== Site Management ==========

    def create_site(self, name: str, description: str = "", representative_photo_id: int = None,
                    confirmed: bool = True) -> int:
        """Create a new site and return its ID. confirmed=False means it's a suggestion."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO sites (name, description, representative_photo_id, created_at, confirmed)
                VALUES (?, ?, ?, datetime('now'), ?)
            """, (name, description, representative_photo_id, 1 if confirmed else 0))
            site_id = cursor.lastrowid
            self.conn.commit()
            return site_id

    def get_site(self, site_id: int) -> Optional[Dict]:
        """Get site by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sites WHERE id = ?", (site_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_site_by_name(self, name: str) -> Optional[Dict]:
        """Get site by name (case-insensitive)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sites WHERE LOWER(name) = LOWER(?)", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_sites(self, include_unconfirmed: bool = True) -> List[Dict]:
        """Get all sites with photo counts (confirmed and suggested)."""
        cursor = self.conn.cursor()
        query = """
            SELECT s.id, s.name, s.description, s.representative_photo_id, s.created_at,
                   COALESCE(s.confirmed, 1) as confirmed,
                   COUNT(DISTINCT pc.id) as photo_count,
                   COUNT(DISTINCT ps.id) as suggested_count
            FROM sites s
            LEFT JOIN photos pc ON pc.site_id = s.id
            LEFT JOIN photos ps ON ps.suggested_site_id = s.id
            GROUP BY s.id
            ORDER BY s.name
        """
        if not include_unconfirmed:
            query = query.replace("GROUP BY", "WHERE COALESCE(s.confirmed, 1) = 1 GROUP BY")
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def update_site(self, site_id: int, name: str = None, description: str = None,
                    representative_photo_id: int = None):
        """Update site details."""
        with self._lock:
            cursor = self.conn.cursor()
            updates = []
            params = []
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if representative_photo_id is not None:
                updates.append("representative_photo_id = ?")
                params.append(representative_photo_id)
            if updates:
                params.append(site_id)
                cursor.execute(f"UPDATE sites SET {', '.join(updates)} WHERE id = ?", params)
                self.conn.commit()

    def delete_site(self, site_id: int):
        """Delete a site (does not delete photos, just unlinks them)."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET site_id = NULL WHERE site_id = ?", (site_id,))
            cursor.execute("DELETE FROM sites WHERE id = ?", (site_id,))
            self.conn.commit()

    def set_photos_site(self, photo_ids: List[int], site_id: Optional[int]):
        """Assign multiple photos to a site."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.executemany(
                "UPDATE photos SET site_id = ? WHERE id = ?",
                [(site_id, pid) for pid in photo_ids]
            )
            self.conn.commit()

    def get_photos_by_site(self, site_id: Optional[int]) -> List[Dict]:
        """Get all photos for a site (None = unassigned)."""
        cursor = self.conn.cursor()
        if site_id is None:
            cursor.execute("""
                SELECT id, file_path, date_taken, camera_model, site_id
                FROM photos WHERE site_id IS NULL
                ORDER BY date_taken DESC
            """)
        else:
            cursor.execute("""
                SELECT id, file_path, date_taken, camera_model, site_id
                FROM photos WHERE site_id = ?
                ORDER BY date_taken DESC
            """, (site_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_unassigned_photo_count(self) -> int:
        """Count photos without a site assignment."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos WHERE site_id IS NULL")
        return cursor.fetchone()[0]

    def get_unassigned_photo_ids(self) -> List[int]:
        """Get IDs of photos without a site assignment."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM photos WHERE site_id IS NULL")
        return [row[0] for row in cursor.fetchall()]

    def get_photo_ids_by_site(self, site_id: int) -> List[int]:
        """Get just photo IDs for a site."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM photos WHERE site_id = ?", (site_id,))
        return [row[0] for row in cursor.fetchall()]

    def set_photo_site(self, photo_id: int, site_id: Optional[int]):
        """Manually assign a photo to a site (confirmed), clearing any suggestion."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE photos SET site_id = ?, suggested_site_id = NULL, suggested_site_confidence = NULL
                WHERE id = ?
            """, (site_id, photo_id))
            self.conn.commit()

    def set_photo_site_suggestion(self, photo_id: int, site_id: int, confidence: float):
        """Set a suggested site for a photo (not confirmed yet)."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE photos SET suggested_site_id = ?, suggested_site_confidence = ?
                WHERE id = ?
            """, (site_id, confidence, photo_id))
            self.conn.commit()

    def confirm_site(self, site_id: int):
        """Mark a site as confirmed and move all suggested photos to confirmed."""
        with self._lock:
            cursor = self.conn.cursor()
            # Mark site as confirmed
            cursor.execute("UPDATE sites SET confirmed = 1 WHERE id = ?", (site_id,))
            # Move suggested photos to confirmed site_id
            cursor.execute("""
                UPDATE photos SET site_id = suggested_site_id,
                                 suggested_site_id = NULL,
                                 suggested_site_confidence = NULL
                WHERE suggested_site_id = ?
            """, (site_id,))
            self.conn.commit()

    def reject_site_suggestion(self, site_id: int):
        """Reject a site suggestion - clear suggested_site_id for those photos."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE photos SET suggested_site_id = NULL, suggested_site_confidence = NULL
                WHERE suggested_site_id = ?
            """, (site_id,))
            # Delete the unconfirmed site
            cursor.execute("DELETE FROM sites WHERE id = ? AND COALESCE(confirmed, 0) = 0", (site_id,))
            self.conn.commit()

    def get_suggested_site_photos(self, site_id: int) -> List[Dict]:
        """Get photos suggested for a site (not yet confirmed)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, file_path, thumbnail_path, suggested_site_confidence
            FROM photos WHERE suggested_site_id = ?
            ORDER BY suggested_site_confidence DESC
        """, (site_id,))
        return [dict(row) for row in cursor.fetchall()]

    # ========== Embeddings for Site Clustering ==========

    def save_embedding(self, photo_id: int, embedding: bytes, model_version: str = None):
        """Save a photo embedding (as raw bytes)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO photo_embeddings (photo_id, embedding, model_version)
            VALUES (?, ?, ?)
        """, (photo_id, embedding, model_version))
        self.conn.commit()

    def get_embedding(self, photo_id: int) -> Optional[Tuple[bytes, str]]:
        """Get embedding bytes and version for a photo."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding, model_version FROM photo_embeddings WHERE photo_id = ?", (photo_id,))
        row = cursor.fetchone()
        return (row[0], row[1]) if row else None

    def get_all_embeddings(self) -> List[Tuple[int, bytes, str]]:
        """Get all photo embeddings as (photo_id, embedding_bytes, model_version) tuples."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT photo_id, embedding, model_version FROM photo_embeddings")
        return [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    def get_photos_without_embeddings(self) -> List[Dict]:
        """Get photos that don't have embeddings yet."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.id, p.file_path, p.thumbnail_path
            FROM photos p
            LEFT JOIN photo_embeddings e ON p.id = e.photo_id
            WHERE e.photo_id IS NULL
        """)
        return [dict(row) for row in cursor.fetchall()]

    def clear_all_embeddings(self):
        """Clear all embeddings (for re-computing)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM photo_embeddings")
        self.conn.commit()

    # ─────────────────────────────────────────────────────────────────────
    # File Hash Methods (for cross-computer sync)
    # ─────────────────────────────────────────────────────────────────────

    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate MD5 hash of a file."""
        import hashlib
        try:
            h = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def calculate_missing_hashes(self, progress_callback=None) -> int:
        """Calculate file hashes for photos that don't have one.

        Args:
            progress_callback: Optional callable(current, total) for progress updates

        Returns:
            Number of hashes calculated
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, file_path, original_name FROM photos WHERE file_hash IS NULL")
        rows = cursor.fetchall()

        count = 0
        duplicates = []
        total = len(rows)
        for i, row in enumerate(rows):
            if progress_callback:
                progress_callback(i, total)

            photo_id = row[0]
            file_path = row[1]
            original_name = row[2] if len(row) > 2 else "unknown"

            if file_path and os.path.exists(file_path):
                file_hash = self._calculate_file_hash(file_path)
                if file_hash:
                    # Check if this hash already exists (would be a duplicate)
                    cursor.execute("SELECT id FROM photos WHERE file_hash = ?", (file_hash,))
                    existing = cursor.fetchone()
                    if existing:
                        # This is a duplicate - log it but don't crash
                        duplicates.append((photo_id, original_name, existing[0]))
                        logger.warning(f"Duplicate photo detected: ID {photo_id} ({original_name}) "
                                       f"has same hash as ID {existing[0]}")
                        continue

                    cursor.execute("UPDATE photos SET file_hash = ? WHERE id = ?",
                                   (file_hash, photo_id))
                    count += 1

        self.conn.commit()

        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate photos that need manual cleanup")

        return count

    def get_hash_stats(self) -> Dict[str, int]:
        """Get stats about file hashes in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM photos WHERE file_hash IS NOT NULL")
        with_hash = cursor.fetchone()[0]
        return {"total": total, "with_hash": with_hash, "missing": total - with_hash}

    def update_photo_hash(self, photo_id: int, file_hash: str):
        """Update the file_hash for a specific photo."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE photos SET file_hash = ? WHERE id = ?", (file_hash, photo_id))
        self.conn.commit()

    # ─────────────────────────────────────────────────────────────────────
    # Supabase Sync Methods
    # ─────────────────────────────────────────────────────────────────────

    def _make_photo_key(self, photo: Dict) -> str:
        """Create a unique key for matching photos across computers."""
        original_name = photo.get("original_name") or ""
        date_taken = photo.get("date_taken") or ""
        camera_model = photo.get("camera_model") or ""
        return f"{original_name}|{date_taken}|{camera_model}"

    def _get_photo_by_key(self, photo_key: str, file_hash: str = None) -> Optional[Dict]:
        """Find a local photo by file_hash (preferred) or photo_key fallback.

        file_hash is stable across devices; photo_key can differ if filenames
        or camera_model strings vary between machines.
        """
        cursor = self.conn.cursor()

        # Prefer file_hash — stable across devices
        if file_hash:
            cursor.execute("SELECT * FROM photos WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            if row:
                return dict(row)

        # Fallback: try matching by the full photo_key
        parts = photo_key.split("|")
        if len(parts) == 3:
            original_name, date_taken, camera_model = parts
            cursor.execute("""
                SELECT * FROM photos
                WHERE original_name = ? AND date_taken = ? AND camera_model = ?
            """, (original_name, date_taken, camera_model))
            row = cursor.fetchone()
            if row:
                return dict(row)

        return None

    # ----------------------------------------------------------------
    # Camera / Club / Roles / Label Suggestions CRUD
    # ----------------------------------------------------------------

    def create_camera(self, name: str, owner: str) -> str:
        """Create a new camera and set owner permission. Returns camera_id."""
        import uuid
        camera_id = str(uuid.uuid4())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO cameras (id, name, owner, verified, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
                (camera_id, name, owner, now, now))
            perm_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO camera_permissions (id, camera_id, username, role, granted_by, granted_at, created_at, updated_at) VALUES (?, ?, ?, 'owner', ?, ?, ?, ?)",
                (perm_id, camera_id, owner, owner, now, now, now))
            self.conn.commit()
        logger.info(f"Created camera '{name}' (id={camera_id}) owned by {owner}")
        return camera_id

    def get_cameras(self, owner: str = None) -> List[Dict]:
        """Get all cameras, optionally filtered by owner."""
        with self._lock:
            cursor = self.conn.cursor()
            if owner:
                cursor.execute("""
                    SELECT c.* FROM cameras c
                    JOIN camera_permissions cp ON c.id = cp.camera_id
                    WHERE cp.username = ? AND cp.role = 'owner'
                    ORDER BY c.name
                """, (owner,))
            else:
                cursor.execute("SELECT * FROM cameras ORDER BY name")
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_all_cameras(self) -> List[Dict]:
        """Return all cameras."""
        return self.get_cameras()

    def get_camera(self, camera_id: str) -> Optional[Dict]:
        """Get a single camera by id."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
            row = cursor.fetchone()
            if row:
                cols = [d[0] for d in cursor.description]
                return dict(zip(cols, row))
        return None

    def get_camera_for_photo(self, photo_id: int) -> Optional[Dict]:
        """Return camera dict for a photo, if assigned."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT camera_id FROM photos WHERE id = ?", (photo_id,))
            row = cursor.fetchone()
            if not row or not row[0]:
                return None
            return self.get_camera(row[0])

    def get_camera_permission(self, camera_id: str, username: str) -> Optional[str]:
        """Return role ('owner', 'editor', 'member') or None for a user on a camera."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT role FROM camera_permissions WHERE camera_id = ? AND username = ?",
                (camera_id, username))
            row = cursor.fetchone()
            if not row and username:
                cursor.execute(
                    "SELECT role FROM camera_permissions WHERE camera_id = ? AND LOWER(username) = ?",
                    (camera_id, username.strip().lower()))
                row = cursor.fetchone()
            return row[0] if row else None

    def get_camera_permission_count(self, camera_id: str) -> int:
        """Return number of permissions assigned to a camera."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM camera_permissions WHERE camera_id = ?", (camera_id,))
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def rename_camera(self, camera_id: str, new_name: str):
        """Rename a camera and mark as verified."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE cameras SET name = ?, verified = 1 WHERE id = ?",
                (new_name, camera_id))
            self.conn.commit()

    def merge_cameras(self, keep_id: str, merge_id: str):
        """Merge merge_id camera into keep_id. Reassigns photos and deletes merge_id."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE photos SET camera_id = ? WHERE camera_id = ?", (keep_id, merge_id))
            cursor.execute("UPDATE label_suggestions SET camera_id = ? WHERE camera_id = ?", (keep_id, merge_id))
            cursor.execute("DELETE FROM camera_permissions WHERE camera_id = ?", (merge_id,))
            cursor.execute("DELETE FROM camera_club_shares WHERE camera_id = ?", (merge_id,))
            cursor.execute("DELETE FROM cameras WHERE id = ?", (merge_id,))
            self.conn.commit()
        logger.info(f"Merged camera {merge_id} into {keep_id}")

    def delete_camera(self, camera_id: str):
        """Delete a camera and related permissions/shares; tombstone suggestions."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM camera_permissions WHERE camera_id = ?", (camera_id,))
            cursor.execute("DELETE FROM camera_club_shares WHERE camera_id = ?", (camera_id,))
            cursor.execute("UPDATE label_suggestions SET deleted_at = ?, updated_at = ? WHERE camera_id = ? AND deleted_at IS NULL",
                           (now, now, camera_id))
            cursor.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
            self.conn.commit()

    def get_user_role_for_photo(self, photo_id: int, username: str) -> str:
        """Check user's role for a photo. Returns 'owner', 'member', or 'none'.

        Owner = has 'owner' permission on the photo's camera.
        Member = member of a club that the photo's camera is shared to.
        """
        with self._lock:
            cursor = self.conn.cursor()
            # Get photo's camera_id
            cursor.execute("SELECT camera_id FROM photos WHERE id = ?", (photo_id,))
            row = cursor.fetchone()
            if not row or not row[0]:
                return 'owner'  # No camera assigned = legacy data, treat as owner
            camera_id = row[0]
            return self._get_role_for_camera(cursor, camera_id, username)

    def _get_role_for_camera(self, cursor, camera_id: str, username: str) -> str:
        """Internal: check role for a specific camera."""
        # Check direct camera permission
        cursor.execute(
            "SELECT role FROM camera_permissions WHERE camera_id = ? AND username = ?",
            (camera_id, username))
        perm = cursor.fetchone()
        if perm:
            return perm[0]  # 'owner' or 'member'

        # Check club membership (member of a club the camera is shared to)
        cursor.execute("""
            SELECT 1 FROM camera_club_shares ccs
            JOIN club_memberships cm ON ccs.club_id = cm.club_id
            WHERE ccs.camera_id = ? AND cm.username = ?
            LIMIT 1
        """, (camera_id, username))
        if cursor.fetchone():
            return 'member'

        return 'none'

    def grant_camera_permission(self, camera_id: str, username: str, role: str, granted_by: str):
        """Grant a user permission on a camera."""
        import uuid
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            perm_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO camera_permissions (id, camera_id, username, role, granted_by, granted_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(camera_id, username) DO UPDATE SET role = ?, granted_by = ?, granted_at = ?
            """, (perm_id, camera_id, username, role, granted_by, now, now, now,
                  role, granted_by, now))
            self.conn.commit()
        logger.info(f"Granted {role} on camera {camera_id} to {username} by {granted_by}")

    def revoke_camera_permission(self, camera_id: str, username: str):
        """Revoke a user's permission on a camera."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "DELETE FROM camera_permissions WHERE camera_id = ? AND username = ?",
                (camera_id, username))
            self.conn.commit()
        logger.info(f"Revoked permission on camera {camera_id} from {username}")

    def create_club(self, name: str, created_by: str) -> str:
        """Create a new club. Returns club_id."""
        import uuid
        club_id = str(uuid.uuid4())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO clubs (id, name, created_by, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (club_id, name, created_by, now, now))
            # Creator is automatically a member
            mem_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO club_memberships (id, club_id, username, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (mem_id, club_id, created_by, now, now))
            self.conn.commit()
        logger.info(f"Created club '{name}' (id={club_id}) by {created_by}")
        return club_id

    def get_clubs(self, username: str = None) -> List[Dict]:
        """Get all clubs, optionally filtered to ones a user belongs to."""
        with self._lock:
            cursor = self.conn.cursor()
            if username:
                cursor.execute("""
                    SELECT c.* FROM clubs c
                    JOIN club_memberships cm ON c.id = cm.club_id
                    WHERE cm.username = ?
                    ORDER BY c.name
                """, (username,))
            else:
                cursor.execute("SELECT * FROM clubs ORDER BY name")
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def add_club_member(self, club_id: str, username: str):
        """Add a member to a club."""
        import uuid
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            mem_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT OR IGNORE INTO club_memberships (id, club_id, username, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (mem_id, club_id, username, now, now))
            self.conn.commit()

    def remove_club_member(self, club_id: str, username: str):
        """Remove a member from a club."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "DELETE FROM club_memberships WHERE club_id = ? AND username = ?",
                (club_id, username))
            self.conn.commit()

    def get_club_members(self, club_id: str) -> List[Dict]:
        """Get all members of a club."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM club_memberships WHERE club_id = ? ORDER BY username",
                (club_id,))
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def share_camera_to_club(self, camera_id: str, club_id: str, shared_by: str, visibility: str = 'full'):
        """Share a camera's photos to a club."""
        import uuid
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            share_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO camera_club_shares (id, camera_id, club_id, shared_by, visibility, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(camera_id, club_id) DO UPDATE SET visibility = ?, shared_by = ?
            """, (share_id, camera_id, club_id, shared_by, visibility, now, now,
                  visibility, shared_by))
            self.conn.commit()
        logger.info(f"Shared camera {camera_id} to club {club_id} by {shared_by}")

    def unshare_camera_from_club(self, camera_id: str, club_id: str):
        """Remove camera sharing from a club."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "DELETE FROM camera_club_shares WHERE camera_id = ? AND club_id = ?",
                (camera_id, club_id))
            self.conn.commit()

    def add_label_suggestion(self, photo_id: int, tag_name: str, suggested_by: str) -> str:
        """Add a label suggestion for a photo. Returns suggestion id."""
        import uuid
        suggestion_id = str(uuid.uuid4())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            # Get file_hash and camera_id from photo
            cursor.execute("SELECT file_hash, camera_id FROM photos WHERE id = ?", (photo_id,))
            row = cursor.fetchone()
            file_hash = row[0] if row else None
            camera_id = row[1] if row else None

            # Check for existing tombstoned suggestion — un-tombstone it
            cursor.execute(
                "SELECT id FROM label_suggestions WHERE photo_id = ? AND tag_name = ? AND suggested_by = ? AND deleted_at IS NOT NULL",
                (photo_id, tag_name, suggested_by))
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    "UPDATE label_suggestions SET deleted_at = NULL, status = 'pending', reviewed_by = NULL, reviewed_at = NULL, updated_at = ? WHERE id = ?",
                    (now, existing[0]))
                self.conn.commit()
                return existing[0]

            try:
                cursor.execute("""
                    INSERT INTO label_suggestions (id, photo_id, file_hash, tag_name, suggested_by, status, camera_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """, (suggestion_id, photo_id, file_hash, tag_name, suggested_by, camera_id, now, now))
                self.conn.commit()
            except sqlite3.IntegrityError:
                # Already exists (not tombstoned) — just return
                cursor.execute(
                    "SELECT id FROM label_suggestions WHERE photo_id = ? AND tag_name = ? AND suggested_by = ?",
                    (photo_id, tag_name, suggested_by))
                row = cursor.fetchone()
                return row[0] if row else suggestion_id
        return suggestion_id

    def approve_suggestion(self, suggestion_id: str, reviewed_by: str):
        """Approve a label suggestion — promotes it to tags."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT photo_id, tag_name FROM label_suggestions WHERE id = ? AND status = 'pending'",
                (suggestion_id,))
            row = cursor.fetchone()
            if not row:
                return
            photo_id, tag_name = row

            # Promote to tags
            self.add_tag(photo_id, tag_name)
            # Update the tag's created_by
            cursor.execute(
                "UPDATE tags SET created_by = ? WHERE photo_id = ? AND tag_name = ? AND deleted_at IS NULL",
                (reviewed_by, photo_id, tag_name))

            # Mark suggestion as approved
            cursor.execute(
                "UPDATE label_suggestions SET status = 'approved', reviewed_by = ?, reviewed_at = ?, updated_at = ? WHERE id = ?",
                (reviewed_by, now, now, suggestion_id))
            self.conn.commit()
        logger.info(f"Approved suggestion {suggestion_id} by {reviewed_by}")

    def accept_suggestion(self, suggestion_id: str, reviewed_by: str) -> bool:
        """Apply suggestion to tags table, mark as 'accepted'."""
        self.approve_suggestion(suggestion_id, reviewed_by)
        return True

    def reject_suggestion(self, suggestion_id: str, reviewed_by: str):
        """Reject a label suggestion."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE label_suggestions SET status = 'rejected', reviewed_by = ?, reviewed_at = ?, updated_at = ? WHERE id = ?",
                (reviewed_by, now, now, suggestion_id))
            self.conn.commit()
        logger.info(f"Rejected suggestion {suggestion_id} by {reviewed_by}")

    def get_pending_suggestions(self, camera_id: str = None, limit: int = 100, owner_username: str = None) -> List[Dict]:
        """Get pending label suggestions, optionally filtered by camera or owner."""
        with self._lock:
            cursor = self.conn.cursor()
            if owner_username:
                cursor.execute("""
                    SELECT ls.*, p.file_path, p.thumbnail_path
                    FROM label_suggestions ls
                    JOIN photos p ON ls.photo_id = p.id
                    JOIN camera_permissions cp ON ls.camera_id = cp.camera_id
                    WHERE ls.status = 'pending' AND ls.deleted_at IS NULL
                      AND cp.username = ? AND cp.role = 'owner'
                    ORDER BY ls.created_at DESC
                    LIMIT ?
                """, (owner_username, limit))
            elif camera_id:
                cursor.execute("""
                    SELECT ls.*, p.file_path, p.thumbnail_path
                    FROM label_suggestions ls
                    JOIN photos p ON ls.photo_id = p.id
                    WHERE ls.status = 'pending' AND ls.deleted_at IS NULL AND ls.camera_id = ?
                    ORDER BY ls.created_at DESC
                    LIMIT ?
                """, (camera_id, limit))
            else:
                cursor.execute("""
                    SELECT ls.*, p.file_path, p.thumbnail_path
                    FROM label_suggestions ls
                    JOIN photos p ON ls.photo_id = p.id
                    WHERE ls.status = 'pending' AND ls.deleted_at IS NULL
                    ORDER BY ls.created_at DESC
                    LIMIT ?
                """, (limit,))
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_suggestions_for_photo(self, photo_id: int) -> List[Dict]:
        """Get all non-deleted suggestions for a photo."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM label_suggestions
                WHERE photo_id = ? AND deleted_at IS NULL
                ORDER BY created_at
            """, (photo_id,))
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def push_to_supabase(self, supabase_client, progress_callback=None, force_full=False) -> Dict[str, int]:
        """Push changed local data to Supabase using batch operations. Returns counts of items pushed.

        Args:
            supabase_client: Supabase client instance
            progress_callback: Optional callable(step, total, message) for progress updates
            force_full: If True, push all data regardless of last sync time
        """
        def report(step, msg):
            if progress_callback:
                progress_callback(step, 9, msg)
            print(msg)

        # Ensure all required columns exist before syncing (handles old databases)
        # These write to DB so need the lock
        with self._lock:
            self._ensure_deer_columns()
            self._ensure_sync_tracking()

        # Check schema_version — force full sync after conflict resolution migration
        schema_version_bump = False
        try:
            with self._lock:
                sv_cursor = self.conn.cursor()
                sv_cursor.execute("SELECT schema_version FROM sync_state WHERE id = 1")
                sv_row = sv_cursor.fetchone()
                if sv_row and (sv_row[0] or 1) < 2:
                    force_full = True
                    schema_version_bump = True
                    logger.info("Schema version < 2: forcing full push to upload sync_ids")
        except Exception:
            pass

        # Get last sync time for incremental push
        last_sync = None if force_full else self.get_last_sync_time('push')
        sync_mode = "full" if last_sync is None else "incremental"
        report(0, f"Starting {sync_mode} sync...")

        # Calculate file hashes for photos that don't have one (for cross-computer matching)
        hash_stats = self.get_hash_stats()
        if hash_stats["missing"] > 0:
            print(f"Calculating file hashes for {hash_stats['missing']} photos...")
            with self._lock:
                self.calculate_missing_hashes()

        from datetime import datetime
        counts = {"photos": 0, "tags": 0, "deer_metadata": 0, "deer_additional": 0,
                  "buck_profiles": 0, "buck_profile_seasons": 0, "annotation_boxes": 0,
                  "cameras": 0, "camera_permissions": 0, "clubs": 0, "club_memberships": 0,
                  "camera_club_shares": 0, "label_suggestions": 0}

        # Use a separate read-only connection for sync queries to avoid blocking main thread
        # This prevents "cannot commit transaction - SQL statements in progress" errors
        sync_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        sync_conn.row_factory = sqlite3.Row

        try:
            cursor = sync_conn.cursor()
            # Use ISO format with UTC timezone suffix to match Supabase TIMESTAMPTZ
            # Use space-separated format without timezone to match SQLite's datetime('now')
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            BATCH_SIZE = 500  # Supabase handles batches well

            # Build WHERE clause for incremental sync (parameterized to prevent SQL injection)
            # Normalize last_sync to match SQLite datetime format (space instead of T)
            normalized_sync = last_sync.replace('T', ' ').replace('+00:00', '') if last_sync else None

            def since_clause(table_alias=""):
                """Returns (clause_string, params_tuple) for parameterized query."""
                if last_sync is None:
                    return ("", ())
                prefix = f"{table_alias}." if table_alias else ""
                return (f" AND {prefix}updated_at > ?", (normalized_sync,))

            def batch_upsert(table_name, data_list, on_conflict):
                """Upsert data in batches."""
                for i in range(0, len(data_list), BATCH_SIZE):
                    batch = data_list[i:i + BATCH_SIZE]
                    if batch:
                        supabase_client.table(table_name).upsert(batch, on_conflict=on_conflict).execute()

            def normalize_datetime(dt_str):
                """Normalize datetime string to ISO format with T separator."""
                if dt_str and ' ' in dt_str and 'T' not in dt_str:
                    return dt_str.replace(' ', 'T')
                return dt_str

            # Push photos_sync (photo identifiers and basic metadata)
            report(1, "Syncing photos...")
            clause, params = since_clause()
            query = "SELECT * FROM photos WHERE 1=1" + clause
            cursor.execute(query, params)
            photos_data = []
            for row in cursor.fetchall():
                photo = dict(row)
                photo_key = self._make_photo_key(photo)
                photos_data.append({
                    "photo_key": photo_key,
                    "file_hash": photo.get("file_hash"),  # For cross-computer matching
                    "original_name": photo.get("original_name"),
                    "date_taken": normalize_datetime(photo.get("date_taken")),
                    "camera_model": photo.get("camera_model"),
                    "camera_location": photo.get("camera_location"),
                    "season_year": photo.get("season_year"),
                    "favorite": bool(photo.get("favorite")),
                    "notes": photo.get("notes"),
                    "collection": photo.get("collection"),  # Club/collection name
                    "r2_photo_id": str(photo.get("id")),  # Local photo ID for R2 URL mapping
                    "archived": bool(photo.get("archived")),  # Hide from mobile by default
                    "updated_at": normalize_datetime(photo.get("updated_at") or now)
                })
            batch_upsert("photos_sync", photos_data, "file_hash")
            counts["photos"] = len(photos_data)

            # Push tags (including tombstones for soft-deleted tags)
            report(2, "Syncing tags...")
            tag_clause, tag_params = since_clause('t')
            cursor.execute(f"""
                SELECT t.tag_name, t.updated_at, t.deleted_at,
                       p.original_name, p.date_taken, p.camera_model, p.file_hash
                FROM tags t
                JOIN photos p ON t.photo_id = p.id
                WHERE 1=1 {tag_clause}
            """, tag_params)
            tags_data = []
            for row in cursor.fetchall():
                t = dict(row)
                photo_key = f"{t['original_name']}|{t['date_taken']}|{t['camera_model']}"
                tags_data.append({
                    "photo_key": photo_key,
                    "file_hash": t["file_hash"],
                    "tag_name": t["tag_name"],
                    "deleted_at": normalize_datetime(t["deleted_at"]) if t["deleted_at"] else None,
                    "updated_at": normalize_datetime(t.get("updated_at") or now)
                })
            batch_upsert("tags", tags_data, "file_hash,tag_name")
            counts["tags"] = len(tags_data)

            # Push deer_metadata
            report(3, "Syncing deer metadata...")
            deer_clause, deer_params = since_clause('d')
            cursor.execute(f"""
                SELECT d.*, p.original_name, p.date_taken, p.camera_model, p.file_hash
                FROM deer_metadata d
                JOIN photos p ON d.photo_id = p.id
                WHERE 1=1 {deer_clause}
            """, deer_params)
            deer_meta_data = []
            for row in cursor.fetchall():
                d = dict(row)
                photo_key = f"{d['original_name']}|{d['date_taken']}|{d['camera_model']}"
                deer_meta_data.append({
                    "photo_key": photo_key,
                    "file_hash": d.get("file_hash"),
                    "deer_id": d.get("deer_id"),
                    "age_class": d.get("age_class"),
                    "left_points_min": d.get("left_points_min"),
                    "left_points_max": d.get("left_points_max"),
                    "right_points_min": d.get("right_points_min"),
                    "right_points_max": d.get("right_points_max"),
                    "left_points_uncertain": bool(d.get("left_points_uncertain")),
                    "right_points_uncertain": bool(d.get("right_points_uncertain")),
                    "left_ab_points_min": d.get("left_ab_points_min"),
                    "left_ab_points_max": d.get("left_ab_points_max"),
                    "right_ab_points_min": d.get("right_ab_points_min"),
                    "right_ab_points_max": d.get("right_ab_points_max"),
                    "abnormal_points_min": d.get("abnormal_points_min"),
                    "abnormal_points_max": d.get("abnormal_points_max"),
                    "broken_antler_side": d.get("broken_antler_side"),
                    "broken_antler_note": d.get("broken_antler_note"),
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            batch_upsert("deer_metadata", deer_meta_data, "file_hash")
            counts["deer_metadata"] = len(deer_meta_data)

            # Push deer_additional
            report(4, "Syncing additional deer...")
            deer_add_clause, deer_add_params = since_clause('d')
            cursor.execute(f"""
                SELECT d.*, p.original_name, p.date_taken, p.camera_model, p.file_hash
                FROM deer_additional d
                JOIN photos p ON d.photo_id = p.id
                WHERE 1=1 {deer_add_clause}
            """, deer_add_params)
            deer_add_data = []
            for row in cursor.fetchall():
                d = dict(row)
                photo_key = f"{d['original_name']}|{d['date_taken']}|{d['camera_model']}"
                deer_add_data.append({
                    "photo_key": photo_key,
                    "file_hash": d.get("file_hash"),
                    "deer_id": d.get("deer_id"),
                    "age_class": d.get("age_class"),
                    "left_points_min": d.get("left_points_min"),
                    "left_points_max": d.get("left_points_max"),
                    "right_points_min": d.get("right_points_min"),
                    "right_points_max": d.get("right_points_max"),
                    "left_points_uncertain": bool(d.get("left_points_uncertain")),
                    "right_points_uncertain": bool(d.get("right_points_uncertain")),
                    "left_ab_points_min": d.get("left_ab_points_min"),
                    "left_ab_points_max": d.get("left_ab_points_max"),
                    "right_ab_points_min": d.get("right_ab_points_min"),
                    "right_ab_points_max": d.get("right_ab_points_max"),
                    "abnormal_points_min": d.get("abnormal_points_min"),
                    "abnormal_points_max": d.get("abnormal_points_max"),
                    "broken_antler_side": d.get("broken_antler_side"),
                    "broken_antler_note": d.get("broken_antler_note"),
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            batch_upsert("deer_additional", deer_add_data, "file_hash,deer_id")
            counts["deer_additional"] = len(deer_add_data)

            # Push buck_profiles
            report(5, "Syncing buck profiles...")
            clause, params = since_clause()
            cursor.execute(f"SELECT * FROM buck_profiles WHERE 1=1{clause}", params)
            profiles_data = []
            for row in cursor.fetchall():
                d = dict(row)
                profiles_data.append({
                    "deer_id": d.get("deer_id"),
                    "display_name": d.get("display_name"),
                    "notes": d.get("notes"),
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            batch_upsert("buck_profiles", profiles_data, "deer_id")
            counts["buck_profiles"] = len(profiles_data)

            # Push buck_profile_seasons
            report(6, "Syncing buck profile seasons...")
            clause, params = since_clause()
            cursor.execute(f"SELECT * FROM buck_profile_seasons WHERE 1=1{clause}", params)
            seasons_data = []
            for row in cursor.fetchall():
                d = dict(row)
                seasons_data.append({
                    "deer_id": d.get("deer_id"),
                    "season_year": d.get("season_year"),
                    "camera_locations": d.get("camera_locations"),
                    "key_characteristics": d.get("key_characteristics"),
                    "left_points_min": d.get("left_points_min"),
                    "left_points_max": d.get("left_points_max"),
                    "right_points_min": d.get("right_points_min"),
                    "right_points_max": d.get("right_points_max"),
                    "left_ab_points_min": d.get("left_ab_points_min"),
                    "left_ab_points_max": d.get("left_ab_points_max"),
                    "right_ab_points_min": d.get("right_ab_points_min"),
                    "right_ab_points_max": d.get("right_ab_points_max"),
                    "abnormal_points_min": d.get("abnormal_points_min"),
                    "abnormal_points_max": d.get("abnormal_points_max"),
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            batch_upsert("buck_profile_seasons", seasons_data, "deer_id,season_year")
            counts["buck_profile_seasons"] = len(seasons_data)

            # Push annotation_boxes - but ONLY if we have a substantial local database
            # This prevents accidentally wiping cloud boxes when pushing from an empty/new computer
            report(7, "Syncing annotation boxes...")
            cursor.execute("SELECT COUNT(*) FROM annotation_boxes")
            local_box_count = cursor.fetchone()[0]

            # Check if cloud annotation_boxes is empty - if so, need full sync
            cloud_box_count = 0
            try:
                result = supabase_client.table("annotation_boxes").select("id").execute()
                cloud_box_count = len(result.data) if result.data else 0
            except Exception:
                pass  # If check fails, assume we need incremental sync

            # Safety check: don't delete cloud boxes if local has very few
            # (protects against pushing from empty/new database)
            if local_box_count == 0 and cloud_box_count > 0:
                print("  Skipping box sync - local box table is empty but cloud has data.")
                print("  Run 'Pull from Cloud' first if this is a new computer.")
                counts["annotation_boxes"] = 0
            else:
                # If cloud is empty but local has data, do full sync (skip since_clause)
                force_full_sync = (cloud_box_count == 0 and local_box_count > 0)
                if force_full_sync:
                    print(f"  Cloud annotation_boxes is empty but local has {local_box_count} - doing full sync")

                if force_full_sync:
                    box_filter, box_params = "", ()
                else:
                    box_filter, box_params = since_clause('b')
                cursor.execute(f"""
                    SELECT b.*, p.original_name, p.date_taken, p.camera_model, p.file_hash
                    FROM annotation_boxes b
                    JOIN photos p ON b.photo_id = p.id
                    WHERE 1=1 {box_filter}
                """, box_params)
                boxes_data = []
                for row in cursor.fetchall():
                    d = dict(row)
                    photo_key = f"{d['original_name']}|{d['date_taken']}|{d['camera_model']}"
                    boxes_data.append({
                        "sync_id": d.get("sync_id"),
                        "photo_key": photo_key,
                        "file_hash": d.get("file_hash"),
                        "label": d.get("label"),
                        "x1": d.get("x1"),
                        "y1": d.get("y1"),
                        "x2": d.get("x2"),
                        "y2": d.get("y2"),
                        "confidence": d.get("confidence"),
                        "species": d.get("species"),
                        "species_conf": d.get("species_conf"),
                        "sex": d.get("sex"),
                        "sex_conf": d.get("sex_conf"),
                        "deleted_at": normalize_datetime(d.get("deleted_at")) if d.get("deleted_at") else None,
                        "updated_at": normalize_datetime(d.get("updated_at") or now)
                    })
                # Upsert by sync_id — handles both full and incremental sync
                if boxes_data:
                    batch_upsert("annotation_boxes", boxes_data, "sync_id")
                counts["annotation_boxes"] = len(boxes_data)

            # Push cameras
            report(8, "Syncing cameras...")
            clause, params = since_clause()
            cursor.execute(f"SELECT * FROM cameras WHERE 1=1{clause}", params)
            cameras_data = []
            for row in cursor.fetchall():
                d = dict(row)
                cameras_data.append({
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "owner": d.get("owner"),
                    "verified": bool(d.get("verified")),
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            if cameras_data:
                batch_upsert("cameras", cameras_data, "id")
            counts["cameras"] = len(cameras_data)

            # Push camera_permissions
            clause, params = since_clause()
            cursor.execute(f"SELECT * FROM camera_permissions WHERE 1=1{clause}", params)
            perms_data = []
            for row in cursor.fetchall():
                d = dict(row)
                perms_data.append({
                    "id": d.get("id"),
                    "camera_id": d.get("camera_id"),
                    "username": d.get("username"),
                    "role": d.get("role"),
                    "granted_by": d.get("granted_by"),
                    "granted_at": normalize_datetime(d.get("granted_at")),
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            if perms_data:
                batch_upsert("camera_permissions", perms_data, "id")
            counts["camera_permissions"] = len(perms_data)

            # Push clubs
            clause, params = since_clause()
            cursor.execute(f"SELECT * FROM clubs WHERE 1=1{clause}", params)
            clubs_data = []
            for row in cursor.fetchall():
                d = dict(row)
                clubs_data.append({
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "created_by": d.get("created_by"),
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            if clubs_data:
                batch_upsert("clubs", clubs_data, "id")
            counts["clubs"] = len(clubs_data)

            # Push club_memberships
            clause, params = since_clause()
            cursor.execute(f"SELECT * FROM club_memberships WHERE 1=1{clause}", params)
            memberships_data = []
            for row in cursor.fetchall():
                d = dict(row)
                memberships_data.append({
                    "id": d.get("id"),
                    "club_id": d.get("club_id"),
                    "username": d.get("username"),
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            if memberships_data:
                batch_upsert("club_memberships", memberships_data, "id")
            counts["club_memberships"] = len(memberships_data)

            # Push camera_club_shares
            clause, params = since_clause()
            cursor.execute(f"SELECT * FROM camera_club_shares WHERE 1=1{clause}", params)
            shares_data = []
            for row in cursor.fetchall():
                d = dict(row)
                shares_data.append({
                    "id": d.get("id"),
                    "camera_id": d.get("camera_id"),
                    "club_id": d.get("club_id"),
                    "shared_by": d.get("shared_by"),
                    "visibility": d.get("visibility"),
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            if shares_data:
                batch_upsert("camera_club_shares", shares_data, "id")
            counts["camera_club_shares"] = len(shares_data)

            # Push label_suggestions (with tombstones)
            clause, params = since_clause()
            cursor.execute(f"SELECT * FROM label_suggestions WHERE 1=1{clause}", params)
            suggestions_data = []
            for row in cursor.fetchall():
                d = dict(row)
                suggestions_data.append({
                    "id": d.get("id"),
                    "file_hash": d.get("file_hash"),
                    "tag_name": d.get("tag_name"),
                    "suggested_by": d.get("suggested_by"),
                    "status": d.get("status"),
                    "reviewed_by": d.get("reviewed_by"),
                    "reviewed_at": normalize_datetime(d.get("reviewed_at")),
                    "camera_id": d.get("camera_id"),
                    "deleted_at": normalize_datetime(d.get("deleted_at")) if d.get("deleted_at") else None,
                    "updated_at": normalize_datetime(d.get("updated_at") or now)
                })
            if suggestions_data:
                batch_upsert("label_suggestions", suggestions_data, "id")
            counts["label_suggestions"] = len(suggestions_data)

            report(9, "Sync complete!")

            # Update last sync time after successful push
            with self._lock:
                self.set_last_sync_time('push', now)
                # Bump schema_version after first successful full push with sync_ids
                if schema_version_bump:
                    self.conn.cursor().execute("UPDATE sync_state SET schema_version = 2 WHERE id = 1")
                    self.conn.commit()
                    logger.info("Schema version bumped to 2 (conflict resolution migration complete)")

            # Garbage-collect old tombstones after successful sync
            self.purge_old_tombstones(days=30)

            return counts
        finally:
            # Always close the dedicated sync connection
            sync_conn.close()

    def pull_from_supabase(self, supabase_client, progress_callback=None, force_full=False) -> Dict[str, int]:
        """Pull data from Supabase and update local database. Returns counts.

        Args:
            supabase_client: Supabase client instance
            progress_callback: Optional callable(step, total, message) for progress updates
            force_full: If True, pull all data regardless of last sync time
        """
        from datetime import datetime

        def report(step, msg):
            if progress_callback:
                progress_callback(step, 10, msg)
            print(msg)

        # Ensure all required columns exist before syncing (handles old databases)
        self._ensure_deer_columns()

        # Get last sync time for incremental pull
        last_sync = None if force_full else self.get_last_sync_time('pull')
        sync_mode = "full" if last_sync is None else "incremental"
        report(0, f"Starting {sync_mode} pull...")

        # Use ISO format with UTC timezone suffix to match Supabase TIMESTAMPTZ
        now = datetime.utcnow().isoformat() + "+00:00"

        counts = {"photos": 0, "tags": 0, "deer_metadata": 0, "deer_additional": 0,
                  "buck_profiles": 0, "buck_profile_seasons": 0, "annotation_boxes": 0,
                  "cameras": 0, "camera_permissions": 0, "clubs": 0, "club_memberships": 0,
                  "camera_club_shares": 0, "label_suggestions": 0}

        cursor = self.conn.cursor()

        def fetch_table(table_name):
            """Fetch from table with optional incremental filter."""
            query = supabase_client.table(table_name).select("*")
            if last_sync:
                query = query.gt("updated_at", last_sync)
            return query.execute(fetch_all=True)

        # Pull photos_sync - update existing or create new cloud-only stubs
        report(1, "Pulling photos...")
        result = fetch_table("photos_sync")
        counts["photos_created"] = 0
        for row in result.data:
            photo = self._get_photo_by_key(row["photo_key"], row.get("file_hash"))
            if photo:
                # Last-write-wins: only update if cloud is newer
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))
                local_updated = self._normalize_ts(photo.get("updated_at", ""))
                if cloud_updated > local_updated:
                    cloud_date = row.get("date_taken")
                    cursor.execute("""
                        UPDATE photos SET
                            camera_location = ?,
                            favorite = COALESCE(?, favorite),
                            notes = ?,
                            date_taken = COALESCE(?, date_taken),
                            updated_at = ?
                        WHERE id = ?
                    """, (row.get("camera_location"), row.get("favorite"), row.get("notes"),
                          cloud_date, self._normalize_ts(row.get("updated_at")), photo["id"]))
                counts["photos"] += 1
            else:
                # Create cloud-only stub for photos that don't exist locally
                file_hash = row.get("file_hash")
                if file_hash:
                    new_id = self.add_cloud_photo(row)
                    if new_id:
                        counts["photos_created"] += 1

        # Pull tags (tombstone-aware + last-write-wins)
        report(2, "Pulling tags...")
        result = fetch_table("tags")
        for row in result.data:
            photo = self._get_photo_by_key(row["photo_key"], row.get("file_hash"))
            if photo:
                cloud_deleted = self._normalize_ts(row.get("deleted_at"))
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))

                cursor.execute("SELECT id, deleted_at, updated_at FROM tags WHERE photo_id = ? AND tag_name = ?",
                              (photo["id"], row["tag_name"]))
                local = cursor.fetchone()

                if local:
                    local_updated = self._normalize_ts(local[2] or "")
                    if cloud_updated > local_updated:
                        cursor.execute("UPDATE tags SET deleted_at = ?, updated_at = ? WHERE id = ?",
                                      (cloud_deleted or None, cloud_updated, local[0]))
                else:
                    cursor.execute("INSERT INTO tags (photo_id, tag_name, deleted_at, updated_at) VALUES (?, ?, ?, ?)",
                                  (photo["id"], row["tag_name"], cloud_deleted or None, cloud_updated))
                counts["tags"] += 1

        # Pull deer_metadata (last-write-wins)
        report(3, "Pulling deer metadata...")
        result = fetch_table("deer_metadata")
        for row in result.data:
            photo = self._get_photo_by_key(row["photo_key"], row.get("file_hash"))
            if photo:
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))
                cursor.execute("SELECT updated_at FROM deer_metadata WHERE photo_id = ?", (photo["id"],))
                local = cursor.fetchone()
                if local:
                    local_updated = self._normalize_ts(local[0] or "")
                    if cloud_updated <= local_updated:
                        counts["deer_metadata"] += 1
                        continue
                cursor.execute("""
                    INSERT OR REPLACE INTO deer_metadata
                    (photo_id, deer_id, age_class, left_points_min, left_points_max,
                     right_points_min, right_points_max, left_points_uncertain, right_points_uncertain,
                     left_ab_points_min, left_ab_points_max, right_ab_points_min, right_ab_points_max,
                     abnormal_points_min, abnormal_points_max, broken_antler_side, broken_antler_note,
                     updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (photo["id"], row.get("deer_id"), row.get("age_class"),
                      row.get("left_points_min"), row.get("left_points_max"),
                      row.get("right_points_min"), row.get("right_points_max"),
                      row.get("left_points_uncertain"), row.get("right_points_uncertain"),
                      row.get("left_ab_points_min"), row.get("left_ab_points_max"),
                      row.get("right_ab_points_min"), row.get("right_ab_points_max"),
                      row.get("abnormal_points_min"), row.get("abnormal_points_max"),
                      row.get("broken_antler_side"), row.get("broken_antler_note"),
                      self._normalize_ts(row.get("updated_at"))))
                counts["deer_metadata"] += 1

        # Pull deer_additional (last-write-wins)
        report(4, "Pulling additional deer...")
        result = fetch_table("deer_additional")
        for row in result.data:
            photo = self._get_photo_by_key(row["photo_key"], row.get("file_hash"))
            if photo and row.get("deer_id"):
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))
                cursor.execute("SELECT updated_at FROM deer_additional WHERE photo_id = ? AND deer_id = ?",
                              (photo["id"], row["deer_id"]))
                local = cursor.fetchone()
                if local:
                    local_updated = self._normalize_ts(local[0] or "")
                    if cloud_updated <= local_updated:
                        counts["deer_additional"] += 1
                        continue
                cursor.execute("""
                    INSERT OR REPLACE INTO deer_additional
                    (photo_id, deer_id, age_class, left_points_min, left_points_max,
                     right_points_min, right_points_max, left_points_uncertain, right_points_uncertain,
                     left_ab_points_min, left_ab_points_max, right_ab_points_min, right_ab_points_max,
                     abnormal_points_min, abnormal_points_max, broken_antler_side, broken_antler_note,
                     updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (photo["id"], row.get("deer_id"), row.get("age_class"),
                      row.get("left_points_min"), row.get("left_points_max"),
                      row.get("right_points_min"), row.get("right_points_max"),
                      row.get("left_points_uncertain"), row.get("right_points_uncertain"),
                      row.get("left_ab_points_min"), row.get("left_ab_points_max"),
                      row.get("right_ab_points_min"), row.get("right_ab_points_max"),
                      row.get("abnormal_points_min"), row.get("abnormal_points_max"),
                      row.get("broken_antler_side"), row.get("broken_antler_note"),
                      self._normalize_ts(row.get("updated_at"))))
                counts["deer_additional"] += 1

        # Pull buck_profiles (last-write-wins)
        report(5, "Pulling buck profiles...")
        result = fetch_table("buck_profiles")
        for row in result.data:
            if row.get("deer_id"):
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))
                cursor.execute("SELECT updated_at FROM buck_profiles WHERE deer_id = ?", (row["deer_id"],))
                local = cursor.fetchone()
                if local:
                    local_updated = self._normalize_ts(local[0] or "")
                    if cloud_updated <= local_updated:
                        counts["buck_profiles"] += 1
                        continue
                cursor.execute("""
                    INSERT OR REPLACE INTO buck_profiles (deer_id, display_name, notes, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (row["deer_id"], row.get("display_name"), row.get("notes"),
                      self._normalize_ts(row.get("updated_at"))))
                counts["buck_profiles"] += 1

        # Pull buck_profile_seasons (last-write-wins)
        report(6, "Pulling buck profile seasons...")
        result = fetch_table("buck_profile_seasons")
        for row in result.data:
            if row.get("deer_id") and row.get("season_year"):
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))
                cursor.execute("SELECT updated_at FROM buck_profile_seasons WHERE deer_id = ? AND season_year = ?",
                              (row["deer_id"], row["season_year"]))
                local = cursor.fetchone()
                if local:
                    local_updated = self._normalize_ts(local[0] or "")
                    if cloud_updated <= local_updated:
                        counts["buck_profile_seasons"] += 1
                        continue
                cursor.execute("""
                    INSERT OR REPLACE INTO buck_profile_seasons
                    (deer_id, season_year, camera_locations, key_characteristics,
                     left_points_min, left_points_max, right_points_min, right_points_max,
                     left_ab_points_min, left_ab_points_max, right_ab_points_min, right_ab_points_max,
                     abnormal_points_min, abnormal_points_max, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (row["deer_id"], row["season_year"], row.get("camera_locations"),
                      row.get("key_characteristics"), row.get("left_points_min"), row.get("left_points_max"),
                      row.get("right_points_min"), row.get("right_points_max"),
                      row.get("left_ab_points_min"), row.get("left_ab_points_max"),
                      row.get("right_ab_points_min"), row.get("right_ab_points_max"),
                      row.get("abnormal_points_min"), row.get("abnormal_points_max"),
                      self._normalize_ts(row.get("updated_at"))))
                counts["buck_profile_seasons"] += 1

        # Pull annotation_boxes (sync_id-based + last-write-wins + tombstones)
        report(7, "Pulling annotation boxes...")
        result = fetch_table("annotation_boxes")

        import uuid as _uuid

        def safe_float(val):
            """Safely convert value to float, handling bytes/blob corruption."""
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, bytes):
                import struct
                try:
                    if len(val) == 4:
                        return struct.unpack('<f', val)[0]
                    elif len(val) == 8:
                        return struct.unpack('<d', val)[0]
                except:
                    pass
                return None
            try:
                return float(val)
            except:
                return None

        for row in result.data:
            photo = self._get_photo_by_key(row["photo_key"], row.get("file_hash"))
            if not photo:
                continue

            x1 = safe_float(row.get("x1"))
            y1 = safe_float(row.get("y1"))
            x2 = safe_float(row.get("x2"))
            y2 = safe_float(row.get("y2"))

            if None in (x1, y1, x2, y2):
                print(f"  Skipping box with invalid coordinates: {row.get('photo_key')}")
                continue

            sync_id = row.get("sync_id")
            cloud_updated = self._normalize_ts(row.get("updated_at", ""))
            cloud_deleted = self._normalize_ts(row.get("deleted_at")) or None

            if sync_id:
                # Preferred path: match by sync_id
                cursor.execute("SELECT id, updated_at FROM annotation_boxes WHERE sync_id = ?", (sync_id,))
                existing = cursor.fetchone()

                if existing:
                    local_updated = self._normalize_ts(existing[1] or "")
                    if cloud_updated > local_updated:
                        cursor.execute("""
                            UPDATE annotation_boxes SET
                                label=?, x1=?, y1=?, x2=?, y2=?, confidence=?,
                                species=?, species_conf=?, sex=?, sex_conf=?,
                                head_x1=?, head_y1=?, head_x2=?, head_y2=?, head_notes=?,
                                deleted_at=?, updated_at=?
                            WHERE sync_id = ?
                        """, (row.get("label"), x1, y1, x2, y2,
                              safe_float(row.get("confidence")),
                              row.get("species"), safe_float(row.get("species_conf")),
                              row.get("sex"), safe_float(row.get("sex_conf")),
                              safe_float(row.get("head_x1")), safe_float(row.get("head_y1")),
                              safe_float(row.get("head_x2")), safe_float(row.get("head_y2")),
                              row.get("head_notes"), cloud_deleted, cloud_updated,
                              sync_id))
                else:
                    # New box from cloud
                    cursor.execute("""
                        INSERT INTO annotation_boxes
                        (photo_id, sync_id, label, x1, y1, x2, y2, confidence,
                         species, species_conf, sex, sex_conf,
                         head_x1, head_y1, head_x2, head_y2, head_notes,
                         deleted_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (photo["id"], sync_id, row.get("label"), x1, y1, x2, y2,
                          safe_float(row.get("confidence")),
                          row.get("species"), safe_float(row.get("species_conf")),
                          row.get("sex"), safe_float(row.get("sex_conf")),
                          safe_float(row.get("head_x1")), safe_float(row.get("head_y1")),
                          safe_float(row.get("head_x2")), safe_float(row.get("head_y2")),
                          row.get("head_notes"), cloud_deleted, cloud_updated))
            else:
                # Legacy box without sync_id — fall back to coordinate matching
                cursor.execute("""
                    SELECT id FROM annotation_boxes
                    WHERE photo_id = ? AND label = ? AND x1 = ? AND y1 = ? AND x2 = ? AND y2 = ?
                """, (photo["id"], row.get("label"), x1, y1, x2, y2))
                match = cursor.fetchone()
                if match:
                    # Assign a sync_id so future syncs use stable identity
                    new_sync_id = str(_uuid.uuid4())
                    cursor.execute("UPDATE annotation_boxes SET sync_id = ?, updated_at = ? WHERE id = ?",
                                  (new_sync_id, cloud_updated, match[0]))
                else:
                    # Insert as new with a sync_id (include deleted_at for tombstoned cloud boxes)
                    cursor.execute("""
                        INSERT INTO annotation_boxes
                        (photo_id, label, x1, y1, x2, y2, confidence, species, species_conf,
                         sex, sex_conf, head_x1, head_y1, head_x2, head_y2, head_notes,
                         sync_id, deleted_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (photo["id"], row.get("label"), x1, y1, x2, y2,
                          safe_float(row.get("confidence")),
                          row.get("species"), safe_float(row.get("species_conf")),
                          row.get("sex"), safe_float(row.get("sex_conf")),
                          safe_float(row.get("head_x1")), safe_float(row.get("head_y1")),
                          safe_float(row.get("head_x2")), safe_float(row.get("head_y2")),
                          row.get("head_notes"), str(_uuid.uuid4()),
                          cloud_deleted, cloud_updated))
            counts["annotation_boxes"] += 1

        # Tombstone-based deletions replace the old destructive set-subtraction
        # Deletions are now synced as tombstones (deleted_at timestamps) in both
        # tags and annotation_boxes, handled by the LWW pull logic above

        # Pull cameras
        report(8, "Pulling cameras...")
        try:
            result = fetch_table("cameras")
            for row in result.data:
                camera_id = row.get("id")
                if not camera_id:
                    continue
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))

                cursor.execute("SELECT id, updated_at FROM cameras WHERE id = ?", (camera_id,))
                existing = cursor.fetchone()

                if existing:
                    local_updated = self._normalize_ts(existing[1] or "")
                    if cloud_updated > local_updated:
                        cursor.execute("""
                            UPDATE cameras SET name=?, owner=?, verified=?, updated_at=?
                            WHERE id = ?
                        """, (row.get("name"), row.get("owner"), row.get("verified"),
                              cloud_updated, camera_id))
                else:
                    cursor.execute("""
                        INSERT INTO cameras (id, name, owner, verified, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (camera_id, row.get("name"), row.get("owner"),
                          row.get("verified"), cloud_updated))
                counts["cameras"] = counts.get("cameras", 0) + 1
        except Exception as e:
            print(f"  Warning: Could not pull cameras: {e}")

        # Pull camera_permissions
        try:
            result = fetch_table("camera_permissions")
            for row in result.data:
                perm_id = row.get("id")
                if not perm_id:
                    continue
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))

                cursor.execute("SELECT id, updated_at FROM camera_permissions WHERE id = ?", (perm_id,))
                existing = cursor.fetchone()

                if existing:
                    local_updated = self._normalize_ts(existing[1] or "")
                    if cloud_updated > local_updated:
                        cursor.execute("""
                            UPDATE camera_permissions SET camera_id=?, username=?, role=?,
                                granted_by=?, granted_at=?, updated_at=?
                            WHERE id = ?
                        """, (row.get("camera_id"), row.get("username"), row.get("role"),
                              row.get("granted_by"), self._normalize_ts(row.get("granted_at")),
                              cloud_updated, perm_id))
                else:
                    cursor.execute("""
                        INSERT INTO camera_permissions (id, camera_id, username, role, granted_by, granted_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (perm_id, row.get("camera_id"), row.get("username"), row.get("role"),
                          row.get("granted_by"), self._normalize_ts(row.get("granted_at")),
                          cloud_updated))
                counts["camera_permissions"] = counts.get("camera_permissions", 0) + 1
        except Exception as e:
            print(f"  Warning: Could not pull camera_permissions: {e}")

        # Pull clubs
        try:
            result = fetch_table("clubs")
            for row in result.data:
                club_id = row.get("id")
                if not club_id:
                    continue
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))

                cursor.execute("SELECT id, updated_at FROM clubs WHERE id = ?", (club_id,))
                existing = cursor.fetchone()

                if existing:
                    local_updated = self._normalize_ts(existing[1] or "")
                    if cloud_updated > local_updated:
                        cursor.execute("""
                            UPDATE clubs SET name=?, created_by=?, updated_at=?
                            WHERE id = ?
                        """, (row.get("name"), row.get("created_by"), cloud_updated, club_id))
                else:
                    cursor.execute("""
                        INSERT INTO clubs (id, name, created_by, updated_at)
                        VALUES (?, ?, ?, ?)
                    """, (club_id, row.get("name"), row.get("created_by"), cloud_updated))
                counts["clubs"] = counts.get("clubs", 0) + 1
        except Exception as e:
            print(f"  Warning: Could not pull clubs: {e}")

        # Pull club_memberships
        try:
            result = fetch_table("club_memberships")
            for row in result.data:
                mem_id = row.get("id")
                if not mem_id:
                    continue
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))

                cursor.execute("SELECT id, updated_at FROM club_memberships WHERE id = ?", (mem_id,))
                existing = cursor.fetchone()

                if existing:
                    local_updated = self._normalize_ts(existing[1] or "")
                    if cloud_updated > local_updated:
                        cursor.execute("""
                            UPDATE club_memberships SET club_id=?, username=?, updated_at=?
                            WHERE id = ?
                        """, (row.get("club_id"), row.get("username"), cloud_updated, mem_id))
                else:
                    cursor.execute("""
                        INSERT INTO club_memberships (id, club_id, username, updated_at)
                        VALUES (?, ?, ?, ?)
                    """, (mem_id, row.get("club_id"), row.get("username"), cloud_updated))
                counts["club_memberships"] = counts.get("club_memberships", 0) + 1
        except Exception as e:
            print(f"  Warning: Could not pull club_memberships: {e}")

        # Pull camera_club_shares
        try:
            result = fetch_table("camera_club_shares")
            for row in result.data:
                share_id = row.get("id")
                if not share_id:
                    continue
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))

                cursor.execute("SELECT id, updated_at FROM camera_club_shares WHERE id = ?", (share_id,))
                existing = cursor.fetchone()

                if existing:
                    local_updated = self._normalize_ts(existing[1] or "")
                    if cloud_updated > local_updated:
                        cursor.execute("""
                            UPDATE camera_club_shares SET camera_id=?, club_id=?, shared_by=?,
                                visibility=?, updated_at=?
                            WHERE id = ?
                        """, (row.get("camera_id"), row.get("club_id"), row.get("shared_by"),
                              row.get("visibility"), cloud_updated, share_id))
                else:
                    cursor.execute("""
                        INSERT INTO camera_club_shares (id, camera_id, club_id, shared_by, visibility, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (share_id, row.get("camera_id"), row.get("club_id"), row.get("shared_by"),
                          row.get("visibility"), cloud_updated))
                counts["camera_club_shares"] = counts.get("camera_club_shares", 0) + 1
        except Exception as e:
            print(f"  Warning: Could not pull camera_club_shares: {e}")

        # Pull label_suggestions (with tombstone support)
        try:
            result = fetch_table("label_suggestions")
            for row in result.data:
                sug_id = row.get("id")
                if not sug_id:
                    continue
                cloud_updated = self._normalize_ts(row.get("updated_at", ""))
                cloud_deleted = self._normalize_ts(row.get("deleted_at")) or None

                cursor.execute("SELECT id, updated_at FROM label_suggestions WHERE id = ?", (sug_id,))
                existing = cursor.fetchone()

                if existing:
                    local_updated = self._normalize_ts(existing[1] or "")
                    if cloud_updated > local_updated:
                        cursor.execute("""
                            UPDATE label_suggestions SET file_hash=?, tag_name=?, suggested_by=?,
                                status=?, reviewed_by=?, reviewed_at=?, camera_id=?,
                                deleted_at=?, updated_at=?
                            WHERE id = ?
                        """, (row.get("file_hash"), row.get("tag_name"), row.get("suggested_by"),
                              row.get("status"), row.get("reviewed_by"),
                              self._normalize_ts(row.get("reviewed_at")),
                              row.get("camera_id"), cloud_deleted, cloud_updated, sug_id))
                else:
                    cursor.execute("""
                        INSERT INTO label_suggestions (id, file_hash, tag_name, suggested_by,
                            status, reviewed_by, reviewed_at, camera_id, deleted_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (sug_id, row.get("file_hash"), row.get("tag_name"), row.get("suggested_by"),
                          row.get("status"), row.get("reviewed_by"),
                          self._normalize_ts(row.get("reviewed_at")),
                          row.get("camera_id"), cloud_deleted, cloud_updated))
                counts["label_suggestions"] = counts.get("label_suggestions", 0) + 1
        except Exception as e:
            print(f"  Warning: Could not pull label_suggestions: {e}")

        report(9, "Finalizing...")

        report(10, "Sync complete!")
        self.conn.commit()

        # Save pull sync time for next incremental pull
        self.set_last_sync_time('pull', now)

        return counts

    # ─────────────────────────────────────────────────────────────────────
    # Claude Review Queue Methods
    # ─────────────────────────────────────────────────────────────────────

    def add_to_claude_review(self, photo_id: int, reason: str, priority: int = 0):
        """Add a photo to the Claude review queue."""
        cursor = self.conn.cursor()
        # Check if already in queue
        cursor.execute("SELECT id FROM claude_review_queue WHERE photo_id = ? AND reviewed_at IS NULL", (photo_id,))
        if cursor.fetchone():
            return  # Already in queue
        cursor.execute(
            "INSERT INTO claude_review_queue (photo_id, reason, priority) VALUES (?, ?, ?)",
            (photo_id, reason, priority)
        )
        self.conn.commit()

    def add_many_to_claude_review(self, photo_ids: List[int], reason: str, priority: int = 0):
        """Add multiple photos to the Claude review queue."""
        cursor = self.conn.cursor()
        for photo_id in photo_ids:
            cursor.execute("SELECT id FROM claude_review_queue WHERE photo_id = ? AND reviewed_at IS NULL", (photo_id,))
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO claude_review_queue (photo_id, reason, priority) VALUES (?, ?, ?)",
                    (photo_id, reason, priority)
                )
        self.conn.commit()

    def get_claude_review_queue(self, limit: int = None) -> List[Dict]:
        """Get pending photos in the Claude review queue."""
        cursor = self.conn.cursor()
        query = """
            SELECT q.id, q.photo_id, q.reason, q.priority, q.added_at, p.file_path
            FROM claude_review_queue q
            JOIN photos p ON q.photo_id = p.id
            WHERE q.reviewed_at IS NULL
            ORDER BY q.priority DESC, q.added_at ASC
        """
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def get_claude_review_count(self) -> int:
        """Get count of pending items in Claude review queue."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM claude_review_queue WHERE reviewed_at IS NULL")
        return cursor.fetchone()[0]

    def mark_claude_reviewed(self, photo_id: int):
        """Mark a photo as reviewed in the Claude queue."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE claude_review_queue SET reviewed_at = CURRENT_TIMESTAMP WHERE photo_id = ? AND reviewed_at IS NULL",
            (photo_id,)
        )
        self.conn.commit()

    def clear_claude_review_queue(self):
        """Clear all pending items from the Claude review queue."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM claude_review_queue WHERE reviewed_at IS NULL")
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()
