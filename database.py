"""
Database module for managing trail camera photo metadata and tags.
"""
import sqlite3
import os
import time
import threading
import logging
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
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access and performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_database()

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
        # Additional indexes for common query patterns
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_archived ON photos(archived)")
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
        self._ensure_deer_columns()
        self._ensure_box_columns()
        self._ensure_sync_tracking()

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
            if table == "deer_metadata":
                add_column(table, "broken_antler_side", "broken_antler_side TEXT")
                add_column(table, "broken_antler_note", "broken_antler_note TEXT")
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

        # Create indexes for updated_at columns
        for table in sync_tables:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_updated_at ON {table}(updated_at)")

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
                  collection: str = "") -> int:
        """Add a photo to the database.

        Returns:
            Photo ID
        """
        if season_year is None:
            season_year = self.compute_season_year(date_taken)
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO photos
                (file_path, original_name, date_taken, camera_model, import_date, thumbnail_path, favorite, notes, season_year, camera_location, key_characteristics, suggested_tag, suggested_confidence, collection)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_path, original_name, date_taken, camera_model,
                  datetime.now().isoformat(), thumbnail_path, favorite, notes, season_year, camera_location, key_characteristics, suggested_tag, suggested_confidence, collection))
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
    
    def get_photo_path(self, photo_id: int) -> Optional[str]:
        """Get file path for a photo ID."""
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
        cursor = self.conn.cursor()
        cursor.execute("UPDATE photos SET camera_model = ? WHERE id = ?", (camera_model or "", photo_id))
        self.conn.commit()

    def set_date_taken(self, photo_id: int, date_taken: Optional[str]):
        """Update date_taken for a photo."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE photos SET date_taken = ? WHERE id = ?", (date_taken, photo_id))
        self.conn.commit()
    
    def add_tag(self, photo_id: int, tag_name: str):
        """Add a tag to a photo."""
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("INSERT INTO tags (photo_id, tag_name) VALUES (?, ?)",
                             (photo_id, tag_name))
                self.conn.commit()
            except sqlite3.IntegrityError:
                # Tag already exists, ignore
                pass
    
    def update_photo_tags(self, photo_id: int, tags: List[str]):
        """Replace all tags for a photo with the provided list."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM tags WHERE photo_id = ?", (photo_id,))
            cursor.executemany(
                "INSERT INTO tags (photo_id, tag_name) VALUES (?, ?)",
                [(photo_id, tag) for tag in tags]
            )
            self.conn.commit()
    
    def get_photo_tags(self, photo_id: int) -> List[str]:
        """Alias for get_tags; returns all tags for a photo."""
        return self.get_tags(photo_id)
    
    def remove_tag(self, photo_id: int, tag_name: str):
        """Remove a tag from a photo."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM tags WHERE photo_id = ? AND tag_name = ?", 
                      (photo_id, tag_name))
        self.conn.commit()
    
    def get_tags(self, photo_id: int) -> List[str]:
        """Get all tags for a photo."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT tag_name FROM tags WHERE photo_id = ? ORDER BY tag_name",
                      (photo_id,))
        return [row['tag_name'] for row in cursor.fetchall()]

    def get_all_distinct_tags(self) -> List[str]:
        """Get all distinct tag names across all photos."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT tag_name FROM tags ORDER BY tag_name")
        return [row['tag_name'] for row in cursor.fetchall()]

    def set_deer_metadata(self, photo_id: int, deer_id: str = None, age_class: str = None,
                          left_points_min: Optional[int] = None, right_points_min: Optional[int] = None,
                          left_points_uncertain: bool = False, right_points_uncertain: bool = False,
                          left_ab_points_min: Optional[int] = None, right_ab_points_min: Optional[int] = None,
                          left_ab_points_uncertain: bool = False, right_ab_points_uncertain: bool = False,
                          abnormal_points_min: Optional[int] = None, abnormal_points_max: Optional[int] = None,
                          broken_antler_side: Optional[str] = None, broken_antler_note: Optional[str] = None):
        """Set deer metadata for a photo."""
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
        # Move profiles
        cursor.execute("UPDATE OR IGNORE buck_profiles SET deer_id = ? WHERE deer_id = ?", (target_id, source_id))
        cursor.execute("UPDATE OR IGNORE buck_profile_seasons SET deer_id = ? WHERE deer_id = ?", (target_id, source_id))
        cursor.execute("DELETE FROM buck_profiles WHERE deer_id = ?", (source_id,))
        cursor.execute("DELETE FROM buck_profile_seasons WHERE deer_id = ?", (source_id,))
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
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE photos SET suggested_tag = ?, suggested_confidence = ? WHERE id = ?
        """, (tag, confidence, photo_id))
        self.conn.commit()

    def set_suggested_sex(self, photo_id: int, sex: str, confidence: Optional[float]):
        """Store AI-suggested sex (buck/doe) and confidence."""
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
        """Replace boxes for a photo. Boxes use relative coords 0-1 and optional per-box data."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM annotation_boxes WHERE photo_id = ?", (photo_id,))
        to_insert = []
        for b in boxes:
            conf = b.get("confidence")
            species = b.get("species", "")
            species_conf = b.get("species_conf")
            to_insert.append((
                photo_id,
                b.get("label", ""),
                float(b["x1"]),
                float(b["y1"]),
                float(b["x2"]),
                float(b["y2"]),
                float(conf) if conf is not None else None,
                species if species else None,
                float(species_conf) if species_conf is not None else None
            ))
        if to_insert:
            cursor.executemany(
                "INSERT INTO annotation_boxes (photo_id, label, x1, y1, x2, y2, confidence, species, species_conf) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                to_insert,
            )
        self.conn.commit()

    def get_boxes(self, photo_id: int) -> List[Dict[str, float]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, label, x1, y1, x2, y2, confidence, species, species_conf,
                   head_x1, head_y1, head_x2, head_y2, head_notes
            FROM annotation_boxes WHERE photo_id = ?
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
            out.append(box)
        return out

    def set_box_species(self, box_id: int, species: str, confidence: float = None):
        """Set species classification for a specific box."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE annotation_boxes SET species = ?, species_conf = ? WHERE id = ?",
            (species, confidence, box_id)
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
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE annotation_boxes SET head_x1 = ?, head_y1 = ?, head_x2 = ?, head_y2 = ?, head_notes = ? WHERE id = ?",
            (x1, y1, x2, y2, notes, box_id)
        )
        self.conn.commit()

    def clear_box_head_line(self, box_id: int):
        """Clear head direction line for a specific box."""
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
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ab.id, ab.photo_id, p.file_path, ab.x1, ab.y1, ab.x2, ab.y2,
                   ab.head_x1, ab.head_y1, ab.head_x2, ab.head_y2, ab.head_notes
            FROM annotation_boxes ab
            JOIN photos p ON ab.photo_id = p.id
            JOIN tags t ON p.id = t.photo_id
            WHERE t.tag_name = 'Deer'
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
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM annotation_boxes WHERE photo_id = ?", (photo_id,))
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
    
    def get_all_photos(self) -> List[Dict]:
        """Get all photos."""
        return self.search_photos()

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
        """Archive multiple photos using batch update."""
        if not photo_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ",".join(["?"] * len(photo_ids))
        cursor.execute(f"UPDATE photos SET archived = 1 WHERE id IN ({placeholders})", photo_ids)
        self.conn.commit()

    def get_archived_count(self) -> int:
        """Get count of archived photos."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos WHERE archived = 1")
        return cursor.fetchone()[0]

    # ========== Site Management ==========

    def create_site(self, name: str, description: str = "", representative_photo_id: int = None,
                    confirmed: bool = True) -> int:
        """Create a new site and return its ID. confirmed=False means it's a suggestion."""
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
        cursor = self.conn.cursor()
        cursor.execute("UPDATE photos SET site_id = NULL WHERE site_id = ?", (site_id,))
        cursor.execute("DELETE FROM sites WHERE id = ?", (site_id,))
        self.conn.commit()

    def set_photo_site(self, photo_id: int, site_id: Optional[int]):
        """Assign a photo to a site."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE photos SET site_id = ? WHERE id = ?", (site_id, photo_id))
        self.conn.commit()

    def set_photos_site(self, photo_ids: List[int], site_id: Optional[int]):
        """Assign multiple photos to a site."""
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
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE photos SET site_id = ?, suggested_site_id = NULL, suggested_site_confidence = NULL
            WHERE id = ?
        """, (site_id, photo_id))
        self.conn.commit()

    def set_photo_site_suggestion(self, photo_id: int, site_id: int, confidence: float):
        """Set a suggested site for a photo (not confirmed yet)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE photos SET suggested_site_id = ?, suggested_site_confidence = ?
            WHERE id = ?
        """, (site_id, confidence, photo_id))
        self.conn.commit()

    def confirm_site(self, site_id: int):
        """Mark a site as confirmed and move all suggested photos to confirmed."""
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

    def get_embedding(self, photo_id: int) -> Optional[bytes]:
        """Get embedding bytes for a photo."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding FROM photo_embeddings WHERE photo_id = ?", (photo_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_all_embeddings(self) -> List[Tuple[int, bytes]]:
        """Get all photo embeddings as (photo_id, embedding_bytes) tuples."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT photo_id, embedding FROM photo_embeddings")
        return [(row[0], row[1]) for row in cursor.fetchall()]

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
        cursor.execute("SELECT id, file_path FROM photos WHERE file_hash IS NULL")
        rows = cursor.fetchall()

        count = 0
        total = len(rows)
        for i, row in enumerate(rows):
            if progress_callback:
                progress_callback(i, total)

            file_path = row[1]
            if file_path and os.path.exists(file_path):
                file_hash = self._calculate_file_hash(file_path)
                if file_hash:
                    cursor.execute("UPDATE photos SET file_hash = ? WHERE id = ?",
                                   (file_hash, row[0]))
                    count += 1

        self.conn.commit()
        return count

    def get_hash_stats(self) -> Dict[str, int]:
        """Get stats about file hashes in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM photos WHERE file_hash IS NOT NULL")
        with_hash = cursor.fetchone()[0]
        return {"total": total, "with_hash": with_hash, "missing": total - with_hash}

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
        """Find a local photo by its sync key, with fallback to file_hash."""
        cursor = self.conn.cursor()

        # First try matching by the full photo_key
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

        # Fallback: try matching by file_hash (for CuddeLink photos with different filenames)
        if file_hash:
            cursor.execute("SELECT * FROM photos WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            if row:
                return dict(row)

        return None

    def push_to_supabase(self, supabase_client, progress_callback=None, force_full=False) -> Dict[str, int]:
        """Push changed local data to Supabase using batch operations. Returns counts of items pushed.

        Args:
            supabase_client: Supabase client instance
            progress_callback: Optional callable(step, total, message) for progress updates
            force_full: If True, push all data regardless of last sync time
        """
        def report(step, msg):
            if progress_callback:
                progress_callback(step, 7, msg)
            print(msg)

        # Ensure all required columns exist before syncing (handles old databases)
        self._ensure_deer_columns()
        self._ensure_sync_tracking()

        # Get last sync time for incremental push
        last_sync = None if force_full else self.get_last_sync_time('push')
        sync_mode = "full" if last_sync is None else "incremental"
        report(0, f"Starting {sync_mode} sync...")

        # Calculate file hashes for photos that don't have one (for cross-computer matching)
        hash_stats = self.get_hash_stats()
        if hash_stats["missing"] > 0:
            print(f"Calculating file hashes for {hash_stats['missing']} photos...")
            self.calculate_missing_hashes()

        from datetime import datetime
        counts = {"photos": 0, "tags": 0, "deer_metadata": 0, "deer_additional": 0,
                  "buck_profiles": 0, "buck_profile_seasons": 0, "annotation_boxes": 0}

        cursor = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        BATCH_SIZE = 500  # Supabase handles batches well

        # Build WHERE clause for incremental sync
        def since_clause(table_alias=""):
            if last_sync is None:
                return ""
            prefix = f"{table_alias}." if table_alias else ""
            return f" AND {prefix}updated_at > '{last_sync}'"

        def batch_upsert(table_name, data_list, on_conflict):
            """Upsert data in batches."""
            for i in range(0, len(data_list), BATCH_SIZE):
                batch = data_list[i:i + BATCH_SIZE]
                if batch:
                    supabase_client.table(table_name).upsert(batch, on_conflict=on_conflict).execute()

        # Push photos_sync (photo identifiers and basic metadata)
        report(1, "Syncing photos...")
        query = "SELECT * FROM photos WHERE 1=1" + since_clause()
        cursor.execute(query)
        photos_data = []
        for row in cursor.fetchall():
            photo = dict(row)
            photo_key = self._make_photo_key(photo)
            photos_data.append({
                "photo_key": photo_key,
                "file_hash": photo.get("file_hash"),  # For cross-computer matching
                "original_name": photo.get("original_name"),
                "date_taken": photo.get("date_taken"),
                "camera_model": photo.get("camera_model"),
                "camera_location": photo.get("camera_location"),
                "season_year": photo.get("season_year"),
                "favorite": bool(photo.get("favorite")),
                "notes": photo.get("notes"),
                "updated_at": now
            })
        batch_upsert("photos_sync", photos_data, "photo_key")
        counts["photos"] = len(photos_data)

        # Push tags
        report(2, "Syncing tags...")
        cursor.execute(f"""
            SELECT t.tag_name, t.updated_at, p.original_name, p.date_taken, p.camera_model, p.file_hash
            FROM tags t
            JOIN photos p ON t.photo_id = p.id
            WHERE 1=1 {since_clause('t')}
        """)
        tags_data = []
        for row in cursor.fetchall():
            photo_key = f"{row['original_name']}|{row['date_taken']}|{row['camera_model']}"
            tags_data.append({
                "photo_key": photo_key,
                "file_hash": row["file_hash"],
                "tag_name": row["tag_name"],
                "updated_at": now
            })
        batch_upsert("tags", tags_data, "photo_key,tag_name")
        counts["tags"] = len(tags_data)

        # Push deer_metadata
        report(3, "Syncing deer metadata...")
        cursor.execute(f"""
            SELECT d.*, p.original_name, p.date_taken, p.camera_model, p.file_hash
            FROM deer_metadata d
            JOIN photos p ON d.photo_id = p.id
            WHERE 1=1 {since_clause('d')}
        """)
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
                "updated_at": now
            })
        batch_upsert("deer_metadata", deer_meta_data, "photo_key")
        counts["deer_metadata"] = len(deer_meta_data)

        # Push deer_additional
        report(4, "Syncing additional deer...")
        cursor.execute(f"""
            SELECT d.*, p.original_name, p.date_taken, p.camera_model, p.file_hash
            FROM deer_additional d
            JOIN photos p ON d.photo_id = p.id
            WHERE 1=1 {since_clause('d')}
        """)
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
                "updated_at": now
            })
        batch_upsert("deer_additional", deer_add_data, "photo_key,deer_id")
        counts["deer_additional"] = len(deer_add_data)

        # Push buck_profiles
        report(5, "Syncing buck profiles...")
        cursor.execute(f"SELECT * FROM buck_profiles WHERE 1=1 {since_clause()}")
        profiles_data = []
        for row in cursor.fetchall():
            d = dict(row)
            profiles_data.append({
                "deer_id": d.get("deer_id"),
                "display_name": d.get("display_name"),
                "notes": d.get("notes"),
                "updated_at": now
            })
        batch_upsert("buck_profiles", profiles_data, "deer_id")
        counts["buck_profiles"] = len(profiles_data)

        # Push buck_profile_seasons
        report(6, "Syncing buck profile seasons...")
        cursor.execute(f"SELECT * FROM buck_profile_seasons WHERE 1=1 {since_clause()}")
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
                "updated_at": now
            })
        batch_upsert("buck_profile_seasons", seasons_data, "deer_id,season_year")
        counts["buck_profile_seasons"] = len(seasons_data)

        # Push annotation_boxes - but ONLY if we have a substantial local database
        # This prevents accidentally wiping cloud boxes when pushing from an empty/new computer
        report(7, "Syncing annotation boxes...")
        cursor.execute("SELECT COUNT(*) FROM annotation_boxes")
        local_box_count = cursor.fetchone()[0]

        # Safety check: don't delete cloud boxes if local has very few
        # (protects against pushing from empty/new database)
        if local_box_count < 100:
            print(f"  Skipping box sync - only {local_box_count} local boxes (need 100+ to sync)")
            print("  This protects cloud data from being overwritten by an empty database.")
            print("  Run 'Pull from Cloud' first if this is a new computer.")
            counts["annotation_boxes"] = 0
        else:
            cursor.execute(f"""
                SELECT b.*, p.original_name, p.date_taken, p.camera_model, p.file_hash
                FROM annotation_boxes b
                JOIN photos p ON b.photo_id = p.id
                WHERE 1=1 {since_clause('b')}
            """)
            boxes_data = []
            for row in cursor.fetchall():
                d = dict(row)
                photo_key = f"{d['original_name']}|{d['date_taken']}|{d['camera_model']}"
                boxes_data.append({
                    "photo_key": photo_key,
                    "file_hash": d.get("file_hash"),
                    "label": d.get("label"),
                    "x1": d.get("x1"),
                    "y1": d.get("y1"),
                    "x2": d.get("x2"),
                    "y2": d.get("y2"),
                    "confidence": d.get("confidence"),
                    "updated_at": now
                })
            # For incremental sync, use upsert instead of delete+insert
            if boxes_data:
                if last_sync is None:
                    # Full sync: delete all and re-insert
                    supabase_client.table("annotation_boxes").delete().neq("photo_key", "").execute()
                    for i in range(0, len(boxes_data), BATCH_SIZE):
                        batch = boxes_data[i:i + BATCH_SIZE]
                        if batch:
                            supabase_client.table("annotation_boxes").insert(batch).execute()
                else:
                    # Incremental sync: upsert changed boxes
                    batch_upsert("annotation_boxes", boxes_data, "photo_key,label,x1,y1")
            counts["annotation_boxes"] = len(boxes_data)

        report(7, "Sync complete!")

        # Update last sync time after successful push
        self.set_last_sync_time('push', now)

        return counts

    def pull_from_supabase(self, supabase_client, progress_callback=None) -> Dict[str, int]:
        """Pull data from Supabase and update local database. Returns counts.

        Args:
            supabase_client: Supabase client instance
            progress_callback: Optional callable(step, total, message) for progress updates
        """
        def report(step, msg):
            if progress_callback:
                progress_callback(step, 7, msg)
            print(msg)

        # Ensure all required columns exist before syncing (handles old databases)
        self._ensure_deer_columns()

        counts = {"photos": 0, "tags": 0, "deer_metadata": 0, "deer_additional": 0,
                  "buck_profiles": 0, "buck_profile_seasons": 0, "annotation_boxes": 0}

        cursor = self.conn.cursor()

        # Pull photos_sync - update local photo metadata (fetch_all to bypass 1000 row limit)
        report(1, "Pulling photos...")
        result = supabase_client.table("photos_sync").select("*").execute(fetch_all=True)
        for row in result.data:
            photo = self._get_photo_by_key(row["photo_key"], row.get("file_hash"))
            if photo:
                cursor.execute("""
                    UPDATE photos SET
                        camera_location = COALESCE(?, camera_location),
                        favorite = COALESCE(?, favorite),
                        notes = COALESCE(?, notes)
                    WHERE id = ?
                """, (row.get("camera_location"), row.get("favorite"), row.get("notes"), photo["id"]))
                counts["photos"] += 1

        # Pull tags (fetch_all to bypass 1000 row limit)
        report(2, "Pulling tags...")
        result = supabase_client.table("tags").select("*").execute(fetch_all=True)
        for row in result.data:
            photo = self._get_photo_by_key(row["photo_key"], row.get("file_hash"))
            if photo:
                cursor.execute("""
                    INSERT OR IGNORE INTO tags (photo_id, tag_name) VALUES (?, ?)
                """, (photo["id"], row["tag_name"]))
                counts["tags"] += 1

        # Pull deer_metadata (fetch_all to bypass 1000 row limit)
        report(3, "Pulling deer metadata...")
        result = supabase_client.table("deer_metadata").select("*").execute(fetch_all=True)
        for row in result.data:
            photo = self._get_photo_by_key(row["photo_key"], row.get("file_hash"))
            if photo:
                cursor.execute("""
                    INSERT OR REPLACE INTO deer_metadata
                    (photo_id, deer_id, age_class, left_points_min, left_points_max,
                     right_points_min, right_points_max, left_points_uncertain, right_points_uncertain,
                     left_ab_points_min, left_ab_points_max, right_ab_points_min, right_ab_points_max,
                     abnormal_points_min, abnormal_points_max, broken_antler_side, broken_antler_note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (photo["id"], row.get("deer_id"), row.get("age_class"),
                      row.get("left_points_min"), row.get("left_points_max"),
                      row.get("right_points_min"), row.get("right_points_max"),
                      row.get("left_points_uncertain"), row.get("right_points_uncertain"),
                      row.get("left_ab_points_min"), row.get("left_ab_points_max"),
                      row.get("right_ab_points_min"), row.get("right_ab_points_max"),
                      row.get("abnormal_points_min"), row.get("abnormal_points_max"),
                      row.get("broken_antler_side"), row.get("broken_antler_note")))
                counts["deer_metadata"] += 1

        # Pull deer_additional (fetch_all to bypass 1000 row limit)
        report(4, "Pulling additional deer...")
        result = supabase_client.table("deer_additional").select("*").execute(fetch_all=True)
        for row in result.data:
            photo = self._get_photo_by_key(row["photo_key"], row.get("file_hash"))
            if photo and row.get("deer_id"):
                cursor.execute("""
                    INSERT OR REPLACE INTO deer_additional
                    (photo_id, deer_id, age_class, left_points_min, left_points_max,
                     right_points_min, right_points_max, left_points_uncertain, right_points_uncertain,
                     left_ab_points_min, left_ab_points_max, right_ab_points_min, right_ab_points_max,
                     abnormal_points_min, abnormal_points_max, broken_antler_side, broken_antler_note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (photo["id"], row.get("deer_id"), row.get("age_class"),
                      row.get("left_points_min"), row.get("left_points_max"),
                      row.get("right_points_min"), row.get("right_points_max"),
                      row.get("left_points_uncertain"), row.get("right_points_uncertain"),
                      row.get("left_ab_points_min"), row.get("left_ab_points_max"),
                      row.get("right_ab_points_min"), row.get("right_ab_points_max"),
                      row.get("abnormal_points_min"), row.get("abnormal_points_max"),
                      row.get("broken_antler_side"), row.get("broken_antler_note")))
                counts["deer_additional"] += 1

        # Pull buck_profiles (fetch_all to bypass 1000 row limit)
        report(5, "Pulling buck profiles...")
        result = supabase_client.table("buck_profiles").select("*").execute(fetch_all=True)
        for row in result.data:
            if row.get("deer_id"):
                cursor.execute("""
                    INSERT OR REPLACE INTO buck_profiles (deer_id, display_name, notes)
                    VALUES (?, ?, ?)
                """, (row["deer_id"], row.get("display_name"), row.get("notes")))
                counts["buck_profiles"] += 1

        # Pull buck_profile_seasons (fetch_all to bypass 1000 row limit)
        report(6, "Pulling buck profile seasons...")
        result = supabase_client.table("buck_profile_seasons").select("*").execute(fetch_all=True)
        for row in result.data:
            if row.get("deer_id") and row.get("season_year"):
                cursor.execute("""
                    INSERT OR REPLACE INTO buck_profile_seasons
                    (deer_id, season_year, camera_locations, key_characteristics,
                     left_points_min, left_points_max, right_points_min, right_points_max,
                     left_ab_points_min, left_ab_points_max, right_ab_points_min, right_ab_points_max,
                     abnormal_points_min, abnormal_points_max)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (row["deer_id"], row["season_year"], row.get("camera_locations"),
                      row.get("key_characteristics"), row.get("left_points_min"), row.get("left_points_max"),
                      row.get("right_points_min"), row.get("right_points_max"),
                      row.get("left_ab_points_min"), row.get("left_ab_points_max"),
                      row.get("right_ab_points_min"), row.get("right_ab_points_max"),
                      row.get("abnormal_points_min"), row.get("abnormal_points_max")))
                counts["buck_profile_seasons"] += 1

        report(7, "Sync complete!")
        self.conn.commit()
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
