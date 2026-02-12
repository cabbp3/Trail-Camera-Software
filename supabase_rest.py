"""
Simple Supabase REST API client using only the requests library.
This eliminates the need for the supabase package which requires C++ build tools on Windows.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)


def _get_bundled_config_path() -> Path:
    """Get path to bundled config, works for dev and PyInstaller bundle."""
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        return Path(sys._MEIPASS) / "cloud_config.json"
    # Running in development
    return Path(__file__).parent / "cloud_config.json"


# Config file locations (checked in order)
USER_CONFIG_PATH = Path.home() / ".trailcam" / "supabase_config.json"
BUNDLED_CONFIG_PATH = _get_bundled_config_path()
SESSION_PATH = Path.home() / ".trailcam" / "session.json"

# Cached client instance
_cached_client: Optional['SupabaseRestClient'] = None


class SupabaseResponse:
    """Mimics the supabase response object."""
    def __init__(self, data: List[Dict], error: Optional[str] = None):
        self.data = data
        self.error = error


def _request_with_retry(method, url, max_attempts=3, **kwargs):
    """Execute an HTTP request with exponential backoff on timeouts and 5xx errors."""
    delay = 0.5
    for attempt in range(max_attempts):
        try:
            response = method(url, **kwargs)
            if response.status_code >= 500 and attempt < max_attempts - 1:
                logger.warning(f"Supabase {response.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1})")
                time.sleep(delay)
                delay *= 2
                continue
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_attempts - 1:
                logger.warning(f"Supabase request failed ({e}), retrying in {delay:.1f}s (attempt {attempt + 1})")
                time.sleep(delay)
                delay *= 2
            else:
                raise


class TableQuery:
    """Builds and executes queries against a Supabase table."""

    def __init__(self, client: 'SupabaseRestClient', table_name: str):
        self.client = client
        self.table_name = table_name
        self._select_cols = None
        self._filters = []
        self._upsert_data = None
        self._upsert_conflict = None
        self._insert_data = None
        self._update_data = None
        self._is_delete = False

    def select(self, columns: str = "*") -> 'TableQuery':
        """Set columns to select."""
        self._select_cols = columns
        return self

    def upsert(self, data: List[Dict], on_conflict: str = None) -> 'TableQuery':
        """Set data to upsert."""
        self._upsert_data = data
        self._upsert_conflict = on_conflict
        return self

    def insert(self, data: List[Dict]) -> 'TableQuery':
        """Set data to insert."""
        self._insert_data = data
        return self

    def update(self, data: Dict) -> 'TableQuery':
        """Set data to update. Must be used with filters (e.g., .eq())."""
        self._update_data = data
        return self

    def delete(self) -> 'TableQuery':
        """Mark as delete operation."""
        self._is_delete = True
        return self

    def neq(self, column: str, value: Any) -> 'TableQuery':
        """Add not-equal filter."""
        self._filters.append(f"{column}=neq.{value}")
        return self

    def eq(self, column: str, value: Any) -> 'TableQuery':
        """Add equal filter."""
        self._filters.append(f"{column}=eq.{value}")
        return self

    def gt(self, column: str, value: Any) -> 'TableQuery':
        """Add greater-than filter (for incremental sync)."""
        self._filters.append(f"{column}=gt.{value}")
        return self

    def gte(self, column: str, value: Any) -> 'TableQuery':
        """Add greater-than-or-equal filter."""
        self._filters.append(f"{column}=gte.{value}")
        return self

    def lt(self, column: str, value: Any) -> 'TableQuery':
        """Add less-than filter."""
        self._filters.append(f"{column}=lt.{value}")
        return self

    def lte(self, column: str, value: Any) -> 'TableQuery':
        """Add less-than-or-equal filter."""
        self._filters.append(f"{column}=lte.{value}")
        return self

    def execute(self, fetch_all: bool = False) -> SupabaseResponse:
        """Execute the query.

        Args:
            fetch_all: If True, paginate through all results (bypasses 1000 row limit)
        """
        url = f"{self.client.url}/rest/v1/{self.table_name}"
        headers = self.client._get_headers()

        try:
            if self._select_cols is not None:
                # SELECT query
                params = {"select": self._select_cols}
                for f in self._filters:
                    key, val = f.split("=", 1)
                    params[key] = val

                if fetch_all:
                    # Paginate through all results
                    all_data = []
                    page_size = 1000
                    offset = 0
                    while True:
                        headers_with_range = headers.copy()
                        headers_with_range["Range"] = f"{offset}-{offset + page_size - 1}"
                        headers_with_range["Prefer"] = "count=exact"
                        response = _request_with_retry(requests.get, url, headers=headers_with_range, params=params, timeout=60)
                        response.raise_for_status()
                        page_data = response.json()
                        all_data.extend(page_data)
                        if len(page_data) < page_size:
                            break  # Last page
                        offset += page_size
                    return SupabaseResponse(all_data)
                else:
                    response = _request_with_retry(requests.get, url, headers=headers, params=params, timeout=30)
                    response.raise_for_status()
                    return SupabaseResponse(response.json())

            elif self._upsert_data is not None:
                # UPSERT query
                # on_conflict must be a URL parameter, not in Prefer header
                headers["Prefer"] = "resolution=merge-duplicates"
                if self._upsert_conflict:
                    url += f"?on_conflict={self._upsert_conflict}"
                response = _request_with_retry(requests.post, url, headers=headers, json=self._upsert_data, timeout=60)
                response.raise_for_status()
                return SupabaseResponse([])

            elif self._insert_data is not None:
                # INSERT query
                response = _request_with_retry(requests.post, url, headers=headers, json=self._insert_data, timeout=60)
                response.raise_for_status()
                return SupabaseResponse([])

            elif self._update_data is not None:
                # UPDATE (PATCH) query - requires filters
                if self._filters:
                    for f in self._filters:
                        url += ("?" if "?" not in url else "&") + f
                response = _request_with_retry(requests.patch, url, headers=headers, json=self._update_data, timeout=60)
                response.raise_for_status()
                return SupabaseResponse([])

            elif self._is_delete:
                # DELETE query
                if self._filters:
                    # Add filters as query params
                    for f in self._filters:
                        url += ("?" if "?" not in url else "&") + f
                response = _request_with_retry(requests.delete, url, headers=headers, timeout=30)
                response.raise_for_status()
                return SupabaseResponse([])

            else:
                raise ValueError("No operation specified")

        except requests.exceptions.RequestException as e:
            return SupabaseResponse([], error=str(e))


class SupabaseRestClient:
    """Simple Supabase REST API client."""

    def __init__(self, url: str, key: str):
        """
        Initialize the client.

        Args:
            url: Supabase project URL (e.g., https://xxx.supabase.co)
            key: Supabase anon/service key
        """
        self.url = url.rstrip("/")
        self.key = key
        self.session_path = SESSION_PATH
        self._session = {}
        self._auth_error = None
        self._load_session()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        try:
            self.refresh_if_needed()
        except Exception:
            pass
        token = self._session.get("access_token")
        auth_token = token if token else self.key
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        """Headers for auth endpoints."""
        return {
            "apikey": self.key,
            "Content-Type": "application/json",
        }

    def _load_session(self) -> None:
        """Load session from disk."""
        if not self.session_path.exists():
            return
        try:
            self._session = json.loads(self.session_path.read_text())
        except Exception:
            self._session = {}

    def _save_session(self) -> None:
        """Persist session to disk."""
        try:
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.session_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(self._session, indent=2))
            os.replace(tmp_path, self.session_path)
            os.chmod(self.session_path, 0o600)
        except Exception:
            pass

    def _clear_session(self) -> None:
        """Clear session from memory and disk."""
        self._session = {}
        try:
            if self.session_path.exists():
                self.session_path.unlink()
        except Exception:
            pass

    def _store_session(self, data: Dict, email: Optional[str] = None, display_name: Optional[str] = None) -> None:
        """Store session tokens and metadata."""
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        expires_at = data.get("expires_at")
        expires_in = data.get("expires_in")
        if not expires_at and expires_in:
            expires_at = int(time.time()) + int(expires_in)

        user = data.get("user") or {}
        user_email = user.get("email") or email
        user_meta = user.get("user_metadata") or {}
        display = user_meta.get("display_name") or display_name

        if access_token:
            self._session = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at,
                "user_email": user_email,
                "display_name": display or user_email,
            }
            self._save_session()

    @property
    def is_authenticated(self) -> bool:
        """Return True if we have a non-expired access token."""
        token = self._session.get("access_token")
        expires_at = self._session.get("expires_at")
        if not token or not expires_at:
            return False
        try:
            return int(expires_at) > int(time.time())
        except Exception:
            return False

    def get_session_info(self) -> Dict[str, Any]:
        """Return current session info."""
        return dict(self._session) if self._session else {}

    def refresh_if_needed(self) -> bool:
        """Refresh access token if expired."""
        refresh_token = self._session.get("refresh_token")
        expires_at = self._session.get("expires_at")
        if not refresh_token or not expires_at:
            return False
        try:
            if int(expires_at) - 30 > int(time.time()):
                return True
        except Exception:
            return False
        ok = self.refresh_token()
        if not ok and self._session:
            self._auth_error = "Session expired or refresh failed"
            logger.warning("Supabase auth refresh failed; falling back to anon key")
        return ok

    def sign_up(self, email: str, password: str, display_name: str) -> Dict[str, Any]:
        """Create a new user account."""
        url = f"{self.url}/auth/v1/signup"
        payload = {
            "email": email,
            "password": password,
            "data": {"display_name": display_name},
        }
        try:
            resp = requests.post(url, headers=self._get_auth_headers(), json=payload, timeout=30)
            if resp.status_code >= 200 and resp.status_code < 300:
                data = resp.json()
                self._store_session(data, email=email, display_name=display_name)
                return {"ok": True, "data": data}
            return {"ok": False, "error": resp.text}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Sign in with email/password."""
        url = f"{self.url}/auth/v1/token?grant_type=password"
        payload = {"email": email, "password": password}
        try:
            resp = requests.post(url, headers=self._get_auth_headers(), json=payload, timeout=30)
            if resp.status_code >= 200 and resp.status_code < 300:
                data = resp.json()
                self._store_session(data, email=email)
                self._auth_error = None
                return {"ok": True, "data": data}
            return {"ok": False, "error": resp.text}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def refresh_token(self) -> bool:
        """Refresh the access token."""
        refresh_token = self._session.get("refresh_token")
        if not refresh_token:
            return False
        url = f"{self.url}/auth/v1/token?grant_type=refresh_token"
        payload = {"refresh_token": refresh_token}
        try:
            resp = requests.post(url, headers=self._get_auth_headers(), json=payload, timeout=30)
            if resp.status_code >= 200 and resp.status_code < 300:
                data = resp.json()
                self._store_session(data, email=self._session.get("user_email"), display_name=self._session.get("display_name"))
                self._auth_error = None
                return True
            self._clear_session()
            return False
        except Exception:
            self._clear_session()
            return False

    def sign_out(self) -> bool:
        """Sign out and clear session."""
        token = self._session.get("access_token")
        if token:
            url = f"{self.url}/auth/v1/logout"
            try:
                headers = self._get_auth_headers()
                headers["Authorization"] = f"Bearer {token}"
                requests.post(url, headers=headers, timeout=15)
            except Exception:
                pass
        self._clear_session()
        return True

    def table(self, name: str) -> TableQuery:
        """Get a table query builder."""
        return TableQuery(self, name)

    def is_configured(self) -> bool:
        """Check if the client has valid URL and key configured."""
        return bool(self.url and self.key)

    def test_connection(self) -> bool:
        """Test if the connection works."""
        try:
            # Try to access the REST API
            url = f"{self.url}/rest/v1/"
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            return response.status_code in (200, 404)  # 404 is ok, means API is reachable
        except Exception:
            return False


def create_client(url: str, key: str) -> SupabaseRestClient:
    """Create a Supabase REST client (drop-in replacement for supabase.create_client)."""
    return SupabaseRestClient(url, key)


def _load_config() -> Optional[Dict]:
    """Load Supabase credentials from config file.

    Checks in order:
    1. User config (~/.trailcam/supabase_config.json)
    2. Bundled config (cloud_config.json in app folder)
    """
    config = None

    # Try user config first
    if USER_CONFIG_PATH.exists():
        try:
            with open(USER_CONFIG_PATH) as f:
                config = json.load(f)
            logger.info("Using user Supabase config")
            return config
        except Exception as e:
            logger.warning(f"Failed to load user Supabase config: {e}")

    # Fall back to bundled config
    if BUNDLED_CONFIG_PATH.exists():
        try:
            with open(BUNDLED_CONFIG_PATH) as f:
                data = json.load(f)
                config = data.get("supabase", data)  # Handle nested or flat format
            logger.info("Using bundled Supabase config")
            return config
        except Exception as e:
            logger.warning(f"Failed to load bundled Supabase config: {e}")

    logger.warning("No Supabase config found")
    return None


def get_client() -> Optional[SupabaseRestClient]:
    """Get a configured Supabase client.

    Loads credentials from config files and returns a cached client instance.
    Returns None if no valid config is found.
    """
    global _cached_client

    if _cached_client is not None:
        try:
            _cached_client.refresh_if_needed()
        except Exception:
            pass
        return _cached_client

    config = _load_config()
    if not config:
        return None

    url = config.get("url")
    key = config.get("key") or config.get("anon_key")  # Support both key names

    if not url or not key:
        logger.error("Supabase config missing 'url' or 'key'/'anon_key'")
        return None

    _cached_client = SupabaseRestClient(url, key)
    logger.info(f"Supabase client initialized for: {url}")
    return _cached_client


def is_configured() -> bool:
    """Check if Supabase is configured and accessible."""
    client = get_client()
    return client is not None and client.is_configured()


def get_cloud_photo_hashes() -> List[str]:
    """Get all file_hashes from Supabase photos_sync table.

    Returns:
        List of file_hash strings from cloud
    """
    client = get_client()
    if not client:
        return []

    try:
        result = client.table("photos_sync").select("file_hash").execute(fetch_all=True)
        hashes = [row["file_hash"] for row in result.data if row.get("file_hash")]
        logger.info(f"Found {len(hashes)} photos in cloud")
        return hashes
    except Exception as e:
        logger.error(f"Failed to get cloud photo hashes: {e}")
        return []


def get_cloud_photos_not_local(local_hashes: set) -> List[Dict]:
    """Get full photo records from Supabase that don't exist locally.

    Args:
        local_hashes: Set of file_hashes that exist in local database

    Returns:
        List of photo records from cloud that are not in local_hashes
    """
    client = get_client()
    if not client:
        return []

    try:
        # Get all photos from cloud
        result = client.table("photos_sync").select("*").execute(fetch_all=True)

        # Filter to only those not in local
        cloud_only = []
        for row in result.data:
            file_hash = row.get("file_hash")
            if file_hash and file_hash not in local_hashes:
                cloud_only.append(row)

        logger.info(f"Found {len(cloud_only)} cloud-only photos (not in local DB)")
        return cloud_only
    except Exception as e:
        logger.error(f"Failed to get cloud photos: {e}")
        return []
