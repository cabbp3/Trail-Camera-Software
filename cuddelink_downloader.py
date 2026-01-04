"""
Minimal CuddeLink downloader (web flow).

This authenticates with env vars (CUDDE_USER/CUDDE_PASS), scrapes photo IDs from
the /photos page, requests a download bundle, then pulls the zip via the GUID.

It returns a list of extracted image file paths (caller handles importing).
"""
import json
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional

import requests


# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubles each retry
RETRYABLE_STATUS_CODES = {502, 503, 504, 520, 521, 522, 523, 524}


def _retry_request(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """
    Retry a request function with exponential backoff for transient errors.

    Handles: 502, 503, 504, Cloudflare errors (520-524), connection errors.
    """
    last_error = None
    delay = RETRY_DELAY

    for attempt in range(max_retries + 1):
        try:
            response = func(*args, **kwargs)
            # Check for retryable status codes
            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt < max_retries:
                    print(f"[CuddeLink] Server error {response.status_code}, retrying in {delay}s... (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries:
                print(f"[CuddeLink] Connection error, retrying in {delay}s... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries:
                print(f"[CuddeLink] Timeout, retrying in {delay}s... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(delay)
                delay *= 2
            else:
                raise

    raise last_error if last_error else RuntimeError("Request failed after retries")
from urllib.parse import urlparse

BASE_URL = "https://camp.cuddeback.com"
LOGIN_URL = f"{BASE_URL}/Identity/Account/Login"
PHOTOS_URL = f"{BASE_URL}/photos"
DOWNLOAD_VIEW_URL = f"{BASE_URL}/photos/downloadphotoview"
DOWNLOAD_FILE_URL = f"{BASE_URL}/photos/Download"
APPLY_FILTER_URL = f"{BASE_URL}/photos/applyFilter"


def _extract_verification_token(html: str) -> Optional[str]:
    """Pull anti-forgery token from login page (handles various HTML formats)."""
    patterns = [
        # name before value, with possible attributes in between
        r'name="__RequestVerificationToken"[^>]*value="([^"]+)"',
        r"name='__RequestVerificationToken'[^>]*value='([^']+)'",
        # value before name
        r'value="([^"]+)"[^>]*name="__RequestVerificationToken"',
        r"value='([^']+)'[^>]*name='__RequestVerificationToken'",
        # meta tag format
        r'content="([^"]+)"[^>]*name="__RequestVerificationToken"',
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _extract_photo_ids_from_html(html: str) -> List[str]:
    """Grab photo IDs from the /photos HTML."""
    ids = set()
    # Hidden inputs with photoId{N} pattern - these contain the actual IDs
    ids.update(re.findall(r'id="photoId\d+"\s+value="([^"]+)"', html, flags=re.IGNORECASE))
    ids.update(re.findall(r'value="([^"]+)"\s+id="photoId\d+"', html, flags=re.IGNORECASE))
    # Legacy patterns
    ids.update(re.findall(r'data-photo-id=["\']([^"\']+)', html, flags=re.IGNORECASE))
    ids.update(re.findall(r'photoid=([A-Za-z0-9_-]+)', html, flags=re.IGNORECASE))
    return list(ids)


def _login(session: requests.Session, user: str, password: str) -> None:
    """Perform form-based login."""
    # Mimic a browser UA; some sites reject default python UA.
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    print(f"[CuddeLink] Getting login page: {LOGIN_URL}")
    resp = session.get(LOGIN_URL, timeout=20)
    print(f"[CuddeLink] Login page status: {resp.status_code}")
    resp.raise_for_status()
    token = _extract_verification_token(resp.text)
    print(f"[CuddeLink] Token found: {'Yes' if token else 'No'}")

    payload = {
        "Input.Email": user,
        "Input.Password": password,
        "Input.RememberMe": "false",
    }
    if token:
        payload["__RequestVerificationToken"] = token

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": BASE_URL,
        "Referer": LOGIN_URL,
    }

    print(f"[CuddeLink] Posting login...")
    post = session.post(LOGIN_URL, data=payload, headers=headers, timeout=20, allow_redirects=True)
    print(f"[CuddeLink] Post status: {post.status_code}, URL: {post.url}")

    if post.status_code >= 400:
        snippet = post.text[:500].strip().replace("\n", " ")
        raise requests.HTTPError(f"Login failed ({post.status_code}): {snippet}")

    # Check if login failed (page contains error message or still on login page)
    if "Invalid login" in post.text or ("invalid" in post.text.lower() and "login" in post.url.lower()):
        raise requests.HTTPError("Invalid email or password. Please check your credentials.")

    # Verify we can access the photos page (proves login succeeded)
    print(f"[CuddeLink] Checking photos page...")
    photos_check = session.get(PHOTOS_URL, timeout=20, allow_redirects=True)
    print(f"[CuddeLink] Photos status: {photos_check.status_code}, URL: {photos_check.url}")
    if "login" in photos_check.url.lower():
        raise requests.HTTPError("Login appeared to succeed but session not authenticated. Please try again.")


def _apply_date_filter(session: requests.Session, start_date: str, end_date: str) -> None:
    """Apply date filter to the photos page session."""
    filter_value = f"date;{start_date};{end_date}"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": PHOTOS_URL,
    }
    print(f"[CuddeLink] Applying date filter: {start_date} to {end_date}")
    resp = session.post(APPLY_FILTER_URL, data={"filter": filter_value}, headers=headers, timeout=30)
    resp.raise_for_status()


def _fetch_photo_ids(session: requests.Session) -> List[str]:
    """Scrape photo IDs from the photos page."""
    resp = session.get(PHOTOS_URL, timeout=30)
    resp.raise_for_status()
    ids = _extract_photo_ids_from_html(resp.text)
    return ids


def _request_download_guid(session: requests.Session, photo_ids: List[str]) -> str:
    """POST photo IDs to request a download bundle; returns GUID."""
    payload = {"photoIds": json.dumps(photo_ids)}
    headers = {
        "Referer": PHOTOS_URL,
        "Origin": BASE_URL,
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    print(f"[CuddeLink] Requesting download for {len(photo_ids)} photos...")
    resp = _retry_request(
        session.post,
        DOWNLOAD_VIEW_URL,
        data=payload,
        headers=headers,
        timeout=120  # Increased timeout for large batches
    )
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if "application/zip" in ctype:
        # Unexpected direct zip; save to temp and return path as GUID surrogate.
        tmp_zip = Path(tempfile.mkstemp(suffix=".zip")[1])
        tmp_zip.write_bytes(resp.content)
        return str(tmp_zip)
    guid = resp.text.strip().strip('"')
    if not guid:
        raise RuntimeError("CuddeLink download GUID missing.")
    return guid


def _download_zip(session: requests.Session, guid: str, progress_callback=None) -> Path:
    """Download the zip using the provided GUID with optional progress reporting."""
    # If guid already points to a file path (direct zip), just return it.
    zip_path = Path(guid)
    if zip_path.exists():
        return zip_path
    params = {"fileGuid": guid, "filename": "CuddebackPhotos"}
    print(f"[CuddeLink] Downloading zip file (this may take a while)...")

    # Stream the download to report progress
    resp = _retry_request(
        session.get,
        DOWNLOAD_FILE_URL,
        params=params,
        timeout=600,  # 10 minutes for large downloads
        stream=True
    )
    resp.raise_for_status()

    # Get content length if available
    total_size = int(resp.headers.get('content-length', 0))

    zip_path = Path(tempfile.mkstemp(suffix=".zip")[1])
    downloaded = 0
    chunk_size = 8192

    with open(zip_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    progress_callback(downloaded, total_size, "download")

    return zip_path


def _extract_images(zip_path: Path, destination: Path) -> List[Path]:
    """Extract image files from zip to destination; return extracted file paths."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            target = destination / Path(member).name
            base = target
            counter = 1
            while target.exists():
                target = base.with_name(f"{base.stem}_{counter}{base.suffix}")
                counter += 1
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted


def check_server_status() -> dict:
    """
    Check if CuddeLink server is reachable and responding.

    Returns dict with:
        - status: 'ok', 'slow', or 'down'
        - message: Human-readable status message
        - response_time: Time in seconds (if successful)
    """
    import time as time_module

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    })

    endpoints = [
        (BASE_URL, "Main site"),
        (LOGIN_URL, "Login page"),
        (PHOTOS_URL, "Photos page"),
    ]

    results = []

    for url, name in endpoints:
        try:
            start = time_module.time()
            resp = session.get(url, timeout=10, allow_redirects=True)
            elapsed = time_module.time() - start

            if resp.status_code in RETRYABLE_STATUS_CODES:
                results.append({
                    "endpoint": name,
                    "status": "down",
                    "code": resp.status_code,
                    "time": elapsed
                })
            elif resp.status_code == 200:
                results.append({
                    "endpoint": name,
                    "status": "ok" if elapsed < 5 else "slow",
                    "code": resp.status_code,
                    "time": elapsed
                })
            else:
                results.append({
                    "endpoint": name,
                    "status": "error",
                    "code": resp.status_code,
                    "time": elapsed
                })
        except requests.exceptions.Timeout:
            results.append({"endpoint": name, "status": "timeout", "code": None, "time": None})
        except requests.exceptions.ConnectionError:
            results.append({"endpoint": name, "status": "unreachable", "code": None, "time": None})

    # Summarize
    down_count = sum(1 for r in results if r["status"] in ("down", "timeout", "unreachable"))
    slow_count = sum(1 for r in results if r["status"] == "slow")

    if down_count >= 2:
        return {
            "status": "down",
            "message": "CuddeLink servers are down or unreachable. Try again later.",
            "details": results
        }
    elif down_count == 1 or slow_count >= 2:
        return {
            "status": "slow",
            "message": "CuddeLink servers are slow or partially unavailable. Downloads may fail.",
            "details": results
        }
    else:
        avg_time = sum(r["time"] for r in results if r["time"]) / len([r for r in results if r["time"]])
        return {
            "status": "ok",
            "message": f"CuddeLink servers are responding normally ({avg_time:.1f}s avg).",
            "details": results
        }


def _download_single_day(session: requests.Session, destination: Path, day_str: str, progress_callback=None) -> List[Path]:
    """Download photos for a single day. Returns list of extracted paths."""
    temp_extract = destination / ".cuddelink_tmp"

    print(f"[CuddeLink] Downloading photos for {day_str}...")
    _apply_date_filter(session, day_str, day_str)
    photo_ids = _fetch_photo_ids(session)
    print(f"[CuddeLink]   Found {len(photo_ids)} photos for {day_str}")

    if not photo_ids:
        return []

    guid = _request_download_guid(session, photo_ids)
    zip_path = _download_zip(session, guid, progress_callback)
    extracted = _extract_images(zip_path, temp_extract)
    print(f"[CuddeLink]   Extracted {len(extracted)} images for {day_str}")

    try:
        zip_path.unlink()
    except Exception:
        pass

    return extracted


def download_new_photos(destination: Path, user: str = None, password: str = None,
                        start_date: str = None, end_date: str = None,
                        progress_callback=None) -> List[Path]:
    """
    Download photos from CuddeLink and extract them.

    Downloads day-by-day to avoid server timeouts on large date ranges.

    Args:
        destination: Where to extract photos
        user: CuddeLink email (falls back to CUDDE_USER env var)
        password: CuddeLink password (falls back to CUDDE_PASS env var)
        start_date: Start date for photo filter (YYYY-MM-DD format, default: 2025-12-11)
        end_date: End date for photo filter (YYYY-MM-DD format, default: today)
        progress_callback: Optional callback(current, total, stage, message) for progress updates
            - stage: "login", "day", "download", "done"
            - For "day" stage: current=day_num, total=num_days
            - For "download" stage: current=bytes_downloaded, total=total_bytes

    Returns:
        List of extracted image file paths (caller should import + clean up if desired).
    """
    from datetime import date, datetime, timedelta

    user = user or os.getenv("CUDDE_USER")
    pw = password or os.getenv("CUDDE_PASS")
    if not user or not pw:
        raise RuntimeError("CuddeLink credentials required. Please set up your email and password.")

    # Default date range: Dec 11, 2025 to today
    if not start_date:
        start_date = "2025-12-11"
    if not end_date:
        end_date = date.today().strftime("%Y-%m-%d")

    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    num_days = (end_dt - start_dt).days + 1

    print(f"[CuddeLink] Starting download for {num_days} day(s): {start_date} to {end_date}")
    session = requests.Session()
    # Use browser-like headers to avoid being blocked
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })

    try:
        print("[CuddeLink] Logging in...")
        if progress_callback:
            progress_callback(0, 1, "login", "Logging in...")
        _login(session, user, pw)
        print("[CuddeLink] Login successful!")

        all_extracted: List[Path] = []

        # Download day by day to avoid server overload
        current_dt = start_dt
        day_num = 0
        while current_dt <= end_dt:
            day_num += 1
            day_str = current_dt.strftime("%Y-%m-%d")
            print(f"[CuddeLink] Day {day_num}/{num_days}: {day_str}")

            if progress_callback:
                progress_callback(day_num, num_days, "day", f"Day {day_num}/{num_days}: {day_str}")

            try:
                # Create a download progress callback that includes day context
                def day_download_callback(downloaded, total, stage):
                    if progress_callback:
                        progress_callback(downloaded, total, "download", f"Downloading {day_str}...")

                day_photos = _download_single_day(session, destination, day_str, day_download_callback)
                all_extracted.extend(day_photos)
            except Exception as e:
                print(f"[CuddeLink] Warning: Failed to download {day_str}: {e}")
                # Continue with next day instead of failing completely

            current_dt += timedelta(days=1)

            # Small delay between days to be nice to the server
            if current_dt <= end_dt:
                time.sleep(1)

        print(f"[CuddeLink] Done! Total extracted: {len(all_extracted)} images")
        if progress_callback:
            progress_callback(num_days, num_days, "done", f"Done! {len(all_extracted)} photos extracted")
        return all_extracted

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        if status in RETRYABLE_STATUS_CODES:
            raise RuntimeError(
                f"CuddeLink server is temporarily unavailable (Error {status}). "
                "Please try again in a few minutes."
            ) from e
        raise
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to CuddeLink servers. "
            "Please check your internet connection and try again."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Connection to CuddeLink timed out. "
            "The server may be slow - please try again."
        )
