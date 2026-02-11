#!/usr/bin/env python3
"""
CuddeLink to R2 Sync Script (Playwright version)

Downloads photos from CuddeLink and uploads them to Cloudflare R2.
Designed to run via GitHub Actions on a schedule.

Uses Playwright headless browser because CuddeLink redesigned their site
as a Blazor WebSocket SPA (Feb 2026). No REST API available.

Environment variables required:
  CUDDELINK_EMAIL, CUDDELINK_PASSWORD
  R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
  SUPABASE_URL, SUPABASE_KEY
"""

import argparse
import asyncio
import hashlib
import io
import json
import os
import random
import re
import shutil
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
from botocore.config import Config
from PIL import Image

BASE_URL = "https://camp.cuddeback.com"


# ============================================================
# Utility functions
# ============================================================

def get_env(name: str) -> str:
    """Get required environment variable."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def safe_url(url: str) -> str:
    """Strip query params from URL for safe logging (no tokens/signatures)."""
    return url.split("?")[0]


async def rate_limit(seconds: float = 1.5):
    """Central throttle between page interactions."""
    await asyncio.sleep(seconds)


async def capture_failure(page, step_name: str):
    """Screenshot + HTML + metadata dump on failure.

    HTML is redacted during login-related failures to prevent
    credentials from leaking into CI artifacts.
    """
    ts = datetime.now().strftime("%H%M%S")
    prefix = f"/tmp/cuddelink_failure_{step_name}_{ts}"
    current_url = safe_url(page.url)
    try:
        await page.screenshot(path=f"{prefix}.png", timeout=5000)
    except Exception:
        pass
    try:
        html = await page.content()
        # Redact credentials from HTML if on login page
        if "login" in step_name.lower() or "bootstrap" in step_name.lower() or "/Account/Login" in page.url:
            html = re.sub(r'(current-value|value)="[^"]*"', r'\1="[REDACTED]"', html)
        with open(f"{prefix}.html", "w") as f:
            f.write(html)
    except Exception:
        pass
    with open(f"{prefix}.meta.txt", "w") as f:
        f.write(f"step: {step_name}\nurl: {current_url}\ntime: {ts}\n")
    print(f"[FAIL] {step_name} at {current_url} — saved to {prefix}.*")


# ============================================================
# R2 / Supabase functions (preserved from original)
# ============================================================

def check_photo_exists(supabase_url: str, supabase_key: str, file_hash: str) -> bool:
    """Check if photo hash already exists in Supabase."""
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    url = f"{supabase_url}/rest/v1/photos_sync?file_hash=eq.{file_hash}&select=file_hash&limit=1"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return len(data) > 0
    except Exception as e:
        print(f"[Supabase] Warning: Could not check for duplicate: {e}")

    return False


def create_thumbnail(image_path: Path, max_size: int = 400) -> bytes:
    """Create a thumbnail from an image. Returns JPEG bytes."""
    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=75, optimize=True)
        return buffer.getvalue()


def get_exif_datetime(image_path: Path):
    """Read DateTimeOriginal from EXIF data.

    Returns ISO datetime string like: 2026-01-20T19:06:54
    Returns None if no EXIF timestamp found.

    IMPORTANT: This is the ONLY reliable source of capture time.
    Filename timestamps from CuddeLink are UPLOAD times, not capture times.
    """
    try:
        from PIL.ExifTags import TAGS
        import subprocess

        timestamp = None

        with Image.open(image_path) as img:
            datetime_original = None
            datetime_tag = None

            try:
                exif_obj = img.getexif()
                if exif_obj:
                    datetime_tag = exif_obj.get(306)
                    try:
                        exif_ifd = exif_obj.get_ifd(0x8769)
                        if exif_ifd:
                            datetime_original = exif_ifd.get(36867)
                            if not datetime_original:
                                datetime_original = exif_ifd.get(0x9003)
                    except (AttributeError, Exception):
                        pass
            except Exception:
                pass

            if not datetime_original:
                try:
                    exif_data = img._getexif()
                    if exif_data:
                        datetime_original = exif_data.get(36867)
                        if not datetime_tag:
                            datetime_tag = exif_data.get(306)
                except (AttributeError, Exception):
                    pass

            timestamp = datetime_original or datetime_tag

        if not timestamp:
            try:
                result = subprocess.run(
                    ['exiftool', '-DateTimeOriginal', '-s3', str(image_path)],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    timestamp = result.stdout.strip()
                    print(f"    [EXIF] Got timestamp via exiftool: {timestamp}")
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                pass

        if timestamp:
            result = str(timestamp).replace(':', '-', 2).replace(' ', 'T')
            return result

        print(f"    [EXIF] No timestamp found in {image_path.name} (tried Pillow + exiftool)")

    except Exception as e:
        print(f"    [EXIF] Failed to read timestamp from {image_path.name}: {e}")

    return None


def parse_cuddelink_datetime(filename: str, fallback_date: str) -> str:
    """DEPRECATED - DO NOT USE. Filename timestamp is CuddeLink upload time, NOT capture time."""
    print(f"    [WARNING] parse_cuddelink_datetime called - this produces WRONG timestamps!")
    return None


def upload_to_r2(s3_client, bucket: str, file_path: Path, r2_key: str) -> bool:
    """Upload file to R2."""
    try:
        s3_client.upload_file(str(file_path), bucket, r2_key)
        return True
    except Exception as e:
        print(f"[R2] Upload failed: {e}")
        return False


def upload_bytes_to_r2(s3_client, bucket: str, data: bytes, r2_key: str) -> bool:
    """Upload bytes to R2."""
    try:
        s3_client.put_object(Bucket=bucket, Key=r2_key, Body=data, ContentType='image/jpeg')
        return True
    except Exception as e:
        print(f"[R2] Upload failed: {e}")
        return False


def save_to_supabase(supabase_url: str, supabase_key: str, file_hash: str, datetime_taken: str, filename: str) -> bool:
    """Save photo metadata to Supabase."""
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    date_part = datetime_taken.split("T")[0]
    photo_key = f"{date_part.replace('-', '/')}/{filename}"

    data = {
        "file_hash": file_hash,
        "photo_key": photo_key,
        "date_taken": datetime_taken,
        "original_name": filename,
        "updated_at": datetime.utcnow().isoformat(),
    }

    try:
        resp = requests.post(
            f"{supabase_url}/rest/v1/photos_sync?on_conflict=file_hash",
            headers=headers,
            json=data,
            timeout=30,
        )
        # 409 = duplicate key, which means the record already exists.
        # Treat as success since Supabase is our idempotent store.
        if resp.status_code == 409:
            return True
        if resp.status_code >= 400:
            print(f"    [Supabase] Save failed: {resp.status_code} {resp.text[:100]}")
            return False
        return True
    except Exception as e:
        print(f"[Supabase] Save failed: {e}")
        return False


# ============================================================
# Playwright functions (new — replaces old requests-based scraping)
# ============================================================

LOGIN_URL = f"{BASE_URL}/Account/Login"
MAX_BOOTSTRAP_ATTEMPTS = 3
BOOTSTRAP_TIMEOUT_MS = 15000


async def check_blazor_error(page) -> bool:
    """Check if the Blazor error UI is visible (= Blazor crashed)."""
    try:
        return await page.evaluate("""() => {
            const el = document.getElementById('blazor-error-ui');
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && style.visibility !== 'hidden';
        }""")
    except Exception:
        return False


async def wait_for_blazor_bootstrap(page, debug: bool = False) -> bool:
    """Ensure Blazor has bootstrapped on the login page.

    Navigates directly to /Account/Login, waits for the SignalR WebSocket
    to connect, then waits for the login form to render.
    Returns True if the login form is ready, False if bootstrap failed.

    IMPORTANT: Do NOT use page.wait_for_url() or page.wait_for_function()
    here — they interfere with Blazor's SignalR/WebSocket initialization
    and cause form submission to silently fail.
    """
    # Set up WebSocket listener BEFORE navigation so we catch the
    # /_blazor handshake as it happens
    blazor_ws_connected = asyncio.Event()

    def on_websocket(ws):
        if "/_blazor" in ws.url:
            if debug:
                print(f"  [Debug] Blazor WebSocket connected: {ws.url[:60]}")
            blazor_ws_connected.set()

    page.on("websocket", on_websocket)

    print("[CuddeLink] Navigating to login page...")
    await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)

    # Wait for Blazor SignalR WebSocket to connect
    try:
        await asyncio.wait_for(blazor_ws_connected.wait(), timeout=BOOTSTRAP_TIMEOUT_MS / 1000)
        if debug:
            print("  [Debug] Blazor circuit established")
    except asyncio.TimeoutError:
        if debug:
            print("  [Debug] Blazor WebSocket not detected (may still work)")

    page.remove_listener("websocket", on_websocket)

    # Wait for the email input field to render (proves Blazor bootstrapped)
    try:
        await page.wait_for_selector('input[type="email"]', timeout=BOOTSTRAP_TIMEOUT_MS)
        if debug:
            print(f"  [Debug] Login form ready at: {safe_url(page.url)}")
        # Extra pause to let Blazor circuit fully stabilize
        await rate_limit(1.0)
        return True
    except Exception:
        pass

    # Check if Blazor crashed (passive check, no polling)
    if await check_blazor_error(page):
        print("[CuddeLink] Blazor error UI is visible — runtime crashed")
        return False

    # Maybe already logged in (session cookie from previous run)
    cloudfront_img = await page.query_selector('img[src*="cloudfront.net"]')
    if cloudfront_img:
        print("[CuddeLink] Already logged in — photos visible!")
        return True

    if debug:
        print("  [Debug] Login form did not render")
    return False


async def submit_credentials(page, email: str, password: str, debug: bool = False):
    """Fill and submit the login form. Assumes Blazor has bootstrapped.

    CuddeLink uses Fluent UI web components (<fluent-text-field>) with Shadow
    DOM. page.fill() puts values in the shadow <input> but doesn't trigger
    Blazor's binding events, leaving the submit button disabled.

    Solution: click the field, type with real keystrokes (keyboard.type),
    then Tab to the next field. This fires keydown/input/keyup events that
    Blazor's @bind handlers listen for.
    """
    # Double-check: if photos are already visible, skip login
    cloudfront_img = await page.query_selector('img[src*="cloudfront.net"]')
    if cloudfront_img:
        print("[CuddeLink] Already logged in — skipping credential submit")
        return

    print("[CuddeLink] Filling login form...")

    # Click email field and type (real keystrokes for Blazor binding)
    # Try #username first, fall back to input[type="email"] if ID changes
    try:
        await page.click('#username', timeout=2000)
    except Exception:
        await page.click('input[type="email"]', timeout=3000)
    await page.keyboard.press('Control+a')
    await page.keyboard.type(email, delay=30)
    await rate_limit(0.3)

    # Tab to password field and type
    await page.keyboard.press('Tab')
    await rate_limit(0.2)
    await page.keyboard.type(password, delay=30)
    await rate_limit(0.3)

    if debug:
        # Check if submit button is enabled now
        disabled = await page.evaluate("""() => {
            const btn = document.querySelector('fluent-button[type="submit"], button[type="submit"]');
            return btn ? btn.hasAttribute('disabled') : 'no button';
        }""")
        print(f"  [Debug] Submit button disabled={disabled}")

    # Click the Sign In button with Playwright's trusted mouse event.
    # JS .click() doesn't create trusted events, so Blazor ignores it.
    # The <fluent-button> is a custom element; try multiple selectors.
    submit_selectors = [
        'fluent-button[type="submit"]',      # outer web component
        'fluent-button:has-text("Sign In")',  # by visible text
        'text=Sign In',                       # Playwright text selector
    ]
    clicked = False
    for sel in submit_selectors:
        try:
            await page.click(sel, timeout=3000)
            clicked = True
            if debug:
                print(f"  [Debug] Clicked submit via: {sel}")
            break
        except Exception:
            continue

    if not clicked:
        # Last resort: submit form via JS requestSubmit
        await page.evaluate("""() => {
            const form = document.querySelector('form[method="post"]');
            if (form && form.requestSubmit) form.requestSubmit();
            else if (form) form.submit();
        }""")
        if debug:
            print("  [Debug] Submitted form via JS requestSubmit")

    # Wait for photos page (CloudFront images appear after login)
    print("[CuddeLink] Waiting for photos page...")
    try:
        await page.wait_for_selector('img[src*="cloudfront.net"]', timeout=30000)
        print("[CuddeLink] Login successful — photos page loaded!")
    except Exception:
        current_url = page.url
        if "/Account/Login" in current_url or "/login" in current_url.lower():
            await capture_failure(page, "login_failed")
            raise RuntimeError(f"Login failed — still on login page: {safe_url(current_url)}")
        print(f"[CuddeLink] Login may have succeeded — URL: {safe_url(current_url)}")
        await rate_limit(5.0)


async def playwright_login(browser, email: str, password: str, debug: bool = False):
    """Login to CuddeLink with retry loop using fresh contexts.

    Each attempt creates a brand-new browser context (clean cookies,
    storage, service workers, and anti-forgery token). This prevents
    stale Blazor circuit state from poisoning retries.

    Returns (context, page, attempts_used) on success — caller must close context.
    """
    for attempt in range(1, MAX_BOOTSTRAP_ATTEMPTS + 1):
        print(f"[CuddeLink] Login attempt {attempt}/{MAX_BOOTSTRAP_ATTEMPTS}...")

        # Fresh context per attempt — no stale cookies/storage/tokens
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        try:
            ready = await wait_for_blazor_bootstrap(page, debug=debug)

            if ready:
                await submit_credentials(page, email, password, debug=debug)
                if attempt > 1:
                    print(f"[CuddeLink] WARNING: Login required {attempt} attempts")
                return context, page, attempt

        except Exception as e:
            await capture_failure(page, f"login_fail_attempt_{attempt}")
            if attempt >= MAX_BOOTSTRAP_ATTEMPTS:
                await context.close()
                raise
            print(f"[CuddeLink] Attempt {attempt} failed: {e}")

        # Bootstrap or login failed — capture state, close context, retry
        if not ready:
            await capture_failure(page, f"bootstrap_fail_attempt_{attempt}")
        await context.close()

        if attempt < MAX_BOOTSTRAP_ATTEMPTS:
            backoff = [2, 5, 9][attempt - 1] + random.uniform(0, 1)
            print(f"[CuddeLink] Retrying in {backoff:.1f}s...")
            await rate_limit(backoff)

    raise RuntimeError(
        f"Login failed after {MAX_BOOTSTRAP_ATTEMPTS} attempts"
    )


async def collect_photos_from_dom(page, max_photos: int = 500, debug: bool = False) -> list[dict]:
    """Extract unique photo URLs from the DOM.

    CuddeLink serves photos via CloudFront CDN:
    https://d9ekiakd1knvt.cloudfront.net/{device_uuid}/{timestamp}

    The Blazor app renders multiple <img> tags per photo (duplicates),
    so we dedupe by URL. Photos are already on the page after login —
    no separate navigation needed.
    """
    print("[Collect] Waiting for photos to load...")
    await rate_limit(5.0)

    # Collect all CloudFront image URLs from the DOM
    # The Blazor app may render photos with lazy loading, so scroll first
    collected = {}
    stall_count = 0
    scroll_count = 0
    start_time = time.time()

    PHOTO_SELECTOR = 'img[src*="cloudfront.net"]'

    while True:
        # Extract unique photo URLs from currently rendered DOM
        photo_data = await page.evaluate("""(selector) => {
            const imgs = document.querySelectorAll(selector);
            return Array.from(imgs).map(img => ({
                src: img.src,
                width: img.naturalWidth || img.width,
                height: img.naturalHeight || img.height,
            }));
        }""", PHOTO_SELECTOR)

        for item in photo_data:
            src = item["src"]
            if src and src not in collected:
                collected[src] = {
                    "src": src,
                    # TODO: these are 960x558 display thumbnails from CloudFront.
                    # Full-res originals may exist behind a lightbox or download
                    # link. Treat full-res discovery as unresolved.
                    "fullres_url": src,
                    "width": item["width"],
                    "height": item["height"],
                }

        prev_count = len(collected)

        # Scroll down to load more
        await page.evaluate("window.scrollBy(0, window.innerHeight)")
        scroll_count += 1
        await rate_limit(2.0)

        new_count = len(collected)
        elapsed = time.time() - start_time

        if debug and new_count > prev_count:
            print(f"  [Collect] Scroll {scroll_count}: {new_count} unique photos ({new_count - prev_count} new)")

        # Stall detection
        if new_count == prev_count:
            stall_count += 1
            if stall_count >= 3:
                print(f"[Collect] All photos loaded after {scroll_count} scrolls")
                break
        else:
            stall_count = 0

        # Safety caps
        if new_count >= max_photos:
            print(f"[Collect] Hit max photo cap ({max_photos})")
            break
        if elapsed > 120:
            print(f"[Collect] Scroll timeout (120s)")
            break

    photos = list(collected.values())
    print(f"[Collect] Found {len(photos)} unique photos (from CloudFront CDN)")
    return photos


async def download_photos(context, photo_metas: list[dict], dest_dir: Path, debug: bool = False) -> list[tuple[Path, dict]]:
    """Download photos using browser context (keeps cookies/auth).

    Returns list of (local_path, meta) tuples.
    """
    if not photo_metas:
        return []

    print(f"[Download] Downloading {len(photo_metas)} photos...")
    downloaded = []
    dl_start = time.time()

    for i, meta in enumerate(photo_metas):
        url = meta.get("fullres_url") or meta.get("src", "")
        if not url:
            continue

        try:
            response = await context.request.get(url, timeout=30000)

            if response.ok:
                body = await response.body()
                # Determine extension from content type
                content_type = response.headers.get("content-type", "image/jpeg")
                ext = ".jpg"
                if "png" in content_type:
                    ext = ".png"

                filename = f"cuddelink_{i:04d}{ext}"
                file_path = dest_dir / filename
                file_path.write_bytes(body)

                file_size_kb = len(body) / 1024
                if debug:
                    print(f"  [Download] {i+1}/{len(photo_metas)}: {filename} ({file_size_kb:.0f} KB)")

                downloaded.append((file_path, meta))
            else:
                print(f"  [Download] Failed {i+1}: HTTP {response.status}")

        except Exception as e:
            print(f"  [Download] Error {i+1}: {e}")

        await rate_limit(1.0)

    dl_elapsed = time.time() - dl_start
    avg_time = dl_elapsed / len(photo_metas) if photo_metas else 0
    print(f"[Download] Downloaded {len(downloaded)} of {len(photo_metas)} photos "
          f"in {dl_elapsed:.0f}s (avg {avg_time:.1f}s/photo)")
    return downloaded


def process_downloaded_photos(downloaded: list[tuple[Path, dict]], s3_client, config: dict, target_date_range: tuple = None) -> tuple[int, int, int]:
    """Process downloaded photos: hash, dedupe, upload to R2, save to Supabase.

    Returns (uploaded, skipped_duplicate, skipped_date).
    """
    uploaded = 0
    skipped_dup = 0
    skipped_date = 0

    for img_path, meta in downloaded:
        # Calculate hash
        file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()

        # Check for duplicate in Supabase
        if check_photo_exists(config["supabase_url"], config["supabase_key"], file_hash):
            print(f"  Skipping duplicate: {img_path.name} (hash: {file_hash[:8]})")
            skipped_dup += 1
            continue

        # Get EXIF datetime
        datetime_taken = get_exif_datetime(img_path)

        # Date range filter (when UI date filter wasn't available)
        if target_date_range and datetime_taken:
            try:
                photo_date = datetime.fromisoformat(datetime_taken).date()
                if photo_date < target_date_range[0] or photo_date > target_date_range[1]:
                    print(f"  Skipping out-of-range: {img_path.name} ({datetime_taken})")
                    skipped_date += 1
                    continue
            except (ValueError, TypeError):
                pass

        if not datetime_taken:
            print(f"    [WARNING] No EXIF timestamp for {img_path.name}")
            datetime_taken = f"{date.today().isoformat()}T00:00:00"
            print(f"    Using date-only placeholder: {datetime_taken}")

        # Upload full image to R2
        r2_key = f"photos/{file_hash}.jpg"
        if upload_to_r2(s3_client, config["r2_bucket"], img_path, r2_key):
            # Generate and upload thumbnail
            try:
                thumb_data = create_thumbnail(img_path)
                thumb_key = f"thumbnails/{file_hash}_thumb.jpg"
                upload_bytes_to_r2(s3_client, config["r2_bucket"], thumb_data, thumb_key)
                print(f"  Uploaded: {img_path.name} + thumbnail ({file_hash[:8]})")
            except Exception as e:
                print(f"  Uploaded: {img_path.name} (thumbnail failed: {e})")

            if datetime_taken:
                print(f"    EXIF timestamp: {datetime_taken}")

            # Save metadata to Supabase
            save_to_supabase(
                config["supabase_url"],
                config["supabase_key"],
                file_hash,
                datetime_taken,
                img_path.name,
            )
            uploaded += 1
        else:
            print(f"  Failed to upload: {img_path.name}")

    return uploaded, skipped_dup, skipped_date


async def apply_date_filter_if_possible(page, days_back: int, debug: bool = False) -> bool:
    """Try to apply a date filter via the UI. Returns True if successful."""
    print(f"[Filter] Looking for date filter UI (days_back={days_back})...")

    date_selectors = [
        'input[type="date"]',
        'input[class*="date"]',
        '[class*="date-picker"]',
        '[class*="datepicker"]',
        'input[placeholder*="date" i]',
    ]

    for selector in date_selectors:
        try:
            el = await page.query_selector(selector)
            if el:
                if debug:
                    print(f"  [Filter] Found date element: {selector}")
                # Found a date picker — try to use it
                today = date.today()
                start = today - timedelta(days=days_back)

                # Try filling it
                await el.fill(start.isoformat())
                await rate_limit(1.0)

                # Look for an "Apply" or "Filter" button
                apply_selectors = [
                    'button:has-text("Apply")',
                    'button:has-text("Filter")',
                    'button:has-text("Search")',
                    'button[type="submit"]',
                ]
                for btn_sel in apply_selectors:
                    btn = await page.query_selector(btn_sel)
                    if btn:
                        await btn.click()
                        await rate_limit(3.0)
                        print("[Filter] Date filter applied")
                        return True

                print("[Filter] Found date input but no apply button")
                return False
        except Exception:
            continue

    print("[Filter] No date filter UI found — will filter by EXIF date after download")
    return False


async def async_main(args):
    """Main async entry point."""
    from playwright.async_api import async_playwright

    # Load config
    config = {
        "cuddelink_email": get_env("CUDDELINK_EMAIL"),
        "cuddelink_password": get_env("CUDDELINK_PASSWORD"),
        "r2_endpoint": get_env("R2_ENDPOINT"),
        "r2_access_key": get_env("R2_ACCESS_KEY"),
        "r2_secret_key": get_env("R2_SECRET_KEY"),
        "r2_bucket": get_env("R2_BUCKET"),
        "supabase_url": get_env("SUPABASE_URL"),
        "supabase_key": get_env("SUPABASE_KEY"),
    }

    debug = args.debug

    # Set up R2 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=config["r2_endpoint"],
        aws_access_key_id=config["r2_access_key"],
        aws_secret_access_key=config["r2_secret_key"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    uploaded = 0
    skipped_dup = 0
    skipped_date = 0

    # Load local URL cache to skip known CloudFront URLs before downloading
    seen_urls_path = Path("/tmp/cuddelink_seen_urls.json")
    seen_urls = set()
    try:
        if seen_urls_path.exists():
            seen_urls = set(json.loads(seen_urls_path.read_text()))
            print(f"[Cache] Loaded {len(seen_urls)} known URLs from local cache")
    except Exception:
        pass

    async with async_playwright() as p:
        # Always run headed — Blazor misbehaves in headless mode.
        # CI uses xvfb-run to provide a virtual display.
        chrome_channel = "chrome"

        # Use system Chrome — Playwright's bundled Chromium headless
        # shell crashes Blazor WebAssembly apps.
        # slow_mo=300 for login phase (Blazor form bindings need it).
        try:
            browser = await p.chromium.launch(
                channel=chrome_channel, headless=False, slow_mo=300,
            )
            print("[Browser] Using system Chrome (slow_mo=300 for login)")
        except Exception:
            chrome_channel = None
            browser = await p.chromium.launch(headless=False, slow_mo=300)
            print("[Browser] Using bundled Chromium (slow_mo=300 for login)")

        login_context = None
        login_attempts = 0
        try:
            # Phase 1: Login — each attempt creates a fresh context
            # (clean cookies, storage, service workers, anti-forgery token)
            login_context, page, login_attempts = await playwright_login(
                browser, config["cuddelink_email"], config["cuddelink_password"], debug=debug,
            )

            # Phase 2: Try date filter (best effort)
            filtered = await apply_date_filter_if_possible(page, args.days_back, debug=debug)

            # Phase 3: Scroll and collect unique CloudFront photo URLs
            photo_metas = await collect_photos_from_dom(page, max_photos=args.max_photos, debug=debug)

            if not photo_metas:
                print("[Sync] No photos found!")
                await capture_failure(page, "no_photos")
                return

            # Phase 4: Filter out already-seen URLs (local cache)
            new_metas = [m for m in photo_metas if m["src"] not in seen_urls]
            if len(new_metas) < len(photo_metas):
                print(f"[Cache] Skipped {len(photo_metas) - len(new_metas)} "
                      f"already-seen URLs, {len(new_metas)} to download")
            photo_metas = new_metas

            if not photo_metas:
                print("[Sync] All photos already seen — nothing to download")
                return

        except Exception as e:
            print(f"[ERROR] {e}")
            raise
        finally:
            if login_context:
                await login_context.close()
            await browser.close()

        # Phase 5: Download with a FAST browser (no slow_mo)
        launch_args = {"headless": False}
        try:
            if chrome_channel:
                dl_browser = await p.chromium.launch(channel=chrome_channel, **launch_args)
            else:
                dl_browser = await p.chromium.launch(**launch_args)
        except Exception:
            dl_browser = await p.chromium.launch(**launch_args)

        dl_context = await dl_browser.new_context()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dest_dir = Path(tmp)

                downloaded = await download_photos(dl_context, photo_metas, dest_dir, debug=debug)

                if not downloaded:
                    print("[Sync] No photos downloaded!")
                    return

                # Phase 6: Process, upload, save
                target_date_range = None
                if not filtered:
                    today = date.today()
                    start = today - timedelta(days=args.days_back)
                    target_date_range = (start, today)

                uploaded, skipped_dup, skipped_date = process_downloaded_photos(
                    downloaded, s3_client, config, target_date_range
                )

                if not filtered and skipped_date > 0:
                    pct = (skipped_date / len(downloaded)) * 100 if downloaded else 0
                    print(f"[Filter] EXIF date filter: kept {len(downloaded) - skipped_date}, "
                          f"skipped {skipped_date} out-of-range ({pct:.0f}%)")
                    if pct > 50:
                        print("[WARNING] Most downloaded photos were outside date range. "
                              "Consider implementing UI date filter for efficiency.")

            # Update local URL cache with all collected URLs (including dupes)
            all_urls = {m["src"] for m in photo_metas}
            seen_urls.update(all_urls)
            try:
                seen_urls_path.write_text(json.dumps(sorted(seen_urls)))
                print(f"[Cache] Saved {len(seen_urls)} URLs to local cache")
            except Exception:
                pass

        finally:
            await dl_browser.close()

    total_skipped = skipped_dup + skipped_date
    print(f"\n[Sync] Complete! Uploaded: {uploaded}, Skipped: {total_skipped} "
          f"(duplicates: {skipped_dup}, out-of-range: {skipped_date})")
    print(f"[Telemetry] Login attempts: {login_attempts}, "
          f"Photos collected: {len(photo_metas)}")


def main():
    parser = argparse.ArgumentParser(description="Sync CuddeLink photos to R2 (Playwright)")
    parser.add_argument("--days-back", type=int, default=1, help="Days to look back")
    parser.add_argument("--max-photos", type=int, default=500,
                        help="Max photos to collect before stopping (default 500)")
    parser.add_argument("--debug", action="store_true",
                        help="Run headed browser with slow_mo + verbose logging")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
