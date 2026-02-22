"""
Auto-update module for Trail Camera Software.
Checks GitHub releases and downloads/installs updates.
"""

import os
import sys
import json
import shutil
import tempfile
import zipfile
import hashlib
import platform
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import requests

from version import __version__, GITHUB_REPO

logger = logging.getLogger(__name__)

# Import manifest utilities if available
try:
    from build_manifest import compare_manifests, compute_sha256
except ImportError:
    # Fallback implementations for compiled app
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def compare_manifests(local: dict, remote: dict) -> dict:
        """Compare two manifests and return changed/added/removed files."""
        local_files = local.get("files", {})
        remote_files = remote.get("files", {})

        local_keys = set(local_files.keys())
        remote_keys = set(remote_files.keys())

        added = remote_keys - local_keys
        removed = local_keys - remote_keys
        common = local_keys & remote_keys

        changed = set()
        for key in common:
            if local_files[key]["sha256"] != remote_files[key]["sha256"]:
                changed.add(key)

        download_size = 0
        for key in added | changed:
            download_size += remote_files[key]["size"]

        return {
            "added": sorted(added),
            "removed": sorted(removed),
            "changed": sorted(changed),
            "download_size": download_size,
            "download_count": len(added) + len(changed)
        }

# GitHub API endpoint
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


def get_current_version() -> str:
    """Get the current app version."""
    return __version__


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string like '1.2.3' into tuple (1, 2, 3)."""
    # Remove 'v' prefix if present
    version_str = version_str.lstrip('v').lstrip('V')
    try:
        parts = version_str.split('.')
        return tuple(int(p) for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def check_for_updates() -> Optional[Dict]:
    """
    Check GitHub for latest release.

    Returns:
        Dict with update info if newer version available, None otherwise.
        Dict contains: version, download_url, release_notes, published_at
    """
    try:
        response = requests.get(
            GITHUB_API_URL,
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10
        )
        response.raise_for_status()

        release = response.json()
        latest_version = release.get("tag_name", "")

        # Compare versions
        current = parse_version(__version__)
        latest = parse_version(latest_version)

        if latest > current:
            # Find the right asset for this platform
            system = platform.system()
            assets = release.get("assets", [])

            download_url = None
            asset_name = None

            for asset in assets:
                name = asset.get("name", "").lower()
                if system == "Windows" and ("windows" in name or name.endswith(".exe") or "win" in name):
                    download_url = asset.get("browser_download_url")
                    asset_name = asset.get("name")
                    break
                elif system == "Darwin" and ("macos" in name or "mac" in name or name.endswith(".app") or name.endswith(".dmg")):
                    download_url = asset.get("browser_download_url")
                    asset_name = asset.get("name")
                    break
                elif ".zip" in name:
                    # Fallback to any zip file
                    download_url = asset.get("browser_download_url")
                    asset_name = asset.get("name")

            return {
                "version": latest_version,
                "current_version": __version__,
                "download_url": download_url,
                "asset_name": asset_name,
                "release_notes": release.get("body", ""),
                "published_at": release.get("published_at", ""),
                "html_url": release.get("html_url", "")
            }

        return None  # No update available

    except requests.RequestException as e:
        logger.error(f"Failed to check for updates: {e}")
        raise Exception(f"Could not connect to update server: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse update response: {e}")
        raise Exception("Invalid response from update server")


def get_local_manifest() -> Optional[Dict]:
    """
    Load the local manifest.json from the app directory.

    Returns:
        Dict with manifest data, or None if not found.
    """
    try:
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            app_dir = Path(sys.executable).parent
        else:
            # Running as script - check dist folder for testing
            app_dir = Path(__file__).parent / "dist" / "TrailCamOrganizer"
            if not app_dir.exists():
                app_dir = Path(__file__).parent

        manifest_path = app_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                return json.load(f)

    except Exception as e:
        logger.warning(f"Could not load local manifest: {e}")

    return None


def get_remote_manifest(release_info: Dict) -> Optional[Dict]:
    """
    Download the remote manifest.json from GitHub release.

    Args:
        release_info: Dict from check_for_updates() containing release info

    Returns:
        Dict with manifest data, or None if not found.
    """
    try:
        # Try to find manifest in release assets
        response = requests.get(
            GITHUB_API_URL,
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10
        )
        response.raise_for_status()
        release = response.json()

        # Look for manifest-windows.json or manifest.json
        system = platform.system()
        manifest_name = f"manifest-{system.lower()}.json" if system == "Windows" else "manifest.json"

        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if name == manifest_name or name == "manifest.json":
                manifest_url = asset.get("browser_download_url")
                if manifest_url:
                    resp = requests.get(manifest_url, timeout=30)
                    resp.raise_for_status()
                    return resp.json()

    except Exception as e:
        logger.warning(f"Could not download remote manifest: {e}")

    return None


def check_delta_update(release_info: Dict) -> Optional[Dict]:
    """
    Check if a delta update is possible and calculate download size.

    Args:
        release_info: Dict from check_for_updates()

    Returns:
        Dict with delta update info, or None if full download required.
        Contains: changed_files, added_files, removed_files, download_size, download_count
    """
    local_manifest = get_local_manifest()
    if not local_manifest:
        logger.info("No local manifest found, full download required")
        return None

    remote_manifest = get_remote_manifest(release_info)
    if not remote_manifest:
        logger.info("No remote manifest found, full download required")
        return None

    diff = compare_manifests(local_manifest, remote_manifest)

    return {
        "changed_files": diff.get("changed", []),
        "added_files": diff.get("added", []),
        "removed_files": diff.get("removed", []),
        "download_size": diff.get("download_size", 0),
        "download_count": diff.get("download_count", 0),
        "local_manifest": local_manifest,
        "remote_manifest": remote_manifest
    }


def download_delta_update(
    download_url: str,
    delta_info: Dict,
    progress_callback=None
) -> Path:
    """
    Download only changed files from the update ZIP.

    This downloads the full ZIP but only extracts changed files,
    which is simpler than hosting individual files separately.
    Future enhancement: download individual files if hosted separately.

    Args:
        download_url: URL to download ZIP from
        delta_info: Dict from check_delta_update()
        progress_callback: Optional callback(downloaded_bytes, total_bytes)

    Returns:
        Path to temp directory with extracted changed files
    """
    # For now, download the full ZIP (GitHub doesn't support range requests on release assets)
    # But only extract the files we need
    zip_path = download_update(download_url, progress_callback)

    # Create extraction directory
    extract_dir = zip_path.parent / "delta_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()

    # Get list of files to extract
    files_to_extract = set(delta_info.get("changed_files", []) + delta_info.get("added_files", []))

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all file names in zip
            zip_names = zip_ref.namelist()

            for zip_name in zip_names:
                # Normalize path (remove leading folder if present)
                # ZIP might contain "TrailCamOrganizer/file.dll" or just "file.dll"
                parts = zip_name.split('/')
                if len(parts) > 1 and parts[0] == "TrailCamOrganizer":
                    rel_path = '/'.join(parts[1:])
                else:
                    rel_path = zip_name

                if rel_path in files_to_extract or zip_name in files_to_extract:
                    # Extract to temp folder maintaining directory structure
                    zip_ref.extract(zip_name, extract_dir)

        # Move manifest.json to extracted folder if present
        for manifest_name in ["manifest.json", "manifest-windows.json"]:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for name in zip_ref.namelist():
                        if name.endswith(manifest_name):
                            zip_ref.extract(name, extract_dir)
                            break
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Failed to extract delta files: {e}")
        raise Exception(f"Delta extraction failed: {e}")

    return extract_dir


def install_delta_update(extract_dir: Path, delta_info: Dict) -> bool:
    """
    Install delta update by replacing only changed files.

    Args:
        extract_dir: Path to extracted delta files
        delta_info: Dict from check_delta_update()

    Returns:
        True if installation started successfully
    """
    system = platform.system()

    if getattr(sys, 'frozen', False):
        app_dir = Path(sys.executable).parent
    else:
        app_dir = Path(__file__).parent

    # Find the extracted app folder
    extracted_app = None
    for item in extract_dir.iterdir():
        if item.is_dir():
            extracted_app = item
            break
    if not extracted_app:
        extracted_app = extract_dir

    # Build list of files to copy
    files_to_copy = delta_info.get("changed_files", []) + delta_info.get("added_files", [])
    files_to_remove = delta_info.get("removed_files", [])

    if system == "Windows":
        return _install_delta_windows(app_dir, extracted_app, files_to_copy, files_to_remove)
    elif system == "Darwin":
        return _install_delta_macos(app_dir, extracted_app, files_to_copy, files_to_remove)

    return False


def _install_delta_windows(
    app_dir: Path,
    extracted_app: Path,
    files_to_copy: List[str],
    files_to_remove: List[str]
) -> bool:
    """Install delta update on Windows."""
    # Create batch script for delta update
    temp_dir = extracted_app.parent
    batch_script = temp_dir / "delta_installer.bat"

    # Build copy commands for each file
    copy_commands = []
    for rel_path in files_to_copy:
        src = extracted_app / rel_path.replace("/", "\\")
        dst = app_dir / rel_path.replace("/", "\\")
        # Create parent directory if needed
        dst_dir = dst.parent
        copy_commands.append(f'if not exist "{dst_dir}" mkdir "{dst_dir}"')
        copy_commands.append(f'copy /y "{src}" "{dst}"')

    # Build delete commands for removed files
    delete_commands = []
    for rel_path in files_to_remove:
        dst = app_dir / rel_path.replace("/", "\\")
        delete_commands.append(f'if exist "{dst}" del /q "{dst}"')

    batch_content = f'''@echo off
echo Waiting for app to close...
timeout /t 2 /nobreak > nul

echo Installing delta update ({len(files_to_copy)} files)...
{chr(10).join(copy_commands)}

echo Removing old files ({len(files_to_remove)} files)...
{chr(10).join(delete_commands)}

echo Starting updated app...
start "" "{app_dir}\\TrailCamOrganizer.exe"

echo Cleaning up...
rmdir /s /q "{temp_dir}"
del "%~f0"
'''

    with open(batch_script, 'w') as f:
        f.write(batch_content)

    subprocess.Popen(
        ['cmd', '/c', str(batch_script)],
        creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
    )

    return True


def _install_delta_macos(
    app_dir: Path,
    extracted_app: Path,
    files_to_copy: List[str],
    files_to_remove: List[str]
) -> bool:
    """Install delta update on macOS."""
    temp_dir = extracted_app.parent
    script_path = temp_dir / "delta_installer.sh"

    # Build copy commands
    copy_commands = []
    for rel_path in files_to_copy:
        src = extracted_app / rel_path
        dst = app_dir / rel_path
        copy_commands.append(f'mkdir -p "$(dirname "{dst}")"')
        copy_commands.append(f'cp -f "{src}" "{dst}"')

    # Build delete commands
    delete_commands = []
    for rel_path in files_to_remove:
        dst = app_dir / rel_path
        delete_commands.append(f'rm -f "{dst}"')

    script_content = f'''#!/bin/bash
sleep 2
echo "Installing delta update ({len(files_to_copy)} files)..."
{chr(10).join(copy_commands)}
echo "Removing old files ({len(files_to_remove)} files)..."
{chr(10).join(delete_commands)}
echo "Restarting app..."
open "{app_dir.parent}"
rm -rf "{temp_dir}"
rm "$0"
'''

    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    subprocess.Popen(['bash', str(script_path)])
    return True


def download_update(download_url: str, progress_callback=None) -> Path:
    """
    Download update file to temp directory.

    Args:
        download_url: URL to download from
        progress_callback: Optional callback(downloaded_bytes, total_bytes)

    Returns:
        Path to downloaded file
    """
    try:
        # Use a session to follow redirects properly (GitHub uses redirects for release assets)
        session = requests.Session()
        response = session.get(
            download_url,
            stream=True,
            timeout=(10, 300),  # (connect timeout, read timeout)
            allow_redirects=True,
            headers={"Accept": "application/octet-stream"}
        )
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Create temp file
        temp_dir = Path(tempfile.gettempdir()) / "trailcam_update"
        temp_dir.mkdir(exist_ok=True)

        filename = download_url.split("/")[-1]
        temp_file = temp_dir / filename

        downloaded = 0
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size:
                        progress_callback(downloaded, total_size)

        # Verify we actually got a zip file and not an HTML error page
        if temp_file.stat().st_size < 1_000_000:
            # Release zip should be >100MB — something went wrong
            with open(temp_file, 'rb') as f:
                header = f.read(4)
            if header[:2] != b'PK':  # ZIP magic bytes
                content_preview = temp_file.read_text(errors='replace')[:200]
                logger.error(f"Download is not a valid ZIP. Size: {temp_file.stat().st_size}, "
                           f"Preview: {content_preview}")
                raise Exception(
                    f"Download failed — received {temp_file.stat().st_size} bytes "
                    f"instead of the expected ZIP file. The server may be temporarily unavailable."
                )

        return temp_file

    except requests.RequestException as e:
        logger.error(f"Failed to download update: {e}")
        raise Exception(f"Download failed: {e}")


def install_update(update_file: Path) -> bool:
    """
    Install the downloaded update.

    For Windows: Extract zip and replace files, then restart.
    For macOS: Extract and replace .app bundle or prompt user.

    Args:
        update_file: Path to downloaded update file

    Returns:
        True if installation started successfully
    """
    system = platform.system()

    try:
        if update_file.suffix.lower() == '.zip':
            # Extract zip file
            extract_dir = update_file.parent / "extracted"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)

            with zipfile.ZipFile(update_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            if system == "Windows":
                return _install_windows(extract_dir)
            elif system == "Darwin":
                return _install_macos(extract_dir)

        elif update_file.suffix.lower() == '.exe':
            # Windows installer - just run it (shell=False for security)
            subprocess.Popen([str(update_file)])
            return True

        elif update_file.suffix.lower() == '.dmg':
            # macOS DMG - open it for user to install
            subprocess.Popen(['open', str(update_file)])
            return True

        return False

    except Exception as e:
        logger.error(f"Failed to install update: {e}")
        raise Exception(f"Installation failed: {e}")


def _install_windows(extract_dir: Path) -> bool:
    """Install update on Windows by replacing files."""
    # Find the app directory
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        app_dir = Path(sys.executable).parent
    else:
        # Running as script
        app_dir = Path(__file__).parent

    # Find extracted app folder (should be TrailCamOrganizer or similar)
    extracted_app = None
    for item in extract_dir.iterdir():
        if item.is_dir():
            extracted_app = item
            break

    if not extracted_app:
        # Files are directly in extract_dir
        extracted_app = extract_dir

    # Create a batch script to replace files after app closes
    batch_script = extract_dir.parent / "update_installer.bat"

    batch_content = f'''@echo off
echo Waiting for app to close...
timeout /t 2 /nobreak > nul

echo Installing update...
xcopy /s /y /q "{extracted_app}\\*" "{app_dir}\\"

echo Starting updated app...
start "" "{app_dir}\\TrailCamOrganizer.exe"

echo Cleaning up...
rmdir /s /q "{extract_dir.parent}"
del "%~f0"
'''

    with open(batch_script, 'w') as f:
        f.write(batch_content)

    # Run the batch script and exit
    subprocess.Popen(['cmd', '/c', str(batch_script)],
                     creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0)

    return True


def _install_macos(extract_dir: Path) -> bool:
    """Install update on macOS."""
    # Find .app bundle in extracted files
    app_bundle = None
    for item in extract_dir.rglob("*.app"):
        app_bundle = item
        break

    if app_bundle:
        # Move to Applications or replace current app
        if getattr(sys, 'frozen', False):
            # Running as .app - replace it
            current_app = Path(sys.executable).parent.parent.parent
            if current_app.suffix == '.app':
                # Create shell script to replace after app closes
                script_path = extract_dir.parent / "update_installer.sh"
                script_content = f'''#!/bin/bash
sleep 2
rm -rf "{current_app}"
cp -R "{app_bundle}" "{current_app.parent}/"
open "{current_app}"
rm -rf "{extract_dir.parent}"
rm "$0"
'''
                with open(script_path, 'w') as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)
                subprocess.Popen(['bash', str(script_path)])
                return True

        # Fallback: Open the folder for manual installation
        subprocess.Popen(['open', str(extract_dir)])
        return True

    return False


def cleanup_update_files():
    """Clean up any leftover update files."""
    temp_dir = Path(tempfile.gettempdir()) / "trailcam_update"
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup update files: {e}")
