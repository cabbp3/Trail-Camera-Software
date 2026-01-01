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
import platform
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import requests

from version import __version__, GITHUB_REPO

logger = logging.getLogger(__name__)

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
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Create temp file
        temp_dir = Path(tempfile.gettempdir()) / "trailcam_update"
        temp_dir.mkdir(exist_ok=True)

        filename = download_url.split("/")[-1]
        temp_file = temp_dir / filename

        downloaded = 0
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size:
                        progress_callback(downloaded, total_size)

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
            # Windows installer - just run it
            subprocess.Popen([str(update_file)], shell=True)
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
