#!/usr/bin/env python3
"""
Generate a manifest.json file for delta updates.

This script scans the built distribution folder and creates a manifest
containing SHA256 hashes of all files. The manifest is used to determine
which files have changed between versions, enabling delta updates.

Usage:
    python build_manifest.py [dist_folder] [output_file]

    dist_folder: Path to the distribution folder (default: dist/TrailCamOrganizer)
    output_file: Path to output manifest (default: dist/TrailCamOrganizer/manifest.json)
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Import version if available
try:
    from version import __version__
except ImportError:
    __version__ = "unknown"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def generate_manifest(dist_folder: Path) -> dict:
    """
    Generate manifest for all files in the distribution folder.

    Returns dict with structure:
    {
        "version": "1.0.3",
        "generated": "2024-01-01T12:00:00",
        "total_size": 488000000,
        "file_count": 150,
        "files": {
            "TrailCamOrganizer.exe": {
                "sha256": "abc123...",
                "size": 12345678
            },
            ...
        }
    }
    """
    files = {}
    total_size = 0

    print(f"Scanning {dist_folder}...")

    for root, dirs, filenames in os.walk(dist_folder):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__"]

        for filename in filenames:
            file_path = Path(root) / filename
            # Get relative path from dist folder
            rel_path = file_path.relative_to(dist_folder)
            # Use forward slashes for cross-platform compatibility
            rel_path_str = str(rel_path).replace("\\", "/")

            try:
                file_hash = compute_sha256(file_path)
                file_size = get_file_size(file_path)

                files[rel_path_str] = {
                    "sha256": file_hash,
                    "size": file_size
                }
                total_size += file_size

            except Exception as e:
                print(f"  Warning: Could not hash {rel_path_str}: {e}")

    manifest = {
        "version": __version__,
        "generated": datetime.utcnow().isoformat() + "Z",
        "total_size": total_size,
        "file_count": len(files),
        "files": files
    }

    print(f"Generated manifest for {len(files)} files ({total_size / 1024 / 1024:.1f} MB)")

    return manifest


def compare_manifests(local: dict, remote: dict) -> dict:
    """
    Compare two manifests and return changed/added/removed files.

    Returns dict with:
    {
        "added": ["new_file.dll", ...],
        "removed": ["old_file.dll", ...],
        "changed": ["modified_file.exe", ...],
        "unchanged": ["same_file.dll", ...],
        "download_size": 12345678  # bytes needed to download
    }
    """
    local_files = local.get("files", {})
    remote_files = remote.get("files", {})

    local_keys = set(local_files.keys())
    remote_keys = set(remote_files.keys())

    added = remote_keys - local_keys
    removed = local_keys - remote_keys
    common = local_keys & remote_keys

    changed = set()
    unchanged = set()

    for key in common:
        if local_files[key]["sha256"] != remote_files[key]["sha256"]:
            changed.add(key)
        else:
            unchanged.add(key)

    # Calculate download size (added + changed files)
    download_size = 0
    for key in added | changed:
        download_size += remote_files[key]["size"]

    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "changed": sorted(changed),
        "unchanged": sorted(unchanged),
        "download_size": download_size,
        "download_count": len(added) + len(changed)
    }


def load_manifest(path: Path) -> dict:
    """Load manifest from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_manifest(manifest: dict, path: Path):
    """Save manifest to JSON file."""
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {path}")


def main():
    # Parse arguments
    if len(sys.argv) >= 2:
        dist_folder = Path(sys.argv[1])
    else:
        # Default to Windows dist folder
        dist_folder = Path("dist/TrailCamOrganizer")

    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = dist_folder / "manifest.json"

    if not dist_folder.exists():
        print(f"Error: Distribution folder not found: {dist_folder}")
        print("Run PyInstaller first to create the distribution.")
        sys.exit(1)

    # Generate manifest
    manifest = generate_manifest(dist_folder)

    # Save to file
    save_manifest(manifest, output_file)

    # Print summary
    print(f"\nManifest Summary:")
    print(f"  Version: {manifest['version']}")
    print(f"  Files: {manifest['file_count']}")
    print(f"  Total Size: {manifest['total_size'] / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
