# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Trail Camera Organizer - Windows build
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Collect PyQt6 completely
pyqt6_datas, pyqt6_binaries, pyqt6_hiddenimports = collect_all('PyQt6')

# Collect all data files
datas = [
    ('models', 'models'),
    ('training', 'training'),
    ('yolov8n.pt', '.'),
    ('supabase_rest.py', '.'),
    ('database.py', '.'),
    ('ai_detection.py', '.'),
    ('ai_suggester.py', '.'),
    ('preview_window.py', '.'),
    ('compare_window.py', '.'),
    ('cuddelink_downloader.py', '.'),
    ('image_processor.py', '.'),
    ('duplicate_dialog.py', '.'),
    ('site_clustering.py', '.'),
    ('organizer_ui.py', '.'),
    ('updater.py', '.'),
    ('version.py', '.'),
    ('r2_storage.py', '.'),
    ('user_config.py', '.'),
    ('site_detector.py', '.'),
    ('site_embedder.py', '.'),
    ('site_identifier.py', '.'),
]
datas += pyqt6_datas

# Collect binaries
binaries = []
binaries += pyqt6_binaries

# Hidden imports
hiddenimports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    'PIL',
    'PIL.Image',
    'PIL.ExifTags',
    'PIL._imaging',
    'numpy',
    'onnxruntime',
    'sqlite3',
    'requests',
    'certifi',
    'pandas',
    'openpyxl',
    'supabase_rest',
    'database',
    'ai_detection',
    'ai_suggester',
    'preview_window',
    'compare_window',
    'cuddelink_downloader',
    'image_processor',
    'duplicate_dialog',
    'organizer_ui',
    'updater',
    'version',
    'r2_storage',
    'user_config',
    'site_detector',
    'site_embedder',
    'site_identifier',
    'boto3',
    'botocore',
    'pytesseract',
]
hiddenimports += pyqt6_hiddenimports
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('onnxruntime')
hiddenimports += collect_submodules('boto3')
hiddenimports += collect_submodules('botocore')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TrailCamOrganizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if Path('icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TrailCamOrganizer',
)
