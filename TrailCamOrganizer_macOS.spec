# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Trail Camera Organizer - macOS build
Creates a standalone .app bundle
"""

import sys
from pathlib import Path

block_cipher = None

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
    ('cloud_config.json', '.'),
]

# Hidden imports - only what we actually need
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
    'urllib3',
    'certifi',
    'charset_normalizer',
    'idna',
    'pandas',
    'openpyxl',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'torch',
        'torchvision',
        'PyQt6.QtQuick',
        'PyQt6.QtQml',
        'PyQt6.QtBluetooth',
        'PyQt6.QtMultimedia',
        'PyQt6.QtNetwork',
        'PyQt6.QtPositioning',
        'PyQt6.QtWebChannel',
        'PyQt6.QtWebEngine',
        'PyQt6.QtWebSockets',
        'PyQt6.Qt3D',
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
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TrailCamOrganizer',
)

app = BUNDLE(
    coll,
    name='TrailCamOrganizer.app',
    icon='icon.icns' if Path('icon.icns').exists() else None,
    bundle_identifier='com.trailcam.organizer',
    info_plist={
        'CFBundleName': 'Trail Camera Organizer',
        'CFBundleDisplayName': 'Trail Camera Organizer',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
    },
)
