# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Trail Camera Organizer - Windows build
"""

import sys
import glob
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Bundle Visual C++ runtime DLLs so the app works on machines without VC++ installed
vcruntime_dlls = []
if sys.platform == 'win32':
    # Find vcruntime and msvcp DLLs from Python's install directory
    python_dir = Path(sys.executable).parent
    for pattern in ['vcruntime*.dll', 'msvcp*.dll', 'concrt*.dll', 'vcomp*.dll']:
        for dll in python_dir.glob(pattern):
            vcruntime_dlls.append((str(dll), '.'))
    # Also check Windows system directory for VC++ and OpenSSL DLLs
    system32 = Path(r'C:\Windows\System32')
    for pattern in ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll',
                     'libcrypto-3-x64.dll', 'libcrypto-3.dll',
                     'libssl-3-x64.dll', 'libssl-3.dll']:
        dll = system32 / pattern
        if dll.exists() and not any(pattern in str(d) for d in vcruntime_dlls):
            vcruntime_dlls.append((str(dll), '.'))
    # Also look for OpenSSL DLLs in Python's DLLs directory
    python_dlls_dir = python_dir / 'DLLs'
    if python_dlls_dir.exists():
        for pattern in ['libcrypto*.dll', 'libssl*.dll']:
            for dll in python_dlls_dir.glob(pattern):
                vcruntime_dlls.append((str(dll), '.'))

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
    ('sync_manager.py', '.'),
    ('updater.py', '.'),
    ('version.py', '.'),
    ('r2_storage.py', '.'),
    ('user_config.py', '.'),
    ('site_detector.py', '.'),
    ('site_embedder.py', '.'),
    ('site_identifier.py', '.'),
    ('cloud_config.json', '.'),
]
datas += pyqt6_datas

# Collect binaries (including VC++ runtime for machines without it installed)
binaries = []
binaries += vcruntime_dlls
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
    'supabase_rest',
    'database',
    'ai_detection',
    'ai_suggester',
    'preview_window',
    'compare_window',
    'cuddelink_downloader',
    'image_processor',
    'duplicate_dialog',
    'sync_manager',
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
    ['trainer_main.py'],
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
        # Exclude training/research-only deps (keeps PyInstaller from importing them)
        'torch',
        'torchvision',
        'onnx',
        'onnx.reference',
        'speciesnet',
        'scipy',
        'sympy',
        'pandas',
        'matplotlib',
        'tensorboard',
        'ultralytics',
        'yolov5',
        'sahi',
        'kagglehub',
        'huggingface_hub',
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
    contents_directory='_internal',
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
