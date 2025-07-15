# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules("webapp")

# Add specific modules that might be needed
hiddenimports += [
    'pydantic_core',
    'pydantic_core._pydantic_core',
    'google.cloud.firestore',
    'google.cloud.firestore_v1',
    'pandasai',
    'pandasai.agent',
    'pandasai.llm',
    'pandasai_openai',
]

a = Analysis(
    ['entrypoint.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    a.scripts,
    [],
    exclude_binaries=False,
    name="entrypoint",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)