# pylint: disable=invalid-name
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs
import os
import glob

# Use collect_all as base
datas, binaries, hiddenimports = collect_all('pydantic_core')

# Explicitly collect any .so files from pydantic_core
try:
    import pydantic_core
    pydantic_core_path = os.path.dirname(pydantic_core.__file__)
    # Look for .so files in the pydantic_core directory
    so_files = glob.glob(os.path.join(pydantic_core_path, '*.so'))
    for so_file in so_files:
        binaries.append((so_file, '.'))
except Exception:
    pass

# Also try to collect dynamic libraries
try:
    dynamic_libs = collect_dynamic_libs('pydantic_core')
    binaries.extend(dynamic_libs)
except Exception:
    pass

# Add specific hidden imports that might be missed
hiddenimports += [
    'pydantic_core._pydantic_core',
    'pydantic_core.core_schema',
]
