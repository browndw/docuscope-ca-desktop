# PyInstaller hook for pandasai-openai
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# Collect all pandasai-openai modules and data
datas, binaries, hiddenimports = collect_all('pandasai_openai')

# Add specific submodules that might be missed
hiddenimports += collect_submodules('pandasai_openai')

# Collect metadata files specifically
datas += collect_data_files('pandasai_openai', include_py_files=True)

# Add specific modules that are commonly needed (if any exist)
hiddenimports += [
    # Most pandasai_openai modules are auto-discovered
]
