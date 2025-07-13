# PyInstaller hook for pandasai
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# Collect all pandasai modules and data
datas, binaries, hiddenimports = collect_all('pandasai')

# Add specific submodules that might be missed
hiddenimports += collect_submodules('pandasai')

# Collect metadata files specifically
datas += collect_data_files('pandasai', include_py_files=True)

# Add specific modules that are commonly needed
hiddenimports += [
    'pandasai.agent',
    'pandasai.helpers',
    'pandasai.llm',
    'pandasai.vectorstores',
    'pandasai.exceptions',
    'pandasai.config',
    'pandasai.constants',
    'pandasai.smart_dataframe',
    'pandasai.smart_datalake',
]
