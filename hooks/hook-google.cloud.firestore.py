# pylint: disable=invalid-name
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all submodules for google.cloud.firestore
hiddenimports = collect_submodules("google.cloud.firestore")

# Also collect core google.cloud modules that firestore depends on
hiddenimports += collect_submodules("google.cloud.firestore_v1")
hiddenimports += collect_submodules("google.cloud.client")
hiddenimports += collect_submodules("google.cloud._helpers")

# Collect any data files that might be needed
datas = collect_data_files("google.cloud.firestore")
