# pylint: disable=invalid-name
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

datas = [("./webapp", "./webapp")]
datas += copy_metadata("docuscope-ca-desktop")
hiddenimports = collect_submodules("webapp")
