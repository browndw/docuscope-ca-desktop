# pylint: disable=invalid-name
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("lingua")
