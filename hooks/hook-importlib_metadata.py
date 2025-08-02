# hook-importlib_metadata.py
from PyInstaller.utils.hooks import collect_submodules

# Ensure importlib.metadata and its submodules are included
hiddenimports = collect_submodules("importlib.metadata")
