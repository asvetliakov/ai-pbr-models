# hook-tqdm.py
from PyInstaller.utils.hooks import copy_metadata

# Include tqdm metadata so transformers can verify the version
datas = copy_metadata("tqdm")
