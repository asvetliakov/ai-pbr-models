# hook-transformers-deps.py
from PyInstaller.utils.hooks import copy_metadata

# Include metadata for all common transformers dependencies
# This prevents importlib.metadata.PackageNotFoundError for various packages
# that transformers checks during import

datas = []

# Core dependencies that transformers always checks
deps_to_include = [
    "tqdm",
    "regex",
    "tokenizers",
    "safetensors",
    "requests",
    "filelock",
    "huggingface-hub",
    "numpy",
    "packaging",
    "pyyaml",
    "transformers",
    "scipy",
    "scikit-learn",
]

for dep in deps_to_include:
    try:
        datas += copy_metadata(dep)
    except Exception:
        # Silently skip if package not found
        pass
