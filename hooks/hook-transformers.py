# hook-transformers.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all transformers submodules
hiddenimports = collect_submodules("transformers")

# Collect transformers data files
datas = collect_data_files("transformers", include_py_files=True)

# Add specific transformers modules for Segformer
hiddenimports += [
    "transformers.models.segformer",
    "transformers.models.segformer.modeling_segformer",
    "transformers.models.segformer.configuration_segformer",
    "transformers.models.segformer.image_processing_segformer",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.image_processing_utils",
    "transformers.utils",
    "transformers.utils.constants",
    "transformers.utils.generic",
    "transformers.utils.hub",
    "transformers.utils.import_utils",
    "transformers.activations",
    "transformers.modeling_outputs",
]
