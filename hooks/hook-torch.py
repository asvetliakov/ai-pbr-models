# hook-torch.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all torch submodules (this should be sufficient for most cases)
hiddenimports = collect_submodules("torch")

# Collect torch data files
datas = collect_data_files("torch", include_py_files=True)

# Add only the torch modules that are actually used in your script
hiddenimports += [
    "torch.nn.functional",  # F.softmax, F.interpolate
    "torch.cuda",  # cuda.is_available, cuda.empty_cache
    "torch.amp",  # autocast functionality
    "torch.amp.autocast_mode",  # autocast class
]

# Note: The collect_submodules above should automatically include most torch functionality.
# The specific imports above are just to ensure critical modules are included.
# If you get import errors, you can uncomment the modules below:

# hiddenimports += [
#     "torch.nn.modules.conv",        # For CNN layers in your models
#     "torch.nn.modules.linear",      # For linear/dense layers
#     "torch.nn.modules.batchnorm",   # For batch normalization
#     "torch.nn.modules.activation",  # For activation functions
# ]
