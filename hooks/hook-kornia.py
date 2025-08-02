# hook-kornia.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all kornia submodules (this should be sufficient)
hiddenimports = collect_submodules("kornia")

# Collect kornia data files
datas = collect_data_files("kornia", include_py_files=True)

# Note: The collect_submodules above should automatically include:
# - kornia.filters.median_blur
# - kornia.filters.gaussian_blur2d
# - kornia.filters.spatial_gradient
# If PyInstaller still misses modules, uncomment the lines below:

# hiddenimports += [
#     "kornia.filters",
#     "kornia.filters.median",
#     "kornia.filters.gaussian",
#     "kornia.filters.sobel",
# ]
