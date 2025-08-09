# create_pbr.spec
# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

# Get the directory where this spec file is located
spec_root = os.path.dirname(os.path.abspath(SPEC))

# Collect metadata for packages that transformers checks
metadata_datas = []
deps_needing_metadata = [
    'tqdm', 'regex', 'tokenizers', 'safetensors', 'requests', 
    'filelock', 'huggingface-hub', 'numpy', 'packaging', 
    'pyyaml', 'transformers', 'scipy', 'scikit-learn'
]

for dep in deps_needing_metadata:
    try:
        metadata_datas += copy_metadata(dep)
    except:
        pass

a = Analysis(
    ['create_pbr.py'],
    pathex=[spec_root],
    binaries=[],
    datas=[
        # Include training_scripts directory for the models (lightweight Python files)
        ('training_scripts', 'training_scripts'),
        # NOTE: stored_weights is excluded to keep executable size manageable
        # Weights will be loaded from external directory at runtime
    ] + metadata_datas,  # Add the collected metadata
    hiddenimports=[
        # PyTorch and related
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torch.cuda',
        'torch.amp',
        'torch.amp.autocast_mode',
        'torchvision',
        'torchvision.transforms',
        'torchvision.transforms.functional',
        # Transformers - comprehensive list for Segformer
        'transformers',
        'transformers.models',
        'transformers.models.segformer',
        'transformers.models.segformer.modeling_segformer',
        'transformers.models.segformer.configuration_segformer',
        'transformers.models.segformer.image_processing_segformer',
        'transformers.models.auto',
        'transformers.models.auto.modeling_auto',
        'transformers.models.auto.auto_factory',
        'transformers.generation',
        'transformers.generation.utils',
        'transformers.generation.candidate_generator',
        'transformers.modeling_utils',
        'transformers.configuration_utils',
        'transformers.image_processing_utils',
        'transformers.activations',
        'transformers.modeling_outputs',
        'transformers.utils',
        'transformers.utils.constants',
        'transformers.utils.generic',
        'transformers.utils.hub',
        'transformers.utils.import_utils',
        'transformers.utils.args_doc',
        # SciPy and sklearn (needed by transformers)
        'scipy',
        'scipy.sparse',
        'scipy.spatial',
        'scipy.spatial.distance',
        'sklearn',
        'sklearn.metrics',
        # Kornia
        'kornia',
        'kornia.filters',
        # Other dependencies
        'numpy',
        'PIL',
        'PIL.Image',
        'tqdm',
        'regex',
        'pathlib',
        'subprocess',
        'argparse',
        'importlib.metadata',
        # Training scripts
        'training_scripts.class_materials',
        'training_scripts.segformer_6ch',
        'training_scripts.unet_models',
    ],
    hookspath=['hooks'],  # Use our custom hooks
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        # 'scipy',  # Removed - needed by transformers
        'pandas',
        'jupyter',
        'notebook',
        'IPython',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out None entries from binaries
a.binaries = [x for x in a.binaries if x is not None]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Build as onedir to avoid per-run unpacking (files are laid out in dist/create_pbr/)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='create_pbr',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Enable UPX compression to reduce file size
    upx_exclude=[],
    console=True,  # Keep console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # You can add an icon file path here
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='create_pbr',
)
