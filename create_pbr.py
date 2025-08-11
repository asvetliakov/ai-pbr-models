import argparse
import sys
import os
import torch
import json
import logging
import time
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from transformers.utils.constants import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from pathlib import Path
from subprocess import run, DEVNULL
from PIL import Image
import kornia as K
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
from training_scripts.class_materials import CLASS_LIST, CLASS_PALETTE
from training_scripts.segformer_6ch import create_segformer
from training_scripts.unet_models import UNetAlbedo, UNetSingleChannel

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# Resolve base directory correctly for both script and PyInstaller executable runs.
if getattr(sys, "frozen", False):
    # Running as a bundled executable (PyInstaller)
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    # Running from source
    BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = (BASE_DIR / "stored_weights").resolve()

parser = argparse.ArgumentParser(description="Create PBR from diffuse + normal")

parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help='Directory containing input images (must contain "textures" folder)',
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save output images. Will create PBR subdirectory automatically",
)

parser.add_argument(
    "--weights_dir",
    type=str,
    default=str(WEIGHTS_DIR),
    help="Directory containing model weights",
)

parser.add_argument(
    "--cpu",
    type=bool,
    default=False,
    help="Force CPU instead of GPU/CUDA for processing (may help if VRAM is low)",
)

parser.add_argument(
    "--format",
    type=str,
    default="dds",
    choices=["dds", "png"],
    help="Output format for the PBR maps (default: dds)",
)

parser.add_argument(
    "--separate_maps",
    type=bool,
    default=False,
    help="If true, saves each PBR map as a separate file. Forces --format to 'png'.",
)

parser.add_argument(
    "--textconv_path",
    type=str,
    default=str((Path(BASE_DIR) / "texconv.exe").resolve()),
    help="Path to the texconv executable",
)

parser.add_argument(
    "--max_tile_size",
    type=int,
    default=2048,
    help="Maximum tile size for processing (default: 2048). Use 1024 if you don't have enough VRAM.",
)

parser.add_argument(
    "--segformer_checkpoint",
    type=str,
    choices=["s4", "s4_alt"],
    help="Segformer checkpoint to use for segmentation",
    default="s4",
)

parser.add_argument(
    "--create_jsons",
    type=bool,
    default=True,
    help="If true, creates PGPatcher's JSON files.",
)


# --- Optional GUI when no CLI args are provided ---
def ensure_console():
    """On Windows, allocate a console if none is attached (useful for windowed builds)."""
    if os.name != "nt":
        return
    try:
        # If stdout is not a TTY, try to allocate a console
        if not sys.stdout or not sys.stdout.isatty():
            import ctypes

            ctypes.windll.kernel32.AllocConsole()
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
            try:
                sys.stdin = open("CONIN$", "r", encoding="utf-8", buffering=1)
            except Exception:
                pass
            globals()["_CONSOLE_WAS_ALLOCATED"] = True
    except Exception:
        pass


# Track if we created a console window
_CONSOLE_WAS_ALLOCATED = False
_ALWAYS_PAUSE_ON_EXIT = False  # set True in GUI mode or via env


def maybe_pause_on_exit(summary: str | None = None) -> None:
    """If we allocated a console, keep it open so user can read the logs."""
    try:
        if summary:
            logging.info(summary)
        must_pause = False
        if os.name == "nt":
            must_pause = bool(
                globals().get("_CONSOLE_WAS_ALLOCATED", False)
                or globals().get("_ALWAYS_PAUSE_ON_EXIT", False)
                or os.environ.get("PBR_ALWAYS_PAUSE") == "1"
            )
        if must_pause:
            print("\nPress Enter to exit...")
            try:
                input()
            except Exception:
                pass
    except Exception:
        pass


def make_desc(parent, text: str) -> tk.Label:
    """Create a small, grey, wrapped description label."""
    lbl = tk.Label(
        parent, text=text, justify=tk.LEFT, foreground="#666666", wraplength=520
    )
    try:
        base = tkfont.nametofont("TkDefaultFont").copy()
        base.configure(size=max(base.cget("size") - 1, 8))
        lbl.configure(font=base)
    except Exception:
        pass
    return lbl


def launch_gui_and_get_args() -> argparse.Namespace:
    """Launch a minimal Tkinter GUI to collect parameters and return an argparse.Namespace."""
    if tk is None:
        raise RuntimeError("Tkinter is not available on this system.")

    ensure_console()

    root = tk.Tk()
    root.title("Create PBR Maps")
    root.geometry("660x500")

    # Vars
    input_dir_v = tk.StringVar(value="")
    output_dir_v = tk.StringVar(value="")
    format_v = tk.StringVar(value="dds")
    separate_maps_v = tk.BooleanVar(value=False)
    create_jsons_v = tk.BooleanVar(value=True)
    tile_size_v = tk.IntVar(value=2048)
    segformer_v = tk.StringVar(value="s4")
    has_cuda = bool(torch.cuda.is_available())
    device_v = tk.StringVar(value=("cuda" if has_cuda else "cpu"))

    # Handlers
    def browse_input():
        d = filedialog.askdirectory(title="Select input directory")
        if d:
            input_dir_v.set(d)

    def browse_output():
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            output_dir_v.set(d)

    # Layout
    frm = ttk.Frame(root)
    frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    # Input dir
    lbl_input = ttk.Label(frm, text="Input directory:")
    lbl_input.grid(row=0, column=0, sticky="w")
    in_entry = ttk.Entry(frm, textvariable=input_dir_v, width=48)
    in_entry.grid(row=0, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_input).grid(
        row=0, column=2, sticky="e"
    )
    make_desc(
        frm, "Folder containing input textures (must contain 'textures' folder)"
    ).grid(row=1, column=1, columnspan=2, sticky="w", pady=(2, 8))

    # Output dir
    lbl_output = ttk.Label(frm, text="Output directory:")
    lbl_output.grid(row=2, column=0, sticky="w")
    out_entry = ttk.Entry(frm, textvariable=output_dir_v, width=48)
    out_entry.grid(row=2, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_output).grid(
        row=2, column=2, sticky="e"
    )
    make_desc(frm, "Where generated maps will be written.").grid(
        row=3, column=1, columnspan=2, sticky="w", pady=(2, 8)
    )

    # Format
    format_lbl = ttk.Label(frm, text="Format:")
    format_lbl.grid(row=4, column=0, sticky="w")
    fmt_box = ttk.Frame(frm)
    fmt_box.grid(row=4, column=1, sticky="w")
    ttk.Radiobutton(fmt_box, text="DDS", variable=format_v, value="dds").pack(
        side=tk.LEFT
    )
    ttk.Radiobutton(fmt_box, text="PNG", variable=format_v, value="png").pack(
        side=tk.LEFT
    )
    make_desc(
        frm,
        "DDS writes GPU-compressed textures via texconv; PNG writes standard .png files.",
    ).grid(row=5, column=1, columnspan=2, sticky="w", pady=(2, 8))

    # Separate maps
    sep_chk = ttk.Checkbutton(
        frm, text="Save separate maps (forces PNG)", variable=separate_maps_v
    )
    sep_chk.grid(row=6, column=1, sticky="w")
    make_desc(
        frm, "Saves Albedo, Roughness, Metallic, AO, and Parallax as individual images."
    ).grid(row=7, column=1, columnspan=2, sticky="w", pady=(2, 8))

    # Tile size
    tile_lbl = ttk.Label(frm, text="Tile size:")
    tile_lbl.grid(row=8, column=0, sticky="w")
    tile_box = ttk.Frame(frm)
    tile_box.grid(row=8, column=1, sticky="w")
    for sz in (1024, 2048):
        ttk.Radiobutton(tile_box, text=str(sz), variable=tile_size_v, value=sz).pack(
            side=tk.LEFT
        )
    make_desc(
        frm,
        "Processing tile size. If you don't have enough VRAM and have OOM(out of memory) errors OR your textures are in 2K, you can set this to 1024. Otherwise leave to 2048.",
    ).grid(row=9, column=1, columnspan=2, sticky="w", pady=(2, 8))

    # Create JSONs
    json_chk = ttk.Checkbutton(
        frm, text="Create PGPatcher JSON files", variable=create_jsons_v
    )
    json_chk.grid(row=10, column=1, sticky="w")
    make_desc(
        frm,
        "Generate JSON files compatible with PGPatcher inside the PBR output tree. Ignored when 'Separate maps' is enabled.",
    ).grid(row=11, column=1, columnspan=2, sticky="w", pady=(2, 8))

    # Keep JSON option disabled when separate maps is on
    def _sync_json_option(*_):
        if separate_maps_v.get() or format_v.get().lower() == "png":
            create_jsons_v.set(False)
            try:
                json_chk.state(["disabled"])  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            try:
                json_chk.state(["!disabled"])  # type: ignore[attr-defined]
            except Exception:
                pass

    try:
        separate_maps_v.trace_add("write", _sync_json_option)  # type: ignore[attr-defined]
        format_v.trace_add("write", _sync_json_option)  # type: ignore[attr-defined]
    except Exception:
        pass
    _sync_json_option()

    # Segformer checkpoint
    seg_lbl = ttk.Label(frm, text="Segmentation")
    seg_lbl.grid(row=12, column=0, sticky="w")
    seg_box = ttk.Frame(frm)
    seg_box.grid(row=12, column=1, sticky="w")
    for lab in ("s4", "s4_alt"):
        ttk.Radiobutton(seg_box, text=lab, variable=segformer_v, value=lab).pack(
            side=tk.LEFT
        )
    make_desc(
        frm,
        "EXPERIMENTAL. Choose the model variant for material segmentation. S4 is the most robust. If there is mislabeling metallic with other materials you can try s4_alt to see if it solves your issue.",
    ).grid(row=13, column=1, columnspan=2, sticky="w", pady=(2, 8))

    # Device selection
    dev_lbl = ttk.Label(frm, text="Compute Device")
    dev_lbl.grid(row=14, column=0, sticky="w")
    dev_box = ttk.Frame(frm)
    dev_box.grid(row=14, column=1, sticky="w")
    ttk.Radiobutton(dev_box, text="CPU", variable=device_v, value="cpu").pack(
        side=tk.LEFT
    )
    cuda_rb = ttk.Radiobutton(dev_box, text="CUDA", variable=device_v, value="cuda")
    if not has_cuda:
        cuda_rb.state(["disabled"])  # disable if CUDA not available
    cuda_rb.pack(side=tk.LEFT)
    make_desc(
        frm, "Use CUDA GPU when available for faster processing; otherwise use CPU."
    ).grid(row=15, column=1, columnspan=2, sticky="w", pady=(2, 8))

    # Actions
    btns = ttk.Frame(frm)
    btns.grid(row=16, column=0, columnspan=3, sticky="e", pady=(12, 0))

    result: dict[str, object] = {"ok": False, "ns": None}

    def on_run():
        if not input_dir_v.get() or not Path(input_dir_v.get()).exists():
            messagebox.showerror("Error", "Please select a valid input directory")
            return
        textures_dir = Path(input_dir_v.get()) / "textures"
        if not textures_dir.is_dir():
            messagebox.showerror(
                "Error",
                "Selected input directory must contain a 'textures' folder.",
            )
            return
        if not output_dir_v.get():
            messagebox.showerror("Error", "Please select an output directory")
            return
        # If separate maps, force PNG
        fmt = "png" if separate_maps_v.get() else format_v.get()
        result["ok"] = True
        result["ns"] = argparse.Namespace(
            input_dir=input_dir_v.get(),
            output_dir=output_dir_v.get(),
            weights_dir=str(WEIGHTS_DIR),
            cpu=(device_v.get() == "cpu"),
            format=fmt,
            separate_maps=bool(separate_maps_v.get()),
            create_jsons=bool(create_jsons_v.get()),
            textconv_path=str((Path(BASE_DIR) / "texconv.exe").resolve()),
            max_tile_size=int(tile_size_v.get()),
            segformer_checkpoint=segformer_v.get(),
        )
        root.destroy()

    def on_cancel():
        result.update(ok=False)
        root.destroy()

    ttk.Button(btns, text="Run", command=on_run).pack(side=tk.RIGHT, padx=6)
    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)

    # Grid config
    frm.columnconfigure(1, weight=1)

    root.mainloop()

    if not bool(result.get("ok")):
        sys.exit(0)
    return result["ns"]  # type: ignore[return-value]


if len(sys.argv) <= 1:
    # GUI mode
    _ALWAYS_PAUSE_ON_EXIT = True
    args = launch_gui_and_get_args()
else:
    # CLI mode
    args = parser.parse_args()
device = (
    torch.device("cuda")
    if not args.cpu and torch.cuda.is_available()
    else torch.device("cpu")
)
logging.info(f"Using device: {device}")

INPUT_DIR = Path(args.input_dir).resolve()
BASE_OUTPUT_DIR = Path(args.output_dir).resolve()
OUTPUT_DIR = BASE_OUTPUT_DIR / "textures" / "PBR"
TEXCONV_PATH = Path(args.textconv_path).resolve()
OUTPUT_PNG = args.format.lower() == "png" or args.separate_maps
SEPARATE_MAPS = args.separate_maps
MAX_TILE_SIZE = args.max_tile_size
WEIGHTS_DIR = Path(args.weights_dir).resolve()
SEGFORMER_STAGE = args.segformer_checkpoint
CREATE_JSONS = False if SEPARATE_MAPS else args.create_jsons

# Validate input structure and compute scan root
TEXTURES_DIR = INPUT_DIR / "textures"
if not TEXTURES_DIR.is_dir():
    logging.error(f"input_dir '{INPUT_DIR}' must contain a 'textures' folder.")
    maybe_pause_on_exit("Aborted due to invalid input directory structure.")
    sys.exit(2)

TEXCONV_ARGS_SRGB_PNG = [
    str(TEXCONV_PATH),
    "-nologo",
    "-f",
    "R8G8B8A8_UNORM_SRGB",
    "-ft",
    "png",
    "--srgb-in",
    "--srgb-out",
    "-y",
]

TEXCONV_ARGS_LINEAR_PNG = [
    str(TEXCONV_PATH),
    "-nologo",
    "-f",
    "R8G8B8A8_UNORM",
    "-ft",
    "png",
    "-y",
]
segformer_weights_path = WEIGHTS_DIR / f"{SEGFORMER_STAGE}/segformer/best_model.pt"
unet_albedo_weights_path = WEIGHTS_DIR / "a4/unet_albedo/best_model.pt"
unet_parallax_weights_path = WEIGHTS_DIR / "m3/unet_parallax/best_model.pt"
unet_ao_weights_path = WEIGHTS_DIR / "m3/unet_ao/best_model.pt"
unet_metallic_weights_path = WEIGHTS_DIR / "m3/unet_metallic/best_model.pt"
unet_roughness_weights_path = WEIGHTS_DIR / "m3/unet_roughness/best_model.pt"

try:
    logging.info("Loading model weights...")
    segformer_weights = torch.load(segformer_weights_path, map_location=device)
    unet_albedo_weights = torch.load(unet_albedo_weights_path, map_location=device)
    unet_parallax_weights = torch.load(unet_parallax_weights_path, map_location=device)
    unet_ao_weights = torch.load(unet_ao_weights_path, map_location=device)
    unet_metallic_weights = torch.load(unet_metallic_weights_path, map_location=device)
    unet_roughness_weights = torch.load(
        unet_roughness_weights_path, map_location=device
    )
except Exception:
    logging.exception(
        "Failed to load model weights. Check --weights_dir and file presence."
    )
    maybe_pause_on_exit("Failed to load weights.")
    sys.exit(3)

segformer = create_segformer(
    num_labels=len(CLASS_LIST),
    device=device,
    lora=False,
    frozen=True,
)
segformer.load_state_dict(segformer_weights)
segformer.eval()

unet_albedo = UNetAlbedo(in_ch=6, cond_ch=512).to(device)
unet_albedo.load_state_dict(unet_albedo_weights)
for param in unet_albedo.parameters():
    param.requires_grad = False
unet_albedo.eval()

unet_parallax = UNetSingleChannel(in_ch=5, cond_ch=512).to(device)
unet_parallax.load_state_dict(unet_parallax_weights)
for param in unet_parallax.parameters():
    param.requires_grad = False
unet_parallax.eval()

unet_ao = UNetSingleChannel(in_ch=5, cond_ch=512).to(device)
unet_ao.load_state_dict(unet_ao_weights)
for param in unet_ao.parameters():
    param.requires_grad = False
unet_ao.eval()

unet_metallic = UNetSingleChannel(in_ch=6 + len(CLASS_LIST), cond_ch=512).to(device)
unet_metallic.load_state_dict(unet_metallic_weights)
for param in unet_metallic.parameters():
    param.requires_grad = False
unet_metallic.eval()

unet_roughness = UNetSingleChannel(in_ch=6 + len(CLASS_LIST), cond_ch=512).to(device)
unet_roughness.load_state_dict(unet_roughness_weights)
for param in unet_roughness.parameters():
    param.requires_grad = False
unet_roughness.eval()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_normal_map(normal: Image.Image) -> Image.Image:
    """
    Take a PIL normal map (RGB, 0–255), renormalize each pixel vector to length 1,
    and return a new PIL.Image in the same mode/range.
    (Does NOT resize or flip any channels.)
    """
    # to (H, W, 3) floats in [0,1]
    arr = np.array(normal, dtype=np.float32) / 255.0

    # map to [-1,1]
    vec = arr * 2.0 - 1.0

    # compute per-pixel length and normalize
    length = np.linalg.norm(vec, axis=2, keepdims=True)
    vec = vec / (length + 1e-6)

    # map back to [0,1]
    out = (vec + 1.0) * 0.5

    # to uint8 PIL
    out8 = (out * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out8, mode="RGB")


def mean_curvature_map(normals: torch.Tensor) -> torch.Tensor:
    """
    normals: (B,3,H,W) in [-1,1]
    returns: (B,1,H,W) curvature map in [-1,1]
    """

    # 1) Median-filter to kill salt-and-pepper spikes
    #    (kernel size 3x3 on each channel)
    normals = K.filters.median_blur(normals, (3, 3))

    # 2) Compute Sobel gradients → shape (B,2,3,H,W)
    grad = K.filters.spatial_gradient(normals, mode="sobel", order=1)
    grad_x, grad_y = grad[:, 0], grad[:, 1]  # each (B,3,H,W)

    # 3) Approximate mean curvature = 0.5*(∂x n_x + ∂y n_y)
    dnx_dx = grad_x[:, 0:1]  # (B,1,H,W)
    dny_dy = grad_y[:, 1:2]  # (B,1,H,W)
    curv = 0.5 * (dnx_dx + dny_dy)

    # 4) Absolute value → all positive
    curv = curv.abs()  # (B,1,H,W)

    # 5) Percentile-normalize per-image (99th percentile)
    #    Flatten spatial dims, compute per-sample 99% value
    flat = curv.view(curv.shape[0], -1)
    p99 = torch.quantile(flat, 0.99, dim=1)  # (B,)
    p99 = p99.view(-1, 1, 1, 1).clamp(min=1e-6)  # avoid zero
    curv = (curv / p99).clamp(0.0, 1.0)

    # Remap to -1 to 1 range
    curv = (curv - 0.5) * 2.0

    return curv


def poisson_coarse_from_normal(normal: torch.Tensor) -> torch.Tensor:
    """
    normal: (B,3,H,W) normal in [-1,1]
    Returns (B,1,H,W) coarse height in [-1,1]
    """

    # Convert to torch tensor for Kornia processing
    # n_torch = torch.from_numpy(n).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)

    # Pre-smooth normals to avoid noise artifacts using Kornia
    normal = K.filters.median_blur(normal, (3, 3))

    # Convert numpy
    n_arr = normal.squeeze(0).permute(1, 2, 0).numpy()

    # print(f"Processing {path}...")
    nz = np.clip(n_arr[..., 2], 1e-3, 1.0)  # avoid divide-by-zero
    gx = n_arr[..., 0] / nz  # slope in +x (cols → right)
    gy = n_arr[..., 1] / nz  # slope in +y (rows → down)

    clip_val = 10.0
    gx = np.clip(gx, -clip_val, +clip_val)
    gy = np.clip(gy, -clip_val, +clip_val)

    H, W = gx.shape
    fx = np.fft.fftfreq(W)[None, :]  # frequency grids
    fy = np.fft.fftfreq(H)[:, None]
    denom = (2 * np.pi) ** 2 * (fx**2 + fy**2)
    denom[0, 0] = np.inf  # zero-mean solution

    div = (1j * 2 * np.pi * fx) * np.fft.fft2(gx) + (1j * 2 * np.pi * fy) * np.fft.fft2(
        gy
    )

    h = np.real(np.fft.ifft2(div / denom))

    span = h.max() - h.min()
    h = (h - h.min()) / span  # [0,1]

    # Avoid outliers
    p1, p99 = np.percentile(h, [1, 99])
    h_clamped = np.clip(h, p1, p99)
    h = (h_clamped - p1) / (p99 - p1 + 1e-6)

    poisson = h.astype(np.float32)

    # Convert to torch tensor for Kornia Gaussian blur
    poisson_torch = (
        torch.from_numpy(poisson).unsqueeze(0).unsqueeze(0).float()
    )  # (1, 1, H, W)

    # Apply Gaussian blur using Kornia (kernel_size=15, sigma=5)
    blur_torch = K.filters.gaussian_blur2d(poisson_torch, (15, 15), (5.0, 5.0))

    # Normalize to [-1, 1]
    blur_torch = (blur_torch - 0.5) * 2.0

    return blur_torch


def predict_albedo(diffuse_img: Image.Image, normal_img: Image.Image) -> Image.Image:
    """
    Predict albedo from diffuse and normal maps.
    """
    # Store alpha channel from diffuse for later use
    diffuse_alpha = diffuse_img.split()[-1]
    # Check if it's true alpha or just white (textconv puts white if input didn't have alpha)
    if diffuse_alpha.getextrema() == (255, 255):
        diffuse_alpha = None

    # Convert to RGB
    diffuse_img = diffuse_img.convert("RGB")
    normal_img = normal_img.convert("RGB")

    # Normalize diffuse & normal image to be the same resolution, use diffuse as base
    W, H = diffuse_img.size
    if normal_img.size != (W, H):
        logging.info(f"Resizing normal from {normal_img.size} to {(W, H)}")
        normal_img = normal_img.resize((W, H), Image.Resampling.BILINEAR)
        normal_img = normalize_normal_map(normal_img)

    # Convert to tensors and normalize
    diffuse = TF.to_tensor(diffuse_img).unsqueeze(0)  # (1, 3, H, W)
    diffuse = TF.normalize(
        diffuse, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    normal = TF.to_tensor(normal_img).unsqueeze(0)  # (1, 3, H, W)
    normal = TF.normalize(
        normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    W, H = diffuse_img.size
    tile_size = min(H, W)
    if tile_size > MAX_TILE_SIZE:
        tile_size = MAX_TILE_SIZE

    overlap = 0
    xs = [0]
    ys = [0]

    # Use tiling if image is large than tile size or not square
    if W != H or tile_size < min(H, W):
        # 1024 -> 128px , 2048 -> 256px
        overlap = int(tile_size / 8)
        stride = tile_size - overlap
        # compute start indices along each axis
        xs = list(range(0, W - tile_size + 1, stride))
        ys = list(range(0, H - tile_size + 1, stride))
        # ensure final tile reaches edge
        if xs[-1] + tile_size < W:
            xs.append(W - tile_size)
        if ys[-1] + tile_size < H:
            ys.append(H - tile_size)

    # Create output tensors on CPU to save GPU memory, move tiles to GPU as needed
    albedo_output = torch.zeros(1, 3, H, W)  # (B, C, H, W)
    weight_albedo = torch.zeros_like(albedo_output)  # (B, C, H, W)

    # Create blending mask (raised cosine window) - avoid zero weights at edges
    eps = 1e-3
    ramp = torch.linspace(eps, 1.0, overlap)

    # Precompute nothing here; we'll build per-tile windows that only taper where a neighbor exists

    num_tiles = len(xs) * len(ys)
    if num_tiles > 1:
        logging.info(
            f"Albedo tiling: tile_size={tile_size}, overlap={overlap}, tiles={num_tiles}"
        )

    for y in ys:
        for x in xs:
            # Use autocast for mixed precision to reduce memory usage
            with torch.no_grad(), autocast(
                enabled=device.type == "cuda",
                device_type=device.type,
            ):
                tile_image = TF.crop(diffuse, y, x, tile_size, tile_size).to(device)
                tile_normal = TF.crop(normal, y, x, tile_size, tile_size).to(device)

                # Get predicted albedo first without segformer
                albedo_input = torch.cat((tile_image, tile_normal), dim=1)
                predicted_albedo = unet_albedo(albedo_input, None)

                # Using previous dirty albedo get segformer features
                predicted_albedo = TF.normalize(
                    predicted_albedo,
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                )

                # Get segformer mask/probabilities
                segformer_input = torch.cat((predicted_albedo, tile_normal), dim=1)
                setformer_output = segformer(segformer_input, output_hidden_states=True)
                seg_feats = setformer_output.hidden_states[-1]  # (B, C, H/4, W/4)

                # Run unet again with segformer features to get better albedo
                predicted_albedo = unet_albedo(albedo_input, seg_feats)
                # Move predictions back to CPU and convert to float32 to save GPU memory
                predicted_albedo = predicted_albedo.cpu().float()

                if overlap > 0:
                    # Build per-tile 1D windows: only taper if a neighbor exists on that side
                    wx = torch.ones(tile_size)
                    wy = torch.ones(tile_size)
                    # Horizontal neighbors
                    if x > 0:
                        wx[:overlap] = ramp
                    if x + tile_size < W:
                        wx[-overlap:] = torch.flip(ramp, dims=[0])
                    # Vertical neighbors
                    if y > 0:
                        wy[:overlap] = ramp
                    if y + tile_size < H:
                        wy[-overlap:] = torch.flip(ramp, dims=[0])
                    # Guard against zeros
                    wx = wx.clamp_min(eps)
                    wy = wy.clamp_min(eps)
                    w2d_cpu = (
                        wy[None, None, :, None] * wx[None, None, None, :]
                    ).float()
                    albedo_output[:, :, y : y + tile_size, x : x + tile_size] += (
                        predicted_albedo * w2d_cpu
                    )
                    weight_albedo[:, :, y : y + tile_size, x : x + tile_size] += w2d_cpu
                else:
                    albedo_output[
                        :, :, y : y + tile_size, x : x + tile_size
                    ] += predicted_albedo
                    weight_albedo[:, :, y : y + tile_size, x : x + tile_size] += 1.0

                # Explicit memory cleanup after each tile
                del (
                    tile_image,
                    tile_normal,
                    albedo_input,
                    predicted_albedo,
                    segformer_input,
                    setformer_output,
                    seg_feats,
                )
                torch.cuda.empty_cache() if device.type == "cuda" else None

    # Normalize output
    if overlap > 0:
        albedo_output = albedo_output / weight_albedo.clamp(min=1e-6)

    albedo_image = TF.to_pil_image(albedo_output.squeeze(0).clamp(0, 1))
    # Copy alpha channel from diffuse if it exists
    if diffuse_alpha is not None:
        albedo_image.putalpha(diffuse_alpha)

    return albedo_image


def predirect_pbr_maps(
    albedo_img: Image.Image, normal_img: Image.Image
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """
    Predict PBR maps from albedo and normal images.
    """
    # Drop alpha if exists
    albedo_img = albedo_img.convert("RGB")
    normal_img = normal_img.convert("RGB")

    original_size = albedo_img.size
    downsample_factor = 1

    if albedo_img.height != albedo_img.width:
        padding = (
            0,
            0,
            # Add right padding to make it square
            (
                albedo_img.height - albedo_img.width
                if albedo_img.height > albedo_img.width
                else 0
            ),
            # Add bottom padding to make it square
            (
                albedo_img.width - albedo_img.height
                if albedo_img.width > albedo_img.height
                else 0
            ),
        )
        logging.info(
            f"Padding to square: pad_right={padding[2]}, pad_bottom={padding[3]}"
        )
        albedo_img = TF.pad(albedo_img, padding, fill=0)  # type: ignore
        normal_img = TF.pad(normal_img, padding, fill=0)  # type: ignore

    # PBR maps have / 2 resolution so downscale if albedo is larger than 1024
    if min(albedo_img.size) > 1024:
        albedo_img = albedo_img.resize(
            (albedo_img.width // 2, albedo_img.height // 2), Image.Resampling.LANCZOS
        )
        downsample_factor = 2
        logging.info("Downscaling PBR base to 1/2 (min side > 1024)")

    # If original input was 8K it's still 4K now. Too much so downscale to max tile size
    # Note we don't want to use tiling similar to albedo to produce roughness/metallic it in always all cases produces tiled roughness/metallic where one tile is different from another
    if min(albedo_img.size) > MAX_TILE_SIZE:
        factor = albedo_img.width // MAX_TILE_SIZE
        albedo_img = albedo_img.resize(
            (albedo_img.width // factor, albedo_img.height // factor),
            Image.Resampling.LANCZOS,
        )
        downsample_factor *= factor
        logging.info(
            f"Downscaling to respect MAX_TILE_SIZE={MAX_TILE_SIZE} (factor={factor})"
        )

    if albedo_img.size != normal_img.size:
        logging.info(f"Resizing normal to {albedo_img.size} to match albedo")
        normal_img = normal_img.resize(
            (albedo_img.width, albedo_img.height), Image.Resampling.BILINEAR
        )
        normal_img = normalize_normal_map(normal_img)

    # Convert to tensors and normalize
    albedo = TF.to_tensor(albedo_img).unsqueeze(0)  # (1, 3, H, W)
    albedo = TF.normalize(
        albedo, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )
    albedo_segformer = TF.to_tensor(albedo_img).unsqueeze(0)  # (1, 3, H, W)
    albedo_segformer = TF.normalize(
        albedo_segformer, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )

    normal = TF.to_tensor(normal_img).unsqueeze(0)  # (1, 3, H, W)
    normal = TF.normalize(
        normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    # Compute poisson-coarse and mean curvature maps
    poisson_coarse = poisson_coarse_from_normal(normal)
    mean_curvature = mean_curvature_map(normal)

    logging.info("Running segmentation + UNets for PBR maps...")
    # Use autocast for mixed precision to reduce memory usage
    with torch.no_grad(), autocast(
        enabled=device.type == "cuda",
        device_type=device.type,
    ):
        albedo = albedo.to(device)
        albedo_segformer = albedo_segformer.to(device)
        normal = normal.to(device)
        poisson_coarse = poisson_coarse.to(device)
        mean_curvature = mean_curvature.to(device)

        # Get segformer mask/probabilities
        segformer_input = torch.cat((albedo_segformer, normal), dim=1)
        setformer_output = segformer(segformer_input, output_hidden_states=True)
        segformer_pred = setformer_output.logits
        seg_feats = setformer_output.hidden_states[-1]  # (B, C, H/4, W/4)

        segformer_pred: torch.Tensor = torch.nn.functional.interpolate(
            segformer_pred,
            size=albedo_segformer.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        seg_probs = F.softmax(segformer_pred, dim=1)
        # Hard confidence gating to avoid ambiguous signals
        max_probs, max_indices = torch.max(seg_probs, dim=1, keepdim=True)
        one_hot_mask = torch.zeros_like(seg_probs)
        one_hot_mask.scatter_(1, max_indices, 1)

        # Use lower confidence threshold but add explicit non-metal suppression
        confidence_thresh = 0.5
        high_conf_mask = (max_probs > confidence_thresh).float()
        final_mask = one_hot_mask * high_conf_mask

        # Run unet for parallax
        parallax_ao_input = torch.cat((normal, mean_curvature, poisson_coarse), dim=1)
        predicted_parallax = unet_parallax(parallax_ao_input, seg_feats)

        # Run unet for AO
        predicted_ao = unet_ao(parallax_ao_input, seg_feats)

        # Run unet for metallic
        metallic_roughness_input = torch.cat(
            (albedo, normal, final_mask),
            dim=1,
        )
        predicted_metallic = unet_metallic(metallic_roughness_input, seg_feats)

        # Run unet for roughness
        predicted_roughness = unet_roughness(metallic_roughness_input, seg_feats)

        # Move predictions back to CPU and convert to float32 to save GPU memory
        predicted_parallax = predicted_parallax.cpu().float()
        predicted_ao = predicted_ao.cpu().float()
        predicted_metallic = predicted_metallic.cpu().float()
        predicted_roughness = predicted_roughness.cpu().float()
        # mask_visual = torch.argmax(final_mask, dim=1).squeeze(0).cpu()

        # Explicit memory cleanup after each tile
        del (
            segformer_input,
            setformer_output,
            segformer_pred,
            seg_feats,
            seg_probs,
            max_probs,
            max_indices,
            one_hot_mask,
            high_conf_mask,
            final_mask,
            parallax_ao_input,
            metallic_roughness_input,
            # mask_visual,
        )
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # If padded crop to original size
    final_size = (
        original_size[0] // downsample_factor,
        original_size[1] // downsample_factor,
    )

    parallax_image: Image.Image = TF.to_pil_image(
        torch.sigmoid(predicted_parallax).squeeze(0).clamp(0, 1)
    )
    parallax_image = parallax_image.crop((0, 0, final_size[0], final_size[1]))

    ao_image: Image.Image = TF.to_pil_image(
        torch.sigmoid(predicted_ao).squeeze(0).clamp(0, 1)
    )
    ao_image = ao_image.crop((0, 0, final_size[0], final_size[1]))

    metallic_image: Image.Image = TF.to_pil_image(
        torch.sigmoid(predicted_metallic).squeeze(0).clamp(0, 1)
    )
    metallic_image = metallic_image.crop((0, 0, final_size[0], final_size[1]))

    roughness_image: Image.Image = TF.to_pil_image(
        torch.sigmoid(predicted_roughness).squeeze(0).clamp(0, 1)
    )
    roughness_image = roughness_image.crop((0, 0, final_size[0], final_size[1]))

    logging.info("PBR inference complete")
    return (
        parallax_image,
        ao_image,
        metallic_image,
        roughness_image,
    )


# Find and process normal + diffuse pairs inside 'textures'
file_list = sorted(TEXTURES_DIR.glob("**/*_n.dds"))
total_found = len(file_list)
processed_ok = 0
processed_fail = 0
skipped_missing_diffuse = 0

logging.info(f"Scanning '{TEXTURES_DIR}' for normal maps: found {total_found}")

for normal_path in file_list:
    diffuse_path = normal_path.with_name(normal_path.name.replace("_n.dds", "_d.dds"))
    if not diffuse_path.exists():
        diffuse_path = normal_path.with_name(normal_path.name.replace("_n.dds", ".dds"))

    if not diffuse_path.exists():
        logging.warning(f"Skipping {normal_path} - diffuse not found (_d.dds or .dds)")
        skipped_missing_diffuse += 1
        continue

    logging.info("Processing pair:")
    logging.info(f"  Normal:  {normal_path}")
    logging.info(f"  Diffuse: {diffuse_path}")

    input_relative_path = normal_path.relative_to(TEXTURES_DIR)
    output_dir = OUTPUT_DIR / input_relative_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call textconv for diffuse
    diffuse_args = TEXCONV_ARGS_SRGB_PNG + ["-o", str(output_dir), str(diffuse_path)]
    normal_args = TEXCONV_ARGS_LINEAR_PNG + ["-o", str(output_dir), str(normal_path)]

    try:
        run(diffuse_args, check=True, stdout=DEVNULL, stderr=DEVNULL)
        run(normal_args, check=True, stdout=DEVNULL, stderr=DEVNULL)
    except Exception:
        logging.exception("texconv failed; skipping this pair")
        processed_fail += 1
        continue

    basename = normal_path.stem.replace("_n", "")
    diffuse_basename = diffuse_path.stem
    diffuse_png = output_dir / (diffuse_path.stem + ".png")
    normal_png = output_dir / (normal_path.stem + ".png")

    diffuse_img = Image.open(diffuse_png)
    has_alpha = diffuse_img.mode == "RGBA"
    # Check if alpha channel exists and is not just white
    if has_alpha:
        alpha_channel = diffuse_img.split()[-1]
        if alpha_channel.getextrema() == (255, 255):
            has_alpha = False

    # Drop alpha (specular if i remember correctly?) from normal
    normal_img = Image.open(normal_png).convert("RGB")

    # Remove PNGs
    diffuse_png.unlink(missing_ok=True)
    normal_png.unlink(missing_ok=True)

    try:
        t0 = time.perf_counter()
        logging.info("Generating Albedo...")
        albedo_img = predict_albedo(diffuse_img, normal_img)
        t1 = time.perf_counter()
        logging.info(f"Albedo generated in {t1 - t0:.2f}s")

        logging.info("Generating PBR maps...")
        parallax_img, ao_img, metallic_img, roughness_img = predirect_pbr_maps(
            albedo_img, normal_img
        )
        t2 = time.perf_counter()
        logging.info(f"PBR maps generated in {t2 - t1:.2f}s")
    except Exception:
        logging.exception("Model inference failed; skipping this pair")
        processed_fail += 1
        continue

    if SEPARATE_MAPS:
        albedo_png = output_dir / (basename + "_albedo.png")
        parallax_png = output_dir / (basename + "_p.png")
        ao_png = output_dir / (basename + "_ao.png")
        metallic_png = output_dir / (basename + "_metallic.png")
        roughness_png = output_dir / (basename + "_roughness.png")

        albedo_img.save(albedo_png)
        parallax_img.save(parallax_png)
        ao_img.save(ao_png)
        metallic_img.save(metallic_png)
        roughness_img.save(roughness_png)
        logging.info(
            f"Saved: {albedo_png.name}, {parallax_png.name}, {ao_png.name}, {metallic_png.name}, {roughness_png.name}"
        )
    else:
        albedo_png = output_dir / (diffuse_basename + ".png")
        rmaos_png = output_dir / (basename + "_rmaos.png")
        parallax_png = output_dir / (basename + "_p.png")

        # Create alpha/specular channel with all values set to 255
        specular = Image.new("L", roughness_img.size, 255)

        rmaos_image = Image.merge(
            "RGBA",
            (
                roughness_img.convert("L"),
                metallic_img.convert("L"),
                ao_img.convert("L"),
                specular,
            ),
        )
        albedo_img.save(albedo_png)
        rmaos_image.save(rmaos_png)
        parallax_img.save(parallax_png)
        logging.info(f"Saved: {albedo_png.name}, {rmaos_png.name}, {parallax_png.name}")

    # Re-save normal map since we need to drop alpha channel here
    normal_img.save(normal_png)

    if not OUTPUT_PNG:
        albedo_dds = albedo_png.with_suffix(".dds")
        rmaos_dds = rmaos_png.with_suffix(".dds")
        parallax_dds = parallax_png.with_suffix(".dds")
        normal_dds = normal_png.with_suffix(".dds")

        albedo_args = [
            str(TEXCONV_PATH),
            "-nologo",
            "-f",
            "BC7_UNORM_SRGB" if has_alpha else "BC1_UNORM_SRGB",
            "-ft",
            "dds",
            "--srgb-in",
            "-y",
            "-m",
            "0",
            "-o",
            str(output_dir),
        ]
        if has_alpha:
            albedo_args.append("--separate-alpha")
        albedo_args.append(str(albedo_png))

        run(albedo_args, check=True, stdout=DEVNULL, stderr=DEVNULL)
        albedo_png.unlink(missing_ok=True)

        rmaos_args = [
            str(TEXCONV_PATH),
            "-nologo",
            "-f",
            "BC1_UNORM",
            "-ft",
            "dds",
            "-y",
            "-m",
            "0",
            "-o",
            str(output_dir),
            str(rmaos_png),
        ]
        run(rmaos_args, check=True, stdout=DEVNULL, stderr=DEVNULL)
        rmaos_png.unlink(missing_ok=True)

        parallax_args = [
            str(TEXCONV_PATH),
            "-nologo",
            "-f",
            "BC4_UNORM",
            "-ft",
            "dds",
            "-y",
            "-m",
            "0",
            "-o",
            str(output_dir),
            str(parallax_png),
        ]
        run(parallax_args, check=True, stdout=DEVNULL, stderr=DEVNULL)
        parallax_png.unlink(missing_ok=True)

        normal_args = [
            str(TEXCONV_PATH),
            "-nologo",
            "-f",
            "BC7_UNORM",
            "-ft",
            "dds",
            "-y",
            "-m",
            "0",
            "-o",
            str(output_dir),
            str(normal_png),
        ]
        run(normal_args, check=True, stdout=DEVNULL, stderr=DEVNULL)
        normal_png.unlink(missing_ok=True)

    if CREATE_JSONS:
        json_base_dir = BASE_OUTPUT_DIR / "PBRNifPatcher"
        json_base_dir.mkdir(parents=True, exist_ok=True)
        json_dir = json_base_dir / input_relative_path.parent
        json_dir.mkdir(parents=True, exist_ok=True)
        json_file = json_dir / (diffuse_basename + ".json")

        with open(json_file, "w") as f:
            json.dump(
                [
                    {
                        "texture": str(input_relative_path.parent / diffuse_basename),
                        "emissive": False,
                        "parallax": True,
                        "subsurface_foliage": False,
                        "subsurface": False,
                        "specular_level": 0.04,
                        "subsurface_color": [1, 1, 1],
                        "roughness_scale": 1,
                        "subsurface_opacity": 1,
                        "smooth_angle": 75,
                        "displacement_scale": 1,
                    }
                ],
                f,
                indent=4,
            )
        logging.info(f"Wrote JSON: {json_file}")

    processed_ok += 1

# Print summary and keep console open if we allocated it
summary = (
    f"\nCompleted. Found: {total_found}, OK: {processed_ok}, "
    f"Failed: {processed_fail}, Skipped (no diffuse): {skipped_missing_diffuse}.\n"
    f"Output root: {OUTPUT_DIR}"
)
maybe_pause_on_exit(summary)
