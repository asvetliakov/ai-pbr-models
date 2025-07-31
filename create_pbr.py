import argparse
import torch
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
from training_scripts.class_materials import CLASS_LIST, CLASS_PALETTE
from training_scripts.segformer_6ch import create_segformer
from training_scripts.unet_models import UNetAlbedo, UNetSingleChannel

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = (BASE_DIR / "stored_weights").resolve()

parser = argparse.ArgumentParser(description="Create PBR from diffuse + normal")

parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Directory containing input images",
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save output images",
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
    default="texconv.exe",
    help="Path to the texconv executable",
)

parser.add_argument(
    "--max_tile_size",
    type=int,
    default=2048,
    help="Maximum tile size for processing (default: 2048). Use 1024 if you don't have enough VRAM.",
)

args = parser.parse_args()
device = (
    torch.device("cuda")
    if not args.cpu and torch.cuda.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")

INPUT_DIR = Path(args.input_dir).resolve()
BASE_OUTPUT_DIR = Path(args.output_dir).resolve()
TEXCONV_PATH = Path(args.textconv_path).resolve()
OUTPUT_PNG = args.format.lower() == "png" or args.separate_maps
SEPARATE_MAPS = args.separate_maps
MAX_TILE_SIZE = args.max_tile_size

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
segformer_weights_path = WEIGHTS_DIR / "s3/segformer/best_model.pt"
unet_albedo_weights_path = WEIGHTS_DIR / "a4/unet_albedo/best_model.pt"
unet_parallax_weights_path = WEIGHTS_DIR / "m3/unet_parallax/best_model.pt"
unet_ao_weights_path = WEIGHTS_DIR / "m3/unet_ao/best_model.pt"
unet_metallic_weights_path = WEIGHTS_DIR / "m3/unet_metallic/best_model.pt"
unet_roughness_weights_path = WEIGHTS_DIR / "m3/unet_roughness/best_model.pt"

segformer_weights = torch.load(segformer_weights_path, map_location=device)
unet_albedo_weights = torch.load(unet_albedo_weights_path, map_location=device)
unet_parallax_weights = torch.load(unet_parallax_weights_path, map_location=device)
unet_ao_weights = torch.load(unet_ao_weights_path, map_location=device)
unet_metallic_weights = torch.load(unet_metallic_weights_path, map_location=device)
unet_roughness_weights = torch.load(unet_roughness_weights_path, map_location=device)

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

BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
        print(f"Resizing normal from {normal_img.size} to {(W, H)}")
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

    # Create blending mask (raised cosine window) - keep on CPU initially
    ramp = torch.linspace(0, 1, overlap)
    window1d = torch.ones(tile_size)

    if overlap > 0:
        window1d[:overlap] = ramp
        window1d[-overlap:] = ramp.flip(0)

    # Make 2d blending mask
    w2d = window1d[None, None, :, None] * window1d[None, None, None, :]

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
                    w2d_cpu = w2d.float()
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

    albedo_min = min(albedo_img.size)
    # PBR maps have / 2 resolution so downscale if albedo is larger than 1024
    if albedo_min > 1024:
        albedo_img = albedo_img.resize(
            (albedo_img.width // 2, albedo_img.height // 2), Image.Resampling.LANCZOS
        )

    if albedo_img.size != normal_img.size:
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

    W, H = albedo_img.size
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
    mask_output = torch.zeros(H, W)
    parallax_output = torch.zeros(1, 1, H, W)  # (B, C, H, W)
    ao_output = torch.zeros(1, 1, H, W)  # (B, C, H, W)
    metallic_output = torch.zeros(1, 1, H, W)  # (B, C, H, W)
    roughness_output = torch.zeros(1, 1, H, W)  # (B, C, H, W)
    weight_1ch = torch.zeros_like(parallax_output)  # (B, C, H, W)

    # Create blending mask (raised cosine window) - keep on CPU initially
    ramp = torch.linspace(0, 1, overlap)
    window1d = torch.ones(tile_size)

    if overlap > 0:
        window1d[:overlap] = ramp
        window1d[-overlap:] = ramp.flip(0)

    # Make 2d blending mask
    w2d = window1d[None, None, :, None] * window1d[None, None, None, :]

    for y in ys:
        for x in xs:
            # Use autocast for mixed precision to reduce memory usage
            with torch.no_grad(), autocast(
                enabled=device.type == "cuda",
                device_type=device.type,
            ):
                tile_albedo = TF.crop(albedo, y, x, tile_size, tile_size).to(device)
                tile_albedo_segformer = TF.crop(
                    albedo_segformer, y, x, tile_size, tile_size
                ).to(device)
                tile_normal = TF.crop(normal, y, x, tile_size, tile_size).to(device)
                tile_poisson = TF.crop(poisson_coarse, y, x, tile_size, tile_size).to(
                    device
                )
                tile_mean_curvature = TF.crop(
                    mean_curvature, y, x, tile_size, tile_size
                ).to(device)

                # Get segformer mask/probabilities
                segformer_input = torch.cat((tile_albedo_segformer, tile_normal), dim=1)
                setformer_output = segformer(segformer_input, output_hidden_states=True)
                segformer_pred = setformer_output.logits
                seg_feats = setformer_output.hidden_states[-1]  # (B, C, H/4, W/4)

                segformer_pred: torch.Tensor = torch.nn.functional.interpolate(
                    segformer_pred,
                    size=tile_albedo_segformer.shape[2:],
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
                parallax_ao_input = torch.cat(
                    (tile_normal, tile_mean_curvature, tile_poisson), dim=1
                )
                predicted_parallax = unet_parallax(parallax_ao_input, seg_feats)

                # Run unet for AO
                predicted_ao = unet_ao(parallax_ao_input, seg_feats)

                # Run unet for metallic
                metallic_roughness_input = torch.cat(
                    (tile_albedo, tile_normal, final_mask),
                    dim=1,
                )
                predicted_metallic = unet_metallic(metallic_roughness_input, seg_feats)

                # Run unet for roughness
                predicted_roughness = unet_roughness(
                    metallic_roughness_input, seg_feats
                )

                # Move predictions back to CPU and convert to float32 to save GPU memory
                predicted_parallax = predicted_parallax.cpu().float()
                predicted_ao = predicted_ao.cpu().float()
                predicted_metallic = predicted_metallic.cpu().float()
                predicted_roughness = predicted_roughness.cpu().float()
                mask_visual = torch.argmax(final_mask, dim=1).squeeze(0).cpu()

                if overlap > 0:
                    w2d_cpu = w2d.float()
                    parallax_output[:, :, y : y + tile_size, x : x + tile_size] += (
                        predicted_parallax * w2d_cpu
                    )
                    ao_output[:, :, y : y + tile_size, x : x + tile_size] += (
                        predicted_ao * w2d_cpu
                    )
                    metallic_output[:, :, y : y + tile_size, x : x + tile_size] += (
                        predicted_metallic * w2d_cpu
                    )
                    roughness_output[:, :, y : y + tile_size, x : x + tile_size] += (
                        predicted_roughness * w2d_cpu
                    )

                    weight_1ch[:, :, y : y + tile_size, x : x + tile_size] += w2d_cpu
                else:
                    parallax_output[
                        :, :, y : y + tile_size, x : x + tile_size
                    ] += predicted_parallax
                    ao_output[
                        :, :, y : y + tile_size, x : x + tile_size
                    ] += predicted_ao
                    metallic_output[
                        :, :, y : y + tile_size, x : x + tile_size
                    ] += predicted_metallic
                    roughness_output[
                        :, :, y : y + tile_size, x : x + tile_size
                    ] += predicted_roughness

                    weight_1ch[:, :, y : y + tile_size, x : x + tile_size] += 1.0

                # Append mask to output
                mask_output[y : y + tile_size, x : x + tile_size] = mask_visual

                # Explicit memory cleanup after each tile
                del (
                    tile_normal,
                    tile_poisson,
                    tile_mean_curvature,
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
                    predicted_parallax,
                    predicted_ao,
                    metallic_roughness_input,
                    predicted_metallic,
                    predicted_roughness,
                    mask_visual,
                )
                torch.cuda.empty_cache() if device.type == "cuda" else None

    # Normalize output
    if overlap > 0:
        parallax_output = parallax_output / weight_1ch.clamp(min=1e-6)
        ao_output = ao_output / weight_1ch.clamp(min=1e-6)
        metallic_output = metallic_output / weight_1ch.clamp(min=1e-6)
        roughness_output = roughness_output / weight_1ch.clamp(min=1e-6)

    parallax_image = TF.to_pil_image(
        torch.sigmoid(parallax_output).squeeze(0).clamp(0, 1)
    )
    ao_image = TF.to_pil_image(torch.sigmoid(ao_output).squeeze(0).clamp(0, 1))
    metallic_image = TF.to_pil_image(
        torch.sigmoid(metallic_output).squeeze(0).clamp(0, 1)
    )
    roughness_image = TF.to_pil_image(
        torch.sigmoid(roughness_output).squeeze(0).clamp(0, 1)
    )

    return (
        parallax_image,
        ao_image,
        metallic_image,
        roughness_image,
    )


# Find and process normal + diffuse pairs
for normal_path in INPUT_DIR.glob("**/*_n.dds"):
    diffuse_path = normal_path.with_name(normal_path.name.replace("_n.dds", "_d.dds"))
    if not diffuse_path.exists():
        diffuse_path = normal_path.with_name(normal_path.name.replace("_n.dds", ".dds"))

    if not diffuse_path.exists():
        print(f"Skipping {normal_path} - corresponding diffuse map not found.")
        continue

    print(f"Processing:")
    print(f"  Normal Map: {normal_path}")
    print(f"  Diffuse Map: {diffuse_path}")

    input_relative_path = normal_path.relative_to(INPUT_DIR)
    output_dir = BASE_OUTPUT_DIR / input_relative_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call textconv for diffuse
    diffuse_args = TEXCONV_ARGS_SRGB_PNG + ["-o", str(output_dir), str(diffuse_path)]
    normal_args = TEXCONV_ARGS_LINEAR_PNG + ["-o", str(output_dir), str(normal_path)]

    run(diffuse_args, check=True, stdout=DEVNULL, stderr=DEVNULL)
    run(normal_args, check=True, stdout=DEVNULL, stderr=DEVNULL)

    basename = normal_path.stem.replace("_n", "")
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

    print("  ...Generating Albedo")
    albedo_img = predict_albedo(diffuse_img, normal_img)
    print("  ...Generating PBR Maps")
    parallax_img, ao_img, metallic_img, roughness_img = predirect_pbr_maps(
        albedo_img, normal_img
    )

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
    else:
        albedo_png = output_dir / (basename + ".png")
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
