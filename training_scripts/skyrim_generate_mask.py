from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import functional as TF
from transformers.utils.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)
from subprocess import run, DEVNULL
from PIL import Image
import math
from class_materials import CLASS_LIST, CLASS_PALETTE
import numpy as np
import torch

# Ensure project root is in sys.path for imports
import sys
import os

from unet_models import UNetAlbedo
from segformer_6ch import create_segformer

BASE_DIR = Path(__file__).resolve().parent

# INPUT_DIR = (BASE_DIR / "../skyrim_samples_for_maps").resolve()
# OUTPUT_DIR = (BASE_DIR / "../skyrim_processed_for_maps").resolve()
INPUT_DIR = (BASE_DIR / "../skyrim_processed").resolve()
OUTPUT_DIR = (BASE_DIR / "../skyrim_processed").resolve()

NON_PBR_INPUT_DIR = INPUT_DIR / "no_pbr"
PBR_INPUT_DIR = INPUT_DIR / "pbr"

device = torch.device("cuda")

unet_alb = UNetAlbedo(
    in_ch=6,
    cond_ch=512,
).to(device)

checkpoint_path = (BASE_DIR / "../weights/a3/unet_albedo/best_model.pt").resolve()
checkpoint = torch.load(checkpoint_path, map_location=device)
unet_alb.load_state_dict(checkpoint["unet_albedo_model_state_dict"])
for param in unet_alb.parameters():
    param.requires_grad = False


# Create segformer and load best weights
segformer = create_segformer(
    num_labels=len(CLASS_LIST),
    device=device,
    lora=True,
    frozen=True,
)
segformer_best_weights_path = (
    BASE_DIR / "../weights/s5/segformer/best_model.pt"
).resolve()
segformer_checkpoint = torch.load(segformer_best_weights_path, map_location=device)
segformer.base_model.load_state_dict(
    segformer_checkpoint["base_model_state_dict"],
)
segformer.load_state_dict(
    segformer_checkpoint["lora_state_dict"],
)


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    """
    Convert a (H, W) torch mask to a (H, W, 3) PIL.Image with PALETTE colors.
    """
    # 1) ensure CPU & numpy
    if mask.ndim == 3:  # batch dim
        mask = mask[0]  # take first in batch, or loop over them
    mask_np = mask.cpu().numpy().astype(np.uint8)  # shape (H, W)

    # 2) build an RGB array
    h, w = mask_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_PALETTE.items():
        color_img[mask_np == cls_idx] = color

    # 3) convert to PIL
    return Image.fromarray(color_img)


def create_albedo(path: Path):
    print(f"Processing: {path}")

    base_name = path.stem.replace("_diffuse", "")
    image = Image.open(path).convert("RGB")
    normal = Image.open(path.with_name(base_name + "_normal.png")).convert("RGB")

    W, H = image.size
    tile_size = min(H, W)
    # tile_size = 1024
    overlap = 0
    if H != W:
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
    else:
        xs = [0]
        ys = [0]

    mask_output = torch.zeros(H, W, device=device)
    # mask_output = torch.zeros((H // 4), W // 4, device=device)
    # print(f"Mask output shape: {mask_output.shape}")

    output = torch.zeros(1, 3, H, W, device=device)  # (B, C, H, W)
    weight = torch.zeros_like(output)  # (B, C, H, W)

    # Create blending mask (raised cosine window)
    ramp = torch.linspace(0, 1, overlap, device=device)
    window1d = torch.ones(tile_size, device=device)

    if overlap > 0:
        window1d[:overlap] = ramp
        window1d[-overlap:] = ramp.flip(0)

    # Make 2d blending mask
    w2d = window1d[None, None, :, None] * window1d[None, None, None, :]

    for y in ys:
        for x in xs:
            with torch.no_grad():
                tile_image = image.crop((x, y, x + tile_size, y + tile_size))
                tile_normal = normal.crop((x, y, x + tile_size, y + tile_size))

                tile_normal = TF.to_tensor(tile_normal)  # (C, H, W)
                tile_normal = (
                    TF.normalize(
                        tile_normal,
                        mean=IMAGENET_STANDARD_MEAN,
                        std=IMAGENET_STANDARD_STD,
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                tile_image = TF.to_tensor(tile_image)  # (C, H, W)

                # Get predirected albedo first
                tile_image = TF.normalize(
                    tile_image,
                    mean=IMAGENET_STANDARD_MEAN,
                    std=IMAGENET_STANDARD_STD,
                )
                tile_image = tile_image.unsqueeze(0).to(device)

                input = torch.cat((tile_image, tile_normal), dim=1)

                # predict albedo image
                predicted_albedo = unet_alb(input, None)

                predicted_albedo = TF.normalize(
                    predicted_albedo,
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                )

                segformer_input = torch.cat((predicted_albedo, tile_normal), dim=1)

                # Get probabilities
                setformer_output = segformer(segformer_input, output_hidden_states=True)
                mask = setformer_output.logits
                seg_feats = setformer_output.hidden_states[-1]  # (B, C, H/4, W/4)

                # Run unet again with segformer features
                predicted_albedo = unet_alb(input, seg_feats)

                if overlap > 0:
                    output[:, :, y : y + tile_size, x : x + tile_size] += (
                        predicted_albedo * w2d
                    )
                    weight[:, :, y : y + tile_size, x : x + tile_size] += w2d
                else:
                    output[
                        :, :, y : y + tile_size, x : x + tile_size
                    ] += predicted_albedo
                    weight[:, :, y : y + tile_size, x : x + tile_size] += 1.0

                mask: torch.Tensor = torch.nn.functional.interpolate(
                    mask,
                    size=tile_image.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                # Convert to class indices
                mask = torch.argmax(mask, dim=1).squeeze(0)  # (H/4, W/4)
                # print(f"Mask shape: {mask.shape}")

                # Append mask to output
                mask_output[y : y + tile_size, x : x + tile_size] = mask
                # mask_output[ds_y : ds_y + ds_tile, ds_x : ds_x + ds_tile] = mask

    # Normalize output
    if overlap > 0:
        output = output / weight.clamp(min=1e-6)

    mask_image = mask_to_pil(mask_output)
    # Concatenate original image and mask
    # final_image = Image.new("RGB", (W + mask_image.width, H))
    # final_image.paste(image, (0, 0))
    # final_image.paste(mask_image, (W, 0))
    out_name = path.with_name(base_name + "_mask.png")
    mask_image.save(out_name, "PNG")

    # Save the albedo image
    albedo_image = TF.to_pil_image(output.squeeze(0).cpu())
    albedo_out_name = path.with_name(base_name + "_basecolor.png")
    albedo_image.save(albedo_out_name, "PNG")


def process_path(path: Path, is_pbr=False):
    print(f"Processing: {path}")

    base_name = path.stem.replace("_diffuse" if not is_pbr else "_basecolor", "")
    image = Image.open(path).convert("RGB")
    normal = Image.open(path.with_name(base_name + "_normal.png")).convert("RGB")

    W, H = image.size
    tile_size = min(H, W)
    # tile_size = 1024
    if H != W:
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
    else:
        xs = [0]
        ys = [0]

    mask_output = torch.zeros(H, W, device=device)
    # mask_output = torch.zeros((H // 4), W // 4, device=device)
    # print(f"Mask output shape: {mask_output.shape}")

    # output = torch.zeros(1, 3, H, W, device=device)  # (B, C, H, W)
    # weight = torch.zeros_like(output)  # (B, C, H, W)

    for y in ys:
        for x in xs:
            with torch.no_grad():
                tile_image = image.crop((x, y, x + tile_size, y + tile_size))
                tile_normal = normal.crop((x, y, x + tile_size, y + tile_size))

                tile_normal = TF.to_tensor(tile_normal)  # (C, H, W)
                tile_normal = (
                    TF.normalize(
                        tile_normal,
                        mean=IMAGENET_STANDARD_MEAN,
                        std=IMAGENET_STANDARD_STD,
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                tile_image = TF.to_tensor(tile_image)  # (C, H, W)

                if is_pbr:
                    # Directly use albedo for segformer input
                    tile_image = TF.normalize(
                        tile_image,
                        mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD,
                    )
                    tile_image = tile_image.unsqueeze(0).to(device)  # (1, C, H, W)
                    segformer_input = torch.cat((tile_image, tile_normal), dim=1)
                else:
                    # Get predirected albedo first
                    tile_image = TF.normalize(
                        tile_image,
                        mean=IMAGENET_STANDARD_MEAN,
                        std=IMAGENET_STANDARD_STD,
                    )
                    tile_image = tile_image.unsqueeze(0).to(device)

                    input = torch.cat((tile_image, tile_normal), dim=1)

                    # predict albedo image
                    predicted_albedo = unet_alb(input, None)

                    predicted_albedo = TF.normalize(
                        predicted_albedo,
                        mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD,
                    )

                    segformer_input = torch.cat((predicted_albedo, tile_normal), dim=1)

                # Get probabilities
                mask = segformer(segformer_input).logits

                mask: torch.Tensor = torch.nn.functional.interpolate(
                    mask,
                    size=tile_image.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                # Convert to class indices
                mask = torch.argmax(mask, dim=1).squeeze(0)  # (H/4, W/4)
                # print(f"Mask shape: {mask.shape}")

                # ds_y = y // 4  # downsampled y
                # ds_x = x // 4  # downsampled x
                # ds_tile = tile_size // 4  # downsampled tile size

                # Append mask to output
                mask_output[y : y + tile_size, x : x + tile_size] = mask
                # mask_output[ds_y : ds_y + ds_tile, ds_x : ds_x + ds_tile] = mask

    mask_image = mask_to_pil(mask_output)
    # Concatenate original image and mask
    # final_image = Image.new("RGB", (W + mask_image.width, H))
    # final_image.paste(image, (0, 0))
    # final_image.paste(mask_image, (W, 0))
    out_name = path.with_name(base_name + "_mask.png")
    mask_image.save(out_name, "PNG")


def main():
    non_pbr_textures = list(NON_PBR_INPUT_DIR.glob("**/*_diffuse.png"))
    pbr_textures = list(PBR_INPUT_DIR.glob("**/*_basecolor.png"))

    # process_non_pbr(non_pbr_textures[66])  # Test single texture processing

    # for path in non_pbr_textures:
    #     process_path(path, is_pbr=False)

    for path in non_pbr_textures:
        create_albedo(path)

    # for path in pbr_textures:
    #     process_path(path, is_pbr=True)

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     print(f"Found {len(non_pbr_textures)} non-PBR textures.")
    #     for path in non_pbr_textures:
    #         executor.submit(process_non_pbr, path)

    # print(f"Found {len(pbr_textures)} PBR textures.")
    # for texture in pbr_textures:
    #     executor.submit(process_pbr, texture)


if __name__ == "__main__":
    main()
