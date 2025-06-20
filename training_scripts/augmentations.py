# Ensure we import it here to set random(seed)
import seed
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional
from noise import pnoise2
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from train_dataset import normalize_normal_map

# import albumentations as A
# import cv2


def get_random_crop(
    albedo: Image.Image,
    normal: Image.Image,
    size: tuple[int, int],
    resize_to: Optional[list[int]],
    augmentations: Optional[bool] = True,
) -> tuple[Image.Image, Image.Image]:
    """
    Crop and resize two images to the same size.
    """
    i, j, h, w = T.RandomCrop.get_params(albedo, output_size=size)  # type: ignore

    final_albedo = TF.crop(albedo, i, j, h, w)  # type: ignore
    final_normal = TF.crop(normal, i, j, h, w)  # type: ignore

    if augmentations:
        if random.random() < 0.5:
            final_albedo = TF.hflip(final_albedo)
            final_normal = TF.hflip(final_normal)

        if random.random() < 0.5:
            final_albedo = TF.vflip(final_albedo)
            final_normal = TF.vflip(final_normal)

        # if random.random() < 0.5:
        # Don't try to rotate if the size is not square
        if size[0] == size[1]:
            k = random.randint(0, 3)
            final_albedo = TF.rotate(final_albedo, angle=k * 90)
            final_normal = TF.rotate(final_normal, angle=k * 90)

    if resize_to is not None:
        final_albedo = TF.resize(
            final_albedo, resize_to, interpolation=TF.InterpolationMode.LANCZOS
        )
        final_normal = TF.resize(
            final_normal, resize_to, interpolation=TF.InterpolationMode.BILINEAR
        )
        final_normal = normalize_normal_map(final_normal)  # type: ignore

    # image not tensors
    return final_albedo, final_normal  # type: ignore


def make_grain_noise(mask_size: tuple[int, int], strength: float = 0.05) -> Image.Image:
    """
    Create a grayscale “grain” mask in [0…1], then turn into an RGBA PIL image
    with alpha = strength, ready to be composited over an RGB albedo crop.
    """
    H, W = mask_size
    # 1) start with standard normal noise
    noise = np.random.randn(H, W).astype(np.float32)
    # 2) normalize to [0,1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    # 3) convert to 8-bit gray
    gray = (noise * 255).astype(np.uint8)
    # 4) create an RGBA image with that gray in RGB and `strength*255` in alpha
    alpha = int(strength * 255)
    rgba = np.stack([gray, gray, gray, np.full_like(gray, fill_value=alpha)], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def overlay_grain(albedo_crop: Image.Image, strength: float = 0.05) -> Image.Image:
    """
    Alpha-composite a grain mask over the given PIL RGB crop.
    """
    H, W = albedo_crop.size[1], albedo_crop.size[0]
    grain = make_grain_noise((H, W), strength=strength)
    return Image.alpha_composite(albedo_crop.convert("RGBA"), grain).convert("RGB")


def make_dirt_overlay(
    size: tuple[int, int],
    scale: float = 100.0,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    strength: float = 0.2,
) -> Image.Image:
    """
    Returns an RGBA PIL image of Perlin‐noise “dirt” for blending:
      - size: (width, height) of the target crop, e.g. (1024,1024)
      - scale: higher → larger blobs
      - octaves/persistence/lacunarity: Perlin parameters
      - strength: alpha of the overlay in [0…1]
    """
    W, H = size
    # 1) Generate a 2D grid of Perlin noise in [0,1]
    dirt = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            nx = x / scale
            ny = y / scale
            # pnoise2 in [−1…1]
            v = pnoise2(
                nx,
                ny,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=random.randint(0, 100),
            )
            dirt[y, x] = (v + 1) / 2.0

    # 2) Normalize to full 0…255
    dirt = (255 * (dirt - dirt.min()) / (dirt.max() - dirt.min() + 1e-6)).astype(
        np.uint8
    )

    # 3) Build an RGBA image: gray in RGB, alpha = strength×255
    alpha = int(strength * 255)
    rgba = np.stack([dirt, dirt, dirt, np.full_like(dirt, alpha)], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def overlay_dirt(
    albedo_crop: Image.Image, strength: float = 0.2, **perlin_kwargs
) -> Image.Image:
    """
    Alpha‐composite a Perlin dirt overlay over your RGB PIL crop.
    """
    W, H = albedo_crop.size
    dirt = make_dirt_overlay((W, H), strength=strength, **perlin_kwargs)
    return Image.alpha_composite(albedo_crop.convert("RGBA"), dirt).convert("RGB")


def make_specular_sprite(
    size: tuple[int, int], radius: float = 0.2, strength: float = 0.15
) -> Image.Image:
    """
    Create a blurred white circle on transparent background.
    - size: (W, H) of your crop, e.g. (1024,1024)
    - radius: fraction of W for the circle radius (0.2 → 20% of width)
    - strength: alpha of the spot (0…1)
    """
    W, H = size
    # 1) blank RGBA image
    sprite = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    # 2) random center somewhere in the upper half (you can tweak)
    cx = random.uniform(W * 0.3, W * 0.7)
    cy = random.uniform(H * 0.1, H * 0.4)
    r = radius * W
    # 3) draw solid white circle
    bbox = [cx - r, cy - r, cx + r, cy + r]
    draw.ellipse(bbox, fill=(255, 255, 255, int(strength * 255)))
    # 4) blur it to soften the edges
    blur_radius = r * 0.5  # half the radius
    sprite = sprite.filter(ImageFilter.GaussianBlur(blur_radius))
    return sprite


def overlay_specular(
    metal_crop: Image.Image, radius: float = 0.2, strength: float = 0.15
) -> Image.Image:
    """
    Alpha-composite a specular highlight sprite onto your metal patch.
    """
    W, H = metal_crop.size
    sprite = make_specular_sprite((W, H), radius=radius, strength=strength)
    return Image.alpha_composite(metal_crop.convert("RGBA"), sprite).convert("RGB")


# elastic_warp = A.ElasticTransform(
#     alpha=10,  # medium-strength warp (±10 px)
#     sigma=6,  # smoothness
#     # alpha_affine=0,  # disable affine component
#     interpolation=cv2.INTER_LANCZOS4,
#     border_mode=cv2.BORDER_REFLECT_101,
#     approximate=False,
#     same_dxdy=False,
#     mask_interpolation=cv2.INTER_NEAREST,
#     noise_distribution="gaussian",
#     p=0.4,  # 40% of images warped
# )

# def elastic_warp(
#     img: Image.Image, grid: int = 4, magnitude: float = 10.0
# ) -> Image.Image:
#     """
#     Apply a small elastic grid warp to a PIL RGB image.

#     Args:
#       img       -- PIL.Image RGB
#       grid      -- number of grid cells per axis (e.g. 4 → 4×4 mesh)
#       magnitude -- max displacement (in pixels) for each mesh point

#     Returns:
#       A new PIL.Image with a slight elastic warp.
#     """
#     W, H = img.size
#     gx = min(grid, W)
#     x_step = max(1, W // gx)
#     gy = min(grid, H)
#     y_step = max(1, H // gy)

#     mesh = []
#     for i in range(gx):
#         for j in range(gy):
#             x0, y0 = i * x_step, j * y_step
#             x1 = min((i + 1) * x_step, W)
#             y1 = min((j + 1) * y_step, H)

#             # destination quad
#             dx0 = random.uniform(-magnitude, magnitude)
#             dy0 = random.uniform(-magnitude, magnitude)
#             dx1 = random.uniform(-magnitude, magnitude)
#             dy1 = random.uniform(-magnitude, magnitude)

#             bbox = (x0, y0, x1, y1)
#             mesh_quad = (
#                 x0 + dx0,
#                 y0 + dy0,  # top-left
#                 x1 + dx1,
#                 y0 + dy0,  # top-right
#                 x1 + dx1,
#                 y1 + dy1,  # bottom-right
#                 x0 + dx0,
#                 y1 + dy1,  # bottom-left
#             )
#             mesh.append((bbox, mesh_quad))

#     return img.transform(
#         (W, H),
#         Image.Transform.MESH,
#         mesh,
#         resample=Image.Resampling.BICUBIC,
#     )


def selective_aug(albedo, normal, category: str):
    if category == "wood":
        if random.random() < 0.5:
            albedo = TF.adjust_brightness(
                albedo, brightness_factor=random.uniform(0.9, 1.1)
            )
        if random.random() < 0.5:
            albedo = TF.adjust_hue(albedo, hue_factor=random.uniform(-0.05, 0.05))
        if random.random() < 0.3:
            albedo = overlay_grain(albedo, strength=random.uniform(0.05, 0.10))  # type: ignore
    elif category == "stone":
        if random.random() < 0.5:
            albedo = TF.adjust_brightness(
                albedo, brightness_factor=random.uniform(0.92, 1.08)
            )
        if random.random() < 0.3:
            albedo = overlay_dirt(
                albedo,  # type: ignore
                strength=random.uniform(0.1, 0.25),  # dirt opacity 10–25%
                scale=random.uniform(200, 400),  # size of dirt blobs
                octaves=4,
                persistence=0.5,
                lacunarity=2.0,
            )
    elif category == "metal":
        if random.random() < 0.3:
            albedo = overlay_specular(
                albedo,
                radius=random.uniform(0.1, 0.2),  # spot size 10–20% of width
                strength=random.uniform(0.1, 0.2),  # opacity 10–20%
            )
    elif category == "fabric":
        if random.random() < 0.5:
            albedo = TF.adjust_hue(albedo, hue_factor=random.uniform(-0.12, 0.12))
        # not worth headache for now
        # if random.random() < 0.3:
        #     alb_np = np.array(albedo)
        #     norm_np = np.array(normal)
        #     data = elastic_warp(image=alb_np, mask=norm_np)
        #     albedo = Image.fromarray(data["image"])
        #     normal = Image.fromarray(data["mask"])
    elif category == "leather":
        if random.random() < 0.5:
            albedo = TF.adjust_brightness(
                albedo, brightness_factor=random.uniform(0.88, 1.12)
            )
        if random.random() < 0.5:
            albedo = TF.adjust_hue(albedo, hue_factor=random.uniform(-0.08, 0.08))
    # Ground, cermaic
    else:
        if random.random() < 0.5:
            albedo = TF.adjust_brightness(
                albedo, brightness_factor=random.uniform(0.9, 1.1)
            )
        if random.random() < 0.3:
            albedo = overlay_dirt(
                albedo,  # type: ignore
                strength=random.uniform(0.10, 0.20),  # 10–20% opacity of the dirt
                scale=random.uniform(300.0, 600.0),  # controls blob size
                octaves=4,  # number of noise layers
                persistence=0.5,  # contrast of layers
                lacunarity=2.0,  # frequency multiplier between octaves
            )
    return albedo, normal


def make_full_image_mask(category_id: int, img_size: tuple[int, int]) -> torch.Tensor:
    """
    Build a segmentation mask of shape (H, W) where every pixel = category_id.
    """
    H, W = img_size
    # numpy array filled with your class index
    mask_np = np.full((H, W), fill_value=category_id, dtype=np.int64)
    # convert to torch LongTensor
    return torch.from_numpy(mask_np)
