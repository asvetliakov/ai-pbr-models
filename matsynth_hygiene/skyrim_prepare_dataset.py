from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from subprocess import run, DEVNULL
from PIL import Image
import math
import numpy as np
from matsynth_prepare_dataset import lambert, colour_cast
from scipy.ndimage import gaussian_filter

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = (BASE_DIR / "../skyrim_samples").resolve()
OUTPUT_DIR = (BASE_DIR / "../skyrim_processed").resolve()

NON_PBR_INPUT_DIR = INPUT_DIR / "no_pbr"
PBR_INPUT_DIR = INPUT_DIR / "pbr"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXCONV_PATH = (BASE_DIR / "../texconv.exe").resolve()

TEXCONV_ARGS_SRGB = [
    str(TEXCONV_PATH),
    "-nologo",
    "-f",
    # "R8G8B8A8_UNORM",
    "R8G8B8A8_UNORM_SRGB",
    "-ft",
    "png",
    "-m",
    "1",
    "--srgb-in",
    "--srgb-out",
    "-y",
]

TEXCONV_ARGS_LINEAR = [
    str(TEXCONV_PATH),
    "-nologo",
    "-f",
    "R8G8B8A8_UNORM",
    "-ft",
    "png",
    "-m",
    "1",
    "-y",
]


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


def screen_ao_from_uint8(
    height_uint8: np.ndarray,
    sigma: float = 9,
    strength: float = 0.25,
) -> np.ndarray:
    # 1) decode 0…255 → –1…1
    h = (height_uint8.astype(np.float32) - 127.0) / 127.0

    # 2) blur
    h_blur = gaussian_filter(h, sigma=sigma, mode="reflect")

    # 3) residual
    delta = h - h_blur

    # 4) AO = 1 - strength * |delta|
    ao = 1.0 - strength * np.abs(delta)

    # clamp to [0,1]
    ao = np.clip(ao, 0.0, 1.0)

    # return as (H,W,1) if you need a channel
    return ao[..., None]


def resize_texture(
    target_size: int,
    path: Path,
    img: Image.Image,
    resampling: Image.Resampling,
    is_normal=False,
) -> Image.Image:

    min_size = min(img.size)
    if min_size == target_size:
        # Already at target resolution, no need to resize
        return img

    factor = target_size / min_size

    width, height = img.size
    final_width = math.ceil(width * factor)
    final_height = math.ceil(height * factor)

    print(f"Resizing {path} from {width}x{height} to {final_width}x{final_height}")
    img = img.resize((final_width, final_height), resample=resampling)
    if is_normal:
        img = normalize_normal_map(img)

    return img


def process_non_pbr(path: Path):
    normal_texture = path
    base_name = normal_texture.stem.replace("_n", "")
    diffuse_texture = path.with_name(path.stem.replace("_n", "_d") + ".dds")
    if not diffuse_texture.exists():
        diffuse_texture = path.with_name(path.stem.replace("_n", "") + ".dds")

    if not diffuse_texture.exists():
        print(f"Missing diffuse texture for {path.name}, skipping.")
        return

    # Call texconv for the diffuse texture
    diffuse_args = TEXCONV_ARGS_SRGB + [
        "-o",
        str(diffuse_texture.parent),
        str(diffuse_texture),
    ]
    run(diffuse_args, check=True, stdout=DEVNULL)
    diffuse_png = diffuse_texture.with_suffix(".png")

    # Call texconv for the normal texture
    normal_args = TEXCONV_ARGS_LINEAR + [
        "-o",
        str(normal_texture.parent),
        str(normal_texture),
    ]
    run(normal_args, check=True, stdout=DEVNULL)
    normal_png = normal_texture.with_suffix(".png")

    # Move the processed textures to the output directory
    relative_path = normal_texture.relative_to(INPUT_DIR).parent
    output_dir = (OUTPUT_DIR / relative_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    diffuse_dst = output_dir / (base_name + "_diffuse.png")
    if diffuse_dst.exists():
        diffuse_dst.unlink()  # Remove existing file if it exists
    diffuse_png = diffuse_png.rename(diffuse_dst)

    normal_dst = output_dir / (base_name + "_normal.png")
    if normal_dst.exists():
        normal_dst.unlink()  # Remove existing file if it exists
    normal_png = normal_png.rename(normal_dst)

    diffuse_img = Image.open(diffuse_png).convert("RGB")
    diffuse_min_size = min(diffuse_img.size)
    if diffuse_min_size < 1024:
        print(
            f"Diffuse resolution is too low for {diffuse_png}: {diffuse_img.size}, dropping."
        )
        diffuse_png.unlink()
        normal_png.unlink()
        return

    normal_img = Image.open(normal_png).convert("RGB")
    normal_min_size = min(normal_img.size)
    if normal_min_size < 1024:
        print(
            f"Normal resolution is too low for {normal_png}: {normal_img.size}, dropping."
        )
        diffuse_png.unlink()
        normal_png.unlink()
        return

    min_size = min(diffuse_min_size, normal_min_size)
    if min_size > 2048:
        min_size = 2048

    diffuse_img = resize_texture(
        min_size, diffuse_png, diffuse_img, Image.Resampling.LANCZOS
    )
    normal_img = resize_texture(
        min_size, normal_png, normal_img, Image.Resampling.BILINEAR, is_normal=True
    )

    diffuse_img.save(diffuse_png, format="PNG")
    normal_img.save(normal_png, format="PNG")


def process_pbr(rmaos_path: Path):
    rmaos_texture = rmaos_path
    base_name = rmaos_path.stem.replace("_rmaos", "")
    albedo_texture = rmaos_path.with_name(
        rmaos_path.stem.replace("_rmaos", "_d") + ".dds"
    )
    if not albedo_texture.exists():
        albedo_texture = rmaos_path.with_name(
            rmaos_path.stem.replace("_rmaos", "") + ".dds"
        )

    if not albedo_texture.exists():
        print(f"Missing albedo texture for {rmaos_path}, skipping.")
        return

    normal_texture = rmaos_path.with_name(
        rmaos_path.stem.replace("_rmaos", "_n") + ".dds"
    )
    if not normal_texture.exists():
        print(f"Missing normal texture for {rmaos_path}, skipping.")
        return

    parallax_texture = rmaos_path.with_name(
        rmaos_path.stem.replace("_rmaos", "_p") + ".dds"
    )
    if not parallax_texture.exists():
        print(f"Missing prallax texture for {rmaos_path}")

    # Move the processed textures to the output directory
    relative_path = rmaos_texture.relative_to(INPUT_DIR).parent
    output_dir = (OUTPUT_DIR / relative_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call texconv for the albedo texture
    albedo_args = TEXCONV_ARGS_SRGB + [
        "-o",
        str(albedo_texture.parent),
        str(albedo_texture),
    ]
    run(albedo_args, check=True, stdout=DEVNULL)
    albedo_png = albedo_texture.with_suffix(".png")

    # Call texconv for the normal texture
    normal_args = TEXCONV_ARGS_LINEAR + [
        "-o",
        str(normal_texture.parent),
        str(normal_texture),
    ]
    run(normal_args, check=True, stdout=DEVNULL)
    normal_png = normal_texture.with_suffix(".png")

    albedo_img = Image.open(albedo_png).convert("RGB")
    normal_img = Image.open(normal_png).convert("RGB")
    min_albedo_size = min(albedo_img.size)
    min_normal_size = min(normal_img.size)
    if min_albedo_size < 1024 or min_normal_size < 1024:
        print(
            f"Albedo or normal resolution is too low for {albedo_png}: albedo size: {albedo_img.size}, normal size: {normal_img.size}, dropping."
        )
        albedo_png.unlink()
        normal_png.unlink()
        return

    # Call texconv for rmaos texture
    rmaos_args = TEXCONV_ARGS_LINEAR + [
        "-o",
        str(rmaos_texture.parent),
        str(rmaos_texture),
    ]
    run(rmaos_args, check=True, stdout=DEVNULL)
    rmaos_png = rmaos_texture.with_suffix(".png")
    rmaos_img = Image.open(rmaos_png)
    min_rmaos_size = min(rmaos_img.size)
    if min_rmaos_size < 1024:
        min_rmaos_size = 1024

    min_size = min(min_albedo_size, min_normal_size, min_rmaos_size)
    if min_size > 2048:
        min_size = 2048

    if min_size == 1024 and min_rmaos_size == 1024 and min_albedo_size > 2048:
        # Better to upscale RMAOS to 2K rather than downscale albedo from 4K to 1K
        min_size = 2048

    albedo_img = resize_texture(
        min_size, albedo_png, albedo_img, Image.Resampling.LANCZOS
    )
    albedo_img.save(albedo_png, format="PNG")
    albedo_dst = output_dir / (base_name + "_basecolor.png")
    if albedo_dst.exists():
        albedo_dst.unlink()  # Remove existing file if it exists
    albedo_png = albedo_png.rename(albedo_dst)

    normal_img = resize_texture(
        min_size, normal_png, normal_img, Image.Resampling.BILINEAR, is_normal=True
    )
    normal_img.save(normal_png, format="PNG")
    normal_dst = output_dir / (base_name + "_normal.png")
    if normal_dst.exists():
        normal_dst.unlink()  # Remove existing file if it exists
    normal_png = normal_png.rename(normal_dst)

    rmaos_img = resize_texture(
        min_size, rmaos_png, rmaos_img, Image.Resampling.BILINEAR
    )

    rmaos_channels = rmaos_img.split()

    rougness_img = rmaos_channels[0].convert("L")
    metallic_img = rmaos_channels[1].convert("L")
    ao_img = rmaos_channels[2].convert("L")
    specular_img = rmaos_channels[3].convert("RGB")

    roughness_dst = output_dir / (base_name + "_roughness.png")
    if roughness_dst.exists():
        roughness_dst.unlink()
    rougness_img.save(roughness_dst, format="PNG")

    metallic_dst = output_dir / (base_name + "_metallic.png")
    if metallic_dst.exists():
        metallic_dst.unlink()
    metallic_img.save(metallic_dst, format="PNG")

    ao_dst = output_dir / (base_name + "_ao.png")
    if ao_dst.exists():
        ao_dst.unlink()
    ao_img.save(ao_dst, format="PNG")

    specular_dst = output_dir / (base_name + "_specular.png")
    if specular_dst.exists():
        specular_dst.unlink()
    specular_img.save(specular_dst, format="PNG")

    rmaos_png.unlink()

    # rmaos_dst = output_dir / (base_name + "_rmaos.png")
    # if rmaos_dst.exists():
    #     rmaos_dst.unlink()
    # rmaos_png = rmaos_png.rename(rmaos_dst)

    parallax_img = None
    if parallax_texture.exists():
        parallax_args = TEXCONV_ARGS_LINEAR + [
            "-o",
            str(parallax_texture.parent),
            str(parallax_texture),
        ]
        run(parallax_args, check=True, stdout=DEVNULL)
        parallax_png = parallax_texture.with_suffix(".png")
        parallax_img = Image.open(parallax_png).convert("L")
        parallax_img = resize_texture(
            min_size, parallax_png, parallax_img, Image.Resampling.BICUBIC
        )
        parallax_img.save(parallax_png, format="PNG")
        parallax_dst = output_dir / (base_name + "_parallax.png")
        if parallax_dst.exists():
            parallax_dst.unlink()
        parallax_png = parallax_png.rename(parallax_dst)

    # Generate synthetic diffuse texture
    alb_arr = np.array(albedo_img, dtype=np.float32) / 255.0
    normal_arr = np.array(normal_img, dtype=np.float32) / 255.0
    ao_arr = np.array(ao_img, dtype=np.float32) / 255.0
    ao_arr = ao_arr[..., None]  # Make it (H, W, 1) for broadcasting

    diffuse_img = lambert(alb_arr, normal_arr)

    # ! fuck it let's just use screen space AO
    # If doesn't have black pixels in AO then apply it otherwise generate AO from height map (if available)
    # This is needed to prevent making UV padding totally black when corresponding albedo is colored
    # If diffuse will have black padding and original albedo is colored then model will be trying to learn to color black pixels
    # if not np.any(ao_arr < 0.01):
    #     print(f"Applying original AO for {albedo_png}")
    #     # diffuse_img *= ao_arr  # Apply AO
    # elif parallax_img is not None:

    # Generate screen-space AO from parallax map
    if parallax_img is not None:
        parallax_arr = np.array(parallax_img, dtype=np.uint8)
        diffuse_img *= screen_ao_from_uint8(parallax_arr, sigma=9, strength=0.25)

    # diffuse_img *= ao_arr  # Apply AO
    # diffuse_img = np.where(ao_arr != 0, diffuse_img * ao_arr, alb_arr)

    diffuse_img = np.clip(diffuse_img, 0.0, 1.0)

    diffuse_img = colour_cast(diffuse_img)
    diffust_dst = output_dir / (base_name + "_diffuse.png")
    if diffust_dst.exists():
        diffust_dst.unlink()
    diffuse_img = Image.fromarray((diffuse_img * 255).astype(np.uint8), mode="RGB")
    diffuse_img.save(diffust_dst, format="PNG")


def main():
    non_pbr_textures = list(NON_PBR_INPUT_DIR.glob("**/*_n.dds"))
    pbr_textures = list(PBR_INPUT_DIR.glob("**/*_rmaos.dds"))

    with ThreadPoolExecutor(max_workers=14) as executor:
        print(f"Found {len(non_pbr_textures)} non-PBR textures.")
        for path in non_pbr_textures:
            executor.submit(process_non_pbr, path)

        print(f"Found {len(pbr_textures)} PBR textures.")
        for texture in pbr_textures:
            executor.submit(process_pbr, texture)


if __name__ == "__main__":
    main()
