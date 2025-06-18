import cv2
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter
import random
import PIL.Image as Image
import json

with open("./matsynth_final_indexes.json", "r") as f:
    index_data = json.load(f)


# def height_to_parallax_png(h_path: Path, p_path: Path, gain: float = 1.0):
#     # load 16-bit absolute height
#     h16 = cv2.imread(str(h_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

#     # normalise 0‥1
#     h01 = (h16 - h16.min()) / (h16.max() - h16.min() + 1e-6)

#     # centre and scale
#     h_gain = 0.5 + gain * (h01 - 0.5)

#     # convert to 8-bit
#     h8 = (h_gain * 255).round().clip(0, 255).astype(np.uint8)

#     cv2.imwrite(str(p_path), h8)  # PNG → convert to DDS later


def height_to_ao(
    pil_h: Image.Image,
    k: float = 4.0,
    blur: int = 9,
    *,
    low_sigma: int = 15,  # ≈ size of planks/boards
    hi_blend: float = 0.20,  # how much tiny detail to mix back
    rough_map: np.ndarray | None = None,  # optional (H,W) ∈[0,1]
    material: str | None = None,  # e.g. "wood"
) -> Image.Image:
    """
    Returns an 8-bit PIL AO image (0=occluded, 255=open).

    - Two-scale trick: large sigma removes wood grain; small blend
      keeps a hint of micro cavities.
    - If `rough_map` is provided, AO is damped in rough regions.
    - If material=="wood" or any string in `SKIP_MATERIALS`, returns flat AO.
    """
    SKIP_MATERIALS = {"wood_parquet", "wood_planks", "wood"}
    if material and material.lower() in SKIP_MATERIALS:
        h, w = pil_h.size
        return Image.fromarray(np.full((w, h), 255, np.uint8), mode="L")

    # --------------- load & normalise height -----------------
    h16 = np.array(pil_h, dtype=np.float32)
    h01 = (h16 - h16.min()) / (h16.max() - h16.min() + 1e-6)

    # --------------- large-scale gradient --------------------
    h_low = gaussian_filter(h01, sigma=low_sigma)
    dx = cv2.Sobel(h_low, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(h_low, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(dx * dx + dy * dy)

    ao = 1.0 - k * grad  # invert crevices → dark
    ao = cv2.GaussianBlur(ao, (blur, blur), 0)

    # --------------- tiny-detail blend (optional) ------------
    if hi_blend > 0:
        h_hi = gaussian_filter(h01, sigma=3)
        dx = cv2.Sobel(h_hi, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(h_hi, cv2.CV_32F, 0, 1, ksize=3)
        grad_hi = np.sqrt(dx * dx + dy * dy)
        ao_hi = 1.0 - k * grad_hi
        ao_hi = cv2.GaussianBlur(ao_hi, (blur, blur), 0)
        ao = ao * (1 - hi_blend) + ao_hi * hi_blend

    # --------------- roughness weighting ---------------------
    if rough_map is not None:
        rough = rough_map.astype(np.float32)
        if rough.ndim == 3:  # squeeze if loaded as H×W×1
            rough = rough[..., 0]
        ao = 1.0 - (1.0 - ao) * (1.0 - rough)  # high rough → flatten AO

    ao = np.clip(ao, 0, 1)
    return Image.fromarray((ao * 255).astype(np.uint8), mode="L")


def lambert(alb, nrm):
    theta, phi = np.random.uniform(0, np.pi / 2), np.random.uniform(0, 2 * np.pi)
    sun = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        np.float32,
    )
    n = nrm * 2 - 1
    n /= np.linalg.norm(n, 2, 2, keepdims=True) + 1e-5
    shade = np.clip((n * sun).sum(2, keepdims=True), 0, 1)
    return alb * (0.4 + 0.6 * shade)


def screen_ao(height):
    if height is None:
        return 1
    ao = 1 - gaussian_filter(height, sigma=9)
    return 1 - 0.25 * ao[..., None]


def specular_highlight(spec_rgb, rough, metallic):
    """
    spec_rgb : (H,W,3) float in [0,1], PBR specular color
    rough    : (H,W) float in [0,1] or None
    metallic : (H,W) float in [0,1] or None

    returns: (H,W,3) float in [0,1], tinted specular highlight
    """
    if spec_rgb is None:
        print("!!! No specular map provided, skipping highlight generation")
        # build a zero-array with same height/width
        template = rough if rough is not None else metallic
        H, W = template.shape[:2]
        return np.zeros((H, W, 3), dtype=np.float32)

    # 1) compute luminance from the RGB specular color
    lum = (
        0.2126 * spec_rgb[..., 0]
        + 0.7152 * spec_rgb[..., 1]
        + 0.0722 * spec_rgb[..., 2]
    )

    # 2) substitute defaults if needed
    if rough is None:
        rough_arr = np.zeros_like(lum)
    else:
        rough_arr = rough

    if metallic is None:
        metallic_arr = np.ones_like(lum)
    else:
        metallic_arr = metallic

    # 3) gloss intensity: colored spec * (1 - rough) * metallic
    gloss = lum * (1.0 - rough_arr) * metallic_arr

    # 4) soft‐blur for a ‘box’ highlight
    gloss = gaussian_filter(gloss, sigma=5)

    # 5) reapply color tint
    #    (broadcast gloss to RGB channels)
    return spec_rgb * gloss[..., None]


def colour_cast(img):
    r, g, b = [random.uniform(0.97, 1.05) for _ in range(3)]
    return img * np.array([r, g, b])


def process_batch(indexes, dataset: Dataset):
    batch_dataset = dataset.select(indexes)

    for i, item in enumerate(batch_dataset):
        name = item["name"]  # type: ignore
        height: Image.Image = item["height"]  # type: ignore
        albedo: Image.Image = item["basecolor"]  # type: ignore
        normal: Image.Image = item["normal"]  # type: ignore
        roughness: Image.Image = item["roughness"]  # type: ignore
        metallic: Image.Image = item["metallic"]  # type: ignore
        specular: Image.Image = item["specular"]  # type: ignore

        albedo = albedo.resize((2048, 2048), resample=Image.Resampling.LANCZOS)

        height = height.resize((2048, 2048), resample=Image.Resampling.BICUBIC)
        normal = normal.resize((2048, 2048), resample=Image.Resampling.BILINEAR)
        roughness = roughness.resize((2048, 2048), resample=Image.Resampling.BILINEAR)
        metallic = metallic.resize((2048, 2048), resample=Image.Resampling.BILINEAR)
        specular = specular.resize((2048, 2048), resample=Image.Resampling.BILINEAR)

        category = index_data["new_category_mapping"].get(name, None)
        if category is None:
            print(f"!!! Skipping {name} ({i}) - category not found")
            continue

        out_path = Path(f"./matsynth_processed/{category}")
        out_path.mkdir(parents=True, exist_ok=True)

        # Generate AO map based on height map data
        print(f"Generating AO for {name} ({i})")
        roughness_arr = np.asarray(roughness.convert("L"), dtype=np.float32) / 255.0
        # default, ceramic, metal
        low_sigma = 15
        hi_blend = 0.20
        if category == "wood":
            low_sigma = 18
            # hi_blend = 0.10
            hi_blend = 0.13
        elif category == "fabric":
            low_sigma = 18
            hi_blend = 0.15
        elif category == "ground":
            low_sigma = 12
            hi_blend = 0.25
        elif category == "stone":
            low_sigma = 12
            hi_blend = 0.25
        elif category == "leather":
            low_sigma = 10
            hi_blend = 0.20

        ao_map = height_to_ao(
            height.convert("I;16"),
            rough_map=roughness_arr,
            low_sigma=low_sigma,
            hi_blend=hi_blend,
        )

        ao_map.save(out_path / f"{name}_ao.png", format="PNG")
        # Save also albedo for inspection
        albedo.save(out_path / f"{name}_basecolor.png", format="PNG")

        if i in index_data["same_diffuse_albedo_indexes"]:
            print(f"Generating synthetic diffuse for {name} ({i})")
            alb_arr = np.asarray(albedo.convert("RGB"), dtype=np.float32) / 255.0
            normal_arr = np.asarray(normal.convert("RGB"), dtype=np.float32) / 255.0
            height_arr = np.asarray(height.convert("I;16"), dtype=np.float32) / 65535.0
            metallic_arr = np.asarray(metallic.convert("L"), dtype=np.float32) / 255.0
            # Specular map in MatSynth is in RGB format with PBR specularity
            specular_arr = np.asarray(specular.convert("RGB"), dtype=np.float32) / 255.0

            synth = lambert(alb_arr, normal_arr)
            synth *= screen_ao(height_arr)
            synth += specular_highlight(specular_arr, roughness_arr, metallic_arr)
            synth = colour_cast(synth)

            synth_img = Image.fromarray(
                (np.clip(synth, 0, 1) * 255).astype(np.uint8), mode="RGB"
            )
            synth_img.save(out_path / f"{name}_diffuse.png", format="PNG")


def main():
    dataset: Dataset = load_dataset("gvecchio/MatSynth", split="train", streaming=False, num_proc=8)  # type: ignore
    dataset = dataset.select_columns(
        [
            "name",
            "category",
            "basecolor",
            "height",
            "normal",
            "roughness",
            "metallic",
            "specular",
        ]
    )
    valid_indexes = index_data["all_valid_indexes"]

    workers = 12
    # Split the valid indexes into chunks for parallel processing
    chunk_size = len(valid_indexes) // workers + 1
    index_chunks = [
        valid_indexes[i : i + chunk_size]
        for i in range(0, len(valid_indexes), chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_batch, chunk, dataset) for chunk in index_chunks
        ]

        for future in futures:
            try:
                future.result()  # Wait for the batch to complete
            except Exception as e:
                print(f"Error processing batch: {e}")


if __name__ == "__main__":
    main()
