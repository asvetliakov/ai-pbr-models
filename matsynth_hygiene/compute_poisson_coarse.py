from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import torch
import cv2
import kornia as K

BASE_DIR = Path(__file__).resolve().parent
# INPUT_DIR = (
#     BASE_DIR
#     / "../skyrim_processed_for_maps/pbr/Faultier's PBR Skyrim AIO 4k/textures/pbr/dungeons/dwemerruins"
# ).resolve()
INPUT_DIR = (BASE_DIR / "../skyrim_processed_for_maps").resolve()
OUTLIER_DIR = (BASE_DIR / "../temp/outliers").resolve()

OUTLIER_DIR.mkdir(parents=True, exist_ok=True)


def poisson_from_normal(n: np.ndarray, path: Path) -> tuple[np.ndarray, bool]:
    """
    n : H×W×3 DirectX tangent normal, already in [-1, 1] (NO G-flip).
    Returns coarse height in [0,1] as float32.
    """

    # Pre-smooth normals to avoid noise artifacts
    for c in range(3):
        n[..., c] = cv2.medianBlur(n[..., c], ksize=3)

    # print(f"Processing {path}...")
    nz = np.clip(n[..., 2], 1e-3, 1.0)  # avoid divide-by-zero
    gx = n[..., 0] / nz  # slope in +x (cols → right)
    gy = n[..., 1] / nz  # slope in +y (rows → down)

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
    is_outlier = span > 100
    if is_outlier:
        print(f"{path}, Height span: {span}")
    h = (h - h.min()) / span  # [0,1]

    # Avoid outliers
    p1, p99 = np.percentile(h, [1, 99])
    h_clamped = np.clip(h, p1, p99)
    h = (h_clamped - p1) / (p99 - p1 + 1e-6)

    return (h.astype(np.float32), is_outlier)


def process_normal(path: Path):
    # print(f"Processing {path}")
    normal = Image.open(path).convert("RGB")
    normal_arr = np.array(normal, dtype=np.float32) / 255.0
    # Convert to -1 to 1 range
    normal_arr = (normal_arr - 0.5) * 2.0

    (poisson, is_outlier) = poisson_from_normal(normal_arr, path)
    blur = cv2.GaussianBlur(poisson, (15, 15), 5)

    out = path.with_suffix(".npy").with_name(
        path.stem.replace("_normal", "_poisson_blur") + ".npy"
    )
    np.save(out, blur)

    h16 = (blur * 65535.0).astype(np.uint16)
    Image.fromarray(h16, mode="I;16").save(out.with_suffix(".png"))

    if is_outlier:
        outlier_path = OUTLIER_DIR / out.with_suffix(".png").name
        Image.fromarray(h16, mode="I;16").save(outlier_path)

    return


def main():
    normals = list(INPUT_DIR.glob("**/*_normal.png"))

    with ThreadPoolExecutor(max_workers=14) as executor:
        print(f"Found {len(normals)} normals.")
        for path in normals:
            executor.submit(process_normal, path)


if __name__ == "__main__":
    main()
