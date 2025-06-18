"""
Matsynth Dataset Preparation Script

This script processes the Matsynth dataset to identify and filter image pairs for training and evaluation. It performs the following steps:
- Loads the dataset splits ("train" and "test").
- Filters out samples where the diffuse image is effectively black.
- Identifies and excludes samples where the diffuse and albedo images are either pixel-identical or highly similar (PSNR > 45 dB).
- Collects and saves the indexes of valid, filtered samples for downstream tasks.

The results are saved as a JSON file containing lists of indexes for each split and filtering category.

Note: MatSynth category names (indexes are starting from 0) are: ['ceramic', 'concrete', 'fabric', 'ground', 'leather', 'marble', 'metal', 'misc', 'plaster', 'plastic', 'stone', 'terracotta', 'wood']
"""

import json
from datasets import load_dataset
from PIL import Image
import numpy as np, math


def psnr(img1, img2):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is a common metric used to measure the quality of reconstruction of lossy compression codecs.
    The higher the PSNR, the better the quality of the reconstructed image.

    Args:
        img1 (np.ndarray): The first image (ground truth or reference image).
        img2 (np.ndarray): The second image (test or reconstructed image).

    Returns:
        float: The PSNR value in decibels (dB).

    Notes:
        - Both images must have the same dimensions.
        - The images are assumed to have pixel values in the range [0, 255].
    """
    mse = ((img1.astype(np.float32) - img2.astype(np.float32)) ** 2).mean()
    return 20 * math.log10(255.0 / math.sqrt(mse + 1e-8))


def is_black_image(image: Image.Image, tol=5):
    """
    Determines if the given image is effectively black within a specified tolerance.

    Args:
        image (Image.Image): The image to check.
        tol (int, optional): The tolerance value for pixel intensity variation.
            If the difference between the maximum and minimum pixel values is less than this value,
            the image is considered black. Defaults to 5.

    Returns:
        bool: True if the image is None or considered black within the given tolerance, False otherwise.
    """
    if image is None:
        return True
    img_array = np.array(image)
    return img_array.max() - img_array.min() < tol


def is_similar_images(diffuse: Image.Image, albedo: Image.Image, threshold=45):
    """
    Determines whether two images are similar based on pixel equality or PSNR (Peak Signal-to-Noise Ratio).

    Args:
        diffuse (Image.Image): The first image to compare (typically the diffuse image).
        albedo (Image.Image): The second image to compare (typically the albedo image).
        threshold (float, optional): The PSNR threshold above which images are considered similar. Defaults to 45.

    Returns:
        bool: True if the images are considered similar (either identical or PSNR exceeds the threshold), False otherwise.
    """
    if diffuse is None or albedo is None:
        return False
    diffuse_arr = np.array(diffuse)
    albedo_arr = np.array(albedo)
    if (
        diffuse_arr.shape == albedo_arr.shape
        and (diffuse_arr == albedo_arr).all()
        or psnr(diffuse_arr, albedo_arr) > threshold
    ):
        return True
    return False


def process_split(split_name):
    ds = load_dataset(
        "gvecchio/MatSynth",
        split=split_name,
        streaming=False,
        num_proc=4,
    )
    all_valid_indexes = []
    without_diffuse = []
    same_diffuse_albedo_indexes = []
    final_train_indexes = []
    for idx, item in enumerate(ds):
        albedo: Image.Image = item["basecolor"]
        diffuse: Image.Image = item["diffuse"]
        name: str = item["name"]
        category: str = item["metadata"]["category"]

        albedo = albedo.convert("RGB")
        diffuse = diffuse.convert("RGB")

        if is_black_image(diffuse):
            without_diffuse.append(idx)
            print(f"{split_name} {category}: Skipping {name} (black diffuse)")
            continue

        all_valid_indexes.append(idx)
        if is_similar_images(diffuse, albedo):
            same_diffuse_albedo_indexes.append(idx)
            print(f"{split_name} {category}: {name} (similar diffuse and albedo)")
            continue

        final_train_indexes.append(idx)
        print(f"{split_name} {category}: {name} (valid pair)")
    return {
        "all_valid_indexes": all_valid_indexes,
        "without_diffuse": without_diffuse,
        "same_diffuse_albedo_indexes": same_diffuse_albedo_indexes,
        "final_train_indexes": final_train_indexes,
    }


results = {}
for split in ["train", "test"]:
    results[split] = process_split(split)

with open("../matsynth_sample_check.json", "w") as f:
    json.dump(results, f, indent=4)
