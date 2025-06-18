"""
Extract all basecolor images from the MatSynth dataset and save them into category-specific directories. For manual review.
"""

from datasets import load_dataset
from PIL import Image
from pathlib import Path

# Load the dataset
split = "train"
ds = load_dataset("gvecchio/MatSynth", split=split, num_proc=6, streaming=False)
ds.select_columns(["name", "basecolor", "category"])  # type: ignore


# Get category label mapping
category_names = ds.features["category"].names  # type: ignore
for item in ds:
    cat = item["category"]  # type: ignore
    cat_name = category_names[cat]
    out_dir = Path(f"matsynth_category_samples/{cat_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    basecolor: Image.Image = item["basecolor"].convert("RGB")  # type: ignore
    basecolor.resize((2048, 2048))
    basecolor.save(out_dir / f"{item["name"]}.png")  # type: ignore
