from datasets import load_dataset

from PIL import Image
from pathlib import Path
import json

with open("matsynth_final_indexes.json", "r") as f:
    index_data = json.load(f)

# Load the dataset
split = "train"
ds = load_dataset("gvecchio/MatSynth", split=split, num_proc=6, streaming=False)
ds.select_columns(["name", "basecolor", "category"])  # type: ignore

EXTRACT_CATEGORIES = ["ground", "leather", "metal"]


category_names = ds.features["category"].names  # type: ignore
for item in ds:
    name = item["name"]  # type: ignore
    cat = item["category"]  # type: ignore
    cat_name = category_names[cat]
    if cat_name not in EXTRACT_CATEGORIES:
        continue

    # We have already this, skip
    if index_data["new_category_mapping"].get(name) is not None:
        continue

    out_dir = Path(f"matsynth_category_samples/{cat_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    basecolor: Image.Image = item["basecolor"].convert("RGB")  # type: ignore
    basecolor.resize((2048, 2048))
    print(f"Saving {name} in {cat_name} category")
    basecolor.save(out_dir / f"{item["name"]}.png")  # type: ignore
