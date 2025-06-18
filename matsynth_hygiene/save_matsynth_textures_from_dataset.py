"""
This script processes the parsed information from matsynth_samples_check.json which contains the possible black diffuse images and same albedo and diffuse images,
then extracts the corresponding images from the MatSynth dataset and saves them into a structured directory format for further manual review.
It also saves 100 samples from each index category (all_valid_indexes, same_diffuse_albedo_indexes, final_train_indexes, without_diffuse) for manual confirmation.
Intended use was to validate automated check for black diffuse images and same albedo and diffuse images.
"""

import json
from datasets import load_dataset, Dataset
import random
from PIL import Image
from pathlib import Path

with open("../matsynth_samples_check.json", "r") as f:
    data = json.load(f)


def process_split(split_name: str, indexes: list[int], indexes_name: str = "indexes"):

    ds: Dataset = load_dataset(
        "gvecchio/MatSynth",
        split=split_name,
        streaming=False,
        num_proc=4,
    )  # type: ignore

    indexed_ds = ds.select(indexes)

    for item in indexed_ds:
        albedo: Image.Image = item["basecolor"]  # type: ignore
        diffuse: Image.Image = item["diffuse"]  # type: ignore
        name: str = item["name"]  # type: ignore

        albedo = albedo.convert("RGB").resize([1024, 1024])
        diffuse = diffuse.convert("RGB").resize([1024, 1024])
        combined = Image.new("RGB", (albedo.width + diffuse.width, albedo.height))
        combined.paste(albedo, (0, 0))
        combined.paste(diffuse, (albedo.width, 0))

        Path(f"matsynth_samples/{split_name}/{indexes_name}").mkdir(
            exist_ok=True, parents=True
        )
        # Example: save the combined image
        combined.save(f"matsynth_samples/{split_name}/{indexes_name}/{name}.png")


results = {}
for split in ["train", "test"]:
    all_valid_indexes = data[split]["all_valid_indexes"]
    random.shuffle(all_valid_indexes)

    same_diffuse_albedo_indexes = data[split]["same_diffuse_albedo_indexes"]
    random.shuffle(same_diffuse_albedo_indexes)

    final_train_indexes = data[split]["final_train_indexes"]
    random.shuffle(final_train_indexes)

    without_diffuse = data[split]["without_diffuse"]

    print("All valid indexes:", len(all_valid_indexes))
    print("Same diffuse and albedo indexes:", len(same_diffuse_albedo_indexes))
    print("Final train indexes:", len(final_train_indexes))
    print("Without diffuse:", len(without_diffuse))

    results[split] = process_split(split, without_diffuse, "without_diffuse")
    results[split] = process_split(
        split, same_diffuse_albedo_indexes[:100], "same_diffuse_albedo_indexes"
    )
    results[split] = process_split(split, all_valid_indexes[:100], "all_valid_indexes")
    results[split] = process_split(
        split, final_train_indexes[:100], "final_train_indexes"
    )
