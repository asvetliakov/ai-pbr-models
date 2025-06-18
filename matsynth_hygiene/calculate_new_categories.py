"""
This script generates a mapping of sample names to their categories
Intended use is to remap MatSynth samples into new categories (supervised by manual review) and drop unnecessary categories & samples.
During the training from the Matsynth dataset, we will use this mapping to filter and organize the samples into new categories based on manual review.

Any sample from MatSynth not present in this mapping must be ignored during training.
Any sample from MatSynth with different category must be remapped to the new category specified in this mapping.
"""

from pathlib import Path

category_map = {}

for path in Path("matsynth_category_samples").rglob("*/*.png"):
    sample_name = path.stem
    sample_category = path.parent.name

    if sample_category not in category_map:
        category_map[sample_category] = []

    category_map[sample_category].append(sample_name)

with open("matsynth_category_mapping.json", "w") as f:
    import json

    json.dump(category_map, f, indent=4)
