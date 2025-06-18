import json
from datasets import load_dataset

with open("./matsynth_samples_check.json", "r") as f:
    # Load old existing index data
    index_data = json.load(f)

with open("./matsynth_category_mapping.json", "r") as f:
    # Load my new category mapping
    new_category_mapping = json.load(f)


ds = load_dataset(
    "gvecchio/MatSynth",
    split="train",
    streaming=False,
    num_proc=6,
)

ds = ds.select_columns(["name", "category"])

# Get MatSynth category label mapping
category_names = ds.features["category"].names  # type: ignore

name_to_category = {}

for category_name, names in new_category_mapping.items():
    for name in names:
        name_to_category[name] = category_name

print(name_to_category)

final_index_data = {
    # Copy without_diffuse for safekeeping
    "without_diffuse": index_data["train"]["without_diffuse"],
    "all_valid_indexes": [],
    "same_diffuse_albedo_indexes": [],
    "different_diffuse_albedo": [],
    "new_category_mapping": name_to_category,
}

for idx, item in enumerate(ds):
    name = item["name"]
    category_idx = item["category"]
    category_name = category_names[category_idx]

    # Check if we excluded it earlier (i.e. without diffuse)
    if idx not in index_data["train"]["all_valid_indexes"]:
        print(f"Skipping item {idx} ({name}) - not in all_valid_indexes")
        continue

    # Check if name has in any of the new category mappings
    if name not in name_to_category.keys():
        print(f"Skipping item {idx} ({name}) - not in new category mapping")
        continue

    final_index_data["all_valid_indexes"].append(idx)

    if idx in index_data["train"]["same_diffuse_albedo_indexes"]:
        final_index_data["same_diffuse_albedo_indexes"].append(idx)
        print(f"Item {idx} ({name}) - same diffuse and albedo")
    else:
        final_index_data["different_diffuse_albedo"].append(idx)
        print(f"Item {idx} ({name}) - different diffuse and albedo")


with open("./matsynth_final_indexes.json", "w") as f:
    json.dump(final_index_data, f, indent=4)
