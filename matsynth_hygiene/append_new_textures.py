from pathlib import Path
import json

with open("./matsynth_final_indexes.json", "r") as f:
    final_index_data = json.load(f)

with open("./matsynth_name_index_map.json", "r") as f:
    name_index_map = json.load(f)

with open("./matsynth_samples_check.json", "r") as f:
    all_matsynth_sample_data = json.load(f)


for path in Path("matsynth_category_samples").rglob("*/*.png"):
    sample_name = path.stem
    sample_category = path.parent.name
    print(f"Processing {sample_name} in category {sample_category}")

    idx = name_index_map["name_to_index"].get(sample_name, None)
    if idx is None:
        print(f"!!!Sample {sample_name} not found in name_index_map, skipping.")
        continue

    if idx in all_matsynth_sample_data["train"]["without_diffuse"]:
        print(f"!!!Sample {sample_name} doesn't have diffuse texture, skipping.")
        continue

    is_same_diffuse_albedo = (
        idx in all_matsynth_sample_data["train"]["same_diffuse_albedo_indexes"]
    )
    print(f"Sample {sample_name} has same diffuse albedo: {is_same_diffuse_albedo}")

    final_index_data["all_valid_indexes"].append(idx)

    if is_same_diffuse_albedo:
        final_index_data["same_diffuse_albedo_indexes"].append(idx)
    else:
        final_index_data["different_diffuse_albedo"].append(idx)

    final_index_data["new_category_mapping"][sample_name] = sample_category

with open("./matsynth_final_indexes.json", "w") as f:
    json.dump(final_index_data, f, indent=4)
