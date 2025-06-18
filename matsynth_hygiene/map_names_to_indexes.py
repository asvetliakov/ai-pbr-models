import json
from datasets import load_dataset

ds = load_dataset(
    "gvecchio/MatSynth",
    split="train",
    streaming=False,
    num_proc=6,
)

ds = ds.select_columns(["name", "category"])

# Get MatSynth category label mapping
category_names = ds.features["category"].names  # type: ignore

data = {
    "name_to_index": {},
    "index_to_name": {},
}

for idx, item in enumerate(ds):
    name = item["name"]
    print(f"{idx}: {name}")
    data["name_to_index"][name] = idx
    data["index_to_name"][idx] = name

with open("matsynth_name_index_map.json", "w") as f:
    json.dump(data, f, indent=4)
