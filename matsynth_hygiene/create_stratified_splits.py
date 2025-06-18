import json
import random

with open("./matsynth_final_indexes.json", "r") as f:
    index_data = json.load(f)

with open("./matsynth_name_index_map.json", "r") as f:
    # Load my new name to index mapping
    name_index_map = json.load(f)


# Sort names by categories
categories = {}
for name, category in index_data["new_category_mapping"].items():
    if category not in categories:
        categories[category] = []

    categories[category].append(name)

train_indexes, train_names, train_a0_indexes, train_a0_names, val_indexes, val_names = (
    [],
    [],
    [],
    [],
    [],
    [],
)

for category, names in categories.items():
    print(f"Processing category: {category} with {len(names)} items")
    random.shuffle(names)
    n = len(names)
    # 5% for validation
    n_val = max(1, int(0.05 * n))

    val_names += names[:n_val]
    cat_train = names[n_val:]
    train_names += cat_train
    # Take 144 samples for train_a_0
    train_a0_names += cat_train[:144]

val_indexes += [name_index_map["name_to_index"][name] for name in val_names]
train_indexes += [name_index_map["name_to_index"][name] for name in train_names]
train_a0_indexes += [name_index_map["name_to_index"][name] for name in train_a0_names]

final_data = {
    "train": {
        "indexes": train_indexes,
        "names": train_names,
    },
    "train_a_0": {
        "indexes": train_a0_indexes,
        "names": train_a0_names,
    },
    "validation": {
        "indexes": val_indexes,
        "names": val_names,
    },
}

with open("matsynth_stratified_splits.json", "w") as f:
    json.dump(final_data, f, indent=4)
