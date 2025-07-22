import json
from pathlib import Path
from PIL import Image
import numpy as np
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = (BASE_DIR / "../skyrim_processed").resolve()

json_path = BASE_DIR / "../skyrim_data_segformer.json"

CLASS_LIST = [
    "fabric",
    "ground",
    "leather",
    "metal",
    "stone",
    "wood",
    # "ceramic",
]

CLASS_PALETTE = {
    0: (216, 27, 96),  # fabric, Raspberry
    1: (139, 195, 74),  # ground, Olive Green
    2: (141, 110, 99),  # leather, Saddle Brown
    3: (0, 145, 233),  # metal, Blue
    4: (120, 144, 156),  # stone, Slate Gray
    5: (229, 115, 115),  # wood, Burnt Sienna
    6: (255, 198, 138),  # ceramic, Pale Orange -> Merged into stone
}

# BACKGROUND_COLOR = (0, 0, 0)

MIN_VAL_SAMPLES_PER_CLASS = 12

# MINORITY_CLASSES_IDX = [CLASS_LIST.index("ceramic"), CLASS_LIST.index("leather")]
MINORITY_CLASSES_IDX = [CLASS_LIST.index("fabric"), CLASS_LIST.index("leather")]

# existing_data = json.load(open(json_path, "r"))


def process_mask(path):
    mask_img = Image.open(path).convert("RGB")
    is_square = mask_img.width == mask_img.height
    mask_array = np.array(mask_img)
    mask_img.close()
    class_counts = {name: 0 for name in CLASS_LIST}
    total = 0
    for idx, color in CLASS_PALETTE.items():
        mask_color = np.all(mask_array == color, axis=-1)
        count = np.sum(mask_color)
        if idx == 6:
            idx = 4  # ceramic merged into stone

        class_counts[CLASS_LIST[idx]] += count
        total += count
    return class_counts, total, path, is_square


def minority_tile_index(material, mask_path, size=256, stride=128):
    TAU_256 = 0.03  # ≥ 3 % ceramic pixels inside a 256×256 tile
    TAU_512 = 0.015  # scale by 256/512
    TAU_768 = 0.010  # scale by 256/768
    TAU = TAU_256 if size == 256 else TAU_512 if size == 512 else TAU_768

    mask_img = Image.open(mask_path).convert("RGB")
    mask_array = np.array(mask_img)
    mask_img.close()

    all_indexes = []

    for y in range(0, mask_array.shape[0] - size + 1, stride):
        for x in range(0, mask_array.shape[1] - size + 1, stride):
            tile = mask_array[y : y + size, x : x + size]
            class_counts = {name: 0 for name in CLASS_LIST}
            total = 0

            for idx, color in CLASS_PALETTE.items():
                mask_color = np.all(tile == color, axis=-1)
                count = np.sum(mask_color)
                if idx == 6:
                    idx = 4  # ceramic merged into stone
                class_counts[CLASS_LIST[idx]] += count
                total += count

            frac = class_counts[material] / total if total > 0 else 0
            if frac >= TAU and class_counts[material] >= 128:
                all_indexes.append((x, y))

    return all_indexes


def main():
    masks = list(INPUT_DIR.glob("**/*_mask.png"))
    random.shuffle(masks)
    pixels_per_class_all = {name: 0 for name in CLASS_LIST}
    # List of relative paths for samples per class
    samples_per_class = {name: [] for name in CLASS_LIST}
    sample_stats = {}
    total_pixels_all = 0

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_mask, path): path for path in masks}
        for future in as_completed(futures):
            class_counts, total, path, is_square = future.result()
            relative_path = str(path.relative_to(INPUT_DIR))
            sample_stats[relative_path] = {
                "class_counts": class_counts,
                "total": total,
                "is_square": is_square,
            }

            for class_name, count in class_counts.items():
                pixels_per_class_all[class_name] += count
                ratio = count / total if total > 0 else 0
                if ratio > 0.40:
                    samples_per_class[class_name].append(relative_path)

            total_pixels_all += total

    mask_paths = [str(p.relative_to(INPUT_DIR)) for p in masks]
    random.shuffle(mask_paths)

    print("Pixels per class (All):")
    for class_name, count in pixels_per_class_all.items():
        percentage = (count / total_pixels_all) * 100 if total_pixels_all > 0 else 0
        print(f"{class_name}: {count} pixels ({percentage:.4f}%)")

    print("\nSamples per class with >40% ratio:")
    for class_name, samples in samples_per_class.items():
        random.shuffle(samples)
        print(f"{class_name}: {len(samples)} samples")

    # Create stratified validation set
    val_set = set()
    for class_name, samples in samples_per_class.items():
        # Pick only square samples for validation
        square_samples = [s for s in samples if sample_stats[s]["is_square"]]
        chosen = random.sample(
            square_samples, min(MIN_VAL_SAMPLES_PER_CLASS, len(square_samples))
        )
        val_set.update(chosen)

    # Fill to 10% of the dataset
    remaining = [i for i in mask_paths if i not in val_set]
    square_remaining = [s for s in remaining if sample_stats[s]["is_square"]]
    target_size = int(0.1 * len(mask_paths))
    val_set.update(random.sample(square_remaining, max(0, target_size - len(val_set))))

    val_dataset = list(val_set)
    train_dataset = [i for i in mask_paths if i not in val_set]

    # train_dataset = existing_data["train"]
    # val_dataset = existing_data["val"]

    print(f"\nValidation set size: {len(val_dataset)}")
    print(f"Training set size: {len(train_dataset)}")

    # Calculate class weights for training dataset
    pixels_per_class_train = {name: 0 for name in CLASS_LIST}
    total_pixels_train = 0
    for sample in train_dataset:
        class_counts = sample_stats[sample]["class_counts"]
        for class_name, count in class_counts.items():
            pixels_per_class_train[class_name] += count
        total_pixels_train += sample_stats[sample]["total"]

    frequency = [
        pixels_per_class_train[name] / total_pixels_train for name in CLASS_LIST
    ]
    # Inverse root weighting
    weights_raw = [1.0 / math.sqrt(freq + 1e-12) for freq in frequency]
    # Normalize & cap
    mean_weight = sum(weights_raw) / len(weights_raw)
    weights = [min(w, 10) / mean_weight for w in weights_raw]

    print("\nClass weights for training dataset:" f"\n{dict(zip(CLASS_LIST, weights))}")

    # Calculate sample weights for training dataset
    sample_weights = []
    minority_sums = []
    for sample in train_dataset:
        class_counts = sample_stats[sample]["class_counts"]
        total_count = sample_stats[sample]["total"]
        # minority_sum = class_counts["ceramic"] + class_counts["leather"]
        minority_sum = class_counts["fabric"] + class_counts["leather"]
        minority_fraction = minority_sum / total_count if total_count > 0 else 0

        minority_sums.append(minority_fraction)

        weight = max((minority_fraction + 1e-6) ** 0.4, 0.3)
        sample_weights.append(weight)

        # weight = math.sqrt(minority_fraction + 1e-6)
        # sample_weights.append(weight)

    sample_weights = np.array(sample_weights, dtype=np.float32)
    sample_weights /= sample_weights.mean()  # Normalize to mean=1

    sample_weights = sample_weights.tolist()

    print(sum(sample_weights))
    print("Min/Max weight:", min(sample_weights), max(sample_weights))
    print(
        "Oversample factor for heaviest image:",
        max(sample_weights) / min(sample_weights),
    )

    # ceramic_crops = {}
    # total_256 = 0
    # total_512 = 0
    # total_768 = 0

    # # Build list of crop "windows" for ceramic
    # for sample in samples_per_class["ceramic"]:
    #     # Skips validation samples
    #     if sample in val_dataset:
    #         print(f"Skipping validation sample for ceramic biased crop: {sample}")
    #         continue
    #     stats = sample_stats[sample]

    #     # Skip homogeneous samples
    #     # if stats["class_counts"]["ceramic"] == stats["total"]:
    #     #     continue

    #     path = INPUT_DIR / sample
    #     indexes_256 = minority_tile_index(path, size=256, stride=128)
    #     indexes_512 = minority_tile_index(path, size=512, stride=256)
    #     indexes_768 = minority_tile_index(path, size=768, stride=256)

    #     ceramic_crops[sample] = {
    #         "256": indexes_256,
    #         "512": indexes_512,
    #         "768": indexes_768,
    #     }
    #     total_256 += len(indexes_256)
    #     total_512 += len(indexes_512)
    #     total_768 += len(indexes_768)

    # all_ceramic_crops_256 = [
    #     (sample, x, y)
    #     for sample, indexes in ceramic_crops.items()
    #     for x, y in indexes["256"]
    # ]

    # all_ceramic_crops_512 = [
    #     (sample, x, y)
    #     for sample, indexes in ceramic_crops.items()
    #     for x, y in indexes["512"]
    # ]

    # all_ceramic_crops_768 = [
    #     (sample, x, y)
    #     for sample, indexes in ceramic_crops.items()
    #     for x, y in indexes["768"]
    # ]

    # print(
    #     f"\nCeramic crops found: 256x256: {total_256}, 512x512: {total_512}, 768x768: {total_768}"
    # )
    # leather_crops = {}
    # total_256 = 0
    # total_512 = 0
    # total_768 = 0

    # # Build list of crop "windows" for leather
    # for sample in samples_per_class["leather"]:
    #     # Skips validation samples
    #     if sample in val_dataset:
    #         print(f"Skipping validation sample for leather biased crop: {sample}")
    #         continue
    #     stats = sample_stats[sample]

    #     # Skip homogeneous samples
    #     # if stats["class_counts"]["leather"] == stats["total"]:
    #     #     continue

    #     path = INPUT_DIR / sample
    #     indexes_256 = minority_tile_index("leather", path, size=256, stride=128)
    #     indexes_512 = minority_tile_index("leather", path, size=512, stride=256)
    #     indexes_768 = minority_tile_index("leather", path, size=768, stride=256)

    #     leather_crops[sample] = {
    #         "256": indexes_256,
    #         "512": indexes_512,
    #         "768": indexes_768,
    #     }
    #     total_256 += len(indexes_256)
    #     total_512 += len(indexes_512)
    #     total_768 += len(indexes_768)

    # all_leather_crops_256 = [
    #     (sample, x, y)
    #     for sample, indexes in leather_crops.items()
    #     for x, y in indexes["256"]
    # ]

    # all_leather_crops_512 = [
    #     (sample, x, y)
    #     for sample, indexes in leather_crops.items()
    #     for x, y in indexes["512"]
    # ]

    # all_leather_crops_768 = [
    #     (sample, x, y)
    #     for sample, indexes in leather_crops.items()
    #     for x, y in indexes["768"]
    # ]

    # print(
    #     f"\nleather crops found: 256x256: {total_256}, 512x512: {total_512}, 768x768: {total_768}"
    # )

    final_data = {
        "train": train_dataset,
        "val": val_dataset,
        "class_weights": weights,
        "sample_weights": sample_weights,
        # "ceramic_crops_256": all_ceramic_crops_256,
        # "ceramic_crops_512": all_ceramic_crops_512,
        # "ceramic_crops_768": all_ceramic_crops_768,
        # "leather_crops_256": all_leather_crops_256,
        # "leather_crops_512": all_leather_crops_512,
        # "leather_crops_768": all_leather_crops_768,
    }

    with open(json_path, "w") as f:
        json.dump(final_data, f, indent=4)


if __name__ == "__main__":
    main()
