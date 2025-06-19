import os
from PIL import Image
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, Tuple


def normalize_normal_map(normal: Image.Image) -> Image.Image:
    """
    Take a PIL normal map (RGB, 0–255), renormalize each pixel vector to length 1,
    and return a new PIL.Image in the same mode/range.
    (Does NOT resize or flip any channels.)
    """
    # to (H, W, 3) floats in [0,1]
    arr = np.array(normal, dtype=np.float32) / 255.0

    # map to [-1,1]
    vec = arr * 2.0 - 1.0

    # compute per-pixel length and normalize
    length = np.linalg.norm(vec, axis=2, keepdims=True)
    vec = vec / (length + 1e-6)

    # map back to [0,1]
    out = (vec + 1.0) * 0.5

    # to uint8 PIL
    out8 = (out * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out8, mode="RGB")


class SimpleImageDataset(Dataset):
    """
    A simple dataset for loading PBR images.
    """

    def __init__(
        self,
        matsynth_dir: str,
        split: str = "train",
        # Used for A0 phase, set to 144
        max_train_samples_per_cat: Optional[int] = None,
    ):
        """
        Args:
            matsynth_dir (str): Matsynth category directory
            transform (callable, optional): A function/transform to apply to PIL images.
        """
        self.matsynth_input_dir = os.path.join(matsynth_dir)
        self.transform: Optional[Callable] = None
        self.split = split

        # Sort train names by categories, need later for weighted sampler
        self.CLASS_LIST = [
            "ceramic",
            "fabric",
            "ground",
            "leather",
            "metal",
            "stone",
            "wood",
        ]
        self.CLASS_LIST_IDX_MAPPING = {
            name: idx for idx, name in enumerate(self.CLASS_LIST)
        }
        self.METAL_IDX = self.CLASS_LIST_IDX_MAPPING["metal"]

        self.all_train_samples = []
        self.all_validation_samples = []

        sample_names_per_category = {name: [] for name in self.CLASS_LIST}

        for metdata in Path(self.matsynth_input_dir).glob("**/*.json"):
            category_name = metdata.parent.name
            name = metdata.stem
            category_idx = self.CLASS_LIST_IDX_MAPPING.get(category_name, None)
            if category_idx is None:
                print(f"Warning: Category '{category_name}' not in CLASS_LIST.")

            sample_names_per_category[category_name].append(name)

        for category, names in sample_names_per_category.items():
            names = sorted(names)
            n = len(names)
            # 5% for validation
            n_val = max(1, int(0.05 * n))

            samples = list(
                map(
                    lambda name: {
                        "source": "matsynth",
                        "name": name,
                        "category_name": category,
                        "category": self.CLASS_LIST_IDX_MAPPING[category],
                        "metadata": f"{category}/{name}.json",
                        "ao": f"{category}/{name}_ao.png",
                        "basecolor": f"{category}/{name}_basecolor.png",
                        "diffuse": f"{category}/{name}_diffuse.png",
                        "height": f"{category}/{name}_height.png",
                        "metallic": f"{category}/{name}_metallic.png",
                        "normal": f"{category}/{name}_normal.png",
                        "roughness": f"{category}/{name}_roughness.png",
                        # "specular": f"{category}/{name}_specular.png",
                    },
                    names,
                )
            )

            # Stratified split
            validation_samples = samples[:n_val]
            train_samples = samples[n_val:]

            if (
                max_train_samples_per_cat is not None
                and len(train_samples) > max_train_samples_per_cat
            ):
                train_samples = train_samples[:max_train_samples_per_cat]

            self.all_validation_samples.extend(validation_samples)
            self.all_train_samples.extend(train_samples)

    def set_transform(self, transform: Callable) -> None:
        self.transform = transform

    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate class weights and sample weights for the dataset.
        """
        if self.split == "train":
            samples = self.all_train_samples
        else:
            samples = self.all_validation_samples

        num_classes = len(self.CLASS_LIST)
        all_labels = [sample["category"] for sample in samples]
        cls_counts = torch.bincount(torch.tensor(all_labels), minlength=num_classes)
        freq = cls_counts / cls_counts.sum()

        loss_weights = 1.0 / torch.sqrt(freq + 1e-6)  # avoid ÷0
        loss_weights *= num_classes / loss_weights.sum()

        sample_weighs_per_class = 1.0 / (cls_counts + 1e-6)
        sample_weights = sample_weighs_per_class[all_labels]

        return loss_weights, sample_weights

    def __len__(self) -> int:
        return (
            self.split == "train"
            and len(self.all_train_samples)
            or len(self.all_validation_samples)
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        samples = (
            self.split == "train"
            and self.all_train_samples
            or self.all_validation_samples
        )

        sample = samples[idx]

        # Clone sample to avoid modifying the original
        sample = sample.copy()
        sample["metadata"] = open(
            os.path.join(self.matsynth_input_dir, sample["metadata"]), "r"
        ).read()
        sample["ao"] = Image.open(
            os.path.join(self.matsynth_input_dir, sample["ao"])
        ).convert("L")
        sample["basecolor"] = Image.open(
            os.path.join(self.matsynth_input_dir, sample["basecolor"])
        ).convert("RGB")
        sample["diffuse"] = Image.open(
            os.path.join(self.matsynth_input_dir, sample["diffuse"])
        ).convert("RGB")
        sample["height"] = Image.open(
            os.path.join(self.matsynth_input_dir, sample["height"])
        ).convert("I;16")
        sample["metallic"] = Image.open(
            os.path.join(self.matsynth_input_dir, sample["metallic"])
        ).convert("L")
        sample["normal"] = Image.open(
            os.path.join(self.matsynth_input_dir, sample["normal"])
        ).convert("RGB")
        sample["roughness"] = Image.open(
            os.path.join(self.matsynth_input_dir, sample["roughness"])
        ).convert("L")
        # sample["specular"] = Image.open(
        #     os.path.join(self.matsynth_input_dir, sample["specular"])
        # ).convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        return sample
