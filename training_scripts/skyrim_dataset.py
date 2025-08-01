import seed
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import random
from torch.utils.data import Dataset
from typing import Callable, Optional
from class_materials import CLASS_PALETTE


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """
    Convert a PIL mask image to a tensor of class indices.
    """
    mask_np = np.array(mask)
    # Create H×W array with class indices, default to 255 for unknown colors
    class_mask = np.full(mask_np.shape[:2], 255, dtype=np.uint8)

    for class_idx, color in CLASS_PALETTE.items():
        color_match = np.all(mask_np == color, axis=-1)
        # ceramic class has been merged into stone
        if class_idx == 6:
            class_idx = 4
        class_mask[color_match] = class_idx

    return torch.from_numpy(class_mask).to(torch.int64)


class SkyrimDataset(Dataset):

    def __init__(
        self,
        skyrim_dir: str,
        data_file: str,
        split: str = "train",
        # skip_init=False,
    ):
        self.skyrim_input_dir = os.path.join(skyrim_dir)
        self.transform: Optional[Callable] = None
        self.split = split

        train_data = json.load(open(data_file, "r"))
        # self.train_dataset = data["train"]
        # self.val_dataset = data["val"]

        self.all_train_samples = []
        self.all_validation_samples = []

        # if skip_init:
        #     return

        glob = "**/*_mask.png"

        for sample in Path(self.skyrim_input_dir).glob(glob):
            relative_path = str(sample.relative_to(self.skyrim_input_dir))

            if (
                relative_path not in train_data["train"]
                and relative_path not in train_data["val"]
            ):
                # print(f"Skipping sample {relative_path} not in train or val set")
                continue

            dataset = (
                self.all_train_samples
                if relative_path in train_data["train"]
                else self.all_validation_samples
            )

            # Use relative path for sample name
            rel_base_name = (
                str(sample.parent.relative_to(self.skyrim_input_dir))
                .replace("\\", "_")
                .replace("/", "_")
                .replace(" ", "_")
            )
            base_name = sample.name.replace("_mask.png", "")
            name = rel_base_name + "_" + base_name
            path = sample.absolute()

            # parallax_path = path.with_name(base_name + "_parallax.png")
            # if ignore_without_parallax and not parallax_path.exists():
            #     continue

            sample = {
                "source": "skyrim",
                "name": name,
                "mask_relative_path": relative_path,
                "pbr": path.with_name(base_name + "_basecolor.png").exists(),
                "diffuse": str(path.with_name(base_name + "_diffuse.png")),
                "basecolor": str(path.with_name(base_name + "_basecolor.png")),
                "normal": str(path.with_name(base_name + "_normal.png")),
                "ao": str(path.with_name(base_name + "_ao.png")),
                "parallax": str(path.with_name(base_name + "_parallax.png")),
                "metallic": str(path.with_name(base_name + "_metallic.png")),
                "roughness": str(path.with_name(base_name + "_roughness.png")),
                "poisson_blur": str(path.with_name(base_name + "_poisson_blur.png")),
                "mask": path.with_name(base_name + "_mask.png"),
            }
            dataset.append(sample)

    def set_transform(self, transform: Callable) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return (
            self.split == "train"
            and len(self.all_train_samples)
            or len(self.all_validation_samples)
        )

    def get_specific_sample_for_relative_mask(
        self, name: str
    ) -> dict[str, Image.Image]:
        samples = (
            self.split == "train"
            and self.all_train_samples
            or self.all_validation_samples
        )

        for sample in samples:
            if sample["mask_relative_path"] == name:
                return self._process_sample(sample, call_tansform=False)

        raise ValueError(f"Sample with name {name} not found")

    def get_specific_sample(
        self, idx: int, call_transform=False
    ) -> dict[str, Image.Image]:
        samples = (
            self.split == "train"
            and self.all_train_samples
            or self.all_validation_samples
        )

        return self._process_sample(samples[idx], call_tansform=call_transform)

    def _process_sample(self, sample, call_tansform: bool) -> dict[str, Image.Image]:
        # Clone sample to avoid modifying the original
        sample = sample.copy()
        sample["diffuse"] = Image.open(sample["diffuse"]).convert("RGB")

        sample["normal"] = Image.open(sample["normal"]).convert("RGB")

        sample["mask"] = Image.open(sample["mask"]).convert("RGB")
        # Convert mask RGB colors to class indices
        # mask_np = np.array(mask)

        # # Create H×W array with class indices, default to 255 for unknown colors
        # class_mask = np.full(mask_np.shape[:2], 255, dtype=np.uint8)

        # # For each class, find matching pixels and set their index
        # for class_idx, color in CLASS_PALETTE.items():
        #     color_match = np.all(mask_np == color, axis=-1)
        #     class_mask[color_match] = class_idx

        # sample["mask"] = torch.from_numpy(class_mask)

        if Path(sample["basecolor"]).exists():
            sample["basecolor"] = Image.open(sample["basecolor"]).convert("RGB")
        else:
            sample["basecolor"] = None

        if Path(sample["ao"]).exists():
            sample["ao"] = Image.open(sample["ao"]).convert("L")
        else:
            sample["ao"] = None

        if Path(sample["parallax"]).exists():
            sample["parallax"] = Image.open(sample["parallax"]).convert("L")
        else:
            sample["parallax"] = None

        if Path(sample["metallic"]).exists():
            sample["metallic"] = Image.open(sample["metallic"]).convert("L")
        else:
            sample["metallic"] = None

        if Path(sample["roughness"]).exists():
            sample["roughness"] = Image.open(sample["roughness"]).convert("L")
        else:
            sample["roughness"] = None

        if Path(sample["poisson_blur"]).exists():
            # Load the Poisson blur as a numpy array
            sample["poisson_blur"] = Image.open(sample["poisson_blur"]).convert("I;16")
        else:
            sample["poisson_blur"] = None

        # sample["specular"] = Image.open(sample["specular"]).convert("RGB")

        if self.transform and call_tansform:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx: int) -> dict[str, Image.Image]:
        samples = (
            self.split == "train"
            and self.all_train_samples
            or self.all_validation_samples
        )

        sample = samples[idx]
        sample = self._process_sample(sample, call_tansform=True)

        return sample
