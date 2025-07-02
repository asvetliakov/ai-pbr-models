import seed
import os
from PIL import Image
from pathlib import Path
import json
import torch
import random
from torch.utils.data import Dataset
from typing import Callable, Optional

BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE_PBR = (BASE_DIR / "skyrim_cache_pbr.json").resolve()
CACHE_FILE_ALL = (BASE_DIR / "skyrim_cache_all.json").resolve()


class SkyrimDataset(Dataset):

    def __init__(
        self,
        skyrim_dir: str,
        split: str = "train",
        load_non_pbr=False,
        skip_init=False,
    ):
        self.skyrim_input_dir = os.path.join(skyrim_dir)
        self.transform: Optional[Callable] = None
        self.split = split

        self.all_train_samples = []
        self.all_validation_samples = []

        if skip_init:
            return

        all_samples = []

        if load_non_pbr and CACHE_FILE_ALL.exists():
            with open(CACHE_FILE_ALL, "r") as f:
                all_samples = json.load(f)
        elif not load_non_pbr and CACHE_FILE_PBR.exists():
            with open(CACHE_FILE_PBR, "r") as f:
                all_samples = json.load(f)
        else:
            glob = "**/*_diffuse.png" if load_non_pbr else "**/*_basecolor.png"

            for sample in Path(self.skyrim_input_dir).glob(glob):
                # Use relative path for sample name
                rel_path = (
                    str(sample.parent.relative_to(self.skyrim_input_dir))
                    .replace("\\", "_")
                    .replace("/", "_")
                    .replace(" ", "_")
                )
                base_name = (
                    sample.stem.replace("_diffuse", "")
                    if load_non_pbr
                    else sample.stem.replace("_basecolor", "")
                )
                name = rel_path + "_" + base_name
                path = sample.absolute()
                sample = {
                    "source": "skyrim",
                    "name": name,
                    "pbr": path.with_name(base_name + "_basecolor.png").exists(),
                    "diffuse": str(path.with_name(base_name + "_diffuse.png")),
                    "basecolor": str(path.with_name(base_name + "_basecolor.png")),
                    "normal": str(path.with_name(base_name + "_normal.png")),
                    "ao": str(path.with_name(base_name + "_ao.png")),
                    "parallax": str(path.with_name(base_name + "_parallax.png")),
                    "metallic": str(path.with_name(base_name + "_metallic.png")),
                    "roughness": str(path.with_name(base_name + "_roughness.png")),
                }
                all_samples.append(sample)
            # Save the samples to cache
            if load_non_pbr:
                with open(CACHE_FILE_ALL, "w") as f:
                    json.dump(all_samples, f, indent=4)
            else:
                with open(CACHE_FILE_PBR, "w") as f:
                    json.dump(all_samples, f, indent=4)

        n_val = max(1, int(0.1 * len(all_samples)))
        random.shuffle(all_samples)
        self.all_validation_samples = all_samples[:n_val]
        self.all_train_samples = all_samples[n_val:]

    def set_transform(self, transform: Callable) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return (
            self.split == "train"
            and len(self.all_train_samples)
            or len(self.all_validation_samples)
        )

    def _process_sample(self, sample, call_tansform: bool) -> dict[str, torch.Tensor]:
        # Clone sample to avoid modifying the original
        sample = sample.copy()
        sample["diffuse"] = Image.open(sample["diffuse"]).convert("RGB")

        sample["normal"] = Image.open(sample["normal"]).convert("RGB")

        if sample["basecolor"].exists():
            sample["basecolor"] = Image.open(sample["basecolor"]).convert("RGB")
        else:
            sample["basecolor"] = None

        if sample["ao"].exists():
            sample["ao"] = Image.open(sample["ao"]).convert("L")
        else:
            sample["ao"] = None

        if sample["parallax"].exists():
            sample["parallax"] = Image.open(sample["parallax"]).convert("L")
        else:
            sample["parallax"] = None

        if sample["metallic"].exists():
            sample["metallic"] = Image.open(sample["metallic"]).convert("L")
        else:
            sample["metallic"] = None

        if sample["roughness"].exists():
            sample["roughness"] = Image.open(sample["roughness"]).convert("L")
        else:
            sample["roughness"] = None

        # sample["specular"] = Image.open(sample["specular"]).convert("RGB")

        if self.transform and call_tansform:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        samples = (
            self.split == "train"
            and self.all_train_samples
            or self.all_validation_samples
        )

        sample = samples[idx]
        sample = self._process_sample(sample, call_tansform=True)

        return sample
