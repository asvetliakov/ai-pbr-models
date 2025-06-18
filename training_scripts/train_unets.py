# Ensure we import it here to set random(seed)
import seed
import json, torch
import numpy as np
import random
import multiprocessing
import torch.nn.functional as F
import lpips
from unet_models import UNetAlbedo, UNetMaps
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from datasets import load_dataset, Dataset, ClassLabel
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from torchmetrics import functional as FM
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

# HYPER_PARAMETERS
BATCH_SIZE = 2  # Batch size for training
EPOCHS = 10  # Number of epochs to train
LR = 1e-4  # Learning rate for the optimizer
WD = 1e-2  # Weight decay for the optimizer
T_MAX = 10  # Max number of epochs for the learning rate scheduler
PHASE = "a0"  # Phase of the training per plan, used for logging and saving

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable for faster training on fixed input sizes

with open("./matsynth_final_indexes.json", "r") as f:
    dataset_index_data = json.load(f)

with open("./matsynth_stratified_splits.json", "r") as f:
    stratified_splits = json.load(f)

# Sort train names by categories, need later for weighted sampler
CLASS_LIST = [
    "ceramic",
    "fabric",
    "ground",
    "leather",
    "metal",
    "stone",
    "wood",
]
NONE_IDX = len(CLASS_LIST)  # Index for "none" category, used for safety
CLASS_LIST_IDX_MAPPING = {name: idx for idx, name in enumerate(CLASS_LIST)}
METAL_IDX = CLASS_LIST_IDX_MAPPING["metal"]

INPUT_IMAGE_SIZE = (2048, 2048)  # Input image size for training, used for resizing

all_labels = []
subset_names = stratified_splits["train_a_0"]["names"]
#  We need 1:1 label mapping in the same order as it appears in the dataset
for name in stratified_splits["train_a_0"]["names"]:
    # Get the category from the mapping
    category = dataset_index_data["new_category_mapping"].get(name, None)
    if category is not None:
        # Get the index of the category in CLASS_LIST
        label_idx = CLASS_LIST_IDX_MAPPING.get(category, None)
        if label_idx is not None:
            all_labels.append(label_idx)
        else:
            print(f"Warning: Category '{category}' not found in CLASS_LIST.")

# Calcualte weights
num_classes = len(CLASS_LIST)
cls_counts = torch.bincount(torch.tensor(all_labels), minlength=num_classes)
freq = cls_counts / cls_counts.sum()

print("Class counts:", cls_counts)
print("Class frequencies:", freq)

# ! DISABLED FOR PHASE A0, REENABLE FOR PHASE A+
# Sample weights for each class
# sample_weighs_per_class = 1.0 / (cls_counts + 1e-6)
# sample_weights = sample_weighs_per_class[all_labels]

# print("Sample weights per class:", sample_weighs_per_class)
# print("Sample weights:", sample_weights)

# # Will pull random samples according to the sample weights
# train_sampler = WeightedRandomSampler(
#     weights=sample_weights.tolist(),
#     num_samples=len(sample_weights),
#     replacement=True,
# )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet_alb = UNetAlbedo(
    in_ch=6,  # RGB + Normal
    cond_ch=256,  # Condition channel size, can be adjusted
).to(
    device
)  # type: ignore

unet_maps = UNetMaps(
    in_ch=6,  # RGB + Normal
    cond_ch=256,  # Condition channel size, can be adjusted
).to(
    device
)  # type: ignore

transform_train_input = T.Compose(
    [
        # Take random crop (consult augment table in plan)
        # T.RandomCrop(size=(256, 256)),
        # T.Resize((1024, 1024), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_STANDARD_MEAN,
            std=IMAGENET_STANDARD_STD,
        ),
    ]
)
transform_train_gt = T.Compose(
    [
        T.ToTensor(),
    ]
)


def transform_validation_input(
    image: Image.Image, interpolation: T.InterpolationMode
) -> torch.Tensor:
    composed = T.Compose(
        [
            # Need to use same crop size for validation
            T.CenterCrop(size=(256, 256)),
            T.Resize((1024, 1024), interpolation=interpolation),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_STANDARD_MEAN,
                std=IMAGENET_STANDARD_STD,
            ),
        ]
    )
    return composed(image)  # type: ignore


def transform_validation_gt(
    image: Image.Image, interpolation: T.InterpolationMode, tensor: bool = True
) -> torch.Tensor:
    funcs = [
        T.CenterCrop(size=(256, 256)),
        T.Resize((1024, 1024), interpolation=interpolation),
    ]
    if tensor:
        funcs.append(T.ToTensor())

    composed = T.Compose(funcs)
    return composed(image)  # type: ignore


def synced_crop_and_resize(
    diffuse: Image.Image,
    normal: Image.Image,
    albedo: Image.Image,
    height: Image.Image,
    metallic: Image.Image,
    roughness: Image.Image,
    ao: Image.Image,
    crop_size: tuple[int, int],
    resize_to: list[int],
) -> tuple[
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
]:
    """
    Crop and resize two images to the same size.
    """
    i, j, h, w = T.RandomCrop.get_params(diffuse, output_size=crop_size)  # type: ignore

    diffuse_crop = TF.crop(diffuse, i, j, h, w)  # type: ignore
    albedo_crop = TF.crop(albedo, i, j, h, w)  # type: ignore
    normal_crop = TF.crop(normal, i, j, h, w)  # type: ignore
    height_crop = TF.crop(height, i, j, h, w)  # type: ignore
    metallic_crop = TF.crop(metallic, i, j, h, w)  # type: ignore
    roughness_crop = TF.crop(roughness, i, j, h, w)  # type: ignore
    ao_crop = TF.crop(ao, i, j, h, w)  # type: ignore

    diffuse_resize = TF.resize(
        diffuse_crop, resize_to, interpolation=T.InterpolationMode.LANCZOS
    )
    albedo_resize = TF.resize(
        albedo_crop, resize_to, interpolation=T.InterpolationMode.LANCZOS
    )
    normal_resize = TF.resize(
        normal_crop, resize_to, interpolation=T.InterpolationMode.BILINEAR
    )
    height_resize = TF.resize(
        height_crop, resize_to, interpolation=T.InterpolationMode.BICUBIC
    )
    metallic_resize = TF.resize(
        metallic_crop, resize_to, interpolation=T.InterpolationMode.BILINEAR
    )
    roughness_resize = TF.resize(
        roughness_crop, resize_to, interpolation=T.InterpolationMode.BILINEAR
    )
    ao_resize = TF.resize(
        ao_crop, resize_to, interpolation=T.InterpolationMode.BILINEAR
    )

    # image not tensors
    return (
        diffuse_resize,
        normal_resize,
        albedo_resize,
        height_resize,
        metallic_resize,
        roughness_resize,
        ao_resize,
    )  # type: ignore


def convert_normal_to_directx_type(normal: Image.Image) -> Image.Image:
    """
    Convert normal map from OpenGL format  to DirectX format.
    OpenGL normal maps have the green channel inverted compared to DirectX.
    """
    np_img = np.array(normal, dtype=np.float32) / 255.0

    R = np_img[..., 0]  # Red channel
    G = np_img[..., 1]  # Green channel
    B = np_img[..., 2]  # Blue channel

    G = 1.0 - G  # Invert green channel for DirectX format

    converted = np.stack((R, G, B), axis=-1)  # Stack channels back together
    return Image.fromarray((converted * 255).astype(np.uint8))


def check_none_category(examples):
    """
    Check if there are any examples with "none" category and warn if so.
    This is used to prevent training on examples with "none" category.
    """
    if NONE_IDX in examples["category"]:
        print(
            "Warning: There are examples with 'none' category in the dataset. "
            "This may lead to incorrect training results."
        )
        # Print affected names
        none_names = [
            name
            for name, category in zip(examples["name"], examples["category"])
            if category == NONE_IDX
        ]
        print(f"Affected names: {none_names}")
        raise ValueError(
            f"Examples {none_names} has 'none' category, which is not allowed in training."
        )


def ensure_input_size(
    im: Image.Image, size: tuple[int, int], resample: Image.Resampling
):
    if im.size == size:
        return im
    return im.resize(size, resample)


def load_diffuse_and_ao(examples: dict) -> tuple[list[Image.Image], list[Image.Image]]:
    # Load from disk
    input_ao = []
    # Either load from disk (if exist) or use existing diffuse
    input_diffuse = []

    categories = examples["category"]
    names = examples["name"]
    diffuse = examples["diffuse"]

    for name, category_idx, diffuse in zip(names, categories, diffuse):
        category = CLASS_LIST[category_idx]
        ao_path = Path(f"./matsynth_processed/{category}/{name}_ao.png")
        ao_img = Image.open(ao_path).convert("L")  # Load AO map as grayscale
        ao_img = ensure_input_size(
            ao_img, INPUT_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
        )
        input_ao.append(ao_img)

        # ! Enable for Phase A+
        # Load generated diffuse if exist (many examples have same diffuse and albedo so we generated synthetic for them)
        # diffuse_path = Path(
        #     f"./matsynth_processed/{category}/{example['name']}_diffuse.png"
        # )
        # if diffuse_path.exists():
        #     # If synthetic diffuse map exists, load it
        #     diffuse_image = Image.open(diffuse_path).convert("RGB")
        #     transformed_diffuse.append(diffuse_image)
        # else:
        diffuse = ensure_input_size(
            diffuse.convert("RGB"), INPUT_IMAGE_SIZE, resample=Image.Resampling.LANCZOS
        )
        input_diffuse.append(diffuse)

    return input_diffuse, input_ao


def make_full_image_mask(category_id: int, img_size: tuple[int, int]) -> torch.Tensor:
    """
    Build a segmentation mask of shape (H, W) where every pixel = category_id.
    """
    H, W = img_size
    # numpy array filled with your class index
    mask_np = np.full((H, W), fill_value=category_id, dtype=np.int64)
    # convert to torch LongTensor
    return torch.from_numpy(mask_np)


def transform_train_fn(examples):
    # Check if there any examples with "none" category and warn if so
    check_none_category(examples)

    # img: Image.Image = Image.open(examples["basecolor"][0])  # Use first image to get size
    # img.convert("RGB").resize((1024, 1024), resample=Image.Resampling.LANCZOS)  # Resize to 1024x1024 for training

    input_albedo = [
        ensure_input_size(
            image.convert("RGB"), INPUT_IMAGE_SIZE, resample=Image.Resampling.LANCZOS
        )
        for image in examples["basecolor"]
    ]
    input_normal = [
        ensure_input_size(
            image.convert("RGB"), INPUT_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
        )
        for image in examples["normal"]
    ]
    input_height = [
        ensure_input_size(
            image.convert("I;16"), INPUT_IMAGE_SIZE, resample=Image.Resampling.BICUBIC
        )
        for image in examples["height"]
    ]
    input_metallic = [
        ensure_input_size(
            image.convert("L"), INPUT_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
        )
        for image in examples["metallic"]
    ]
    input_roughness = [
        ensure_input_size(
            image.convert("L"), INPUT_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
        )
        for image in examples["roughness"]
    ]
    input_category = examples["category"]
    input_names = examples["name"]

    # Load diffuse and AO maps. AO is always loaded from disk, diffuse is either loaded from disk (when sample has same albedo = diffuse) or taken from the example (when different)
    input_diffuse, input_ao = load_diffuse_and_ao(examples)

    # For UNet-Albedo
    final_diffuse_and_normal = []
    final_albedo = []
    final_hieght = []
    final_metallic = []
    final_roughness = []
    final_ao = []
    final_normal = []

    final_masks = []

    for diffuse, normal, albedo, height, metallic, roughness, ao, category in zip(
        input_diffuse,
        input_normal,
        input_albedo,
        input_height,
        input_metallic,
        input_roughness,
        input_ao,
        input_category,
    ):
        diffuse, normal, albedo, height, metallic, roughness, ao = (
            synced_crop_and_resize(
                diffuse,
                normal,
                albedo,
                height,
                metallic,
                roughness,
                ao,
                crop_size=(256, 256),  # Crop size for training
                resize_to=[1024, 1024],  # Resize to 1024x1024 for training
            )
        )

        final_masks.append(make_full_image_mask(category, img_size=(1024, 1024)))

        diffuse = transform_train_input(diffuse)
        # MatSynth dataset uses OpenGL normal maps, we need to convert them to DirectX format
        # Using it here to avoid converting 4k images
        normal = transform_train_input(convert_normal_to_directx_type(normal))
        # Concatenate albedo and normal along the channel dimension
        final_diffuse_and_normal.append(torch.cat((diffuse, normal), dim=0))  # type: ignore
        final_normal.append(normal)

        albedo = transform_train_gt(albedo)
        final_albedo.append(albedo)

        # ToTensor() is normalizing 8 bit images ( / 255 ) so for 16bit we need to do it manually
        height_arr = np.array(height, dtype=np.uint16)
        height_arr = height_arr.astype(np.float32) / 65535.0  # Normalize to [0, 1]
        height_tensor = torch.from_numpy(height_arr).unsqueeze(0)
        final_hieght.append(height_tensor)

        metallic = transform_train_gt(metallic)
        final_metallic.append(metallic)

        roughness = transform_train_gt(roughness)
        final_roughness.append(roughness)

        ao = transform_train_gt(ao)
        final_ao.append(ao)

    return {
        "diffuse_and_normal": final_diffuse_and_normal,
        "height": final_hieght,
        "albedo": final_albedo,
        "normal": final_normal,
        "metallic": final_metallic,
        "roughness": final_roughness,
        "ao": final_ao,
        "masks": torch.stack(
            final_masks, dim=0
        ),  # Concatenate masks along batch dimension
        "category": examples["category"],  # keep for reference
        "name": input_names,  # keep for reference
    }


def transform_val_fn(examples):
    # Check if there any examples with "none" category and warn if so
    check_none_category(examples)

    input_albedo = [
        ensure_input_size(
            image.convert("RGB"), INPUT_IMAGE_SIZE, resample=Image.Resampling.LANCZOS
        )
        for image in examples["basecolor"]
    ]
    input_normal = [
        ensure_input_size(
            image.convert("RGB"), INPUT_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
        )
        for image in examples["normal"]
    ]
    input_height = [
        ensure_input_size(
            image.convert("I;16"), INPUT_IMAGE_SIZE, resample=Image.Resampling.BICUBIC
        )
        for image in examples["height"]
    ]
    input_metallic = [
        ensure_input_size(
            image.convert("L"), INPUT_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
        )
        for image in examples["metallic"]
    ]
    input_roughness = [
        ensure_input_size(
            image.convert("L"), INPUT_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
        )
        for image in examples["roughness"]
    ]
    input_category = examples["category"]
    input_names = examples["name"]

    # Load diffuse and AO maps. AO is always loaded from disk, diffuse is either loaded from disk (when sample has same albedo = diffuse) or taken from the example (when different)
    input_diffuse, input_ao = load_diffuse_and_ao(examples)

    # For UNet-Albedo
    final_diffuse_and_normal = []
    final_albedo = []
    final_height = []
    final_metallic = []
    final_roughness = []
    final_ao = []
    final_masks = []
    final_normal = []

    for diffuse, normal, albedo_gt, height, metallic, roughness, ao, category in zip(
        input_diffuse,
        input_normal,
        input_albedo,
        input_height,
        input_metallic,
        input_roughness,
        input_ao,
        input_category,
    ):
        final_masks.append(make_full_image_mask(category, img_size=(1024, 1024)))

        diffuse = transform_validation_input(diffuse, T.InterpolationMode.LANCZOS)
        normal = transform_validation_input(
            convert_normal_to_directx_type(normal), T.InterpolationMode.BILINEAR
        )
        final_diffuse_and_normal.append(
            torch.cat((diffuse, normal), dim=0)  # type: ignore
        )
        final_normal.append(normal)

        albedo = transform_validation_gt(albedo_gt, T.InterpolationMode.LANCZOS)
        final_albedo.append(albedo)

        height: Image.Image = transform_validation_gt(height, T.InterpolationMode.BICUBIC, False)  # type: ignore
        height_arr = np.array(height, dtype=np.uint16)
        height_arr = height_arr.astype(np.float32) / 65535.0  # Normalize to [0, 1]
        height_tensor = torch.from_numpy(height_arr).unsqueeze(0)
        final_height.append(height_tensor)

        metallic = transform_validation_gt(metallic, T.InterpolationMode.BILINEAR)
        final_metallic.append(metallic)

        roughness = transform_validation_gt(roughness, T.InterpolationMode.BILINEAR)
        final_roughness.append(roughness)

        ao = transform_validation_gt(ao, T.InterpolationMode.BILINEAR)
        final_ao.append(ao)

    return {
        "diffuse_and_normal": final_diffuse_and_normal,
        "height": final_height,
        "albedo": final_albedo,
        "normal": final_normal,
        "metallic": final_metallic,
        "roughness": final_roughness,
        "ao": final_ao,
        "masks": torch.stack(
            final_masks, dim=0
        ),  # Concatenate masks along batch dimension
        "category": examples["category"],  # keep for reference
        "name": input_names,  # keep for reference
    }


def load_my_dataset() -> tuple[Dataset, Dataset]:
    # I have prepared specific dataset indexes so there shouldn't be actual none categories when training
    # But putting here "none" for safety
    CLASS_LIST_WITH_NONE = CLASS_LIST + ["none"]
    CLASS_LIST_IDX_MAPPING["none"] = NONE_IDX

    dataset: Dataset = load_dataset("gvecchio/MatSynth", split="train", streaming=False, num_proc=8)  # type: ignore
    # Process dataset to remmap categories, use temp_ds with only name and category to avoid loading images
    temp_ds = dataset.select_columns(["name", "category"])
    temp_ds = temp_ds.map(
        lambda item: {
            "name": item["name"],
            "category": CLASS_LIST_IDX_MAPPING.get(
                dataset_index_data["new_category_mapping"].get(item["name"], "none"),
                NONE_IDX,
            ),
        },
    )

    # For SegFormer we need only the basecolor(albedo) and category
    dataset = dataset.select_columns(
        ["name", "basecolor", "diffuse", "normal", "height", "metallic", "roughness"]
    )

    # Add our category mapping to the dataset
    dataset = dataset.add_column(
        name="category",
        column=temp_ds["category"],
        feature=ClassLabel(
            names=CLASS_LIST_WITH_NONE,
            num_classes=len(CLASS_LIST_WITH_NONE),
        ),
        new_fingerprint="my_category_mapping_v2",
    )

    # Select our prepared indexes for train & val datasets
    # train_ds = dataset.select(stratified_splits["train"]["indexes"])
    # ! Phase A0 uses only 1000 samples for training
    random.shuffle(stratified_splits["train_a_0"]["indexes"])
    train_ds = dataset.select(stratified_splits["train_a_0"]["indexes"])

    val_ds = dataset.select(stratified_splits["validation"]["indexes"])

    train_ds.set_transform(transform_train_fn)
    val_ds.set_transform(transform_val_fn)

    return train_ds, val_ds


def masked_l1(pred, target, material_mask, w_fg=3.0, w_bg=1.0):
    """
    Loss re‑weighting
    Give pixels whose material matches the ground‑truth map name a higher weight (so the “metal” area influences metallic map loss more, etc.):

    Weighted L1 where material_mask==1 are foreground (important).
    """
    weight = torch.where(material_mask, w_fg, w_bg).float()
    errors = torch.abs(pred - target)
    return (weight * errors).sum() / weight.sum()


_lpips = lpips.LPIPS(net="vgg").to(device).eval()


def to_lpips_space(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W) in [0,1]
    returns: (B,3,H,W) in [-1,1]
    """
    return x.mul_(2).sub_(1)


@torch.no_grad()
def lpips_batch(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    # remap into LPIPS’s expected [-1,1]
    p = to_lpips_space(pred_rgb)
    t = to_lpips_space(target_rgb)

    return _lpips(p, t).mean()


def calculate_unet_albedo_loss(
    albedo_pred: torch.Tensor,
    albedo_gt: torch.Tensor,
    categories: list,
    ecpoch_data: dict,
    key="train",
) -> torch.Tensor:
    if ecpoch_data.get("unet_albedo") is None:
        ecpoch_data["unet_albedo"] = {}

    if ecpoch_data["unet_albedo"].get(key) is None:
        ecpoch_data["unet_albedo"][key] = {
            "l1_loss": 0.0,
            "total_loss": 0.0,
            "batch_count": 0,
            # "per_class_loss": {name: 0.0 for name in CLASS_LIST},
            # "per_class_sample_count": {name: 0 for name in CLASS_LIST},
        }

    # L1 Loss
    l1_loss = masked_l1(
        albedo_pred,
        albedo_gt,
        material_mask=torch.ones_like(
            albedo_pred[:, :1], dtype=torch.bool, device=device
        ),
    )

    ecpoch_data["unet_albedo"][key]["l1_loss"] += l1_loss.item()

    # SSIM
    # ssim_val = FM.structural_similarity_index_measure(
    #     albedo_pred.clamp(0, 1), albedo_gt.clamp(0, 1), data_range=1.0
    # )
    # if isinstance(ssim_val, tuple):
    #     ssim_val = ssim_val[0]
    # ssim_loss = 1 - ssim_val.item()

    # # LPIPS
    # lpips = lpips_batch(
    #     albedo_pred.clamp(0, 1), albedo_gt.clamp(0, 1)
    # )

    # Total loss
    total_loss = l1_loss

    ecpoch_data["unet_albedo"][key]["total_loss"] += total_loss.item()
    ecpoch_data["unet_albedo"][key]["batch_count"] += 1

    return total_loss


def calculate_unet_maps_loss(
    roughness_pred: torch.Tensor,
    metallic_pred: torch.Tensor,
    ao_pred: torch.Tensor,
    height_pred: torch.Tensor,
    roughness_gt: torch.Tensor,
    metallic_gt: torch.Tensor,
    ao_gt: torch.Tensor,
    height_gt: torch.Tensor,
    masks: torch.Tensor,
    categories: list,
    ecpoch_data: dict,
    key="train",
) -> torch.Tensor:
    if ecpoch_data.get("unet_maps") is None:
        ecpoch_data["unet_maps"] = {}

    if ecpoch_data["unet_maps"].get(key) is None:
        ecpoch_data["unet_maps"][key] = {
            "total_loss": 0.0,
            "rough_loss": 0.0,
            "rough_l1_loss": 0.0,
            "metal_loss": 0.0,
            "ao_loss": 0.0,
            "height_loss": 0.0,
            "height_l1_loss": 0.0,
            "height_tv": 0.0,
            "batch_count": 0,
        }

    # Calculate masks
    mask_all = torch.ones_like(roughness_gt, dtype=torch.bool)  # (B, 1, H, W)
    mask_metal = (masks == METAL_IDX).unsqueeze(1)  # (B, 1, H, W)

    # Roughness, since every pixel is important, we use a mask of ones
    l1_rough = masked_l1(
        pred=roughness_pred, target=roughness_gt, material_mask=mask_all
    )
    ecpoch_data["unet_maps"][key]["rough_l1_loss"] += l1_rough.item()
    # ssim_rough = FM.structural_similarity_index_measure(
    #     roughness_pred.clamp(0, 1),
    #     roughness_gt.clamp(0, 1),
    #     data_range=1.0,
    # )
    # if isinstance(ssim_rough, tuple):
    #     ssim_rough = ssim_rough[0]

    # loss_rough = l1_rough + 0.05 * (1 - ssim_rough)
    loss_rough = l1_rough
    ecpoch_data["unet_maps"][key]["rough_loss"] += loss_rough.item()

    # Metal
    # loss_metal = F.binary_cross_entropy_with_logits(
    #     metallic_pred,
    #     metallic_gt,
    #     weight=mask_metal,  # Zeros out non-metal regions
    #     reduction="sum",
    # )
    # loss_metal = loss_metal / mask_metal.sum().clamp(min=1.0)  # Avoid division by zero
    # Phase A0
    loss_metal = masked_l1(metallic_pred, metallic_gt, material_mask=mask_all)
    ecpoch_data["unet_maps"][key]["metal_loss"] += loss_metal.item()

    # AO, since every pixel is important, we use a mask of ones
    loss_ao = masked_l1(ao_pred, ao_gt, material_mask=mask_all)
    ecpoch_data["unet_maps"][key]["ao_loss"] += loss_ao.item()

    # Height
    loss_height = masked_l1(height_pred, height_gt, mask_all)
    ecpoch_data["unet_maps"][key]["height_l1_loss"] += loss_height.item()
    # Gradient total variation (TV) smoothness penalty
    # [w0 - w1, w1 - w2, ..., wN-1 - wN]
    # dx = torch.abs(height_pred[..., :-1] - height_pred[..., 1:]).mean()
    # [h0 - h1, h1 - h2, ..., hN-1 - hN]
    # dy = torch.abs(height_pred[..., :-1, :] - height_pred[..., 1:, :]).mean()
    # tv = dx + dy
    # loss_height = masked_l1(height_pred, height_gt, mask_all) + 0.01 * tv
    # Since every pixel is important, we use a mask of ones
    ecpoch_data["unet_maps"][key]["height_loss"] += loss_height.item()

    loss_total = (loss_rough + loss_metal + loss_ao + loss_height) / 4.0
    ecpoch_data["unet_maps"][key]["total_loss"] += loss_total.item()
    ecpoch_data["unet_maps"][key]["batch_count"] += 1

    return loss_total


def calculate_avg(epoch_data, key="train"):
    total_batches = epoch_data[key]["batch_count"]

    epoch_data["unet_albedo"][key]["l1_loss"] = (
        epoch_data["unet_albedo"][key]["l1_loss"] / total_batches
    )
    epoch_data["unet_albedo"][key]["total_loss"] = (
        epoch_data["unet_albedo"][key]["total_loss"] / total_batches
    )

    epoch_data["unet_maps"][key]["rough_l1_loss"] = (
        epoch_data["unet_maps"][key]["rough_l1_loss"] / total_batches
    )
    epoch_data["unet_maps"][key]["rough_loss"] = (
        epoch_data["unet_maps"][key]["rough_loss"] / total_batches
    )
    epoch_data["unet_maps"][key]["metal_loss"] = (
        epoch_data["unet_maps"][key]["metal_loss"] / total_batches
    )
    epoch_data["unet_maps"][key]["ao_loss"] = (
        epoch_data["unet_maps"][key]["ao_loss"] / total_batches
    )
    epoch_data["unet_maps"][key]["height_l1_loss"] = (
        epoch_data["unet_maps"][key]["height_l1_loss"] / total_batches
    )
    epoch_data["unet_maps"][key]["height_loss"] = (
        epoch_data["unet_maps"][key]["height_loss"] / total_batches
    )
    epoch_data["unet_maps"][key]["total_loss"] = (
        epoch_data["unet_maps"][key]["total_loss"] / total_batches
    )

    return (
        epoch_data["unet_albedo"][key]["total_loss"],
        epoch_data["unet_maps"][key]["total_loss"],
    )


def to_rgb(x):
    # x: (1, H, W) → (3, H, W) by repeating the gray channel
    return x.repeat(3, 1, 1)


# Training loop
def do_train():
    train_ds, val_ds = load_my_dataset()
    print(
        f"Starting training for {EPOCHS} epochs, on {len(train_ds)} samples, validation on {len(val_ds)} samples."
    )

    train_loader = DataLoader(
        train_ds,  # type: ignore
        batch_size=BATCH_SIZE,
        # sampler=train_sampler,
        # num_workers=4,
        shuffle=True,  # ! DISABLE FOR PHASE A+
    )

    validation_loader = DataLoader(
        val_ds,  # type: ignore
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation data
        # num_workers=6,
    )

    optimizer = torch.optim.AdamW(
        list(unet_alb.parameters()) + list(unet_maps.parameters()),
        lr=LR,
        weight_decay=WD,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    scaler = GradScaler(device.type)  # AMP scaler for mixed precision

    best_val_loss_albedo = float("inf")
    best_val_loss_maps = float("inf")
    patience = 4
    no_improvement_count_albedo = 0
    no_improvement_count_maps = 0
    albedo_frozen = False
    maps_frozen = False

    output_dir = Path(f"./weights/{PHASE}/unets")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        if albedo_frozen and maps_frozen:
            print("Both UNet-Albedo and UNet-Maps are auto frozen, stopping training.")
            break

        unet_alb.train()
        unet_maps.train()

        # Should we use GT albedo for UNet-maps
        teacher_epochs = 10 if PHASE.lower() == "a" or PHASE.lower() == "a0" else 0
        # If true don't detach Unet-albedo gradients
        joint_finetune = (PHASE.lower() == "c") and (epoch >= 0.5 * EPOCHS)
        # My RTX 5090 doesn't have enough memory for batch size 4, so using 2 with accumulation
        accum_steps = 2

        epoch_data = {
            "epoch": epoch + 1,
        }

        optimizer.zero_grad()

        for i, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")
        ):
            diffuse_and_normal = batch["diffuse_and_normal"]
            normal = batch["normal"]
            category = batch["category"]
            albedo_gt = batch["albedo"]
            height = batch["height"]
            metallic = batch["metallic"]
            roughness = batch["roughness"]
            ao = batch["ao"]
            masks = batch["masks"]

            diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
            normal = normal.to(device, non_blocking=True)
            albedo_gt = albedo_gt.to(device, non_blocking=True)
            height = height.to(device, non_blocking=True)
            metallic = metallic.to(device, non_blocking=True)
            roughness = roughness.to(device, non_blocking=True)
            ao = ao.to(device, non_blocking=True)

            with autocast(device_type=device.type):
                #  Get Segoformer ouput for FiLM
                # seg_feats = segformer(inputs6, output_hidden_states=True).hidden_states[-1].detach()      # (B,256,H/16,W/16)

                # Get UNet-Albedo prediction
                albedo_pred = unet_alb(diffuse_and_normal, None)

            # Get albedo input for UNet-maps
            if epoch < teacher_epochs:
                # predirected albedo is not good enough in earlier phases on early epochs so use GT albedo
                maps_input_albedo = albedo_gt
            else:
                # Joint finetuning only in some phases
                maps_input_albedo = (
                    albedo_pred if joint_finetune else albedo_pred.detach()
                )

            # Normalize albedo_pred
            maps_input_albedo = TF.normalize(
                maps_input_albedo,
                mean=IMAGENET_STANDARD_MEAN,
                std=IMAGENET_STANDARD_STD,
            )

            maps_input = torch.cat(
                [maps_input_albedo, normal],
                dim=1,  # Concatenate albedo and normal along the channel dimension (B, 6, H, W)
            )

            with autocast(device_type=device.type):
                # Get UNet-Maps prediction
                maps_pred = unet_maps(maps_input, None)

            roughness_pred = maps_pred["rough"]
            metallic_pred = maps_pred["metal"]
            ao_pred = maps_pred["ao"]
            height_pred = maps_pred["height"]

            # Unet-albedo loss
            unet_albedo_loss = calculate_unet_albedo_loss(
                albedo_pred, albedo_gt, category, epoch_data, key="train"
            )

            unet_maps_loss = calculate_unet_maps_loss(
                roughness_pred,
                metallic_pred,
                ao_pred,
                height_pred,
                roughness_gt=roughness,
                metallic_gt=metallic,
                ao_gt=ao,
                height_gt=height,
                masks=masks,
                categories=category,
                ecpoch_data=epoch_data,
                key="train",
            )

            # Total loss
            total_loss = unet_albedo_loss + unet_maps_loss

            # loss.backward()
            # optimizer.step()

            # ① scale down so that sum over accum_steps equals real batch gradient
            total_loss = total_loss / accum_steps
            scaler.scale(total_loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        unet_alb.eval()
        unet_maps.eval()
        samples_saved_per_class = {name: 0 for name in CLASS_LIST}

        with torch.no_grad():
            for batch in tqdm(
                validation_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation"
            ):
                diffuse_and_normal = batch["diffuse_and_normal"]
                albedo_gt = batch["albedo"]
                normal = batch["normal"]
                category = batch["category"]
                height = batch["height"]
                metallic = batch["metallic"]
                roughness = batch["roughness"]
                ao = batch["ao"]
                masks = batch["masks"]
                names = batch["name"]

                diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
                normal = normal.to(device, non_blocking=True)
                albedo_gt = albedo_gt.to(device, non_blocking=True)
                height = height.to(device, non_blocking=True)
                metallic = metallic.to(device, non_blocking=True)
                roughness = roughness.to(device, non_blocking=True)
                ao = ao.to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    # seg_feats = segformer(inputs6, output_hidden_states=True)\
                    #     .hidden_states[-1].detach()      # (B,256,H/16,W/16)
                    albedo_pred = unet_alb(diffuse_and_normal, None)

                # Get albedo input for UNet-maps
                if epoch < teacher_epochs:
                    # predirected albedo is not good enough in earlier phases on early epochs so use GT albedo
                    maps_input_albedo = albedo_gt
                else:
                    # Joint finetuning only in some phases
                    maps_input_albedo = (
                        albedo_pred if joint_finetune else albedo_pred.detach()
                    )

                # Normalize albedo_pred
                maps_input_albedo = TF.normalize(
                    maps_input_albedo,
                    mean=IMAGENET_STANDARD_MEAN,
                    std=IMAGENET_STANDARD_STD,
                )

                maps_input = torch.cat(
                    [maps_input_albedo, normal],
                    dim=1,  # Concatenate albedo and normal along the channel dimension (B, 6, H, W)
                )

                with autocast(device_type=device.type):
                    # Get UNet-Maps prediction
                    maps_pred = unet_maps(maps_input, None)

                roughness_pred = maps_pred["rough"]
                metallic_pred = maps_pred["metal"]
                ao_pred = maps_pred["ao"]
                height_pred = maps_pred["height"]

                calculate_unet_albedo_loss(
                    albedo_pred, albedo_gt, category, epoch_data, key="validation"
                )

                calculate_unet_maps_loss(
                    roughness_pred,
                    metallic_pred,
                    ao_pred,
                    height_pred,
                    roughness_gt=roughness,
                    metallic_gt=metallic,
                    ao_gt=ao,
                    height_gt=height,
                    masks=masks,
                    categories=category,
                    ecpoch_data=epoch_data,
                    key="validation",
                )

                for k in range(len(category)):
                    # Accumulate per-class loss
                    cat_name = CLASS_LIST[category[k].item()]  # Get the category name

                    if samples_saved_per_class[cat_name] < 2:
                        # Save 2 samples per class for inspection
                        output_path = output_dir / f"val_samples_{epoch + 1}/{cat_name}"
                        output_path.mkdir(parents=True, exist_ok=True)

                        # Save diffuse, normal, GT albedo and predicted albedo side by side
                        combined_gt = torch.cat(
                            [
                                diffuse_and_normal[k][
                                    :3, ...
                                ],  # Diffuse - first 3 channels
                                diffuse_and_normal[k][
                                    3:, ...
                                ],  # Normal - next 3 channels
                                albedo_gt[k],  # GT Albedo
                                # height[i],  # Height
                                to_rgb(metallic[k]),  # Metallic
                                to_rgb(roughness[k]),  # Roughness
                                to_rgb(ao[k]),  # AO
                            ],
                            dim=2,  # Concatenate along width
                        )
                        predicted = torch.cat(
                            [
                                diffuse_and_normal[k][
                                    :3, ...
                                ],  # Diffuse - first 3 channels
                                diffuse_and_normal[k][
                                    3:, ...
                                ],  # Normal - next 3 channels
                                albedo_pred[k],  # Predicted Albedo
                                # height_pred[i],  # Height
                                to_rgb(metallic_pred[k]),  # Metallic
                                to_rgb(roughness_pred[k]),  # Roughness
                                to_rgb(ao_pred[k]),  # AO
                            ],
                            dim=2,  # Concatenate along width
                        )

                        combined = torch.cat(
                            [
                                combined_gt,  # GT
                                predicted,  # Predicted
                            ],
                            dim=1,  # Concatenate along height
                        ).clamp(
                            0, 1
                        )  # Clamp to [0, 1] for saving

                        # Height is saved as a separate image since it is 16-bit
                        height = torch.cat(
                            [
                                height[k],  # Height
                                height_pred[k],  # Predicted Height
                            ],
                            dim=2,  # Concatenate along width
                        ).clamp(
                            0, 1
                        )  # Clamp to [0, 1] for saving

                        save_image(combined, output_path / f"{cat_name}_{names[k]}.png")

                        # Save height as 16-bit PNG, save_image() doesn't work for 16-bit images
                        h16 = (height.squeeze(0).cpu().numpy() * 65535).astype(
                            np.uint16
                        )
                        height_im = Image.fromarray(h16, mode="I;16")
                        height_im.save(
                            output_path / f"{cat_name}_{names[k]}_height.png",
                            format="PNG",
                        )

                        samples_saved_per_class[cat_name] += 1

        calculate_avg(epoch_data, key="train")
        unet_albedo_total_val_loss, unet_maps_total_val_loss = calculate_avg(
            epoch_data, key="validation"
        )

        print(json.dumps(epoch_data, indent=4))

        if unet_albedo_total_val_loss < best_val_loss_albedo:
            best_val_loss_albedo = unet_albedo_total_val_loss
            no_improvement_count_albedo = 0
        else:
            no_improvement_count_albedo += 1
            print(
                f"UNet-Albedo: no improvement at epoch {epoch + 1}, validation loss: {unet_albedo_total_val_loss:.4f}"
            )
            if no_improvement_count_albedo >= patience:
                print(
                    f"UNet-Albedo: Early freezing at epoch {epoch + 1}, no improvement for {patience} epochs."
                )
                # Freeze UNet-Albedo parameters
                for p in unet_alb.parameters():
                    p.requires_grad = False
                    p.grad = None  # Clear stale gradients

                albedo_frozen = True

        if unet_maps_total_val_loss < best_val_loss_maps:
            best_val_loss_maps = unet_maps_total_val_loss
            no_improvement_count_maps = 0
        else:
            no_improvement_count_maps += 1
            print(
                f"UNet-Maps: no improvement at epoch {epoch + 1}, validation loss: {unet_maps_total_val_loss:.4f}"
            )
            if no_improvement_count_maps >= patience:
                print(
                    f"UNet-Maps: Early freezing at epoch {epoch + 1}, no improvement for {patience} epochs."
                )
                # Freeze UNet-Maps parameters
                for p in unet_maps.parameters():
                    p.requires_grad = False
                    p.grad = None  # Clear stale gradients

                maps_frozen = True

        # Save checkopoint after each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "unet_albedo_model_state_dict": unet_alb.state_dict(),
                "unet_maps_model_state_dict": unet_maps.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch_data": epoch_data,
            },
            output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
        )
        # Save epoch data to a JSON file
        with open(output_dir / f"epoch_{epoch + 1}_stats.json", "w") as f:
            json.dump(epoch_data, f, indent=4)

        scheduler.step()

    print("Training completed.")


if __name__ == "__main__":
    # On Windows frozen executables need this; harmless otherwise
    multiprocessing.freeze_support()
    do_train()
