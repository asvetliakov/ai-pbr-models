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
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from torchmetrics import functional as FM
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from train_dataset import SimpleImageDataset, normalize_normal_map

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

BASE_DIR = Path(__file__).resolve().parent

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

matsynth_dir = (BASE_DIR / "../matsynth_processed").resolve()

train_dataset = SimpleImageDataset(
    matsynth_dir=str(matsynth_dir),
    split="train",
    max_train_samples_per_cat=144,  # For Phase A0
)

validation_dataset = SimpleImageDataset(
    matsynth_dir=str(matsynth_dir),
    split="validation",
)
# Phase A0 temp since using max_train_samples_per_cat, otherwise it deterministic
validation_dataset.all_train_samples = train_dataset.all_train_samples
validation_dataset.all_validation_samples = train_dataset.all_validation_samples

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


def center_crop(
    image: Image.Image,
    size: tuple[int, int],
    resize_to: list[int],
    interpolation: T.InterpolationMode,
) -> Image.Image:
    crop = TF.center_crop(image, size)  # type: ignore
    resized = TF.resize(crop, resize_to, interpolation=interpolation)  # type: ignore
    return resized  # type: ignore


transform_validation_input = T.Compose(
    [
        # Need to use same crop size for validation
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_STANDARD_MEAN,
            std=IMAGENET_STANDARD_STD,
        ),
    ]
)

transform_validation_gt = T.Compose(
    [
        T.ToTensor(),
    ]
)


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
    normal_resize = normalize_normal_map(TF.resize(normal_crop, resize_to, interpolation=T.InterpolationMode.BILINEAR))  # type: ignore
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


def make_full_image_mask(category_id: int, img_size: tuple[int, int]) -> torch.Tensor:
    """
    Build a segmentation mask of shape (H, W) where every pixel = category_id.
    """
    H, W = img_size
    # numpy array filled with your class index
    mask_np = np.full((H, W), fill_value=category_id, dtype=np.int64)
    # convert to torch LongTensor
    return torch.from_numpy(mask_np)


def transform_train_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    height = example["height"]
    metallic = example["metallic"]
    roughness = example["roughness"]
    diffuse = example["diffuse"]
    ao = example["ao"]
    category = example["category"]
    name = example["name"]

    diffuse, normal, albedo, height, metallic, roughness, ao = synced_crop_and_resize(
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

    mask = make_full_image_mask(category, img_size=(1024, 1024))
    # Store normal for later visualization

    diffuse = transform_train_input(diffuse)
    normal = transform_train_input(normal)

    # Concatenate albedo and normal along the channel dimension
    diffuse_and_normal = torch.cat((diffuse, normal), dim=0)  # type: ignore

    albedo = transform_train_gt(albedo)

    # ToTensor() is normalizing 8 bit images ( / 255 ) so for 16bit we need to do it manually
    height_arr = np.array(height, dtype=np.uint16)
    height_arr = height_arr.astype(np.float32) / 65535.0  # Normalize to [0, 1]
    height = torch.from_numpy(height_arr).unsqueeze(0)

    metallic = transform_train_gt(metallic)
    roughness = transform_train_gt(roughness)
    ao = transform_train_gt(ao)

    return {
        "diffuse_and_normal": diffuse_and_normal,
        "height": height,
        "albedo": albedo,
        "normal": normal,
        "metallic": metallic,
        "roughness": roughness,
        "ao": ao,
        "masks": mask,
        "category": category,
        "name": name,
    }


def transform_val_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    height = example["height"]
    metallic = example["metallic"]
    roughness = example["roughness"]
    diffuse = example["diffuse"]
    ao = example["ao"]
    category = example["category"]
    name = example["name"]

    mask = make_full_image_mask(category, img_size=(1024, 1024))

    # Store original non normalized diffuse and normal for visual inspection in validation loop
    albedo = center_crop(
        albedo,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=T.InterpolationMode.LANCZOS,
    )
    diffuse = center_crop(
        diffuse,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=T.InterpolationMode.LANCZOS,
    )

    normal = normalize_normal_map(
        center_crop(
            normal,
            size=(256, 256),
            resize_to=[1024, 1024],
            interpolation=T.InterpolationMode.BILINEAR,
        )
    )
    height = center_crop(
        height,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=T.InterpolationMode.BICUBIC,
    )
    metallic = center_crop(
        metallic,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=T.InterpolationMode.BILINEAR,
    )
    roughness = center_crop(
        roughness,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=T.InterpolationMode.BILINEAR,
    )
    ao = center_crop(
        ao,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=T.InterpolationMode.BILINEAR,
    )

    original_normal = transform_validation_gt(normal)
    original_diffuse = transform_validation_gt(diffuse)

    diffuse = transform_validation_input(diffuse)
    normal = transform_validation_input(normal)
    diffuse_and_normal = torch.cat((diffuse, normal), dim=0)  # type: ignore

    albedo = transform_validation_gt(albedo)

    height_arr = np.array(height, dtype=np.uint16)
    height_arr = height_arr.astype(np.float32) / 65535.0  # Normalize to [0, 1]
    height = torch.from_numpy(height_arr).unsqueeze(0)

    metallic = transform_validation_gt(metallic)

    roughness = transform_validation_gt(roughness)

    ao = transform_validation_gt(ao)

    return {
        "diffuse_and_normal": diffuse_and_normal,
        "height": height,
        "albedo": albedo,
        "normal": normal,
        "metallic": metallic,
        "roughness": roughness,
        "ao": ao,
        "masks": mask,
        "category": category,
        "name": name,
        "original_diffuse": original_diffuse,
        "original_normal": original_normal,
    }


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
        }

    # Calculate masks
    mask_all = torch.ones_like(roughness_gt, dtype=torch.bool)  # (B, 1, H, W)
    mask_metal = (masks == train_dataset.METAL_IDX).unsqueeze(1)  # (B, 1, H, W)

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


train_dataset.set_transform(transform_train_fn)
validation_dataset.set_transform(transform_val_fn)


# Training loop
def do_train():
    print(
        f"Starting training for {EPOCHS} epochs, on {len(train_dataset)} samples, validation on {len(validation_dataset)} samples."
    )

    train_loader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=BATCH_SIZE,
        # sampler=train_sampler,
        # num_workers=4,
        shuffle=True,  # ! DISABLE FOR PHASE A+
    )

    validation_loader = DataLoader(
        validation_dataset,  # type: ignore
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
            "train": {
                "batch_count": 0,
            },
            "validation": {
                "batch_count": 0,
            },
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
                unet_maps_input_albedo = albedo_gt
            else:
                # Joint finetuning only in some phases
                unet_maps_input_albedo = (
                    albedo_pred if joint_finetune else albedo_pred.detach()
                )

            # Normalize albedo_pred
            unet_maps_input_albedo = TF.normalize(
                unet_maps_input_albedo,
                mean=IMAGENET_STANDARD_MEAN,
                std=IMAGENET_STANDARD_STD,
            )

            unet_maps_input = torch.cat(
                [unet_maps_input_albedo, normal],
                dim=1,  # Concatenate albedo and normal along the channel dimension (B, 6, H, W)
            )

            with autocast(device_type=device.type):
                # Get UNet-Maps prediction
                maps_pred = unet_maps(unet_maps_input, None)

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

            epoch_data["train"]["batch_count"] += 1

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
        samples_saved_per_class = {name: 0 for name in train_dataset.CLASS_LIST}

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
                original_diffuse = batch["original_diffuse"]
                original_normal = batch["original_normal"]

                diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
                normal = normal.to(device, non_blocking=True)
                albedo_gt = albedo_gt.to(device, non_blocking=True)
                height = height.to(device, non_blocking=True)
                metallic = metallic.to(device, non_blocking=True)
                roughness = roughness.to(device, non_blocking=True)
                original_diffuse = original_diffuse.to(device, non_blocking=True)
                original_normal = original_normal.to(device, non_blocking=True)
                ao = ao.to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    # seg_feats = segformer(inputs6, output_hidden_states=True)\
                    #     .hidden_states[-1].detach()      # (B,256,H/16,W/16)
                    albedo_pred = unet_alb(diffuse_and_normal, None)

                # Normalize albedo_pred
                unet_maps_input_albedo = TF.normalize(
                    albedo_pred,
                    mean=IMAGENET_STANDARD_MEAN,
                    std=IMAGENET_STANDARD_STD,
                )

                unet_maps_input = torch.cat(
                    [unet_maps_input_albedo, normal],
                    dim=1,  # Concatenate albedo and normal along the channel dimension (B, 6, H, W)
                )

                with autocast(device_type=device.type):
                    # Get UNet-Maps prediction
                    maps_pred = unet_maps(unet_maps_input, None)

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

                epoch_data["validation"]["batch_count"] += 1

                for k in range(len(category)):
                    # Accumulate per-class loss
                    cat_name = train_dataset.CLASS_LIST[
                        category[k].item()
                    ]  # Get the category name

                    if samples_saved_per_class[cat_name] < 4:
                        # Save 2 samples per class for inspection
                        output_path = output_dir / f"val_samples_{epoch + 1}/{cat_name}"
                        output_path.mkdir(parents=True, exist_ok=True)

                        # Save diffuse, normal, GT albedo and predicted albedo side by side
                        visual_sample_gt = torch.cat(
                            [
                                original_diffuse[k],  # Diffuse
                                original_normal[k],  # Normal
                                albedo_gt[k],  # GT Albedo
                                # height[i],  # Height
                                to_rgb(metallic[k]),  # Metallic
                                to_rgb(roughness[k]),  # Roughness
                                to_rgb(ao[k]),  # AO
                            ],
                            dim=2,  # Concatenate along width
                        )
                        visual_sample_predicted = torch.cat(
                            [
                                original_diffuse[k],  # Diffuse
                                original_normal[k],  # Normal
                                albedo_pred[k],  # Predicted Albedo
                                # height_pred[i],  # Height
                                to_rgb(metallic_pred[k]),  # Metallic
                                to_rgb(roughness_pred[k]),  # Roughness
                                to_rgb(ao_pred[k]),  # AO
                            ],
                            dim=2,  # Concatenate along width
                        )

                        visual_sample = torch.cat(
                            [
                                visual_sample_gt,  # GT
                                visual_sample_predicted,  # Predicted
                            ],
                            dim=1,  # Concatenate along height
                        ).clamp(
                            0, 1
                        )  # Clamp to [0, 1] for saving

                        # Height is saved as a separate image since it is 16-bit
                        visual_sample_height = torch.cat(
                            [
                                height[k],  # Height
                                height_pred[k],  # Predicted Height
                            ],
                            dim=2,  # Concatenate along width
                        ).clamp(
                            0, 1
                        )  # Clamp to [0, 1] for saving

                        save_image(
                            visual_sample, output_path / f"{cat_name}_{names[k]}.png"
                        )

                        # Save height as 16-bit PNG, save_image() doesn't work for 16-bit images
                        h16 = (
                            visual_sample_height.squeeze(0).cpu().numpy() * 65535
                        ).astype(np.uint16)
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
