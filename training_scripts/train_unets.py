# Ensure we import it here to set random(seed)
import seed
import json, torch
import numpy as np
import random
import multiprocessing
import torch.nn.functional as F
import lpips
import math
import argparse
from typing import Callable
from unet_models import UNetAlbedo, UNetMaps
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from torchmetrics import functional as FM
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from train_dataset import SimpleImageDataset, normalize_normal_map
from augmentations import (
    get_random_crop,
    make_full_image_mask,
    selective_aug,
    center_crop,
)

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Train Segformer for Semantic Segmentation"
)
parser.add_argument(
    "--phase",
    type=str,
    default="a",
    help="Phase of the training per plan, used for logging and saving",
)
parser.add_argument(
    "--load_checkpoint",
    type=str,
    default=None,
    help="Path to the checkpoint to load the model from",
)

parser.add_argument(
    "--resume",
    type=bool,
    default=False,
    help="Whether to resume training from the last checkpoint",
)

args = parser.parse_args()
print(f"Training phase: {args.phase}")

# HYPER_PARAMETERS
BATCH_SIZE = 2  # Batch size for training
EPOCHS = 35  # Number of epochs to train
LR = 5e-5  # Learning rate for the optimizer
WD = 1e-2  # Weight decay for the optimizer
# T_MAX = 10  # Max number of epochs for the learning rate scheduler
PHASE = args.phase  # Phase of the training per plan, used for logging and saving

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable for faster training on fixed input sizes

matsynth_dir = (BASE_DIR / "../matsynth_processed").resolve()

train_dataset = SimpleImageDataset(
    matsynth_dir=str(matsynth_dir),
    split="train",
)

validation_dataset = SimpleImageDataset(
    matsynth_dir=str(matsynth_dir), split="validation", skip_init=True
)

validation_dataset.all_validation_samples = train_dataset.all_validation_samples
loss_weights, sample_weights = train_dataset.get_weights()

# # Will pull random samples according to the sample weights
train_sampler = WeightedRandomSampler(
    weights=sample_weights.tolist(),
    num_samples=len(sample_weights),
    replacement=True,
)


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


checkpoint = None
resume_training = args.resume
if (args.load_checkpoint is not None) and Path(args.load_checkpoint).resolve().exists():
    load_checkpoint_path = Path(args.load_checkpoint).resolve()
    print(
        f"Loading model from checkpoint: {load_checkpoint_path}, resume={resume_training}"
    )
    checkpoint = torch.load(load_checkpoint_path, map_location=device)

if checkpoint is not None:
    print("Loading model weights from checkpoint...")
    unet_alb.load_state_dict(checkpoint["unet_albedo_model_state_dict"])
    unet_maps.load_state_dict(checkpoint["unet_maps_model_state_dict"])


# ! Set metal bias # to a value that corresponds to 8.4% metal pixels in the dataset to prevent early collapse
p0 = 0.084
b0 = -math.log((1 - p0) / p0)  # ≈ -2.36
with torch.no_grad():
    torch.nn.init.constant_(unet_maps.head_metal[0].bias, b0)  # type: ignore
    # print(unet_maps.head_metal[0].bias)


def get_transform_train(
    current_epoch: int,
    safe_augmentations=True,
    color_augmentations=True,
) -> Callable:
    def transform_train_fn(example):
        albedo = example["basecolor"]
        normal = example["normal"]
        height = example["height"]
        metallic = example["metallic"]
        roughness = example["roughness"]
        diffuse = example["diffuse"]
        ao = example["ao"]
        category = example["category"]
        category_name = example["category_name"]
        name = example["name"]

        albedo, normal, diffuse, height, metallic, roughness, ao = get_random_crop(
            albedo=albedo,
            normal=normal,
            size=(256, 256),  # Crop size for training
            diffuse=diffuse,
            height=height,
            metallic=metallic,
            roughness=roughness,
            ao=ao,
            resize_to=[1024, 1024],  # Resize to 1024x1024 for training
            augmentations=safe_augmentations,
        )

        albedo_orig = albedo
        albedo_aug = albedo

        if color_augmentations:
            diffuse = selective_aug(diffuse, category=category_name)
            # For teacher mode
            albedo_aug = selective_aug(albedo, category=category_name)

        mask = make_full_image_mask(category, img_size=(1024, 1024))

        diffuse = TF.to_tensor(diffuse)  # type: ignore
        diffuse = TF.normalize(
            diffuse, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
        )

        normal = TF.to_tensor(normal)  # type: ignore
        normal = TF.normalize(
            normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
        )

        # Concatenate albedo and normal along the channel dimension
        diffuse_and_normal = torch.cat((diffuse, normal), dim=0)  # type: ignore

        albedo_orig = TF.to_tensor(albedo_orig)  # type: ignore
        # ! Note we normalize to standard mean/std inside training loop
        albedo_aug = TF.to_tensor(albedo_aug)  # type: ignore

        # to_tensor() is normalizing 8 bit images ( / 255 ) so for 16bit we need to do it manually
        height_arr = np.array(height, dtype=np.uint16)
        height_arr = height_arr.astype(np.float32) / 65535.0  # Normalize to [0, 1]
        height = torch.from_numpy(height_arr).unsqueeze(0)

        metallic = TF.to_tensor(metallic)  # type: ignore
        roughness = TF.to_tensor(roughness)  # type: ignore
        ao = TF.to_tensor(ao)  # type: ignore

        return {
            "diffuse_and_normal": diffuse_and_normal,
            "height": height,
            "albedo_aug": albedo_aug,
            "albedo_orig": albedo_orig,
            "normal": normal,
            "metallic": metallic,
            "roughness": roughness,
            "ao": ao,
            "masks": mask,
            "category": category,
            "name": name,
        }

    return transform_train_fn


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

    albedo = center_crop(
        albedo,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=TF.InterpolationMode.LANCZOS,
    )
    diffuse = center_crop(
        diffuse,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=TF.InterpolationMode.LANCZOS,
    )

    normal = normalize_normal_map(
        center_crop(
            normal,
            size=(256, 256),
            resize_to=[1024, 1024],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
    )
    height = center_crop(
        height,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=TF.InterpolationMode.BICUBIC,
    )
    metallic = center_crop(
        metallic,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    roughness = center_crop(
        roughness,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    ao = center_crop(
        ao,
        size=(256, 256),
        resize_to=[1024, 1024],
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    # Store original non normalized diffuse and normal for visual inspection in validation loop
    original_normal = TF.to_tensor(normal)
    original_diffuse = TF.to_tensor(diffuse)

    diffuse = TF.to_tensor(diffuse)
    diffuse = TF.normalize(
        diffuse, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    normal = TF.to_tensor(normal)
    normal = TF.normalize(
        normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )
    diffuse_and_normal = torch.cat((diffuse, normal), dim=0)

    albedo = TF.to_tensor(albedo)

    height_arr = np.array(height, dtype=np.uint16)
    height_arr = height_arr.astype(np.float32) / 65535.0  # Normalize to [0, 1]
    height = torch.from_numpy(height_arr).unsqueeze(0)

    metallic = TF.to_tensor(metallic)
    roughness = TF.to_tensor(roughness)
    ao = TF.to_tensor(ao)

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


def masked_l1(pred, target, material_mask, w_fg=1.0, w_bg=1.0):
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
            "ssim_loss": 0.0,
            "lpips": 0.0,
            # "per_class_loss": {name: 0.0 for name in CLASS_LIST},
            # "per_class_sample_count": {name: 0 for name in CLASS_LIST},
        }

    # L1 Loss
    # l1_loss = masked_l1(
    #     albedo_pred,
    #     albedo_gt,
    #     material_mask=torch.ones_like(
    #         albedo_pred[:, :1], dtype=torch.bool, device=device
    #     ),
    # )
    l1_loss = F.l1_loss(albedo_pred.clamp(0, 1), albedo_gt.clamp(0, 1))
    l1_loss = l1_loss.float()

    ecpoch_data["unet_albedo"][key]["l1_loss"] += l1_loss.item()

    # SSIM
    ssim_val = FM.structural_similarity_index_measure(
        albedo_pred.clamp(0, 1).float(), albedo_gt.clamp(0, 1).float(), data_range=1.0
    )
    if isinstance(ssim_val, tuple):
        ssim_val = ssim_val[0]
    ssim_val = torch.nan_to_num(ssim_val, nan=1.0).float()
    ssim_loss = 1 - ssim_val
    ecpoch_data["unet_albedo"][key]["ssim_loss"] += ssim_loss.item()

    # LPIPS
    with autocast(device_type=device.type, enabled=False):
        lpips = lpips_batch(
            albedo_pred.clamp(0, 1).float(), albedo_gt.clamp(0, 1).float()
        )
    lpips = torch.nan_to_num(lpips, nan=0.0).float()
    ecpoch_data["unet_albedo"][key]["lpips"] += lpips.item()

    # Total loss
    total_loss = l1_loss + 0.1 * ssim_loss + 0.05 * lpips

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
            "rough_l1_loss": 0.0,
            "rought_ssim_loss": 0.0,
            "rough_loss": 0.0,
            "metal_loss": 0.0,
            "ao_loss": 0.0,
            "height_l1_loss": 0.0,
            "height_tv_penalty": 0.0,
            "height_loss": 0.0,
            "total_loss": 0.0,
        }

    # Calculate masks
    # mask_all = torch.ones_like(roughness_gt, dtype=torch.bool)  # (B, 1, H, W)
    # mask_metal = (
    #     (masks == train_dataset.METAL_IDX).unsqueeze(1).to(device)
    # )  # (B, 1, H, W)

    # Roughness, since every pixel is important, we use a mask of ones
    # l1_rough = masked_l1(
    #     pred=roughness_pred, target=roughness_gt, material_mask=mask_all
    # )
    l1_rough = F.l1_loss(roughness_pred, roughness_gt).float()

    ecpoch_data["unet_maps"][key]["rough_l1_loss"] += l1_rough.item()
    ssim_rough = FM.structural_similarity_index_measure(
        roughness_pred.clamp(0, 1).float(),
        roughness_gt.clamp(0, 1).float(),
        data_range=1.0,
    )
    if isinstance(ssim_rough, tuple):
        ssim_rough = ssim_rough[0]
    ssim_rough = torch.nan_to_num(ssim_rough, nan=1.0).float()

    ssim_rough_loss = (1 - ssim_rough).float()
    ecpoch_data["unet_maps"][key]["rought_ssim_loss"] += ssim_rough_loss.item()

    loss_rough = l1_rough + 0.05 * ssim_rough_loss
    ecpoch_data["unet_maps"][key]["rough_loss"] += loss_rough.item()

    # Metal
    metal_positive = metallic_gt.sum().float()
    metal_negative = (metallic_gt.numel() - metal_positive).float()
    metal_weights = ((metal_negative + 1e-6) / (metal_positive + 1e-6)).clamp(
        min=1.0, max=20.0
    )
    loss_metal = F.binary_cross_entropy_with_logits(
        metallic_pred, metallic_gt, pos_weight=metal_weights, reduction="mean"
    )

    # metal_ratio_raw = (metal_negative / metal_positive).sqrt().clamp(max=20.0)
    # metal_normaliser = (metal_ratio_raw * metal_positive + metal_negative) / (
    #     metal_positive + metal_negative
    # )
    # metal_weight = metal_ratio_raw / metal_normaliser
    # non_metal_weight = 1.0 / metal_normaliser
    # weight_map = torch.where(metallic_gt > 0.5, metal_weight, non_metal_weight)

    # loss_metal = F.binary_cross_entropy_with_logits(
    #     metallic_pred, metallic_gt, weight=weight_map, reduction="mean"
    # )

    # loss_metal = F.binary_cross_entropy_with_logits(
    #     metallic_pred,
    #     metallic_gt,
    #     weight=mask_metal,  # Zeros out non-metal regions
    #     reduction="sum",
    # )
    # loss_metal = (
    #     loss_metal / mask_metal.sum().clamp(min=1.0)  # Avoid division by zero
    # ).float()
    ecpoch_data["unet_maps"][key]["metal_loss"] += loss_metal.item()

    # AO, since every pixel is important, we use a mask of ones
    # loss_ao = masked_l1(ao_pred, ao_gt, material_mask=mask_all)
    loss_ao = F.l1_loss(ao_pred, ao_gt).float()
    ecpoch_data["unet_maps"][key]["ao_loss"] += loss_ao.item()

    # Height
    # l1_height = masked_l1(height_pred, height_gt, mask_all)
    l1_height = F.l1_loss(height_pred, height_gt).float()
    ecpoch_data["unet_maps"][key]["height_l1_loss"] += l1_height.item()
    # Gradient total variation (TV) smoothness penalty
    # [w0 - w1, w1 - w2, ..., wN-1 - wN]
    dx = torch.abs(height_pred[..., :-1] - height_pred[..., 1:]).mean().float()
    # [h0 - h1, h1 - h2, ..., hN-1 - hN]
    dy = torch.abs(height_pred[..., :-1, :] - height_pred[..., 1:, :]).mean().float()
    tv = dx + dy
    grad_penalty = 0.01 * tv
    ecpoch_data["unet_maps"][key]["height_tv_penalty"] += grad_penalty.item()

    loss_height = l1_height + grad_penalty
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
    epoch_data["unet_albedo"][key]["ssim_loss"] = (
        epoch_data["unet_albedo"][key]["ssim_loss"] / total_batches
    )
    epoch_data["unet_albedo"][key]["lpips"] = (
        epoch_data["unet_albedo"][key]["lpips"] / total_batches
    )

    epoch_data["unet_albedo"][key]["total_loss"] = (
        epoch_data["unet_albedo"][key]["total_loss"] / total_batches
    )

    epoch_data["unet_maps"][key]["rough_l1_loss"] = (
        epoch_data["unet_maps"][key]["rough_l1_loss"] / total_batches
    )
    epoch_data["unet_maps"][key]["rought_ssim_loss"] = (
        epoch_data["unet_maps"][key]["rought_ssim_loss"] / total_batches
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
    epoch_data["unet_maps"][key]["height_tv_penalty"] = (
        epoch_data["unet_maps"][key]["height_tv_penalty"] / total_batches
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
    print(
        f"Starting training for {EPOCHS} epochs, on {len(train_dataset)} samples, validation on {len(validation_dataset)} samples."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        # num_workers=4,
        shuffle=False,
        pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation data
        # num_workers=6,
        pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
    )

    optimizer = torch.optim.AdamW(
        list(unet_alb.parameters()) + list(unet_maps.parameters()),
        lr=LR,
        weight_decay=WD,
    )
    if checkpoint is not None and resume_training:
        print("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=EPOCHS * len(train_loader),
        # 15% warm-up 85% cooldown
        pct_start=0.15,
        div_factor=5.0,  # start LR = 1e-5
        final_div_factor=5.0,  # End LR = 1e-5
    )
    if checkpoint is not None and resume_training:
        print("Loading scheduler state from checkpoint.")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    scaler = GradScaler(device.type)  # AMP scaler for mixed precision
    if checkpoint is not None and resume_training:
        print("Loading scaler state from checkpoint.")
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    best_val_loss_albedo = float("inf")
    best_val_loss_maps = float("inf")
    patience = 6
    no_improvement_count_albedo = 0
    no_improvement_count_maps = 0
    albedo_frozen = False
    maps_frozen = False

    output_dir = Path(f"./weights/{PHASE}/unets")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if checkpoint is not None and resume_training:
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, EPOCHS):
        if albedo_frozen and maps_frozen:
            print("Both UNet-Albedo and UNet-Maps are auto frozen, stopping training.")
            break

        unet_alb.train()
        unet_maps.train()

        train_dataset.set_transform(
            get_transform_train(
                epoch + 1,
                # flips are enabled from epoch 1
                safe_augmentations=True,
                # Color augmentations are enabled after warm-up (from epoch 6)
                color_augmentations=(epoch + 1) > 5,
            )
        )
        validation_dataset.set_transform(transform_val_fn)

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
            albedo_gt = batch["albedo_orig"]
            albedo_aug = batch["albedo_aug"]
            height = batch["height"]
            metallic = batch["metallic"]
            roughness = batch["roughness"]
            ao = batch["ao"]
            masks = batch["masks"]

            diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
            normal = normal.to(device, non_blocking=True)
            albedo_gt = albedo_gt.to(device, non_blocking=True)
            albedo_aug = albedo_aug.to(device, non_blocking=True)
            height = height.to(device, non_blocking=True)
            metallic = metallic.to(device, non_blocking=True)
            roughness = roughness.to(device, non_blocking=True)
            ao = ao.to(device, non_blocking=True)

            with autocast(device_type=device.type):
                #  Get Segoformer ouput for FiLM
                # seg_feats = segformer(inputs6, output_hidden_states=True).hidden_states[-1].detach()      # (B,256,H/16,W/16)

                # Get UNet-Albedo prediction
                albedo_pred = unet_alb(diffuse_and_normal, None)

            # Unet-albedo loss
            unet_albedo_loss = calculate_unet_albedo_loss(
                albedo_pred, albedo_gt, category, epoch_data, key="train"
            )
            if torch.isnan(unet_albedo_loss):
                raise ValueError(
                    "Unet-Albedo loss is NaN, stopping training to avoid further issues."
                )

            # Get albedo input for UNet-maps
            if epoch < teacher_epochs:
                # predirected albedo is not good enough in earlier phases on early epochs so use GT albedo (potentially augmented)
                unet_maps_input_albedo = albedo_aug
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

            if torch.isnan(unet_maps_loss):
                raise ValueError(
                    "Unet-Maps loss is NaN, stopping training to avoid further issues."
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
                scheduler.step()

        calculate_avg(epoch_data, key="train")

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

                calculate_unet_albedo_loss(
                    albedo_pred, albedo_gt, category, epoch_data, key="validation"
                )

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

                    if samples_saved_per_class[cat_name] < 8:
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

        unet_albedo_total_val_loss, unet_maps_total_val_loss = calculate_avg(
            epoch_data, key="validation"
        )

        print(json.dumps(epoch_data, indent=4))

        # Save checkopoint after each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "unet_albedo_model_state_dict": unet_alb.state_dict(),
                "unet_maps_model_state_dict": unet_maps.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch_data": epoch_data,
            },
            output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
        )
        # Save epoch data to a JSON file
        with open(output_dir / f"epoch_{epoch + 1}_stats.json", "w") as f:
            json.dump(epoch_data, f, indent=4)

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

    print("Training completed.")


if __name__ == "__main__":
    # On Windows frozen executables need this; harmless otherwise
    multiprocessing.freeze_support()
    do_train()
