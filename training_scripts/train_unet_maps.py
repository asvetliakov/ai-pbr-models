# Ensure we import it here to set random(seed)
import seed
import json, torch
import multiprocessing
import torch.nn.functional as F
import lpips
import argparse
import math
from skyrim_dataset import SkyrimDataset
from unet_models import UNetMaps, UNetAlbedo
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from torchmetrics import functional as FM
import warnings
from transformers.utils.constants import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from train_dataset import SimpleImageDataset
from augmentations import (
    get_random_crop,
    center_crop,
)
from skyrim_photometric_aug import SkyrimPhotometric
from segformer_6ch import create_segformer

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast


BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Train UNet-Maps")
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

parser.add_argument(
    "--load_best_loss",
    type=bool,
    default=False,
    help="Whether to load the best validation loss from the checkpoint",
)

parser.add_argument(
    "--weight_donor",
    type=str,
    default=None,
    help="Path to the weight donor checkpoint to load the model from",
)

args = parser.parse_args()
print(f"Training phase: {args.phase}")

# HYPER_PARAMETERS
EPOCHS = 6  # Number of epochs to train
LR = 1e-6  # Learning rate for the optimizer
WD = 1e-2  # Weight decay for the optimizer
# T_MAX = 10  # Max number of epochs for the learning rate scheduler
PHASE = args.phase  # Phase of the training per plan, used for logging and saving

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True  # Enable for faster training on fixed input sizes
torch.backends.cudnn.benchmark = (
    False  # For some reason it may slows down consequent epochs
)

VISUAL_SAMPLES_COUNT = 16  # Number of samples to visualize in validation

matsynth_dir = (BASE_DIR / "../matsynth_processed").resolve()
skyrim_dir = (BASE_DIR / "../skyrim_processed_for_maps").resolve()

device = torch.device("cuda")

# need only for class list
matsynth_train_dataset = SimpleImageDataset(
    matsynth_dir=str(matsynth_dir),
    split="train",
    skip_init=True,
)


skyrim_train_dataset = SkyrimDataset(
    skyrim_dir=str(skyrim_dir),
    split="train",
    load_non_pbr=False,
    ignore_without_parallax=True,
)

skyrim_validation_dataset = SkyrimDataset(
    skyrim_dir=str(skyrim_dir),
    split="validation",
    load_non_pbr=False,
    skip_init=True,
)
skyrim_validation_dataset.all_validation_samples = (
    skyrim_train_dataset.all_validation_samples
)

CROP_SIZE = 768

BATCH_SIZE = 6
BATCH_SIZE_VALIDATION = 6

SKYRIM_PHOTOMETRIC = 0.0

MIN_SAMPLES_TRAIN = len(skyrim_train_dataset)
MIN_SAMPLES_VALIDATION = len(skyrim_validation_dataset)
STEPS_PER_EPOCH_TRAIN = math.ceil(MIN_SAMPLES_TRAIN / BATCH_SIZE)
STEPS_PER_EPOCH_VALIDATION = math.ceil(MIN_SAMPLES_VALIDATION / BATCH_SIZE_VALIDATION)

resume_training = args.resume


def get_model():
    unet_maps = UNetMaps(
        in_ch=6,  # RGB + Normal
        cond_ch=512,  # Condition channel size, can be adjusted
    ).to(
        device
    )  # type: ignore

    if args.weight_donor is not None:
        weight_donor_path = Path(args.weight_donor).resolve()
        print(f"Loading model from weight donor: {weight_donor_path}")
        weight_donor_checkpoint = torch.load(weight_donor_path, map_location=device)
        unet_maps.load_state_dict(
            weight_donor_checkpoint["unet_albedo_model_state_dict"], strict=False
        )

        # ! Set metal bias # to a value that corresponds to 15% metal pixels in the dataset to prevent early collapse
        # Doing this only on the first init on the first phase. If we're setting weight donor then we're at the first phase
        p0 = 0.15
        b0 = -math.log((1 - p0) / p0)
        with torch.no_grad():
            torch.nn.init.constant_(unet_maps.head_metal[0].bias, b0)  # type: ignore

    checkpoint = None
    if (args.load_checkpoint is not None) and Path(
        args.load_checkpoint
    ).resolve().exists():
        load_checkpoint_path = Path(args.load_checkpoint).resolve()
        print(
            f"Loading model from checkpoint: {load_checkpoint_path}, resume={resume_training}"
        )
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        unet_maps.load_state_dict(checkpoint["unet_maps_model_state_dict"])

    # Create segformer and load best weights
    segformer = create_segformer(
        num_labels=len(matsynth_train_dataset.CLASS_LIST),
        device=device,
        lora=True,
        frozen=True,
    )
    segformer_best_weights_path = (
        BASE_DIR / "../weights/s5/segformer/best_model.pt"
    ).resolve()
    print("Loading Segformer weights from:", segformer_best_weights_path)
    segformer_checkpoint = torch.load(segformer_best_weights_path, map_location=device)
    segformer.base_model.load_state_dict(
        segformer_checkpoint["base_model_state_dict"],
    )
    segformer.load_state_dict(
        segformer_checkpoint["lora_state_dict"],
    )

    unet_albedo = UNetAlbedo(
        in_ch=6,  # RGB + Normal
        cond_ch=512,  # Condition channel size, can be adjusted
    ).to(device)
    unet_albedo_best_weights_path = (
        BASE_DIR / "../weights/a3/unet_albedo/best_model.pt"
    ).resolve()
    print("Loading Unet-albedo weights from:", unet_albedo_best_weights_path)
    unet_albedo_checkpoint = torch.load(
        unet_albedo_best_weights_path, map_location=device
    )
    unet_albedo.load_state_dict(
        unet_albedo_checkpoint["unet_albedo_model_state_dict"],
    )
    for param in unet_albedo.parameters():
        param.requires_grad = False

    return unet_maps, segformer, unet_albedo, checkpoint


def transform_train_fn(example):
    name = example["name"]
    albedo = example["basecolor"]
    diffuse = example["diffuse"]
    normal = example["normal"]
    parallax = example["parallax"]
    ao = example["ao"]
    metallic = example["metallic"]
    roughness = example["roughness"]

    albedo, normal, diffuse, parallax, metallic, roughness, ao = get_random_crop(
        albedo,
        normal,
        size=(CROP_SIZE, CROP_SIZE),
        diffuse=diffuse,
        height=parallax,
        ao=ao,
        metallic=metallic,
        roughness=roughness,
        augmentations=True,
        resize_to=None,
    )

    albedo_orig = albedo
    albedo_segformer = albedo

    albedo_orig = TF.to_tensor(albedo_orig)

    if SKYRIM_PHOTOMETRIC > 0.0:
        skyrim_photometric = SkyrimPhotometric(p_aug=SKYRIM_PHOTOMETRIC)
        diffuse = skyrim_photometric(diffuse)

    normal = TF.to_tensor(normal)  # type: ignore
    normal = TF.normalize(
        normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    albedo_segformer = TF.to_tensor(albedo_segformer)
    albedo_segformer = TF.normalize(
        albedo_segformer, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )
    albedo_and_normal_segformer = torch.cat((albedo_segformer, normal), dim=0)

    diffuse = TF.to_tensor(diffuse)  # type: ignore
    diffuse = TF.normalize(
        diffuse, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )
    diffuse_and_normal = torch.cat((diffuse, normal), dim=0)

    ao = TF.to_tensor(ao)  # type: ignore
    metallic = TF.to_tensor(metallic)  # type: ignore
    roughness = TF.to_tensor(roughness)  # type: ignore

    parallax = TF.to_tensor(parallax) if parallax is not None else torch.zeros_like(ao)

    return {
        # Unet-albedo input
        "diffuse_and_normal": diffuse_and_normal,
        # Segformer input
        "albedo_and_normal_segformer": albedo_and_normal_segformer,
        # GT albedo, used for visualization
        "albedo": albedo_orig,
        # normalized normal, used for UNet-maps input (along with predirected albedo)
        "normal": normal,
        # GT parallax
        "parallax": parallax,
        # GT AO
        "ao": ao,
        # GT metallic
        "metallic": metallic,
        # GT roughness
        "roughness": roughness,
        "name": name,
    }


def transform_val_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    name = example["name"]
    diffuse = example["diffuse"]
    parallax = example["parallax"]
    ao = example["ao"]
    metallic = example["metallic"]
    roughness = example["roughness"]

    albedo = center_crop(
        albedo,
        size=(CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.LANCZOS,
    )

    normal = center_crop(
        normal,
        size=(CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    diffuse = center_crop(
        diffuse,
        size=(CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.LANCZOS,
    )

    parallax = (
        center_crop(
            parallax,
            size=(CROP_SIZE, CROP_SIZE),
            resize_to=None,
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        if parallax is not None
        else None
    )

    ao = center_crop(
        ao,
        size=(CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    metallic = center_crop(
        metallic,
        size=(CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    roughness = center_crop(
        roughness,
        size=(CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    albedo_orig = albedo
    albedo_segformer = albedo
    diffuse_orig = diffuse
    normal_orig = normal

    albedo_orig = TF.to_tensor(albedo_orig)

    normal = TF.to_tensor(normal)  # type: ignore
    normal = TF.normalize(
        normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    albedo_segformer = TF.to_tensor(albedo_segformer)
    albedo_segformer = TF.normalize(
        albedo_segformer, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )
    albedo_and_normal_segformer = torch.cat((albedo_segformer, normal), dim=0)

    diffuse = TF.to_tensor(diffuse)  # type: ignore
    diffuse = TF.normalize(
        diffuse, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )
    diffuse_and_normal = torch.cat((diffuse, normal), dim=0)

    ao = TF.to_tensor(ao)  # type: ignore
    metallic = TF.to_tensor(metallic)  # type: ignore
    roughness = TF.to_tensor(roughness)  # type: ignore
    parallax = TF.to_tensor(parallax) if parallax is not None else torch.zeros_like(ao)

    diffuse_orig = TF.to_tensor(diffuse_orig)  # type: ignore
    normal_orig = TF.to_tensor(normal_orig)  # type: ignore

    return {
        # Unet-albedo input
        "diffuse_and_normal": diffuse_and_normal,
        # Segformer input
        "albedo_and_normal_segformer": albedo_and_normal_segformer,
        # GT albedo, used for visualization
        "albedo": albedo_orig,
        # normalized normal, used for UNet-maps input (along with predirected albedo)
        "normal": normal,
        # GT parallax
        "parallax": parallax,
        # GT AO
        "ao": ao,
        # GT metallic
        "metallic": metallic,
        # GT roughness
        "roughness": roughness,
        # GT diffuse, used for visualization
        "orig_diffuse": diffuse_orig,
        # GT normal, used for visualization
        "orig_normal": normal_orig,
        "name": name,
    }


def calculate_unet_maps_loss(
    roughness_pred: torch.Tensor,
    metallic_pred: torch.Tensor,
    ao_pred: torch.Tensor,
    height_pred: torch.Tensor,
    roughness_gt: torch.Tensor,
    metallic_gt: torch.Tensor,
    ao_gt: torch.Tensor,
    height_gt: torch.Tensor,
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

    w_rough = 1.0  # Weight for roughness loss
    w_metal = 1.0  # Weight for metallic loss
    w_ao = 1.0  # Weight for AO loss
    w_height = 1.0  # Weight for height loss

    # Calculate masks
    # mask_all = torch.ones_like(roughness_gt, dtype=torch.bool)  # (B, 1, H, W)
    # mask_metal = (
    #     (masks == train_dataset.METAL_IDX).unsqueeze(1).to(device)
    # )  # (B, 1, H, W)

    # Roughness, since every pixel is important, we use a mask of ones
    # l1_rough = masked_l1(
    #     pred=roughness_pred, target=roughness_gt, material_mask=mask_all
    # )
    if w_rough != 0.0:
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
    else:
        loss_rough = torch.tensor(0.0, device=device)

    loss_rough = loss_rough * w_rough  # Apply weight
    ecpoch_data["unet_maps"][key]["rough_loss"] += loss_rough.item()

    # Metal
    if w_metal != 0.0:
        metal_positive = metallic_gt.sum().float()
        metal_negative = (metallic_gt.numel() - metal_positive).float()
        metal_weights = (
            ((metal_negative + 1e-6) / (metal_positive + 1e-6))
            .clamp(min=1.0, max=16.0)
            .to(device)
        )
        loss_metal = F.binary_cross_entropy_with_logits(
            metallic_pred, metallic_gt, pos_weight=metal_weights, reduction="mean"
        )
    else:
        loss_metal = torch.tensor(0.0, device=device)

    loss_metal = loss_metal * w_metal  # Apply weight

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
    if w_ao != 0.0:
        loss_ao = F.l1_loss(ao_pred, ao_gt).float()
    else:
        loss_ao = torch.tensor(0.0, device=device)

    loss_ao = loss_ao * w_ao  # Apply weight
    ecpoch_data["unet_maps"][key]["ao_loss"] += loss_ao.item()

    # Height
    # l1_height = masked_l1(height_pred, height_gt, mask_all)
    if w_height != 0.0:
        l1_height = F.l1_loss(height_pred, height_gt).float()
        ecpoch_data["unet_maps"][key]["height_l1_loss"] += l1_height.item()
        # Gradient total variation (TV) smoothness penalty
        # [w0 - w1, w1 - w2, ..., wN-1 - wN]
        dx = torch.abs(height_pred[..., :-1] - height_pred[..., 1:]).mean().float()
        # [h0 - h1, h1 - h2, ..., hN-1 - hN]
        dy = (
            torch.abs(height_pred[..., :-1, :] - height_pred[..., 1:, :]).mean().float()
        )
        tv = dx + dy
        grad_penalty = 0.005 * tv
        ecpoch_data["unet_maps"][key]["height_tv_penalty"] += grad_penalty.item()

        loss_height = l1_height + grad_penalty
    else:
        loss_height = torch.tensor(0.0, device=device)

    loss_height = loss_height * w_height  # Apply weight
    # Since every pixel is important, we use a mask of ones
    ecpoch_data["unet_maps"][key]["height_loss"] += loss_height.item()

    # loss_total = (loss_rough + loss_metal + loss_ao + loss_height) / 4.0
    loss_total = (loss_rough + loss_metal + loss_ao + loss_height) / (
        w_rough + w_metal + w_ao + w_height
    )
    ecpoch_data["unet_maps"][key]["total_loss"] += loss_total.item()

    return loss_total


def calculate_avg(epoch_data, key="train"):
    total_batches = epoch_data[key]["batch_count"]

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

    return epoch_data["unet_maps"][key]["total_loss"]


def cycle(dl: DataLoader):
    while True:
        for batch in dl:
            yield batch


def to_rgb(x):
    # x: (1, H, W) → (3, H, W) by repeating the gray channel
    return x.repeat(3, 1, 1)


skyrim_train_dataset.set_transform(transform_train_fn)
skyrim_validation_dataset.set_transform(transform_val_fn)


# Training loop
def do_train():
    print(
        f"Starting training for {EPOCHS} epochs, on {(STEPS_PER_EPOCH_TRAIN * BATCH_SIZE)} Samples, validation on {MIN_SAMPLES_VALIDATION} samples."
    )

    unet_maps, segformer, unet_albedo, checkpoint = get_model()

    # for param in unet_maps.parameters():
    #     param.requires_grad = False

    # for param in unet_maps.head_rough.parameters():
    #     param.requires_grad = True

    skyrim_train_loader = DataLoader(
        skyrim_train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        prefetch_factor=2,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    skyrim_validation_loader = DataLoader(
        skyrim_validation_dataset,
        batch_size=BATCH_SIZE_VALIDATION,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    skyrim_train_iter = cycle(skyrim_train_loader)

    skyrim_validation_iter = cycle(skyrim_validation_loader)

    # ❶ encoder with LLRD (0.8^depth)
    base_enc_lr = 2e-5
    base_dec_lr = 1e-4

    param_groups = []
    depth = len(unet_maps.unet.encoder)
    for i, block in enumerate(unet_maps.unet.encoder):
        lr_i = base_enc_lr * (0.8 ** (depth - i - 1))
        param_groups.append(
            {
                "params": block.parameters(),
                "lr": lr_i,
                "weight_decay": WD,
            }
        )

    # decoder + film + each head all at base_dec_lr
    param_groups += [
        {
            "params": unet_maps.unet.decoder.parameters(),
            "lr": base_dec_lr,
            "weight_decay": WD,
        },
        {"params": unet_maps.unet.film.parameters(), "lr": base_dec_lr, "weight_decay": WD},  # type: ignore
        {"params": unet_maps.head_rough.parameters(), "lr": base_dec_lr},
        {"params": unet_maps.head_metal.parameters(), "lr": base_dec_lr},
        {"params": unet_maps.head_ao.parameters(), "lr": base_dec_lr},
        {"params": unet_maps.head_h.parameters(), "lr": base_dec_lr},
    ]

    # trainable = [p for p in unet_maps.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        param_groups,
        # trainable,
        # param_groups,
        # lr=LR,
        # weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    if checkpoint is not None and resume_training:
        print("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=2e-6
    )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if checkpoint is not None and resume_training:
        print("Loading scheduler state from checkpoint.")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    scaler = GradScaler(device.type)  # AMP scaler for mixed precision
    if checkpoint is not None and resume_training:
        print("Loading scaler state from checkpoint.")
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    best_val_loss = float("inf")
    if checkpoint is not None and args.load_best_loss and resume_training:
        best_val_loss = checkpoint["epoch_data"]["unet_maps"]["validation"][
            "total_loss"
        ]
        print(f"Loading best validation loss from checkpoint: {best_val_loss}")

    patience = 6
    no_improvement_count = 0

    output_dir = Path(f"./weights/{PHASE}/unet_maps")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if checkpoint is not None and resume_training:
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, EPOCHS):
        unet_maps.train()

        epoch_data = {
            "epoch": epoch + 1,
            "train": {
                "batch_count": 0,
            },
            "validation": {
                "batch_count": 0,
            },
        }

        bar = tqdm(
            range(STEPS_PER_EPOCH_TRAIN),
            desc=f"Epoch {epoch + 1}/{EPOCHS} - Training",
            unit="batch",
        )

        for i in bar:
            skyrim_batch = next(skyrim_train_iter)

            diffuse_and_normal = skyrim_batch["diffuse_and_normal"]
            albedo_and_normal_segformer = skyrim_batch["albedo_and_normal_segformer"]
            normal = skyrim_batch["normal"]
            parallax_gt = skyrim_batch["parallax"]
            ao_gt = skyrim_batch["ao"]
            metallic_gt = skyrim_batch["metallic"]
            roughness_gt = skyrim_batch["roughness"]

            diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
            albedo_and_normal_segformer = albedo_and_normal_segformer.to(
                device, non_blocking=True
            )
            normal = normal.to(device, non_blocking=True)
            parallax_gt = parallax_gt.to(device, non_blocking=True)
            ao_gt = ao_gt.to(device, non_blocking=True)
            metallic_gt = metallic_gt.to(device, non_blocking=True)
            roughness_gt = roughness_gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.no_grad():
                with autocast(device_type=device.type):
                    #  Get Segoformer ouput for FiLM
                    seg_feats = (
                        segformer(
                            albedo_and_normal_segformer, output_hidden_states=True
                        )
                        .hidden_states[-1]
                        .detach()
                    )
                    predicted_albedo = unet_albedo(
                        diffuse_and_normal, seg_feats
                    ).detach()

            predicted_albedo = TF.normalize(
                predicted_albedo, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
            )
            predicted_albedo = predicted_albedo.to(device, non_blocking=True)

            albedo_and_normal = torch.cat((predicted_albedo, normal), dim=1)

            with autocast(device_type=device.type):
                # Get predicted maps from UNet
                maps_pred = unet_maps(albedo_and_normal, seg_feats)

            pred_parallax = maps_pred["height"]
            pred_ao = maps_pred["ao"]
            pred_metallic = maps_pred["metal"]
            pred_roughness = maps_pred["rough"]

            # Unet-albedo loss
            unet_loss = calculate_unet_maps_loss(
                pred_roughness,
                pred_metallic,
                pred_ao,
                pred_parallax,
                roughness_gt,
                metallic_gt,
                ao_gt,
                parallax_gt,
                epoch_data,
                key="train",
            )

            if torch.isnan(unet_loss):
                raise ValueError(
                    "Unet loss is NaN, stopping training to avoid further issues."
                )

            epoch_data["train"]["batch_count"] += 1

            # Total loss
            total_loss = unet_loss

            # loss.backward()
            # optimizer.step()

            scaler.scale(total_loss).backward()

            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()

        calculate_avg(epoch_data, key="train")

        unet_maps.eval()
        with torch.no_grad():
            bar = tqdm(
                range(STEPS_PER_EPOCH_VALIDATION),
                desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation",
                unit="batch",
            )

            num_samples_saved = 0

            for _ in bar:
                skyrim_batch = next(skyrim_validation_iter)

                diffuse_and_normal = skyrim_batch["diffuse_and_normal"]
                albedo_and_normal_segformer = skyrim_batch[
                    "albedo_and_normal_segformer"
                ]
                normal = skyrim_batch["normal"]
                parallax_gt = skyrim_batch["parallax"]
                ao_gt = skyrim_batch["ao"]
                metallic_gt = skyrim_batch["metallic"]
                roughness_gt = skyrim_batch["roughness"]
                albedo_gt = skyrim_batch["albedo"]
                diffuse_gt = skyrim_batch["orig_diffuse"]
                normal_gt = skyrim_batch["orig_normal"]
                name = skyrim_batch["name"]

                diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
                albedo_and_normal_segformer = albedo_and_normal_segformer.to(
                    device, non_blocking=True
                )
                normal = normal.to(device, non_blocking=True)
                parallax_gt = parallax_gt.to(device, non_blocking=True)
                ao_gt = ao_gt.to(device, non_blocking=True)
                metallic_gt = metallic_gt.to(device, non_blocking=True)
                roughness_gt = roughness_gt.to(device, non_blocking=True)
                albedo_gt = albedo_gt.to(device, non_blocking=True)
                diffuse_gt = diffuse_gt.to(device, non_blocking=True)
                normal_gt = normal_gt.to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    #  Get Segoformer ouput for FiLM
                    seg_feats = (
                        segformer(
                            albedo_and_normal_segformer, output_hidden_states=True
                        )
                        .hidden_states[-1]
                        .detach()
                    )
                    predicted_albedo = unet_albedo(
                        diffuse_and_normal, seg_feats
                    ).detach()

                predicted_albedo_orig = predicted_albedo
                predicted_albedo = TF.normalize(
                    predicted_albedo,
                    mean=IMAGENET_STANDARD_MEAN,
                    std=IMAGENET_STANDARD_STD,
                )
                predicted_albedo = predicted_albedo.to(device, non_blocking=True)

                albedo_and_normal = torch.cat((predicted_albedo, normal), dim=1)

                with autocast(device_type=device.type):
                    # Get predicted maps from UNet
                    maps_pred = unet_maps(albedo_and_normal, seg_feats)

                pred_parallax = maps_pred["height"]
                pred_ao = maps_pred["ao"]
                pred_metallic = maps_pred["metal"]
                pred_roughness = maps_pred["rough"]

                calculate_unet_maps_loss(
                    pred_roughness,
                    pred_metallic,
                    pred_ao,
                    pred_parallax,
                    roughness_gt,
                    metallic_gt,
                    ao_gt,
                    parallax_gt,
                    epoch_data,
                    key="validation",
                )

                epoch_data["validation"]["batch_count"] += 1

                if (
                    VISUAL_SAMPLES_COUNT > 0
                    and num_samples_saved < VISUAL_SAMPLES_COUNT
                ):
                    for (
                        sample_diffuse,
                        sample_normal,
                        sample_albedo_gt,
                        sample_roughness_gt,
                        sample_metallic_gt,
                        sample_ao_gt,
                        sample_parallax_gt,
                        sample_albedo_pred,
                        sample_pred_roughness,
                        sample_pred_metallic,
                        sample_pred_ao,
                        sample_pred_parallax,
                        sample_name,
                    ) in zip(
                        diffuse_gt,
                        normal_gt,
                        albedo_gt,
                        roughness_gt,
                        metallic_gt,
                        ao_gt,
                        parallax_gt,
                        predicted_albedo_orig,
                        pred_roughness,
                        pred_metallic,
                        pred_ao,
                        pred_parallax,
                        name,
                    ):

                        # Save few samples per class for inspection
                        output_path = output_dir / f"val_samples_{epoch + 1}"
                        output_path.mkdir(parents=True, exist_ok=True)

                        visual_sample_gt = torch.cat(
                            [
                                sample_diffuse,
                                sample_normal,
                                sample_albedo_gt,
                                to_rgb(sample_roughness_gt),
                                to_rgb(sample_metallic_gt),
                                to_rgb(sample_ao_gt),
                                to_rgb(sample_parallax_gt),
                            ],
                            dim=2,  # Concatenate along width
                        )
                        visual_sample_predicted = torch.cat(
                            [
                                sample_diffuse,
                                sample_normal,
                                sample_albedo_pred,
                                to_rgb(sample_pred_roughness),
                                to_rgb(sample_pred_metallic),
                                to_rgb(sample_pred_ao),
                                to_rgb(sample_pred_parallax),
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

                        save_image(visual_sample, output_path / f"{sample_name}.png")

                        num_samples_saved += 1

        unet_total_val_loss = calculate_avg(epoch_data, key="validation")

        scheduler.step()
        print(json.dumps(epoch_data, indent=4))

        # Save checkopoint after each epoch
        torch.save(
            {
                "epoch": epoch + 1,
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

        if unet_total_val_loss < best_val_loss:
            best_val_loss = unet_total_val_loss
            no_improvement_count = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "unet_maps_model_state_dict": unet_maps.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch_data": epoch_data,
                },
                output_dir / "best_model.pt",
            )

            # Save epoch data to a JSON file
            with open(output_dir / "best_model_stats.json", "w") as f:
                json.dump(epoch_data, f, indent=4)

            print(
                f"Saved new best model at epoch {epoch + 1} with loss {best_val_loss:.4f}"
            )
        else:
            no_improvement_count += 1
            print(
                f"UNet: no improvement at epoch {epoch + 1}, validation loss: {unet_total_val_loss:.4f}"
            )
            if no_improvement_count >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}, no improvement for {patience} epochs."
                )
                break

    print("Training completed.")


if __name__ == "__main__":
    # On Windows frozen executables need this; harmless otherwise
    multiprocessing.freeze_support()
    do_train()
