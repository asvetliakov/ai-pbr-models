# Ensure we import it here to set random(seed)
import seed
import json, torch
import numpy as np
import multiprocessing
import random
import math
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
import argparse

# from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from torchmetrics import functional as FM
from segformer_6ch import create_segformer
from transformers.utils.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)
from train_dataset import SimpleImageDataset, normalize_normal_map
from skyrim_dataset import SkyrimDataset
from augmentations import (
    get_random_crop,
    make_full_image_mask,
    selective_aug,
    center_crop,
    get_crop_size,
)
from skyrim_photometric_aug import SkyrimPhotometric

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

parser.add_argument(
    "--load_best_loss",
    type=bool,
    default=False,
    help="Whether to load the best validation loss from the checkpoint",
)

args = parser.parse_args()

print(f"Training phase: {args.phase}")

BASE_DIR = Path(__file__).resolve().parent

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

# HYPER_PARAMETERS
EPOCHS = 10  # Number of epochs to train
LR = 1e-5  # Learning rate for the optimizer
WD = 1e-2  # Weight decay for the optimizer
# T_MAX = 10  # Max number of epochs for the learning rate scheduler
# PHASE = "a"  # Phase of the training per plan, used for logging and saving
PHASE = args.phase  # Phase of the training per plan, used for logging and saving

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable for faster training on fixed input sizes

matsynth_dir = (BASE_DIR / "../matsynth_processed").resolve()
skyrim_dir = (BASE_DIR / "../skyrim_processed").resolve()

device = torch.device("cuda")

matsynth_train_dataset = SimpleImageDataset(
    matsynth_dir=str(matsynth_dir),
    split="train",
)

matsynth_validation_dataset = SimpleImageDataset(
    matsynth_dir=str(matsynth_dir),
    split="validation",
    skip_init=True,
)
matsynth_validation_dataset.all_validation_samples = (
    matsynth_train_dataset.all_validation_samples
)

loss_weights, sample_weights = matsynth_train_dataset.get_weights()

loss_weights = loss_weights.to(device)  # type: ignore

with autocast(device_type=device.type):
    matsynth_seg_loss_fn = torch.nn.CrossEntropyLoss(
        weight=loss_weights, ignore_index=255
    )
    matsynth_seq_per_class_fn = torch.nn.CrossEntropyLoss(
        weight=loss_weights, ignore_index=255, reduction="none"
    )
    skyrim_seg_loss_fn = torch.nn.CrossEntropyLoss()

# # Will pull random samples according to the sample weights
train_sampler = WeightedRandomSampler(
    weights=sample_weights.tolist(),
    num_samples=len(sample_weights),
    replacement=True,
)

skyrim_train_dataset = SkyrimDataset(
    skyrim_dir=str(skyrim_dir),
    split="train",
    load_non_pbr=True,
)

skyrim_validation_dataset = SkyrimDataset(
    skyrim_dir=str(skyrim_dir),
    split="validation",
    load_non_pbr=True,
    skip_init=True,
)
skyrim_validation_dataset.all_validation_samples = (
    skyrim_train_dataset.all_validation_samples
)

CROP_SIZE = 512

BATCH_SIZE_MATSYNTH = 18
BATCH_SIZE_SKYRIM = 6
BATCH_SIZE_VALIDATION_MATSYNTH = 18
BATCH_SIZE_VALIDATION_SKYRIM = 6

MATSYNTH_COMPOSITES = True
MATSYNTH_COLOR_AUGMENTATIONS = True
MATSYNTH_2_CROP_CHANGE = 0.25
MATSYNTH_4_CROP_CHANGE = 0.1
SKYRIM_PHOTOMETRIC = 0.6  # Photometric augmentation strength for Skyrim dataset

BATCH_SIZE = (
    BATCH_SIZE_MATSYNTH
    if BATCH_SIZE_MATSYNTH > BATCH_SIZE_SKYRIM
    else BATCH_SIZE_SKYRIM
)

MIN_SAMPLES_TRAIN = len(
    matsynth_train_dataset
    if BATCH_SIZE_MATSYNTH > BATCH_SIZE_SKYRIM
    else skyrim_train_dataset
)
MIN_SAMPLES_VALIDATION = len(matsynth_validation_dataset)
STEPS_PER_EPOCH_TRAIN = math.ceil(MIN_SAMPLES_TRAIN / BATCH_SIZE)
STEPS_PER_EPOCH_VALIDATION = math.ceil(
    MIN_SAMPLES_VALIDATION / BATCH_SIZE_VALIDATION_MATSYNTH
)

resume_training = args.resume


def get_model():
    best_model_checkpoint = None

    if (args.load_checkpoint is not None) and Path(
        args.load_checkpoint
    ).resolve().exists():
        load_checkpoint_path = Path(args.load_checkpoint).resolve()
        print(f"Loading checkpoint: {load_checkpoint_path}, resume={resume_training}")
        best_model_checkpoint = torch.load(load_checkpoint_path, map_location=device)

    model = create_segformer(
        num_labels=len(
            matsynth_train_dataset.CLASS_LIST
        ),  # Number of classes for segmentation
        device=device,
        lora=True,
        # For loading S1 checkpoint
        base_model_state=(
            best_model_checkpoint.get("model_state_dict", None)
            if best_model_checkpoint is not None
            else None
        ),
    )

    if best_model_checkpoint is not None:
        if best_model_checkpoint.get("base_model_state_dict") is not None:
            print("Loading base model state dict from checkpoint.")
            model.base_model.load_state_dict(
                best_model_checkpoint["base_model_state_dict"],
            )

        if best_model_checkpoint.get("lora_state_dict") is not None:
            # Load LoRA state dict if it exists
            print("Loading LoRA state dict from checkpoint.")
            model.load_state_dict(
                best_model_checkpoint["lora_state_dict"],
            )

    return model, best_model_checkpoint


# # Freeze everything except the LoRA layers and decode head
# for p in model.parameters():
#     p.requires_grad = False

# Phase S4, unfreeze all BatchNorm and LayerNorm layers only
# for name, module in model.named_modules():
#     if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
#         # both weight (gamma) and bias (beta) of each norm layer
#         module.weight.requires_grad = True
#         module.bias.requires_grad = True

# Unfreeze LoRA layers and decode head
# for name, p in model.named_parameters():
#     if "decode_head" in name:
#         # if "decode_head" in name or "lora_" in name
#         p.requires_grad = True

# # Override the BN momentum on all BatchNorm2d layers
# for m in model.modules():
#     if isinstance(m, torch.nn.BatchNorm2d):
#         # Default momentum is 0.1; we lower it so running stats update faster
#         m.momentum = 0.01

# Phase S3
# Unfreeze decode head
# for p in model.base_model.model.decode_head.parameters():  # type: ignore
#     p.requires_grad = True

# Unfreeze TOP 1/2 of the encoder blocks (+ their LoRA)
# enc_blocks = model.base_model.model.segformer.encoder.block  # type: ignore # nn.ModuleList
# start_idx = len(enc_blocks) // 2  # type: ignore # halfway index
# for blk in enc_blocks[start_idx:]:  # type: ignore
#     for p in blk.parameters():  # type: ignore
#         p.requires_grad = True

# for n, p in model.named_parameters():
#     if p.requires_grad:
#         print(f"parameter: {n}")


def matsynth_transform_train_fn(example):
    name = example["name"]
    # sample_cat = example["category_name"]

    # Upper left corner tuple for each cro
    positions = [(0, 0)]
    # h, w
    crop_size = (CROP_SIZE, CROP_SIZE)
    # h, w
    tile_size = [CROP_SIZE, CROP_SIZE]
    samples = [example]

    if MATSYNTH_COMPOSITES:
        if random.random() < MATSYNTH_4_CROP_CHANGE:
            positions = [
                (0, 0),
                (int(CROP_SIZE / 2), 0),
                (0, int(CROP_SIZE / 2)),
                (int(CROP_SIZE / 2), int(CROP_SIZE / 2)),
            ]
            tile_size = [int(CROP_SIZE / 2), int(CROP_SIZE / 2)]
            samples = [
                example,
                matsynth_train_dataset.get_random_sample(),
                matsynth_train_dataset.get_random_sample(),
                matsynth_train_dataset.get_random_sample(),
            ]
        elif random.random() < MATSYNTH_2_CROP_CHANGE:
            positions = [(0, 0), (int(CROP_SIZE / 2), 0)]
            tile_size = [CROP_SIZE, int(CROP_SIZE / 2)]
            samples = [
                example,
                matsynth_train_dataset.get_random_sample(),
            ]

    final_albedo = Image.new("RGB", crop_size)
    final_normal = Image.new("RGB", crop_size)
    final_mask = torch.zeros(crop_size, dtype=torch.int64)
    # final_color_mask = Image.new("RGB", crop_size)
    # original_albedo = Image.new("RGB", crop_size)
    # original_normal = Image.new("RGB", crop_size)

    for sample, pos in zip(samples, positions):
        albedo = sample["basecolor"]
        normal = sample["normal"]
        category = sample["category"]
        category_name = sample["category_name"]

        albedo, normal, *_ = get_random_crop(
            albedo,
            normal,
            size=(tile_size[0], tile_size[1]),
            augmentations=True,
            resize_to=None,
        )
        if MATSYNTH_COLOR_AUGMENTATIONS:
            # original_albedo.paste(albedo, box=pos)  # type: ignore
            # original_normal.paste(normal, box=pos)  # type: ignore

            albedo, normal = selective_aug(albedo, normal, category=category_name)

        # albedo = TF.resize(
        #     albedo, tile_size, interpolation=TF.InterpolationMode.LANCZOS  # type: ignore
        # )
        # normal = TF.resize(
        #     normal, tile_size, interpolation=TF.InterpolationMode.BILINEAR  # type: ignore
        # )
        # normal = normalize_normal_map(normal)  # type: ignore

        final_albedo.paste(albedo, box=pos)  # type: ignore
        final_normal.paste(normal, box=pos)  # type: ignore

        mask = make_full_image_mask(
            category_id=category,
            # height comes first
            img_size=(tile_size[0], tile_size[1]),
        )  # (H, W)

        final_mask[pos[1] : pos[1] + tile_size[0], pos[0] : pos[0] + tile_size[1]] = (
            mask
        )

        # Mask visualization
        # color_mask = np.zeros((tile_size[0], tile_size[1], 3), dtype=np.uint8)
        # color_cat = matsynth_train_dataset.CLASS_PALETTE[category]  # type: ignore
        # color_mask[:, :] = color_cat  # type: ignore
        # final_color_mask.paste(
        #     Image.fromarray(color_mask, mode="RGB"), box=pos  # type: ignore
        # )

    # img_test_dir = (BASE_DIR / f"test_images/matsynth").resolve()
    # visual_check = Image.new("RGB", (CROP_SIZE * 5, CROP_SIZE))
    # visual_check.paste(final_albedo, (0, 0))  # type: ignore
    # visual_check.paste(final_normal, (CROP_SIZE, 0))  # type: ignore
    # visual_check.paste(original_albedo, (CROP_SIZE * 2, 0))  # type: ignore
    # visual_check.paste(original_normal, (CROP_SIZE * 3, 0))  # type: ignore
    # visual_check.paste(final_color_mask, (CROP_SIZE * 4, 0))  # type: ignore
    # img_test_dir.mkdir(parents=True, exist_ok=True)
    # visual_check.save(
    #     img_test_dir / f"{sample_cat}_{name}.png",
    # )

    final_albedo = TF.to_tensor(final_albedo)
    # Segformer has been trained with ImageNet default normalization
    final_albedo = TF.normalize(
        final_albedo, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )

    final_normal = TF.to_tensor(final_normal)
    final_normal = TF.normalize(
        final_normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    # Concatenate albedo and normal along the channel dimension
    final_sample = torch.cat((final_albedo, final_normal), dim=0)  # type: ignore

    return {
        "pixel_values": final_sample,
        "labels": final_mask,
        # "category": category,  # keep for reference
    }


def skyrim_transform_train_fn(example):
    # name = example["name"]
    # current_crop_size = get_crop_size(current_epoch, CROP_SIZES_PER_EPOCH)

    image = example["basecolor"] if example["pbr"] else example["diffuse"]
    normal = example["normal"]

    final_image, final_normal, *_ = get_random_crop(
        image,
        normal,
        size=(CROP_SIZE, CROP_SIZE),
        augmentations=True,
        resize_to=None,
    )
    if SKYRIM_PHOTOMETRIC > 0.0:
        skyrim_photometric = SkyrimPhotometric(p_aug=SKYRIM_PHOTOMETRIC)
        final_image = skyrim_photometric(final_image)

    final_image = TF.to_tensor(final_image)
    # Segformer has been trained with ImageNet default normalization
    final_image = TF.normalize(
        final_image, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )

    final_normal = TF.to_tensor(final_normal)
    final_normal = TF.normalize(
        final_normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    # Concatenate albedo and normal along the channel dimension
    final_sample = torch.cat((final_image, final_normal), dim=0)  # type: ignore

    return {
        "pixel_values": final_sample,
    }


def matsynth_transform_val_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    category = example["category"]

    albedo = center_crop(
        albedo,
        (CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.LANCZOS,
    )
    albedo = TF.to_tensor(albedo)
    albedo = TF.normalize(albedo, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    normal = center_crop(
        normal,
        (CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    normal = TF.to_tensor(normal)
    normal = TF.normalize(
        normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    # Concatenate albedo and normal along the channel dimension
    final = torch.cat((albedo, normal), dim=0)  # type: ignore

    mask = make_full_image_mask(category_id=category, img_size=(CROP_SIZE, CROP_SIZE))

    return {
        "pixel_values": final,
        "labels": mask,
        "category": category,  # keep for reference
    }


def skyrim_transform_val_fn(example):
    image = example["basecolor"] if example["pbr"] else example["diffuse"]
    normal = example["normal"]

    final_image = center_crop(
        image,
        (CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.LANCZOS,
    )
    final_image = TF.to_tensor(final_image)
    final_image = TF.normalize(
        final_image, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )

    final_normal = center_crop(
        normal,
        (CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    final_normal = TF.to_tensor(final_normal)
    final_normal = TF.normalize(
        final_normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    # Concatenate albedo and normal along the channel dimension
    final = torch.cat((final_image, final_normal), dim=0)  # type: ignore

    return {
        "pixel_values": final,
    }


def cycle(dl: DataLoader):
    while True:
        for batch in dl:
            yield batch


matsynth_train_dataset.set_transform(matsynth_transform_train_fn)
matsynth_validation_dataset.set_transform(matsynth_transform_val_fn)

skyrim_train_dataset.set_transform(skyrim_transform_train_fn)
skyrim_validation_dataset.set_transform(skyrim_transform_val_fn)


# Training loop
def do_train():
    print(
        f"Starting training for {EPOCHS} epochs, on {STEPS_PER_EPOCH_TRAIN * BATCH_SIZE} Samples, MatSynth/Skyrim Batch: {BATCH_SIZE_MATSYNTH}/{BATCH_SIZE_SKYRIM}, validation on {MIN_SAMPLES_VALIDATION} MatSynth samples."
    )

    model, best_model_checkpoint = get_model()
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "decode_head" in name or "lora_" in name:
            p.requires_grad = True

    matsynth_train_loader = DataLoader(
        matsynth_train_dataset,  # type: ignore
        batch_size=BATCH_SIZE_MATSYNTH,
        sampler=train_sampler,
        num_workers=8,
        prefetch_factor=2,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    matsynth_validation_loader = DataLoader(
        matsynth_validation_dataset,  # type: ignore
        batch_size=BATCH_SIZE_VALIDATION_MATSYNTH,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    skyrim_train_loader = DataLoader(
        skyrim_train_dataset,
        batch_size=BATCH_SIZE_SKYRIM,
        num_workers=2,
        prefetch_factor=2,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    skyrim_validation_loader = DataLoader(
        skyrim_validation_dataset,
        batch_size=BATCH_SIZE_VALIDATION_SKYRIM,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    matsynth_train_iter = cycle(matsynth_train_loader)
    skyrim_train_iter = cycle(skyrim_train_loader)

    matsynth_validation_iter = cycle(matsynth_validation_loader)
    skyrim_validation_iter = cycle(skyrim_validation_loader)

    # head_params, enc_params, lora_params = [], [], []
    # for name, p in model.named_parameters():
    #     if not p.requires_grad:
    #         continue
    #     if ".decode_head." in name:
    #         head_params.append(p)
    #     elif "lora_" in name:
    #         lora_params.append(p)
    #     else:
    #         # top-half encoder (its base weights)
    #         enc_params.append(p)

    # # Give LoRA layers higher learning rate and no weight decay
    # param_groups = [
    #     {"params": head_params, "lr": LR, "weight_decay": WD},
    #     {"params": enc_params, "lr": LR, "weight_decay": WD},
    #     {"params": lora_params, "lr": 1e-5, "weight_decay": 0.0},
    # ]

    # lora_params = [
    #     p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    # ]
    # head_params = [
    #     p for n, p in model.named_parameters() if "decode_head" in n and p.requires_grad
    # ]
    # encoder_params = [
    #     p
    #     for n, p in model.named_parameters()
    #     if ("encoder.block.3" in n or "encoder.block.2" in n)
    #     and p.requires_grad
    #     and "lora_" not in n
    # ]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        # [
        #     # {"params": head_params, "lr": 6e-6, "weight_decay": WD},  # decode-head
        #     # {"params": lora_params, "lr": 3e-6, "weight_decay": 0.0},  # LoRA
        #     # {"params": encoder_params, "lr": 4e-6, "weight_decay": WD},
        #     # {"params": head_params, "lr": 1e-6, "weight_decay": WD},  # decode-head
        #     # {"params": lora_params, "lr": 4e-6, "weight_decay": 0},  # LoRA
        #     # {"params": encoder_params, "lr": 2e-6, "weight_decay": WD},
        # ],
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    if best_model_checkpoint is not None and resume_training:
        print("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(best_model_checkpoint["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=2e-6
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=1,
    #     T_mult=1,
    #     eta_min=2e-6,  # Minimum learning rate
    # )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=LR,
    #     total_steps=EPOCHS * STEPS_PER_EPOCH_TRAIN,
    #     # 15% warm-up 85% cooldown
    #     pct_start=0.15,
    #     div_factor=5.0,  # start LR = 1e-5
    #     final_div_factor=5.0,  # End LR = 1e-5
    # )
    if best_model_checkpoint is not None and resume_training:
        print("Loading scheduler state from checkpoint.")
        scheduler.load_state_dict(best_model_checkpoint["scheduler_state_dict"])

    scaler = GradScaler(device.type)  # AMP scaler for mixed precision
    if best_model_checkpoint is not None and resume_training:
        print("Loading scaler state from checkpoint.")
        scaler.load_state_dict(best_model_checkpoint["scaler_state_dict"])

    best_val_loss = float("inf")
    if best_model_checkpoint is not None and args.load_best_loss and resume_training:
        best_val_loss = best_model_checkpoint["epoch_data"]["val_loss"]
        print(f"Loaded best validation loss: {best_val_loss}")

    patience = 4
    no_improvement_count = 0

    output_dir = Path(f"./weights/{PHASE}/segformer")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if best_model_checkpoint is not None and resume_training:
        start_epoch = best_model_checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()

        # For IoU
        # ! need to reest it here early due to GPU memory issues
        val_all_labels = []
        val_all_preds = []

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": 0.0,
            "train_matsynth_loss": 0.0,
            "train_skyrim_loss": 0.0,
            "val_loss": 0.0,
            "val_matsynth_loss": 0.0,
            "val_skyrim_loss": 0.0,
            "val_skyrim_confidence": 0.0,
            "per_class_loss": {name: 0.0 for name in matsynth_train_dataset.CLASS_LIST},
            "IoU": {name: 0.0 for name in matsynth_train_dataset.CLASS_LIST},
            "mIoU": 0.0,
        }

        train_loss_sum = 0.0
        matsynth_loss_sum = 0.0
        skyrim_loss_sum = 0.0
        train_batch_count = 0

        bar = tqdm(
            range(STEPS_PER_EPOCH_TRAIN),
            desc=f"Epoch {epoch + 1}/{EPOCHS} - Training",
            unit="batch",
        )

        for i in bar:
            matsynth_batch = next(matsynth_train_iter)
            skyrim_batch = next(skyrim_train_iter)

            input = torch.cat(
                [matsynth_batch["pixel_values"], skyrim_batch["pixel_values"]], dim=0
            )
            # input = matsynth_batch["pixel_values"]
            matsynth_labels_gt = matsynth_batch["labels"]

            domain = torch.cat(
                [
                    torch.zeros(
                        len(matsynth_batch["pixel_values"]),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.ones(
                        len(skyrim_batch["pixel_values"]),
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )  # False=MatSynth, True=Skyrim

            input = input.to(device, non_blocking=True)
            matsynth_labels_gt = matsynth_labels_gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                logits: torch.Tensor = model(pixel_values=input).logits

                # upsample logits to match the input size
                logits_up: torch.Tensor = torch.nn.functional.interpolate(
                    logits,
                    size=input.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                # matsynth_logits = logits_up  # MatSynth logits
                matsynth_logits = logits_up[~domain]  # MatSynth logits
                skyrim_logits = logits_up[domain]

                matsynth_loss = matsynth_seg_loss_fn(
                    matsynth_logits, matsynth_labels_gt
                )

                if torch.isnan(matsynth_loss):
                    raise ValueError("Loss is NaN")

                skyrim_confidence, skyrim_pred_labels = skyrim_logits.softmax(
                    dim=1
                ).max(dim=1)
                # build a mask of high-confidence pixels
                skyrim_base_mask = skyrim_confidence >= 0.8

                # Our confidence for fabric & leather is not great from S1 checkpoint so use a lower threshold
                skyrim_fabric_mask = (
                    skyrim_pred_labels
                    == matsynth_train_dataset.CLASS_LIST_IDX_MAPPING["fabric"]
                ) & (skyrim_confidence > 0.6)

                skyrim_leather_mask = (
                    skyrim_pred_labels
                    == matsynth_train_dataset.CLASS_LIST_IDX_MAPPING["leather"]
                ) & (skyrim_confidence > 0.6)

                skyrim_mask = (
                    skyrim_base_mask | skyrim_fabric_mask | skyrim_leather_mask
                )

                if skyrim_mask.any():

                    # for the masked CE, we still need “labels”—
                    # use the network’s own argmax predictions where conf >= 0.8
                    skyrim_labels = skyrim_pred_labels[skyrim_mask]

                    # pick only those pixels
                    skyrim_logits_flat = skyrim_logits.permute(0, 2, 3, 1)[
                        skyrim_mask
                    ]  # (N_masked, C)

                    skyrim_loss = skyrim_seg_loss_fn(skyrim_logits_flat, skyrim_labels)
                    if torch.isnan(skyrim_loss):
                        raise ValueError("Skyrim loss is NaN")
                else:
                    skyrim_loss = torch.tensor(0.0, device=device)

                # skyrim_loss = torch.tensor(0.0, device=device)

                total_loss = matsynth_loss + 0.2 * skyrim_loss

            matsynth_loss_sum += matsynth_loss.item()
            skyrim_loss_sum += skyrim_loss.item()

            train_loss_sum += total_loss.item()
            train_batch_count += 1

            # loss.backward()
            # optimizer.step()

            scaler.scale(total_loss).backward()
            # — Gradient clipping —
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()

        matsynth_train_loss_avg = matsynth_loss_sum / train_batch_count
        skyrim_train_loss_avg = skyrim_loss_sum / train_batch_count
        train_loss_avg = train_loss_sum / train_batch_count
        epoch_data["train_loss"] = train_loss_avg
        epoch_data["train_matsynth_loss"] = matsynth_train_loss_avg
        epoch_data["train_skyrim_loss"] = skyrim_train_loss_avg

        model.eval()
        val_loss_sum = 0.0
        val_matsynth_loss_sum = 0.0
        val_skyrim_loss_sum = 0.0
        val_skyrim_confidence_sum = 0.0
        val_batch_count = 0

        # Per class cross-entropy loss
        val_total_loss_per_class = torch.zeros(
            len(matsynth_train_dataset.CLASS_LIST), dtype=torch.float32, device="cuda"
        )
        val_total_pixels_per_class = torch.zeros(
            len(matsynth_train_dataset.CLASS_LIST), dtype=torch.float32, device="cuda"
        )

        with torch.no_grad():
            bar = tqdm(
                range(STEPS_PER_EPOCH_VALIDATION),
                desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation",
                unit="batch",
            )

            for _ in bar:
                matsynth_batch = next(matsynth_validation_iter)
                skyrim_batch = next(skyrim_validation_iter)

                input = torch.cat(
                    [matsynth_batch["pixel_values"], skyrim_batch["pixel_values"]],
                    dim=0,
                )
                # input = matsynth_batch["pixel_values"]
                matsynth_labels_gt = matsynth_batch["labels"]

                domain = torch.cat(
                    [
                        torch.zeros(
                            len(matsynth_batch["pixel_values"]),
                            dtype=torch.bool,
                            device=device,
                        ),
                        torch.ones(
                            len(skyrim_batch["pixel_values"]),
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )  # False=MatSynth, True=Skyrim

                input = input.to(device, non_blocking=True)
                matsynth_labels_gt = matsynth_labels_gt.to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    logits: torch.Tensor = model(pixel_values=input).logits

                    # upsample logits to match the input size
                    logits_up: torch.Tensor = torch.nn.functional.interpolate(
                        logits,
                        size=input.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    # matsynth_logits = logits_up  # MatSynth logits
                    matsynth_logits = logits_up[~domain]  # MatSynth logits
                    skyrim_logits = logits_up[domain]

                    matsynth_loss = matsynth_seg_loss_fn(
                        matsynth_logits, matsynth_labels_gt
                    )

                    if torch.isnan(matsynth_loss):
                        raise ValueError("Loss is NaN")

                    skyrim_confidence, skyrim_pred_labels = skyrim_logits.softmax(
                        dim=1
                    ).max(dim=1)
                    # build a mask of high-confidence pixels
                    skyrim_base_mask = skyrim_confidence >= 0.6

                    # Our confidence for fabric is not great from S1 checkpoint so use a lower threshold
                    skyrim_fabric_mask = (
                        skyrim_pred_labels
                        == matsynth_train_dataset.CLASS_LIST_IDX_MAPPING["fabric"]
                    ) & (skyrim_confidence > 0.6)

                    skyrim_leather_mask = (
                        skyrim_pred_labels
                        == matsynth_train_dataset.CLASS_LIST_IDX_MAPPING["leather"]
                    ) & (skyrim_confidence > 0.6)

                    skyrim_mask = (
                        skyrim_base_mask | skyrim_fabric_mask | skyrim_leather_mask
                    )

                    if skyrim_mask.any():
                        # % of pixels above 0.8
                        skyrim_confident_pixels = skyrim_mask.float().mean().item()

                        # for the masked CE, we still need “labels”—
                        # use the network’s own argmax predictions where conf >= 0.8
                        skyrim_labels = skyrim_pred_labels[skyrim_mask]

                        # pick only those pixels
                        skyrim_logits_flat = skyrim_logits.permute(0, 2, 3, 1)[
                            skyrim_mask
                        ]  # (N_masked, C)

                        skyrim_loss = skyrim_seg_loss_fn(
                            skyrim_logits_flat, skyrim_labels
                        )
                        if torch.isnan(skyrim_loss):
                            raise ValueError("Skyrim loss is NaN")
                    else:
                        skyrim_loss = torch.tensor(0.0, device=device)
                        skyrim_confident_pixels = 0.0

                    # skyrim_loss = torch.tensor(0.0, device=device)

                    total_loss = matsynth_loss + 0.2 * skyrim_loss

                    pixel_loss = matsynth_seq_per_class_fn(
                        matsynth_logits, matsynth_labels_gt
                    )

                    flat_loss = pixel_loss.view(-1)
                    flat_labels = matsynth_labels_gt.view(-1)

                    for c in range(len(matsynth_train_dataset.CLASS_LIST)):
                        mask = flat_labels == c
                        val_total_loss_per_class[c] += flat_loss[mask].sum().float()
                        val_total_pixels_per_class[c] += mask.sum().float()

                    labels_pred = matsynth_logits.argmax(dim=1)
                    val_all_preds.append(labels_pred)
                    val_all_labels.append(matsynth_labels_gt)

                val_skyrim_loss_sum += skyrim_loss.item()
                val_skyrim_confidence_sum += skyrim_confident_pixels
                val_matsynth_loss_sum += matsynth_loss.item()
                val_loss_sum += total_loss.item()
                val_batch_count += 1

        val_loss_avg = val_loss_sum / val_batch_count
        val_matsynth_loss_avg = val_matsynth_loss_sum / val_batch_count
        val_skyrim_loss_avg = val_skyrim_loss_sum / val_batch_count
        val_skyrim_confidence_avg = val_skyrim_confidence_sum / val_batch_count
        epoch_data["val_loss"] = val_loss_avg
        epoch_data["val_matsynth_loss"] = val_matsynth_loss_avg
        epoch_data["val_skyrim_loss"] = val_skyrim_loss_avg
        epoch_data["val_skyrim_confidence"] = val_skyrim_confidence_avg

        # Calculate IoU (MatSynth)
        val_all_preds = torch.cat(val_all_preds, dim=0)
        val_all_labels = torch.cat(val_all_labels, dim=0)

        jaccard_index = FM.jaccard_index(
            val_all_preds,
            val_all_labels,
            num_classes=len(matsynth_train_dataset.CLASS_LIST),
            ignore_index=255,  # Ignore the background class
            task="multiclass",
            average="none",  # Calculate IoU for each class separately
        )
        for idx, name in enumerate(matsynth_train_dataset.CLASS_LIST):
            epoch_data["IoU"][name] = jaccard_index[idx].item()

        epoch_data["mIoU"] = jaccard_index.mean().item()

        val_avg_loss_per_class = val_total_loss_per_class / (
            val_total_pixels_per_class + 1e-6
        )
        for idx, name in enumerate(matsynth_train_dataset.CLASS_LIST):
            epoch_data["per_class_loss"][name] = val_avg_loss_per_class[idx].item()

        scheduler.step()
        print(json.dumps(epoch_data, indent=4))

        # Save checkopoint after each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                # "model_state_dict": model.state_dict(),
                "base_model_state_dict": model.base_model.state_dict(),
                "lora_state_dict": model.state_dict(),
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

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            no_improvement_count = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    # "model_state_dict": model.state_dict(),
                    "base_model_state_dict": model.base_model.state_dict(),
                    "lora_state_dict": model.state_dict(),
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
                f"No improvement at epoch {epoch + 1}, validation loss: {val_loss_avg:.4f}"
            )
            if no_improvement_count >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}, no improvement for {patience} epochs."
                )
                break

        torch.cuda.synchronize()  # ensure all kernels are done
        torch.cuda.empty_cache()  # release fragmented blocks

    print("Training completed.")


if __name__ == "__main__":
    # On Windows frozen executables need this; harmless otherwise
    multiprocessing.freeze_support()
    do_train()
