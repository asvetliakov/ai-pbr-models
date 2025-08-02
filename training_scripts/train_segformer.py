# Ensure we import it here to set random(seed)
import seed
import json, torch
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
from sklearn.metrics import confusion_matrix
from segformer_6ch import create_segformer
from transformers.utils.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)
from skyrim_dataset import SkyrimDataset, mask_to_tensor
from augmentations import (
    get_random_crop,
)
from skyrim_photometric_aug import SkyrimPhotometric
from class_materials import CLASS_LIST

parser = argparse.ArgumentParser(
    description="Train Segformer for Semantic Segmentation"
)
parser.add_argument(
    "--phase",
    type=str,
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
    "--lr_scale",
    type=float,
    default=1.0,
    help="Scale factor for learning rate when resuming (e.g., 0.5 to halve LR)",
)

args = parser.parse_args()

print(f"Training phase: {args.phase}")

BASE_DIR = Path(__file__).resolve().parent

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

PHASE = args.phase  # Phase of the training per plan, used for logging and saving

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = (
    False  # For some reason it may slows down consequent epochs
)

skyrim_dir = (BASE_DIR / "../skyrim_processed").resolve()

skyrim_data_file_path = (BASE_DIR / "../skyrim_data_segformer.json").resolve()

device = torch.device("cuda")


skyrim_train_dataset = SkyrimDataset(
    skyrim_dir=str(skyrim_dir),
    split="train",
    data_file=str(skyrim_data_file_path),
)

skyrim_validation_dataset = SkyrimDataset(
    skyrim_dir=str(skyrim_dir),
    split="validation",
    data_file=str(skyrim_data_file_path),
)


skyrim_data_file = json.load(open(skyrim_data_file_path, "r"))
skyrim_sample_weights = skyrim_data_file["sample_weights"]
skyrim_class_weights = torch.tensor(
    skyrim_data_file["class_weights"], dtype=torch.float32, device=device
)

skyrim_train_sampler = WeightedRandomSampler(
    weights=skyrim_sample_weights,
    num_samples=len(skyrim_sample_weights),
    replacement=True,
)


CROP_SIZE = 1024  # S3 Light - between 768-1024

BATCH_SIZE_VALIDATION_SKYRIM = 1

SKYRIM_PHOTOMETRIC = 0.4

SKYRIM_CERAMIC_CROP_BIAS_CHANCE = 0
SKYRIM_LEATHER_CROP_BIAS_CHANCE = 0

SKYRIM_WORKERS = 0
BATCH_SIZE_SKYRIM = 0

USE_ACCUMULATION = False

if CROP_SIZE == 256:
    BATCH_SIZE_SKYRIM = 40
    SKYRIM_WORKERS = 16
    # SKYRIM_LEATHER_CROP_BIAS_CHANCE = 0.3
    # SKYRIM_CERAMIC_CROP_BIAS_CHANCE = 0.3

if CROP_SIZE == 512:
    BATCH_SIZE_SKYRIM = 12
    SKYRIM_WORKERS = 12
    # SKYRIM_LEATHER_CROP_BIAS_CHANCE = 0.25
    # SKYRIM_CERAMIC_CROP_BIAS_CHANCE = 0.2

if CROP_SIZE == 768:
    BATCH_SIZE_SKYRIM = 6
    SKYRIM_WORKERS = 6
    USE_ACCUMULATION = True
    # SKYRIM_LEATHER_CROP_BIAS_CHANCE = 0.15
    # SKYRIM_CERAMIC_CROP_BIAS_CHANCE = 0.1

if CROP_SIZE == 896:
    BATCH_SIZE_SKYRIM = 4
    SKYRIM_WORKERS = 6
    USE_ACCUMULATION = True

if CROP_SIZE == 1024:
    BATCH_SIZE_SKYRIM = 3
    SKYRIM_WORKERS = 6
    USE_ACCUMULATION = True
    # SKYRIM_LEATHER_CROP_BIAS_CHANCE = 0.15


BATCH_SIZE = BATCH_SIZE_SKYRIM

MIN_SAMPLES_TRAIN = len(skyrim_train_dataset)
STEPS_PER_EPOCH_TRAIN = math.ceil(MIN_SAMPLES_TRAIN / BATCH_SIZE)

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
        num_labels=len(CLASS_LIST),  # Number of classes for segmentation
        device=device,
        lora=False,
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


def skyrim_transform_train_fn(example):
    # name = example["name"]
    # current_crop_size = get_crop_size(current_epoch, CROP_SIZES_PER_EPOCH)

    # if (
    #     SKYRIM_CERAMIC_CROP_BIAS_CHANCE > 0
    #     and random.random() < SKYRIM_CERAMIC_CROP_BIAS_CHANCE
    # ):
    #     crop_data = skyrim_data_file[f"ceramic_crops_{CROP_SIZE}"]
    #     (sample_name, x, y) = random.choice(crop_data)
    #     specific_crop_pos = (x, y)
    #     example = skyrim_train_dataset.get_specific_sample_for_relative_mask(
    #         sample_name
    #     )
    specific_crop_pos = None
    if (
        SKYRIM_LEATHER_CROP_BIAS_CHANCE > 0
        and random.random() < SKYRIM_LEATHER_CROP_BIAS_CHANCE
    ):
        crop_data = skyrim_data_file[f"leather_crops_{CROP_SIZE}"]
        # crop_data = skyrim_data_file[f"leather_crops_768"]
        (sample_name, x, y) = random.choice(crop_data)
        specific_crop_pos = (x, y)
        example = skyrim_train_dataset.get_specific_sample_for_relative_mask(
            sample_name
        )

    albedo = example["basecolor"]
    normal = example["normal"]
    mask = example["mask"]

    used_diffuse = False
    # if random.random() < 0.3:
    #     albedo = example["diffuse"]
    #     used_diffuse = True

    crop_result = get_random_crop(
        albedo=albedo,  # type: ignore
        normal=normal,  # type: ignore
        mask=mask,
        size=(CROP_SIZE, CROP_SIZE),
        augmentations=True,
        resize_to=None,
        specific_crop_pos=specific_crop_pos,
    )
    final_albedo = crop_result["albedo"]
    final_normal = crop_result["normal"]
    final_mask = crop_result["mask"]

    if SKYRIM_PHOTOMETRIC > 0.0 and not used_diffuse:
        skyrim_photometric = SkyrimPhotometric(
            p_aug=SKYRIM_PHOTOMETRIC, ao_enabled=True
        )
        final_albedo = skyrim_photometric(final_albedo)

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

    final_mask = mask_to_tensor(final_mask)  # type: ignore

    return {
        "pixel_values": final_sample,
        "labels": final_mask,
    }


def skyrim_transform_val_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    mask = example["mask"]

    # final_albedo = center_crop(
    #     albedo,
    #     (CROP_SIZE, CROP_SIZE),
    #     resize_to=None,
    #     interpolation=TF.InterpolationMode.LANCZOS,
    # )
    albedo = TF.to_tensor(albedo)
    albedo = TF.normalize(albedo, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    # final_normal = center_crop(
    #     normal,
    #     (CROP_SIZE, CROP_SIZE),
    #     resize_to=None,
    #     interpolation=TF.InterpolationMode.BILINEAR,
    # )
    normal = TF.to_tensor(normal)
    normal = TF.normalize(
        normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    # Concatenate albedo and normal along the channel dimension
    final = torch.cat((albedo, normal), dim=0)  # type: ignore

    final_mask = mask_to_tensor(mask)

    return {
        "pixel_values": final,
        "labels": final_mask,
    }


PER_CLASS_GAMMA_MAP = torch.tensor(
    [
        1.2,  # fabric   <- moderate class, moderate damping
        1.2,  # ground   <- moderate class, moderate damping
        1.0,  # leather  <- minority class, minimal damping (keep CE-like)
        1.5,  # metal    <- frequent class, more damping
        2.0,  # stone    <- majority class, strong damping for easy pixels
        1.5,  # wood     <- frequent class, more damping
    ],
    device=device,
)  # gamma for focal loss - higher gamma = more damping of easy examples


def dropout_mask(
    labels: torch.Tensor,
    keep_prob: float = 0.80,
):
    classes = [
        CLASS_LIST.index("stone"),
        # CLASS_LIST.index("wood"),
    ]
    """
    Returns a boolean mask (B,H,W) with True for pixels kept in the loss.
    """
    mask = torch.ones_like(labels, dtype=torch.bool)
    if keep_prob >= 1.0:
        return mask  # nothing to drop

    rand = torch.rand_like(labels.float())  # U(0,1) for every pixel
    drop_condition = rand > keep_prob  # True where we drop

    # Combine for all specified classes in one shot
    target_pixels = torch.zeros_like(mask)
    for cls in classes:
        target_pixels |= labels == cls

    mask &= ~(target_pixels & drop_condition.bool())
    return mask


IGNORE_INDEX = 255


def _make_valid_mask(target: torch.Tensor) -> torch.Tensor:
    """mask == 1 where target ∈ [0, C-1]; 0 where target == 255"""
    return target != IGNORE_INDEX


# Dynamic focal term for minority classes
def focal_ce(
    logits: torch.Tensor, target: torch.Tensor, keep_mask=None
) -> torch.Tensor:
    valid = _make_valid_mask(target)  # (B,H,W) bool
    if keep_mask is not None:
        valid &= keep_mask  # combine masks

    if not valid.any():
        return logits.new_tensor(0.0, requires_grad=True)  # nothing to learn

    # restrict tensors to valid pixels only  → 1-D views (flat)
    logits_flat = logits.permute(0, 2, 3, 1)[valid]  # (P,C)
    target_flat = target[valid]  # (P,)

    log_p = torch.nn.functional.log_softmax(logits_flat, dim=1)
    p = torch.exp(log_p)

    one_hot = torch.nn.functional.one_hot(
        target_flat, num_classes=logits.size(1)
    ).float()

    gamma = PER_CLASS_GAMMA_MAP.to(logits.device, logits.dtype)
    focal = (1 - p).pow(gamma) * log_p
    focal_loss = -(one_hot * focal).float().sum(1)

    return focal_loss.mean()


def dice_loss(pred_probs, target, keep_mask=None):
    valid = _make_valid_mask(target)
    if keep_mask is not None:
        valid &= keep_mask

    if not valid.any():
        return pred_probs.new_tensor(0.0, requires_grad=True)

    # Build 4D mask for broadcasting
    valid4 = valid[:, None, :, :].float()  # (B,1,H,W)

    # Zero out invalid / dropped pixels
    pred_m = pred_probs * valid4  # (B,C,H,W)
    # One-hot for valid pixels only
    gt_onehot = (
        torch.nn.functional.one_hot(
            torch.where(valid, target, 0), num_classes=pred_probs.size(1)
        )
        .permute(0, 3, 1, 2)
        .float()
        * valid4
    )  # (B,C,H,W), zeros elsewhere

    # Sum across spatial dims ONLY -> per-class stats
    intersection = (pred_m * gt_onehot).sum(dim=(2, 3))  # (B,C)
    union = pred_m.sum(dim=(2, 3)) + gt_onehot.sum(dim=(2, 3))  # (B,C)

    dice_per_class = 1 - (2 * intersection + 1e-6) / (union + 1e-6)  # (B,C)

    # Average across classes & batch → scalar
    return dice_per_class.mean()


def calculate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    epoch_data: dict,
    key: str = "train",
) -> torch.Tensor:
    keep_mask = None
    if PHASE in {"s0", "s1", "s2", "s3"} and key != "validation":
        keep_mask = dropout_mask(
            labels,
        )

    if key == "validation":
        preds = logits.argmax(dim=1)  # Get the predicted class for each pixel
        flat_labels = labels.view(-1)
        jaccard_index = FM.jaccard_index(
            preds,
            labels,
            num_classes=len(CLASS_LIST),
            ignore_index=255,  # Ignore the background class
            task="multiclass",
            average="none",  # Calculate IoU for each class separately
        )
        # Accumulate Jaccard index for each class as scalars, but only for classes present in batch
        if epoch_data[key].get("jaccard_sum") is None:
            epoch_data[key]["jaccard_sum"] = [0.0] * len(CLASS_LIST)
        if epoch_data[key].get("jaccard_batch_count") is None:
            epoch_data[key]["jaccard_batch_count"] = [0] * len(CLASS_LIST)

        # Initialize confusion matrix accumulator
        if epoch_data[key].get("confusion_matrix") is None:
            epoch_data[key]["confusion_matrix"] = torch.zeros(
                len(CLASS_LIST), len(CLASS_LIST), dtype=torch.long
            )

        # Accumulate confusion matrix (only for valid pixels)
        valid_mask = labels != 255
        valid_preds = preds[valid_mask].cpu()
        valid_labels = labels[valid_mask].cpu()

        if len(valid_preds) > 0:  # Only if there are valid pixels
            batch_cm = confusion_matrix(
                valid_labels.numpy(),
                valid_preds.numpy(),
                labels=list(range(len(CLASS_LIST))),
                sample_weight=None,
            )
            epoch_data[key]["confusion_matrix"] += torch.from_numpy(batch_cm)

        # Check which classes are present in this batch
        for class_id in range(len(CLASS_LIST)):
            class_present = (flat_labels == class_id).any().item()
            if class_present:  # Only accumulate IoU for classes present in this batch
                epoch_data[key]["jaccard_sum"][class_id] += (
                    jaccard_index[class_id].float().item()
                )
                epoch_data[key]["jaccard_batch_count"][class_id] += 1

    total_loss = torch.tensor(0.0, device=logits.device)

    if epoch_data[key].get("focal_loss") is None:
        epoch_data[key]["focal_loss"] = 0.0

    if epoch_data[key].get("dice") is None:
        epoch_data[key]["dice"] = 0.0

    focal_loss = focal_ce(logits, labels, keep_mask=keep_mask)
    dice = dice_loss(
        logits.softmax(dim=1),  # Convert logits to probabilities
        labels,
        keep_mask=keep_mask,  # Apply keep_mask if it exists
    )

    epoch_data[key]["focal_loss"] += focal_loss.item()
    epoch_data[key]["dice"] += dice.item()

    # New weighting: Focal loss is the main component, Dice loss regularizes it.
    total_loss = 0.8 * focal_loss + 0.2 * dice

    epoch_data[key]["total_loss"] += total_loss.item()
    epoch_data[key]["batch_count"] += 1

    return total_loss


def calculate_final_statistics(
    epoch_data: dict,
    key: str = "train",
):
    if key == "validation":
        epoch_data[key]["per_class_iou"] = {}

        # Calculate average Jaccard index over batches where each class was present
        for class_id, class_name in enumerate(CLASS_LIST):
            class_batch_count = epoch_data[key]["jaccard_batch_count"][class_id]
            if class_batch_count > 0:
                avg_iou = epoch_data[key]["jaccard_sum"][class_id] / class_batch_count
            else:
                avg_iou = 0.0  # Class never appeared in any batch
            epoch_data[key]["per_class_iou"][class_name] = avg_iou

        # Calculate mean IoU across all classes (only for classes that appeared)
        valid_ious = []
        for class_id in range(len(CLASS_LIST)):
            class_batch_count = epoch_data[key]["jaccard_batch_count"][class_id]
            if class_batch_count > 0:
                avg_iou = epoch_data[key]["jaccard_sum"][class_id] / class_batch_count
                valid_ious.append(avg_iou)

        if valid_ious:
            epoch_data[key]["mean_iou"] = sum(valid_ious) / len(valid_ious)
        else:
            epoch_data[key]["mean_iou"] = 0.0

        # Clean up temporary accumulators
        del epoch_data[key]["jaccard_sum"]
        del epoch_data[key]["jaccard_batch_count"]

    epoch_data[key]["dice"] = epoch_data[key]["dice"] / epoch_data[key]["batch_count"]
    epoch_data[key]["focal_loss"] = (
        epoch_data[key]["focal_loss"] / epoch_data[key]["batch_count"]
    )

    epoch_data[key]["total_loss"] /= epoch_data[key]["batch_count"]


def print_confusion_matrix(epoch_data: dict):
    cm = epoch_data["validation"]["confusion_matrix"]

    # Calculate percentages (row-wise normalization)
    cm_percent = cm.float()
    row_sums = cm_percent.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_percent = (cm_percent / row_sums) * 100

    # Header
    header = "True\\Pred".ljust(12)
    for i, class_name in enumerate(CLASS_LIST):
        header += f"{class_name[:8]:>10}"
    print(header)

    # Matrix rows
    for i, class_name in enumerate(CLASS_LIST):
        row = f"{class_name[:8]:8}".ljust(12)
        for j in range(len(CLASS_LIST)):
            row += f"{cm_percent[i, j].item():>9.1f}%"
        print(row)
    print()  # Empty line after table
    del epoch_data["validation"]["confusion_matrix"]


def cycle(dl: DataLoader):
    while True:
        for batch in dl:
            yield batch


skyrim_train_dataset.set_transform(skyrim_transform_train_fn)
skyrim_validation_dataset.set_transform(skyrim_transform_val_fn)


def is_norm_param(name, module):
    return (
        isinstance(
            module,
            (
                torch.nn.LayerNorm,
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
                torch.nn.GroupNorm,
                torch.nn.InstanceNorm1d,
                torch.nn.InstanceNorm2d,
                torch.nn.InstanceNorm3d,
            ),
        )
        or "norm" in name.lower()
        or "bn" in name.lower()
        or "ln" in name.lower()
    )


# Training loop
def do_train():
    EPOCHS = 8

    print(
        f"Starting training for {EPOCHS} epochs, on {STEPS_PER_EPOCH_TRAIN * BATCH_SIZE} samples, validation on {len(skyrim_validation_dataset)} samples."
    )

    model, best_model_checkpoint = get_model()
    # for n, p in model.named_parameters():
    #     print(f"Parameter: {n}")

    # for p in model.parameters():
    #     p.requires_grad = False

    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only last encoder block (block.3) + decoder
    for n, p in model.named_parameters():
        if (
            "decode_head." in n
            or ".encoder.block.3." in n
            or ".patch_embeddings." in n
            or ".layer_norm." in n
            or ".encoder.block.2" in n
        ):
            p.requires_grad = True

    for n, p in model.named_parameters():
        if "decode_head." in n:
            p.requires_grad = True

    # for n, p in model.named_parameters():
    #     # freeze all blocks in stage-0, not freezing patch embeddings 0
    #     if ".encoder.block.0." in n:
    #         p.requires_grad = False

    # # Override the BN momentum on all BatchNorm2d layers
    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         # Default momentum is 0.1; we lower it so running stats update faster
    #         m.momentum = 0.01

    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(f"Trainable parameter: {n}")

    skyrim_train_loader = DataLoader(
        skyrim_train_dataset,
        batch_size=BATCH_SIZE_SKYRIM,
        sampler=skyrim_train_sampler,
        num_workers=SKYRIM_WORKERS,
        prefetch_factor=2,
        shuffle=False,
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

    skyrim_train_iter = cycle(skyrim_train_loader)

    LR_DEC = 1e-4
    LR_ENC = 1e-5
    # WD = 1e-2

    # --- map every parameter to its encoder depth (None = decoder / head) ---
    gamma = 0.9
    depth_map = {}

    encoder_blocks = model.segformer.encoder.block  # type: ignore # list of 4 lists-of-MixFFNs
    for depth, blk in enumerate(encoder_blocks):  # type: ignore
        for p in blk.parameters(recurse=True):
            depth_map[id(p)] = depth

    for i, pe in enumerate(model.segformer.encoder.patch_embeddings):  # type: ignore
        for p in pe.parameters(recurse=True):
            depth_map[id(p)] = i
    for i, ln in enumerate(model.segformer.encoder.layer_norm):  # type: ignore
        for p in ln.parameters(recurse=True):
            depth_map[id(p)] = i

    param_groups = {}  # key = (lr, wd) ➜ list(params)

    for module_name, module in model.named_modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue

            full_name = f"{module_name}.{name}" if module_name else name

            # Decide weight-decay
            wd = (
                0.0  # No weight decay for normalization layers
                if (
                    is_norm_param(name, module)
                    or "pos_embed" in full_name
                    or "position" in full_name
                )
                else (
                    1e-3 if full_name.endswith(".bias") else 1e-2
                )  # Stronger WD for main weights (increased from 1e-2)
            )

            # Decide learning-rate (encoder depth or decoder/head)
            depth = depth_map.get(id(p), None)
            if depth is None:
                lr = LR_DEC  # decoder / head
            else:
                # LLRD: deepest block (B3) gets full LR, shallower get LR*γ^(Δdepth)
                lr = LR_ENC * (gamma ** (3 - depth))

            # print(f"Parameter: {full_name}, LR: {lr}, WD: {wd}, Depth: {depth}")

            # Collect by (lr, wd)
            param_groups.setdefault((lr, wd), []).append(p)

    optimizer_groups = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in param_groups.items()
    ]

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    if best_model_checkpoint is not None and resume_training:
        print("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(best_model_checkpoint["optimizer_state_dict"])

    effective_scheduler_steps = (
        int(STEPS_PER_EPOCH_TRAIN / 2) if USE_ACCUMULATION else STEPS_PER_EPOCH_TRAIN
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=EPOCHS * effective_scheduler_steps, eta_min=2e-6
    # )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=effective_scheduler_steps // 2,  # Half epoch warmup
    )

    cosine_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int((EPOCHS - 0.5) * effective_scheduler_steps), eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine_stage1],
        milestones=[
            effective_scheduler_steps // 2
        ],  # After half epoch switch to cosine
    )

    # warmup = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=0.1,
    #     end_factor=1.0,
    #     total_iters=effective_scheduler_steps,
    # )

    # cosine_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=(EPOCHS - 1) * effective_scheduler_steps, eta_min=1e-6
    # )

    # Stage 3: Aggressive cosine decay from epoch 3 onwards (0.3x current LR as max)
    # cosine_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=(EPOCHS - 2) * effective_scheduler_steps, eta_min=2e-6
    # )

    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     # schedulers=[warmup, cosine_stage1, cosine_stage2],
    #     schedulers=[warmup, cosine_stage1],
    #     milestones=[
    #         effective_scheduler_steps,  # After epoch 1 switch to cosine
    #         # 2 * effective_scheduler_steps,  # After epoch 2 switch to aggressive decay
    #     ],
    # )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=LR_DEC,  # 4e-4 according to README
    #     total_steps=EPOCHS * STEPS_PER_EPOCH_TRAIN,
    #     # 15% warm-up 85% cooldown per README
    #     pct_start=0.15,
    #     div_factor=4.0,  # start LR = max_lr/4 = 1e-4
    #     # div_factor=10.0,  # start LR = max_lr/10 = 4e-5
    #     final_div_factor=40.0,  # End LR = max_lr/final_div = 1e-5
    # )
    if best_model_checkpoint is not None and resume_training:
        print("Loading scheduler state from checkpoint.")
        scheduler.load_state_dict(best_model_checkpoint["scheduler_state_dict"])

        # Manual LR reduction after resuming
        if args.lr_scale != 1.0:
            print(f"Manually scaling learning rate by factor of {args.lr_scale}")
            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                param_group["lr"] = old_lr * args.lr_scale
                print(f"LR changed from {old_lr:.2e} to {param_group['lr']:.2e}")

    scaler = GradScaler(device.type)  # AMP scaler for mixed precision
    if best_model_checkpoint is not None and resume_training:
        print("Loading scaler state from checkpoint.")
        scaler.load_state_dict(best_model_checkpoint["scaler_state_dict"])

    best_val_loss = float("inf")
    if best_model_checkpoint is not None and args.load_best_loss and resume_training:
        # Use the correct key from epoch_data structure
        try:
            best_val_loss = best_model_checkpoint["epoch_data"]["validation"][
                "total_loss"
            ]
            print(f"Loaded best validation loss: {best_val_loss}")
        except KeyError:
            print(
                "Warning: Could not load best validation loss from checkpoint, using inf"
            )
            best_val_loss = float("inf")

    patience = 6
    no_improvement_count = 0

    output_dir = Path(f"./weights/{PHASE}/segformer")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if best_model_checkpoint is not None and resume_training:
        start_epoch = best_model_checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()

        epoch_data = {
            "epoch": epoch + 1,
            "train": {
                "total_loss": 0.0,
                "batch_count": 0,
            },
            "validation": {
                "total_loss": 0.0,
                "batch_count": 0,
            },
        }

        accum_steps = 2

        bar = tqdm(
            range(STEPS_PER_EPOCH_TRAIN),
            desc=f"Epoch {epoch + 1}/{EPOCHS} - Training",
            unit="batch",
        )

        if USE_ACCUMULATION:
            optimizer.zero_grad(set_to_none=True)

        for i in bar:
            skyrim_batch = next(skyrim_train_iter)

            input = skyrim_batch["pixel_values"]
            labels_gt = skyrim_batch["labels"]

            input = input.to(device, non_blocking=True)
            labels_gt = labels_gt.to(device, non_blocking=True)

            if not USE_ACCUMULATION:
                optimizer.zero_grad()

            with autocast(device_type=device.type):
                logits: torch.Tensor = model(pixel_values=input)["logits"]

                # upsample logits to match the input size
                logits_up: torch.Tensor = torch.nn.functional.interpolate(
                    logits,
                    size=input.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

                skyrim_loss = calculate_loss(
                    logits_up,
                    labels_gt,
                    epoch_data,
                    key="train",
                )
                if torch.isnan(skyrim_loss):
                    raise ValueError("Loss is NaN")

                total_loss = skyrim_loss

            # loss.backward()
            # optimizer.step()

            if USE_ACCUMULATION:
                total_loss = total_loss / accum_steps  # Scale loss for accumulation

            scaler.scale(total_loss).backward()

            if not USE_ACCUMULATION:
                # — Gradient clipping —
                scaler.unscale_(optimizer)  # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            if USE_ACCUMULATION:
                if (i + 1) % accum_steps == 0 or (i + 1) == STEPS_PER_EPOCH_TRAIN:
                    # — Gradient clipping —
                    scaler.unscale_(optimizer)  # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    # Step per effective batch
                    scheduler.step()

        calculate_final_statistics(epoch_data, key="train")

        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(
                tqdm(
                    skyrim_validation_loader,
                    desc=f"Epoch {epoch + 1}/{EPOCHS} - Skyrim Validation",
                    unit="batch",
                )
            ):
                input = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    logits: torch.Tensor = model(pixel_values=input)["logits"]

                    # upsample logits to match the input size
                    logits_up: torch.Tensor = torch.nn.functional.interpolate(
                        logits,
                        size=input.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                    loss = calculate_loss(
                        logits_up,
                        labels,
                        epoch_data,
                        key="validation",
                    )

                    if torch.isnan(loss):
                        raise ValueError("Loss is NaN")

            calculate_final_statistics(epoch_data, key="validation")

        # use only skyrim validation loss for early stopping
        epoch_val_loss = epoch_data["validation"]["total_loss"]

        # scheduler.step()

        print_confusion_matrix(epoch_data)
        print(json.dumps(epoch_data, indent=4))

        # Save checkopoint after each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                # "base_model_state_dict": model.base_model.state_dict(),
                # "lora_state_dict": model.state_dict(),
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

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            no_improvement_count = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    # "base_model_state_dict": model.base_model.state_dict(),
                    # "lora_state_dict": model.state_dict(),
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
                f"No improvement at epoch {epoch + 1}, validation loss: {epoch_val_loss:.4f}"
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
