# Ensure we import it here to set random(seed)
import seed
import json, torch
import multiprocessing
import torch.nn.functional as F
import lpips
import argparse
from typing import Callable
from unet_models import UNetAlbedo, UNetMaps
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from torchmetrics import functional as FM
from transformers.utils.constants import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from train_dataset import SimpleImageDataset, normalize_normal_map
from augmentations import (
    get_random_crop,
    selective_aug,
    center_crop,
)
from segformer_6ch import create_segformer

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Train UNet-Albedo")
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


# Create segformer and load best weights
segformer = create_segformer(
    num_labels=len(train_dataset.CLASS_LIST),  # Number of classes for segmentation
    device=device,
)
segformer_best_weights_path = (
    BASE_DIR / "../weights/a/segformer/best_model.pt"
).resolve()
segformer_checkpoint = torch.load(segformer_best_weights_path, map_location=device)
segformer.load_state_dict(
    segformer_checkpoint["model_state_dict"],
)
# Freeze segformer parameters
for param in segformer.parameters():
    param.requires_grad = False


def get_transform_train(
    current_epoch: int,
    safe_augmentations=True,
    color_augmentations=True,
) -> Callable:
    def transform_train_fn(example):
        albedo = example["basecolor"]
        normal = example["normal"]
        diffuse = example["diffuse"]
        category = example["category"]
        category_name = example["category_name"]
        name = example["name"]

        albedo, normal, diffuse, *_ = get_random_crop(
            albedo=albedo,
            normal=normal,
            size=(256, 256),  # Crop size for training
            diffuse=diffuse,
            resize_to=[1024, 1024],  # Resize to 1024x1024 for training
            augmentations=safe_augmentations,
        )

        albedo_orig = albedo
        albedo_segformer = albedo

        if color_augmentations:
            diffuse = selective_aug(diffuse, category=category_name)

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

        albedo_orig = TF.to_tensor(albedo_orig)

        albedo_segformer = TF.to_tensor(albedo_segformer)
        albedo_segformer = TF.normalize(
            albedo_segformer, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
        albedo_and_normal_segformer = torch.cat((albedo_segformer, normal), dim=0)

        return {
            "diffuse_and_normal": diffuse_and_normal,
            "albedo_and_normal_segformer": albedo_and_normal_segformer,
            "albedo": albedo_orig,
            "normal": normal,
            "category": category,
            "name": name,
        }

    return transform_train_fn


def transform_val_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    diffuse = example["diffuse"]
    category = example["category"]
    name = example["name"]

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

    albedo_orig = albedo
    albedo_segformer = albedo

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

    albedo_orig = TF.to_tensor(albedo_orig)

    albedo_segformer = TF.to_tensor(albedo_segformer)
    albedo_segformer = TF.normalize(
        albedo_segformer, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )

    albedo_and_normal_segformer = torch.cat((albedo_segformer, normal), dim=0)

    return {
        "diffuse_and_normal": diffuse_and_normal,
        "albedo_and_normal_segformer": albedo_and_normal_segformer,
        "albedo": albedo_orig,
        "normal": normal,
        "category": category,
        "name": name,
        "original_diffuse": original_diffuse,
        "original_normal": original_normal,
    }


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

    return epoch_data["unet_albedo"][key]["total_loss"]


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
        # Hungs sometimes if enabled
        # pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation data
        # num_workers=6,
        # Hungs sometimes if enabled
        # pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet_alb.parameters()),
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
    if checkpoint is not None and args.load_best_loss and resume_training:
        best_val_loss_albedo = checkpoint["epoch_data"]["unet_albedo"]["validation"][
            "total_loss"
        ]
        print(f"Loading best validation loss from checkpoint: {best_val_loss_albedo}")

    patience = 6
    no_improvement_count_albedo = 0

    output_dir = Path(f"./weights/{PHASE}/unet_albedo")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if checkpoint is not None and resume_training:
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, EPOCHS):
        unet_alb.train()

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
            category = batch["category"]
            albedo_gt = batch["albedo"]
            albedo_and_normal_segformer = batch["albedo_and_normal_segformer"]

            diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
            albedo_gt = albedo_gt.to(device, non_blocking=True)
            albedo_and_normal_segformer = albedo_and_normal_segformer.to(
                device, non_blocking=True
            )

            with torch.no_grad():
                #  Get Segoformer ouput for FiLM
                with autocast(device_type=device.type):
                    seg_feats = (
                        segformer(
                            albedo_and_normal_segformer, output_hidden_states=True
                        )
                        .hidden_states[-1]
                        .detach()
                    )  # (B,256,H/16,W/16)

            with autocast(device_type=device.type):
                # Get UNet-Albedo prediction
                albedo_pred = unet_alb(diffuse_and_normal, seg_feats)

            # Unet-albedo loss
            unet_albedo_loss = calculate_unet_albedo_loss(
                albedo_pred, albedo_gt, category, epoch_data, key="train"
            )
            if torch.isnan(unet_albedo_loss):
                raise ValueError(
                    "Unet-Albedo loss is NaN, stopping training to avoid further issues."
                )

            epoch_data["train"]["batch_count"] += 1

            # Total loss
            total_loss = unet_albedo_loss

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
        samples_saved_per_class = {name: 0 for name in train_dataset.CLASS_LIST}

        with torch.no_grad():
            for batch in tqdm(
                validation_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation"
            ):
                diffuse_and_normal = batch["diffuse_and_normal"]
                albedo_and_normal_segformer = batch["albedo_and_normal_segformer"]
                albedo_gt = batch["albedo"]
                normal = batch["normal"]
                category = batch["category"]
                names = batch["name"]
                original_diffuse = batch["original_diffuse"]
                original_normal = batch["original_normal"]

                diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
                albedo_and_normal_segformer = albedo_and_normal_segformer.to(
                    device, non_blocking=True
                )
                normal = normal.to(device, non_blocking=True)
                albedo_gt = albedo_gt.to(device, non_blocking=True)
                original_diffuse = original_diffuse.to(device, non_blocking=True)
                original_normal = original_normal.to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    seg_feats = (
                        segformer(
                            albedo_and_normal_segformer, output_hidden_states=True
                        )
                        .hidden_states[-1]
                        .detach()
                    )  # (B,256,H/16,W/16)
                    albedo_pred = unet_alb(diffuse_and_normal, seg_feats)

                calculate_unet_albedo_loss(
                    albedo_pred, albedo_gt, category, epoch_data, key="validation"
                )

                epoch_data["validation"]["batch_count"] += 1

                for k in range(len(category)):
                    # Accumulate per-class loss
                    cat_name = train_dataset.CLASS_LIST[
                        category[k].item()
                    ]  # Get the category name

                    if samples_saved_per_class[cat_name] < 8:
                        # Save few samples per class for inspection
                        output_path = output_dir / f"val_samples_{epoch + 1}/{cat_name}"
                        output_path.mkdir(parents=True, exist_ok=True)

                        # Save diffuse, normal, GT albedo and predicted albedo side by side
                        visual_sample_gt = torch.cat(
                            [
                                original_diffuse[k],  # Diffuse
                                original_normal[k],  # Normal
                                albedo_gt[k],  # GT Albedo
                            ],
                            dim=2,  # Concatenate along width
                        )
                        visual_sample_predicted = torch.cat(
                            [
                                original_diffuse[k],  # Diffuse
                                original_normal[k],  # Normal
                                albedo_pred[k],  # Predicted Albedo
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

                        save_image(
                            visual_sample, output_path / f"{cat_name}_{names[k]}.png"
                        )

                        samples_saved_per_class[cat_name] += 1

        unet_albedo_total_val_loss = calculate_avg(epoch_data, key="validation")

        print(json.dumps(epoch_data, indent=4))

        # Save checkopoint after each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "unet_albedo_model_state_dict": unet_alb.state_dict(),
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
            torch.save(
                {
                    "epoch": epoch + 1,
                    "unet_albedo_model_state_dict": unet_alb.state_dict(),
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
                f"Saved new best model at epoch {epoch + 1} with loss {best_val_loss_albedo:.4f}"
            )
        else:
            no_improvement_count_albedo += 1
            print(
                f"UNet-Albedo: no improvement at epoch {epoch + 1}, validation loss: {unet_albedo_total_val_loss:.4f}"
            )
            if no_improvement_count_albedo >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}, no improvement for {patience} epochs."
                )
                break

    print("Training completed.")


if __name__ == "__main__":
    # On Windows frozen executables need this; harmless otherwise
    multiprocessing.freeze_support()
    do_train()
