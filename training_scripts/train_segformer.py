# Ensure we import it here to set random(seed)
import seed
import json, torch
import numpy as np
import multiprocessing
import random
from typing import Callable
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
import argparse

# from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from torchmetrics import functional as FM
from transformers import (
    SegformerForSemanticSegmentation,
)
from transformers.utils.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)
from train_dataset import SimpleImageDataset, normalize_normal_map
from augmentations import (
    get_random_crop,
    make_full_image_mask,
    selective_aug,
    center_crop,
)

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

BASE_DIR = Path(__file__).resolve().parent

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

# HYPER_PARAMETERS
BATCH_SIZE = 4  # Batch size for training
EPOCHS = 35  # Number of epochs to train
LR = 5e-5  # Learning rate for the optimizer
WD = 1e-2  # Weight decay for the optimizer
# T_MAX = 10  # Max number of epochs for the learning rate scheduler
# PHASE = "a"  # Phase of the training per plan, used for logging and saving
PHASE = args.phase  # Phase of the training per plan, used for logging and saving

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable for faster training on fixed input sizes

matsynth_dir = (BASE_DIR / "../matsynth_processed").resolve()

device = torch.device("cuda")

train_dataset = SimpleImageDataset(
    matsynth_dir=str(matsynth_dir),
    split="train",
)

validation_dataset = SimpleImageDataset(
    matsynth_dir=str(matsynth_dir),
    split="validation",
    skip_init=True,
)
validation_dataset.all_validation_samples = train_dataset.all_validation_samples

loss_weights, sample_weights = train_dataset.get_weights()

loss_weights = loss_weights.to(device)  # type: ignore

seg_loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights, ignore_index=255)
seq_per_class_fn = torch.nn.CrossEntropyLoss(
    weight=loss_weights, ignore_index=255, reduction="none"
)

# # Will pull random samples according to the sample weights
train_sampler = WeightedRandomSampler(
    weights=sample_weights.tolist(),
    num_samples=len(sample_weights),
    replacement=True,
)


model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=len(train_dataset.CLASS_LIST),  # Number of classes for segmentation
    ignore_mismatched_sizes=True,  # Ignore size mismatch for classification head
).to(
    device  # type: ignore
)

# Patch segformer for 6channel input
old_embed = model.segformer.encoder.patch_embeddings[0]
old_conv = old_embed.proj  # Conv2d(3,64,kernel=7,stride=4,pad=3)

# 2) Create a new one for 6-channel input
new_conv = torch.nn.Conv2d(
    in_channels=6,  # RGB + Normal
    out_channels=old_conv.out_channels,  # 64 # type: ignore
    kernel_size=old_conv.kernel_size,  # (7,7) # type: ignore
    stride=old_conv.stride,  # (4,4) # type: ignore
    padding=old_conv.padding,  # (3,3) # type: ignore
    bias=old_conv.bias is not None,  # True if bias is used # type: ignore
).to(
    device
)  # type: ignore

# 3) Copy pretrained RGB weights → channels 0–2
with torch.no_grad():
    new_conv.weight[:, :3, :, :] = old_conv.weight  # type: ignore
    # 4) Init the extra normal channels → 3–5
    torch.nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode="fan_out")
    if old_conv.bias is not None:  # type: ignore
        new_conv.bias.copy_(old_conv.bias)  # type: ignore # Copy bias if it exists

# 5) Replace it back in the model
model.segformer.encoder.patch_embeddings[0].proj = new_conv

# 6) Update config so future code knows to expect 6 channels
model.config.num_channels = 6

resume_training = args.resume
if (args.load_checkpoint is not None) and Path(args.load_checkpoint).resolve().exists():
    load_checkpoint_path = Path(args.load_checkpoint).resolve()
    print(
        f"Loading model from checkpoint: {load_checkpoint_path}, resume={resume_training}"
    )
    best_model_checkpoint = torch.load(load_checkpoint_path, map_location=device)

if best_model_checkpoint is not None:
    print("Loading model state from checkpoint.")
    model.load_state_dict(
        best_model_checkpoint["model_state_dict"],
    )


def get_transform_train(
    current_epoch: int,
    safe_augmentations=True,
    composites=True,
    color_augmentations=True,
) -> Callable:
    def transform_train_fn(example):
        # name = example["name"]

        # Upper left corner tuple for each cro
        positions = [(0, 0)]
        # h, w
        crop_size = (256, 256)
        # h, w
        tile_size = [1024, 1024]
        samples = [example]

        if composites:
            # 15% chance of 4 random crops
            if random.random() < 0.15:
                positions = [(0, 0), (512, 0), (0, 512), (512, 512)]
                tile_size = [512, 512]
                crop_size = (256, 256)
                samples = [
                    example,
                    train_dataset.get_random_sample(),
                    train_dataset.get_random_sample(),
                    train_dataset.get_random_sample(),
                ]
            # 30% chance of 2 random crops
            elif random.random() < 0.3:
                positions = [(0, 0), (512, 0)]
                tile_size = [1024, 512]
                crop_size = (512, 256)
                samples = [
                    example,
                    train_dataset.get_random_sample(),
                ]

        final_albedo = Image.new("RGB", (1024, 1024))
        final_normal = Image.new("RGB", (1024, 1024))
        final_mask = torch.zeros((1024, 1024), dtype=torch.int64)
        # final_color_mask = Image.new("RGB", (1024, 1024))

        for sample, pos in zip(samples, positions):
            albedo = sample["basecolor"]
            normal = sample["normal"]
            category = sample["category"]
            category_name = sample["category_name"]

            albedo, normal = get_random_crop(
                albedo,
                normal,
                size=crop_size,
                augmentations=safe_augmentations,
                resize_to=None,
            )
            if color_augmentations:
                albedo, normal = selective_aug(
                    albedo,
                    normal,
                    category=category_name,
                )

            albedo = TF.resize(
                albedo, tile_size, interpolation=TF.InterpolationMode.LANCZOS  # type: ignore
            )
            normal = TF.resize(
                normal, tile_size, interpolation=TF.InterpolationMode.BILINEAR  # type: ignore
            )
            normal = normalize_normal_map(normal)  # type: ignore

            final_albedo.paste(albedo, box=pos)  # type: ignore
            final_normal.paste(normal, box=pos)  # type: ignore

            mask = make_full_image_mask(
                category_id=category,
                # height comes first
                img_size=(tile_size[0], tile_size[1]),
            )  # (H, W)

            final_mask[
                pos[1] : pos[1] + tile_size[0], pos[0] : pos[0] + tile_size[1]
            ] = mask

            # Mask visualization
            # color_mask = np.zeros((tile_size[0], tile_size[1], 3), dtype=np.uint8)
            # color_cat = PALETTE[category]  # type: ignore
            # color_mask[:, :] = color_cat  # type: ignore
            # final_color_mask.paste(
            #     Image.fromarray(color_mask, mode="RGB"), box=pos  # type: ignore
            # )

        # visual_check = Image.new("RGB", (2048, 1024))
        # visual_check.paste(final_albedo, (0, 0))  # type: ignore
        # visual_check.paste(final_color_mask, (1024, 0))  # type: ignore
        # visual_check.paste(final_normal, (1024, 0))  # type: ignore
        # img_test_dir.mkdir(parents=True, exist_ok=True)
        # visual_check.save(
        #     img_test_dir / f"{name}.png",
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

    return transform_train_fn


def transform_val_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    category = example["category"]

    albedo = center_crop(albedo, (256, 256), [1024, 1024], TF.InterpolationMode.LANCZOS)
    albedo = TF.to_tensor(albedo)
    albedo = TF.normalize(albedo, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    normal = normalize_normal_map(
        center_crop(normal, (256, 256), [1024, 1024], TF.InterpolationMode.BILINEAR)
    )
    normal = TF.to_tensor(normal)
    normal = TF.normalize(
        normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    # Concatenate albedo and normal along the channel dimension
    final = torch.cat((albedo, normal), dim=0)  # type: ignore

    mask = make_full_image_mask(category_id=category, img_size=(1024, 1024))

    return {
        "pixel_values": final,
        "labels": mask,
        "category": category,  # keep for reference
    }


# Training loop
def do_train():
    print(
        f"Starting training for {EPOCHS} epochs, on {len(train_dataset)} samples, validation on {len(validation_dataset)} samples."
    )

    train_loader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        # num_workers=4,
        shuffle=False,
    )

    validation_loader = DataLoader(
        validation_dataset,  # type: ignore
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation data
        # num_workers=6,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    if best_model_checkpoint is not None and resume_training:
        print("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(best_model_checkpoint["optimizer_state_dict"])

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
    if best_model_checkpoint is not None and resume_training:
        print("Loading scheduler state from checkpoint.")
        scheduler.load_state_dict(best_model_checkpoint["scheduler_state_dict"])

    scaler = GradScaler(device.type)  # AMP scaler for mixed precision
    if best_model_checkpoint is not None and resume_training:
        print("Loading scaler state from checkpoint.")
        scaler.load_state_dict(best_model_checkpoint["scaler_state_dict"])

    best_val_loss = float("inf")
    patience = 6
    patience_min_delta = 0.005
    no_improvement_count = 0

    output_dir = Path(f"./weights/{PHASE}/segformer")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if best_model_checkpoint is not None and resume_training:
        print(f"Resuming training from epoch {start_epoch}.")
        start_epoch = best_model_checkpoint["epoch"]

    for epoch in range(start_epoch, EPOCHS):
        model.train()

        train_dataset.set_transform(
            get_transform_train(
                epoch + 1,
                # Composites & flips are enabled from epoch 1
                safe_augmentations=True,
                composites=True,
                # Color augmentations are enabled after warm-up (from epoch 6)
                color_augmentations=(epoch + 1) > 5,
            )
        )
        validation_dataset.set_transform(transform_val_fn)

        # For IoU
        # ! need to reest it here early due to GPU memory issues
        val_all_labels = []
        val_all_preds = []

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "per_class_loss": {name: 0.0 for name in train_dataset.CLASS_LIST},
            "IoU": {name: 0.0 for name in train_dataset.CLASS_LIST},
            "mIoU": 0.0,
        }

        train_loss_sum = 0.0
        train_batch_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training"):
            input = batch["pixel_values"]
            labels_gt = batch["labels"]

            input = input.to(device, non_blocking=True)
            labels_gt = labels_gt.to(device, non_blocking=True)

            with autocast(device_type=device.type):
                logits = model(input).logits

            # upsample logits to match the input size
            logits_up = torch.nn.functional.interpolate(
                logits,
                size=input.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            with autocast(device_type=device.type):
                loss = seg_loss_fn(logits_up, labels_gt)

            # Alternatively can skip batch but let's see how it goes
            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            train_loss_sum += loss.item()
            train_batch_count += 1

            optimizer.zero_grad()

            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss_avg = train_loss_sum / train_batch_count
        epoch_data["train_loss"] = train_loss_avg

        model.eval()
        val_loss_sum = 0.0
        val_batch_count = 0

        # Per class cross-entropy loss
        val_total_loss_per_class = torch.zeros(
            len(train_dataset.CLASS_LIST), dtype=torch.float32, device="cuda"
        )
        val_total_pixels_per_class = torch.zeros(
            len(train_dataset.CLASS_LIST), dtype=torch.float32, device="cuda"
        )

        with torch.no_grad():
            for batch in tqdm(
                validation_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation"
            ):
                input = batch["pixel_values"]
                labels_gt = batch["labels"]
                input = input.to(device, non_blocking=True)
                labels_gt = labels_gt.to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    logits = model(input).logits

                # upsample logits to match the input size
                logits_up = torch.nn.functional.interpolate(
                    logits,
                    size=input.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                with autocast(device_type=device.type):
                    loss = seg_loss_fn(logits_up, labels_gt)

                if torch.isnan(loss):
                    raise ValueError("Loss is NaN")

                val_loss_sum += loss.item()
                val_batch_count += 1

                with autocast(device_type=device.type):
                    pixel_loss = seq_per_class_fn(logits_up, labels_gt)

                flat_loss = pixel_loss.view(-1)
                flat_labels = labels_gt.view(-1)

                for c in range(len(train_dataset.CLASS_LIST)):
                    mask = flat_labels == c
                    val_total_loss_per_class[c] += flat_loss[mask].sum()
                    val_total_pixels_per_class[c] += mask.sum()

                labels_pred = logits_up.argmax(dim=1)
                val_all_preds.append(labels_pred)
                val_all_labels.append(labels_gt)

        val_loss_avg = val_loss_sum / val_batch_count
        epoch_data["val_loss"] = val_loss_avg

        # Calculate IoU
        val_all_preds = torch.cat(val_all_preds, dim=0)
        val_all_labels = torch.cat(val_all_labels, dim=0)

        jaccard_index = FM.jaccard_index(
            val_all_preds,
            val_all_labels,
            num_classes=len(train_dataset.CLASS_LIST),
            ignore_index=255,  # Ignore the background class
            task="multiclass",
            average="none",  # Calculate IoU for each class separately
        )
        for idx, name in enumerate(train_dataset.CLASS_LIST):
            epoch_data["IoU"][name] = jaccard_index[idx].item()

        epoch_data["mIoU"] = jaccard_index.mean().item()

        val_avg_loss_per_class = val_total_loss_per_class / (
            val_total_pixels_per_class + 1e-6
        )
        for idx, name in enumerate(train_dataset.CLASS_LIST):
            epoch_data["per_class_loss"][name] = val_avg_loss_per_class[idx].item()

        print(json.dumps(epoch_data, indent=4))

        # Save checkopoint after each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
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

        if val_loss_avg < best_val_loss - patience_min_delta:
            best_val_loss = val_loss_avg
            no_improvement_count = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
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

        scheduler.step()

    print("Training completed.")


if __name__ == "__main__":
    # On Windows frozen executables need this; harmless otherwise
    multiprocessing.freeze_support()
    do_train()
