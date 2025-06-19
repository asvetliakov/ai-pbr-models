# Ensure we import it here to set random(seed)
import seed
import json, torch
import numpy as np
import multiprocessing
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
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

BASE_DIR = Path(__file__).resolve().parent

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

# HYPER_PARAMETERS
BATCH_SIZE = 4  # Batch size for training
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

# loss_weights, sample_weights = train_dataset.get_weights()

# seg_loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights, ignore_index=255)
seg_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
# seq_per_class_fn = torch.nn.CrossEntropyLoss(
#     weight=loss_weights, ignore_index=255, reduction="none"
# )
seq_per_class_fn = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="none")

# ! DISABLED FOR PHASE A0, REENABLE FOR PHASE A+
# # Will pull random samples according to the sample weights
# train_sampler = WeightedRandomSampler(
#     weights=sample_weights.tolist(),
#     num_samples=len(sample_weights),
#     replacement=True,
# )


device = torch.device("cuda")

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

transform_train = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]
)

transform_train_normal = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_STANDARD_MEAN,
            std=IMAGENET_STANDARD_STD,
        ),
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


transform_val = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]
)

transform_val_normal = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_STANDARD_MEAN,
            std=IMAGENET_STANDARD_STD,
        ),
    ]
)


def synced_crop_and_resize(
    albedo: Image.Image,
    normal: Image.Image,
    size: tuple[int, int],
    resize_to: list[int],
) -> tuple[Image.Image, Image.Image]:
    """
    Crop and resize two images to the same size.
    """
    i, j, h, w = T.RandomCrop.get_params(albedo, output_size=size)  # type: ignore

    albedo_crop = TF.crop(albedo, i, j, h, w)  # type: ignore
    normal_crop = TF.crop(normal, i, j, h, w)  # type: ignore

    albedo_resize = TF.resize(
        albedo_crop, resize_to, interpolation=T.InterpolationMode.LANCZOS
    )
    normal_resize = TF.resize(
        normal_crop, resize_to, interpolation=T.InterpolationMode.BILINEAR
    )
    normal_resize = normalize_normal_map(normal_resize)  # type: ignore

    # image not tensors
    return albedo_resize, normal_resize  # type: ignore


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
    category = example["category"]

    albedo, normal = synced_crop_and_resize(
        albedo, normal, size=(256, 256), resize_to=[1024, 1024]
    )
    albedo = transform_train(albedo)
    normal = transform_train_normal(normal)

    # Concatenate albedo and normal along the channel dimension
    final = torch.cat((albedo, normal), dim=0)  # type: ignore

    mask = make_full_image_mask(category_id=category, img_size=(1024, 1024))  # (H, W)

    return {
        "pixel_values": final,
        "labels": mask,
        "category": category,  # keep for reference
    }


def transform_val_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    category = example["category"]

    albedo = transform_val(
        center_crop(albedo, (256, 256), [1024, 1024], T.InterpolationMode.LANCZOS)
    )

    normal = transform_val_normal(
        normalize_normal_map(
            center_crop(normal, (256, 256), [1024, 1024], T.InterpolationMode.BILINEAR)
        )
    )

    # Concatenate albedo and normal along the channel dimension
    final = torch.cat((albedo, normal), dim=0)  # type: ignore

    mask = make_full_image_mask(category_id=category, img_size=(1024, 1024))

    return {
        "pixel_values": final,
        "labels": mask,
        "category": category,  # keep for reference
    }


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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    # scaler = GradScaler(device.type)  # AMP scaler for mixed precision

    best_val_loss = float("inf")
    patience = 3
    no_improvement_count = 0

    output_dir = Path(f"./weights/{PHASE}/segformer")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()

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

            # with autocast(device_type=device.type):
            logits = model(input).logits

            # upsample logits to match the input size
            logits_up = torch.nn.functional.interpolate(
                logits,
                size=input.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = seg_loss_fn(logits_up, labels_gt)

            train_loss_sum += loss.item()
            train_batch_count += 1

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

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

                # with autocast(device_type=device.type):
                logits = model(input).logits

                # upsample logits to match the input size
                logits_up = torch.nn.functional.interpolate(
                    logits,
                    size=input.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = seg_loss_fn(logits_up, labels_gt)
                val_loss_sum += loss.item()
                val_batch_count += 1

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

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            no_improvement_count = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
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

        # Save checkopoint after each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
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
