# Ensure we import it here to set random(seed)
import seed
import json, torch
import numpy as np
import random
import multiprocessing
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from datasets import load_dataset, Dataset, ClassLabel
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchmetrics import functional as FM
from transformers import (
    SegformerForSemanticSegmentation,
)
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# from torch.amp.grad_scaler import GradScaler
# from torch.amp.autocast_mode import autocast

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

# Loss weights are inversely proportional to the square root of the frequency of each class
loss_weights = 1.0 / torch.sqrt(freq + 1e-6)  # avoid ÷0
loss_weights *= num_classes / loss_weights.sum()
# seg_loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights, ignore_index=255)
seg_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
# seq_per_class_fn = torch.nn.CrossEntropyLoss(
#     weight=loss_weights, ignore_index=255, reduction="none"
# )
seq_per_class_fn = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="none")
print("Loss weights:", loss_weights)

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


device = torch.device("cuda")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=len(CLASS_LIST),  # Number of classes for segmentation
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
        # Take random crop (consult augment table in plan)
        # T.RandomCrop(size=(256, 256)),
        # T.Resize((1024, 1024), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]
)


def transform_val(
    image: Image.Image, interpolation: T.InterpolationMode
) -> torch.Tensor:
    composed = T.Compose(
        [
            # Need to use same crop size for validation
            T.CenterCrop(size=(256, 256)),
            T.Resize((1024, 1024), interpolation=interpolation),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ]
    )
    return composed(image)  # type: ignore


def synced_crop_and_resize(
    albedo: Image.Image, normal: Image.Image, size: tuple[int, int]
) -> tuple[Image.Image, Image.Image]:
    """
    Crop and resize two images to the same size.
    """
    i, j, h, w = T.RandomCrop.get_params(albedo, output_size=size)  # type: ignore

    albedo_crop = TF.crop(albedo, i, j, h, w)  # type: ignore
    normal_crop = TF.crop(normal, i, j, h, w)  # type: ignore

    albedo_resize = TF.resize(
        albedo_crop, [1024, 1024], interpolation=T.InterpolationMode.LANCZOS
    )
    normal_resize = TF.resize(
        normal_crop, [1024, 1024], interpolation=T.InterpolationMode.BILINEAR
    )

    # image not tensors
    return albedo_resize, normal_resize  # type: ignore


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
    # Transform
    transformed_albedo = [image.convert("RGB") for image in examples["basecolor"]]
    transformed_normal = [image.convert("RGB") for image in examples["normal"]]

    # Check if there any examples with "none" category and warn if so
    if NONE_IDX in examples["category"]:
        print(
            "Warning: There are examples with 'none' category in the training set. "
            "This may lead to incorrect training results."
        )
        # Print affected names
        none_names = [
            name
            for name, category in zip(examples["name"], examples["category"])
            if category == NONE_IDX
        ]
        print(f"Affected names: {none_names}")

    final = []
    for albedo, normal in zip(transformed_albedo, transformed_normal):
        albedo, normal = synced_crop_and_resize(albedo, normal, size=(256, 256))
        albedo = transform_train(albedo)
        # MatSynth dataset uses OpenGL normal maps, we need to convert them to DirectX format
        # Using it here to avoid converting 4k images
        normal = transform_train(convert_normal_to_directx_type(normal))

        # Concatenate albedo and normal along the channel dimension
        final.append(torch.cat((albedo, normal), dim=0))  # type: ignore

    masks = [
        make_full_image_mask(category_id=category, img_size=(1024, 1024))
        for category in examples["category"]
    ]

    return {
        "pixel_values": final,
        "labels": torch.stack(masks, dim=0),  # (B, H, W)
        "category": examples["category"],  # keep for reference
    }


def transform_val_fn(examples):
    # Transform
    transformed = [
        transform_val(image.convert("RGB"), T.InterpolationMode.LANCZOS)
        for image in examples["basecolor"]
    ]

    transformed_normal = [
        # MatSynth dataset uses OpenGL normal maps, we need to convert them to DirectX format
        transform_val(
            convert_normal_to_directx_type(image.convert("RGB")),
            T.InterpolationMode.BILINEAR,
        )
        for image in examples["normal"]
    ]

    final = []
    for albedo, normal in zip(transformed, transformed_normal):
        # Concatenate albedo and normal along the channel dimension
        final.append(torch.cat((albedo, normal), dim=0))  # type: ignore

    masks = [
        # MatSynth dataset uses OpenGL normal maps, we need to convert them to DirectX format
        make_full_image_mask(category_id=category, img_size=(1024, 1024))
        for category in examples["category"]
    ]

    return {
        "pixel_values": final,
        "labels": torch.stack(masks, dim=0),  # (B, H, W)
        "category": examples["category"],  # keep for reference
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
                len(CLASS_LIST),
            ),
        },
    )

    # For SegFormer we need only the basecolor(albedo) and category
    dataset = dataset.select_columns(["name", "basecolor", "normal"])

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
            "per_class_loss": {name: 0.0 for name in CLASS_LIST},
            "IoU": {name: 0.0 for name in CLASS_LIST},
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
            len(CLASS_LIST), dtype=torch.float32, device="cuda"
        )
        val_total_pixels_per_class = torch.zeros(
            len(CLASS_LIST), dtype=torch.float32, device="cuda"
        )

        with torch.no_grad():
            for batch in tqdm(
                validation_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation"
            ):
                input = batch["pixel_values"]
                labels_gt = batch["labels"]
                input = input.to(device, non_blocking=True)
                labels_gt = labels_gt.to(device, non_blocking=True)

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

                for c in range(len(CLASS_LIST)):
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
            num_classes=len(CLASS_LIST),
            ignore_index=255,  # Ignore the background class
            task="multiclass",
            average="none",  # Calculate IoU for each class separately
        )
        for idx, name in enumerate(CLASS_LIST):
            epoch_data["IoU"][name] = jaccard_index[idx].item()

        epoch_data["mIoU"] = jaccard_index.mean().item()

        val_avg_loss_per_class = val_total_loss_per_class / (
            val_total_pixels_per_class + 1e-6
        )
        for idx, name in enumerate(CLASS_LIST):
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
