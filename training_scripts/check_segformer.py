# Ensure we import it here to set random(seed)
import seed
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import argparse

# from torchvision.utils import save_image
from torchvision.transforms import functional as TF
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
from augmentations import center_crop

BASE_DIR = Path(__file__).resolve().parent

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable for faster training on fixed input sizes

parser = argparse.ArgumentParser(
    description="Check segformer model performance by testing on validation set, predicating masks and saving them."
)

parser.add_argument(
    "--load_checkpoint",
    type=str,
    default=None,
    required=True,
    help="Path to the checkpoint to load the model from",
)

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

checkpoint_path = Path(parser.parse_args().load_checkpoint).resolve()
print(f"Loading checkpoint from {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(
    checkpoint["model_state_dict"],
)


def transform_val_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    category = example["category"]
    name = example["name"]

    albedo = center_crop(albedo, (256, 256), [1024, 1024], TF.InterpolationMode.LANCZOS)
    orig_albedo = albedo
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

    return {
        "pixel_values": final,
        "albedo": TF.to_tensor(orig_albedo),  # keep for reference
        "category": category,  # keep for reference
        "name": name,  # keep for reference
    }


validation_dataset.set_transform(transform_val_fn)

validation_loader = DataLoader(
    validation_dataset,  # type: ignore
    batch_size=4,
    shuffle=False,  # No need to shuffle validation data
    # num_workers=6,
)


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    """
    Convert a (H, W) torch mask to a (H, W, 3) PIL.Image with PALETTE colors.
    """
    # 1) ensure CPU & numpy
    if mask.ndim == 3:  # batch dim
        mask = mask[0]  # take first in batch, or loop over them
    mask_np = mask.cpu().numpy().astype(np.uint8)  # shape (H, W)

    # 2) build an RGB array
    h, w = mask_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in validation_dataset.CLASS_PALETTE.items():
        color_img[mask_np == cls_idx] = color

    # 3) convert to PIL
    return Image.fromarray(color_img)


test_images = Path(BASE_DIR / "../segformer_sample_test").resolve()
test_images.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    for batch in tqdm(
        validation_loader,
        desc="Processing validation dataset",
        total=len(validation_loader),
    ):
        input = batch["pixel_values"]
        input = input.to(device, non_blocking=True)
        albedo = batch["albedo"]
        name = batch["name"]
        category = batch["category"]

        logits = model(input).logits

        logits_up = torch.nn.functional.interpolate(
            logits,
            size=input.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        pred_mask = torch.argmax(logits_up, dim=1)

        for albedo, mask, name, category in zip(albedo, pred_mask, name, category):

            cat_name = validation_dataset.CLASS_LIST[category]

            albedo_img = TF.to_pil_image(albedo)

            pred_mask_img = mask_to_pil(mask)

            sample_img = Image.new("RGB", (2048, 1024))
            sample_img.paste(albedo_img, (0, 0))
            sample_img.paste(pred_mask_img, (1024, 0))

            sample_img.save(test_images / f"{cat_name}_{name}_sample.png")
