# Ensure we import it here to set random(seed)
import seed
import json, torch
import multiprocessing
import torch.nn.functional as F
import lpips
import argparse
import math
from skyrim_dataset import SkyrimDataset
from unet_models import UNetAlbedo
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
from train_dataset import SimpleImageDataset, normalize_normal_map
from augmentations import (
    get_random_crop,
    selective_aug,
    center_crop,
)
from skyrim_photometric_aug import SkyrimPhotometric
from segformer_6ch import create_segformer

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast


# Supress VGG warnings from lpips
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models")


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
EPOCHS = 35  # Number of epochs to train
LR = 2e-4  # Learning rate for the optimizer
WD = 1e-2  # Weight decay for the optimizer
# T_MAX = 10  # Max number of epochs for the learning rate scheduler
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
    matsynth_dir=str(matsynth_dir), split="validation", skip_init=True
)

matsynth_validation_dataset.all_validation_samples = (
    matsynth_train_dataset.all_validation_samples
)
loss_weights, sample_weights = matsynth_train_dataset.get_weights()

loss_weights = loss_weights.to(device)  # type: ignore

train_sampler = WeightedRandomSampler(
    weights=sample_weights.tolist(),
    num_samples=len(sample_weights),
    replacement=True,
)

skyrim_train_dataset = SkyrimDataset(
    skyrim_dir=str(skyrim_dir),
    split="train",
    load_non_pbr=False,
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

CROP_SIZE = 256

BATCH_SIZE_MATSYNTH = 16
BATCH_SIZE_SKYRIM = 16
BATCH_SIZE_VALIDATION_MATSYNTH = 16
BATCH_SIZE_VALIDATION_SKYRIM = 16

MATSYNTH_COLOR_AUGMENTATIONS = False
SKYRIM_PHOTOMETRIC = 0.0

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
MIN_SAMPLES_VALIDATION = len(skyrim_validation_dataset)
STEPS_PER_EPOCH_TRAIN = math.ceil(MIN_SAMPLES_TRAIN / BATCH_SIZE)
STEPS_PER_EPOCH_VALIDATION = math.ceil(
    MIN_SAMPLES_VALIDATION / BATCH_SIZE_VALIDATION_SKYRIM
)

resume_training = args.resume


def get_model():
    unet_alb = UNetAlbedo(
        in_ch=6,  # RGB + Normal
        cond_ch=512,  # Condition channel size, can be adjusted
    ).to(
        device
    )  # type: ignore

    checkpoint = None
    if (args.load_checkpoint is not None) and Path(
        args.load_checkpoint
    ).resolve().exists():
        load_checkpoint_path = Path(args.load_checkpoint).resolve()
        print(
            f"Loading model from checkpoint: {load_checkpoint_path}, resume={resume_training}"
        )
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        unet_alb.load_state_dict(checkpoint["unet_albedo_model_state_dict"])

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
    return unet_alb, segformer, checkpoint


def matsynth_transform_train_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    diffuse = example["diffuse"]
    category_name = example["category_name"]
    name = example["name"]

    albedo, normal, diffuse, *_ = get_random_crop(
        albedo=albedo,
        normal=normal,
        size=(CROP_SIZE, CROP_SIZE),
        diffuse=diffuse,
        resize_to=None,
        augmentations=True,
    )

    albedo_orig = albedo
    albedo_segformer = albedo

    albedo_orig = TF.to_tensor(albedo_orig)

    if MATSYNTH_COLOR_AUGMENTATIONS:
        diffuse, normal = selective_aug(diffuse, normal, category=category_name)

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
        "name": name,
    }


def skyrim_transform_train_fn(example):
    name = example["name"]
    albedo = example["basecolor"]
    normal = example["normal"]
    diffuse = example["diffuse"]

    albedo, normal, diffuse, *_ = get_random_crop(
        albedo,
        normal,
        size=(CROP_SIZE, CROP_SIZE),
        diffuse=diffuse,
        augmentations=True,
        resize_to=None,
    )

    albedo_orig = albedo
    albedo_segformer = albedo

    albedo_orig = TF.to_tensor(albedo_orig)

    if SKYRIM_PHOTOMETRIC > 0.0:
        skyrim_photometric = SkyrimPhotometric(p_aug=SKYRIM_PHOTOMETRIC)
        diffuse = skyrim_photometric(diffuse)

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
        "name": name,
    }


def transform_val_fn(example):
    albedo = example["basecolor"]
    normal = example["normal"]
    diffuse = example["diffuse"]
    name = example["name"]

    albedo = center_crop(
        albedo,
        size=(CROP_SIZE, CROP_SIZE),
        resize_to=None,
        interpolation=TF.InterpolationMode.LANCZOS,
    )
    diffuse = center_crop(
        diffuse,
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

    original_albedo = albedo
    albedo_segformer = albedo

    # Store original non normalized diffuse and normal for visual inspection in validation loop
    original_normal = TF.to_tensor(normal)
    original_diffuse = TF.to_tensor(diffuse)
    original_albedo = TF.to_tensor(original_albedo)

    diffuse = TF.to_tensor(diffuse)
    diffuse = TF.normalize(
        diffuse, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )

    normal = TF.to_tensor(normal)
    normal = TF.normalize(
        normal, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD
    )
    diffuse_and_normal = torch.cat((diffuse, normal), dim=0)

    albedo_segformer = TF.to_tensor(albedo_segformer)
    albedo_segformer = TF.normalize(
        albedo_segformer, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )

    albedo_and_normal_segformer = torch.cat((albedo_segformer, normal), dim=0)

    return {
        "diffuse_and_normal": diffuse_and_normal,
        "albedo_and_normal_segformer": albedo_and_normal_segformer,
        "albedo": original_albedo,
        "normal": normal,
        "name": name,
        "original_diffuse": original_diffuse,
        "original_normal": original_normal,
    }


_lpips = lpips.LPIPS(net="vgg").to(device).eval()
for p in _lpips.parameters():
    p.requires_grad = False


def to_lpips_space(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W) in [0,1]
    returns: (B,3,H,W) in [-1,1]
    """
    return x * 2 - 1


def lpips_batch(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    # remap into LPIPS’s expected [-1,1]
    p = to_lpips_space(pred_rgb)
    t = to_lpips_space(target_rgb)

    return _lpips(p, t).mean()


def calculate_unet_albedo_loss(
    albedo_pred: torch.Tensor,
    albedo_gt: torch.Tensor,
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
    lpips = lpips_batch(albedo_pred.clamp(0, 1).float(), albedo_gt.clamp(0, 1).float())
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


def cycle(dl: DataLoader):
    while True:
        for batch in dl:
            yield batch


matsynth_train_dataset.set_transform(matsynth_transform_train_fn)
matsynth_validation_dataset.set_transform(transform_val_fn)

skyrim_train_dataset.set_transform(skyrim_transform_train_fn)
skyrim_validation_dataset.set_transform(transform_val_fn)


# Training loop
def do_train():
    print(
        f"Starting training for {EPOCHS} epochs, on {(STEPS_PER_EPOCH_TRAIN * BATCH_SIZE) * 2} Samples, MatSynth/Skyrim Batch: {BATCH_SIZE_MATSYNTH}/{BATCH_SIZE_SKYRIM}, validation on {MIN_SAMPLES_VALIDATION * 2} samples."
    )

    unet_alb, segformer, checkpoint = get_model()

    matsynth_train_loader = DataLoader(
        matsynth_train_dataset,  # type: ignore
        batch_size=BATCH_SIZE_MATSYNTH,
        sampler=train_sampler,
        num_workers=6,
        prefetch_factor=2,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    matsynth_validation_loader = DataLoader(
        matsynth_validation_dataset,  # type: ignore
        batch_size=BATCH_SIZE_VALIDATION_MATSYNTH,
        num_workers=3,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    skyrim_train_loader = DataLoader(
        skyrim_train_dataset,
        batch_size=BATCH_SIZE_SKYRIM,
        num_workers=6,
        prefetch_factor=2,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    skyrim_validation_loader = DataLoader(
        skyrim_validation_dataset,
        batch_size=BATCH_SIZE_VALIDATION_SKYRIM,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
    )

    matsynth_train_iter = cycle(matsynth_train_loader)
    skyrim_train_iter = cycle(skyrim_train_loader)

    matsynth_validation_iter = cycle(matsynth_validation_loader)
    skyrim_validation_iter = cycle(skyrim_validation_loader)

    enc_params = {"params": unet_alb.unet.encoder.parameters(), "lr": 2e-4}
    dec_params = {"params": unet_alb.unet.decoder.parameters(), "lr": 2e-4}
    film_params = {"params": unet_alb.unet.film.parameters(), "lr": 3e-4}  # type: ignore
    head_params = {"params": unet_alb.out.parameters(), "lr": 2.5e-4}

    optimizer = torch.optim.AdamW(
        # filter(lambda p: p.requires_grad, unet_alb.parameters()),
        # lr=LR,
        # weight_decay=WD,
        [enc_params, dec_params, film_params, head_params],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WD,
    )
    if checkpoint is not None and resume_training:
        print("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=LR,
    #     total_steps=EPOCHS * len(train_loader),
    #     # 15% warm-up 85% cooldown
    #     pct_start=0.15,
    #     div_factor=5.0,  # start LR = 1e-5
    #     final_div_factor=5.0,  # End LR = 1e-5
    # )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[2e-4, 2e-4, 3e-4, 2.5e-4],
        total_steps=EPOCHS * STEPS_PER_EPOCH_TRAIN,
        pct_start=0.2,
        anneal_strategy="cos",
        final_div_factor=20,  # final LR ≈ max/20 ≈ 1e-5
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

        # # Batch size = 4 fits into VRAM of 5090 but it's slow. Much faster to use gradient accumulation with batch size = 2
        # accum_steps = 2

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
        # optimizer.zero_grad()

        for i in bar:
            matsynth_batch = next(matsynth_train_iter)
            skyrim_batch = next(skyrim_train_iter)

            diffuse_and_normal = torch.cat(
                [
                    matsynth_batch["diffuse_and_normal"],
                    skyrim_batch["diffuse_and_normal"],
                ],
                dim=0,
            )
            albedo_gt = torch.cat(
                [matsynth_batch["albedo"], skyrim_batch["albedo"]], dim=0
            )
            albedo_and_normal_segformer = torch.cat(
                [
                    matsynth_batch["albedo_and_normal_segformer"],
                    skyrim_batch["albedo_and_normal_segformer"],
                ],
                dim=0,
            )

            diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
            albedo_gt = albedo_gt.to(device, non_blocking=True)
            albedo_and_normal_segformer = albedo_and_normal_segformer.to(
                device, non_blocking=True
            )

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

            with autocast(device_type=device.type):
                # Get UNet-Albedo prediction
                albedo_pred = unet_alb(diffuse_and_normal, seg_feats)

            # Unet-albedo loss
            unet_albedo_loss = calculate_unet_albedo_loss(
                albedo_pred, albedo_gt, epoch_data, key="train"
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
            # total_loss = total_loss / accum_steps
            scaler.scale(total_loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     scheduler.step()

        calculate_avg(epoch_data, key="train")

        unet_alb.eval()
        with torch.no_grad():
            bar = tqdm(
                range(STEPS_PER_EPOCH_VALIDATION),
                desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation",
                unit="batch",
            )

            num_samples_saved = 0

            for _ in bar:
                matsynth_batch = next(matsynth_validation_iter)
                skyrim_batch = next(skyrim_validation_iter)

                diffuse_and_normal = torch.cat(
                    [
                        matsynth_batch["diffuse_and_normal"],
                        skyrim_batch["diffuse_and_normal"],
                    ],
                    dim=0,
                )
                albedo_and_normal_segformer = torch.cat(
                    [
                        matsynth_batch["albedo_and_normal_segformer"],
                        skyrim_batch["albedo_and_normal_segformer"],
                    ],
                    dim=0,
                )
                albedo_gt = torch.cat(
                    [matsynth_batch["albedo"], skyrim_batch["albedo"]], dim=0
                )
                normal = torch.cat(
                    [matsynth_batch["normal"], skyrim_batch["normal"]], dim=0
                )
                names = list(matsynth_batch["name"]) + list(skyrim_batch["name"])
                original_diffuse = torch.cat(
                    [
                        matsynth_batch["original_diffuse"],
                        skyrim_batch["original_diffuse"],
                    ],
                    dim=0,
                )
                original_normal = torch.cat(
                    [
                        matsynth_batch["original_normal"],
                        skyrim_batch["original_normal"],
                    ],
                    dim=0,
                )

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
                    )
                    albedo_pred = unet_alb(diffuse_and_normal, seg_feats)

                calculate_unet_albedo_loss(
                    albedo_pred, albedo_gt, epoch_data, key="validation"
                )

                epoch_data["validation"]["batch_count"] += 1

                if num_samples_saved < 64:
                    for (
                        sample_diffuse,
                        sample_normal,
                        sample_albedo_gt,
                        sample_albedo_pred,
                        sample_name,
                    ) in zip(
                        original_diffuse, original_normal, albedo_gt, albedo_pred, names
                    ):

                        # Save few samples per class for inspection
                        output_path = output_dir / f"val_samples_{epoch + 1}"
                        output_path.mkdir(parents=True, exist_ok=True)

                        # Save diffuse, normal, GT albedo and predicted albedo side by side
                        visual_sample_gt = torch.cat(
                            [
                                sample_diffuse,  # Diffuse
                                sample_normal,  # Normal
                                sample_albedo_gt,  # GT Albedo
                            ],
                            dim=2,  # Concatenate along width
                        )
                        visual_sample_predicted = torch.cat(
                            [
                                sample_diffuse,  # Diffuse
                                sample_normal,  # Normal
                                sample_albedo_pred,  # Predicted Albedo
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
