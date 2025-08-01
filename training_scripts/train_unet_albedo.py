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
from augmentations import (
    get_random_crop,
    center_crop,
)
from skyrim_photometric_aug import SkyrimPhotometric
from segformer_6ch import create_segformer
from class_materials import CLASS_LIST

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

# T_MAX = 10  # Max number of epochs for the learning rate scheduler
PHASE = args.phase  # Phase of the training per plan, used for logging and saving

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True  # Enable for faster training on fixed input sizes
torch.backends.cudnn.benchmark = (
    False  # For some reason it may slows down consequent epochs
)

VISUAL_SAMPLES_COUNT = 10  # Number of samples to visualize in validation

skyrim_dir = (BASE_DIR / "../skyrim_processed/pbr").resolve()

skyrim_data_file_path = (BASE_DIR / "../skyrim_data_unet_albedo.json").resolve()

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
skyrim_train_sampler = WeightedRandomSampler(
    weights=skyrim_sample_weights,
    num_samples=len(skyrim_sample_weights),
    replacement=True,
)

CROP_SIZE = 256

BATCH_SIZE_VALIDATION_SKYRIM = 1

SKYRIM_WORKERS = 0
BATCH_SIZE_SKYRIM = 0

SKYRIM_PHOTOMETRIC = 0.6

USE_ACCUMULATION = False
ACCUM_STEPS = 2

if CROP_SIZE == 256:
    BATCH_SIZE_SKYRIM = 16
    SKYRIM_WORKERS = 16
    # SKYRIM_LEATHER_CROP_BIAS_CHANCE = 0.2
    # SKYRIM_CERAMIC_CROP_BIAS_CHANCE = 0.3

if CROP_SIZE == 512:
    BATCH_SIZE_SKYRIM = 4
    SKYRIM_WORKERS = 4
    # SKYRIM_LEATHER_CROP_BIAS_CHANCE = 0.2
    # SKYRIM_CERAMIC_CROP_BIAS_CHANCE = 0.2
    USE_ACCUMULATION = True

if CROP_SIZE == 768:
    BATCH_SIZE_SKYRIM = 2
    SKYRIM_WORKERS = 4
    # SKYRIM_LEATHER_CROP_BIAS_CHANCE = 0.1
    # SKYRIM_CERAMIC_CROP_BIAS_CHANCE = 0.1
    USE_ACCUMULATION = True

if CROP_SIZE == 1024:
    BATCH_SIZE_SKYRIM = 1
    SKYRIM_WORKERS = 2
    USE_ACCUMULATION = True
    ACCUM_STEPS = 4

BATCH_SIZE = BATCH_SIZE_SKYRIM

MIN_SAMPLES_TRAIN = len(skyrim_train_dataset)
STEPS_PER_EPOCH_TRAIN = math.ceil(MIN_SAMPLES_TRAIN / BATCH_SIZE)

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
        num_labels=len(CLASS_LIST),
        device=device,
        lora=False,
        frozen=True,
    )
    segformer_best_weights_path = (
        BASE_DIR / "../weights/s3/segformer/best_model.pt"
    ).resolve()
    print("Loading Segformer weights from:", segformer_best_weights_path)
    segformer_checkpoint = torch.load(segformer_best_weights_path, map_location=device)

    segformer.load_state_dict(
        segformer_checkpoint["model_state_dict"],
    )
    return unet_alb, segformer, checkpoint


def skyrim_transform_train_fn(example):
    name = example["name"]
    albedo = example["basecolor"]
    normal = example["normal"]
    diffuse = example["diffuse"]

    crop_result = get_random_crop(
        albedo=albedo,
        normal=normal,
        size=(CROP_SIZE, CROP_SIZE),
        diffuse=diffuse,
        augmentations=True,
        resize_to=None,
    )
    albedo = crop_result["albedo"]
    normal = crop_result["normal"]
    diffuse = crop_result["diffuse"]

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

    # Validate at production resolution: 1K for A4, 2K for future A5
    val_crop_size = (1024, 1024)  # Production resolution for A4

    albedo = center_crop(
        albedo,
        size=val_crop_size,
        resize_to=None,
        interpolation=TF.InterpolationMode.LANCZOS,
    )
    diffuse = center_crop(
        diffuse,
        size=val_crop_size,
        resize_to=None,
        interpolation=TF.InterpolationMode.LANCZOS,
    )

    normal = center_crop(
        normal,
        size=val_crop_size,
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
    # ssim_val = FM.structural_similarity_index_measure(
    #     albedo_pred.clamp(0, 1).float(), albedo_gt.clamp(0, 1).float(), data_range=1.0
    # )
    # if isinstance(ssim_val, tuple):
    #     ssim_val = ssim_val[0]
    # ssim_val = torch.nan_to_num(ssim_val, nan=1.0).float()
    # ssim_loss = 1 - ssim_val

    ssim = FM.multiscale_structural_similarity_index_measure(
        albedo_pred.clamp(0, 1).float(), albedo_gt.clamp(0, 1).float(), data_range=1.0
    )
    ssim = torch.nan_to_num(ssim, nan=1.0).float()
    ssim_loss = 1 - ssim
    ecpoch_data["unet_albedo"][key]["ssim_loss"] += ssim_loss.item()

    # LPIPS
    lpips = lpips_batch(albedo_pred.clamp(0, 1).float(), albedo_gt.clamp(0, 1).float())
    lpips = torch.nan_to_num(lpips, nan=0.0).float()
    ecpoch_data["unet_albedo"][key]["lpips"] += lpips.item()

    # Total loss
    total_loss = l1_loss + 0.15 * ssim_loss + 0.05 * lpips

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


skyrim_train_dataset.set_transform(skyrim_transform_train_fn)
skyrim_validation_dataset.set_transform(transform_val_fn)


# Training loop
def do_train():
    EPOCHS = 45

    print(
        f"Starting training for {EPOCHS} epochs, on {(STEPS_PER_EPOCH_TRAIN * BATCH_SIZE)} samples, validation on {len(skyrim_validation_dataset)} samples."
    )

    unet_alb, segformer, checkpoint = get_model()

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
        prefetch_factor=2,
        persistent_workers=True,
    )

    skyrim_train_iter = cycle(skyrim_train_loader)

    # LR = 1e-6
    # WD = 1e-2
    WD = 1e-2

    # A3.1: Full UNet adaptation (except bottleneck & FiLM)
    # Train entire feature processing pipeline for 6-channel input

    # for param in unet_alb.parameters():
    #     param.requires_grad = False

    # for param in unet_alb.unet.decoder.parameters():
    #     param.requires_grad = True

    # for param in unet_alb.out.parameters():
    #     param.requires_grad = True

    # for n, p in unet_alb.named_parameters():
    #     if p.requires_grad:
    #         print(f"Trainable parameter: {n}")

    param_groups = []

    enc_params = [
        p
        for n, p in unet_alb.named_parameters()
        if (
            "unet.inc." in n
            or "unet.encoder." in n
            or "unet.bot." in n
            or "unet.bottleneck_attention." in n
        )
        and p.requires_grad
    ]
    param_groups.append({"params": enc_params, "lr": 2e-4, "weight_decay": WD})

    decoder_params = [
        p
        for n, p in unet_alb.named_parameters()
        if "unet.decoder." in n and p.requires_grad
    ]
    param_groups.append({"params": decoder_params, "lr": 2e-4, "weight_decay": WD})

    film_params = [
        p
        for n, p in unet_alb.named_parameters()
        if "unet.film." in n and p.requires_grad
    ]
    param_groups.append({"params": film_params, "lr": 3e-4, "weight_decay": 0.0})

    head_params = [
        p for n, p in unet_alb.named_parameters() if "out." in n and p.requires_grad
    ]
    param_groups.append({"params": head_params, "lr": 2.5e-4, "weight_decay": WD})

    # trainable = [p for p in unet_alb.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    if checkpoint is not None and resume_training:
        print("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    effective_scheduler_steps = (
        int(STEPS_PER_EPOCH_TRAIN / ACCUM_STEPS)
        if USE_ACCUMULATION
        else STEPS_PER_EPOCH_TRAIN
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=EPOCHS, eta_min=LR * 0.1
    # )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[2e-4, 2e-4, 3e-4, 2.5e-4],
        total_steps=EPOCHS * STEPS_PER_EPOCH_TRAIN,
        pct_start=0.2,
        anneal_strategy="cos",
        final_div_factor=20,  # final LR ≈ max/20 ≈ 1e-5
    )
    # 1 epoch warm-up to the base LR
    # warmup = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=0.1,
    #     end_factor=1.0,
    #     total_iters=effective_scheduler_steps,
    # )

    # cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=(EPOCHS - 1) * effective_scheduler_steps, eta_min=2e-6
    # )
    # Gentle cosine decay for inc layer adaptation
    # Start at 2e-4, decay to 5e-5 over 3 epochs for stable convergence
    # cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=(EPOCHS - 1) * effective_scheduler_steps,
    #     eta_min=5e-5,  # End at 25% of starting LR
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=3 * effective_scheduler_steps,  # 4 epochs per restart
    #     T_mult=1,
    #     eta_min=2e-6,
    # )

    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[warmup, cosine],
    #     milestones=[
    #         effective_scheduler_steps,
    #     ],  # After first epoch switch to cosine
    # )

    if checkpoint is not None and resume_training and scheduler is not None:
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

    patience = 8  # Increased patience for higher LRs
    no_improvement_count_albedo = 0

    output_dir = Path(f"./weights/{PHASE}/unet_albedo")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if checkpoint is not None and resume_training:
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, EPOCHS):
        unet_alb.train()

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

        if USE_ACCUMULATION:
            optimizer.zero_grad(set_to_none=True)

        for i in bar:
            skyrim_batch = next(skyrim_train_iter)

            diffuse_and_normal = skyrim_batch["diffuse_and_normal"]
            albedo_gt = skyrim_batch["albedo"]
            albedo_and_normal_segformer = skyrim_batch["albedo_and_normal_segformer"]

            diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
            albedo_gt = albedo_gt.to(device, non_blocking=True)
            albedo_and_normal_segformer = albedo_and_normal_segformer.to(
                device, non_blocking=True
            )

            if not USE_ACCUMULATION:
                optimizer.zero_grad()

            with torch.no_grad():
                with autocast(device_type=device.type):
                    #  Get Segoformer ouput for FiLM
                    seg_feats = segformer(
                        albedo_and_normal_segformer, output_hidden_states=True
                    )["hidden_states"][-1].detach()

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

            if USE_ACCUMULATION:
                total_loss = total_loss / ACCUM_STEPS  # Scale loss for accumulation

            scaler.scale(total_loss).backward()

            if not USE_ACCUMULATION:
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            else:  # USE_ACCUMULATION is True
                if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == STEPS_PER_EPOCH_TRAIN:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    # Step per effective batch
                    if scheduler is not None:
                        scheduler.step()

        calculate_avg(epoch_data, key="train")

        unet_alb.eval()
        with torch.no_grad():

            num_samples_saved = 0

            for _, batch in enumerate(
                tqdm(
                    skyrim_validation_loader,
                    desc=f"Epoch {epoch + 1}/{EPOCHS} - Skyrim Validation",
                    unit="batch",
                )
            ):
                diffuse_and_normal = batch["diffuse_and_normal"]
                albedo_and_normal_segformer = batch["albedo_and_normal_segformer"]
                albedo_gt = batch["albedo"]
                normal = batch["normal"]
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
                    seg_feats = segformer(
                        albedo_and_normal_segformer, output_hidden_states=True
                    )["hidden_states"][-1].detach()
                    albedo_pred = unet_alb(diffuse_and_normal, seg_feats)

                calculate_unet_albedo_loss(
                    albedo_pred, albedo_gt, epoch_data, key="validation"
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
                        sample_albedo_pred,
                        sample_name,
                    ) in zip(
                        original_diffuse, original_normal, albedo_gt, albedo_pred, names
                    ):

                        # Save few samples per class for inspection
                        output_path = output_dir / f"val_samples_{epoch + 1}"
                        output_path.mkdir(parents=True, exist_ok=True)

                        # Save diffuse, normal, GT albedo and predicted albedo side by side
                        visual_sample = torch.cat(
                            [
                                sample_diffuse,  # Diffuse
                                sample_normal,  # Normal
                                sample_albedo_gt,  # GT Albedo
                                sample_albedo_pred,  # Predicted Albedo
                            ],
                            dim=2,  # Concatenate along width
                        ).clamp(0, 1)

                        save_image(visual_sample, output_path / f"{sample_name}.png")

                        num_samples_saved += 1

        unet_albedo_total_val_loss = calculate_avg(epoch_data, key="validation")

        # scheduler.step()
        print(json.dumps(epoch_data, indent=4))

        # Save checkopoint after each epoch
        checkpoint_data = {
            "epoch": epoch + 1,
            "unet_albedo_model_state_dict": unet_alb.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch_data": epoch_data,
        }
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(
            checkpoint_data,
            output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
        )
        # Save epoch data to a JSON file
        with open(output_dir / f"epoch_{epoch + 1}_stats.json", "w") as f:
            json.dump(epoch_data, f, indent=4)

        if unet_albedo_total_val_loss < best_val_loss_albedo:
            best_val_loss_albedo = unet_albedo_total_val_loss
            no_improvement_count_albedo = 0

            best_model_data = {
                "epoch": epoch + 1,
                "unet_albedo_model_state_dict": unet_alb.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch_data": epoch_data,
            }
            if scheduler is not None:
                best_model_data["scheduler_state_dict"] = scheduler.state_dict()

            torch.save(
                best_model_data,
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
