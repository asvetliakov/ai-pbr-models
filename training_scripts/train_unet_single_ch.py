# Ensure we import it here to set random(seed)
import seed
import json, torch
import random
import multiprocessing
import torch.nn.functional as F
import argparse
import math
from skyrim_dataset import SkyrimDataset
from unet_models import UNetSingleChannel, UNetAlbedo
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms import functional as TF, RandomErasing
from torchmetrics import functional as FM
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
import kornia as K


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

parser.add_argument(
    "--map-to-train",
    type=str,
    choices=["roughness", "metallic", "ao", "parallax"],
    required=True,
    help="Map to train: roughness, metallic, ao, or parallax",
)

args = parser.parse_args()

UNET_MAP = args.map_to_train

print(f"Training phase: {args.phase}, map to train: {UNET_MAP}")

# HYPER_PARAMETERS
EPOCHS = 7  # Number of epochs to train
# LR = 1e-3  # Learning rate for the optimizer
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
    ignore_without_parallax=UNET_MAP == "parallax",
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

DATASET_WORKERS = 4
if CROP_SIZE == 768:
    # Batch size 6 for 768
    DATASET_WORKERS = 8
elif CROP_SIZE < 768:
    # Batch size 16 for 512 and 32 for 256
    DATASET_WORKERS = 12

resume_training = args.resume


def get_model():
    # AO, height
    unet_channels = 5
    if UNET_MAP == "roughness":
        unet_channels = 6
    elif UNET_MAP == "metallic":
        unet_channels = 3

    unet = UNetSingleChannel(in_ch=unet_channels, cond_ch=512).to(device)  # type: ignore

    if args.weight_donor is not None:
        weight_donor_path = Path(args.weight_donor).resolve()
        print(f"Loading model from weight donor: {weight_donor_path}")
        weight_donor_checkpoint = torch.load(weight_donor_path, map_location=device)

        # print("Taking normal channels for convolution")
        # w_old = weight_donor_checkpoint["unet_albedo_model_state_dict"][
        #     "unet.inc.conv.0.weight"
        # ]  # torch.Size([64, 6, 3, 3])

        # # Take normal channels from A2 weights
        # w_norm = w_old[:, 3:, :, :].clone()  # torch.Size([64, 3, 3, 3])

        # if UNET_MAP == "parallax":
        #     albedo_gray = torch.empty(
        #         (w_old.size(0), 1, *w_old.shape[2:]),
        #         device=w_old.device,
        #         dtype=w_old.dtype,
        #     )
        #     torch.nn.init.kaiming_normal_(albedo_gray, mode="fan_out")
        #     w_new = torch.cat([w_norm, albedo_gray], dim=1)  # shape [64,4,3,3]
        # else:
        # w_new = w_norm  # shape [64,3,3,3]

        # Replace
        # weight_donor_checkpoint["unet_albedo_model_state_dict"][
        #     "unet.inc.conv.0.weight"
        # ] = w_new

        unet.load_state_dict(
            weight_donor_checkpoint["unet_albedo_model_state_dict"], strict=False
        )

        # ! Set metal bias # to a value that corresponds to 15% metal pixels in the dataset to prevent early collapse
        # Doing this only on the first init on the first phase. If we're setting weight donor then we're at the first phase
        if UNET_MAP == "metallic":
            p0 = 0.15
            b0 = -math.log((1 - p0) / p0)
            with torch.no_grad():
                torch.nn.init.constant_(unet.out.bias, b0)  # type: ignore

    checkpoint = None
    if (args.load_checkpoint is not None) and Path(
        args.load_checkpoint
    ).resolve().exists():
        load_checkpoint_path = Path(args.load_checkpoint).resolve()
        print(
            f"Loading model from checkpoint: {load_checkpoint_path}, resume={resume_training}"
        )
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        unet.load_state_dict(checkpoint["unet_maps_model_state_dict"])

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

    return unet, segformer, unet_albedo, checkpoint


def transform_train_fn(example):
    name = example["name"]
    albedo = example["basecolor"]
    diffuse = example["diffuse"]
    normal = example["normal"]
    parallax = example["parallax"]
    ao = example["ao"]
    metallic = example["metallic"]
    roughness = example["roughness"]
    poisson_blur = example["poisson_blur"]

    albedo, normal, diffuse, parallax, metallic, roughness, ao, poisson_blur = (
        get_random_crop(
            albedo,
            normal,
            size=(CROP_SIZE, CROP_SIZE),
            diffuse=diffuse,
            height=parallax,
            ao=ao,
            metallic=metallic,
            roughness=roughness,
            poisson_blur=poisson_blur,
            augmentations=True,
            resize_to=None,
        )
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

    if poisson_blur is not None:
        poisson_blur_arr = np.array(poisson_blur, dtype=np.uint16)
        poisson_blur_arr = (
            poisson_blur_arr.astype(np.float32) / 65535.0
        )  # Normalize to [0,1]

        poisson_blur = torch.from_numpy(poisson_blur_arr).unsqueeze(0)
        # Normalize to [-1, 1]
        poisson_blur = (poisson_blur - 0.5) * 2.0
    else:
        poisson_blur = torch.zeros((1, CROP_SIZE, CROP_SIZE), dtype=torch.float32)

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
        "poisson_blur": poisson_blur,
        "name": name,
    }


def transform_val_fn(example):
    name = example["name"]
    albedo = example["basecolor"]
    normal = example["normal"]
    name = example["name"]
    diffuse = example["diffuse"]
    parallax = example["parallax"]
    ao = example["ao"]
    metallic = example["metallic"]
    roughness = example["roughness"]
    poisson_blur = example["poisson_blur"]

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

    poisson_blur = (
        center_crop(
            poisson_blur,
            size=(CROP_SIZE, CROP_SIZE),
            resize_to=None,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        if poisson_blur is not None
        else None
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

    if poisson_blur is not None:
        poisson_blur_arr = np.array(poisson_blur, dtype=np.uint16)
        poisson_blur_arr = (
            poisson_blur_arr.astype(np.float32) / 65535.0
        )  # Normalize to [0,1]

        poisson_blur = torch.from_numpy(poisson_blur_arr).unsqueeze(0)
        # Normalize to [-1, 1]
        poisson_blur = (poisson_blur - 0.5) * 2.0
    else:
        poisson_blur = torch.zeros((1, CROP_SIZE, CROP_SIZE), dtype=torch.float32)

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
        "poisson_blur": poisson_blur,
        "name": name,
    }


def gradient_difference_loss(pred, gt):
    # pred, gt: (B,1,H,W)
    # compute forward-differences
    dh_pred_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dh_pred_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dh_gt_x = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    dh_gt_y = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    return F.l1_loss(dh_pred_x, dh_gt_x).float() + F.l1_loss(dh_pred_y, dh_gt_y).float()


def edge_aware_sobel_loss(pred, gt):
    grads_pred = K.filters.spatial_gradient(pred, mode="sobel", order=1)
    grads_gt = K.filters.spatial_gradient(gt, mode="sobel", order=1)
    # collapse gradient channels to magnitude
    # grads_pred: (B,1,2,H,W) → (B,1,H,W)
    gx_pred, gy_pred = grads_pred.unbind(dim=2)
    sobel_pred = gx_pred.abs() + gy_pred.abs()

    gx_gt, gy_gt = grads_gt.unbind(dim=2)
    sobel_gt = gx_gt.abs() + gy_gt.abs()

    loss_edge = F.l1_loss(sobel_pred, sobel_gt).float()
    return loss_edge


def dice_loss(pred, gt, eps=1e-6):
    # pred, gt in [0,1], shape (B,1,H,W)
    intersection = (pred * gt).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice_score = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice_score.mean()


def tv2_loss(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,1,H,W) height map
    returns: scalar mean of second-order TV
    """
    # second differences along width (x-axis)
    d2x = x[..., :, :-2] - 2 * x[..., :, 1:-1] + x[..., :, 2:]
    # second differences along height (y-axis)
    d2y = x[..., :, :-2, :] - 2 * x[..., :, 1:-1, :] + x[..., :, 2:, :]

    # take absolute value and mean
    loss_x = d2x.abs().float().mean()
    loss_y = d2y.abs().float().mean()
    return loss_x + loss_y


def sobel_to_normal(pred: torch.Tensor) -> torch.Tensor:
    # 1) cast to float32 for stable Sobel
    H32 = pred.to(torch.float32)

    # 2) get Sobel ∂H/∂x, ∂H/∂y in one call
    #    returns (B,C,2,H,W): channel 0 = dx, 1 = dy
    grads = K.filters.spatial_gradient(H32, mode="sobel", order=1)
    dx, dy = grads[..., 0, :, :], grads[..., 1, :, :]

    # 3) build the predicted normals
    nz = torch.ones_like(dx, dtype=torch.float32)
    # DirectX predirected normal
    N_pred = torch.cat([-dx, +dy, nz], dim=1)
    N_pred = F.normalize(N_pred, dim=1)

    return N_pred


def normal_consistency_loss(
    pred: torch.Tensor, normal_map: torch.Tensor
) -> torch.Tensor:
    """
    pred:       (B,1,H,W) predicted height ∈ [0,1]
    normal_map: (B,3,H,W) input normals ∈ [-1,1]
    """
    N_pred = sobel_to_normal(pred)

    # ensure input normals are float32 & unit
    N_in = F.normalize(normal_map.to(torch.float32), dim=1)

    # cosine‐distance
    cos = (N_pred * N_in).sum(dim=1)  # (B,H,W)
    loss = (1.0 - cos).mean()

    return loss


def normal_reprojection_loss(
    pred: torch.Tensor, normal_map: torch.Tensor
) -> torch.Tensor:
    """
    pred:       (B,1,H,W) predicted height ∈ [0,1]
    normal_map: (B,3,H,W) input normals ∈ [-1,1]
    """
    N_pred = sobel_to_normal(pred)

    # ensure input normals are float32 & unit
    N_in = F.normalize(normal_map.to(torch.float32), dim=1)

    # reprojection loss
    reproj_loss = F.l1_loss(N_pred, N_in)

    return reproj_loss


def laplacian_pyramid(img, levels=3):
    pyr, cur = [], img
    for _ in range(levels):
        lo = F.avg_pool2d(cur, 2)
        hi = cur - F.interpolate(
            lo, scale_factor=2, mode="bilinear", align_corners=False
        )
        pyr.append(hi)
        cur = lo
    pyr.append(cur)
    return pyr  # list of tensors


def focal_bce_with_logits(pred, targets, gamma=2.0, pos_weight=None, reduction="mean"):
    """
    Focal BCE for a single-channel (metal) mask.

    logits  : (B,1,H,W) raw network outputs
    targets : (B,1,H,W) ground-truth ∈ {0,1} or [0,1]
    """
    # standard BCE, no reduction yet
    bce = F.binary_cross_entropy_with_logits(
        pred, targets, pos_weight=pos_weight, reduction="none"
    )
    # convert logits → probability of the *true* class
    pred = pred * targets + (1.0 - pred) * (1.0 - targets)

    focal_factor = (1.0 - pred).pow(gamma)  # (B,1,H,W)
    loss = focal_factor * bce  # shape like bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:  # "none"
        return loss


def focal_l1_loss(pred, gt, alpha=2.0, gamma=1.5, eps=1e-6):
    """
    pred, gt: (B,C,H,W) tensors
    alpha: weight factor for error-based scaling
    gamma: focal exponent
    """
    err = torch.abs(pred - gt) + eps  # (B,C,H,W)
    weight = 1 + alpha * err  # linear focus
    loss = (weight * err.pow(gamma)).mean()  # combine focal & relative
    return loss


def calculate_rougness_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    epoch_data: dict,
    key="train",
) -> torch.Tensor:
    if epoch_data[key].get("total_loss") is None:
        epoch_data[key]["l1_loss"] = 0.0
        epoch_data[key]["ssim_loss"] = 0.0
        epoch_data[key]["edge_loss"] = 0.0
        epoch_data[key]["total_loss"] = 0.0

    prob = torch.sigmoid(pred)
    ssim_rough = FM.multiscale_structural_similarity_index_measure(
        prob, gt.clamp(0, 1).float(), data_range=1.0
    )
    ssim_rough = torch.nan_to_num(ssim_rough, nan=1.0).float()

    ssim_rough_loss = (1 - ssim_rough).float()
    epoch_data[key]["ssim_loss"] += ssim_rough_loss.item()

    # l1_rough = F.l1_loss(prob, gt).float()
    l1_rough = focal_l1_loss(prob, gt, alpha=1.5, gamma=1.5, eps=1e-6).float()
    epoch_data[key]["l1_loss"] += l1_rough.item()

    loss_edge = edge_aware_sobel_loss(prob, gt)
    epoch_data[key]["edge_loss"] += loss_edge.item()

    loss_rough = l1_rough + 0.1 * ssim_rough_loss + 0.02 * loss_edge
    epoch_data[key]["total_loss"] += loss_rough.item()

    return loss_rough


def calculate_metallic_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    epoch_data: dict,
    key="train",
) -> torch.Tensor:
    if epoch_data[key].get("total_loss") is None:
        epoch_data[key]["bce"] = 0
        epoch_data[key]["l1_loss"] = 0.0
        epoch_data[key]["edge_loss"] = 0.0
        epoch_data[key]["dice_loss"] = 0.0
        epoch_data[key]["total_loss"] = 0.0

    metal_positive = gt.float().sum()
    metal_negative = (gt.numel() - metal_positive).float()
    if metal_positive == 0:
        metal_weights = torch.tensor(1.0, device=device)
        skip_bce = True
    else:
        metal_weights = (
            ((metal_negative + 1e-6) / (metal_positive + 1e-6))
            .clamp(min=1.0, max=16.0)
            .to(device)
        )
        skip_bce = False

    if not skip_bce:
        bce = focal_bce_with_logits(
            pred, gt, pos_weight=metal_weights, gamma=2.0, reduction="mean"
        )
    else:
        bce = pred.new_zeros(())

    epoch_data[key]["bce"] += bce.item()

    prob = torch.sigmoid(pred)

    l1_metal_loss = F.l1_loss(prob, gt, reduction="mean").float()
    epoch_data[key]["l1_loss"] += l1_metal_loss.item()

    edge_metal_loss = edge_aware_sobel_loss(prob, gt).float()
    epoch_data[key]["edge_loss"] += edge_metal_loss.item()

    dice_metal_loss = dice_loss(prob, gt).float()
    epoch_data[key]["dice_loss"] += dice_metal_loss.item()

    loss_metal = (
        bce + 0.2 * l1_metal_loss + 0.05 * edge_metal_loss + 0.5 * dice_metal_loss
    )
    epoch_data[key]["total_loss"] += loss_metal.item()

    return loss_metal


def calculate_ao_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    epoch_data: dict,
    key="train",
) -> torch.Tensor:

    if epoch_data[key].get("total_loss") is None:
        epoch_data[key]["l1_loss"] = 0.0
        epoch_data[key]["edge_loss"] = 0.0
        epoch_data[key]["ssim_loss"] = 0.0
        epoch_data[key]["total_loss"] = 0.0

    prob = torch.sigmoid(pred)

    l1_ao_loss = F.l1_loss(prob, gt).float()
    epoch_data[key]["l1_loss"] += l1_ao_loss.item()

    # Sobel gradient for AO
    loss_edge = edge_aware_sobel_loss(prob, gt)
    epoch_data[key]["edge_loss"] += loss_edge.item()

    ssim = FM.multiscale_structural_similarity_index_measure(
        prob, gt.clamp(0, 1).float(), data_range=1.0
    )
    ssim = torch.nan_to_num(ssim, nan=1.0).float()
    ssim_loss = (1 - ssim).float()
    epoch_data[key]["ssim_loss"] += ssim_loss.item()

    loss_ao = l1_ao_loss + 0.15 * loss_edge + 0.1 * ssim_loss
    epoch_data[key]["total_loss"] += loss_ao.item()

    return loss_ao


def calculate_height_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    normal_map: torch.Tensor,
    epoch_data: dict,
    key="train",
) -> torch.Tensor:
    if epoch_data[key].get("total_loss") is None:
        epoch_data[key]["l1_loss"] = 0.0
        epoch_data[key]["tv"] = 0.0
        epoch_data[key]["grad_diff_loss"] = 0.0
        epoch_data[key]["ssim_loss"] = 0.0
        epoch_data[key]["reproj_loss"] = 0.0
        epoch_data[key]["lp_loss"] = 0.0
        epoch_data[key]["total_loss"] = 0.0

    prob = torch.sigmoid(pred)

    l1_height = F.l1_loss(prob, gt).float()
    epoch_data[key]["l1_loss"] += l1_height.item()

    # Gradient total variation (TV) smoothness penalty
    # # [w0 - w1, w1 - w2, ..., wN-1 - wN]
    dx = torch.abs(prob[..., :-1] - prob[..., 1:]).mean().float()
    # # [h0 - h1, h1 - h2, ..., hN-1 - hN]
    dy = torch.abs(prob[..., :-1, :] - prob[..., 1:, :]).mean().float()
    tv = dx + dy
    epoch_data[key]["tv"] += tv.item()

    # Gradient difference loss
    height_grad_diff_loss = gradient_difference_loss(prob, gt)
    epoch_data[key]["grad_diff_loss"] += height_grad_diff_loss.item()

    # SSIM loss for height
    ssim_height = FM.multiscale_structural_similarity_index_measure(
        prob,
        gt.clamp(0, 1).float(),
        data_range=1.0,
    )
    ssim_height = torch.nan_to_num(ssim_height, nan=1.0).float()
    height_ssim_loss = (1 - ssim_height).float()
    epoch_data[key]["ssim_loss"] += height_ssim_loss.item()

    # tv2 = tv2_loss(pred)
    # epoch_data[key]["tv2"] += tv2.item()

    # normal_loss = normal_consistency_loss(prob, normal_map)
    # epoch_data[key]["normal_consistency_loss"] += normal_loss.item()

    # linear decay 0->6 epochs: 0.25 → 0.1, then flat
    # normal_loss_weight = 0.25 if epoch < 6 else 0.10
    # normal_loss_weight = 0.25 - (0.25 - 0.10) * min(epoch, 6) / 6

    # Variance matching loss
    # pred_std = prob.view(prob.size(0), -1).std(dim=1).float()
    # gt_std = gt.view(gt.size(0), -1).std(dim=1).float()

    # var_loss = ((pred_std - gt_std).pow(2)).mean()
    # epoch_data[key]["var_loss"] += var_loss.item()

    reproj_loss = normal_reprojection_loss(prob, normal_map)
    if PHASE == "p0" or PHASE == "p1":
        reproj_weight = 0.15
    elif PHASE == "p2":
        reproj_weight = 0.10
    else:
        reproj_weight = 0.10

    epoch_data[key]["reproj_loss"] += reproj_loss.item()

    lp_pred = laplacian_pyramid(prob)
    lp_gt = laplacian_pyramid(gt)

    loss_lp = sum(
        F.l1_loss(a, b) * (2**-i) for i, (a, b) in enumerate(zip(lp_pred, lp_gt))
    )
    epoch_data[key]["lp_loss"] += loss_lp.item()  # type: ignore

    loss_height = (
        l1_height
        + 0.25 * height_grad_diff_loss
        + 0.06 * tv
        + 0.06 * height_ssim_loss
        + reproj_weight * reproj_loss
        + 0.1 * loss_lp
    )

    epoch_data[key]["total_loss"] += loss_height.item()
    return loss_height


def get_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    normal_map: torch.Tensor,
    epoch_data: dict,
    key="train",
) -> torch.Tensor:
    if UNET_MAP == "roughness":
        return calculate_rougness_loss(pred, gt, epoch_data, key)
    elif UNET_MAP == "metallic":
        return calculate_metallic_loss(pred, gt, epoch_data, key)
    elif UNET_MAP == "ao":
        return calculate_ao_loss(pred, gt, epoch_data, key)
    elif UNET_MAP == "parallax":
        return calculate_height_loss(pred, gt, normal_map, epoch_data, key)
    else:
        raise ValueError(f"Unknown loss type: {UNET_MAP}")


def calculate_avg(epoch_data, key="train"):
    total_batches = epoch_data[key]["batch_count"]

    for k in epoch_data[key].keys():
        if k != "batch_count":
            epoch_data[key][k] = epoch_data[key][k] / total_batches

    return epoch_data[key]["total_loss"]


def cycle(dl: DataLoader):
    while True:
        for batch in dl:
            yield batch


def to_rgb(x):
    # x: (1, H, W) → (3, H, W) by repeating the gray channel
    return x.repeat(3, 1, 1)


def mean_curvature_map(normals: torch.Tensor) -> torch.Tensor:
    """
    normals: (B,3,H,W) in [-1,1]
    returns: (B,1,H,W) curvature map in [-1,1]
    """

    # 1) Median-filter to kill salt-and-pepper spikes
    #    (kernel size 3x3 on each channel)
    normals = K.filters.median_blur(normals, (3, 3))

    # 2) Compute Sobel gradients → shape (B,2,3,H,W)
    grad = K.filters.spatial_gradient(normals, mode="sobel", order=1)
    grad_x, grad_y = grad[:, 0], grad[:, 1]  # each (B,3,H,W)

    # 3) Approximate mean curvature = 0.5*(∂x n_x + ∂y n_y)
    dnx_dx = grad_x[:, 0:1]  # (B,1,H,W)
    dny_dy = grad_y[:, 1:2]  # (B,1,H,W)
    curv = 0.5 * (dnx_dx + dny_dy)

    # 4) Absolute value → all positive
    curv = curv.abs()  # (B,1,H,W)

    # 5) Percentile-normalize per-image (99th percentile)
    #    Flatten spatial dims, compute per-sample 99% value
    flat = curv.view(curv.shape[0], -1)
    p99 = torch.quantile(flat, 0.99, dim=1)  # (B,)
    p99 = p99.view(-1, 1, 1, 1).clamp(min=1e-6)  # avoid zero
    curv = (curv / p99).clamp(0.0, 1.0)

    # Remap to -1 to 1 range
    curv = (curv - 0.5) * 2.0

    return curv


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

    # for param in unet_maps.unet.decoder.parameters():
    #     param.requires_grad = True

    # for param in unet_maps.unet.film.parameters():  # type: ignore
    #     param.requires_grad = True

    # for param in unet_maps.head.parameters():
    #     param.requires_grad = True

    skyrim_train_loader = DataLoader(
        skyrim_train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATASET_WORKERS,
        prefetch_factor=2,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    skyrim_validation_loader = DataLoader(
        skyrim_validation_dataset,
        batch_size=BATCH_SIZE_VALIDATION,
        shuffle=False,
        num_workers=DATASET_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    skyrim_train_iter = cycle(skyrim_train_loader)

    skyrim_validation_iter = cycle(skyrim_validation_loader)

    # base_enc_lr = 1e-5
    # base_dec_lr = 4e-5
    base_enc_lr = 5e-5
    base_dec_lr = 2e-4

    # ❶ encoder with LLRD
    depth_map = {
        0: ["unet.inc."],
        1: ["unet.encoder.0."],
        2: ["unet.encoder.1."],
        3: ["unet.encoder.2."],
        4: ["unet.bot."],
    }

    n_blocks = len(depth_map)
    param_groups = []
    gamma = 0.9
    for depth, prefixes in depth_map.items():
        lr = base_enc_lr * (gamma ** (n_blocks - depth - 1))
        params = [
            p
            for n, p in unet_maps.named_parameters()
            if any(n.startswith(pref) for pref in prefixes)
        ]
        param_groups.append({"params": params, "lr": lr, "weight_decay": WD})

    # decoder + film + each head all at base_dec_lr
    param_groups += [
        # {
        #     "params": unet_maps.unet.encoder.parameters(),
        #     "lr": base_enc_lr,
        #     "weight_decay": WD,
        # },
        # {
        #     "params": unet_maps.unet.inc.parameters(),
        #     "lr": base_enc_lr,
        #     "weight_decay": WD,
        # },
        # {
        #     "params": unet_maps.unet.bot.parameters(),
        #     "lr": base_enc_lr,
        #     "weight_decay": WD,
        # },
        {
            "params": unet_maps.unet.decoder.parameters(),
            "lr": base_dec_lr,
            "weight_decay": WD,
        },
        {"params": unet_maps.unet.film.parameters(), "lr": base_dec_lr, "weight_decay": 0.0},  # type: ignore # No WD for FiLM
        # {"params": unet_maps.unet.mask_film.parameters(), "lr": base_dec_lr, "weight_decay": 0.0},  # type: ignore # No WD for FiLM
        {
            "params": unet_maps.head.parameters(),
            "lr": base_dec_lr,
            "weight_decay": WD,
        },
    ]
    # for n, p in unet_maps.named_parameters():
    #     print(f"Param: {n}")

    # trainable = [p for p in unet_maps.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        param_groups,
        # trainable,
        # param_groups,
        # lr=base_dec_lr,
        # weight_decay=WD,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    if checkpoint is not None and resume_training:
        print("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 1 epoch warm-up to the base LR
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.3, end_factor=1.0, total_iters=STEPS_PER_EPOCH_TRAIN
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(EPOCHS - 1) * STEPS_PER_EPOCH_TRAIN,
        # eta_min=base_enc_lr * 0.05,
        eta_min=base_dec_lr * 0.05,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[STEPS_PER_EPOCH_TRAIN],  # After first epoch switch to cosine
    )

    if checkpoint is not None and resume_training:
        print("Loading scheduler state from checkpoint.")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    scaler = GradScaler(device.type)  # AMP scaler for mixed precision
    if checkpoint is not None and resume_training:
        print("Loading scaler state from checkpoint.")
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # for pg in optimizer.param_groups:
    #     pg["lr"] = pg["lr"] * 0.1
    #     print(
    #         f"Param group: {pg['params'][0].name}, LR: {pg['lr']:.1e}, WD: {pg['weight_decay']}"
    #     )

    best_val_loss = float("inf")
    if checkpoint is not None and args.load_best_loss and resume_training:
        best_val_loss = checkpoint["epoch_data"]["validation"]["total_loss"]
        print(f"Loading best validation loss from checkpoint: {best_val_loss}")

    patience = 6
    no_improvement_count = 0

    output_dir = Path(f"./weights/{PHASE}/unet_{UNET_MAP}").resolve()
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
        # accum_steps = 2

        bar = tqdm(
            range(STEPS_PER_EPOCH_TRAIN),
            desc=f"Epoch {epoch + 1}/{EPOCHS} - Training",
            unit="batch",
        )

        # optimizer.zero_grad(set_to_none=True)
        for i in bar:
            skyrim_batch = next(skyrim_train_iter)

            diffuse_and_normal = skyrim_batch["diffuse_and_normal"]
            albedo_and_normal_segformer = skyrim_batch["albedo_and_normal_segformer"]
            normal = skyrim_batch["normal"]
            poisson_blur = skyrim_batch["poisson_blur"]

            gt = skyrim_batch[UNET_MAP]

            diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
            albedo_and_normal_segformer = albedo_and_normal_segformer.to(
                device, non_blocking=True
            )
            normal = normal.to(device, non_blocking=True)
            poisson_blur = poisson_blur.to(device, non_blocking=True)

            gt = gt.to(device, non_blocking=True)

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

            if UNET_MAP == "parallax" or UNET_MAP == "ao":
                curvature = mean_curvature_map(normal)
                input = torch.cat(
                    [
                        normal,
                        curvature,
                        poisson_blur,
                    ],
                    dim=1,
                )

            if UNET_MAP == "roughness":
                input = torch.cat(
                    [
                        predicted_albedo,
                        normal,
                    ],
                    dim=1,
                )

            with autocast(device_type=device.type):
                # Get predicted map from UNet
                predicted = unet_maps(input, seg_feats)

            loss = get_loss(
                predicted,
                gt,
                normal,
                epoch_data,
                key="train",
            )

            if torch.isnan(loss):
                raise ValueError(
                    "Unet loss is NaN, stopping training to avoid further issues."
                )

            epoch_data["train"]["batch_count"] += 1

            # loss = loss / accum_steps  # Scale loss for accumulation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet_maps.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            # if (i + 1) % accum_steps == 0 or (i + 1) == STEPS_PER_EPOCH_TRAIN:
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad(set_to_none=True)
            scheduler.step()

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
                name = skyrim_batch["name"]
                albedo_gt = skyrim_batch["albedo"]
                diffuse_gt = skyrim_batch["orig_diffuse"]
                normal_gt = skyrim_batch["orig_normal"]
                poisson_blur = skyrim_batch["poisson_blur"]

                gt = skyrim_batch[UNET_MAP]

                diffuse_and_normal = diffuse_and_normal.to(device, non_blocking=True)
                albedo_and_normal_segformer = albedo_and_normal_segformer.to(
                    device, non_blocking=True
                )
                normal = normal.to(device, non_blocking=True)
                diffuse_gt = diffuse_gt.to(device, non_blocking=True)
                normal_gt = normal_gt.to(device, non_blocking=True)
                albedo_gt = albedo_gt.to(device, non_blocking=True)
                poisson_blur = poisson_blur.to(device, non_blocking=True)

                gt = gt.to(device, non_blocking=True)

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

                # predicted_albedo_orig = predicted_albedo
                predicted_albedo = TF.normalize(
                    predicted_albedo,
                    mean=IMAGENET_STANDARD_MEAN,
                    std=IMAGENET_STANDARD_STD,
                )
                predicted_albedo = predicted_albedo.to(device, non_blocking=True)

                if UNET_MAP == "parallax" or UNET_MAP == "ao":
                    curvature = mean_curvature_map(normal)
                    input = torch.cat(
                        [
                            normal,
                            curvature,
                            poisson_blur,
                        ],
                        dim=1,
                    )
                if UNET_MAP == "roughness":
                    input = torch.cat(
                        [
                            predicted_albedo,
                            normal,
                        ],
                        dim=1,
                    )

                with autocast(device_type=device.type):
                    # Get predicted map from UNet
                    predicted = unet_maps(input, seg_feats)

                get_loss(
                    predicted,
                    gt,
                    normal,
                    epoch_data,
                    key="validation",
                )

                epoch_data["validation"]["batch_count"] += 1

                if (
                    VISUAL_SAMPLES_COUNT > 0
                    and num_samples_saved < VISUAL_SAMPLES_COUNT
                ):
                    for (
                        sample_name,
                        sample_diffuse,
                        sample_normal,
                        # sample_albedo_gt,
                        gt,
                        # sample_albedo_pred,
                        predicted,
                    ) in zip(
                        name,
                        diffuse_gt,
                        normal_gt,
                        # albedo_gt,
                        gt,
                        # predicted_albedo_orig,
                        predicted,
                    ):

                        # Save few samples per class for inspection
                        output_path = output_dir / f"val_samples_{epoch + 1}"
                        output_path.mkdir(parents=True, exist_ok=True)

                        visual_sample_gt = torch.cat(
                            [
                                sample_diffuse,
                                sample_normal,
                                # sample_albedo_gt,
                                to_rgb(gt),
                            ],
                            dim=2,  # Concatenate along width
                        )
                        visual_sample_predicted = torch.cat(
                            [
                                sample_diffuse,
                                sample_normal,
                                # sample_albedo_pred,
                                to_rgb(torch.sigmoid(predicted)),
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

        # scheduler.step()
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
