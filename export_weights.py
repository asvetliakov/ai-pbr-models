import training_scripts.seed
from pathlib import Path
import torch
from training_scripts.class_materials import CLASS_LIST
from training_scripts.segformer_6ch import create_segformer
from training_scripts.unet_models import UNetAlbedo, UNetSingleChannel

BASE_DIR = Path(__file__).resolve().parent

WEIGHT_DIR = (BASE_DIR / "weights").resolve()
EXPORT_DIR = (BASE_DIR / "stored_weights").resolve()

EXPORT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda")

segformer_weights_path = WEIGHT_DIR / "s3/segformer/best_model.pt"
print(f"Loading segformer weights from {segformer_weights_path}")
segformer_checkpoint = torch.load(segformer_weights_path, map_location=device)
segformer_weights = segformer_checkpoint["model_state_dict"]
torch.save(segformer_weights, EXPORT_DIR / "s3/segformer/best_model.pt")

segformer_weights_path_alt = WEIGHT_DIR / "s4/segformer/best_model.pt"
print(f"Loading segformer weights from {segformer_weights_path_alt}")
segformer_checkpoint_alt = torch.load(segformer_weights_path_alt, map_location=device)
segformer_weights_alt = segformer_checkpoint_alt["model_state_dict"]
torch.save(segformer_weights_alt, EXPORT_DIR / "s4/segformer/best_model.pt")

unet_albedo_best_weights_path = WEIGHT_DIR / "a4/unet_albedo/best_model.pt"
print("Loading Unet-albedo weights from:", unet_albedo_best_weights_path)
unet_albedo_checkpoint = torch.load(unet_albedo_best_weights_path, map_location=device)
unet_albedo_weigths = unet_albedo_checkpoint["unet_albedo_model_state_dict"]
torch.save(unet_albedo_weigths, EXPORT_DIR / "a4/unet_albedo/best_model.pt")

unet_parallax_best_weights_path = WEIGHT_DIR / "m3/unet_parallax/best_model.pt"
print("Loading Unet-parallax weights from:", unet_parallax_best_weights_path)
unet_parallax_checkpoint = torch.load(
    unet_parallax_best_weights_path, map_location=device
)
unet_parallax_weights = unet_parallax_checkpoint["unet_maps_model_state_dict"]
torch.save(unet_parallax_weights, EXPORT_DIR / "m3/unet_parallax/best_model.pt")

unet_ao_best_weights_path = WEIGHT_DIR / "m3/unet_ao/best_model.pt"
print("Loading Unet-ao weights from:", unet_ao_best_weights_path)
unet_ao_checkpoint = torch.load(unet_ao_best_weights_path, map_location=device)
unet_ao_weights = unet_ao_checkpoint["unet_maps_model_state_dict"]
torch.save(unet_ao_weights, EXPORT_DIR / "m3/unet_ao/best_model.pt")

unet_metallic_best_weights_path = WEIGHT_DIR / "m3/unet_metallic/best_model.pt"
print("Loading Unet-metallic weights from:", unet_metallic_best_weights_path)
unet_metallic_checkpoint = torch.load(
    unet_metallic_best_weights_path, map_location=device
)
unet_metallic_weights = unet_metallic_checkpoint["unet_maps_model_state_dict"]
torch.save(unet_metallic_weights, EXPORT_DIR / "m3/unet_metallic/best_model.pt")

unet_roughness_best_weights_path = WEIGHT_DIR / "m3/unet_roughness/best_model.pt"
print("Loading Unet-roughness weights from:", unet_roughness_best_weights_path)
unet_roughness_checkpoint = torch.load(
    unet_roughness_best_weights_path, map_location=device
)
unet_roughness_weights = unet_roughness_checkpoint["unet_maps_model_state_dict"]
torch.save(unet_roughness_weights, EXPORT_DIR / "m3/unet_roughness/best_model.pt")
