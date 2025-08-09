import argparse
from pathlib import Path
import torch

# Local imports
from training_scripts.class_materials import CLASS_LIST
from training_scripts.segformer_6ch import create_segformer
from training_scripts.unet_models import UNetAlbedo, UNetSingleChannel


class SegformerWrapper(torch.nn.Module):
    """
    Wraps a HuggingFace SegFormer to accept 6-channel input and return (logits, last_hidden).
    The base model must already be patched for 6ch via create_segformer.
    """

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor):
        # Always return logits + last hidden state
        out = self.base(pixel_values=x, output_hidden_states=True)
        logits = out.logits
        last_hidden = out.hidden_states[-1]
        return logits, last_hidden


def main():
    parser = argparse.ArgumentParser(
        description="Trace TorchScript models (UNets and SegFormer wrapper)"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=str((Path(__file__).parent / "stored_weights").resolve()),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str((Path(__file__).parent / "weights_ts").resolve()),
    )
    parser.add_argument(
        "--segformer_checkpoint", type=str, choices=["s4", "s4_alt"], default="s4"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device(args.device)
    H, W = args.height, args.width
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    # --- Resolve weights paths (match create_pbr.py) ---
    segformer_weights_path = (
        weights_dir / f"{args.segformer_checkpoint}/segformer/best_model.pt"
    )
    unet_albedo_weights_path = weights_dir / "a4/unet_albedo/best_model.pt"
    unet_parallax_weights_path = weights_dir / "m3/unet_parallax/best_model.pt"
    unet_ao_weights_path = weights_dir / "m3/unet_ao/best_model.pt"
    unet_metallic_weights_path = weights_dir / "m3/unet_metallic/best_model.pt"
    unet_roughness_weights_path = weights_dir / "m3/unet_roughness/best_model.pt"

    # --- Load SegFormer and wrap ---
    print("Loading SegFormer…")
    segformer = create_segformer(
        num_labels=len(CLASS_LIST), device=device, lora=False, frozen=True
    )
    segformer_sd = torch.load(segformer_weights_path, map_location=device)
    segformer.load_state_dict(segformer_sd)
    segformer.eval()

    seg_wrap = SegformerWrapper(segformer).to(device).eval()

    # Trace SegFormer wrapper with a 6ch dummy input
    print("Tracing SegFormer wrapper…")
    ex_seg = torch.randn(1, 6, H, W, device=device)
    with torch.no_grad():
        traced_seg = torch.jit.trace(seg_wrap, ex_seg, strict=False)
    seg_out_path = out_dir / "segformer.ts"
    torch.jit.save(traced_seg, str(seg_out_path))
    print(f"Saved: {seg_out_path}")

    # --- UNet Albedo ---
    print("Tracing UNet Albedo…")
    unet_albedo = UNetAlbedo(in_ch=6, cond_ch=512).to(device)
    unet_albedo_sd = torch.load(unet_albedo_weights_path, map_location=device)
    unet_albedo.load_state_dict(unet_albedo_sd)
    unet_albedo.eval()

    img_albedo = torch.randn(1, 6, H, W, device=device)
    segfeat = torch.randn(1, 512, max(1, H // 16), max(1, W // 16), device=device)

    # Trace conditional version (with segfeats)
    with torch.no_grad():
        traced_albedo = torch.jit.trace(
            unet_albedo, (img_albedo, segfeat), strict=False
        )
    out = out_dir / "unet_albedo.ts"
    torch.jit.save(traced_albedo, str(out))
    print(f"Saved: {out}")

    # Trace unconditional version (without segfeats)
    with torch.no_grad():
        traced_albedo_uncond = torch.jit.trace(unet_albedo, (img_albedo), strict=False)
    out_uncond = out_dir / "unet_albedo_uncond.ts"
    torch.jit.save(traced_albedo_uncond, str(out_uncond))
    print(f"Saved: {out_uncond}")

    # --- UNet Parallax ---
    print("Tracing UNet Parallax…")
    unet_parallax = UNetSingleChannel(in_ch=5, cond_ch=512).to(device)
    unet_parallax_sd = torch.load(unet_parallax_weights_path, map_location=device)
    unet_parallax.load_state_dict(unet_parallax_sd)
    unet_parallax.eval()

    img_par = torch.randn(1, 5, H, W, device=device)
    with torch.no_grad():
        traced_parallax = torch.jit.trace(
            unet_parallax, (img_par, segfeat), strict=False
        )
    out = out_dir / "unet_parallax.ts"
    torch.jit.save(traced_parallax, str(out))
    print(f"Saved: {out}")

    # --- UNet AO ---
    print("Tracing UNet AO…")
    unet_ao = UNetSingleChannel(in_ch=5, cond_ch=512).to(device)
    unet_ao_sd = torch.load(unet_ao_weights_path, map_location=device)
    unet_ao.load_state_dict(unet_ao_sd)
    unet_ao.eval()

    img_ao = torch.randn(1, 5, H, W, device=device)
    with torch.no_grad():
        traced_ao = torch.jit.trace(unet_ao, (img_ao, segfeat), strict=False)
    out = out_dir / "unet_ao.ts"
    torch.jit.save(traced_ao, str(out))
    print(f"Saved: {out}")

    # --- UNet Metallic ---
    print("Tracing UNet Metallic…")
    in_ch_mr = 6 + len(CLASS_LIST)
    unet_metallic = UNetSingleChannel(in_ch=in_ch_mr, cond_ch=512).to(device)
    unet_metallic_sd = torch.load(unet_metallic_weights_path, map_location=device)
    unet_metallic.load_state_dict(unet_metallic_sd)
    unet_metallic.eval()

    img_met = torch.randn(1, in_ch_mr, H, W, device=device)
    with torch.no_grad():
        traced_metallic = torch.jit.trace(
            unet_metallic, (img_met, segfeat), strict=False
        )
    out = out_dir / "unet_metallic.ts"
    torch.jit.save(traced_metallic, str(out))
    print(f"Saved: {out}")

    # --- UNet Roughness ---
    print("Tracing UNet Roughness…")
    unet_roughness = UNetSingleChannel(in_ch=in_ch_mr, cond_ch=512).to(device)
    unet_roughness_sd = torch.load(unet_roughness_weights_path, map_location=device)
    unet_roughness.load_state_dict(unet_roughness_sd)
    unet_roughness.eval()

    img_rough = torch.randn(1, in_ch_mr, H, W, device=device)
    with torch.no_grad():
        traced_roughness = torch.jit.trace(
            unet_roughness, (img_rough, segfeat), strict=False
        )
    out = out_dir / "unet_roughness.ts"
    torch.jit.save(traced_roughness, str(out))
    print(f"Saved: {out}")

    print("All models traced and saved.")


if __name__ == "__main__":
    main()
