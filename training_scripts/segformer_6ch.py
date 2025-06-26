import torch
from transformers import SegformerForSemanticSegmentation
from transformers.utils import logging


def create_segformer(
    num_labels: int, device: torch.device
) -> SegformerForSemanticSegmentation:
    logging.set_verbosity_error()  # Suppress warnings from transformers
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_labels,  # Number of classes for segmentation
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

    return model
