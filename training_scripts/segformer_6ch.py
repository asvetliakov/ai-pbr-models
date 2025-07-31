import torch
from transformers import SegformerForSemanticSegmentation
from transformers.utils import logging
from typing import Optional

# from types import MethodType
# from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def create_segformer(
    num_labels: int,
    device: torch.device,
    lora=False,
    base_model_state: Optional[dict] = None,
    frozen=False,
) -> SegformerForSemanticSegmentation:
    logging.set_verbosity_error()  # Suppress warnings from transformers
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_labels,  # Number of classes for segmentation
        ignore_mismatched_sizes=True,  # Ignore size mismatch for classification head
    ).to(
        device  # type: ignore
    )

    # 1. Patch segformer for 6channel input
    old_embed = model.segformer.encoder.patch_embeddings[0]
    old_conv = old_embed.proj  # Conv2d(3,64,kernel=7,stride=4,pad=3)

    # Create a new one for 6-channel input
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

    # Copy pretrained RGB weights → channels 0–2
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight  # type: ignore
        # 4) Init the extra normal channels → 3–5
        torch.nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode="fan_out")
        if old_conv.bias is not None:  # type: ignore
            new_conv.bias.copy_(old_conv.bias)  # type: ignore # Copy bias if it exists

    # Replace it back in the model
    model.segformer.encoder.patch_embeddings[0].proj = new_conv

    # Update config so future code knows to expect 6 channels
    model.config.num_channels = 6

    if base_model_state is not None:
        print("Loading base model state dict from checkpoint. (no-LoRA)")
        model.load_state_dict(base_model_state)

    # if lora:
    #     # LoRA added in S2
    #     lora_config = LoraConfig(
    #         r=16,  # Rank of the LoRA layers # type: ignore
    #         lora_alpha=32,  # Scaling factor for the LoRA layers # type: ignore
    #         target_modules=[  # type: ignore
    #             "attention.self.query",
    #             "attention.self.value",
    #         ],  # Target modules to apply LoRA to
    #         lora_dropout=0.05,  # Dropout rate for the LoRA layers # type: ignore
    #         bias="none",  # No bias in LoRA layers # type: ignore
    #         task_type=TaskType.FEATURE_EXTRACTION,  # type: ignore
    #     )
    #     model = get_peft_model(model, lora_config).to(device)

    #     def segformer_peft_forward(self, *args, **kwargs):
    #         # 1) If PEFT passes input_ids=…, treat it as pixel_values.
    #         if "input_ids" in kwargs and "pixel_values" not in kwargs:
    #             kwargs["pixel_values"] = kwargs.pop("input_ids")
    #         # 2) Call the original PEFT forward
    #         return super(type(self), self).forward(*args, **kwargs)  # type: ignore

    #         # attach the shim

    #     model.forward = MethodType(segformer_peft_forward, model)

    if frozen:
        for param in model.parameters():
            param.requires_grad = False

    return model  # type: ignore
