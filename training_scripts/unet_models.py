import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence


# -------------------------------------------------
# Core blocks
# -------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv → GN → SiLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad in case of odd input dimensions
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.pad(x, [0, skip.size(-1) - x.size(-1), 0, skip.size(-2) - x.size(-2)])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -------------------------------------------------
# FiLM (γ, β) conditioning block
# -------------------------------------------------
class FiLM(nn.Module):
    """
    Learns two 1×1 convs that map SegFormer feature maps
    to channel‑wise γ (scale) and β (shift) parameters.
    """

    def __init__(self, cond_ch: int, target_ch: int):
        super().__init__()
        self.to_gamma = nn.Conv2d(cond_ch, target_ch, 1)
        self.to_beta = nn.Conv2d(cond_ch, target_ch, 1)

    def forward(self, x, cond):
        gamma = torch.tanh(self.to_gamma(cond))  # keep scale in [-1,1]
        beta = self.to_beta(cond)
        return x * (1 + gamma) + beta


# -------------------------------------------------
# Backbone UNet
# -------------------------------------------------
class _UNetBackbone(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        base: int = 64,
        depth: int = 4,
        cond_ch: Optional[int] = None,  # channels of SegFormer feature map
    ):
        super().__init__()
        self.depth = depth
        self.inc = DoubleConv(in_ch, base)
        # encoder
        enc = []
        for i in range(1, depth):
            enc.append(Down(base * 2 ** (i - 1), base * 2**i))
        self.encoder = nn.ModuleList(enc)
        # bottleneck
        self.bot = Down(base * 2 ** (depth - 1), base * 2**depth)
        # decoder
        dec = []
        for i in reversed(range(1, depth + 1)):
            in_ch_dec = base * 2**i + base * 2 ** (i - 1)
            dec.append(Up(in_ch_dec, base * 2 ** (i - 1)))
        self.decoder = nn.ModuleList(dec)

        # optional FiLM
        self.film = FiLM(cond_ch, base * 2**depth) if cond_ch is not None else None

    def forward(self, x, cond: Optional[torch.Tensor] = None):
        # x  : (B, in_ch, H, W)
        # cond: (B, cond_ch, H/16, W/16)  ← SegFormer stage 4
        skips = [self.inc(x)]
        for down in self.encoder:
            skips.append(down(skips[-1]))

        bottleneck = self.bot(skips[-1])
        if self.film is not None and cond is not None:
            # up‑sample cond to bottleneck spatial size
            cond = F.interpolate(
                cond, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False
            )
            bottleneck = self.film(bottleneck, cond)

        out = bottleneck
        for up, skip in zip(self.decoder, reversed(skips)):
            out = up(out, skip)
        return out


# -------------------------------------------------
# 2a · UNet‑Albedo
# -------------------------------------------------
class UNetAlbedo(nn.Module):
    """
    Outputs a 3‑channel linear‑space albedo (RGB ∈ [0,1]).
    """

    def __init__(self, in_ch: int = 3, cond_ch: Optional[int] = 256):
        super().__init__()
        self.unet = _UNetBackbone(in_ch, base=64, depth=4, cond_ch=cond_ch)
        self.out = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),  # keeps outputs in (0,1)
        )

    def forward(self, img, segfeat=None):
        x = self.unet(img, segfeat)
        return self.out(x)


# -------------------------------------------------
# 2b · UNet‑Maps
# -------------------------------------------------
class UNetMaps(nn.Module):
    """
    Predicts 4 PBR maps in a single forward pass:

        • Roughness  ‖ 1ch, [0,1]
        • Metallic   ‖ 1ch, [0,1]
        • AO         ‖ 1ch, [0,1]
        • Height     ‖ 1ch, [0,1]
    """

    def __init__(self, in_ch: int = 3, cond_ch: Optional[int] = 256):
        super().__init__()
        self.unet = _UNetBackbone(in_ch, base=64, depth=4, cond_ch=cond_ch)

        # individual 1×1 heads
        self.head_rough = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.head_metal = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.head_ao = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.head_h = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, img, segfeat=None):
        x = self.unet(img, segfeat)
        return {
            "rough": self.head_rough(x),
            "metal": self.head_metal(x),
            "ao": self.head_ao(x),
            "height": self.head_h(x),
        }


class UNetSingleChannel(nn.Module):
    def __init__(self, in_ch: int = 3, cond_ch: Optional[int] = 256):
        super().__init__()
        self.unet = _UNetBackbone(in_ch, base=64, depth=4, cond_ch=cond_ch)
        self.out = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, img, segfeat=None):
        x = self.unet(img, segfeat)
        return self.out(x)
