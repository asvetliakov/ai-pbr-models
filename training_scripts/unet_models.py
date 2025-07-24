import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence


# -------------------------------------------------
# Lightweight Self-Attention for small datasets
# -------------------------------------------------
class LightweightSelfAttention(nn.Module):
    """Minimal self-attention with heavy regularization for small datasets"""

    def __init__(self, channels: int, reduction: int = 16, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        reduced_ch = max(channels // reduction, 8)  # High reduction to minimize params

        self.query = nn.Conv2d(channels, reduced_ch, 1, bias=False)
        self.key = nn.Conv2d(channels, reduced_ch, 1, bias=False)
        self.value = nn.Conv2d(channels, reduced_ch, 1, bias=False)
        self.out_proj = nn.Conv2d(reduced_ch, channels, 1, bias=False)

        # Start with very small influence
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout2d(dropout)

        # Small weight initialization
        for m in [self.query, self.key, self.value, self.out_proj]:
            nn.init.xavier_normal_(m.weight, gain=0.1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Always use pooling for efficiency and regularization
        if H > 24 or W > 24:
            pool_size = H // 24 + 1 if H > 24 else 1
            x_pool = F.avg_pool2d(x, pool_size)
            _, _, H_p, W_p = x_pool.shape

            q = self.query(x_pool).view(B, -1, H_p * W_p).permute(0, 2, 1)
            k = self.key(x_pool).view(B, -1, H_p * W_p)
            v = self.value(x_pool).view(B, -1, H_p * W_p)

            attn = torch.bmm(q, k) / math.sqrt(q.size(-1))
            attn = F.softmax(attn, dim=-1)

            out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, -1, H_p, W_p)
            out = self.out_proj(out)
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        else:
            q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
            k = self.key(x).view(B, -1, H * W)
            v = self.value(x).view(B, -1, H * W)

            attn = torch.bmm(q, k) / math.sqrt(q.size(-1))
            attn = F.softmax(attn, dim=-1)

            out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, -1, H, W)
            out = self.out_proj(out)

        if self.training:
            # Only apply dropout during training
            out = self.dropout(out)

        # Very conservative blending to prevent overfitting
        return x + self.gamma * out * 0.05


# -------------------------------------------------
# Core blocks
# -------------------------------------------------


class DoubleConv(nn.Module):
    """(Conv → GN → SiLU) × 2 with adaptive normalization"""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        # Adaptive groups: fewer groups for smaller channel counts (more stable)
        groups = min(groups, out_ch // 4) if out_ch >= 16 else min(groups, out_ch)
        groups = max(1, groups)  # Ensure at least 1 group

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
    def __init__(self, in_ch: int, out_ch: int, use_antialiasing: bool = True):
        super().__init__()
        if use_antialiasing:
            # Anti-aliasing: smooth downsampling to prevent high-frequency artifacts
            self.block = nn.Sequential(
                # First: depthwise convolution with stride=2 (smoother than max pooling)
                nn.Conv2d(
                    in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False
                ),
                # Then: pointwise convolution to change channels
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(min(8, out_ch // 4) if out_ch >= 16 else 1, out_ch),
                nn.SiLU(inplace=True),
                # Finally: refinement with DoubleConv
                DoubleConv(out_ch, out_ch),
            )
        else:
            # Original approach with max pooling
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

        # Improved skip connections: feature alignment and fusion
        skip_ch = out_ch  # Skip connection channel count

        # Skip connection processing: refine features before fusion
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_ch, skip_ch, 1, bias=False),
            nn.GroupNorm(min(8, skip_ch // 4) if skip_ch >= 16 else 1, skip_ch),
            nn.SiLU(inplace=True),
        )

        # Channel attention for better feature selection
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 16, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch // 16, in_ch, 1),
            nn.Sigmoid(),
        )

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad in case of odd input dimensions
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.pad(x, [0, skip.size(-1) - x.size(-1), 0, skip.size(-2) - x.size(-2)])

        # Process skip connection for better feature quality
        skip_refined = self.skip_conv(skip)

        # Concatenate and apply channel attention
        fused = torch.cat([skip_refined, x], dim=1)
        attention_weights = self.channel_attention(fused)
        fused = fused * attention_weights

        return self.conv(fused)


# -------------------------------------------------
# FiLM (γ, β) conditioning block
# -------------------------------------------------
class FiLM(nn.Module):
    """
    Learns two 1×1 convs that map SegFormer feature maps
    to channel‑wise γ (scale) and β (shift) parameters.
    """

    def __init__(self, cond_ch: int, target_ch: int, kernel_size: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.to_gamma = nn.Conv2d(
            cond_ch, target_ch, kernel_size=kernel_size, padding=pad
        )
        self.to_beta = nn.Conv2d(
            cond_ch, target_ch, kernel_size=kernel_size, padding=pad
        )

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
        mask_film: bool = False,
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

        # Lightweight attention at bottleneck for high-res (minimal params, heavy regularization)
        self.bottleneck_attention = LightweightSelfAttention(
            base * 2**depth, reduction=32, dropout=0.2  # Very conservative settings
        )

        # decoder
        dec = []
        for i in reversed(range(1, depth + 1)):
            in_ch_dec = base * 2**i + base * 2 ** (i - 1)
            dec.append(Up(in_ch_dec, base * 2 ** (i - 1)))
        self.decoder = nn.ModuleList(dec)

        # Segformer FiLM
        self.film = FiLM(cond_ch, base * 2**depth) if cond_ch is not None else None
        # Late-fusion mask FiLM
        self.mask_film = FiLM(1, base * 2**depth, kernel_size=3) if mask_film else None

        if self.mask_film is not None:
            # initialize so mask has near-zero influence at start
            nn.init.constant_(self.mask_film.to_gamma.bias, -2.0)  # type: ignore
            nn.init.constant_(self.mask_film.to_beta.bias, 0.0)  # type: ignore

    def forward(
        self,
        x,
        cond: Optional[torch.Tensor] = None,
        gray_mask: Optional[torch.Tensor] = None,
    ):
        # x  : (B, in_ch, H, W)
        # cond: (B, cond_ch, H/16, W/16)  ← SegFormer stage 4
        skips = [self.inc(x)]
        for down in self.encoder:
            skips.append(down(skips[-1]))

        bottleneck = self.bot(skips[-1])

        # Apply lightweight attention for better high-res modeling (only if training)
        # if self.training:  # Only during training to prevent overfitting
        bottleneck = self.bottleneck_attention(bottleneck)

        # --- SegFormer FiLM  ---
        if self.film is not None and cond is not None:
            # up‑sample cond to bottleneck spatial size
            cond = F.interpolate(
                cond, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False
            )
            bottleneck = self.film(bottleneck, cond)

        # --- Late-fusion mask FiLM ---
        if self.mask_film is not None and gray_mask is not None:
            gm = F.interpolate(
                gray_mask, bottleneck.shape[-2:], mode="bilinear", align_corners=False
            )
            bottleneck = self.mask_film(bottleneck, gm)

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
        self.unet = _UNetBackbone(in_ch, base=96, depth=4, cond_ch=cond_ch)
        self.out = nn.Sequential(
            nn.Conv2d(96, 3, 1),
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
        self.unet = _UNetBackbone(in_ch, base=96, depth=4, cond_ch=cond_ch)

        # individual 1×1 heads
        self.head_rough = nn.Sequential(nn.Conv2d(96, 1, 1), nn.Sigmoid())
        self.head_metal = nn.Sequential(nn.Conv2d(96, 1, 1), nn.Sigmoid())
        self.head_ao = nn.Sequential(nn.Conv2d(96, 1, 1), nn.Sigmoid())
        self.head_h = nn.Sequential(nn.Conv2d(96, 1, 1), nn.Sigmoid())

    def forward(self, img, segfeat=None):
        x = self.unet(img, segfeat)
        return {
            "rough": self.head_rough(x),
            "metal": self.head_metal(x),
            "ao": self.head_ao(x),
            "height": self.head_h(x),
        }


class UNetSingleChannel(nn.Module):
    def __init__(
        self, in_ch: int = 3, cond_ch: Optional[int] = 256, mask_film: bool = False
    ):
        super().__init__()
        self.unet = _UNetBackbone(
            in_ch, base=96, depth=4, cond_ch=cond_ch, mask_film=mask_film
        )
        self.head = nn.Conv2d(96, 1, 1)

    def forward(self, img, segfeat=None, gray_mask=None):
        x = self.unet(img, segfeat, gray_mask)
        return self.head(x)
