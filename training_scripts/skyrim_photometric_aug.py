import seed

"""
Plug‑and‑play torchvision-style transforms that implement:
1. AO tint      (approximate: darken by inverted Gaussian-blurred luminance)
2. Cold WB      (simple RGB gains)
3. Vignette     (radial falloff)
Attach them with probability p_aug (e.g. 0.6) to Skyrim diffuse inputs.

How to integrate:

from torchvision import transforms
from skyrim_photometric_aug import SkyrimPhotometric

skyrim_transform = transforms.Compose([
        SkyrimPhotometric(p_aug=0.6),     # <= NEW
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),            # rest of your pipeline…
])
"""

import math, random, torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class AOTint:
    def __init__(self, strength_range=(0.10, 0.25), kernel_size=23):
        self.strength_range = strength_range
        self.kernel_size = kernel_size

    def __call__(self, img):
        """
        img: Float tensor in [0,1], shape (C,H,W)
        """
        strength = random.uniform(*self.strength_range)
        # fast approx.: AO mask = blurred inverse‑luma
        luma = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]  # (H,W)
        luma = luma.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        pad = self.kernel_size // 2
        weight = torch.ones(
            1, 1, self.kernel_size, self.kernel_size, device=img.device
        ) / (self.kernel_size**2)
        luma_blur = F.conv2d(luma, weight, padding=pad)  # (1,1,H,W)
        ao_mask = 1.0 - luma_blur.squeeze()  # (H,W)
        ao_tint = 1.0 - strength * ao_mask
        img = img * ao_tint.clamp(0, 1)
        return img.clamp(0, 1)


class ColdWhiteBalance:
    def __init__(self, r_mult=(0.90, 0.98), b_mult=(1.02, 1.10)):
        self.r_mult = r_mult
        self.b_mult = b_mult

    def __call__(self, img):
        r_gain = random.uniform(*self.r_mult)
        b_gain = random.uniform(*self.b_mult)
        img = img.clone()
        img[0] = img[0] * r_gain
        img[2] = img[2] * b_gain
        return img.clamp(0, 1)


class Vignette:
    def __init__(self, edge_gain=(0.85, 0.95)):
        self.edge_gain = edge_gain

    def __call__(self, img):
        h, w = img.shape[-2:]
        y = torch.linspace(-1, 1, h, device=img.device).view(-1, 1)
        x = torch.linspace(-1, 1, w, device=img.device).view(1, -1)
        radius = torch.sqrt(x**2 + y**2)
        # soft fall‑off:  smoothstep
        edge = random.uniform(*self.edge_gain)
        weight = 1 - (1 - edge) * torch.clamp((radius - 0.5) / 0.5, 0, 1) ** 2
        img = img * weight
        return img.clamp(0, 1)


class SkyrimPhotometric:
    """
    Combine all three with independent 50% probability each,
    wrapped in an outer p_aug probability.
    """

    def __init__(self, p_aug=0.6):
        self.p_aug = p_aug
        self.ao = AOTint()
        self.wb = ColdWhiteBalance()
        self.vg = Vignette()

    def __call__(self, img):
        if random.random() > self.p_aug:
            return img
        # torchvision img → tensor in [0,1]
        img = TF.to_tensor(img) if not torch.is_tensor(img) else img
        if random.random() < 0.5:
            img = self.ao(img)
        if random.random() < 0.5:
            img = self.wb(img)
        if random.random() < 0.5:
            img = self.vg(img)
        return TF.to_pil_image(img)  # return same type as input
