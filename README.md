# 🏔️ Skyrim PBR Pipeline — Rev 6.0 (26 Jun 2025)

---

## 0. Legend

-   **S–phases** = SegFormer
-   **A–phases** = UNet-Albedo (`UNetAlbedo` in code)
-   **M–phases** = UNet-Maps (`UNetMaps`)

`cond_ch = 512` ⇒ FiLM conditioning **enabled from the first epoch** for both U-Nets.  
Class weights = `1 / √freq(class)`; WeightedRandomSampler active in every phase except the final hi-res stages.

---

## 1. SegFormer (S)

| Phase | Dataset mix         | Trainables        | **Crop / Feed** | Augment†                                                        | Epochs | Opt & LR        | Scheduler         | Loss                       |
| ----- | ------------------- | ----------------- | --------------- | --------------------------------------------------------------- | ------ | --------------- | ----------------- | -------------------------- |
| S1    | 100 % MatSynth      | enc + dec         | **256**         | none (first 10 ep) flips · rot · colour · composite (30 %/15 %) | 55     | AdamW 1e‑4→1e‑5 | OneCycle          | CE (+√freq)                |
| S2    | 75 % Mat / 25 % Sky | heads + LoRA      | **512**         | S1 + SkyPhotometric 0.6 + composite (25 %/10 %)                 | 10     | AdamW 1e‑5      | cosine‑10, η=2e‑6 | CE + masked‑CE (Sky p>0.8) |
| S3    | 50 % / 50 %         | top‑½ enc + heads | **768**         | SkyPhotometric 0.6                                              | 10     | AdamW 5e‑6      | cosine‑12         | same                       |
| S4    | 50 % / 50 %         | **BN/LN only**    | **1024**        | none                                                            | 2      | AdamW 3e‑6      | cosine‑restart    | CE                         |
| S5    | 50 % / 50 %         | dec‑head + LoRA   | **1024**        | SkyPhotometric 0.5                                              | 8      | AdamW 1e‑6      | cosine‑8          | CE                         |

---

## 2. UNet‑Albedo (A)

|  Phase | Dataset mix         | Trainables          | **Crop / Feed (px)** | Augment†                  | Epochs | Optimiser & LR (per‑group)                            | Scheduler                           | Loss                       |
| -----: | ------------------- | ------------------- | -------------------- | ------------------------- | -----: | ----------------------------------------------------- | ----------------------------------- | -------------------------- |
| **A1** | 50 % Mat / 50 % Sky | full UNet + FiLM    | **256**              | flips · rot, SkyPhoto 0.6 |     45 | AdamW — enc 2e‑4 · dec 2e‑4 · FiLM 3e‑4 · head 2.5e‑4 | OneCycle (pct 0.2, cos, final 1e‑5) | L1 + 0.1 SSIM + 0.08 LPIPS |
| **A2** | 25 % Mat / 75 % Sky | decoder + FiLM      | **512**              | A1 aug + SkyPhoto 0.6     |     14 | AdamW 1e‑5                                            | cosine‑14                           | same                       |
| **A3** | 100 % Sky           | **1 × 1 head only** | **1 024**            | none                      |      5 | Adam 5e‑7                                             | Exp 0.9                             | same                       |

_Save the **best A2** checkpoint → encoder donor for Maps._

---

## 3. Separate Unet per map (M)

## 3.1 Roughness & Metallic:

Import weights from A2, re-init first conv (kaiming normal on conv.weight)

|  Phase | Dataset   | Encoder init           | Trainables        | **Crop / Feed (px)** | Epochs | Optimiser & LR                                | Scheduler              | Core losses |
| -----: | --------- | ---------------------- | ----------------- | -------------------- | -----: | --------------------------------------------- | ---------------------- | ----------- |
| **M0** | 100 % Sky | best A2 (strict False) | enc + dec + heads | **768**              |      6 | AdamW: enc 5e‑5 (LLRD 0.9^d) · dec/heads 2e‑4 | cosine‑6,eta_min=5e-6  | See table   |
| **M1** | 100 % Sky | from M0                | enc + dec + heads | **1 024**            |     12 | AdamW: enc 1e‑5 (LLRD) · dec/heads 4e‑5       | cosine‑12,eta_min=1e-6 | same        |

## 3.2 Height & AO

Start from scratch

| Phase  | Crop | Epochs | Enc LR            | Dec LR | Optimizer | Scheduler                                                                          |
| ------ | ---- | ------ | ----------------- | ------ | --------- | ---------------------------------------------------------------------------------- |
| **P0** | 256  | 8      | 1e‑4              | 2e‑4   | AdamW     | warmup linearLR 1 epoch, start_lr=0.3, cosine t_max=epochs-1,eta_min=`enc_lr*0.05` |
| **P1** | 512  | 8      | 1e‑4 (LLRD 0.8^d) | 2e‑4   | AdamW     | same                                                                               |
| **P2** | 768  | 14     | 8e‑5 (LLDR 0.8^d) | 1.6e‑4 | AdamW     | same                                                                               |
| **P3** | 1024 | 5      | frozen            | 1.0e‑4 | AdamW     | same                                                                               |

## 3.4 Unet-Maps input

| Map       | Input                                                               |
| --------- | ------------------------------------------------------------------- |
| height    | normal + mean curvature + poisson-coarse                            |
| ao        | normal + mean curvature + poisson-coarse                            |
| roughness | albedo + normal + segformer mask (K channels) (both channel & FiLM) |
| metallic  | albedo + segformer mask (K channels) (both as channel and FiLM)     |

## 3.5 Per‑map network & loss recipes

| map        | loss = _λᵢ·termᵢ_                                                                                              |
| ---------- | -------------------------------------------------------------------------------------------------------------- |
| **Height** | `1.0·L1 + 0.25·GradDiff + 0.06·TV + 0.15->0.10(decay in P2)·Normal‑Reproj + 0.06·MS‑SSIM + 0.1*Laplacian-Pyr ` |
| **AO**     | `1.0·L1 + 0.15·Sobel + 0.1·MS‑SSIM`                                                                            |
| **Rough**  | `1.0·Focal-Relative-L1 + 0.1·MS‑SSIM + 0.02·Sobel`                                                             |
| **Metal**  | `1.0*Focal BCE(a=0.25,g=2.0) + 0.7*Focal-Tversky(a=0.7,b=0.3,g=1.5) + 0.05*Sobel + 0.05*L1`                    |

## 4. Composite‑mosaic rules

| Feed ≤ (px) | Mosaic active? | Grid        | Share (2 crop / 4 crop) |
| ----------- | -------------- | ----------- | ----------------------- |
| 256         | yes            | 2 × 2 (128) | 30 % / 15 %             |
| 512         | yes            | 2 × 2 (256) | 25 % / 10 %             |
| ≥ 768       | **no**         | —           | 0 %                     |

### Metallic scarcity fixes

```python
# before M-pre
p = 0.06                 # prior metal pixel ratio
b0 = math.log(p / (1 - p))
model.head_metal[0].bias.data.fill_(b0)          # 1×1 conv bias
bce = torch.nn.BCEWithLogitsLoss(pos_weight = neg/pos)
```

## 5. Augmentation key

-   Global safe – h/v flip, 90° rot (all phases)
-   Colour-jitter – ±5 % hue/sat (MatSynth only)
-   Composite mosaics – MatSynth only, % per table
-   SkyPhotometric(p=0.6) – light tint, γ, grain; p = 0.5 in hi-res stages

## 6. Implementation notes

```python
# ❶  Weight-transfer Maps ⇐ Albedo
maps.unet.load_state_dict(albedo.unet.state_dict(), strict=False)

# ❷  LLRD parameter groups (encoder only)
for i, blk in enumerate(maps.unet.encoder):
    lr = base_enc_lr * (0.8 ** (len(maps.unet.encoder) - i - 1))
    param_groups.append({"params": blk.parameters(), "lr": lr, "weight_decay": 1e-2})

# ❸  Detach albedo when feeding Maps
alb = albedo(diffuse_normal, segfeat).detach()
maps_in = torch.cat([alb, normal], 1)
out = maps(maps_in, segfeat)

```

## 7. MatSynth category hygiene

| Category                                                 | Action & Reason                                                                          |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **plastic**                                              | **Drop** – anachronistic.                                                                |
| **concrete**                                             | **Merge → stone** – roughness/height similar; makes SegFormer’s job easier.              |
| **marble**                                               | If Skyrim mod pack has no marble, **merge into stone**; else keep (rare indoor pillars). |
| **plaster**                                              | **Drop**                                                                                 |
| **terracotta**                                           | Very rare → **drop**.                                                                    |
| **misc**                                                 | Contains heterogeneous, often modern designs → **drop**.                                 |
| **ceramic, fabric, ground, leather, metal, wood, stone** | **Keep**. Add `fur` if you have ≥ 100 samples.                                           |

## 8. Texture augmentation table

| Category                       | **Safe for ALL domains**<br>(apply blindly) | **Category-Selective**<br>(only if label is known or confidence > 0.8) | **Exclude / Never**                |
| ------------------------------ | ------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------- |
| **wood**                       | flips, 90° rot                              | ±10 % brightness, ±5 % hue, small grain-noise mask                     | heavy tint (green, purple)         |
| **stone**                      | flips, 90° rot                              | ±8 % brightness, Perlin dirt overlay                                   | hue shift (changes mineral colour) |
| **metal**                      | flips, 90° rot                              | subtle specular highlight sprite (white blotch α=0.15)                 | hue shift (turns iron blue)        |
| **fabric / fur**               | flips, 90° rot                              | ±12 % hue/ sat, small warp (elastic-grid)                              | specular sprite                    |
| **leather**                    | flips, 90° rot                              | ±8 % hue, ±12 % brightness                                             | specular sprite                    |
| **ground / ceramic / plaster** | flips, 90° rot                              | ±10 % brightness, Perlin dirt                                          | hue shift > 5 %                    |
| **misc (dropped)**             | —                                           | —                                                                      | —                                  |
