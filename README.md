# 🏔️ Skyrim PBR Pipeline — Rev 7.0 (19 Jule 2025)

---

## 0. Legend

-   **S–phases** = SegFormer
-   **A–phases** = UNet-Albedo (`UNetAlbedo` in code)
-   **M–phases** = UNet-Maps (`UNetMaps`)

`cond_ch = 512` ⇒ FiLM conditioning **enabled from the first epoch** for both U-Nets.  
Class weights = `1 / √freq(class)`; WeightedRandomSampler active in every phase except the final hi-res stages.

---

## 1. SegFormer (S)

| Phase                    | Data mix (train) | **Trainables**                                    | Crop (px) | Augment † (p per sample)           | Epochs | LR & Scheduler                                | Loss                                      |
| ------------------------ | ---------------- | ------------------------------------------------- | --------- | ---------------------------------- | -----: | --------------------------------------------- | ----------------------------------------- |
| **S0 – Warm-up**         | 100 % Sky        | **enc + dec + heads**                             | 256       | flips, rot90 (1.0), SkyPhoto (0.6) |     30 | OneCycle LR 1e-4 → 4e-4 → 1e-5 (pct 0.15)     | 0.6 CE (w_c) + 0.3 Focal (γ=2) + 0.1 Dice |
| **S1 – Domain focus**    | 100 % Sky        | **enc (block 0 frozen) + dec + heads** (LLRD 0.9) | 512       | same                               |     15 | Cosine start 1e-4, ηₘᵢₙ 8e-6                  | 0.6 CE + 0.25 Focal + 0.15 Dice           |
| **S2 – Hi-res mix**      | 100 % Sky        | **enc + dec + heads** (LLRD 0.9)                  | 768       | same                               |     12 | Cosine 8e-5 → ηₘᵢₙ 8e-6                       | 0.6 CE + 0.2 Focal + 0.2 Dice             |
| **S3 – Full-res polish** | 100 % Sky        | **enc + dec + heads** (LLRD 0.9)                  | 1024      | _none_                             |      6 | Cosine enc 5e-6 → 2e-6, dec/head 1e-5 -> 5e-6 | 0.8 CE + 0.2 Dice                         |
| **S4 – Decoder final**   | 100 % Sky        | **dec-head only**                                 | 1024      | _none_                             |      6 | Cosine 1e-5                                   | 0.9 CE + 0.1 Dice                         |

### 1.1 SegFormer Class‑balancing Strategy

| Layer                                 | Purpose                                                        | Implementation                                                                       |
| ------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Per‑image `WeightedRandomSampler`** | Over‑sample images rich in _minority_ materials.               | Soft exponent + Floor  *i* = `max(frac ** 0.4, 0.3)`<br>normalise so mean ≈ 1.       |
| **Patch‑aware oversampling**          | Guarantee minority pixels in every crop for 256–768 px stages. | 30/20/10% crops drawn from a pre‑built minority‑tile index (freq >= 3% and >= 128px) |
| **Per‑pixel class weights**           | Down‑weight stone/wood inside loss.                            | `w_c = 1 / √freq_c` then normalise                                                   |
| **Adaptive focal γ**                  | Extra penalty on easy majority pixels.                         | See table below                                                                      |
| **Loss mask drop‑out**                | Reduce gradient dominance of majority.                         | Randomly drop 20 % of stone+wood pixels _in loss_ (`keep & rand > p`).               |

### 1.2 Gamma map

| Class   | Pixel share | Recommended γ    | Why                                                                                                 |
| ------- | ----------- | ---------------- | --------------------------------------------------------------------------------------------------- |
| stone   | **36 %**    | **2.0**          | Dominant & usually easy; strong damping keeps its gradients in check.                               |
| metal   | 18 %        | **1.5**          | Still frequent but slightly harder (different hues).                                                |
| wood    | 17 %        | **1.5**          | Similar to metal in share and difficulty.                                                           |
| fabric  | 10 %        | **1.2**          | Mid-tier class; moderate damping.                                                                   |
| ground  | 9 %         | **1.2**          | Same tier as fabric.                                                                                |
| leather | 7 %         | **1.0**          | Getting sparse—leave nearly CE-like.                                                                |
| ceramic | **2 %**     | **0.5** _(or 0)_ | Ultra-minority: keep full CE signal; γ=0.5 still gives focal’s stability without killing gradients. |

_From S1_ Apply 0.9 LLRD to encoder blocks.
_Weighted Sampler_: S0-S2, disable from stage S3
_Loss mask dropout for majority classes_: 20% in S0/S1, disabled from S2

---

## 2. UNet‑Albedo (A)

| Phase  | Dataset mix | Trainables                  | **Crop / Feed (px)** | Augment†                  | Epochs | Optimiser & LR (per‑group)                            | Scheduler                           | Loss                           |
| ------ | ----------- | --------------------------- | -------------------- | ------------------------- | -----: | ----------------------------------------------------- | ----------------------------------- | ------------------------------ |
| **A1** | 100 % Sky   | full UNet + FiLM            | **256**              | flips · rot, SkyPhoto 0.6 |     45 | AdamW — enc 2e‑4 · dec 2e‑4 · FiLM 3e‑4 · head 2.5e‑4 | OneCycle (pct 0.2, cos, final 1e‑5) | L1 + 0.15 MS-SSIM + 0.05 LPIPS |
| **A2** | 100 % Sky   | **enc + dec + FiLM**        | **512**              | A1 aug + SkyPhoto 0.6     |     14 | AdamW — enc 8e‑6 · dec 3e‑5 · FiLM 4e‑5 · head 3e‑5   | cosine‑14, eta_min=5e-6             | same                           |
| **A3** | 100 % Sky   | **enc + dec + FiLM**        | **768**              | A1 aug + SkyPhoto 0.3     |     12 | AdamW — enc 5e‑6 · dec 4e‑5 · FiLM 5e‑5 · head 4e‑5   | cosine‑12, eta_min=3e-6             | same                           |
| **A4** | 100 % Sky   | **dec + head** (enc frozen) | **1 024**            | A1 aug + SkyPhoto 0.15    |      3 | AdamW — dec 2e‑5 · head 3e‑5                          | fixed LR (no scheduler)             | same                           |

_Save the **best A3** checkpoint → encoder donor for Maps._

---

## 3. Separate Unet per map (M)

## 3.1 Roughness & Metallic:

Import weights from A3, re-init first conv (kaiming normal on conv.weight)

|  Phase | Dataset   | Encoder init           | Trainables        | **Crop / Feed (px)** | Epochs | Optimiser & LR                                | Scheduler              | Core losses |
| -----: | --------- | ---------------------- | ----------------- | -------------------- | -----: | --------------------------------------------- | ---------------------- | ----------- |
| **M0** | 100 % Sky | best A4 (strict False) | enc + dec + heads | **768**              |      6 | AdamW: enc 5e‑5 (LLRD 0.9^d) · dec/heads 2e‑4 | cosine‑6,eta_min=5e-6  | See table   |
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
