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

| Phase                    | Data mix (train) | **Trainables**                                          | Crop (px) | Augment † (p per sample)           | Epochs | LR & Scheduler                               | Loss                    |
| ------------------------ | ---------------- | ------------------------------------------------------- | --------- | ---------------------------------- | -----: | -------------------------------------------- | ----------------------- |
| **S0 – Warm-up**         | 100 % Sky        | **enc + dec + heads**                                   | 256       | flips, rot90 (1.0), SkyPhoto (0.6) |     30 | OneCycle LR 1e-4 → 4e-4 → 1e-5 (pct 0.15)    | 0.8 Focal CE + 0.2 Dice |
| **S1 – Domain focus**    | 100 % Sky        | **enc (block 0 frozen) + dec + heads** (LLRD 0.9)       | 512       | same                               |     15 | Cosine start 1e-4, ηₘᵢₙ 8e-6                 | same                    |
| **S2 – Hi-res mix**      | 100 % Sky        | **enc (block 0 frozen) + dec + heads** (LLRD 0.9)       | 768       | same                               |     12 | Cosine 8e-5 → ηₘᵢₙ 8e-6                      | same                    |
| **S3 – Full-res polish** | 100 % Sky        | **enc block 3 (0,1,2 frozen) + dec + heads** (LLRD 0.9) | 896       | same                               |      4 | Cosine enc 3e-7, dec/head 3e-6, eta_min=1e-7 | same                    |
| **S4 – Decoder final**   | 100 % Sky        | **dec-head only**                                       | 1024      | fils, rot90                        |      6 | Cosine 5e-6 -> 5e-7                          | same                    |

### 1.1 SegFormer Class‑balancing Strategy

| Layer                                 | Purpose                                                        | Implementation                                                                       |
| ------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Per‑image `WeightedRandomSampler`** | Over‑sample images rich in _minority_ materials.               | Soft exponent + Floor  *i* = `max(frac ** 0.4, 0.3)`<br>normalise so mean ≈ 1.       |
| **Patch‑aware oversampling**          | Guarantee minority pixels in every crop for 256–768 px stages. | 30/20/10% crops drawn from a pre‑built minority‑tile index (freq >= 3% and >= 128px) |
| **Per‑pixel class weights**           | Down‑weight stone/wood inside loss.                            | `w_c = 1 / √freq_c` then normalise                                                   |
| **Adaptive focal γ**                  | Extra penalty on easy majority pixels.                         | See table below                                                                      |
| **Loss mask drop‑out**                | Reduce gradient dominance of majority.                         | Randomly drop 20 % of stone pixels _in loss_ (`keep & rand > p`).                    |

### 1.2 Gamma map

| Class   | Pixel share | Recommended γ | Why                                                                   |
| ------- | ----------- | ------------- | --------------------------------------------------------------------- |
| stone   | **36 %**    | **2.0**       | Dominant & usually easy; strong damping keeps its gradients in check. |
| metal   | 18 %        | **1.5**       | Still frequent but slightly harder (different hues).                  |
| wood    | 17 %        | **1.5**       | Similar to metal in share and difficulty.                             |
| fabric  | 10 %        | **1.2**       | Mid-tier class; moderate damping.                                     |
| ground  | 9 %         | **1.2**       | Same tier as fabric.                                                  |
| leather | 7 %         | **1.0**       | Getting sparse—leave nearly CE-like.                                  |

_From S1_ Apply 0.9 LLRD to encoder blocks.
_Loss mask dropout for majority classes_: 20% in S0/S1, disabled from S2

---

## 2. UNet‑Albedo (A)

| Phase  | Dataset mix | Trainables                  | **Crop / Feed (px)** | Augment†                  | Epochs | Optimiser & LR (per‑group)                            | Scheduler                           | Loss                           |
| ------ | ----------- | --------------------------- | -------------------- | ------------------------- | -----: | ----------------------------------------------------- | ----------------------------------- | ------------------------------ |
| **A1** | 100 % Sky   | full UNet + FiLM            | **256**              | flips · rot, SkyPhoto 0.6 |     45 | AdamW — enc 2e‑4 · dec 2e‑4 · FiLM 3e‑4 · head 2.5e‑4 | OneCycle (pct 0.2, cos, final 1e‑5) | L1 + 0.15 MS-SSIM + 0.05 LPIPS |
| **A2** | 100 % Sky   | **enc + dec + FiLM**        | **512**              | A1 aug + SkyPhoto 0.6     |     14 | AdamW — enc 5e‑5 · dec 8e‑5 · FiLM 1e‑4 · head 8e‑5   | cosine‑14, eta_min=5e-6             | same                           |
| **A3** | 100 % Sky   | **enc + dec + FiLM**        | **768**              | A1 aug + SkyPhoto 0.3     |     12 | AdamW — enc 5e‑6 · dec 4e‑5 · FiLM 5e‑5 · head 4e‑5   | cosine‑12, eta_min=3e-6             | same                           |
| **A4** | 100 % Sky   | **dec + head** (enc frozen) | **1 024**            | A1 aug + SkyPhoto 0.15    |      8 | AdamW — dec 1.5e‑5 · head 2e‑5                        | cosine-5, eta_min=3e-6              | same                           |

---

## 3. Separate Unet per map (M)

## 3.1 Training plan

### Height/AO - Start from scratch

| Phase  | Crop | Epochs | Enc LR            | Dec LR | Optimizer | Scheduler                                                                         |
| ------ | ---- | ------ | ----------------- | ------ | --------- | --------------------------------------------------------------------------------- |
| **P0** | 256  | 8      | 1e‑4              | 2e‑4   | AdamW     | warmup linearLR 1 epoch, start_lr=0.3, cosine t_max=epochs-1,eta_min=`enc_lr*0.1` |
| **P1** | 512  | 8      | 1e‑4 (LLRD 0.9^d) | 2e‑4   | AdamW     | same                                                                              |
| **P2** | 768  | 14     | 8e‑5 (LLDR 0.9^d) | 1.6e‑4 | AdamW     | same                                                                              |
| **P3** | 1024 | 5      | frozen            | 1.0e‑4 | AdamW     | same                                                                              |

### Roughness/Metallic - Import weights from A4

| Phase  | Crop | Epochs | Enc LR            | Dec LR | Optimizer | Scheduler                                                                         |
| ------ | ---- | ------ | ----------------- | ------ | --------- | --------------------------------------------------------------------------------- |
| **M0** | 256  | 20     | 2e‑4              | 2e‑4   | AdamW     | warmup linearLR 2 epoch, start_lr=0.3, cosine t_max=epochs-1,eta_min=`enc_lr*0.1` |
| **M1** | 512  | 13     | 1e‑4 (LLRD 0.9^d) | 2e‑4   | AdamW     | same, but warmup 1 epoch                                                          |
| **M2** | 768  | 13     | 8e‑5 (LLDR 0.9^d) | 1.6e‑4 | AdamW     | same                                                                              |
| **M3** | 1024 | 6      | frozen            | 1.0e‑4 | AdamW     | same                                                                              |

## 3.2 Unet-Maps input

| Map       | Input                                                                                         |
| --------- | --------------------------------------------------------------------------------------------- |
| height    | normal + mean curvature + poisson-coarse                                                      |
| ao        | normal + mean curvature + poisson-coarse                                                      |
| roughness | albedo + normal + segformer mask (K channels, gated by confidence) (both channel & FiLM)      |
| metallic  | albedo + normal + segformer mask (K channels, gated by confidence) (both as channel and FiLM) |

## 3.3 Per‑map network & loss recipes

| map        | loss = _λᵢ·termᵢ_                                                                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Height** | `1.0·L1 + 0.25·GradDiff + 0.06·TV + 0.15->0.10(decay in P2)·Normal‑Reproj + 0.06·MS‑SSIM + 0.1*Laplacian-Pyr `                   |
| **AO**     | `1.0·L1 + 0.15·Sobel + 0.1·MS‑SSIM`                                                                                              |
| **Rough**  | `1.0·Focal-Relative-L1 + 0.15·MS‑SSIM + 0.05·Sobel`                                                                              |
| **Metal**  | `1.0*Focal BCE(a=0.25,g=2.0) + 0.3*Focal-Tversky(a=0.7,b=0.3,g=1.5) + 0.05*Sobel + 0.1*L1 + 0.1*MS-SSIM + 0.15*Material-Penalty` |

## 4. Augmentation key

-   Global safe – h/v flip, 90° rot (all phases)
-   SkyPhotometric(p=0.6) – light tint, γ, grain; p = 0.5 in hi-res stages
