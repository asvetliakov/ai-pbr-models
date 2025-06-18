# PBR Texture Conversion Plan

**Task objective**:
Convert **old Skyrim SE (diffuse + normal) textures** to a modern PBR set (**albedo, roughness, metallic, AO, height**) using custom-trained AI models, leveraging the **MatSynth dataset** (high-quality PBR) and Skyrim SE-specific texture samples.

**Models used**:

-   **ViT classifier** for material type
-   **SegFormer** for metallic masks
-   **Custom UNet-Albedo**: diffuse+normal → albedo
-   **Custom UNet-Maps**: albedo+normal → roughness/metallic/AO/height

---

## 0️⃣ **Dataset Pre‑flight (one‑time)**

| Step                       | Script                    | Purpose                                                                                                                  |
| -------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Duplicate pair removal** | `find_identical_pairs.py` | Move _diffuse ≡ albedo_ pairs to `duplicates/` (they will be used later only for auto‑encoder self‑reconstruction).      |
| **Black‑diffuse audit**    | `verify_black_diffuse.py` | Detect genuine “no‑diffuse” materials vs. conversion mistakes. Fix bad PNGs; treat “no‑diffuse” cases as _unsupervised_. |

---

**NOTE (AI IGNORE)** If decided that Skyrim PBR conversion is not a good quality for training then reduce Skyrim SE % for Phase B to 15% and in Phase C to 30%. Lower also LR in phase D to 5e-7 (still train on 100% skyrim pbr)

## 🪜 Phase Overview

| Phase  | Data Mix                               | Res.       | Purpose                                       | Trainable                                         | New Rules / Gates                                       |
| ------ | -------------------------------------- | ---------- | --------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------- |
| **A0** | 100% MatSynth                          | 1K (1024²) | Sanity check & baseline training              | Full Fine-tune                                    |                                                         |
| **A**  | 100% MatSynth                          | 1K         | Solidify clean PBR priors                     | Full Fine-tune                                    |
| **B**  | 75% MatSynth + 25% Skyrim SE           | 1K         | Initial Skyrim domain-adaptation (heads only) | Heads + LoRA (Freeze/LoRA)                        | **enable “SkyrimPhotometric” augment**                  |
| **C**  | 50% MatSynth + 50% Skyrim SE           | 1K         | Deeper domain adaptation                      | heads + top ½ enc. + LoRA (Partial Unfreeze/LoRA) | switch **AdamW β₂ = 0.9995** (or **Lion** if available) |
| **C′** | 50% MatSynth + 50% Skyrim SE           | 1 K        | Re‑warm‑up BN/LN stats                        | BN/LN only                                        | 1‑2 epochs; prepares stats for 2 K jump.                |
| **D**  | 100% Skyrim SE (**per-material jobs**) | 2K (2048²) | High-res detail refinement                    | isolated heads                                    | freeze backbones **Adam β₂ = 0.9995**; LR = 1e‑6.       |

### 🔒 **Validation Gate (applied before advancing):**

Advance _only if_ for the _last 3 epochs_

1. **Skyrim‑val loss** ↓ ≤ 5 % (moving average) **and**
2. **MatSynth‑val loss** ↑ ≤ 10 %.

Early‑stop any phase on plateau ≥ 4 epochs.

---

| Area                               | What You Already Do Well                                             | Suggested Improvements                                                                                                                                                                                                | Rationale                                                                                              |
| ---------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Phase structure & checkpoints**  | Clear, progressive domain‑adaptation schedule with checkpoint reuse. | Add **formal validation gates** between phases. E.g. require: <br>• Skyrim‑val ↓ ≤ 5 % over last three epochs **and**<br>• MatSynth‑val ↑ ≤ 10 % (to control catastrophic forgetting) before advancing.               | Prevents moving on while either domain is deteriorating.                                               |
| **Resolution ramp‑up (1 K → 2 K)** | Correct order (learn first, refine later).                           | Insert a **1–2‑epoch re‑warm‑up** at 1 K with only frozen backbones but _unfrozen batch‑norm/Layer‑Norm stats_.                                                                                                       | Avoids distribution shock when switching tiling strategy & batch size at 2 K.                          |
| **Optimizer & LR**                 | Sensible AdamW / Adam schedule.                                      | In Phases C & D, switch to **AdamW(β₂ = 0.9995)** or **Lion** (if available).                                                                                                                                         | Lower β₂ (or Lion) gives faster adaptation when very few parameters remain trainable.                  |
| **Augmentation**                   | Creative cut‑and‑paste composites.                                   | Add **photometric augmentations that mimic Skyrim lighting**: <br>• slight baked‑in AO tint,<br>• colder white‑balance,<br>• light vignette.<br>Do this _only_ for Skyrim samples to enlarge its appearance manifold. | PBR priors stay intact while Skyrim distribution broadens.                                             |
| **Evaluation metrics**             | L1/L2, SSIM, IoU, Accuracy.                                          | Add **Perceptual (LPIPS)** on generated maps and a **material similarity metric** (e.g. cosine Δ in 3D BRDF space rendered at random view/light dir).                                                                 | Subjective visual quality correlates better with LPIPS & BRDF similarity than with pixel losses alone. |
| **Logging**                        | Rich job tags.                                                       | Pipe metrics into **Weights & Biases** or **TensorBoard hparams** plugin so you can correlate domain mix %, freeze depth, LR, etc.                                                                                    | Makes hyper‑parameter search transparent and reproducible.                                             |

---

## ✨ **Augmentation Matrix**

| Phase  | Domain             | Spatial (existing)                              | **Photometric (NEW)**                                               |
| ------ | ------------------ | ----------------------------------------------- | ------------------------------------------------------------------- |
| A0 / A | MatSynth           | flips, rot, colour jitter, 2‑/4‑crop composites | —                                                                   |
| B      | MatSynth<br>Skyrim | idem                                            | **Skyrim only**: `SkyrimPhotometric` (AO tint + cold WB + vignette) |
| C / C′ | idem               | idem (composite rates ↓ to 20 % / 10 %)         | Skyrim only (same)                                                  |
| D      | Skyrim             | none (full 2 K)                                 | Skyrim only (same, but _strength × 0.5_)                            |

> **Implementation**: drop‑in PyTorch transform `SkyrimPhotometric` (see `skyrim_photometric_aug.py`).  
> adds _baked‑in AO tint_, _cold WB_, _light vignette_ to **diffuse input only**.  
> Apply _after_ MatSynth/Skyrim mixing so each Skyrim sample may appear pristine **and** distorted within the same epoch.  
> Ground‑truth maps are **never** photometrically altered.

---

## 🏗️ **Model‑Specific Updates**

| Component            | Change                                                                                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ViT classifier**   | Switch to **multi‑label** mode – use `BCEWithLogitsLoss` (`multilabel_bce_loss.py`).                                                               |
| **UNet‑Maps**        | Add **FiLM conditioning** with per‑pixel material logits (`film_conditioning.py`).                                                                 |
| **Curriculum crops** | In B → C schedule crop size: 256 → 768 px using `compute_crop_size()` to start with mostly homogeneous patches then expose mixed‑material context. |
| **Losses**           | Replace plain L1/L2 with **masked‑L1** (`masked_l1.py`) weighted by material relevance; add **LPIPS** and **BRDF‑cosine** to validation.           |
| **Metrics logging**  | Pipe to **Weights & Biases**; log: phase tag, mix %, freeze depth, LR, optimiser.                                                                  |

---

## ⚙️ **Optimiser & LR Schedules (summary) - updated**

| Phase | Optimiser                       | Base LR     | Scheduler             |
| ----- | ------------------------------- | ----------- | --------------------- |
| A0    | AdamW β₂ = 0.999                | 1e‑4        | Cosine (Tₘₐₓ = 15)    |
| A     | AdamW β₂ = 0.999                | 5e‑5 → 1e‑5 | One‑Cycle             |
| B     | AdamW β₂ = 0.999                | 1e‑5        | Step (6, 0.5)         |
| C     | **AdamW β₂ = 0.9995** (or Lion) | 5e‑6        | Cosine (Tₘₐₓ = 12)    |
| C′    | same as C                       | 3e‑6        | cosine warm‑restart   |
| D     | **Adam β₂ = 0.9995**            | 1e‑6        | Exponential (γ = 0.9) |

---

## 1️⃣ Phase A0 — Quick Sanity-Check (1K)

-   **Purpose**: Verify pipeline, models, losses on minimal data.
-   **Data**: Sample ~1 000 MatSynth patches at 1024×1024
-   **Epochs**: 10–15
-   **Optimizer**: AdamW, weight decay 1e-2
-   **LR**: 1×10⁻⁴
-   **Scheduler**: CosineAnnealingLR (T_max=15)
-   **What to train**:
    -   ViT classifier
    -   SegFormer (metallic mask)
    -   UNet-Albedo
    -   UNet-Maps (roughness/metallic/AO/height)
-   **Augmentation**: None yet—focus on data pipeline sanity.
-   **Freeze/LoRA**: None
-   **Notes**:
    -   Validate end-to-end pipeline: data loading, augmentations, definitions, losses.
    -   Expect rapid loss decrease; fix mismatches or errors now.
-   **AI Notes**:
    -   “A0 context: confirming that datasets and models are wired correctly at 1K.”
    -   “No domain or style adaptation—pure functionality check.”
    -   “Use logs of A0 to decide hyper-param adjustments for Phase A.”

---

## 2️⃣ Phase A — Full MatSynth Pre-training (1K)

-   **Purpose**: Train models fully on MatSynth to learn accurate PBR priors.
-   **Init**: **Resume** from A0 checkpoint
-   **Data**: Full MatSynth @1024²
-   **Epochs**: 30–40
-   **Optimizer**: AdamW, weight decay 1e-2
-   **LR**: start 5×10⁻⁵ → decay to 1×10⁻⁵
-   **Scheduler**: OneCycleLR (max_lr=5e-5, pct_start=0.3)
-   **What to train**:
    -   All models end-to-end (UNets from scratch; ViT/SegFormer warm-started).
-   **Augmentation**: Standard MatSynth augmentations (random flips, rotations, color jitter).
-   **Random Mix/Combine** (add to dataset, do not replace originals):
    -   **2-crop composites**: Take two random 512×512 MatSynth crops and tile side-by-side or top-bottom within a 1024×1024 canvas. Apply to ~30% of training samples.
    -   **4-crop composites**: Take four random 256×256 MatSynth crops and arrange in a 2×2 grid. Apply to ~15% of training samples.
-   **Freeze/LoRA**: None
-   **Notes**:
    -   Learn robust, physics-based PBR priors.
    -   Monitor MatSynth-val: UNet L1/L2 & SSIM, SegFormer IoU, ViT accuracy.
-   **AI Notes**:
    -   “A context: model is learning ‘ideal’ PBR behavior on clean, varied data.”
    -   “No domain mixing yet—focus on generic material understanding.”
    -   “Checkpoint at end of A will serve as base for all adaptation.”

---

## 3️⃣ Phase B — Head-Only Domain Adaptation (1K)

-   **Purpose**: Begin adapting models to Skyrim SE domain with minimal forgetting.
-   **Data**: 75% MatSynth + 25% Skyrim SE @1024²
-   **Epochs**: 8–12
-   **Optimizer**: AdamW, weight decay 1e-2
-   **LR**: 1×10⁻⁵
-   **Scheduler**: StepLR (step_size=6, gamma=0.5)
-   **What to train**:
    -   **Freeze** encoders/backbones (ViT blocks, SegFormer encoder, UNet encoders).
    -   **Train** decoder heads for all models.
    -   **Use LoRA** adapters on ViT & SegFormer last layers.
-   **Augmentation & Mix**:
    -   Continue random composites from Phase A (2-crop, ~30%; 4-crop, ~15%).
    -   Keep original samples as ~60% of each batch.
-   **Curriculum crop:** start 256 px.
-   **Notes**:
    -   Introduce Skyrim style cues while preserving MatSynth priors.
    -   Keep batch mixing random each epoch.
-   **AI Notes**:
    -   “B context: small-scale domain shift via adapters and head-tuning.”
    -   “Skyrim SE textures now influencing heads, but core features still frozen.”
    -   “Monitor both MatSynth-val and Skyrim-val to catch forgetting.”

---

## 4️⃣ Phase C — Partial Unfreeze Deep Adaptation (1K)

-   **Purpose**: Strengthen domain-specific representation of Skyrim textures.
-   **Data**: 50% MatSynth + 50% Skyrim SE @1024²
-   **Epochs**: 8–12
-   **Optimizer**: AdamW, weight decay 1e-2
-   **LR**: 5×10⁻⁶
-   **Scheduler**: CosineAnnealingLR (T_max=12)
-   **What to train**:
    -   **Unfreeze** top 50% of encoder layers (last ViT blocks; deepest UNet blocks).
    -   **Continue** training decoder heads + existing LoRA adapters.
-   **Augmentation & Mix**:
    -   Maintain composites at reduced rate (2-crop ~20%; 4-crop ~10%).
    -   Originals remain ~70% of each batch.
-   **Curriculum crop:** progress toward 768 px.
-   **Notes**:
    -   Grant more capacity for Skyrim-specific patterns.
    -   Early-stop when mixed-val loss plateaus or overfitting begins.
-   **AI Notes**:
    -   “C context: deeper layers now adjust to domain; lower layers still hold clean priors.”
    -   “Key metrics: convergence on mixed-val gives signal to proceed to high-res.”
    -   “Prepare checkpoint for 2K refinement stage.”

---

## 5️⃣ Phase D — Per-Material 2K Refinement

**Phase D context for AI**:

High-resolution, final fine-tuning. Fully frozen backbones. Separate independent jobs per material head. This ensures maximal Skyrim-specific detail with minimal parameter updates.

1. **Setup Steps**

    - Load Phase C checkpoint
    - Data loader to 2048²; adjust convolutional strides accordingly.
    - Freeze backbones entirely. **train** only designated decoder/upsample heads.
    - Photometric: SkyrimPhotometric (half strength).

2. **Independent Jobs**

| Model                    | Heads to Train                    | Epochs | Optimizer       | LR       | Scheduler                 | Freeze & LoRA Notes                                                              |
| ------------------------ | --------------------------------- | ------ | --------------- | -------- | ------------------------- | -------------------------------------------------------------------------------- |
| **SegFormer**            | Metallic-mask decoder layers      | 5–8    | AdamW (wd=1e-2) | 1 × 10⁻⁶ | StepLR(step=4, gamma=0.5) | Freeze encoder; retain LoRA adapters from B/C; train decoder only.               |
| **UNet → Albedo**        | Final upsampling & output blocks  | 6–10   | Adam            | 1 × 10⁻⁶ | ExponentialLR(gamma=0.9)  | Freeze encoder & early decoder; train final upsampling layers; no LoRA.          |
| **UNet → R/M/AO/Height** | Each map-head separately (4 jobs) | 6–10   | Adam            | 1 × 10⁻⁶ | ExponentialLR(gamma=0.9)  | Freeze shared trunk; train one map-head per run; no LoRA.                        |
| **ViT Classifier**       | — (skip 2K)                       | —      | —               | —        | —                         | Classification is resolution-agnostic—retain best 1K weights; no 2K pass needed. |

-   **AI Notes**:
    -   “D context: isolate high-frequency detail learning to minimal decoder parameters at 2K.”
    -   “Each head’s job name should include `phase=D`, `res=2K`, `model=<head>`, for easy recall.”
    -   “Backbones remain frozen to preserve combined MatSynth + Skyrim priors.”

---

## 🔧 When to Use LoRA vs. Freeze

-   **ViT & SegFormer**

    -   **Phases B/C**: freeze backbone; insert & train LoRA adapters + heads.
    -   **Phase D**: freeze backbone; do not add new adapters; train only heads.

-   **Custom UNets**
    -   **Phases B/C**: freeze encoder; train decoder heads (no LoRA).
    -   **Phase D**: freeze trunk; train only final upsampling/output heads (no LoRA).

**AI Notes**:

-   “Adapters only on transformers; UNets use classic freeze-and-tune.”
-   “LoRA modules persist from B/C into D but aren’t expanded further.”

---

# **✅ General Training & AI Context Tips**:

-   Always resume training checkpoints when possible.
-   Clearly tag all jobs: phase, res, model, mix, aug.
-   Augmentation: add composites, don't replace originals.
-   Monitor separate validation metrics: MatSynth-val vs Skyrim-val.
-   Early-stop immediately upon plateauing validation performance.

## AI Context Note:

-   This detailed tagging/logging facilitates accurate future AI consultation.

## 🎯 Final AI Contextualization

_This complete document is structured explicitly to enable future AI-based troubleshooting and guidance. All phases, motivations, and choices are explicitly clarified to provide stable context whenever future questions arise._
