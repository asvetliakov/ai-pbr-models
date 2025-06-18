# 🍰 Step‑by‑Step PBR Texture Conversion & Training Plan

_(Edition 3 – 13 Jun 2025)_  
This version is **human‑friendly** (you can follow it even if you have never trained a network before) **and AI‑friendly** (all stages include tags and file names that an assistant can parse).

---

## 🗂 Directory Layout (copy/paste)

project_root/
│
├── data/
│ ├── MatSynth/ # original MatSynth PNGs
│ ├── Skyrim/ # Skyrim source textures
│ ├── duplicates/ # auto‑detected diffuse=albedo pairs
│ └── reports/ # JSON stats dumped by scripts
│
├── scripts/ # helper code (listed later)
├── checkpoints/ # saved after every stage
└── train_logs/ # TensorBoard / Weights & Biases

---

**Synthetic diffuse sampler implementation tip**: create two SubsetRandomSampler objects and wrap them in a WeightedRandomSampler

---

## 🌐 Phase Overview (read this table once)

| Phase             | Mix (MatSynth / Skyrim) | Input Size  | What is _trained_                 | Epochs        | Tag            |
| ----------------- | ----------------------- | ----------- | --------------------------------- | ------------- | -------------- |
| **A0**            | 100 / 0                 | 1 K (1024²) | **all** layers                    | 10–15         | `phase_A0`     |
| **A**             | 100 / 0                 | 1 K         | **all** layers                    | 30–40         | `phase_A`      |
| **A-Albedo-Sync** | 100 / 0                 | 1 K         | **all** layers                    | 15            | `A‑Albedo‑Syn` |
| **B**             | 75 / 25                 | 1 K         | _decoder heads_ + LoRA            | 8–12          | `phase_B`      |
| **C**             | 50 / 50                 | 1 K         | heads + **top ½ encoders** + LoRA | 8–12          | `phase_C`      |
| **C′**            | 50 / 50                 | 1 K         | **BN/LN stats only**              | 1–2           | `phase_Cp`     |
| **D**             | 0 / 100                 | 2 K (2048²) | **one output head** at a time     | 5–10 per head | `phase_D`      |

> **Rule of thumb:** each phase _inherits_ the best checkpoint of the previous phase.

### Gate explanation

| Key            | What it is                                |
| -------------- | ----------------------------------------- |
| `mat_val_loss` | total validation loss on **MatSynth-val** |
| `sky_val_loss` | total validation loss on **Skyrim-val**   |

_Skyrim rule_: `sky_val_loss_e ≤ 0.95 × sky_val_loss_e-2`

_MatSynth rule_: `mat_val_loss_e ≤ 1.10 × mat_val_loss_e-2`

> “↓ ≤ 5 %” means final ≤ 95 % of the value two epochs earlier.

> “↑ ≤ 10 %” means final ≤ 110 % of the value two epochs earlier.

### Passing Example

| Epoch | `sky_val_loss` | `mat_val_loss` |
| ----- | -------------- | -------------- |
| 14    | 0.1000         | 0.0900         |
| 15    | 0.0975         | 0.0910         |
| 16    | 0.0948         | 0.0935         |

Skyrim 0.0948 ≤ 0.95 × 0.1000 (= 0.095) ✔︎
MatSynth 0.0935 ≤ 1.10 × 0.0900 (= 0.099) ✔︎

### Failing example

| Epoch | `sky_val_loss` | `mat_val_loss` |
| ----- | -------------- | -------------- |
| 22    | 0.0830         | 0.0800         |
| 23    | 0.0835         | 0.0860         |
| 24    | 0.0795         | 0.0885         |

Skyrim 0.0795 ≤ 0.95 × 0.0830 (= 0.0789) ✘ (only -4.2 %)
MatSynth 0.0885 ≤ 1.10 × 0.0800 (= 0.088) ✘ (+10.6 %)

script `check_gate.py`

---

## 📑 Detailed, Self‑Contained Stages

### 1️⃣ Phase A0 — _“Does it even run?”_

| Item                 | Setting                                                    |
| -------------------- | ---------------------------------------------------------- |
| **Goal**             | Make sure data loader, models, losses, GPU setup all work. |
| **Dataset**          | `data/MatSynth/` sample of 1 000 random patches.           |
| **Batch size**       | 4 (fits into 8 GB GPU).                                    |
| **Trainable layers** | Everything (ViT, SegFormer, both UNets).                   |
| **Optimizer**        | `AdamW(lr=1e-4, weight_decay=1e-2)`                        |
| **Scheduler**        | `CosineAnnealingLR(T_max=15)`                              |
| **Augmentations**    | None.                                                      |
| **Metrics**          | print losses every 50 it; run `val.py` each epoch.         |
| **Checkpoint**       | `checkpoints/A0_best.pth`                                  |
| **AI‑note tag**      | `phase_A0_checkpoint`                                      |
| **When done**        | Loss curves drop sharply and stabilise → proceed.          |

---

### 2️⃣ Phase A — _Learn clean PBR priors_

| Item              | Setting                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| **Dataset**       | Full MatSynth at 1024².                                                                          |
| **Resume from**   | `A0_best.pth`                                                                                    |
| **Epochs**        | 30–40                                                                                            |
| **Optimizer**     | `AdamW(lr=5e-5 start → 1e-5 end)`                                                                |
| **Scheduler**     | `OneCycleLR(max_lr=5e-5, pct_start=0.3)`                                                         |
| **Augmentations** | _Spatial_ → flips, 90° rotations, colour‑jitter.<br>_Composites_ → 2‑crop (30 %), 4‑crop (15 %). |
| **Metrics**       | L1/L2, SSIM, IoU, Accuracy.                                                                      |
| **Extra**         | TensorBoard or Weights & Biases (`wandb`) logging.                                               |
| **Checkpoint**    | `checkpoints/A_best.pth`                                                                         |
| **Tag**           | `phase_A_checkpoint`                                                                             |
| **Proceed when**  | val losses plateau < 4 epochs.                                                                   |

---

### Phase A‑Albedo‑Syn — enlarge paired set

| Item              | Setting                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| **Dataset**       | MatSynth with synthetic diffuse overwritten by synth_diffuse_v2.py (5 350 pairs at 1 K)          |
| **Resume from**   | `A0_best.pth`                                                                                    |
| **Epochs**        | 15                                                                                               |
| **Optimizer**     | `AdamW(lr=5e‑5 → 1e‑5, weight_decay=1e‑2)`                                                       |
| **Scheduler**     | `OneCycleLR(max_lr=5e-5, pct_start=0.3)`                                                         |
| **Augmentations** | _Spatial_ → flips, 90° rotations, colour‑jitter.<br>_Composites_ → 2‑crop (30 %), 4‑crop (15 %). |
| **Metrics**       | `masked_L1` (w_fg=3) + LPIPS 0.1× (see `lpips_val.py`)                                           |
| **Extra**         | TensorBoard or Weights & Biases (`wandb`) logging.                                               |
| **Checkpoint**    | `checkpoints/A_alb_syn_best.pth`                                                                 |
| **Tag**           | `phase_Aalb_checkpoint`                                                                          |
| **Proceed when**  | LPIPS↓ and L1↓ across validation set.                                                            |

---

### 3️⃣ Phase B — _Head‑only Skyrim adaptation_

| Item                                       | Setting                                                                      |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| **Dataset mix**                            | 75 % MatSynth : 25 % Skyrim at 1 K.                                          |
| **Trainable**                              | **freeze encoders**; train decoder heads + LoRA.                             |
| **Optimizer**                              | `AdamW(lr=1e-5, weight_decay=1e-2)`                                          |
| **Scheduler**                              | `StepLR(step_size=6, gamma=0.5)`                                             |
| **Augmentations**                          | Spatial (same) **+** `SkyrimPhotometric(p=0.6)` **for Skyrim samples only**. |
| **MatSynth Synthetic Pairs Sample Weight** | 1.0 (same as real) – keeps volume high.                                      |
| **Curriculum crop**                        | Start 256 px inside `compute_crop_size.py`.                                  |
| **Extra losses**                           | import `masked_l1.py`; set `w_fg=3.0`.                                       |
| **Metrics**                                | add LPIPS via `lpips_val.py` (example below).                                |
| **Validation gate**                        | _advance only if_ `<Skyrim‑val ↓ 5 % AND MatSynth‑val ↑ 10 %>`.              |
| **Checkpoint**                             | `checkpoints/B_best.pth`                                                     |

---

### 4️⃣ Phase C — _Partial encoder unfreeze_

| Item                                       | Setting                                                                            |
| ------------------------------------------ | ---------------------------------------------------------------------------------- |
| **Dataset mix**                            | 50 % / 50 % at 1 K.                                                                |
| **Trainable**                              | unfreeze **top 50 % of each encoder** + heads + LoRA.                              |
| **Optimizer**                              | `AdamW(lr=5e-6, betas=(0.9, 0.9995))` _(or Lion)_                                  |
| **Scheduler**                              | `CosineAnnealingLR(T_max=12)`                                                      |
| **Augmentations**                          | Spatial; Photometric (Skyrim) unchanged; composites ↓ to 2‑crop 20 %, 4‑crop 10 %. |
| **MatSynth Synthetic Pairs Sample Weight** | 0.5 – synthetic appears every other epoch on average.                              |
| **Curriculum crop**                        | grows toward 768 px.                                                               |
| **Checkpoint**                             | `checkpoints/C_best.pth`                                                           |
| **Tag**                                    | `phase_C_checkpoint`                                                               |

---

### 4️⃣′ Phase C′ — _Re‑warm‑up BN / LN stats_ (optional but recommended)

| Item                                       | Setting                                           |
| ------------------------------------------ | ------------------------------------------------- |
| **Goal**                                   | Stabilise running means/vars for later 2 K jump.  |
| **Trainable**                              | **Only** Batch‑Norm and Layer‑Norm affine params. |
| **Epochs**                                 | 1–2                                               |
| **MatSynth Synthetic Pairs Sample Weight** | 0.25 – very light presence.                       |
| **Optimizer**                              | `AdamW(lr=3e-6)`                                  |
| **Scheduler**                              | small cosine warm‑restart (use PyTorch default).  |
| **Checkpoint**                             | `checkpoints/Cp_best.pth`                         |
| **Tag**                                    | `phase_Cp_checkpoint`                             |

---

### 5️⃣ Phase D — _2 K high‑detail per‑map refinement_

Run **one map head at a time** (6 jobs total: metallic mask, albedo, roughness, metallic map, AO, height).

| Item                                       | Setting                                                                           |
| ------------------------------------------ | --------------------------------------------------------------------------------- |
| **Dataset**                                | 100 % Skyrim at 2048².                                                            |
| **Trainable**                              | **freeze backbone**; train the selected _output head_ and its up‑sampling layers. |
| **Epochs**                                 | 5–10                                                                              |
| **Optimizer**                              | `Adam(lr=1e-6, betas=(0.9, 0.9995))`                                              |
| **Scheduler**                              | `ExponentialLR(gamma=0.9)`                                                        |
| **Augmentations**                          | **no spatial scaling**; `SkyrimPhotometric` with _half_ strength.                 |
| **MatSynth Synthetic Pairs Sample Weight** | 0 or 0.1 – optional; by now real Skyrim pairs dominate.                           |
| **Metrics**                                | same + visual validation on 3‑D viewer if possible.                               |
| **Job name**                               | `phase_D_res2K_<map>.sh` _(important for AI search)_                              |
| **Early‑stop**                             | patience = 3 epochs per head.                                                     |
| **Output checkpoint**                      | `checkpoints/D_<map>_best.pth`                                                    |

---

## 🔬 Supporting Scripts (all placed in `scripts/`)

| File                          | Purpose                                        |
| ----------------------------- | ---------------------------------------------- |
| **skyrim_photometric_aug.py** | implements AO‑tint, cold‑WB, vignette.         |
| **compute_crop_size.py**      | returns crop dims for curriculum learning.     |
| **vi_multilabel_loss.py**     | ViT multi‑label loss helper.                   |
| **film_conditioning.py**      | FiLM block for UNet‑Maps conditioning.         |
| **masked_l1.py**              | weighted L1 loss for material‑relevant pixels. |
| **synth_diffuse_v2.py**       | Generate Synthetic Diffuse from PBR            |
| **lpips_val.py**              | snippet below – adds LPIPS metric.             |
