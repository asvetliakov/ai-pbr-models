# 🌄 End‑to‑End PBR Conversion Plan — _No‑ViT, Multi‑Mask SegFormer_

_(Version 4.5 · 18 Jun 2025)_

---

## MatSynth category hygiene

| Category                                                 | Action & Reason                                                                          |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **plastic**                                              | **Drop** – anachronistic.                                                                |
| **concrete**                                             | **Merge → stone** – roughness/height similar; makes SegFormer’s job easier.              |
| **marble**                                               | If Skyrim mod pack has no marble, **merge into stone**; else keep (rare indoor pillars). |
| **plaster**                                              | **Drop**                                                                                 |
| **terracotta**                                           | Very rare → **drop**.                                                                    |
| **misc**                                                 | Contains heterogeneous, often modern designs → **drop**.                                 |
| **ceramic, fabric, ground, leather, metal, wood, stone** | **Keep**. Add `fur` if you have ≥ 100 samples.                                           |

---

## 🪜 Phase Ladder — with goals, metrics & advice

| Phase  | Goal (one‑liner)                                      | Dataset Mix    | Trainable Modules                        | Crop     | Augment (new → old)                                |
| ------ | ----------------------------------------------------- | -------------- | ---------------------------------------- | -------- | -------------------------------------------------- |
| **A0** | Smoke‑test pipeline; get first masks & maps.          | 100 % MatSynth | SegFormer, UNet‑Albedo, UNet‑Maps (all)  | 256 px   | none                                               |
| **A**  | Learn clean PBR priors on single‑material textures.   | 100 % MatSynth | same                                     | 256 px   | flips · rot · jitter · composites (SegFormer only) |
| **B**  | Introduce Skyrim; adapt heads with FiLM conditioning. | 75 %/25 %      | SegFormer heads+LoRA; UNet decoder heads | 256→512  | Photometric (Skyrim) · composites (All)            |
| **C**  | Deep adaptation; unfreeze upper encoder layers.       | 50 %/50 %      | top ½ encoders + heads                   | 512→768  | same, lower composite %                            |
| **C′** | Stabilise BN/LN stats for 2 K jump.                   | 50 %/50 %      | BN/LN only                               | 1 K      | none                                               |
| **D**  | High‑res detail for each map at 2 K.                  | 100 % Skyrim   | per‑map head job                         | full 2 K | Photometric × 0.5                                  |

Below, each phase is _self‑contained_.

---

### 1️⃣ Phase A0 — _“Does it run?”_

| Item                       | Setting                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Description**            | 10‑epoch sprint: ensure dataloader, losses, GPU work.                                                    |
| **Optimizer**              | `AdamW(lr=1e‑4, wd=1e‑2)`                                                                                |
| **Scheduler**              | `CosineAnnealingLR(T_max=10)`                                                                            |
| **Augment**                | none                                                                                                     |
| **SegFormer ground‑truth** | _whole image mask_ = class id (easy single‑material)                                                     |
| **Losses**                 | SegFormer → `CrossEntropy2d` (mask = class id) <br>UNet‑Albedo → `maksed L1` <br>UNet‑Maps → `masked L1` |
| **Log / Watch**            | `mat_val_loss` (expect sharp ↓), `seg_iou` (≥ 0.55), `u_alb_L1`                                          |
| **Recommendations**        | If any loss is `nan`, fix data normalisation before advancing.                                           |

---

### 2️⃣ Phase A — _Clean MatSynth Learning_

| Item                   | Setting                                                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Epochs**             | 35                                                                                                                                 |
| **Description**        | 35 epochs to build strong physics priors. Composites automatically create multi‑material masks for SegFormer.                      |
| **Optimizer**          | `AdamW(lr 5e‑5→1e‑5)`                                                                                                              |
| **Scheduler**          | `OneCycleLR(max_lr=5e‑5, pct_start=0.15)`                                                                                          |
| **Augment**            | flips, 90° rot, colour‑jitter; **composites (SegFormer only)**<br>  • 2‑crop 30 %<br>  • 4‑crop 15 %                               |
| **SegFormer GT**       | composites know patch coordinates ⇒ auto mask                                                                                      |
| **Curriculum crop**    | fixed 256 px                                                                                                                       |
| **Losses**             | same as A0 plus, `UNet-albedo`: `masked L1 + 0.1 * SSIM + 0.05 * LPIPS` (both training & validation), `Unet-maps`: see table below |
| **Additional Metrics** | LPIPS on albedo; IoU on SegFormer per‑class                                                                                        |
| **Gate**               | proceed when losses plateau < 4 epochs                                                                                             |
| **Advice**             | Over‑fitting shows as LPIPS ↑ while L1 ↓. Stop early if that happens.                                                              |

### Unet-maps loss table

| Map           | Range        | Loss terms                                                                          |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| **Roughness** | 0-1          | `masked_L1` + **0.05 × SSIM**                                                       |
| **Metallic**  | 0-1          | Full-image BCE (From Metallic mask GT) + pos_weight = #neg / #pos. No explicit mask |
| **AO**        | 0-1          | `masked_L1`                                                                         |
| **Height**    | unrestricted | `masked_L1` + **0.01 × Grad-penalty** (encourage smoothness)                        |

_Validation loss fro UNet-maps_
Same losses but WITHOUT gradient penalty (TV) to save time

_Masked L1 examles:_

| Model-head                               | What is **foreground** (`material_mask==1`)                                                                                                                                                                        | When to apply                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| **UNet-Albedo**                          | All _valid_ pixels (because the whole albedo image is relevant) → just pass `material_mask = torch.ones_like(pred[:, :1])`. Effectively your `masked_l1` simplifies to plain L1 but keeps the same call-signature. | Phases A → D                                                  |
| **UNet-Maps** (rough, metal, AO, height) | Pixels **belonging to the map’s material**. Example for metallic head:<br>`material_mask = (segformer_pred == metal_idx)` (upsampled to H×W).                                                                      | Phases B → D (when SegFormer is good enough to provide masks) |

_Metal mask_ per phases
| phase | supervision available | suggested loss |
| --------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **A** (MatSynth 100 %) | _perfect_ GT metallic maps | **Full-image BCE** + `pos_weight = #neg / #pos`. No explicit mask. |
| **B / C** (MatSynth + Skyrim, masks from SegFormer) | SegFormer masks have some error | Two equally good options: <br>**(i)** Keep full-image BCE + `pos_weight`. The BCEloss is hardy to a few wrong pixels.<br>**(ii)** Go back to masked BCE but use a small background weight, e.g. `weight = 0.2 + 0.8*mask_metal` so negatives still matter a bit. |
| **D** (2 K, masks fixed) | SegFormer very stable | Either strategy works – most people stay with whatever they used in C. |

```python
# Albedo
loss_alb = masked_l1(alb_pred, alb_gt,
                     material_mask = torch.ones_like(alb_pred[:, :1]))

# Metallic map (example)
metal_mask = (seg_pred == metal_idx).unsqueeze(1)          # (B,1,H,W) bool
loss_met = masked_l1(metal_pred, metal_gt, metal_mask)

```

_Loss examples:_

```python
# unet-albedo
l1      = masked_l1(pred_alb, gt_alb, torch.ones_like(gt_alb[:, :1]))
ssim    = 1 - pytorch_msssim.ssim(pred_alb, gt_alb, data_range=1.0)
lpips_v = lpips_fn(pred_alb, gt_alb)          # e.g. LPIPS-Alex, avg over batch
loss_alb = l1 + 0.10*ssim + 0.05*lpips_v

# Unet-maps
# Roughness (foreground = any non-empty mask)
rough_mask = (gt_metal < 0.5) | (gt_metal >= 0.5)  # simply ones
loss_rough = masked_l1(pred_rough, gt_rough, rough_mask) \
           + 0.05 * (1 - ssim(pred_rough, gt_rough))

# Metallic (only evaluate where material truly metal)
# loss_metal = F.binary_cross_entropy(pred_metal, gt_metal,
#                                     weight = metal_mask.float())
# count positive / negative pixels in this mini-batch
pos = metallic_gt.sum()
neg = metallic_gt.numel() - pos
pos_weight = neg.float() / pos.clamp(min=1.)   # scalar

loss_metal = F.binary_cross_entropy_with_logits(
                 metallic_pred, metallic_gt,
                 pos_weight = pos_weight)   # reduction='mean'


# AO
loss_ao = masked_l1(pred_ao, gt_ao, torch.ones_like(gt_ao))

# Height (add TV penalty)
tv = torch.mean(torch.abs(pred_h[:, :, :, :-1] - pred_h[:, :, :, 1:])) \
   + torch.mean(torch.abs(pred_h[:, :, :-1, :] - pred_h[:, :, 1:, :]))
loss_h = masked_l1(pred_h, gt_h, torch.ones_like(gt_h)) + 0.01*tv

loss_maps = (loss_rough + loss_metal + loss_ao + loss_h) / 4
```

_Summary_:
| Stage | UNet-Albedo loss | UNet-Maps loss |
| ---------------- | ---------------------------- | ---------------------------------------------- |
| **A0 smoke** | only `masked_L1` (quick) | skip |
| **A (MatSynth)** | `L1 + 0.1·SSIM + 0.05·LPIPS` | 4-map average as above (mask from GT category) |
| **B / C (mix)** | same as A | same, but masks now from **SegFormer preds** |
| **D (2 K)** | _freeze_ ALB; no loss | per-head training, use respective map-loss |

## Where to use GT albedo vs. predicted albedo

| Phase                        | Maps-net **input**                                                          | `detach()`?                                                                                         | Why this is the safest default                                                                                                                                                                                                                       |
| ---------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A0** Smoke-test            | **GT albedo** only                                                          | _irrelevant_ – no grad path through constants                                                       | Keep the pipeline trivial while you-re still shaking out loaders & losses.                                                                                                                                                                           |
| **A** Clean MatSynth (35 ep) | **Warm-up (≈ first 5-10 ep)**: GT albedo →<br>**Main run**: **pred albedo** | **Yes** (keep heads independent)                                                                    | GT warm-up lets Maps net learn the pure mapping.<br>Once Albedo stabilises you expose Maps to realistic noise, but you _don’t_ want its loss to drag Albedo away from its own objective.                                                             |
| **B** Head-only domain-adapt | **pred albedo**                                                             | **Yes**                                                                                             | You’re freezing most backbones; the Maps loss shouldn’t be trying to move Albedo’s frozen layers.                                                                                                                                                    |
| **C** Partial unfreeze       | **pred albedo**                                                             | **Two-step**<br>• first 50 % of epochs: **Yes**<br>• final 3-4 ep: **No** (joint finetune, tiny lr) | Early in C you’re still chasing stability after the unfreeze – keep heads decoupled. Once losses flatten, letting the Maps gradients polish Albedo can give a small boost to global coherence. Use a very low LR (e.g. ×0.25) during the joint pass. |
| **C′** BN/LN warm-up         | **pred albedo**                                                             | **Yes** (or leave Albedo frozen)                                                                    | Goal is just stats refresh; keep optimisation local.                                                                                                                                                                                                 |
| **D** 2 K per-map heads      | **pred albedo**                                                             | _irrelevant_ – Albedo backbone/head is frozen                                                       | At this point Albedo is immutable; detach for clarity but it won’t matter.                                                                                                                                                                           |

---

### 4️⃣ Phase B — _Head‑Level Domain Adapt_

| Item                      | Setting                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------ |
| **Dataset mix**           | 75 % MatSynth · 25 % Skyrim                                                          |
| **Trainable**             | SegFormer heads + LoRA; UNet decoder heads; enable **FiLM conditioning**             |
| **Epochs**                | 10                                                                                   |
| **Curriculum crop**       | 256 px → 512 px (linear each epoch)                                                  |
| **Augment (Skyrim only)** | `SkyrimPhotometric(p=0.6)`                                                           |
| **Pseudo‑label trick**    | For Skyrim pixels where SegFormer `softmax > 0.8`, include them in CE loss.          |
| **Optimizer / Sched**     | `AdamW(lr 1e‑5)`                                                                     |
| **Scheduler**             | `StepLR(6, γ=0.5)`                                                                   |
| **Losses**                | SegFormer: CE on MatSynth + **masked CE** on Skyrim;<br>UNet‑Albedo/Maps: same as A. |
| **Metrics**               | `sky_val_loss`, `mat_val_loss`, `seg_iou_sky`, `seg_iou_mat`                         |
| **Gate rule**             | last‑3 epochs:  `sky_val_loss ≤ 0.95*prev` & `mat_val_loss ≤ 1.10*prev`              |
| **Personal advice**       | Watch SegFormer IoU on _metal_ channel; should climb from ~0.30 → 0.55+.             |

---

### 5️⃣ Phase C — _Partial Unfreeze Deep Adapt_

| Item                  | Setting                                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------- |
| **Goal**              | Let upper encoders specialise to Skyrim while retaining MatSynth priors.                      |
| **Dataset mix**       | 50 % / 50 %                                                                                   |
| **Trainable**         | top 50 % encoders + heads; LoRA still active. keep FiLM                                       |
| **Epochs**            | 10                                                                                            |
| **Optimizer / Sched** | `AdamW(lr 5e‑6, β₂=0.9995)`                                                                   |
| **Scheduler**         | `CosineAnnealingLR(T_max=12)`                                                                 |
| **Crop schedule**     | 512 px → 768 px                                                                               |
| **Augment**           | composites rate ↓ (2‑crop 20 %, 4‑crop 10 %); Photometric unchanged                           |
| **Metrics**           | same + per‑class SegFormer IoU; monitor catastrophic forgetting (stone/wood should not dive). |
| **Advice**            | If MatSynth IoU falls >10 % from Phase A, reduce unfreeze depth to 25 %.                      |

---

### 6️⃣ Phase C′ — _BN/LN Stats Warm‑up_

| Item          | Setting                                                          |
| ------------- | ---------------------------------------------------------------- |
| **Purpose**   | Correct running mean/var for up‑coming 2 K strides.              |
| **Trainable** | **only** BN/LN affine params.                                    |
| **Epochs**    | 2                                                                |
| **Optimizer** | `AdamW(lr 3e‑6)`                                                 |
| **Scheduler** | cosine warm‑restart                                              |
| **Crop**      | full 1 K                                                         |
| **Metrics**   | ensure val losses don’t spike; if they do, increase to 4 epochs. |
| **Advice**    | Disable dropout during this phase.                               |

---

### 7️⃣ Phase D — _2 K Per‑Map High‑Detail_

| Common settings | value                            |
| --------------- | -------------------------------- |
| **Dataset**     | 100 % Skyrim @ 2048²             |
| **Photometric** | SkyrimPhotometric strength × 0.5 |
| **Freeze**      | **all backbones**                |
| **Early‑stop**  | patience 3 epochs/head           |

| Job (script)                                | Trainable Layers             | Epochs | Optimizer                  | Scheduler       | Metrics to watch       |
| ------------------------------------------- | ---------------------------- | ------ | -------------------------- | --------------- | ---------------------- |
| `seg_D` (all masks)                         | SegFormer decoder head       | 6      | `AdamW(lr 1e‑6, wd 1e‑2)`  | `StepLR(4,0.5)` | IoU_metal ↑, IoU_avg ↑ |
| `u_alb_D`                                   | final upsample / output conv | 8      | `Adam(lr 1e‑6, β₂=0.9995)` | `ExpLR(γ=0.9)`  | LPIPS_sky ↓ (< 0.18)   |
| `u_maps_D_<map>` (rough, metal, ao, height) | map‑specific output conv     | 5‑8    | same                       | same            | masked*L1*<map> ↓      |

> **Recommendation:** run `seg_D` first so the freshest logits feed UNet‑Maps jobs.

---

## 🤖 Personal Recommendations

-   GPU memory – keep per‑map Phase D jobs under 12 GB by --channels_last and torch.compile (PyTorch 2.1).
-   Synthetic diffuse – after Phase B you may lower its sampling weight to 0.3 to avoid over‑regularising shadows.
-   Fur class (optional) – if added, monitor IoU_fur; it’s usually the hardest.
-   SegFormer depth – tiny B1 model is enough (≈ 25 M params). Large models slow Phase D.

---

## General training tips

1. Training order per phase: SegFormer, UNet-Albedo, UNet-Maps

-   UNet-Maps uses SegFormer best checkpoint from this Phase

2. In Phase D freeze everything and leave only specific head (roughness, metallic, etc...) then train in separate runs per specific head

-   Each head continue from the best checkpoint from previous head, e.g. train roughness head first -> metallic loads results from roughness train and continue with metallic head, etc

## Sampling

| Tactic                                      | What you do                                                                                              | Pros                                                | Cons                                                                                  | When to use                                                                                  |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **1. Per-class loss weights** _(easy)_      | Multiply the **SegFormer CE loss** and **masked-L1 losses** by `1 / √freq(class)` (or focal-like power). | One-liner; no change to sampling.                   | Large classes still dominate the **features learned** in early epochs.                | Always do this as a baseline.                                                                |
| **2. Balanced sampler**                     | Use `torch.utils.data.WeightedRandomSampler` so every mini-batch has \~uniform class distribution.       | Equal gradient signal per class; simple to plug in. | Over-sampling a rare class shows the _same_ images more often → slight over-fit risk. | When minority class < ½ of majority (e.g. ground 260 vs metal 800).                          |
| **3. Synthetic augmentation of minorities** | Apply extra augmentations **only** to minority textures (e.g. hue shift, cut-mix two ground variants).   | Extra diversity avoids over-fit in 2.               | More code; careful not to change semantics (e.g. don’t tint ground purple).           | If you have < 200 unique textures in a class.                                                |
| **4. Class merging or dropping**            | Fold extremely small or irrelevant classes into a parent (you already merged concrete→stone).            | Simplifies head; removes imbalance completely.      | Loses fine category detail.                                                           | Only when class < 50 images **and** not visually distinct (you’ve done most merges already). |

### Minimal pytorch code for 1 + 2

```python
# dataset_build.py
import json, torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# ---------- 1) compute class frequencies ----------
json_labels = json.load(open("splits/MatSynth_train.json"))
all_labels  = [item["class_id"] for item in json_labels]  # list[int]

num_classes = len(set(all_labels))
cls_counts  = torch.bincount(torch.tensor(all_labels), minlength=num_classes)
freq        = cls_counts.float() / cls_counts.sum()

# ---------- loss weights (tactic 1) ----------
loss_weights = 1.0 / torch.sqrt(freq + 1e-6)
seg_loss_fn  = torch.nn.CrossEntropyLoss(weight=loss_weights, ignore_index=255)

# ---------- sampler weights (tactic 2) ----------
inv_freq   = 1.0 / (cls_counts[all_labels].float())  # per-sample weight
sampler    = WeightedRandomSampler(inv_freq, num_samples=len(inv_freq), replacement=True)

train_loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=4)
```

_Explanation_

-   weight ∝ 1/√freq for loss keeps gradients stable without exploding rare classes.
-   Weighted sampler picks minority images more often, yielding roughly balanced batches.

| Phase        | Use sampler?                             | Use loss weights?                                                         |
| ------------ | ---------------------------------------- | ------------------------------------------------------------------------- |
| A0           | **No** (keep pipeline minimal)           | **Yes**                                                                   |
| A, A-Alb-Syn | Yes                                      | Yes                                                                       |
| B, C, C′     | Yes _(on combined MatSynth+Skyrim list)_ | Yes                                                                       |
| D            | **No sampler** (100 % Skyrim)            | Keep loss weights – within Skyrim, fur vs ground may still be imbalanced. |

**Tips & gotchas**

-   Always use class-weighted losses.
-   Add a simple WeightedRandomSampler whenever a class is ≤ 50 % of the largest class (ground 260 vs wood 800 qualifies).
-   Log per-class IoU in train_logs. If ground still lags (< 0.4 when others are 0.6+), bump sampler weights by 1/freq (instead of 1/√freq).
-   Freeze sampler after Phase B if the network starts to over-fit (val loss diverges while train loss falls).
-   Synthetic augment for ground – easiest win: add small random gamma (±10 %) or overlay 5 % Perlin-noise dirt masks; do not hue-shift stones or tree bark (looks wrong).
-   Checkpoint names – append bal tag, e.g. seg_B_bal_best.pth, to remind you that a balanced sampler was used.
-   Keep Phase D simple—by then you have only Skyrim, so imbalance is smaller.

## Augment table

---

Phase A0 : none
Phase A : flip, rot, jitter (Mat), composites (Mat)
Phase B : flip, rot, jitter (Mat), composites (Mat), Photometric (Sky)
Phase C : same as B but composites 20 / 10 %
Phase D : none (+Photometric 0.5× Sky)

| Augmentation                               | Why you keep it                                                                                                     | Which phases                                          | Domain                                                                                           |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Horizontal/vertical flip**               | Doubles effective dataset size; harmless for texture orientation.                                                   | A, B, C                                               | MatSynth + Skyrim                                                                                |
| **90 ° rotations**                         | Adds rotational variety; needed because many Skyrim textures tile in both axes.                                     | A, B, C                                               | MatSynth + Skyrim                                                                                |
| **Colour‑jitter** _(±5 % hue/sat)_         | Prevents the network from over‑fitting to a single white‑balance in MatSynth.                                       | A, B, C                                               | **MatSynth only** (Skyrim already gets Photometric).                                             |
| **Composite crops** (2‑ & 4‑patch mosaics) | **Critical**: they are your _only_ source of _pixel‑accurate multi‑material masks_ for SegFormer during Phases A–C. | A (30 % / 15 %)<br>B (30 % / 15 %)<br>C (20 % / 10 %) | MatSynth only. For composite crops take random samples from whole dataset not just current batch |

### 🖼️ How cropping works vs. composite mosaics

| Term                   | What you actually do                                                                                                                                                                   | Where it happens                                                                 |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Random crop (N px)** | Take a **square patch of side =N pixels** from the input texture, then **_resize it back_** to the model’s fixed input size (1 K for Phases A–C, or 2 K in Phase D).                   | Every phase that lists a crop size (256, 512, 768).                              |
| **Composite mosaic**   | Tile **2 or 4 independent _already-cropped_ patches** side-by-side to fill a 1 K canvas. Each sub-patch keeps its own ground-truth mask → you get an automatic _multi-material_ label. | Only on **MatSynth** samples in Phases A, B, C (with the percentages we listed). |

So: crop_size = 256 px means “model sees a 1024×1024 image whose content originated from a random 256-pixel window.”

**Why the curriculum crop schedule exists**

-   Small crops first (256 px) – forces SegFormer and UNets to learn fine local patterns (wood grain, stone pores).
-   Larger crops later (512→768 px) – introduce bigger structures (rivets, seams) once the lower-level filters are in place.
-   Full image in Phase D (2 K) – no resizing at all, so you optimize true high-frequency detail.

### Phase-by-Phase cheat sheet

| Phase  | `crop_size`           | Composite mosaics?        | What the network finally receives                                                                                                                       |
| ------ | --------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A0** | **256 px**            | **OFF**                   | 1024² image made from one 256 px crop up-scaled to 1 K.                                                                                                 |
| **A**  | 256 px                | 2-crop 30 % / 4-crop 15 % | • Single-crop images (same as A0).<br>• **30 % of batches**: two 256 px crops side-by-side → still 1024².<br>• **15 %**: four 256 px crops in 2×2 grid. |
| **B**  | 256 → 512 px (linear) | Same composite rates as A | Early epochs: small crops; late epochs: larger crops. Composites constructed from whichever crop size is current.                                       |
| **C**  | 512 → 768 px          | Composites at 20 % / 10 % | Even bigger context + sparser mosaics.                                                                                                                  |
| **C′** | full 1 K              | OFF                       | Pure resizing disabled; each texture scaled to exactly 1024² without cropping.                                                                          |
| **D**  | full 2 K              | OFF                       | Native 2048² textures, no cropping, no mosaics.                                                                                                         |

## Texture augmentation table

| Category                       | **Safe for ALL domains**<br>(apply blindly) | **Category-Selective**<br>(only if label is known or confidence > 0.8) | **Exclude / Never**                |
| ------------------------------ | ------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------- |
| **wood**                       | flips, 90° rot                              | ±10 % brightness, ±5 % hue, small grain-noise mask                     | heavy tint (green, purple)         |
| **stone**                      | flips, 90° rot                              | ±8 % brightness, Perlin dirt overlay                                   | hue shift (changes mineral colour) |
| **metal**                      | flips, 90° rot                              | subtle specular highlight sprite (white blotch α=0.15)                 | hue shift (turns iron blue)        |
| **fabric / fur**               | flips, 90° rot                              | ±12 % hue/ sat, small warp (elastic-grid)                              | specular sprite                    |
| **leather**                    | flips, 90° rot                              | ±8 % hue, ±12 % brightness                                             | specular sprite                    |
| **ground / ceramic / plaster** | flips, 90° rot                              | ±10 % brightness, Perlin dirt                                          | hue shift > 5 %                    |
| **misc (dropped)**             | —                                           | —                                                                      | —                                  |

_Composites are built after augmentation_

### How to integrate

```python
def safe_aug(img):
    # always on
    if random.random() < 0.5: img = TF.hflip(img)
    if random.random() < 0.5: img = TF.vflip(img)
    k = random.randint(0,3); img = img.rotate(90*k)
    return img

def selective_aug(img, cls_name):
    # MatSynth or confidently-labeled Skyrim
    if cls_name in ["wood","fabric","leather"]:
        img = TF.adjust_hue(img, random.uniform(-0.05,0.05))
    if cls_name in ["wood","stone","ground","ceramic"]:
        img = TF.adjust_brightness(img, random.uniform(0.9,1.1))
    if cls_name in ["fabric","fur"] and random.random()<0.3:
        img = elastic_warp(img)            # small cloth stretch
    if cls_name=="metal" and random.random()<0.3:
        img = overlay_highlight(img)
    return img

# MatSynth (labels are known)
img = safe_aug(img)
img = selective_aug(img, cls_name_from_folder)

# Skyrim (labels unknown)
# Before Phase B - apply only safe_aug
# Phase B onward - run SegFormer on the crop first

logits = segformer(img_small)          # (C,H,W)
conf, cls = logits.softmax(0).max(0)
if conf.mean() > 0.8:                  # high-confidence crop
    img = selective_aug(img, CLASSES[cls])
else:
    img = safe_aug(img)

```

---

## LPIPS

| Usage style                                          | Pros                                                                        | Cons                                                           | Recommended spot                                                                          |
| ---------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Validation‑only metric**                           | • Zero extra back‑prop cost.<br>• Simpler to code.                          | • Model optimises purely for L1/L2 → can look overly smooth.   | **All phases** (always log `lpips_val`).                                                  |
| **Small‑weight training term** (e.g. `0.05 × LPIPS`) | • Encourages sharper, perceptually pleasing results (important for albedo). | Adds one extra VGG forward pass per mini‑batch (≈8 ms on 1 K). | **Phase A‑Alb‑Syn only** — that’s where you fight baked lighting and “identity shortcut.” |
