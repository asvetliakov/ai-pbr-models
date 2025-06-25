# 🏔️ Skyrim PBR Pipeline — Rev 6.0 (26 Jun 2025)

_All hyper-params assume fp16 + `channels_last` on a 24 GB GPU. Adjust batch-size ↔ LR linearly._

---

## 0. Legend

-   **S–phases** = SegFormer
-   **A–phases** = UNet-Albedo (`UNetAlbedo` in code)
-   **M–phases** = UNet-Maps (`UNetMaps`)

`cond_ch = 256` ⇒ FiLM conditioning **enabled from the first epoch** for both U-Nets.  
Class weights = `1 / √freq(class)`; WeightedRandomSampler active in every phase except the final hi-res stages.

---

## 1. SegFormer (S)

| Phase  | Dataset mix         | Trainables        | Crop / Res | Augment†                                         | Epochs | Opt & LR        | Scheduler      | Loss                               |
| ------ | ------------------- | ----------------- | ---------- | ------------------------------------------------ | ------ | --------------- | -------------- | ---------------------------------- |
| **S0** | 100 % MatSynth      | enc + dec         | 256²       | none                                             | 10     | AdamW 1e-4      | cosine-10      | CE                                 |
| **S1** | 100 % MatSynth      | enc + dec         | 256²       | flips · rot · colour · **composite (30 %/15 %)** | 45     | AdamW 5e-5→1e-5 | OneCycle       | CE (+√freq)                        |
| **S2** | 75 % Mat + 25 % Sky | heads + LoRA      | 256→512    | S1 + SkyPhotometric 0.6                          | 10     | AdamW 1e-5      | Step 6×0.5     | CE + masked-CE (Sky softmax > 0.8) |
| **S3** | 50 % / 50 %         | top-½ enc + heads | 512→768    | composite 20 %/10 %                              | 10     | AdamW 5e-6      | cosine-12      | same                               |
| **S4** | 50 % / 50 %         | **BN/LN only**    | full 1 K   | none                                             | 2      | AdamW 3e-6      | cosine-restart | CE                                 |
| **S5** | 100 % Sky           | dec-head          | full 2 K   | SkyPhoto 0.5                                     | 6      | AdamW 1e-6      | Step 4×0.5     | CE                                 |

---

## 2. UNet-Albedo (A)

| Phase          | Dataset mix         | Trainables           | Crop         | Augment†             | Epochs | Opt & LR        | Scheduler | Loss                       |
| -------------- | ------------------- | -------------------- | ------------ | -------------------- | ------ | --------------- | --------- | -------------------------- |
| **A1**         | 100 % MatSynth      | **full UNet + FiLM** | 256²         | flips · rot · colour | 35     | AdamW 5e-5→1e-5 | OneCycle  | L1 + 0.1 SSIM + 0.05 LPIPS |
| **A2**         | 25 % Mat / 75 % Sky | decoder + FiLM       | 512→768      | A1 + SkyPhoto 0.6    | 10     | AdamW 1e-5      | cosine-10 | same                       |
| **A3-default** | 100 % Sky           | **1 × 1 head only**  | **full 2 K** | none                 | 3      | Adam 5e-7       | Exp 0.9   | same                       |

_Save **best A2 checkpoint** → encoder weight donor for Maps._

---

## 3. UNet-Maps (M) — stand-alone model

| Phase                       | Dataset   | **Encoder init**                                         | Trainables                                           | Res / Crop     | Epochs   | Opt & LR                                                  | Scheduler | Core losses                                                  |
| --------------------------- | --------- | -------------------------------------------------------- | ---------------------------------------------------- | -------------- | -------- | --------------------------------------------------------- | --------- | ------------------------------------------------------------ |
| **M-pre** _(cheap warm-up)_ | 100 % Sky | **copy `unet.*` weights from best A2**<br>`strict=False` | enc + dec + heads                                    | 768→1 K (rand) | 6        | AdamW:<br>• enc 2e-5 (LLRD 0.8^depth)<br>• dec/heads 1e-4 | cosine-6  | Rough L1 + .05 SSIM · Metal BCE • AO L1 · Height L1 + .01 TV |
| **M0**                      | 100 % Sky | from M-pre                                               | enc + dec + heads                                    | full 1 K       | 8        | AdamW:<br>• enc 1e-5<br>• dec/heads 5e-5                  | cosine-8  | same                                                         |
| **M1**                      | 100 % Sky | best M0                                                  | **one head at a time** (rough → metal → AO → height) | full 2 K       | 5–7 each | Adam 1e-6                                                 | Exp 0.9   | same (detach Albedo)                                         |

### Metallic scarcity fixes

```python
# before M-pre
p = 0.06                 # prior metal pixel ratio
b0 = math.log(p / (1 - p))
model.head_metal[0].bias.data.fill_(b0)          # 1×1 conv bias
bce = torch.nn.BCEWithLogitsLoss(pos_weight = neg/pos)
```

## 4. Curriculum snapshot

| Block | SegFormer       | Albedo               | Maps                       |
| ----- | --------------- | -------------------- | -------------------------- |
| Early | S0–S1 (256)     | A1 (256)             | —                          |
| Mid   | S2–S3 (512-768) | A2 (512-768)         | M-pre (768-1 K) → M0 (1 K) |
| Late  | S5 (2 K)        | **A3-default (2 K)** | M1 (2 K per-head)          |

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
