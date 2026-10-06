# vindr_mammo_resnet18

Single-view, per-image binary classifier on VinDr-Mammo, **ResNet-18**. Originally a straight
copy of the `cbis_mass_resnet18` recipe; now upgraded with the ingredients that make
`experiments/vindr_multiview` (a replication of Petrini & Kim, arXiv 2503.19945) strong — whose
*single-view* model already reaches AUC ~0.851.

**Task — binary BI-RADS risk** (matches the `binary_suspicious` convention):
- `label 0` = normal/benign — BI-RADS 1-2
- `label 1` = suspicious — BI-RADS 3-5

## Pipeline (2 stages)
```bash
python ../preprocessing/build_index.py                          # once: png_index.csv (full-image labels/splits)

# Stage 1 — patch classifier (the key ingredient)
python make_patches.py                         # 224px lesion/background patches -> patch_index.csv
python train_patch.py                          # ResNet-18 4-class -> patch_trunk.pt (conv trunk)

# Stage 2 — full-image classifier
python s_main.py --pretrained-trunk patch_trunk.pt   # patch-pretrained trunk (best)
python s_main.py                                     # plain ImageNet init (A/B baseline)

# or run everything:
python run_all.py                              # full pipeline
python run_all.py --no-patch-pretrain          # ImageNet baseline, for comparison
```
`python s_main.py --help` lists all flags (backbone, pretrained-trunk, img-size, batch, epochs,
warmup-epochs, lr, weight-decay, dropout, seed, rounds, workers, amp, cache-ram, ckpt).

### Regularization & best-of-N (fighting the rare-class overfit)
The val AUROC peaks early (~epoch 7) then declines — a rare-class overfit (only ~1.3k suspicious
train images, oversampled by the balanced sampler). Two knobs address it:
- `--weight-decay` (default 0.05) + `--dropout` (default 0.2, a dropout before the classifier
  `fc`) — regularize to lift the peak. The dropout head keeps the `fc.` prefix so the
  patch-trunk transfer still loads cleanly.
- `--rounds N` + `--seed S` — the val curve is noisy, so train N seeded rounds (seeds S, S+1, …),
  keep the best val AUROC (mirrors the paper's 3-round selection). Round r saves `best_png_rN.pt`;
  the winner is copied to `--ckpt`. Best current recipe:
```bash
python s_main.py --pretrained-trunk patch_trunk.pt --img-size 768 --batch 16 --workers 4 \
    --epochs 15 --rounds 3 --cache-ram --no-plot
```
`train_patch.py` also takes `--weight-decay` and `--seed` for a reproducible trunk.

### Ablation knobs (parametrized)
The pipeline is ablation-ready. The **backbone flows through both stages** (shared builders in
`pipeline.py`), so the Stage-1 trunk always matches the Stage-2 model.

| knob | flag | notes |
|---|---|---|
| Backbone | `--backbone` | torchvision `resnet18`/`resnet50`/`efficientnet_b0` (trunk-compatible) **or any timm model** (`convnext_tiny`, `vit_*`, …). Threaded through `train_patch.py`, `s_main.py`, `run_all.py`. |
| Optimizer | `--optimizer {adamw,adam,sgd,rmsprop}` `--momentum` | SGD/RMSprop want a higher `--lr` than Adam. |
| Train mode | `--train-mode {full,head}` | `head` = freeze backbone, train only the classifier (linear probe; BN kept in eval). |
| Quantization (fp16) | `--quantize {none,fp16}` | quick in-run check: after fp32 eval, also evaluates a half-precision GPU copy and reports size / AUROC / latency. INT8 serving is measured separately in `../deployment/`. |
| Grad checkpointing | `--grad-checkpoint` | timm models only; trades compute for memory. Needed to fit heavy transformers like **ViT-Small @768** (its 224px pos-embeddings are auto-interpolated to the training resolution via `dynamic_img_size`). ViT should run **no-trunk** (pos-embeddings are resolution-specific, so the 224px patch trunk doesn't transfer). |

`train_patch.py` writes `trunk_<backbone>.pt` (+ a `.energy.json` sidecar). `s_main.py` writes
**`best_<tag>.pt`** — a per-run name (the full `<tag>`, not just the backbone), so two runs of the
same backbone at different img-size/optimizer/… never clobber each other's checkpoint; pass
`--ckpt` to override. A trunk/backbone mismatch (e.g. a resnet trunk into a convnext) fails
loudly. End-to-end with a different backbone:
```bash
python run_all.py --skip-patches --backbone convnext_tiny \
    --extra "--img-size 768 --batch 16 --epochs 15 --cache-ram --no-plot"
```

### Reported metrics, predictions, and energy
Every `s_main.py` run reports, in addition to per-image + per-breast AUROC:
- **Confusion-matrix metrics** at the 0.5 and val-selected thresholds: **MCC, TPR/sensitivity,
  TNR/specificity, PPV/precision, NPV** (+ TP/TN/FP/FN).
- **Energy / CO₂ / time** via CodeCarbon for three regions, reported separately: Stage-1 patch
  training, Stage-2 training, and inference (Wh + gCO₂ + wall-seconds). `--no-energy` disables it;
  it degrades to wall-time only if CodeCarbon is missing/offline.

Outputs:
- `predictions/pred_<tag>.csv` — one row per test image (`patient_id, laterality, view, label,
  prob, pred_0.5, pred_opt`, + `prob_fp16` when quantized); enough to recompute any metric.
- `results.csv` — **one appended row per run** with all parameters + metrics + all three energy
  regions (Stage-1 energy pulled from the trunk sidecar), so seed/backbone/optimizer sweeps
  accumulate for side-by-side comparison. `<tag>` encodes the run's parameters.

Install extras for these: `timm` (non-torchvision backbones) and `codecarbon` (energy) — see
`requirements.txt`.

### Pareto plot
`../common/plot_pareto.py` draws accuracy against training energy from `results.csv` and marks the strict-improvement frontier:
```bash
python ../common/plot_pareto.py results.csv mcc_opt --logx
```

## What changed vs the original recipe (ported from `vindr_multiview`)
Roughly in order of expected impact:

1. **Patch-classifier pretraining ("PBC") — the main lever.** `make_patches.py` cuts 224px
   patches (4-class: background / BI-RADS 3/4/5) with **exact per-finding BI-RADS** labels joined
   from `finding_annotations.csv`. The boxes are re-derived through the same geometric transform
   that baked `processed_dataset_1024` (`preprocessing/preprocess.py`) — verified identical to
   the stored boxes to 0.0px — carrying each finding's BI-RADS, then patches are cropped from the
   already-baked PNGs. (We don't reuse `processed_index.csv`'s `bboxes_xyxy_json` because it drops
   out-of-crop boxes and omits per-finding BI-RADS.) `train_patch.py` trains a ResNet-18 and
   exports the conv trunk (`state_dict` minus `fc.*`); `s_main.py --pretrained-trunk` loads it via
   `strict=False` in place of raw ImageNet weights — the ResNet analogue of the paper's
   `children[:-2]` extractor transfer. Patch-val is carved from **training** studies, so the
   full-image test set is never touched in Stage 1.
2. **Higher resolution** — default `--img-size 1024` (native PNG size, no resize on read), vs the
   old 512.
3. **Per-image z-score normalization** (`pipeline.py`) instead of `/255` + ImageNet mean/std.
4. **Balanced oversampling sampler** (inverse-freq `WeightedRandomSampler`) with **plain CE**,
   replacing the class-weighted loss.
5. **Stronger augmentation** — rotation ±25°, shear ±12°, scale ±20%, intensity ±20%,
   `BORDER_REFLECT` (ported from the official VinDr affine/flip).
6. **Warmup + cosine LR**, all params fine-tuned (no freeze phase), mirroring the paper.
7. **Per-breast evaluation** — test probs aggregated by `(patient_id, laterality)` with max and
   mean over CC+MLO, reported alongside per-image AUROC.

`pipeline.py` holds the shared dataset/normalization/augmentation/sampler/LR helpers used by both
stages, so the patch and full-image pixel pipelines are identical (required for a valid transfer).

### Key experiment
Compare Stage-2 val/test AUROC **with vs without** `--pretrained-trunk` (i.e. `run_all.py` vs
`run_all.py --no-patch-pretrain`) to quantify what the patch-pretraining transfer is worth — the
core hypothesis behind the gap with `vindr_multiview`.

## GPU / VRAM (6 GB RTX A1000)
AMP (on by default) + `channels_last` use the Ampere tensor cores and halve activation memory.
ResNet-18 peak VRAM: ~2.2 GB at batch 32 / 512px; roughly 4× per-sample at 1024px, so the
default batch 12 @ 1024px fits comfortably. Lower `--img-size` (768/512) or `--batch` if needed;
`--no-amp` reverts to fp32.

## Preloading into RAM (`--cache-ram`)
Profiling showed disk isn't the steady-state bottleneck: the OS **page cache** already serves the
~4.4 GB of PNGs at ~0.04 ms/img. The real per-image cost is **PNG decode (~3.7 ms @1024)**.
`--cache-ram` preloads decoded images into one contiguous `uint8` array (shared across workers via
copy-on-write), cutting the per-image cost to ~0.013 ms:

```bash
python s_main.py --cache-ram                 # cache train+val (~16 GB @1024), stream test
python s_main.py --cache-ram --cache-test    # also cache test (~21 GB total)
python s_main.py --cache-ram --img-size 512  # ~4 GB, plenty of headroom
```

Footprint scales with `img_size²`: train+val @1024 ≈ 16 GB, all 20k ≈ 21 GB; @512 those become
~4 / ~5 GB. The cache is one contiguous array shared across workers (verified: 6 workers add only
the shared copy, not 6×). On top of it the DataLoader keeps **prefetch buffers** as float32 tensors
in flight (`workers × prefetch × batch × 3 × img² × 4 B`) — several GB at 1024px — so the RAM guard
counts those too. With a RAM cache there's no decode to hide, so **fewer `--workers` (e.g. 2–4) is
usually fine** and cuts that overhead.

**This box has no swap**, so the guard refuses an unsafe allocation unless `--force-cache`; at
1024 GB-scale, close the IDE and any other processes first. Note: at 1024px the model is often
already **GPU-bound** (decode is only ~6 s/epoch across 8 workers), so the cache may help little —
check the `img/s` heartbeat with vs without it. The cache pays off most at 512px.

## Data / splits
`build_index.py` reshapes `processed_dataset_1024/processed_index.csv` into
`png_path,patient_id,laterality,view,label,split`, fixing the missing `experiments/` path prefix
and mapping the 0-indexed `breast_birads` (0 = BI-RADS 1 … 4 = BI-RADS 5) to the binary label.
VinDr's official `training`/`test` folds are used; val is carved from `training` grouped by
`study_id` so CC/MLO of a study never straddle train/val. Counts: train ~13.6k / val ~2.4k /
test 4k images, ~9.6% suspicious.
