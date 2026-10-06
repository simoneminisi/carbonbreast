# Keras 3 replication of the Pareto frontier (TensorFlow or JAX backend)

Keras 3 (TensorFlow or JAX backend — see §4) port of `training_pytorch/`, restricted to the **7
configurations on the `pareto_mcc_opt.png` frontier** plus the 2 Stage-1 patch trunks two of
them need. Same augmentation (cv2, verbatim), same per-image z-score, same balanced sampler,
same warmup+cosine per-step LR, same val-AUROC model selection, same 48-column results row
and run-tag format — so Keras rows and the existing PyTorch rows plot on one Pareto chart.

## 1. Layout

This directory runs in place inside the repository. The data directory is `../training_pytorch/` once the preprocessing of the root README has been run (`png_index.csv`, `processed_dataset_1024/`, `patches/`, `patch_index.csv`); pass it as `DATA_DIR=../training_pytorch/`.

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_keras.txt
```

Verify BEFORE launching the batch:

```bash
# GPU visible to TF
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# resnet18 ImageNet preset available in this keras-hub
python -c "import keras_hub; print([p for p in keras_hub.models.ResNetBackbone.presets if '18' in p])"
```

The second command must list `resnet_18_imagenet` (or a close variant — the code falls back
to any `*18*imagenet*` ResNet preset automatically and prints which one it picked).

## 3. Run

```bash
bash run_pareto_keras.sh
```

What it does, in order (each step skipped if its output already exists):

1. Stage-1 trunk `resnet18` (patches @224, bs64, 30 ep) → `trunk_keras_resnet18.weights.h5` + `.energy.json`
2. Stage-1 trunk `convnext_small` → `trunk_keras_convnext_small.weights.h5` + `.energy.json`
3. The 7 Stage-2 frontier runs (all adamw / full / lr 1e-4 / seed 42 / q-fp16 / AMP):

| backbone | img | bs | ep | trunk |
|---|---|---|---|---|
| resnet18 | 384 | 48 | 15 | – |
| mobilenetv3_small_100 | 512 | 48 | 15 | – |
| resnet18 | 512 | 32 | 15 | – |
| mobilenetv3_small_100 | 768 | 32 | 15 | – |
| resnet18 | 768 | 16 | 15 | – |
| resnet18 | 768 | 16 | 15 | ✓ |
| convnext_small | 768 | 8 | 40 (warmup 5) | ✓ |

Env overrides: `BACKEND=... DATA_DIR=... RESULTS=... SEED=... ROUNDS=... WORKERS=... bash run_pareto_keras.sh`

Outputs land in THIS dir: `results_keras.csv`, `best_<tag>.weights.h5` (+ `_fp16`),
`predictions/pred_<tag>.csv`, `test_confusion_matrix_*.png`, trunks + energy sidecars.

## 4. JAX backend

The same code runs on the Keras 3 **JAX** backend (`BACKEND=jax`); results go to
`results_keras_jax.csv`. Use a NEW directory and a FRESH venv — jax, tensorflow[and-cuda]
and torch pin **conflicting** `nvidia-*-cu12` wheels and must never share an environment
(installing one on top of another breaks the other, as seen with torch+TF on this VM):

```bash
cd training_keras
python -m venv .venv_jax && source .venv_jax/bin/activate
pip install --upgrade pip
pip install -r requirements_keras_jax.txt

# pre-flight: JAX must see the GPU (tensorflow-cpu intentionally cannot)
python -c "import jax; print(jax.devices())"          # expect [CudaDevice(id=0)]
python -c "import keras_hub; print([p for p in keras_hub.models.ResNetBackbone.presets if '18' in p])"

BACKEND=jax bash run_pareto_keras.sh
```

tf.data (CPU) still feeds the input pipeline — that is the supported Keras 3 pattern on JAX.
The Stage-1 trunk `.weights.h5` files are backend-portable, but retraining them in the JAX
env keeps the Stage-1 energy measurement attributable to JAX (the script does this
automatically since the fresh dir has no trunk files).

## 5. Plots

```bash
# single-framework frontier
python plot_pareto.py results_keras.csv mcc_opt --unit wh --logx --rowid

# combined frontier (same columns, same tags -> rows concatenate directly);
# add whichever frameworks you have:
head -1 ../training_pytorch/results.csv > combined.csv
tail -n +2 ../training_pytorch/results.csv >> combined.csv                # torch
tail -n +2 ../training_keras/results_keras.csv >> combined.csv    # keras/TF
tail -n +2 results_keras_jax.csv >> combined.csv                                # keras/JAX
python plot_pareto.py combined.csv mcc_opt --unit wh --logx --rowid
```

Use `--unit wh` (or `--co2-factor 475` for gCO2 at a fixed grid intensity) — CodeCarbon's
per-run CO2 factors proved inconsistent across the torch runs.

## 6. Sizing / troubleshooting

- `--cache-ram` at 768px needs ~16 GB free RAM (train+val uint8) — same as the torch runs.
- TF's allocator differs from torch: if a run OOMs (most likely convnext_small @768), edit its
  `bs` in the CONFIGS array of `run_pareto_keras.sh` (halving it creates a new tag/row).
- Workers only affect the RAM-cache decode; the training input pipeline is tf.data AUTOTUNE.
- If codecarbon prints "unavailable", training still runs and logs wall-time only.

## 7. Known deltas vs the PyTorch runs (accepted, documented for the paper)

- **Weight provenance**: resnet18 = KerasHub timm-converted; mobilenetv3/convnext =
  Keras/Google ImageNet weights (torch used torchvision V1 / timm). Small metric shifts are
  part of what the framework comparison measures.
- **mobilenetv3_small_100 head**: keras `include_top=False` pools to 576-d features; timm's
  head had an extra 1024-d conv stage. Slightly smaller classifier input.
- **convnext_small head-LayerNorm**: timm ships it pretrained; keras only inside
  `include_top=True`, so ours starts random — irrelevant here because the convnext run uses
  the Stage-1-retrained trunk (which trains and exports that LayerNorm).
- **Loss-scale skips**: torch skipped the LR-schedule step when AMP's GradScaler skipped the
  optimizer; Keras ties the schedule to `optimizer.iterations` (no skipping).
- Input pipeline is seeded but not bitwise-reproducible across runs (parallel numpy augment
  threads) — same caveat as torch's multi-worker DataLoader.
