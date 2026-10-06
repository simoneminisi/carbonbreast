# Plain-TensorFlow replication of the Pareto frontier

Keras-free TensorFlow port of the 7 `pareto_mcc_opt` frontier configs (plus the 2 Stage-1 patch
trunks two of them need). Same augmentation (cv2, verbatim), per-image z-score, balanced sampler,
warmup+cosine per-step LR, val-AUROC model selection, energy regions and 49-column results row /
run-tag format as the PyTorch and Keras runs — so `results_tf.csv` rows concatenate with theirs.

What is different from the Keras port, on purpose (this is what the experiment measures):

- **No Keras, no tf.data.** Models are `tf.Module`s built from `tf.nn` ops (`layers_tf.py`,
  `backbones_tf.py`); the whole training step is ONE `tf.function` (`train_tf.py`). The input
  pipeline is a numpy + cv2 worker-process pool writing batches into shared-memory ring buffers
  (`../common/pipeline_np.py`, shared with the JAX port).
- **timm ImageNet weights for all three backbones** (`pretrained/<backbone>.npz`, converted with
  `../common/tools/convert_timm_to_npz.py`): the PyTorch runs' provenance. resnet18 = `resnet18.a1_in1k`
  (byte-identical to the KerasHub preset the Keras runs used); mobilenetv3_small_100 keeps timm's
  1024-d `conv_head` (Keras pooled at 576-d); convnext_small has the pretrained `head.norm`.
- Every backbone is verified layer-by-layer against timm activations (`parity_test_tf.py`).

## 1. Layout

This directory runs in place inside the repository. The shared, framework-free modules (loader, metrics, energy regions, results schema) live in `../common`; the run script puts it on `PYTHONPATH`, and so must you for manual commands: `export PYTHONPATH=../common`. The converted ImageNet weights go in `./pretrained/` (see `../pretrained/README.md`). The data directory is `../training_pytorch/` once the preprocessing of the root README has been run (`png_index.csv`, `processed_dataset_1024/`, `patches/`, `patch_index.csv`).

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_tf.txt
```

`tensorflow[and-cuda]`, `jax[cuda12]` and `torch` pin conflicting `nvidia-*-cu12` wheels — one
framework per venv/directory, never co-install.

Pre-flight (all must pass before launching the batch):

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # [PhysicalDevice(... GPU:0)]
export PYTHONPATH=../common
python parity_test_tf.py --all          # 3x "PARITY OK" vs the timm reference activations (CPU, ~1 min)
python test_optim_tf.py                 # AdamW closed form + LR-schedule parity
python ../common/test_pipeline_np.py --folder ../training_pytorch/   # loader determinism (numpy only)
```

Optional throughput sanity before the batch (1 epoch, ~15 s/epoch expected for resnet18 @384 on
a torch-class GPU; the Keras-TF run took 41 s/epoch):

```bash
python s_main_tf.py --backbone resnet18 --img-size 384 --batch 48 --epochs 1 --warmup-epochs 1 \
  --folder ../training_pytorch/ --cache-ram --no-plot --results-csv sanity.csv
nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv -lms 500   # in another shell
```

GPU utilisation should sit above ~85 %; if it does not, raise `WORKERS` / `PREFETCH`.

## 3. Run

```bash
bash run_pareto_tf.sh
```

In order (each step skipped if its output already exists):

1. Stage-1 trunk `resnet18` (patches @224, bs64, 30 ep) → `trunk_tf_resnet18.npz` + `.energy.json`
2. Stage-1 trunk `convnext_small` → `trunk_tf_convnext_small.npz` + `.energy.json`
3. The 7 Stage-2 frontier runs (adamw / full / lr 1e-4 / seed 42 / q-fp16 / AMP):

| backbone | img | bs | ep | trunk |
|---|---|---|---|---|
| resnet18 | 384 | 48 | 15 | – |
| mobilenetv3_small_100 | 512 | 48 | 15 | – |
| resnet18 | 512 | 32 | 15 | – |
| mobilenetv3_small_100 | 768 | 32 | 15 | – |
| resnet18 | 768 | 16 | 15 | – |
| resnet18 | 768 | 16 | 15 | ✓ |
| convnext_small | 768 | 8 | 40 (warmup 5) | ✓ |

Env overrides: `DATA_DIR=... RESULTS=... SEED=... ROUNDS=... WORKERS=... PREFETCH=... XLA=1 bash run_pareto_tf.sh`
(`XLA=1` compiles the train/predict steps with `jit_compile=True`; off by default so the default run
is the plain-graph TensorFlow measurement. If you run both, use a different `RESULTS` file.)

Outputs land in THIS dir: `results_tf.csv`, `best_<tag>.npz` (+ `_fp16.npz`),
`predictions/pred_<tag>.csv`, `test_confusion_matrix_*.png`, trunks + energy sidecars.

## 4. Plots

```bash
python plot_pareto.py results_tf.csv mcc_opt --unit wh --logx --rowid
# combined frontier (same columns, same tags -> rows concatenate directly):
head -1 ../training_pytorch/results.csv > combined.csv
tail -n +2 ../training_pytorch/results.csv >> combined.csv            # torch
tail -n +2 ../training_keras/results_keras.csv >> combined.csv # keras/TF
tail -n +2 results_tf.csv >> combined.csv                                    # bare TF
python plot_pareto.py combined.csv mcc_opt --unit wh --logx --rowid
```

Use `--unit wh` (or `--co2-factor 475`): CodeCarbon's per-run CO2 factors are inconsistent.

## 5. Sizing / troubleshooting

- `--cache-ram` at 768px needs ~10 GB (train+val uint8) plus the loader ring buffers:
  `(WORKERS + PREFETCH + 1) × batch × S² × 12 B` per loader (3 loaders) ≈ 4.4 GB at 768/bs16 with
  8 workers — the RAM guard accounts for both. Lower `WORKERS`/`PREFETCH` on small boxes.
- Loader workers are forked BEFORE TensorFlow initialises the GPU and never import TF. `--workers 0`
  runs the loader inline (debugging only).
- OOM on convnext_small @768: try `TF_GPU_ALLOCATOR=cuda_malloc_async`, else halve `bs` in the
  CONFIGS array (new tag/row, as in the Keras README).
- The first training step includes graph tracing (a few seconds, printed as `first step`); it is
  inside the tracked energy region, as the Keras/torch first-step overheads were.
- If codecarbon prints "unavailable", training still runs and logs wall-time only.

## 6. Known deltas vs the Keras / PyTorch runs (documented for the paper)

- Weights: timm for all three backbones (= torch provenance), incl. the 1024-d mobilenetv3 head and
  the pretrained convnext head-LayerNorm — NOT the Keras/Google weights of the Keras runs.
- AdamW `eps = 1e-8` (torch/optax default; Keras used 1e-7). Decoupled lr-scaled weight decay on
  all params in both.
- LR schedule is stepped on every optimizer call, including loss-scale-skipped steps (Keras
  behaviour); dynamic loss scale 2^15, ×2 / 2000 finite steps, ÷2 on overflow.
- Head init: glorot-uniform kernel / zero bias (Keras Dense default).
- Loader is bitwise deterministic per seed regardless of worker scheduling (the torch / Keras
  pipelines were not); the balanced index stream and augmentation code are verbatim.
- `size_mb_*` counts parameter bytes (fp32 incl. BN statistics; the fp16 copy casts everything
  incl. the head); `latency_ms_fp32` is the tracked test pass / n_test, `latency_ms_fp16` a raw timer.
