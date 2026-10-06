# Plain-JAX (flax + optax) replication of the Pareto frontier

Keras-free, TensorFlow-free JAX port of the 7 `pareto_mcc_opt` frontier configs (plus the 2
Stage-1 patch trunks two of them need). Same augmentation (cv2, verbatim), per-image z-score,
balanced sampler, warmup+cosine per-step LR, val-AUROC model selection, energy regions and
49-column results row / run-tag format as the PyTorch, Keras and bare-TF runs — so
`results_jax.csv` rows concatenate with theirs.

What is different from the Keras-JAX run, on purpose:

- **No Keras, no tf.data, no tensorflow at all.** Models are `flax.linen` modules
  (`layers_jax.py`, `backbones_jax.py`), the optimizer is `optax.adamw`, and the whole training
  step is ONE `jax.jit` (`train_jax.py`). The input pipeline is a numpy + cv2 worker-process pool
  writing batches into shared-memory ring buffers (`../common/pipeline_np.py`, shared with the TensorFlow port).
- **timm ImageNet weights for all three backbones** (`pretrained/<backbone>.npz`): the PyTorch runs'
  provenance (resnet18 = `resnet18.a1_in1k`, identical to the KerasHub preset; mobilenetv3 with
  timm's 1024-d `conv_head`; convnext_small with the pretrained `head.norm`).
- Every backbone is verified layer-by-layer against timm activations (`parity_test_jax.py`).

## 1. Layout

This directory runs in place inside the repository. The shared, framework-free modules (loader, metrics, energy regions, results schema) live in `../common`; the run script puts it on `PYTHONPATH`, and so must you for manual commands: `export PYTHONPATH=../common`. The converted ImageNet weights go in `./pretrained/` (see `../pretrained/README.md`). The data directory is `../training_pytorch/` once the preprocessing of the root README has been run.

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_jax.txt
```

`jax[cuda12]`, `tensorflow[and-cuda]` and `torch` pin conflicting `nvidia-*-cu12` wheels — one
framework per venv/directory, never co-install. Do NOT install tensorflow(-cpu) here.

Pre-flight:

```bash
python -c "import jax; print(jax.devices())"        # [CudaDevice(id=0)]
export PYTHONPATH=../common
JAX_PLATFORMS=cpu python parity_test_jax.py --all   # 3x "PARITY OK" vs the timm reference activations
JAX_PLATFORMS=cpu python test_optax_adamw.py        # optax.adamw closed form + LR-schedule parity
python ../common/test_pipeline_np.py --folder ../training_pytorch/
```

Optional throughput sanity (1 epoch; the Keras-JAX run took 47 s/epoch for resnet18 @384):

```bash
python s_main_jax.py --backbone resnet18 --img-size 384 --batch 48 --epochs 1 --warmup-epochs 1 \
  --folder ../training_pytorch/ --cache-ram --no-plot --results-csv sanity.csv
nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv -lms 500
```

## 3. Run

```bash
bash run_pareto_jax.sh
```

Stage-1 trunks `trunk_jax_resnet18.npz` / `trunk_jax_convnext_small.npz` (+ `.energy.json`), then the
same 7 Stage-2 configs as the Keras / TF batches (see `run_pareto_jax.sh`), written to
`results_jax.csv`. Env overrides: `DATA_DIR=... RESULTS=... SEED=... ROUNDS=... WORKERS=...
PREFETCH=... bash run_pareto_jax.sh`. `XLA_PYTHON_CLIENT_PREALLOCATE=false` is exported by the
script (as in the Keras-JAX run) so OOMs are honest and CodeCarbon sees real GPU usage.

Outputs: `results_jax.csv`, `best_<tag>.npz` (+ `_fp16.npz`), `predictions/pred_<tag>.csv`,
`test_confusion_matrix_*.png`, trunks + energy sidecars.

## 4. Plots

```bash
python plot_pareto.py results_jax.csv mcc_opt --unit wh --logx --rowid
head -1 ../training_pytorch/results.csv > combined.csv
tail -n +2 ../training_pytorch/results.csv >> combined.csv
tail -n +2 results_jax.csv >> combined.csv
python plot_pareto.py combined.csv mcc_opt --unit wh --logx --rowid
```

## 5. Sizing / troubleshooting

- RAM: 768px cache ≈ 10 GB + loader rings `(WORKERS + PREFETCH + 1) × batch × S² × 12 B` per loader.
- OOM on convnext_small @768: `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, else halve `bs` in CONFIGS.
- The first step includes XLA compilation (printed as `first step`), inside the tracked region like
  the Keras/torch first-step overheads.
- Loader workers are forked before JAX initialises the GPU and never import jax.

## 6. Known deltas vs the Keras / PyTorch runs

Same list as the TensorFlow port (`../training_tf/README.md` §6): timm weights (torch provenance) for all three
backbones incl. the 1024-d mobilenetv3 head and pretrained convnext head-LN; AdamW eps 1e-8 with
decoupled lr-scaled weight decay on all params (`optax.adamw`, `mask=None`); schedule stepped on
every call incl. loss-scale skips; dynamic loss scale 2^15 / ×2 per 2000 finite steps / ÷2 on
overflow; glorot head init; deterministic loader; `size_mb_*` = parameter bytes.
