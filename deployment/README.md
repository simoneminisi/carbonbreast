# Deployment matrix: model-only inference benchmark (GPU + CPU)

Measures the trained frontier checkpoints as they would be *deployed*, on the same 4000-image test set:

| mode | runtime | device | precision |
|---|---|---|---|
| `torch_fp32` | PyTorch eager | GPU | fp32 (TF32 disabled) |
| `torch_fp16` | PyTorch eager | GPU | fp16 weights + inputs |
| `trt_fp16` | TensorRT | GPU | fp16 |
| `trt_int8` | TensorRT | GPU | INT8 (entropy calibration, 300 train images) + fp16 fallback |
| `ov_fp32` | OpenVINO | CPU | fp32 |
| `ov_int8` | OpenVINO + NNCF PTQ | CPU | INT8 weights + activations (same 300 calibration images) |

Why a separate benchmark: the training rows' `infer_wh` / `latency_ms_*` time the whole test
DataLoader (PNG decode on the fly, GPU mostly idle), so models 30× apart in FLOPs come out within
2× of each other and fp16 looks "4% faster". Here every image is decoded + normalised once, the
model runs on device-resident batches, and power comes from the hardware counters — so precision,
runtime and device differences are actually visible. Per (backbone, mode) the script does:

1. **accuracy pass** — full test set (batch 16): AUROC (per-image / per-breast), MCC at 0.5 and at
   the val-Youden threshold, predictions CSV, cross-check against the parent training row's AUROC;
2. **steady-state loop** for batch 1 and 16 — 3 s warm-up, then 20 s of timed execution on 64
   resident images → img/s, ms/img, synchronous p50/p90 latency (batch 1), GPU energy
   (`nvml` energy counter, else 20 ms power samples), CPU package energy (RAPL, if the VM exposes
   it — usually not), system energy (CodeCarbon at 1 s sampling, the training rows' tracker), and
   the GPU idle power measured just before the loop.

One row per (backbone, mode, batch) in `results_deploy.csv`; existing cells are skipped.

## 1. Layout

This directory runs in place inside the repository. Put the four trained checkpoints in `./checkpoints/` under the names listed in `run_deploy.sh`; the data directory is `../training_pytorch/` after preprocessing (`DATA_DIR`, default `../training_pytorch/`).

## 2. Setup (fresh venv — torch + TensorRT + OpenVINO; never inside the TF or JAX venvs)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_deploy.txt
python -c "import torch,tensorrt as trt; print(torch.cuda.get_device_name(0), 'trt', trt.__version__, trt.Builder(trt.Logger()).platform_has_fast_int8)"
```

`requirements_deploy.txt` pins **`tensorrt-cu12`** on purpose: the bare `tensorrt` package now resolves
to CUDA-13 wheels, which fail with "CUDA initialization failure (35)" on CUDA-12 drivers.

## 3. Run

```bash
LIMIT=64 LOOP_SECS=3 RESULTS=preflight.csv bash run_deploy.sh   # pre-flight: every mode on 64 images (~10 min, builds the engines)
bash run_deploy.sh                                              # the real matrix (engines/IRs from the pre-flight are reused)
```

Runs the four checkpoints (`resnet18`, `efficientnet_b0`, `mobilenetv3_small_100`, `convnext_small`;
resolution and parent training row per checkpoint are listed in `run_deploy.sh`) through all six
modes. Budget ~1–2 h wall-time, dominated by INT8 calibration/engine builds and convnext on CPU;
the energy is negligible. Env overrides: `MODES=... BATCHES=... BACKBONES="resnet18 efficientnet_b0"
LOOP_SECS=... CALIB=... CPU_THREADS=... RESULTS=... bash run_deploy.sh`. `env/` collects
`nvidia-smi -q`, `lscpu`, `free`, and library versions; `env_deploy.json` has the same in JSON.

Outputs: `results_deploy.csv`, `predictions_deploy/pred_<backbone>_img<S>_<mode>.csv`,
`trt_engines/` (ONNX, engines, INT8 calibration cache), `ov_ir/` (OpenVINO IR per mode).

## 4. Reading the numbers

- `gpu_w` / `gpu_wh_per_1k` — GPU power from 20 ms nvml power samples over the exact timed window
  (`gpu_energy_src` says how many samples); `gpu_w_counter` is the nvml energy counter over the same
  window — it lags by ~1 s, so it over-reads short loops (fine at 20 s, garbage at 3 s). `cpu_wh_per_1k`
  comes from RAPL when readable (`cpu_energy_src`); `sys_wh_per_1k` — CodeCarbon's system figure for the same loop,
  comparable to the training rows' `infer_wh`. For GPU modes `sys` includes the idle CPU/RAM; for CPU
  modes it includes the idle GPU if the box has one — compare within a device with the device
  counter, across devices with `sys` and with `gpu_idle_w` in mind.
- `img_per_s` / `ms_per_img` — throughput at steady state (async launches, one sync at the end);
  `lat_ms_p50_sync` — single-image, synchronous, the "one mammogram arrives" latency.
- Accuracy cells repeat across the two batch rows of a mode. A `notes` value flags a >0.03 AUROC
  gap from the parent row: for fp32 modes that means the checkpoint is not the run the tag claims;
  for INT8 modes it means the quantization damaged that architecture (expected for convnext PTQ).
- `cpu_vnni` / `cpu_amx` say whether the CPU has INT8 acceleration (AVX-512 VNNI / AMX) — the reason
  OpenVINO INT8 may or may not pay off on a given VM.
