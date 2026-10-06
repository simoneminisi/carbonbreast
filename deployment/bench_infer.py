"""VinDr-Mammo deployment matrix -- MODEL-ONLY inference benchmark of trained checkpoints.

Why: the training rows' `infer_wh` / `latency_ms_*` time the whole test DataLoader (PNG decode on
the fly at low GPU utilisation), so models 30x apart in FLOPs differ by 2x and fp16 is "4% faster".
Here the test images are decoded + normalised ONCE, the model runs on resident inputs, and power is
read from the hardware counters, so precision / runtime / device differences actually show.

Modes (--modes):
    torch_fp32   PyTorch eager, true fp32 (TF32 disabled)         GPU
    torch_fp16   PyTorch eager, .half() weights + inputs          GPU
    trt_fp16     TensorRT engine, FP16                            GPU
    trt_int8     TensorRT engine, INT8 (entropy calibration on --calib-size train images) + FP16 fallback
    ov_fp32      OpenVINO fp32                                    CPU
    ov_int8      OpenVINO + NNCF post-training INT8 (weights+activations, same calibration set)  CPU

For every (backbone, mode):
  A. accuracy pass: full test set (4000 imgs) -> AUROC / MCC at 0.5 and at the val-Youden threshold,
     predictions CSV, cross-check against the parent training row's AUROC (--parent-tag).
  B. steady-state loop for each --batches size on GPU-/host-resident pre-normalised batches:
     --warmup-secs, then --loop-secs of timed execution -> img/s, ms/img, sync p50/p90 latency,
     GPU energy (nvml energy counter, else power samples), CPU package energy (RAPL, if readable),
     system energy (CodeCarbon, 1 s sampling), idle GPU power before the loop.
One row per (backbone, mode, batch) -> results_deploy.csv; cells already present are skipped.

    python bench_infer.py --backbone resnet18 --ckpt checkpoints/best_resnet18.pt \
        --parent-tag resnet18_adamw_full_q-fp16_img768_bs16_ep15_r1_s42_trunk-0 \
        --folder ../training_pytorch/ --modes torch_fp32,torch_fp16,trt_fp16,trt_int8,ov_fp32,ov_int8
"""
import argparse
import csv
import datetime
import glob
import json
import os
import platform
import statistics
import threading
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from metrics import confusion_metrics, per_breast_auc
from pipeline import create_backbone, decode_all, zscore_3ch

GPU_MODES = ("torch_fp32", "torch_fp16", "trt_fp16", "trt_int8")
CPU_MODES = ("ov_fp32", "ov_int8")
ALL_MODES = GPU_MODES + CPU_MODES
BACKENDS = {"torch_fp32": "torch", "torch_fp16": "torch", "trt_fp16": "tensorrt", "trt_int8": "tensorrt",
            "ov_fp32": "openvino", "ov_int8": "openvino"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True)
    p.add_argument("--ckpt", required=True, help="trained fp32 state_dict (best_<...>.pt)")
    p.add_argument("--parent-tag", default="", help="training-row tag (img size is parsed from it, AUROC cross-checked)")
    p.add_argument("--parent-results", default="parent_results.csv", help="results.csv with the parent rows")
    p.add_argument("--img-size", type=int, default=None, help="default: img<S> parsed from --parent-tag, else 768")
    p.add_argument("--modes", default=",".join(ALL_MODES))
    p.add_argument("--batches", default="1,16", help="steady-state batch sizes")
    p.add_argument("--acc-batch", type=int, default=16, help="batch for the accuracy pass (<= max(--batches))")
    p.add_argument("--folder", default="../training_pytorch/", help="data dir prefix (must end with /)")
    p.add_argument("--png-index", default=None, help="default: <folder>png_index.csv")
    p.add_argument("--calib-size", type=int, default=300, help="# train images for INT8 calibration")
    p.add_argument("--warmup-secs", type=float, default=3.0)
    p.add_argument("--loop-secs", type=float, default=20.0)
    p.add_argument("--loop-imgs", type=int, default=64, help="# distinct resident images cycled in the loop")
    p.add_argument("--sync-iters", type=int, default=200, help="# synchronous calls for p50/p90 latency (batch 1 only)")
    p.add_argument("--workers", type=int, default=8, help="decode threads")
    p.add_argument("--cpu-threads", type=int, default=0, help="OpenVINO inference threads (0 = runtime default)")
    p.add_argument("--results-csv", default="results_deploy.csv")
    p.add_argument("--pred-dir", default="predictions_deploy")
    p.add_argument("--engine-dir", default="trt_engines")
    p.add_argument("--ir-dir", default="ov_ir")
    p.add_argument("--limit", type=int, default=0, help="DEBUG: use only the first N test/val images")
    p.add_argument("--no-energy", dest="energy", action="store_false", help="disable CodeCarbon")
    p.add_argument("--force", action="store_true", help="re-run cells already in the results csv")
    return p.parse_args()


# ------------------------------------------------------------------ data
def split_trainval(df, val_frac=0.15, seed=42):
    train = df[df["split"] == "train"]
    tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=val_frac, random_state=seed)
                          .split(train, groups=train["patient_id"]))
    return train.iloc[tr_idx], train.iloc[va_idx]


def norm_batch(u8s, S):
    """(B,S,S) uint8 -> (B,3,S,S) float32 numpy, the exact training/eval preprocessing."""
    return np.stack([zscore_3ch(u, S).numpy() for u in u8s]).astype(np.float32, copy=False)


# ------------------------------------------------------------------ hardware counters
class Nvml:
    def __init__(self):
        self.ok = False
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device() if torch.cuda.is_available() else 0)
            self.name = pynvml.nvmlDeviceGetName(self.h)
            self.name = self.name.decode() if isinstance(self.name, bytes) else self.name
            self.driver = pynvml.nvmlSystemGetDriverVersion()
            self.driver = self.driver.decode() if isinstance(self.driver, bytes) else self.driver
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
                self.has_energy = True
            except Exception:  # noqa: BLE001
                self.has_energy = False
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[nvml] unavailable ({type(e).__name__}: {e}); GPU power not measured")
            self.name, self.driver, self.has_energy = "", "", False

    def energy_mj(self):
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) if self.ok and self.has_energy else None

    def power_w(self):
        return self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0 if self.ok else None


class PowerSampler(threading.Thread):
    """Background nvml power sampler (fallback when the energy counter is unsupported)."""

    def __init__(self, nvml, period=0.02):
        super().__init__(daemon=True)
        self.nvml, self.period, self.samples, self._stop_evt = nvml, period, [], threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            t = time.perf_counter()
            self.samples.append((t, self.nvml.power_w()))
            time.sleep(self.period)

    def stop(self):
        self._stop_evt.set()
        self.join()
        return self.samples


def rapl_uj():
    """Sum of intel-rapl package energy counters (uJ), or None if unreadable (typical on cloud VMs)."""
    tot = 0
    found = False
    for path in sorted(glob.glob("/sys/class/powercap/intel-rapl:[0-9]*/energy_uj")):
        try:
            with open(path) as f:
                tot += int(f.read().strip())
            found = True
        except OSError:
            return None
    return tot if found else None


def rapl_delta_uj(e0, e1):
    if e0 is None or e1 is None:
        return None
    if e1 < e0:  # counter wrap (sum of packages: approximate with the largest range file)
        rng = 0
        for path in glob.glob("/sys/class/powercap/intel-rapl:[0-9]*/max_energy_range_uj"):
            try:
                rng += int(open(path).read().strip())
            except OSError:
                pass
        e1 += rng
    return e1 - e0


class SysTracker:
    """CodeCarbon system energy (GPU+CPU+RAM) with 1 s sampling -- the training rows' tracker."""

    def __init__(self, enabled):
        self.t = None
        if enabled:
            try:
                from codecarbon import EmissionsTracker  # noqa: PLC0415
                self.t = EmissionsTracker(log_level="error", save_to_file=False, measure_power_secs=1)
                self.t.start()
            except Exception as e:  # noqa: BLE001
                print(f"[energy] CodeCarbon unavailable ({type(e).__name__}: {e})")

    def stop_wh(self):
        if self.t is None:
            return None
        try:
            self.t.stop()
            return self.t._total_energy.kWh * 1000.0
        except Exception as e:  # noqa: BLE001
            print(f"[energy] CodeCarbon stop failed ({type(e).__name__}: {e})")
            return None


def cpu_info():
    name, flags = platform.processor() or "", ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and not name.startswith("model"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("flags"):
                flags = line.split(":", 1)[1]
                break
    except OSError:
        pass
    fl = set(flags.split())
    return name, {"vnni": bool(fl & {"avx512_vnni", "avx_vnni"}), "amx": "amx_int8" in fl, "avx512": "avx512f" in fl}


# ------------------------------------------------------------------ runners
class TorchRunner:
    def __init__(self, model, half):
        self.dev = "cuda"
        self.dtype = torch.float16 if half else torch.float32
        self.model = model.cuda().eval()
        if half:
            self.model.half()
        self.size_mb = sum(p.numel() * p.element_size() for p in self.model.state_dict().values()) / 1e6

    def prepare(self, x):
        return torch.from_numpy(x).to("cuda", self.dtype, non_blocking=False)

    @torch.no_grad()
    def run(self, inp):
        return self.model(inp).float().cpu().numpy()

    @torch.no_grad()
    def run_timed(self, inp):
        self.model(inp)

    def sync(self):
        torch.cuda.synchronize()


class TrtRunner:
    """TensorRT 10 (execute_async_v3) with an 8.x fallback (execute_async_v2)."""

    def __init__(self, engine_bytes, S, max_batch):
        import tensorrt as trt  # noqa: PLC0415
        self.trt = trt
        self.dev, self.S = "cuda", S
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.ctx = self.engine.create_execution_context()
        self.v3 = hasattr(self.ctx, "execute_async_v3")
        if self.v3:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.in_name = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
            self.out_name = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.out = torch.empty((max_batch, 2), dtype=torch.float32, device="cuda")
        self.stream = torch.cuda.current_stream().cuda_stream
        self.size_mb = len(engine_bytes) / 1e6
        self._shape = None

    def prepare(self, x):
        return torch.from_numpy(x).to("cuda", torch.float32).contiguous()

    def _set(self, inp):
        b = inp.shape[0]
        if self._shape != b:
            if self.v3:
                self.ctx.set_input_shape(self.in_name, tuple(inp.shape))
            else:
                self.ctx.set_binding_shape(0, tuple(inp.shape))
            self._shape = b
        if self.v3:
            self.ctx.set_tensor_address(self.in_name, int(inp.data_ptr()))
            self.ctx.set_tensor_address(self.out_name, int(self.out.data_ptr()))

    def run_timed(self, inp):
        self._set(inp)
        if self.v3:
            self.ctx.execute_async_v3(self.stream)
        else:
            self.ctx.execute_async_v2([int(inp.data_ptr()), int(self.out.data_ptr())], self.stream)

    def run(self, inp):
        self.run_timed(inp)
        torch.cuda.synchronize()
        return self.out[: inp.shape[0]].cpu().numpy().copy()

    def sync(self):
        torch.cuda.synchronize()


class OvRunner:
    def __init__(self, compiled, size_mb):
        self.dev = "cpu"
        self.req = compiled.create_infer_request()
        self.size_mb = size_mb

    def prepare(self, x):
        return np.ascontiguousarray(x, dtype=np.float32)

    def run_timed(self, inp):
        self.req.infer({0: inp})

    def run(self, inp):
        self.req.infer({0: inp})
        return self.req.get_output_tensor(0).data.copy()

    def sync(self):
        pass


# ------------------------------------------------------------------ model / engine builders
def load_model(backbone, ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    for dr in (0.2, 0.0):  # s_main wraps the head as Sequential(Dropout, Linear) when dropout > 0
        m = create_backbone(backbone, 2, dropout=dr, pretrained=False).eval()
        try:
            m.load_state_dict(sd)
            return m
        except RuntimeError:
            continue
    raise RuntimeError(f"{ckpt}: state_dict does not match a {backbone} head (dropout 0.2 / 0.0)")


def export_onnx(model, S, path):
    if os.path.exists(path):
        return path
    model = model.float().eval().cpu()
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, torch.zeros(1, 3, S, S), path, dynamo=False, **kw)
    except TypeError:  # older torch: no `dynamo` kwarg
        torch.onnx.export(model, torch.zeros(1, 3, S, S), path, **kw)
    print(f"  exported {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def build_trt_engine(onnx_path, engine_path, S, max_batch, int8, calib_u8=None, calib_cache=None):
    """Build (or load) a TensorRT engine with an optimisation profile batch 1..max_batch."""
    import tensorrt as trt  # noqa: PLC0415
    if os.path.exists(engine_path):
        print(f"  engine exists -> {engine_path}")
        return open(engine_path, "rb").read()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, S, S), (max_batch, 3, S, S), (max_batch, 3, S, S))
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = EntropyCalibrator(calib_u8, S, min(16, max_batch), calib_cache)
        if hasattr(config, "set_calibration_profile"):
            config.set_calibration_profile(profile)
    t0 = time.time()
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError("TensorRT engine build failed")
    data = bytes(data)
    with open(engine_path, "wb") as f:
        f.write(data)
    print(f"  built {engine_path} ({len(data) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
    return data


def _make_calibrator_class():
    import tensorrt as trt  # noqa: PLC0415

    class _Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self, u8, S, batch, cache_file):
            super().__init__()
            self.u8, self.S, self.batch, self.cache_file = u8, S, batch, cache_file
            self.i = 0
            self.buf = torch.empty((batch, 3, S, S), dtype=torch.float32, device="cuda")

        def get_batch_size(self):
            return self.batch

        def get_batch(self, names):
            if self.i + self.batch > len(self.u8):
                return None
            x = norm_batch(self.u8[self.i:self.i + self.batch], self.S)
            self.buf.copy_(torch.from_numpy(x))
            self.i += self.batch
            return [int(self.buf.data_ptr())]

        def read_calibration_cache(self):
            if self.cache_file and os.path.exists(self.cache_file):
                return open(self.cache_file, "rb").read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_file:
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

    return _Calib


def EntropyCalibrator(*a, **k):  # noqa: N802 (lazy class creation: tensorrt is optional)
    return _make_calibrator_class()(*a, **k)


def build_ov(model, S, int8, calib_u8, ir_path, cpu_threads):
    import openvino as ov  # noqa: PLC0415
    core = ov.Core()
    if os.path.exists(ir_path):
        m = core.read_model(ir_path)
        print(f"  IR exists -> {ir_path}")
    else:
        with torch.no_grad():
            m = ov.convert_model(model.float().eval().cpu(), example_input=torch.zeros(1, 3, S, S))
        m.reshape({0: [-1, 3, S, S]})
        if int8:
            import nncf  # noqa: PLC0415
            items = [norm_batch(calib_u8[i:i + 1], S) for i in range(len(calib_u8))]  # (1,3,S,S) each, the training-time preprocessing
            t0 = time.time()
            m = nncf.quantize(m, nncf.Dataset(items, lambda x: x), subset_size=len(items))
            print(f"  nncf.quantize: {time.time() - t0:.0f}s")
        ov.save_model(m, ir_path, compress_to_fp16=False)
    size_mb = os.path.getsize(os.path.splitext(ir_path)[0] + ".bin") / 1e6
    props = {"INFERENCE_NUM_THREADS": cpu_threads} if cpu_threads > 0 else {}
    compiled = core.compile_model(m, "CPU", props)
    try:
        n_thr = compiled.get_property("INFERENCE_NUM_THREADS")
    except Exception:  # noqa: BLE001
        n_thr = cpu_threads
    return compiled, size_mb, n_thr, ov.__version__


# ------------------------------------------------------------------ measurement
def accuracy_pass(runner, u8, S, batch):
    probs = []
    n = len(u8)
    for i in range(0, n, batch):
        x = norm_batch(u8[i:i + batch], S)
        b = x.shape[0]
        if b < batch:  # fixed-shape engines: pad the last partial batch
            x = np.concatenate([x, np.repeat(x[-1:], batch - b, axis=0)])
        logits = runner.run(runner.prepare(x))[:b]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs.append(e[:, 1] / e.sum(axis=1))
    return np.concatenate(probs)


def metrics_block(test_df, labels, probs, thr):
    row = {"thr_opt": thr, "test_auroc_img": roc_auc_score(labels, probs)}
    pb = per_breast_auc(test_df, probs)
    row["test_auroc_breast_max"], row["test_auroc_breast_mean"] = pb["max"], pb["mean"]
    for pref, preds in (("0.5", (probs > 0.5).astype(int)), ("opt", (probs > thr).astype(int))):
        m = confusion_metrics(labels, preds)
        for k in ("acc", "mcc", "tpr", "tnr", "ppv", "npv"):
            row[f"{k}_{pref}"] = m[k]
    return row


def steady_state(runner, batches, warmup_s, loop_s, nvml, energy, sync_iters):
    """Cycle resident batches: warm-up, then a timed loop with hardware counters around it."""
    K = len(batches)
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < warmup_s:
        runner.run_timed(batches[i % K])
        i += 1
    runner.sync()
    time.sleep(1.0)  # let the (lagging) nvml energy counter catch up with the warm-up
    sampler = None
    if runner.dev == "cuda" and nvml.ok:  # 20 ms power samples = window-exact primary GPU figure
        sampler = PowerSampler(nvml)
        sampler.start()
    gpu_e0 = nvml.energy_mj() if runner.dev == "cuda" else None
    cpu_e0 = rapl_uj()
    tracker = SysTracker(energy)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < loop_s:
        runner.run_timed(batches[n % K])
        n += 1
    runner.sync()
    t = time.perf_counter() - t0
    sys_wh = tracker.stop_wh()
    cpu_e1 = rapl_uj()
    gpu_e1 = nvml.energy_mj() if runner.dev == "cuda" else None
    imgs = n * batches[0].shape[0]
    out = {"loop_s": t, "loop_imgs": imgs, "img_per_s": imgs / t, "ms_per_img": 1000.0 * t / imgs,
           "gpu_w": "", "gpu_w_counter": "", "gpu_wh_per_1k": "", "gpu_energy_src": "", "cpu_w": "",
           "cpu_wh_per_1k": "", "cpu_energy_src": "", "sys_wh_per_1k": "", "sys_w": "", "sys_energy_src": ""}
    samples = [(tt, p) for tt, p in (sampler.stop() if sampler is not None else []) if p is not None and t0 <= tt <= t0 + t]
    if len(samples) > 1:
        w = statistics.fmean(p for _, p in samples)
        out.update(gpu_w=w, gpu_wh_per_1k=w * t / 3600 / imgs * 1000, gpu_energy_src=f"nvml_power_samples({len(samples)}@20ms)")
    if gpu_e0 is not None and gpu_e1 is not None:
        wh = (gpu_e1 - gpu_e0) / 3.6e6
        out["gpu_w_counter"] = wh * 3600 / t  # nvml energy counter; lags ~1 s, trust only for long loops
        if len(samples) <= 1:
            out.update(gpu_w=wh * 3600 / t, gpu_wh_per_1k=wh / imgs * 1000, gpu_energy_src="nvml_energy_counter")
    d = rapl_delta_uj(cpu_e0, cpu_e1)
    if d is not None:
        wh = d / 3.6e9
        out.update(cpu_w=wh * 3600 / t, cpu_wh_per_1k=wh / imgs * 1000, cpu_energy_src="rapl_package")
    if sys_wh is not None:
        out.update(sys_wh_per_1k=sys_wh / imgs * 1000, sys_w=sys_wh * 3600 / t, sys_energy_src="codecarbon_1s")
    # synchronous per-call latency (the "one image arrives" deployment number)
    lat = []
    if batches[0].shape[0] == 1 and sync_iters > 0:
        for j in range(sync_iters):
            ts = time.perf_counter()
            runner.run_timed(batches[j % K])
            runner.sync()
            lat.append((time.perf_counter() - ts) * 1000)
        lat.sort()
        out["lat_ms_p50_sync"] = lat[len(lat) // 2]
        out["lat_ms_p90_sync"] = lat[int(len(lat) * 0.9)]
    else:
        out["lat_ms_p50_sync"] = out["lat_ms_p90_sync"] = ""
    return out


def idle_gpu_w(nvml, secs=2.0):
    if not nvml.ok:
        return ""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    s = PowerSampler(nvml)
    s.start()
    time.sleep(secs)
    vals = [p for _, p in s.stop() if p is not None]
    return statistics.fmean(vals) if vals else ""


DEPLOY_FIELDS = [
    "timestamp", "backbone", "parent_tag", "img_size", "mode", "device", "backend", "batch",
    "n_test", "n_val", "calib_size", "thr_opt",
    "test_auroc_img", "test_auroc_breast_max", "test_auroc_breast_mean",
    "acc_0.5", "mcc_0.5", "tpr_0.5", "tnr_0.5", "ppv_0.5", "npv_0.5",
    "acc_opt", "mcc_opt", "tpr_opt", "tnr_opt", "ppv_opt", "npv_opt",
    "size_mb", "img_per_s", "ms_per_img", "lat_ms_p50_sync", "lat_ms_p90_sync",
    "loop_s", "loop_imgs", "warmup_s",
    "gpu_w", "gpu_w_counter", "gpu_idle_w", "gpu_wh_per_1k", "gpu_energy_src",
    "cpu_w", "cpu_wh_per_1k", "cpu_energy_src",
    "sys_w", "sys_wh_per_1k", "sys_energy_src",
    "gpu_name", "gpu_driver", "cpu_name", "cpu_count", "cpu_threads", "cpu_vnni", "cpu_amx",
    "torch_version", "backend_version", "parent_auroc", "notes",
]


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DEPLOY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"  appended -> {path}")


def done_cells(path):
    """(backbone, parent_tag, img_size, mode, batch) cells already in the results csv."""
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d.backbone, d.parent_tag.fillna("").astype(str), d.img_size.astype(int), d["mode"], d.batch.astype(int)))


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in ALL_MODES]
    assert not bad, f"unknown modes {bad}; choose from {ALL_MODES}"
    batches = sorted({int(b) for b in args.batches.split(",")})
    max_batch = max(batches + [args.acc_batch])
    S = args.img_size
    if S is None:
        import re  # noqa: PLC0415
        m = re.search(r"_img(\d+)_", args.parent_tag or "")
        S = int(m.group(1)) if m else 768
    if args.png_index is None:
        args.png_index = args.folder + "png_index.csv"
    assert os.path.exists(args.png_index), f"{args.png_index} not found"
    assert os.path.exists(args.ckpt), f"{args.ckpt} not found"
    for d in (args.pred_dir, args.engine_dir, args.ir_dir):
        os.makedirs(d, exist_ok=True)
    gpu_needed = any(m in GPU_MODES for m in modes)
    if gpu_needed and not torch.cuda.is_available():
        raise SystemExit("GPU modes requested but torch.cuda.is_available() is False")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False      # torch_fp32 = honest fp32
    torch.backends.cuda.matmul.allow_tf32 = False

    nvml = Nvml()
    cpu_name, cpu_flags = cpu_info()
    env = {"gpu_name": nvml.name if nvml.ok else (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""),
           "gpu_driver": nvml.driver, "cpu_name": cpu_name, "cpu_count": os.cpu_count(),
           "cpu_vnni": cpu_flags["vnni"], "cpu_amx": cpu_flags["amx"], "torch_version": torch.__version__}
    print(f"env: {json.dumps(env)}")
    with open("env_deploy.json", "w") as f:
        json.dump({**env, "cpu_flags": cpu_flags, "python": platform.python_version()}, f, indent=1)

    df = pd.read_csv(args.png_index)
    train_df, val_df = split_trainval(df)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    calib_df = train_df.head(args.calib_size).reset_index(drop=True)
    if args.limit:
        test_df, val_df = test_df.head(args.limit), val_df.head(args.limit)
        print(f"DEBUG --limit {args.limit}")
    print(f"backbone {args.backbone}  ckpt {args.ckpt}  img {S}  test {len(test_df)}  val {len(val_df)}  calib {len(calib_df)}")
    test_u8 = decode_all(test_df, S, args.workers, args.folder, "test")
    val_u8 = decode_all(val_df, S, args.workers, args.folder, "val")
    calib_u8 = decode_all(calib_df, S, args.workers, args.folder, "calib") if any(m in ("trt_int8", "ov_int8") for m in modes) else None
    test_labels = test_df["label"].astype(int).to_numpy()
    val_labels = val_df["label"].astype(int).to_numpy()
    loop_u8 = test_u8[: args.loop_imgs]

    parent_auc = ""
    if args.parent_tag and os.path.exists(args.parent_results):
        pr = pd.read_csv(args.parent_results)
        m = pr[pr.tag == args.parent_tag]
        if not m.empty:
            parent_auc = float(m.iloc[0]["test_auroc_img"])
            if len(m) > 1:
                print(f"  [note] {len(m)} rows share --parent-tag; cross-checking against the first "
                      f"({m.iloc[0]['timestamp']}, AUROC {parent_auc:.4f}); others: "
                      + ", ".join(f"{r.timestamp} {r.test_auroc_img:.4f}" for r in m.iloc[1:].itertuples()))
    done = set() if args.force else done_cells(args.results_csv)
    model = load_model(args.backbone, args.ckpt)
    stem = f"{args.backbone}_img{S}"
    onnx_path = os.path.join(args.engine_dir, f"{stem}.onnx")

    for mode in modes:
        todo = [b for b in batches if (args.backbone, args.parent_tag, S, mode, b) not in done]
        if not todo:
            print(f"\n=== {mode}: all batches already in {args.results_csv} -- skip")
            continue
        print(f"\n=== {mode} ===")
        backend_version, notes, n_thr = "", "", ""
        if mode.startswith("torch"):
            backend_version = torch.__version__
            runner = TorchRunner(load_model(args.backbone, args.ckpt), half=(mode == "torch_fp16"))
        elif mode.startswith("trt"):
            import tensorrt as trt  # noqa: PLC0415
            backend_version = trt.__version__
            export_onnx(load_model(args.backbone, args.ckpt), S, onnx_path)
            eng = os.path.join(args.engine_dir, f"{stem}_{mode}_b{max_batch}.engine")
            data = build_trt_engine(onnx_path, eng, S, max_batch, int8=(mode == "trt_int8"), calib_u8=calib_u8,
                                    calib_cache=os.path.join(args.engine_dir, f"{stem}_int8.calib"))
            runner = TrtRunner(data, S, max_batch)
        else:
            ir = os.path.join(args.ir_dir, f"{stem}_{mode}.xml")
            compiled, size_mb, n_thr, backend_version = build_ov(load_model(args.backbone, args.ckpt), S, mode == "ov_int8",
                                                                  calib_u8, ir, args.cpu_threads)
            runner = OvRunner(compiled, size_mb)

        # A. accuracy (val threshold + test metrics), once per mode
        t0 = time.time()
        val_probs = accuracy_pass(runner, val_u8, S, args.acc_batch)
        fpr, tpr, thr = roc_curve(val_labels, val_probs)
        best_thr = float(thr[np.argmax(tpr - fpr)])
        if not np.isfinite(best_thr):  # degenerate val subset (only with --limit): fall back to 0.5
            best_thr = 0.5
        test_probs = accuracy_pass(runner, test_u8, S, args.acc_batch)
        acc = metrics_block(test_df, test_labels, test_probs, best_thr)
        print(f"  accuracy: AUROC {acc['test_auroc_img']:.4f}  MCC@0.5 {acc['mcc_0.5']:.3f}  MCC@opt {acc['mcc_opt']:.3f} "
              f"(thr {best_thr:.3f})  [{time.time() - t0:.0f}s]")
        if parent_auc != "" and not args.limit and abs(acc["test_auroc_img"] - parent_auc) > 0.03:
            notes = ("checkpoint may not match --parent-tag" if mode in ("torch_fp32", "ov_fp32")
                     else "quantization degraded accuracy")
            print(f"  [WARN] AUROC {acc['test_auroc_img']:.3f} vs parent {parent_auc:.3f} -- {notes}")
        out = test_df[["patient_id", "laterality", "view", "label"]].copy()
        out["prob"] = test_probs
        out.to_csv(os.path.join(args.pred_dir, f"pred_{stem}_{mode}.csv"), index=False)

        # B. steady state per batch size
        for B in todo:
            res = [runner.prepare(norm_batch(loop_u8[i:i + B], S)) for i in range(0, len(loop_u8) - B + 1, B)]
            idle = idle_gpu_w(nvml) if runner.dev == "cuda" else ""
            ss = steady_state(runner, res, args.warmup_secs, args.loop_secs, nvml, args.energy, args.sync_iters)
            gw = f"{ss['gpu_w']:.0f} W" if ss["gpu_w"] != "" else "n/a"
            cw = f"{ss['cpu_w']:.0f} W" if ss["cpu_w"] != "" else "n/a"
            sw = f"{ss['sys_wh_per_1k']:.3f} Wh/1k" if ss["sys_wh_per_1k"] != "" else "n/a"
            print(f"  batch {B:2d}: {ss['img_per_s']:8.1f} img/s  {ss['ms_per_img']:.3f} ms/img  gpu {gw}  cpu {cw}  sys {sw}")
            row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "backbone": args.backbone,
                   "parent_tag": args.parent_tag, "img_size": S, "mode": mode, "device": runner.dev,
                   "backend": BACKENDS[mode], "batch": B, "n_test": len(test_df), "n_val": len(val_df),
                   "calib_size": len(calib_df) if mode.endswith("int8") else "", **acc,
                   "size_mb": runner.size_mb, **ss, "warmup_s": args.warmup_secs, "gpu_idle_w": idle,
                   **env, "cpu_threads": n_thr, "backend_version": backend_version, "parent_auroc": parent_auc,
                   "notes": notes}
            append_row(args.results_csv, row)
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
