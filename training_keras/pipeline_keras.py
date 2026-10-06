"""Shared data pipeline for the VinDr-Mammo experiment — Keras 3 / TensorFlow port.

Faithful port of ../training_pytorch/pipeline.py so the Keras runs are comparable
with the PyTorch results row-for-row:

  * per-image z-score normalization (not /255 + ImageNet mean/std);
  * identical cv2 augmentation -- rotation +-25, shear +-12, scale [0.8,1.2],
    intensity +-20%, BORDER_REFLECT, random flip (copied verbatim);
  * inverse-frequency balanced sampling with replacement (WeightedRandomSampler semantics);
  * per-step warmup+cosine LR schedule (same curve as make_lr_factor);
  * CodeCarbon track_energy (copied verbatim).

Both train_patch_keras.py (Stage 1) and s_main_keras.py (Stage 2) build backbones through
the SAME build_backbone(), which is what makes the exported trunk weights transferable.
"""
import contextlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # export KERAS_BACKEND=jax for the JAX runs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if os.environ["KERAS_BACKEND"] == "jax":
    # JAX analog of TF memory growth: no 75%-of-VRAM preallocation, so OOMs are honest and
    # codecarbon/pynvml sees real usage. Must be set before jax initializes.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import tensorflow as tf  # noqa: E402  (tf.data input pipeline on every backend)
import keras  # noqa: E402
from keras import layers  # noqa: E402

MAX = 255.0  # our processed PNGs are 8-bit
ANGLE, SHEAR, ZOOM_IN, ZOOM_OUT, INTENSITY = 25, 12, 0.20, 0.20, 0.20


# --- affine / flip primitives (verbatim from pipeline.py) ---
def _affine_matrix(center, angle, scale, shear):
    angle = math.radians(angle)
    shear = math.radians(shear)
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
    RSS = np.array([[math.cos(angle) * scale, -math.sin(angle + shear) * scale, 0],
                    [math.sin(angle) * scale, math.cos(angle + shear) * scale, 0],
                    [0, 0, 1]])
    M = C @ RSS @ np.linalg.inv(C)
    return M[:2, :]


def augment(u8):
    """Geometric + intensity augmentation on a uint8 HxW image (official VinDr style)."""
    beta = INTENSITY * np.random.uniform(-1, 1)
    img = cv2.add(u8.astype(np.float32), beta * MAX)  # additive brightness jitter

    angle = np.random.randint(-ANGLE, ANGLE + 1)
    scale = 1 + np.random.uniform(0, ZOOM_OUT) if np.random.randint(0, 2) == 0 \
        else 1 - np.random.uniform(0, ZOOM_IN)
    shear = np.random.randint(-SHEAR, SHEAR + 1)
    h, w = img.shape
    M = _affine_matrix((w * 0.5 + 0.5, h * 0.5 + 0.5), angle, scale, shear)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    opt = np.random.randint(0, 4)  # {vertical, horizontal, both, none}
    if opt == 0:
        img = np.flip(img, 0)
    elif opt == 1:
        img = np.flip(img, 1)
    elif opt == 2:
        img = np.flip(np.flip(img, 0), 1)
    return np.ascontiguousarray(np.clip(img, 0, MAX))


def zscore_3ch_hwc(img, img_size):
    """uint8/float HxW -> normalized HxWx3 float32 (per-image z-score, channels-last)."""
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size))
    x = img.astype(np.float32) / MAX
    x -= x.mean()
    x /= max(x.std(), 1e-5)
    return np.ascontiguousarray(np.repeat(x[:, :, None], 3, axis=2))


def make_lr_factor(warmup_epochs, total_epochs, steps_per_epoch, peak_mult=1.0, floor_mult=0.02):
    """Reference python multiplier (verbatim from pipeline.py) -- kept for the parity
    unit-test against WarmupCosine below."""
    decay_epochs = max(total_epochs - warmup_epochs, 1)

    def factor(step):
        e = step / max(steps_per_epoch, 1)  # fractional epoch
        if e < warmup_epochs:
            return peak_mult * e / max(warmup_epochs, 1e-9)
        t = min((e - warmup_epochs) / decay_epochs, 1.0)  # 0..1
        return floor_mult + (peak_mult - floor_mult) * 0.5 * (1.0 + math.cos(math.pi * t))

    return factor


class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Per-step LR: linear warmup 0->peak over warmup_epochs, then cosine peak->0.02*peak.
    Same curve as make_lr_factor() * peak_lr (stepped per batch, like the torch LambdaLR)."""

    def __init__(self, peak_lr, warmup_epochs, total_epochs, steps_per_epoch, floor_mult=0.02):
        self.peak_lr = float(peak_lr)
        self.warmup_epochs = float(warmup_epochs)
        self.total_epochs = float(total_epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.floor_mult = float(floor_mult)
        self.decay_epochs = max(self.total_epochs - self.warmup_epochs, 1.0)

    def __call__(self, step):
        from keras import ops
        e = ops.cast(step, "float32") / max(self.steps_per_epoch, 1)
        warm = e / max(self.warmup_epochs, 1e-9)
        t = ops.minimum((e - self.warmup_epochs) / self.decay_epochs, 1.0)
        cos = self.floor_mult + (1.0 - self.floor_mult) * 0.5 * (1.0 + ops.cos(math.pi * t))
        return self.peak_lr * ops.where(e < self.warmup_epochs, warm, cos)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_epochs": self.warmup_epochs,
                "total_epochs": self.total_epochs, "steps_per_epoch": self.steps_per_epoch,
                "floor_mult": self.floor_mult}


def _mem_available_bytes():
    """Bytes of MemAvailable from /proc/meminfo (0 if unreadable)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except OSError:
        pass
    return 0


def assert_ram_fits(n_bytes, headroom=0.85, force=False):
    """Refuse to allocate an in-RAM cache that would risk an OOM kill.
    Raises MemoryError unless the array fits within `headroom` of MemAvailable (or force=True)."""
    avail = _mem_available_bytes()
    gb = n_bytes / 1e9
    print(f"cache footprint: {gb:.1f} GB  |  MemAvailable: {avail / 1e9:.1f} GB")
    if not force and avail and n_bytes > headroom * avail:
        raise MemoryError(
            f"in-RAM cache needs {gb:.1f} GB but only {avail / 1e9:.1f} GB available. "
            f"Close other processes, cache train+val only (drop --cache-test), lower "
            f"--img-size, or pass --force-cache to override.")


def decode_all(df, img_size, workers=8, folder="", desc="cache"):
    """Decode every df['png_path'] to uint8 at img_size into ONE contiguous (N, S, S) array,
    row-aligned to df. cv2.imread releases the GIL, so a ThreadPool parallelizes the decode."""
    n = len(df)
    arr = np.empty((n, img_size, img_size), dtype=np.uint8)
    paths = df["png_path"].tolist()

    def load(i):
        u8 = cv2.imread(folder + paths[i], cv2.IMREAD_GRAYSCALE)
        if u8.shape[0] != img_size or u8.shape[1] != img_size:
            u8 = cv2.resize(u8, (img_size, img_size))
        arr[i] = u8

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(load, range(n)))
    print(f"{desc}: decoded {n} imgs @ {img_size}px into RAM ({arr.nbytes / 1e9:.1f} GB) "
          f"in {time.time() - t0:.0f}s")
    return arr


@contextlib.contextmanager
def track_energy(label, enabled=True):
    """Report energy (Wh) + CO2 (g) via CodeCarbon and wall-time (s, measured independently).
    Yields a dict filled on exit with keys wh / co2_g / seconds. Degrades to wall-time only if
    CodeCarbon is absent or errors (e.g. offline box), so a run never crashes."""
    out = {"wh": None, "co2_g": None, "seconds": None}
    tracker = None
    if enabled:
        try:
            from codecarbon import EmissionsTracker  # noqa: PLC0415 (optional dep)
            tracker = EmissionsTracker(log_level="error", save_to_file=False)
            tracker.start()
        except Exception as e:  # noqa: BLE001
            print(f"[energy] CodeCarbon unavailable ({type(e).__name__}: {e}); wall-time only")
    t0 = time.time()
    try:
        yield out
    finally:
        out["seconds"] = time.time() - t0
        if tracker is not None:
            try:
                co2_kg = tracker.stop()
                out["co2_g"] = (co2_kg or 0.0) * 1000.0
                out["wh"] = tracker._total_energy.kWh * 1000.0
            except Exception as e:  # noqa: BLE001
                print(f"[energy] CodeCarbon stop failed ({type(e).__name__}: {e}); wall-time only")
        wh = f"{out['wh']:.2f}" if out["wh"] is not None else "n/a"
        co2 = f"{out['co2_g']:.3f}" if out["co2_g"] is not None else "n/a"
        print(f"[energy] {label}: {wh} Wh  {co2} gCO2  {out['seconds']:.1f} s")


def energy_sidecar_path(path):
    """trunk_keras_x.weights.h5 -> trunk_keras_x.energy.json (splitext would only strip .h5)."""
    if path.endswith(".weights.h5"):
        return path[:-len(".weights.h5")] + ".energy.json"
    return os.path.splitext(path)[0] + ".energy.json"


def setup_gpu():
    """Backend-aware GPU setup. Returns True if the TRAINING backend sees a GPU.

    tensorflow: enable memory growth (plays nicer with CodeCarbon/pynvml, honest OOMs).
    jax: report jax.devices(); ALSO hide any GPU from TF so the tf.data input pipeline
    stays on CPU even if a full (non -cpu) tensorflow happens to be installed."""
    if keras.backend.backend() == "jax":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception as e:  # noqa: BLE001
            print(f"[gpu] could not hide GPUs from tf.data: {e}")
        import jax  # noqa: PLC0415
        gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda", "rocm")]
        print(f"[gpu] jax devices: {jax.devices()}")
        return len(gpus) > 0
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:  # noqa: BLE001
            print(f"[gpu] set_memory_growth failed on {g}: {e}")
    return len(gpus) > 0


# --- tf.data pipeline (mirrors PngDataset + WeightedRandomSampler + DataLoader) ---
def _make_fetch(df, folder, img_size, train, cache):
    """Row-index -> (HWC float32, int32 label). Reads the RAM cache when given, else decodes
    from disk at access time, exactly like PngDataset.__getitem__."""
    paths = df["png_path"].tolist()
    labels = df["label"].astype(int).to_numpy()
    assert cache is None or len(cache) == len(df), "cache must be row-aligned to df"

    def fetch(i):
        i = int(i)
        if cache is not None:
            u8 = cache[i]
        else:
            u8 = cv2.imread(folder + paths[i], cv2.IMREAD_GRAYSCALE)
        if train:
            u8 = augment(u8)
        return zscore_3ch_hwc(u8, img_size), np.int32(labels[i])

    return fetch


def make_train_ds(df, img_size, batch, seed, folder="", cache=None):
    """Infinite balanced training dataset. Index stream = inverse-class-frequency draws with
    replacement, len(df) per epoch (WeightedRandomSampler semantics); batch(drop_remainder=True)
    + steps_per_epoch=len(df)//batch reproduces drop_last=True."""
    labels = df["label"].astype(int).to_numpy()
    fetch = _make_fetch(df, folder, img_size, train=True, cache=cache)

    def index_stream():
        counts = np.bincount(labels)
        counts[counts == 0] = 1
        w = (1.0 / counts)[labels]
        p = w / w.sum()
        rng = np.random.default_rng(seed)
        n = len(labels)
        while True:
            yield from rng.choice(n, size=n, replace=True, p=p)

    ds = tf.data.Dataset.from_generator(
        index_stream, output_signature=tf.TensorSpec(shape=(), dtype=tf.int64))
    ds = ds.map(lambda i: tf.numpy_function(fetch, [i], (tf.float32, tf.int32)),
                num_parallel_calls=tf.data.AUTOTUNE)

    def set_shapes(x, y):
        x.set_shape((img_size, img_size, 3))
        y.set_shape(())
        return x, y

    return ds.map(set_shapes).batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def make_eval_ds(df, img_size, batch, folder="", cache=None):
    """Order-preserving eval dataset yielding images only (labels come from the df).
    tf.data map is deterministic by default, so predictions stay row-aligned to df."""
    fetch = _make_fetch(df, folder, img_size, train=False, cache=cache)

    def fetch_x(i):
        return fetch(i)[0]

    ds = tf.data.Dataset.from_tensor_slices(np.arange(len(df), dtype=np.int64))
    ds = ds.map(lambda i: tf.numpy_function(fetch_x, [i], tf.float32),
                num_parallel_calls=tf.data.AUTOTUNE)

    def set_shape(x):
        x.set_shape((img_size, img_size, 3))
        return x

    return ds.map(set_shape).batch(batch).prefetch(tf.data.AUTOTUNE)


# --- backbone / trunk / optimizer (shared by Stage 1 and Stage 2) ---
KERAS_BACKBONES = ("resnet18", "mobilenetv3_small_100", "convnext_small")


def _resnet18_trunk(img_size):
    """ImageNet resnet18 via KerasHub (keras.applications has no resnet18)."""
    import keras_hub  # noqa: PLC0415
    preset = "resnet_18_imagenet"
    presets = getattr(keras_hub.models.ResNetBackbone, "presets", {})
    if presets and preset not in presets:
        cands = sorted(p for p in presets if "18" in p and "imagenet" in p)
        assert cands, (f"no resnet18 imagenet preset in this keras_hub "
                       f"(available: {sorted(presets)})")
        preset = cands[0]
        print(f"[backbone] resnet_18_imagenet not found, using preset: {preset}")
    bb = keras_hub.models.ResNetBackbone.from_preset(preset)
    inp = keras.Input((img_size, img_size, 3), name="trunk_in")
    x = bb(inp)
    if isinstance(x, dict):  # some KerasHub backbones return a feature pyramid
        x = x[sorted(x.keys())[-1]]
    x = layers.GlobalAveragePooling2D(name="trunk_pool")(x)
    return keras.Model(inp, x, name="resnet18_trunk")


def build_backbone(name, img_size):
    """ImageNet-pretrained feature extractor: (S,S,3) z-scored input -> (features,) pooled.
    include_preprocessing=False is MANDATORY for the applications models -- their built-in
    rescaling would destroy the per-image z-scored input."""
    if name == "resnet18":
        return _resnet18_trunk(img_size)
    if name == "mobilenetv3_small_100":  # = keras MobileNetV3Small alpha=1.0
        bb = keras.applications.MobileNetV3Small(
            input_shape=(img_size, img_size, 3), alpha=1.0, minimalistic=False,
            include_top=False, weights="imagenet", pooling="avg",
            include_preprocessing=False)
        return keras.Model(bb.input, bb.output, name="mobilenetv3_small_100_trunk")
    if name == "convnext_small":
        try:
            bb = keras.applications.ConvNeXtSmall(
                input_shape=(img_size, img_size, 3), include_top=False,
                weights="imagenet", pooling="avg", include_preprocessing=False)
        except TypeError:  # older keras without the kwarg: its preprocessing is identity-safe
            print("[backbone] ConvNeXtSmall has no include_preprocessing kwarg; "
                  "building without it -- VERIFY no Normalization layer is applied!")
            bb = keras.applications.ConvNeXtSmall(
                input_shape=(img_size, img_size, 3), include_top=False,
                weights="imagenet", pooling="avg")
        # timm keeps a LayerNorm between pooling and the classifier (head.norm) and the torch
        # pipeline shipped it inside the trunk; keras only bundles it with include_top=True,
        # so append a fresh one here. It is random at first but Stage-1 trains it before export.
        x = layers.LayerNormalization(epsilon=1e-6, name="head_layernorm")(bb.output)
        return keras.Model(bb.input, x, name="convnext_small_trunk")
    raise ValueError(f"unknown backbone: {name} (keras port supports {KERAS_BACKBONES})")


def build_model(name, num_classes, img_size, dropout=0.0, trunk_path=None):
    """backbone -> Dropout -> Dense(num_classes) logits (fp32 head under mixed precision).
    Returns (model, backbone); the backbone sub-model is what trunk save/load operates on.
    load_weights hard-fails on a backbone/trunk mismatch (same guarantee as torch load_trunk)."""
    backbone = build_backbone(name, img_size)
    if trunk_path:
        backbone.load_weights(trunk_path)
        print(f"loaded patch-pretrained trunk from {trunk_path} (only the head is random)")
    inp = keras.Input((img_size, img_size, 3), name="image")
    x = backbone(inp)
    if dropout > 0:
        x = layers.Dropout(dropout, name="head_dropout")(x)
    out = layers.Dense(num_classes, dtype="float32", name="head")(x)
    return keras.Model(inp, out, name=f"{name}_classifier"), backbone


def build_optimizer(name, learning_rate, weight_decay=0.0, momentum=0.9):
    """Keras AdamW uses decoupled, lr-scaled weight decay on ALL params -- same as the torch
    AdamW config (which excluded nothing). Note: keras Adam's weight_decay is also decoupled,
    unlike torch Adam's L2 -- only adamw is used by the pareto runs."""
    name = name.lower()
    wd = weight_decay if weight_decay > 0 else None
    if name == "adamw":
        return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=wd)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=wd)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, weight_decay=wd,
                                    momentum=momentum, nesterov=momentum > 0)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate, weight_decay=wd,
                                        momentum=momentum)
    raise ValueError(f"unknown optimizer: {name}")


def weights_mb(model):
    """Total variable bytes in MB (counterpart of torch state_dict_mb; includes BN stats)."""
    total = 0
    for w in model.weights:
        total += int(np.prod(w.shape)) * np.dtype(keras.backend.standardize_dtype(w.dtype)).itemsize
    return total / 1e6


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
