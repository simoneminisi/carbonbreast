"""Shared data pipeline for the VinDr-Mammo ResNet-18 experiment.

Ports the ingredients that make experiments/vindr_multiview strong, adapted to our
8-bit processed PNGs (processed_dataset_1024, breast-crop + CLAHE already baked in):

  * per-image z-score normalization (not /255 + ImageNet mean/std) -- scale-invariant,
    mirrors vindr_multiview/data/augment.py:30-39;
  * stronger augmentation -- rotation +-25, shear +-12, scale [0.8,1.2], intensity +-20%,
    BORDER_REFLECT, random flip -- ported from vindr_multiview/vendored/img_process.py;
  * inverse-frequency WeightedRandomSampler -- copy of vindr_multiview/data/samplers.py.

Both the Stage-1 patch classifier and the Stage-2 full-image model import from here so the
normalization/augmentation is identical (required for a valid conv-trunk transfer).
"""
import contextlib
import math
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler

# Containers (Docker/k8s) often ship a tiny /dev/shm (64 MB), which the default DataLoader
# tensor-sharing (file_descriptor) exhausts -> "bus error / insufficient shared memory". Route
# worker->main batches through the filesystem (/tmp) instead so any --workers count is safe.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:  # noqa: BLE001
    pass

MAX = 255.0  # our processed PNGs are 8-bit
ANGLE, SHEAR, ZOOM_IN, ZOOM_OUT, INTENSITY = 25, 12, 0.20, 0.20, 0.20


# --- affine / flip primitives (ported from vindr_multiview/vendored/img_process.py) ---
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


def zscore_3ch(img, img_size):
    """uint8/float HxW -> normalized 3xHxW float32 tensor (per-image z-score)."""
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size))
    x = img.astype(np.float32) / MAX
    x -= x.mean()
    x /= max(x.std(), 1e-5)
    x = np.repeat(x[None, :, :], 3, axis=0)  # gray -> 3ch (CHW)
    return torch.from_numpy(np.ascontiguousarray(x))


def make_lr_factor(warmup_epochs, total_epochs, steps_per_epoch, peak_mult=1.0, floor_mult=0.02):
    """LambdaLR multiplier: linear warmup 0->peak over `warmup_epochs`, then cosine decay
    peak->floor over the rest. Set the optimizer base LR to the peak LR; ported/simplified
    from vindr_multiview/train/lr_schedule.py."""
    decay_epochs = max(total_epochs - warmup_epochs, 1)

    def factor(step):
        e = step / max(steps_per_epoch, 1)  # fractional epoch
        if e < warmup_epochs:
            return peak_mult * e / max(warmup_epochs, 1e-9)
        t = min((e - warmup_epochs) / decay_epochs, 1.0)  # 0..1
        return floor_mult + (peak_mult - floor_mult) * 0.5 * (1.0 + math.cos(math.pi * t))

    return factor


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
    """Refuse to allocate an in-RAM cache that would risk an OOM kill (this box has NO swap).
    Raises MemoryError unless the array fits within `headroom` of MemAvailable (or force=True)."""
    avail = _mem_available_bytes()
    gb = n_bytes / 1e9
    print(f"cache footprint: {gb:.1f} GB  |  MemAvailable: {avail / 1e9:.1f} GB")
    if not force and avail and n_bytes > headroom * avail:
        raise MemoryError(
            f"in-RAM cache needs {gb:.1f} GB but only {avail / 1e9:.1f} GB available "
            f"(no swap on this box). Close the IDE / other processes, cache train+val only "
            f"(drop --cache-test), lower --img-size, or pass --force-cache to override.")


def decode_all(df, img_size, workers=8, folder="", desc="cache"):
    """Decode every df['png_path'] to uint8 at img_size into ONE contiguous (N, S, S) array,
    row-aligned to df. A single array (not a list of arrays) is shared across DataLoader workers
    via fork copy-on-write, so it stays a single physical copy. cv2.imread releases the GIL, so a
    ThreadPool parallelizes the decode."""
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


def balanced_sampler(labels):
    """Inverse-frequency WeightedRandomSampler over integer class labels."""
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels)
    counts[counts == 0] = 1
    sample_w = (1.0 / counts)[labels]
    return WeightedRandomSampler(weights=sample_w.tolist(),
                                 num_samples=len(labels), replacement=True)


# --- backbone / trunk / optimizer (shared by train_patch.py Stage 1 and s_main.py Stage 2) ---
# The torchvision set keeps its exact key layout so the existing patch_trunk.pt (and prior results)
# stay valid; any other name is resolved by timm. Stage 1 and Stage 2 build backbones through the
# SAME function so the exported conv trunk transfers cleanly.
TORCHVISION_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0")


def _tv_head(in_features, num_classes, dropout):
    if dropout > 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def create_backbone(name, num_classes, dropout=0.0, pretrained=True):
    """Build an ImageNet-pretrained backbone with a `num_classes` head. torchvision for the native
    set (trunk-compatible), else timm.create_model."""
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = _tv_head(m.fc.in_features, num_classes, dropout)
    elif name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = _tv_head(m.fc.in_features, num_classes, dropout)
    elif name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        if dropout > 0:
            m.classifier[0].p = dropout  # efficientnet already has a dropout before the fc
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        import timm  # noqa: PLC0415 (optional dep; only needed for non-torchvision backbones)
        kw = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
        # transformers have fixed-resolution positional embeddings; dynamic_img_size interpolates
        # them so e.g. vit_small_patch16_224 runs at 768px. Conv backbones ignore/reject the kwarg.
        if any(t in name for t in ("vit", "deit", "beit", "eva", "swin")):
            kw["dynamic_img_size"] = True
        try:
            m = timm.create_model(name, **kw)
        except TypeError:
            kw.pop("dynamic_img_size", None)
            m = timm.create_model(name, **kw)
    return m


def get_head(model, name):
    """The classifier sub-module (for freezing / linear-probe). Uniform across torchvision + timm."""
    if name in ("resnet18", "resnet50"):
        return model.fc
    if name == "efficientnet_b0":
        return model.classifier
    return model.get_classifier()  # timm


def _head_key_names(model, name):
    """state_dict keys (params + buffers) belonging to the classifier head, by object identity --
    backbone-agnostic, so it works for any torchvision or timm architecture."""
    head = get_head(model, name)
    head_ids = {id(t) for t in list(head.parameters()) + list(head.buffers())}
    names = set()
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        if id(t) in head_ids:
            names.add(n)
    return names


def trunk_state_dict(model, name):
    """model.state_dict() minus the classifier head -> the transferable conv trunk."""
    head_names = _head_key_names(model, name)
    return {k: v for k, v in model.state_dict().items() if k not in head_names}


def load_trunk(model, path, name):
    """Load a trunk into `model` (strict=False). Asserts nothing unexpected and that only head keys
    are missing -- which also loudly rejects a backbone/trunk mismatch (e.g. resnet trunk -> convnext)."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    res = model.load_state_dict(sd, strict=False)
    head_names = _head_key_names(model, name)
    unexpected = list(res.unexpected_keys)
    missing = [k for k in res.missing_keys if k not in head_names]
    assert not unexpected and not missing, (
        f"trunk transfer mismatch for backbone '{name}': "
        f"unexpected={unexpected[:5]} missing(non-head)={missing[:5]} "
        f"-- is this trunk for a different backbone?")
    print(f"loaded patch-pretrained trunk from {path} ({len(sd)} tensors; only the head is random)")
    return len(sd)


def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                               momentum=momentum, nesterov=momentum > 0)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"unknown optimizer: {name}")


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


class PngDataset(Dataset):
    """Reads a dataframe with `png_path` + `label` columns. Used for both patches (224px)
    and full images (1024px): only img_size and the source df differ.

    If `cache` (a row-aligned uint8 (N, img_size, img_size) array from decode_all) is given,
    images are fetched from RAM instead of decoded from disk each access -- eliminating the
    ~3.7ms/img PNG decode. The cache is read-only and shared across workers via copy-on-write."""

    def __init__(self, df, folder="", img_size=1024, train=False, cache=None):
        self.df = df.reset_index(drop=True)
        self.folder = folder
        self.img_size = img_size
        self.train = train
        self.cache = cache
        assert cache is None or len(cache) == len(self.df), "cache must be row-aligned to df"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache is not None:
            u8 = self.cache[idx]
        else:
            u8 = cv2.imread(self.folder + self.df.iloc[idx]["png_path"], cv2.IMREAD_GRAYSCALE)
        if self.train:
            u8 = augment(u8)
        x = zscore_3ch(u8, self.img_size)
        return x, int(self.df.iloc[idx]["label"])
