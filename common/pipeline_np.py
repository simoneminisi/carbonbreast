"""Framework-free input pipeline for the VinDr-Mammo experiment (numpy + cv2 + multiprocessing).

Shared VERBATIM by the bare-TensorFlow and bare-JAX ports, so the two frameworks differ in
nothing but the framework:

  * per-image z-score normalization (not /255 + ImageNet mean/std);
  * identical cv2 augmentation -- rotation +-25, shear +-12, scale [0.8,1.2], intensity +-20%,
    BORDER_REFLECT, random flip (copied verbatim from pipeline.py / pipeline_keras.py);
  * inverse-frequency balanced sampling with replacement (WeightedRandomSampler semantics),
    len(df) draws per epoch, drop_last;
  * a forked worker pool doing all cv2 work; batches are assembled directly into a ring of
    shared-memory float32 slots (no tf.data, no per-sample Python round-trips).

Contract for consumers: the array yielded by a loader is valid until the NEXT item is
requested -- copy it to the device (tf.constant / jax.device_put) before advancing.
The pool must be started BEFORE the training framework initializes its GPU (fork safety);
`WorkerPool.start()` is the explicit hook for that, and loaders auto-start it otherwise.
"""
import ctypes
import math
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# cv2 must never spawn its internal thread pool in the parent: forked workers would inherit a
# phantom pool and deadlock on their first parallel_for (imread/resize/warpAffine). All
# parallelism here comes from threads (decode) and processes (loaders), one cv2 call at a time.
cv2.setNumThreads(0)

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


# --- split / subset helpers (moved here from s_main so both ports share them) ---
def split_trainval(df, val_frac=0.15, seed=42):
    """Patient-grouped val split of the official training split. NOTE: the split is fixed at
    random_state=42 regardless of --seed, exactly like the torch and Keras runs."""
    train = df[df["split"] == "train"]
    tr_idx, va_idx = next(
        GroupShuffleSplit(1, test_size=val_frac, random_state=seed).split(train, groups=train["patient_id"]))
    return train.iloc[tr_idx], train.iloc[va_idx]


def limit_df(d, n):
    """Stratified head(n) so tiny smoke-test splits still contain both classes."""
    per = max(1, n // max(d["label"].nunique(), 1))
    return d.groupby("label", group_keys=False).head(per).head(n).reset_index(drop=True)


def balanced_probs(labels):
    """Inverse-class-frequency sampling distribution (WeightedRandomSampler semantics)."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels)
    counts[counts == 0] = 1
    w = (1.0 / counts)[labels]
    return w / w.sum()


# --- RAM cache (verbatim) ---
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
    row-aligned to df. cv2.imread releases the GIL, so a ThreadPool parallelizes the decode.
    (The thread pool is joined before any worker process is forked.)"""
    n = len(df)
    arr = np.empty((n, img_size, img_size), dtype=np.uint8)
    paths = df["png_path"].tolist()

    def load(i):
        u8 = cv2.imread(folder + paths[i], cv2.IMREAD_GRAYSCALE)
        if u8.shape[0] != img_size or u8.shape[1] != img_size:
            u8 = cv2.resize(u8, (img_size, img_size))
        arr[i] = u8

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
        list(ex.map(load, range(n)))
    print(f"{desc}: decoded {n} imgs @ {img_size}px into RAM ({arr.nbytes / 1e9:.1f} GB) "
          f"in {time.time() - t0:.0f}s")
    return arr


def loader_buffer_bytes(workers, prefetch, batch, img_size, n_loaders=3):
    """Ring-buffer footprint of the loaders, for the RAM guard (float32 NHWC slots)."""
    slots = workers + prefetch + 1
    return n_loaders * slots * batch * img_size * img_size * 3 * 4


# --- worker pool + shared-memory ring loaders ---
# Loader state lives in this module-level registry so forked workers inherit it (copy-on-
# write for the uint8 caches, genuinely shared for the RawArray ring slots).
_SHARED = {}


def _worker_init():
    cv2.setNumThreads(0)  # keep cv2 single-threaded inside each worker as well


def _work_batch(sid, idx, seed, slot):
    """Fill ring slot `slot` of loader `sid` with the z-scored (augmented) images of rows `idx`.
    np.random is re-seeded per batch so results do not depend on which worker ran it."""
    st = _SHARED[sid]
    out = st["ring"][slot]
    cache, paths, folder, img_size, train = st["cache"], st["paths"], st["folder"], st["img_size"], st["train"]
    if train:
        np.random.seed(seed)
    for j, i in enumerate(idx):
        i = int(i)
        u8 = cache[i] if cache is not None else cv2.imread(folder + paths[i], cv2.IMREAD_GRAYSCALE)
        if train:
            u8 = augment(u8)
        out[j] = zscore_3ch_hwc(u8, img_size)
    return slot


class WorkerPool:
    """Forked process pool shared by all loaders of a run. `workers=0` runs batches inline
    (same code path, no processes) for debugging and tiny smoke tests."""

    def __init__(self, workers):
        self.workers = int(workers)
        self._pool = None
        self._started = False

    def is_started(self):
        return self._started

    def start(self):
        if self._started:
            return self
        if self.workers > 0:
            ctx = mp.get_context("fork")
            self._pool = ctx.Pool(self.workers, initializer=_worker_init)
        self._started = True
        return self

    def submit(self, sid, idx, seed, slot):
        if not self._started:
            print("[loader] WorkerPool auto-started at first batch (start() it before GPU init)")
            self.start()
        if self._pool is None:
            return _InlineResult(_work_batch(sid, idx, seed, slot))
        return self._pool.apply_async(_work_batch, (sid, idx, seed, slot))

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        self._started = False


class _InlineResult:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


class _RingLoader:
    """Base: registers its state, owns a ring of `slots` shared float32 (B,S,S,3) buffers."""

    def __init__(self, df, img_size, batch, folder, cache, pool, train, prefetch):
        assert cache is None or len(cache) == len(df), "cache must be row-aligned to df"
        if pool.is_started():
            raise RuntimeError("construct every loader BEFORE WorkerPool.start(): forked workers "
                               "only see loader state that existed at fork time")
        self.n = len(df)
        self.img_size, self.batch, self.pool, self.train = int(img_size), int(batch), pool, train
        self.labels = df["label"].astype(int).to_numpy().astype(np.int32)
        self.slots = max(pool.workers, 1) + int(prefetch) + 1
        n_float = self.slots * self.batch * self.img_size * self.img_size * 3
        raw = mp.RawArray(ctypes.c_float, n_float)  # anonymous shared mmap, inherited on fork
        self.ring = np.frombuffer(raw, dtype=np.float32).reshape(
            self.slots, self.batch, self.img_size, self.img_size, 3)
        self._raw = raw
        self.sid = f"{id(self)}"
        _SHARED[self.sid] = {"ring": self.ring, "cache": cache, "paths": df["png_path"].tolist(),
                             "folder": folder, "img_size": self.img_size, "train": train}

    def _run(self, index_batches, seeds):
        """Dispatch every (indices, seed) pair to the pool, keep `slots-1` in flight, yield in order."""
        pending = deque()
        free = deque(range(self.slots))
        it = iter(zip(index_batches, seeds))
        exhausted = False

        def dispatch():
            nonlocal exhausted
            try:
                idx, seed = next(it)
            except StopIteration:
                exhausted = True
                return
            slot = free.popleft()
            pending.append((self.pool.submit(self.sid, idx, seed, slot), idx, slot))

        while not exhausted and len(pending) < self.slots - 1:
            dispatch()
        while pending:
            ar, idx, slot = pending.popleft()
            ar.get()
            yield self.ring[slot], idx
            free.append(slot)  # consumer has moved on: the slot may be overwritten now
            if not exhausted:
                dispatch()


class TrainLoader(_RingLoader):
    """Infinite balanced index stream, `len(df)//batch` batches per epoch (drop_last).

    One `np.random.default_rng(seed)` generator draws `rng.choice(n, n, replace=True, p)` chunks
    forever (exactly the Keras `index_stream`; the remainder of a chunk carries into the next
    epoch and across rounds). A second generator (seed+1) draws one augmentation seed per batch.
    """

    def __init__(self, df, img_size, batch, seed, folder="", cache=None, pool=None, prefetch=4):
        super().__init__(df, img_size, batch, folder, cache, pool, train=True, prefetch=prefetch)
        self.steps = max(self.n // self.batch, 1)
        self.p = balanced_probs(self.labels)
        self.rng = np.random.default_rng(seed)
        self.aug_rng = np.random.default_rng(seed + 1)
        self._chunk = deque()

    def __len__(self):
        return self.steps

    def _next_indices(self):
        while len(self._chunk) < self.batch:
            self._chunk.extend(self.rng.choice(self.n, size=self.n, replace=True, p=self.p))
        return [self._chunk.popleft() for _ in range(self.batch)]

    def __iter__(self):
        idx_batches = [self._next_indices() for _ in range(self.steps)]
        seeds = [int(self.aug_rng.integers(0, 2 ** 31 - 1)) for _ in range(self.steps)]
        for x, idx in self._run(idx_batches, seeds):
            yield x, self.labels[idx]


class EvalLoader(_RingLoader):
    """Order-preserving, FIXED-SHAPE batches: yields (x, n_valid); the last batch is padded by
    repeating the last row so tf.function / jax.jit trace exactly once. Slice `[:n_valid]`."""

    def __init__(self, df, img_size, batch, folder="", cache=None, pool=None, prefetch=2):
        super().__init__(df, img_size, batch, folder, cache, pool, train=False, prefetch=prefetch)
        self.steps = int(math.ceil(self.n / self.batch)) if self.n else 0

    def __len__(self):
        return self.steps

    def __iter__(self):
        batches, valid = [], []
        for s in range(self.steps):
            idx = list(range(s * self.batch, min((s + 1) * self.batch, self.n)))
            valid.append(len(idx))
            idx = idx + [idx[-1]] * (self.batch - len(idx))
            batches.append(idx)
        for (x, _), nv in zip(self._run(batches, [0] * self.steps), valid):
            yield x, nv


def eval_probs(loader, predict_logits, softmax):
    """Run `predict_logits(x) -> (B, C) logits` over an EvalLoader; returns prob of class 1
    for every row in order (or the full softmax when C > 2 via `softmax`)."""
    outs = []
    for x, nv in loader:
        outs.append(np.asarray(predict_logits(x), dtype=np.float32)[:nv])
    logits = np.concatenate(outs, axis=0) if outs else np.zeros((0, 2), np.float32)
    return softmax(logits)
