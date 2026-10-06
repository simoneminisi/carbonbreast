"""Framework-neutral weight files: flat `/`-separated keys -> numpy arrays in one .npz.

Used for the timm-converted ImageNet weights (`pretrained/<backbone>.npz`), the Stage-1 trunks
(`trunk_<fw>_<backbone>.npz`, backbone subtree only) and the Stage-2 checkpoints
(`best_<tag>.npz`, backbone + head). Both ports load/save the SAME files.

Leaf vocabulary (flax names, so the JAX side needs no renaming):
    conv / dense      kernel, bias         conv (kH,kW,I,O)  depthwise (kH,kW,C,1)  dense (in,out)
    batch-norm        scale, bias, mean, var
    layer-norm        scale, bias
    layer-scale       gamma
Structural prefixes (see backbones_*.py): `backbone/...` and `head/dense/...`.
"""
import numpy as np


def save_npz(path, flat):
    np.savez(path, **{k: np.ascontiguousarray(v) for k, v in flat.items()})
    return path


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def subtree(flat, prefix, strip=True):
    """Entries whose key starts with `prefix`, optionally with the prefix stripped."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return {(k[len(prefix):] if strip else k): v for k, v in flat.items() if k.startswith(prefix)}


def add_prefix(flat, prefix):
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return {prefix + k: v for k, v in flat.items()}


def assert_same_keys(expected, got, what="weights"):
    """Strict both-way key check (the counterpart of the torch load_trunk assertion)."""
    expected, got = set(expected), set(got)
    missing, unexpected = sorted(expected - got), sorted(got - expected)
    if missing or unexpected:
        raise KeyError(f"{what}: key mismatch -- missing {len(missing)} {missing[:8]} | "
                       f"unexpected {len(unexpected)} {unexpected[:8]}")


def assert_same_shapes(expected, got, what="weights"):
    bad = [(k, tuple(expected[k]), tuple(got[k].shape)) for k in expected
           if tuple(expected[k]) != tuple(got[k].shape)]
    if bad:
        raise ValueError(f"{what}: shape mismatch on {len(bad)} keys, e.g. {bad[:5]}")


def cast_fp16(flat):
    """fp16 copy of every float array (integer arrays untouched) -- the torch `.half()` analog."""
    return {k: (v.astype(np.float16) if np.issubdtype(v.dtype, np.floating) else v)
            for k, v in flat.items()}


def size_mb(flat):
    """Total bytes / 1e6 (counterpart of torch state_dict_mb / keras weights_mb)."""
    return sum(int(v.nbytes) for v in flat.values()) / 1e6
