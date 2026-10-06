"""V2 -- layer-by-layer parity of the bare-TF backbones vs the timm reference activations.

    python parity_test_tf.py --backbone resnet18 [--pretrained-dir pretrained]
    python parity_test_tf.py --all

Compares every captured stage on the (2,224,224,3) and (2,250,250,3) reference inputs
(eval mode, fp32): max|delta| / max(1, max|ref|) < 1e-4, logits < 1e-3,
and the fp32 size_mb of the loaded npz. Also checks the training-mode BatchNorm running-stat
update against numpy (unbiased variance, momentum 0.1).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

from backbones_tf import build_backbone
from layers_tf import BatchNorm, Dense
from npzio import load_npz, size_mb

EXPECTED_MB = {"resnet18": 44.744448, "mobilenetv3_small_100": 6.119872, "convnext_small": 197.818752}


def check_backbone(name, pdir, tol=1e-4, tol_logits=1e-3):
    ref = load_npz(os.path.join(pdir, f"ref_{name}.npz"))
    meta = json.loads(str(ref["meta"]))
    flat = load_npz(os.path.join(pdir, f"{name}.npz"))
    bb = build_backbone(name)
    bb.load_flat(flat)
    mb = size_mb(flat)
    assert abs(mb - EXPECTED_MB[name]) < 1e-3, f"size_mb {mb} != {EXPECTED_MB[name]}"
    head = Dense(bb.feat_dim, 1000, name="fc")
    head.load_flat({"fc/kernel": ref["fc/kernel"], "fc/bias": ref["fc/bias"]}, prefix="fc/")
    worst = 0.0
    for s in (224, 250):
        x = tf.constant(ref[f"input_{s}"])
        t0 = time.time()
        outs = bb.forward_stages(x, tf.float32)
        logits = head(outs[list(outs)[-1]], tf.float32).numpy()
        dt = time.time() - t0
        for label in meta["stages"]:
            got, exp = outs[label].numpy(), ref[f"{s}/{label}"]
            if exp.size == got.size and exp.shape != got.shape:
                exp = exp.reshape(got.shape)  # timm hooks keep pooled maps as (B,1,1,C)
            assert got.shape == exp.shape, f"{name}@{s} {label}: shape {got.shape} != {exp.shape}"
            err = float(np.max(np.abs(got - exp)))
            scale = max(float(np.max(np.abs(exp))), 1.0)  # relative to the activation scale
            rel = err / scale
            worst = max(worst, rel)
            assert rel < tol, f"{name}@{s} {label}: max|delta|={err:.3e} (rel {rel:.3e}) > {tol}"
        lerr = float(np.max(np.abs(logits - ref[f"{s}/logits"])))
        assert lerr < tol_logits, f"{name}@{s} logits: max|delta|={lerr:.3e}"
        print(f"  {name}@{s}: {len(meta['stages'])} stages within {worst:.2e}, logits {lerr:.2e}  ({dt:.1f}s)")
    print(f"PARITY OK  {name}  ({meta['timm_tag']}, {mb:.6f} MB)")


def check_batchnorm():
    """Training-mode fused BN must normalise with the biased batch var and update running var
    with the UNBIASED one (torch semantics), momentum 0.1."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5, 5, 3)).astype(np.float32) * 2 + 1
    bn = BatchNorm(3)
    bn.load_flat({"bn/scale": np.array([1.5, 0.5, 2.0], np.float32), "bn/bias": np.array([0.1, -0.2, 0.3], np.float32),
                  "bn/mean": np.zeros(3, np.float32), "bn/var": np.ones(3, np.float32)}, prefix="bn/")
    y = bn(tf.constant(x), training=True).numpy()
    m, v = x.mean(axis=(0, 1, 2)), x.var(axis=(0, 1, 2))
    n = x.size / 3
    y_ref = (x - m) / np.sqrt(v + 1e-5) * np.array([1.5, 0.5, 2.0]) + np.array([0.1, -0.2, 0.3])
    assert np.max(np.abs(y - y_ref)) < 1e-4, "BN train-mode normalisation mismatch"
    rm, rv = bn.leaves["mean"].numpy(), bn.leaves["var"].numpy()
    assert np.max(np.abs(rm - 0.1 * m)) < 1e-5, f"running mean {rm} != {0.1 * m}"
    exp_rv = 0.9 * 1.0 + 0.1 * v * n / (n - 1)
    assert np.max(np.abs(rv - exp_rv)) < 1e-5, f"running var {rv} != {exp_rv} (unbiased expected)"
    y_eval = bn(tf.constant(x), training=False).numpy()
    y_eval_ref = (x - rm) / np.sqrt(rv + 1e-5) * np.array([1.5, 0.5, 2.0]) + np.array([0.1, -0.2, 0.3])
    assert np.max(np.abs(y_eval - y_eval_ref)) < 1e-4
    print("BATCHNORM OK (biased normalisation, unbiased running var, momentum 0.1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pretrained-dir", default="pretrained")
    a = ap.parse_args()
    tf.config.set_visible_devices([], "GPU")  # CPU is enough and deterministic
    check_batchnorm()
    names = list(EXPECTED_MB) if a.all else [a.backbone]
    if names == [None]:
        sys.exit("pass --backbone or --all")
    for n in names:
        check_backbone(n, a.pretrained_dir)


if __name__ == "__main__":
    main()
