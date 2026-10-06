"""V3 -- one AdamW step of the bare-TF optimizer vs the closed form, plus LR-schedule parity.

    python test_optim_tf.py
"""
import numpy as np
import tensorflow as tf

from lr import WarmupCosineSpec
from test_lr import check_schedule
from train_tf import Optimizer, lr_numpy_tf


def main():
    tf.config.set_visible_devices([], "GPU")
    rng = np.random.default_rng(0)
    w0 = rng.standard_normal((5, 3)).astype(np.float32)
    g = rng.standard_normal((5, 3)).astype(np.float32)
    v = tf.Variable(w0)
    spec = WarmupCosineSpec(1e-3, 0, 10, 10)  # no warmup -> lr(step 0) = peak
    opt = Optimizer([v], "adamw", spec, weight_decay=0.05)
    lr = float(opt.lr())
    assert abs(lr - 1e-3) < 1e-9  # fp32 schedule value
    opt.apply([tf.constant(g)])
    opt.step_schedule()
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = (1 - b1) * g
    s = (1 - b2) * g * g
    m_hat, v_hat = m / (1 - b1), s / (1 - b2)
    expected = w0 - lr * (m_hat / (np.sqrt(v_hat) + eps) + 0.05 * w0)
    err = float(np.max(np.abs(v.numpy() - expected)))
    assert err < 1e-6, f"AdamW step mismatch {err:.3e}"  # fp32: ~1 ulp on O(1) weights
    assert int(opt.iterations) == 1 and int(opt.adam_t) == 1
    print(f"ADAMW OK (max|err| {err:.2e}; decoupled lr-scaled wd; iterations/adam_t = 1/1)")
    worst = check_schedule(lr_numpy_tf, tol=1e-9, verbose=False)
    print(f"LR PARITY OK (tf_lr worst err {worst:.2e})")


if __name__ == "__main__":
    main()
