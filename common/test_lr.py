"""V3 -- per-step LR parity: the `where` formulation (lr_numpy, and optionally a framework
callable) must equal peak_lr * make_lr_factor(step) for every step of all 8 configs.

    python test_lr.py                      # numpy formulation only
    from test_lr import check_schedule     # ports call check_schedule(fn) with their tensor fn
"""
import numpy as np

from lr import WarmupCosineSpec, lr_numpy, make_lr_factor

# (peak_lr, warmup, epochs, n_train, batch) for the 7 stage-2 configs + stage-1 (13600 ~ train rows)
CONFIGS = [
    (1e-4, 3, 15, 13600, 48), (1e-4, 3, 15, 13600, 48), (1e-4, 3, 15, 13600, 32),
    (1e-4, 3, 15, 13600, 32), (1e-4, 3, 15, 13600, 16), (1e-4, 3, 15, 13600, 16),
    (1e-4, 5, 40, 13600, 8), (3e-4, 3, 30, 27376, 64),
]


def check_schedule(fn=None, tol=1e-9, verbose=True):
    """fn(spec, steps_np_int_array) -> np.ndarray of LRs (defaults to lr_numpy)."""
    fn = fn or lr_numpy
    worst = 0.0
    for peak, warm, ep, n, bs in CONFIGS:
        spe = max(n // bs, 1)
        spec = WarmupCosineSpec(peak, warm, ep, spe)
        factor = make_lr_factor(warm, ep, spe)
        steps = np.arange(0, spe * ep + 1, dtype=np.int64)
        ref = np.array([peak * factor(int(s)) for s in steps], dtype=np.float64)
        got = np.asarray(fn(spec, steps), dtype=np.float64)
        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)
        assert err < tol, f"LR mismatch {err:.3e} for config {(peak, warm, ep, n, bs)}"
        if verbose:
            print(f"  lr parity OK  peak={peak:g} warm={warm} ep={ep} spe={spe}  max|err|={err:.2e}")
    return worst


if __name__ == "__main__":
    w = check_schedule()
    print(f"LR SCHEDULE PARITY OK (worst {w:.2e})")
