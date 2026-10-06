"""Unit checks of the bare-JAX training core:
  * one AdamW step from a fresh state vs the closed form  w - lr*(m_hat/(sqrt(v_hat)+eps) + wd*w)
    (with m_hat = g, v_hat = g^2 after one step), within 1e-7;
  * one nesterov-SGD step vs  w - lr*((g + mom*g) + wd*w);
  * per-step LR parity of lr_numpy_jax against the python reference (test_lr.check_schedule).

    python test_optax_adamw.py            # default device;  JAX_PLATFORMS=cpu python test_optax_adamw.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax

import train_jax as T
from test_lr import check_schedule


def _closed_form_adamw(w, g, lr, wd, b1=0.9, b2=0.999, eps=1e-8):
    m = (1 - b1) * g
    v = (1 - b2) * g * g
    m_hat, v_hat = m / (1 - b1), v / (1 - b2)
    return w - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * w)


def check_adamw(lr=1e-3, wd=0.05, tol=1e-7):
    rng = np.random.default_rng(0)
    w = {"backbone": {"k": rng.standard_normal((4, 3)).astype(np.float32)},
         "head": {"b": rng.standard_normal((3,)).astype(np.float32)}}
    g = jax.tree_util.tree_map(lambda a: rng.standard_normal(a.shape).astype(np.float32), w)
    params = jax.tree_util.tree_map(jnp.asarray, w)
    grads = jax.tree_util.tree_map(jnp.asarray, g)
    tx = T.make_optimizer("adamw", wd)
    st = T.with_lr(tx.init(params), lr)
    updates, st2 = tx.update(grads, st, params)
    new = jax.tree_util.tree_map(np.asarray, optax.apply_updates(params, updates))
    ref = jax.tree_util.tree_map(lambda a, b: _closed_form_adamw(a.astype(np.float64), b.astype(np.float64), lr, wd), w, g)
    err = max(float(np.max(np.abs(n - r))) for n, r in zip(jax.tree_util.tree_leaves(new), jax.tree_util.tree_leaves(ref)))
    assert err < tol, f"adamw step mismatch {err:.3e}"
    assert int(st2.inner_state[0].count) == 1, "adam count must be 1 after one applied step"
    print(f"  adamw one-step vs closed form: max|err|={err:.2e}  OK")

    # masked (head-only) variant: backbone untouched, head identical to the unmasked step
    txm = T.make_optimizer("adamw", wd, mask=T.head_mask(params))
    stm = T.with_lr(txm.init(params), lr)
    um, _ = txm.update(grads, stm, params)
    newm = jax.tree_util.tree_map(np.asarray, optax.apply_updates(params, um))
    assert np.array_equal(newm["backbone"]["k"], w["backbone"]["k"]), "masked: backbone must not change"
    assert np.max(np.abs(newm["head"]["b"] - new["head"]["b"])) == 0.0, "masked: head step must match"
    print("  adamw masked (head-only) step: backbone frozen, head identical  OK")


def check_sgd(lr=1e-2, wd=0.05, mom=0.9, tol=1e-7):
    rng = np.random.default_rng(1)
    w = rng.standard_normal((5,)).astype(np.float32)
    g = rng.standard_normal((5,)).astype(np.float32)
    tx = T.make_optimizer("sgd", wd, mom)
    st = T.with_lr(tx.init({"w": jnp.asarray(w)}), lr)
    u, _ = tx.update({"w": jnp.asarray(g)}, st, {"w": jnp.asarray(w)})
    new = np.asarray(optax.apply_updates({"w": jnp.asarray(w)}, u)["w"])
    ref = w.astype(np.float64) - lr * ((g + mom * g) + wd * w).astype(np.float64)
    err = float(np.max(np.abs(new - ref)))
    assert err < tol, f"sgd step mismatch {err:.3e}"
    print(f"  sgd(nesterov) one-step vs closed form: max|err|={err:.2e}  OK")


def main():
    print(f"[jax] {jax.__version__}  devices: {jax.devices()}")
    check_adamw()
    check_sgd()
    worst = check_schedule(T.lr_numpy_jax)
    print(f"LR SCHEDULE PARITY OK for lr_numpy_jax (worst {worst:.2e})")
    print("OK")


if __name__ == "__main__":
    main()
