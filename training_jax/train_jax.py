"""Bare-JAX training core: seeding, device setup, per-step warmup-cosine LR, AdamW/Adam/SGD with
decoupled weight decay (optax), dynamic loss scaling for mixed fp16, and the single jitted
train / predict steps. No Keras, no TensorFlow.

Semantics mirror train_tf.py (and the Keras/torch runs it follows):
  * LR schedule is stepped on EVERY optimizer call (`TrainState.step`, always +1), including steps
    skipped by the loss scaler; Adam bias correction counts only applied steps (the optax state is
    only replaced when the gradients are finite).
  * AdamW weight decay is decoupled and lr-scaled on ALL trainable params (no BN/bias exclusion).
  * Dynamic loss scale: init 2^15, x2 after 2000 consecutive finite steps, /2 + skip on non-finite.
  * `--train-mode head`: optax.masked so only `head/` params update, backbone applied in inference
    mode (BN running stats), gradients through the backbone stopped.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # allocate on demand (TF memory growth analog)

import random  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import struct  # noqa: E402

from layers_jax import F32, cast_tree_fp16, flatten_vars, unflatten_vars  # noqa: E402,F401 (re-exported)
from lr import WarmupCosineSpec  # noqa: E402,F401 (re-exported for the scripts)
from metrics import softmax_np  # noqa: E402
from pipeline_np import eval_probs  # noqa: E402

LOSS_SCALE_INIT, GROWTH_INTERVAL = 2.0 ** 15, 2000


def setup_gpu():
    devs = jax.devices()
    gpu = any(d.platform == "gpu" for d in devs)
    print(f"[gpu] jax {jax.__version__}  devices: {devs}")
    return gpu


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    return jax.random.PRNGKey(s)


def compute_dtype(amp):
    return jnp.float16 if amp else F32


def jax_lr(spec, step):
    """Same 5-line `where` formulation as tf_lr / Keras WarmupCosine (see lr.py)."""
    e = jnp.asarray(step).astype(F32) / spec.step_div
    warm = e / spec.warm_div
    t = jnp.minimum((e - spec.warmup_epochs) / spec.decay_epochs, 1.0)
    cos = spec.floor_mult + (1.0 - spec.floor_mult) * 0.5 * (1.0 + jnp.cos(np.pi * t))
    return spec.peak_lr * jnp.where(e < spec.warmup_epochs, warm, cos)


def lr_numpy_jax(spec, steps):
    """Vectorised host-side evaluation of jax_lr for test_lr.check_schedule."""
    return np.asarray(jax_lr(spec, jnp.asarray(np.asarray(steps), dtype=jnp.int32)))


# ---------------------------------------------------------------- state / optimizer
@struct.dataclass
class TrainState:
    params: Any
    batch_stats: Any
    opt_state: Any
    step: jax.Array        # schedule counter, always incremented
    loss_scale: jax.Array  # dynamic loss scale (f32)
    good_steps: jax.Array  # consecutive finite steps (int32)

    @classmethod
    def create(cls, variables, tx):
        params = variables["params"]
        return cls(params=params, batch_stats=variables["batch_stats"], opt_state=tx.init(params),
                   step=jnp.zeros((), jnp.int32), loss_scale=jnp.asarray(LOSS_SCALE_INIT, F32),
                   good_steps=jnp.zeros((), jnp.int32))

    def variables(self):
        return {"params": self.params, "batch_stats": self.batch_stats}


def head_mask(params):
    """Bool pytree: True for `head/...` leaves (the only trainable ones in --train-mode head)."""
    return jax.tree_util.tree_map_with_path(lambda path, _: path[0].key == "head", params)


def _sgd(learning_rate, momentum, weight_decay, nesterov):
    """torch/Keras SGD: p -= lr * (nesterov_momentum_step(g) + wd * p)  (decay added AFTER momentum)."""
    steps = [optax.trace(decay=momentum, nesterov=nesterov)] if momentum > 0 else []
    steps += [optax.add_decayed_weights(weight_decay), optax.scale_by_learning_rate(learning_rate)]
    return optax.chain(*steps)


def make_optimizer(kind, weight_decay=0.0, momentum=0.9, mask=None, beta1=0.9, beta2=0.999, eps=1e-8):
    """adamw / adam (both decoupled wd, like keras.optimizers.Adam(weight_decay=...)) and sgd
    (nesterov when momentum > 0); the learning rate is injected per step by `with_lr`."""
    kind = kind.lower()
    if kind in ("adamw", "adam"):
        tx = optax.inject_hyperparams(optax.adamw)(
            learning_rate=0.0, b1=beta1, b2=beta2, eps=eps, weight_decay=float(weight_decay), mask=None)
    elif kind == "sgd":
        tx = optax.inject_hyperparams(_sgd, static_args=("momentum", "nesterov"))(
            learning_rate=0.0, momentum=float(momentum), weight_decay=float(weight_decay), nesterov=momentum > 0)
    else:
        raise NotImplementedError(f"optimizer {kind} not implemented in the bare-JAX port")
    if mask is not None:  # head-only: optimizer on the masked-in leaves, zero update on the rest
        frozen = jax.tree_util.tree_map(lambda m: not m, mask)
        tx = optax.chain(optax.masked(tx, mask), optax.masked(optax.set_to_zero(), frozen))
    return tx


def with_lr(opt_state, lr):
    """Copy of the optimizer state with `learning_rate` replaced in every inject_hyperparams state
    (walks optax.chain tuples and MaskedState wrappers)."""
    if isinstance(opt_state, optax.MaskedState):
        return opt_state._replace(inner_state=with_lr(opt_state.inner_state, lr))
    if hasattr(opt_state, "hyperparams"):  # InjectHyperparamsState / InjectStatefulHyperparamsState
        hp = dict(opt_state.hyperparams)
        hp["learning_rate"] = jnp.asarray(lr, hp["learning_rate"].dtype)
        return opt_state._replace(hyperparams=hp)
    if isinstance(opt_state, tuple) and not hasattr(opt_state, "_fields"):  # optax.chain state
        return tuple(with_lr(s, lr) for s in opt_state)
    return opt_state  # EmptyState & co.


# ---------------------------------------------------------------- train / predict steps
def _all_finite(grads):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))


def make_train_step(module, tx, spec, amp, head_only=False):
    """One jitted function (state, x, y, dropout_key) -> (state, loss, finite):
    forward -> CE -> (scaled) grads -> unscale -> finite? -> update. Fixed input shapes
    (drop_last training batches) => exactly one compilation."""
    dtype = compute_dtype(amp)

    def loss_fn(params, batch_stats, x, y, key, scale):
        if head_only:
            params = jax.tree_util.tree_map(lambda m, p: p if m else jax.lax.stop_gradient(p),
                                            head_mask(params), params)
        logits, mut = module.apply({"params": params, "batch_stats": batch_stats}, x, True, dtype,
                                   backbone_train=not head_only, rngs={"dropout": key}, mutable=["batch_stats"])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()  # fp32 logits
        return loss * scale, (loss, mut["batch_stats"])

    @jax.jit
    def step(state, x, y, dropout_key):
        key = jax.random.fold_in(dropout_key, state.step)
        lr = jax_lr(spec, state.step)
        scale = state.loss_scale if amp else jnp.asarray(1.0, F32)
        grads, (loss, new_bs) = jax.grad(loss_fn, has_aux=True)(
            state.params, state.batch_stats, x.astype(dtype), y, key, scale)
        opt_state = with_lr(state.opt_state, lr)
        if amp:
            grads = jax.tree_util.tree_map(lambda g: g / scale, grads)
            finite = _all_finite(grads)
        else:
            finite = jnp.asarray(True)
        updates, new_opt = tx.update(grads, opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        if amp:  # apply only when finite; batch_stats and the schedule step advance regardless
            new_params = jax.tree_util.tree_map(lambda n, o: jnp.where(finite, n, o), new_params, state.params)
            new_opt = jax.tree_util.tree_map(lambda n, o: jnp.where(finite, n, o), new_opt, opt_state)
            good = state.good_steps + 1
            grow = good >= GROWTH_INTERVAL
            new_scale = jnp.where(finite, jnp.where(grow, scale * 2.0, scale), jnp.maximum(scale / 2.0, 1.0))
            new_good = jnp.where(finite, jnp.where(grow, 0, good), 0).astype(jnp.int32)
        else:
            new_scale, new_good = state.loss_scale, state.good_steps
        state = state.replace(params=new_params, batch_stats=new_bs, opt_state=new_opt,
                              step=state.step + 1, loss_scale=new_scale, good_steps=new_good)
        return state, loss, finite

    return step


def make_predict_fn(module, dtype):
    """Jitted (variables, x) -> fp32 logits; traced once per input shape (EvalLoader pads batches)."""
    @jax.jit
    def predict(variables, x):
        return module.apply(variables, x, False, dtype)

    return predict


def to_device(x):
    """Copy a loader batch to the default device. The loader reuses its ring slot after the next
    item is requested; GPU device_put copies during the call, but CPU device_put aliases host
    memory, so a numpy copy is taken first there."""
    if jax.default_backend() == "cpu":
        x = np.array(x, copy=True)
    return jax.device_put(x)


def predict_probs(predict_fn, variables, loader):
    """Class-1 probability for every row of an EvalLoader (fixed-shape, padded batches)."""
    return eval_probs(loader, lambda x: np.asarray(predict_fn(variables, to_device(x))),
                      lambda lg: softmax_np(lg)[:, 1])


def predict_logits(predict_fn, variables, loader):
    return eval_probs(loader, lambda x: np.asarray(predict_fn(variables, to_device(x))), lambda lg: lg)
