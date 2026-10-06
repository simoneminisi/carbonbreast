"""Bare-TensorFlow training core: seeding, GPU setup, per-step warmup-cosine LR, AdamW/Adam/SGD
with decoupled weight decay, dynamic loss scaling for mixed fp16, and the single-graph
tf.function train / predict steps. No Keras.

Semantics mirror the Keras runs (and torch where Keras followed torch):
  * LR schedule is stepped on EVERY optimizer call, including steps skipped by the loss scaler
    (Keras LossScaleOptimizer behaviour); Adam bias correction counts only applied steps.
  * AdamW weight decay is decoupled and lr-scaled on ALL trainable params (no BN/bias exclusion).
  * Dynamic loss scale: init 2^15, x2 after 2000 consecutive finite steps, /2 + skip on non-finite.
"""
import random

import numpy as np
import tensorflow as tf

from lr import WarmupCosineSpec  # noqa: F401 (re-exported for the scripts)
from metrics import softmax_np
from pipeline_np import eval_probs


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:  # noqa: BLE001
            print(f"[gpu] set_memory_growth failed on {g}: {e}")
    print(f"[gpu] tensorflow {tf.__version__}  GPUs: {gpus}")
    return len(gpus) > 0


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    return tf.random.Generator.from_seed(s)


def tf_lr(spec, step):
    """Same 5-line `where` formulation as Keras WarmupCosine (see lr.py)."""
    e = tf.cast(step, tf.float32) / spec.step_div
    warm = e / spec.warm_div
    t = tf.minimum((e - spec.warmup_epochs) / spec.decay_epochs, 1.0)
    cos = spec.floor_mult + (1.0 - spec.floor_mult) * 0.5 * (1.0 + tf.cos(np.pi * t))
    return spec.peak_lr * tf.where(e < spec.warmup_epochs, warm, cos)


def lr_numpy_tf(spec, steps):
    """Vectorised host-side evaluation of tf_lr for test_lr.check_schedule."""
    return tf_lr(spec, tf.constant(np.asarray(steps, dtype=np.int64))).numpy()


class Optimizer(tf.Module):
    """adamw / adam (both with decoupled wd, like keras.optimizers.Adam(weight_decay=...)) and
    sgd (nesterov when momentum > 0). Owns two counters: `iterations` (schedule, always
    incremented) and `adam_t` (bias correction, incremented only on applied updates)."""

    def __init__(self, variables, kind, spec, weight_decay=0.0, momentum=0.9,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(name="opt")
        kind = kind.lower()
        if kind not in ("adamw", "adam", "sgd"):
            raise NotImplementedError(f"optimizer {kind} not implemented in the bare-TF port")
        self.kind, self.spec = kind, spec
        self.wd, self.momentum = float(weight_decay), float(momentum)
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.vars = list(variables)
        self.iterations = tf.Variable(0, dtype=tf.int64, trainable=False, name="iterations")
        self.adam_t = tf.Variable(0, dtype=tf.int64, trainable=False, name="adam_t")
        self.m = [tf.Variable(tf.zeros_like(v), trainable=False, name="m") for v in self.vars]
        self.v = [tf.Variable(tf.zeros_like(v), trainable=False, name="v") for v in self.vars] \
            if kind != "sgd" else []

    def lr(self):
        return tf_lr(self.spec, self.iterations)

    def apply(self, grads):
        lr = self.lr()
        if self.kind == "sgd":
            for p, g, buf in zip(self.vars, grads, self.m):
                if self.momentum > 0:
                    buf.assign(self.momentum * buf + g)
                    step = g + self.momentum * buf  # nesterov
                else:
                    step = g
                p.assign_sub(lr * (step + self.wd * p))
            return
        t = tf.cast(self.adam_t + 1, tf.float32)
        bc1 = 1.0 - tf.pow(self.b1, t)
        bc2 = 1.0 - tf.pow(self.b2, t)
        for p, g, m, v in zip(self.vars, grads, self.m, self.v):
            m.assign(self.b1 * m + (1.0 - self.b1) * g)
            v.assign(self.b2 * v + (1.0 - self.b2) * tf.square(g))
            update = (m / bc1) / (tf.sqrt(v / bc2) + self.eps)
            p.assign_sub(lr * (update + self.wd * p))
        self.adam_t.assign_add(1)

    def step_schedule(self):
        self.iterations.assign_add(1)


class DynamicLossScale(tf.Module):
    def __init__(self, initial=2.0 ** 15, growth_interval=2000):
        super().__init__(name="loss_scale")
        self.scale = tf.Variable(initial, dtype=tf.float32, trainable=False, name="scale")
        self.good_steps = tf.Variable(0, dtype=tf.int64, trainable=False, name="good_steps")
        self.growth_interval = growth_interval

    def update(self, finite):
        def on_finite():
            g = self.good_steps + 1
            grow = g >= self.growth_interval
            self.scale.assign(tf.where(grow, self.scale * 2.0, self.scale))
            self.good_steps.assign(tf.where(grow, tf.zeros_like(g), g))
            return 0

        def on_nonfinite():
            self.scale.assign(tf.maximum(self.scale / 2.0, 1.0))
            self.good_steps.assign(0)
            return 0

        tf.cond(finite, on_finite, on_nonfinite)


def _all_finite(grads):
    return tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in grads])


def make_train_step(model, opt, loss_scale, batch, img_size, amp, xla, rng, head_only=False):
    """One tf.function doing forward -> CE -> (scaled) grads -> unscale -> finite? -> update.
    Fixed input signature (drop_last training batches) => exactly one trace."""
    compute_dtype = tf.float16 if amp else tf.float32
    train_vars = opt.vars
    sig = [tf.TensorSpec((batch, img_size, img_size, 3), tf.float32), tf.TensorSpec((batch,), tf.int32)]

    @tf.function(input_signature=sig, jit_compile=True if xla else None)
    def step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True, dtype=compute_dtype, rng=rng, backbone_training=not head_only)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
            scaled = loss * loss_scale.scale if amp else loss
        grads = tape.gradient(scaled, train_vars)
        if amp:
            grads = [g / loss_scale.scale for g in grads]
            finite = _all_finite(grads)

            def apply():
                opt.apply(grads)
                return 0

            tf.cond(finite, apply, lambda: 0)
            loss_scale.update(finite)
        else:
            opt.apply(grads)
            finite = tf.constant(True)
        opt.step_schedule()
        return loss, finite

    return step


def make_predict_fn(model, compute_dtype, batch, img_size, xla=False):
    sig = [tf.TensorSpec((batch, img_size, img_size, 3), tf.float32)]

    @tf.function(input_signature=sig, jit_compile=True if xla else None)
    def predict(x):
        return model(x, training=False, dtype=compute_dtype)

    return predict


def predict_probs(predict_fn, loader):
    """Class-1 probability for every row of an EvalLoader (fixed-shape, padded batches)."""
    return eval_probs(loader, lambda x: predict_fn(tf.constant(x)).numpy(), lambda lg: softmax_np(lg)[:, 1])


def predict_logits(predict_fn, loader):
    return eval_probs(loader, lambda x: predict_fn(tf.constant(x)).numpy(), lambda lg: lg)


def trainable_variables(model, head_only=False):
    flat = model.flat()
    return [v for k, v in flat.items() if v.trainable and (not head_only or k.startswith("head/"))]
