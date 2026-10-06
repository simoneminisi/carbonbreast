"""Bare-TensorFlow building blocks (tf.Module + tf.Variable, NHWC, explicit torch-style padding).

No Keras anywhere. Parameters are fp32 variables; every layer takes a `dtype` and casts its
kernel to it, so the same module runs fp32, mixed fp16 (fp32 master weights) and pure fp16
(after `load_flat(cast_fp16(...))` the variables themselves are fp16).

Variable naming follows the npz key scheme of npzio.py (`stem/conv/kernel`, `layer1/block0/bn1/scale`
...): every layer owns `leaves` (leaf name -> tf.Variable) and containers own `children`
(prefix -> layer); `flat()` walks the tree.
"""
import numpy as np
import tensorflow as tf


class Layer(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.leaves = {}
        self.children_ = {}

    def flat(self, prefix=""):
        out = {}
        for k, v in self.leaves.items():
            out[prefix + k] = v
        for k, c in self.children_.items():
            out.update(c.flat(prefix + k + "/"))
        return out

    def load_flat(self, flat, prefix=""):
        """Assign every variable from a flat dict (strict: raises on missing/extra keys)."""
        mine = self.flat(prefix)
        missing = sorted(set(mine) - set(flat))
        extra = sorted(set(flat) - set(mine))
        if missing or extra:
            raise KeyError(f"load_flat: missing {len(missing)} {missing[:6]} | extra {len(extra)} {extra[:6]}")
        for k, v in mine.items():
            a = flat[k]
            if tuple(a.shape) != tuple(v.shape):
                raise ValueError(f"load_flat: {k} shape {a.shape} != variable {tuple(v.shape)}")
            v.assign(tf.cast(a, v.dtype))

    def export_flat(self, prefix=""):
        return {k: v.numpy() for k, v in self.flat(prefix).items()}

    def cast_variables(self, dtype):
        """Rebuild every variable in `dtype` (used for the pure-fp16 evaluation copy)."""
        for k, v in list(self.leaves.items()):
            self.leaves[k] = tf.Variable(tf.cast(v, dtype), trainable=v.trainable, name=v.name.split(":")[0])
        for c in self.children_.values():
            c.cast_variables(dtype)
        return self


def _var(shape, init, name, trainable=True):
    return tf.Variable(init(shape).astype(np.float32), trainable=trainable, name=name)


def glorot(shape):
    fan_in, fan_out = int(np.prod(shape[:-1])), int(shape[-1])
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-lim, lim, size=shape)


def kaiming(shape):  # he-normal for convs (only matters for --no-pretrained)
    fan_in = int(np.prod(shape[:-1]))
    return np.random.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


class Conv2d(Layer):
    """(kH,kW,I,O) conv or (kH,kW,C,1) depthwise, explicit symmetric zero padding `pad`."""

    def __init__(self, cin, cout, k, stride=1, pad=0, bias=False, depthwise=False, name="conv"):
        super().__init__(name)
        self.k, self.stride, self.pad, self.depthwise = k, stride, pad, depthwise
        shape = (k, k, cin, 1) if depthwise else (k, k, cin, cout)
        self.leaves["kernel"] = _var(shape, kaiming, name + "_kernel")
        if bias:
            self.leaves["bias"] = _var((cout,), np.zeros, name + "_bias")

    def __call__(self, x, dtype):
        w = tf.cast(self.leaves["kernel"], dtype)
        if self.pad:
            p = self.pad
            x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        if self.depthwise:
            y = tf.nn.depthwise_conv2d(x, w, strides=[1, self.stride, self.stride, 1], padding="VALID")
        else:
            y = tf.nn.conv2d(x, w, strides=self.stride, padding="VALID")
        if "bias" in self.leaves:
            y = y + tf.cast(self.leaves["bias"], dtype)
        return y


class BatchNorm(Layer):
    """Torch-semantics BN: eps 1e-5, running stats updated with momentum 0.1 and UNBIASED batch
    variance. Uses the fused kernel (fp16 input allowed, fp32 stats/math inside)."""

    def __init__(self, c, eps=1e-5, momentum=0.1, name="bn"):
        super().__init__(name)
        self.eps, self.momentum = eps, momentum
        self.leaves["scale"] = _var((c,), np.ones, name + "_scale")
        self.leaves["bias"] = _var((c,), np.zeros, name + "_bias")
        self.leaves["mean"] = _var((c,), np.zeros, name + "_mean", trainable=False)
        self.leaves["var"] = _var((c,), np.ones, name + "_var", trainable=False)

    def __call__(self, x, training):
        L = self.leaves
        scale, offset = tf.cast(L["scale"], tf.float32), tf.cast(L["bias"], tf.float32)
        mean, var = tf.cast(L["mean"], tf.float32), tf.cast(L["var"], tf.float32)
        if training:
            y, new_mean, new_var = tf.compat.v1.nn.fused_batch_norm(
                x, scale, offset, mean=mean, variance=var, epsilon=self.eps, is_training=True,
                exponential_avg_factor=self.momentum, data_format="NHWC")
            L["mean"].assign(tf.cast(new_mean, L["mean"].dtype))
            L["var"].assign(tf.cast(new_var, L["var"].dtype))
            return y
        y, _, _ = tf.compat.v1.nn.fused_batch_norm(
            x, scale, offset, mean=mean, variance=var, epsilon=self.eps, is_training=False,
            data_format="NHWC")
        return y


class LayerNorm(Layer):
    """LayerNorm over the last axis in fp32 (eps 1e-6 for ConvNeXt), result cast back."""

    def __init__(self, c, eps=1e-6, name="ln"):
        super().__init__(name)
        self.eps = eps
        self.leaves["scale"] = _var((c,), np.ones, name + "_scale")
        self.leaves["bias"] = _var((c,), np.zeros, name + "_bias")

    def __call__(self, x):
        xf = tf.cast(x, tf.float32)
        mean, var = tf.nn.moments(xf, axes=[-1], keepdims=True)
        y = (xf - mean) * tf.math.rsqrt(var + self.eps)
        y = y * tf.cast(self.leaves["scale"], tf.float32) + tf.cast(self.leaves["bias"], tf.float32)
        return tf.cast(y, x.dtype)


class Dense(Layer):
    """(in,out) kernel + bias; works on (B,in) and, via a 1x1 conv, on (B,H,W,in)."""

    def __init__(self, cin, cout, bias=True, name="dense"):
        super().__init__(name)
        self.leaves["kernel"] = _var((cin, cout), glorot, name + "_kernel")
        if bias:
            self.leaves["bias"] = _var((cout,), np.zeros, name + "_bias")

    def __call__(self, x, dtype):
        w = tf.cast(self.leaves["kernel"], dtype)
        if len(x.shape) == 4:
            y = tf.nn.conv2d(x, w[None, None], strides=1, padding="VALID")
        else:
            y = tf.matmul(x, w)
        if "bias" in self.leaves:
            y = y + tf.cast(self.leaves["bias"], dtype)
        return y


# --- activations / pooling (timm formulas) ---
def relu(x):
    return tf.nn.relu(x)


def hard_swish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0


def hard_sigmoid(x):
    return tf.nn.relu6(x + 3.0) / 6.0


def gelu(x):
    return tf.nn.gelu(x, approximate=False)


ACTS = {"relu": relu, "hs": hard_swish, "gelu": gelu}


def max_pool_3x3_s2_p1(x):
    """torch MaxPool2d(3, stride 2, padding 1): pad with -inf, then VALID max-pool."""
    pad_value = x.dtype.min
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=pad_value)
    return tf.nn.max_pool2d(x, ksize=3, strides=2, padding="VALID")


def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[1, 2])


def dropout(x, rate, rng):
    """Inverted dropout driven by a stateful tf.random.Generator (deterministic per seed)."""
    if rate <= 0:
        return x
    keep = tf.cast(rng.uniform(tf.shape(x), dtype=tf.float32) >= rate, x.dtype)
    return x * keep / tf.cast(1.0 - rate, x.dtype)
