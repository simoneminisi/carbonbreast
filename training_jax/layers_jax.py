"""Bare-JAX building blocks (flax.linen + lax, NHWC, explicit torch-style padding). No Keras, no TF.

Mirrors layers_tf.py block for block. Parameters live in the `params` collection (fp32), BatchNorm
running stats in `batch_stats`; every layer takes a compute `dtype` and casts its kernel to it,
so one module runs fp32, mixed fp16 (fp32 master weights) and pure fp16 (after `unflatten_vars(
cast_fp16(flat))` the leaves themselves are fp16). Normalisation math is always fp32.

Naming follows the npz key scheme of npzio.py: structural levels that carry no parameters of
their own (`stem`, `layer1`, `blocks/2/1`, `head`, ...) are `Group` containers, so
`flax.traverse_util.flatten_dict(sep='/')` of the variables yields the structural keys directly
(`stem/conv/kernel`, `layer1/block0/bn1/scale`, `blocks/2/1/se/reduce/kernel`, ...).
The ONLY layout translation is the depthwise kernel: flax/lax store it as (k,k,1,C), the npz
as (k,k,C,1); `flatten_vars` / `unflatten_vars` transpose it.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

F32 = jnp.float32
DEPTHWISE_MODULES = ("conv_dw", "dwconv")  # module names whose `kernel` is a depthwise (k,k,1,C) kernel

kaiming = nn.initializers.variance_scaling(2.0, "fan_in", "normal")  # he-normal (only for --no-pretrained)
glorot = nn.initializers.glorot_uniform()


class Group(nn.Module):
    """Pure naming container: everything `fn` creates lives under this module's name.
    (`Group(name="stem")(body, x)` <-> the TF port's `children_["stem/conv"]` string nesting.)"""

    @nn.compact
    def __call__(self, fn, *args):
        return fn(*args)


class Conv(nn.Module):
    """(kH,kW,I,O) conv or depthwise (feature_group_count=C, kernel (kH,kW,1,C)), explicit
    symmetric zero padding `pad` (never SAME). Depthwise convs must be named conv_dw / dwconv."""
    cout: int
    k: int
    stride: int = 1
    pad: int = 0
    bias: bool = False
    depthwise: bool = False

    @nn.compact
    def __call__(self, x, dtype):
        cin = x.shape[-1]
        if self.depthwise != (self.name in DEPTHWISE_MODULES):
            raise ValueError(f"Conv {self.name!r}: depthwise={self.depthwise} but depthwise modules are "
                             f"exactly {DEPTHWISE_MODULES} (needed by flatten_vars/unflatten_vars)")
        if self.depthwise:
            assert cin == self.cout, "depthwise conv needs cin == cout"
            shape, groups = (self.k, self.k, 1, cin), cin
        else:
            shape, groups = (self.k, self.k, cin, self.cout), 1
        w = self.param("kernel", kaiming, shape, F32).astype(dtype)
        p = self.pad
        y = jax.lax.conv_general_dilated(
            x, w, window_strides=(self.stride, self.stride), padding=[(p, p), (p, p)],
            dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=groups)
        if self.bias:
            y = y + self.param("bias", nn.initializers.zeros, (self.cout,), F32).astype(dtype)
        return y


class BatchNorm(nn.Module):
    """Torch-semantics BN: eps 1e-5; train mode normalises with the BIASED batch variance and
    updates `batch_stats` with momentum 0.1 (running = 0.9*running + 0.1*batch) using the
    UNBIASED batch variance n/(n-1). fp32 math on an upcast input, result cast back."""
    c: int
    eps: float = 1e-5
    momentum: float = 0.1

    @nn.compact
    def __call__(self, x, train):
        c = self.c
        scale = self.param("scale", nn.initializers.ones, (c,), F32).astype(F32)
        bias = self.param("bias", nn.initializers.zeros, (c,), F32).astype(F32)
        ra_mean = self.variable("batch_stats", "mean", lambda: jnp.zeros((c,), F32))
        ra_var = self.variable("batch_stats", "var", lambda: jnp.ones((c,), F32))
        xf = x.astype(F32)
        if train:
            axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(xf, axis=axes)
            var = jnp.mean(jnp.square(xf - mean), axis=axes)  # biased, for the normalisation
            if not self.is_initializing():
                n = float(np.prod([x.shape[a] for a in axes]))
                m = self.momentum
                ra_mean.value = ((1.0 - m) * ra_mean.value.astype(F32) + m * mean).astype(ra_mean.value.dtype)
                ra_var.value = ((1.0 - m) * ra_var.value.astype(F32)
                                + m * var * (n / max(n - 1.0, 1.0))).astype(ra_var.value.dtype)
        else:
            mean, var = ra_mean.value.astype(F32), ra_var.value.astype(F32)
        y = (xf - mean) * jax.lax.rsqrt(var + self.eps) * scale + bias
        return y.astype(x.dtype)


class LayerNorm(nn.Module):
    """LayerNorm over the last axis in fp32 (eps 1e-6 for ConvNeXt), result cast back."""
    c: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.c,), F32).astype(F32)
        bias = self.param("bias", nn.initializers.zeros, (self.c,), F32).astype(F32)
        xf = x.astype(F32)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        y = (xf - mean) * jax.lax.rsqrt(var + self.eps) * scale + bias
        return y.astype(x.dtype)


class Dense(nn.Module):
    """(in,out) kernel + bias; works on (B,in) and, via a batched matmul (== 1x1 conv), on (B,H,W,in)."""
    cout: int
    bias: bool = True

    @nn.compact
    def __call__(self, x, dtype):
        w = self.param("kernel", glorot, (x.shape[-1], self.cout), F32).astype(dtype)
        y = jnp.matmul(x.astype(dtype), w)
        if self.bias:
            y = y + self.param("bias", nn.initializers.zeros, (self.cout,), F32).astype(dtype)
        return y


# --- activations / pooling (timm formulas) ---
def relu(x):
    return jax.nn.relu(x)


def hard_swish(x):
    return x * jax.nn.relu6(x + 3.0) / 6.0


def hard_sigmoid(x):
    return jax.nn.relu6(x + 3.0) / 6.0


def gelu(x):
    return jax.nn.gelu(x, approximate=False)


ACTS = {"relu": relu, "hs": hard_swish, "gelu": gelu}


def max_pool_3x3_s2_p1(x):
    """torch MaxPool2d(3, stride 2, padding 1): nn.max_pool pads with its -inf init value."""
    return nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))


def global_avg_pool(x):
    return jnp.mean(x, axis=(1, 2))


# --- flat <-> nested variable trees (npz key scheme) ---
def _is_depthwise_key(key):
    parts = key.split("/")
    return len(parts) >= 2 and parts[-1] == "kernel" and parts[-2] in DEPTHWISE_MODULES


def flatten_vars(params, batch_stats=None):
    """Nested flax collections -> flat {structural key: np.ndarray} in the npz scheme
    (`.../bn/scale` from params, `.../bn/mean` from batch_stats; depthwise (k,k,1,C)->(k,k,C,1))."""
    flat = flatten_dict(jax.device_get(params), sep="/")
    if batch_stats:
        flat.update(flatten_dict(jax.device_get(batch_stats), sep="/"))
    out = {}
    for k, v in flat.items():
        a = np.asarray(v)
        if _is_depthwise_key(k):
            assert a.ndim == 4 and a.shape[2] == 1, f"{k}: expected flax depthwise (k,k,1,C), got {a.shape}"
            a = np.ascontiguousarray(a.transpose(0, 1, 3, 2))
        out[k] = a
    return out


def unflatten_vars(flat):
    """Inverse of flatten_vars: flat npz dict -> (params, batch_stats) nested trees of jnp arrays
    (dtype preserved, so a cast_fp16 dict yields fp16 leaves)."""
    params, stats = {}, {}
    for k, v in flat.items():
        a = np.asarray(v)
        if _is_depthwise_key(k):
            assert a.ndim == 4 and a.shape[3] == 1, f"{k}: expected npz depthwise (k,k,C,1), got {a.shape}"
            a = np.ascontiguousarray(a.transpose(0, 1, 3, 2))
        leaf = k.rsplit("/", 1)[-1]
        (stats if leaf in ("mean", "var") else params)[k] = jnp.asarray(a)
    return unflatten_dict(params, sep="/"), unflatten_dict(stats, sep="/")


def cast_tree_fp16(tree):
    """fp16 copy of every floating leaf of a pytree (the in-memory counterpart of npzio.cast_fp16)."""
    return jax.tree_util.tree_map(
        lambda a: a.astype(jnp.float16) if jnp.issubdtype(a.dtype, jnp.floating) else a, tree)
