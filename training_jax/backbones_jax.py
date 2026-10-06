"""resnet18 / mobilenetv3_small_100 / convnext_small in bare JAX (flax.linen), block-for-block
re-implementations of the timm models whose weights `pretrained/<backbone>.npz` carries --
the same structure, names and formulas as backbones_tf.py.

Every backbone module exposes
    __call__(x, train, dtype, stages=False)  -> (B, feat_dim) pooled features, or the stage dict
    features(x, train, dtype)                -> (B, feat_dim)
    forward_stages(x, dtype)                 -> dict of stage outputs (eval mode) for the parity test
and the Classifier wraps it: backbone -> cast fp32 -> dropout -> Dense(num_classes) fp32 logits.
Variables: {"params": ..., "batch_stats": ...}; flat keys `backbone/...` + `head/dense/...`.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn

from layers_jax import (
    ACTS, F32, BatchNorm, Conv, Dense, Group, LayerNorm, flatten_vars, gelu, global_avg_pool,
    hard_sigmoid, hard_swish, max_pool_3x3_s2_p1, relu, unflatten_vars,
)
from npzio import add_prefix, assert_same_keys, assert_same_shapes, load_npz, subtree


# ---------------------------------------------------------------- resnet18
class BasicBlock(nn.Module):
    cout: int
    stride: int

    @nn.compact
    def __call__(self, x, train, dtype):
        cin, cout, s = x.shape[-1], self.cout, self.stride
        y = relu(BatchNorm(cout, name="bn1")(Conv(cout, 3, s, 1, name="conv1")(x, dtype), train))
        y = BatchNorm(cout, name="bn2")(Conv(cout, 3, 1, 1, name="conv2")(y, dtype), train)
        if s != 1 or cin != cout:
            def down(x):
                return BatchNorm(cout, name="bn")(Conv(cout, 1, s, 0, name="conv")(x, dtype), train)
            x = Group(name="down")(down, x)
        return relu(y + x)


class ResNet18(nn.Module):
    feat_dim = 512

    @nn.compact
    def __call__(self, x, train, dtype, stages=False):
        out = {}

        def stem(x):
            return relu(BatchNorm(64, name="bn")(Conv(64, 7, 2, 3, name="conv")(x, dtype), train))

        x = out["stem_act"] = Group(name="stem")(stem, x)
        x = out["maxpool"] = max_pool_3x3_s2_p1(x)
        cfg = [(64, 1), (128, 2), (256, 2), (512, 2)]
        for li, (cout, s) in enumerate(cfg, start=1):
            def layer(x, cout=cout, s=s):
                x = BasicBlock(cout, s, name="block0")(x, train, dtype)
                return BasicBlock(cout, 1, name="block1")(x, train, dtype)

            x = out[f"layer{li}"] = Group(name=f"layer{li}")(layer, x)
        out["pooled"] = global_avg_pool(x)
        return out if stages else out["pooled"]

    def features(self, x, train, dtype):
        return self(x, train, dtype)

    def forward_stages(self, x, dtype):
        return self(x, False, dtype, stages=True)


# ---------------------------------------------------------------- mobilenetv3_small_100
# (kind, k, stride, cin, mid, cout, se_rd, act) -- from timm's mobilenetv3_small_100 (non-minimal)
MNV3_SMALL = [
    [("ds", 3, 2, 16, None, 16, 8, "relu")],
    [("ir", 3, 2, 16, 72, 24, None, "relu"), ("ir", 3, 1, 24, 88, 24, None, "relu")],
    [("ir", 5, 2, 24, 96, 40, 24, "hs"), ("ir", 5, 1, 40, 240, 40, 64, "hs"), ("ir", 5, 1, 40, 240, 40, 64, "hs")],
    [("ir", 5, 1, 40, 120, 48, 32, "hs"), ("ir", 5, 1, 48, 144, 48, 40, "hs")],
    [("ir", 5, 2, 48, 288, 96, 72, "hs"), ("ir", 5, 1, 96, 576, 96, 144, "hs"), ("ir", 5, 1, 96, 576, 96, 144, "hs")],
    [("cn", 1, 1, 96, None, 576, None, "hs")],
]


class SqueezeExcite(nn.Module):
    rd: int

    @nn.compact
    def __call__(self, x, dtype):
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = relu(Conv(self.rd, 1, 1, 0, bias=True, name="reduce")(s, dtype))
        s = hard_sigmoid(Conv(x.shape[-1], 1, 1, 0, bias=True, name="expand")(s, dtype))
        return x * s


class MnV3Block(nn.Module):
    spec: tuple

    @nn.compact
    def __call__(self, x, train, dtype):
        kind, k, s, cin, mid, cout, se_rd, act = self.spec
        act = ACTS[act]
        assert x.shape[-1] == cin, f"MnV3Block: expected {cin} channels, got {x.shape[-1]}"
        skip = s == 1 and cin == cout
        p = (k - 1) // 2
        if kind == "ds":
            y = act(BatchNorm(cin, name="bn1")(Conv(cin, k, s, p, depthwise=True, name="conv_dw")(x, dtype), train))
            if se_rd:
                y = SqueezeExcite(se_rd, name="se")(y, dtype)
            y = BatchNorm(cout, name="bn2")(Conv(cout, 1, 1, 0, name="conv_pw")(y, dtype), train)
        elif kind == "ir":
            y = act(BatchNorm(mid, name="bn1")(Conv(mid, 1, 1, 0, name="conv_pw")(x, dtype), train))
            y = act(BatchNorm(mid, name="bn2")(Conv(mid, k, s, p, depthwise=True, name="conv_dw")(y, dtype), train))
            if se_rd:
                y = SqueezeExcite(se_rd, name="se")(y, dtype)
            y = BatchNorm(cout, name="bn3")(Conv(cout, 1, 1, 0, name="conv_pwl")(y, dtype), train)
        else:  # cn
            y = act(BatchNorm(cout, name="bn1")(Conv(cout, k, s, 0, name="conv")(x, dtype), train))
        return y + x if skip else y


class MobileNetV3Small(nn.Module):
    feat_dim = 1024

    @nn.compact
    def __call__(self, x, train, dtype, stages=False):
        out = {}

        def stem(x):
            return hard_swish(BatchNorm(16, name="bn")(Conv(16, 3, 2, 1, name="conv")(x, dtype), train))

        x = out["stem_act"] = Group(name="stem")(stem, x)
        blocks = Group(name="blocks")
        for si, stage in enumerate(MNV3_SMALL):
            def run_stage(x, si=si, stage=stage):
                def body(x):
                    for bi, spec in enumerate(stage):
                        x = MnV3Block(spec, name=str(bi))(x, train, dtype)
                    return x
                return Group(name=str(si))(body, x)

            x = out[f"blocks{si}"] = blocks(run_stage, x)

        def head(x):
            p = jnp.mean(x, axis=(1, 2), keepdims=True)  # (B,1,1,576)
            h = hard_swish(Conv(self.feat_dim, 1, 1, 0, bias=True, name="conv")(p, dtype))
            return h.reshape(-1, self.feat_dim)

        out["pre_logits"] = Group(name="head")(head, x)
        return out if stages else out["pre_logits"]

    def features(self, x, train, dtype):
        return self(x, train, dtype)

    def forward_stages(self, x, dtype):
        return self(x, False, dtype, stages=True)


# ---------------------------------------------------------------- convnext_small
class ConvNeXtBlock(nn.Module):
    c: int

    @nn.compact
    def __call__(self, x, dtype):
        c = self.c
        y = Conv(c, 7, 1, 3, bias=True, depthwise=True, name="dwconv")(x, dtype)
        y = LayerNorm(c, 1e-6, name="ln")(y)

        def mlp(y):
            return Dense(c, name="fc2")(gelu(Dense(4 * c, name="fc1")(y, dtype)), dtype)

        y = Group(name="mlp")(mlp, y)
        gamma = self.param("gamma", lambda key, shape: jnp.full(shape, 1e-6, F32), (c,))
        return x + y * gamma.astype(dtype)


class ConvNeXtSmall(nn.Module):
    feat_dim = 768
    depths, dims = (3, 3, 27, 3), (96, 192, 384, 768)

    @nn.compact
    def __call__(self, x, train, dtype, stages=False):
        out = {}
        depths, dims = self.depths, self.dims

        def stem(x):
            return LayerNorm(96, 1e-6, name="ln")(Conv(96, 4, 4, 0, bias=True, name="conv")(x, dtype))

        x = out["stem"] = Group(name="stem")(stem, x)
        stages_g = Group(name="stages")
        for si, (d, dim) in enumerate(zip(depths, dims)):
            def run_stage(x, si=si, d=d, dim=dim):
                def body(x):
                    if si > 0:
                        def down(x):
                            return Conv(dim, 2, 2, 0, bias=True, name="conv")(
                                LayerNorm(dims[si - 1], 1e-6, name="ln")(x), dtype)
                        x = Group(name="down")(down, x)

                    def blocks(x):
                        for bi in range(d):
                            x = ConvNeXtBlock(dim, name=str(bi))(x, dtype)
                        return x
                    return Group(name="blocks")(blocks, x)
                return Group(name=str(si))(body, x)

            x = out[f"stages{si}"] = stages_g(run_stage, x)
        out["pre_logits"] = Group(name="head")(lambda x: LayerNorm(768, 1e-6, name="ln")(global_avg_pool(x)), x)
        return out if stages else out["pre_logits"]

    def features(self, x, train, dtype):
        return self(x, train, dtype)

    def forward_stages(self, x, dtype):
        return self(x, False, dtype, stages=True)


BACKBONES = {"resnet18": ResNet18, "mobilenetv3_small_100": MobileNetV3Small, "convnext_small": ConvNeXtSmall}


def build_backbone(backbone, **kw):
    return BACKBONES[backbone](**kw)


# ---------------------------------------------------------------- classifier
class Classifier(nn.Module):
    """backbone -> cast fp32 -> dropout -> Dense(num_classes); logits always fp32 (mixed-precision safe)."""
    backbone_name: str
    num_classes: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train, dtype, backbone_train=None):
        bt = train if backbone_train is None else backbone_train
        f = build_backbone(self.backbone_name, name="backbone").features(x.astype(dtype), bt, dtype)
        f = f.astype(F32)
        f = nn.Dropout(float(self.dropout_rate))(f, deterministic=not train)  # 'dropout' rng collection
        return Group(name="head")(lambda f: Dense(self.num_classes, name="dense")(f, F32), f)


def init_variables(module, rng_key, size=64):
    """{"params", "batch_stats"} of a freshly initialised module (shape-independent: GAP heads)."""
    v = module.init({"params": rng_key}, jnp.zeros((1, size, size, 3), F32), False, F32)
    return {"params": v["params"], "batch_stats": v.get("batch_stats", {})}


def load_flat(variables, flat, prefix="", what="weights"):
    """Strict load of a flat npz dict into (a `prefix` subtree of) `variables`: both-way key check
    + shape check against the current variables, then unflatten (dtype of `flat` preserved, so a
    cast_fp16 dict gives fp16 leaves -- the counterpart of the TF cast_variables + load_flat)."""
    cur = flatten_vars(variables["params"], variables["batch_stats"])
    expected = subtree(cur, prefix) if prefix else cur
    assert_same_keys(expected.keys(), flat.keys(), what=what)
    assert_same_shapes({k: v.shape for k, v in expected.items()}, flat, what=what)
    merged = dict(cur)
    merged.update(add_prefix(flat, prefix) if prefix else flat)
    params, stats = unflatten_vars(merged)
    return {"params": params, "batch_stats": stats}


def build_model(name, num_classes, dropout_rate, pretrained_npz=None, trunk_npz=None, rng_key=None, verbose=True):
    """(module, variables) with ImageNet (pretrained_npz) or Stage-1 (trunk_npz) backbone weights.
    Both loads are strict on the backbone key set (the torch load_trunk guarantee); the head is
    glorot-uniform / zeros from `rng_key`."""
    module = Classifier(name, num_classes, dropout_rate)
    variables = init_variables(module, rng_key if rng_key is not None else jax.random.PRNGKey(0))
    src = trunk_npz or pretrained_npz
    if src:
        flat = load_npz(src)
        variables = load_flat(variables, flat, prefix="backbone", what=f"{src} vs {name} backbone")
        if verbose:
            print(f"loaded {'patch-pretrained trunk' if trunk_npz else 'ImageNet weights'} from {src} "
                  f"({len(flat)} arrays; only the head is random)")
    return module, variables
