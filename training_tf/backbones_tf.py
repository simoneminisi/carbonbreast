"""resnet18 / mobilenetv3_small_100 / convnext_small in bare TensorFlow, block-for-block
re-implementations of the timm models whose weights `pretrained/<backbone>.npz` carries.

Every backbone exposes
    features(x, training, dtype)           -> (B, feat_dim) pooled features
    forward_stages(x, dtype)               -> dict of stage outputs (eval mode) for the parity test
and the classifier wraps it: backbone -> dropout -> Dense(num_classes) fp32 logits.
"""
import numpy as np
import tensorflow as tf

from layers_tf import (
    ACTS, BatchNorm, Conv2d, Dense, Layer, LayerNorm, dropout, global_avg_pool, hard_sigmoid,
    hard_swish, max_pool_3x3_s2_p1, relu,
)
from npzio import add_prefix, assert_same_keys, load_npz, subtree


# ---------------------------------------------------------------- resnet18
class BasicBlock(Layer):
    def __init__(self, cin, cout, stride, name):
        super().__init__(name)
        self.children_["conv1"] = Conv2d(cin, cout, 3, stride, 1, name=name + "_conv1")
        self.children_["bn1"] = BatchNorm(cout, name=name + "_bn1")
        self.children_["conv2"] = Conv2d(cout, cout, 3, 1, 1, name=name + "_conv2")
        self.children_["bn2"] = BatchNorm(cout, name=name + "_bn2")
        self.down = stride != 1 or cin != cout
        if self.down:
            self.children_["down/conv"] = Conv2d(cin, cout, 1, stride, 0, name=name + "_down_conv")
            self.children_["down/bn"] = BatchNorm(cout, name=name + "_down_bn")

    def __call__(self, x, training, dtype):
        c = self.children_
        y = relu(c["bn1"](c["conv1"](x, dtype), training))
        y = c["bn2"](c["conv2"](y, dtype), training)
        s = c["down/bn"](c["down/conv"](x, dtype), training) if self.down else x
        return relu(y + s)


class ResNet18(Layer):
    feat_dim = 512

    def __init__(self, name="resnet18"):
        super().__init__(name)
        self.children_["stem/conv"] = Conv2d(3, 64, 7, 2, 3, name="stem_conv")
        self.children_["stem/bn"] = BatchNorm(64, name="stem_bn")
        cfg = [(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
        for li, (cin, cout, s) in enumerate(cfg, start=1):
            self.children_[f"layer{li}/block0"] = BasicBlock(cin, cout, s, f"layer{li}_block0")
            self.children_[f"layer{li}/block1"] = BasicBlock(cout, cout, 1, f"layer{li}_block1")

    def _stem(self, x, training, dtype):
        c = self.children_
        x = relu(c["stem/bn"](c["stem/conv"](x, dtype), training))
        return x

    def features(self, x, training, dtype):
        x = max_pool_3x3_s2_p1(self._stem(x, training, dtype))
        for li in range(1, 5):
            for b in range(2):
                x = self.children_[f"layer{li}/block{b}"](x, training, dtype)
        return global_avg_pool(x)

    def forward_stages(self, x, dtype):
        out = {}
        x = out["stem_act"] = self._stem(x, False, dtype)
        x = out["maxpool"] = max_pool_3x3_s2_p1(x)
        for li in range(1, 5):
            for b in range(2):
                x = self.children_[f"layer{li}/block{b}"](x, False, dtype)
            out[f"layer{li}"] = x
        out["pooled"] = global_avg_pool(x)
        return out


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


class SqueezeExcite(Layer):
    def __init__(self, c, rd, name):
        super().__init__(name)
        self.children_["reduce"] = Conv2d(c, rd, 1, 1, 0, bias=True, name=name + "_reduce")
        self.children_["expand"] = Conv2d(rd, c, 1, 1, 0, bias=True, name=name + "_expand")

    def __call__(self, x, dtype):
        s = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        s = relu(self.children_["reduce"](s, dtype))
        s = hard_sigmoid(self.children_["expand"](s, dtype))
        return x * s


class MnV3Block(Layer):
    def __init__(self, spec, name):
        super().__init__(name)
        kind, k, s, cin, mid, cout, se_rd, act = spec
        self.kind, self.act = kind, ACTS[act]
        self.skip = s == 1 and cin == cout
        c = self.children_
        if kind == "ds":
            c["conv_dw"] = Conv2d(cin, cin, k, s, (k - 1) // 2, depthwise=True, name=name + "_dw")
            c["bn1"] = BatchNorm(cin, name=name + "_bn1")
            if se_rd:
                c["se"] = SqueezeExcite(cin, se_rd, name + "_se")
            c["conv_pw"] = Conv2d(cin, cout, 1, 1, 0, name=name + "_pw")
            c["bn2"] = BatchNorm(cout, name=name + "_bn2")
        elif kind == "ir":
            c["conv_pw"] = Conv2d(cin, mid, 1, 1, 0, name=name + "_pw")
            c["bn1"] = BatchNorm(mid, name=name + "_bn1")
            c["conv_dw"] = Conv2d(mid, mid, k, s, (k - 1) // 2, depthwise=True, name=name + "_dw")
            c["bn2"] = BatchNorm(mid, name=name + "_bn2")
            if se_rd:
                c["se"] = SqueezeExcite(mid, se_rd, name + "_se")
            c["conv_pwl"] = Conv2d(mid, cout, 1, 1, 0, name=name + "_pwl")
            c["bn3"] = BatchNorm(cout, name=name + "_bn3")
        else:  # cn
            c["conv"] = Conv2d(cin, cout, k, s, 0, name=name + "_conv")
            c["bn1"] = BatchNorm(cout, name=name + "_bn1")

    def __call__(self, x, training, dtype):
        c = self.children_
        if self.kind == "ds":
            y = self.act(c["bn1"](c["conv_dw"](x, dtype), training))
            if "se" in c:
                y = c["se"](y, dtype)
            y = c["bn2"](c["conv_pw"](y, dtype), training)
        elif self.kind == "ir":
            y = self.act(c["bn1"](c["conv_pw"](x, dtype), training))
            y = self.act(c["bn2"](c["conv_dw"](y, dtype), training))
            if "se" in c:
                y = c["se"](y, dtype)
            y = c["bn3"](c["conv_pwl"](y, dtype), training)
        else:
            y = self.act(c["bn1"](c["conv"](x, dtype), training))
        return y + x if self.skip else y


class MobileNetV3Small(Layer):
    feat_dim = 1024

    def __init__(self, name="mobilenetv3_small_100"):
        super().__init__(name)
        self.children_["stem/conv"] = Conv2d(3, 16, 3, 2, 1, name="stem_conv")
        self.children_["stem/bn"] = BatchNorm(16, name="stem_bn")
        self.n_stages = len(MNV3_SMALL)
        for si, stage in enumerate(MNV3_SMALL):
            for bi, spec in enumerate(stage):
                self.children_[f"blocks/{si}/{bi}"] = MnV3Block(spec, f"b{si}_{bi}")
        self.children_["head/conv"] = Conv2d(576, 1024, 1, 1, 0, bias=True, name="head_conv")

    def _stem(self, x, training, dtype):
        c = self.children_
        return hard_swish(c["stem/bn"](c["stem/conv"](x, dtype), training))

    def _stage(self, si, x, training, dtype):
        for bi in range(len(MNV3_SMALL[si])):
            x = self.children_[f"blocks/{si}/{bi}"](x, training, dtype)
        return x

    def _head(self, x, dtype):
        p = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # (B,1,1,576)
        h = hard_swish(self.children_["head/conv"](p, dtype))
        return tf.reshape(h, [-1, self.feat_dim])

    def features(self, x, training, dtype):
        x = self._stem(x, training, dtype)
        for si in range(self.n_stages):
            x = self._stage(si, x, training, dtype)
        return self._head(x, dtype)

    def forward_stages(self, x, dtype):
        out = {}
        x = out["stem_act"] = self._stem(x, False, dtype)
        for si in range(self.n_stages):
            x = out[f"blocks{si}"] = self._stage(si, x, False, dtype)
        out["pre_logits"] = self._head(x, dtype)
        return out


# ---------------------------------------------------------------- convnext_small
class ConvNeXtBlock(Layer):
    def __init__(self, c, name):
        super().__init__(name)
        self.children_["dwconv"] = Conv2d(c, c, 7, 1, 3, bias=True, depthwise=True, name=name + "_dw")
        self.children_["ln"] = LayerNorm(c, 1e-6, name=name + "_ln")
        self.children_["mlp/fc1"] = Dense(c, 4 * c, name=name + "_fc1")
        self.children_["mlp/fc2"] = Dense(4 * c, c, name=name + "_fc2")
        self.leaves["gamma"] = tf.Variable(np.full((c,), 1e-6, np.float32), name=name + "_gamma")

    def __call__(self, x, dtype):
        c = self.children_
        y = c["dwconv"](x, dtype)
        y = c["ln"](y)
        y = c["mlp/fc2"](tf.nn.gelu(c["mlp/fc1"](y, dtype), approximate=False), dtype)
        return x + y * tf.cast(self.leaves["gamma"], dtype)


class ConvNeXtSmall(Layer):
    feat_dim = 768
    depths, dims = (3, 3, 27, 3), (96, 192, 384, 768)

    def __init__(self, name="convnext_small"):
        super().__init__(name)
        c = self.children_
        c["stem/conv"] = Conv2d(3, 96, 4, 4, 0, bias=True, name="stem_conv")
        c["stem/ln"] = LayerNorm(96, 1e-6, name="stem_ln")
        for si, (d, dim) in enumerate(zip(self.depths, self.dims)):
            if si > 0:
                c[f"stages/{si}/down/ln"] = LayerNorm(self.dims[si - 1], 1e-6, name=f"s{si}_down_ln")
                c[f"stages/{si}/down/conv"] = Conv2d(self.dims[si - 1], dim, 2, 2, 0, bias=True, name=f"s{si}_down_conv")
            for bi in range(d):
                c[f"stages/{si}/blocks/{bi}"] = ConvNeXtBlock(dim, f"s{si}_b{bi}")
        c["head/ln"] = LayerNorm(768, 1e-6, name="head_ln")

    def _stem(self, x, dtype):
        c = self.children_
        return c["stem/ln"](c["stem/conv"](x, dtype))

    def _stage(self, si, x, dtype):
        c = self.children_
        if si > 0:
            x = c[f"stages/{si}/down/conv"](c[f"stages/{si}/down/ln"](x), dtype)
        for bi in range(self.depths[si]):
            x = c[f"stages/{si}/blocks/{bi}"](x, dtype)
        return x

    def _head(self, x):
        return self.children_["head/ln"](global_avg_pool(x))

    def features(self, x, training, dtype):
        x = self._stem(x, dtype)
        for si in range(4):
            x = self._stage(si, x, dtype)
        return self._head(x)

    def forward_stages(self, x, dtype):
        out = {}
        x = out["stem"] = self._stem(x, dtype)
        for si in range(4):
            x = out[f"stages{si}"] = self._stage(si, x, dtype)
        out["pre_logits"] = self._head(x)
        return out


BACKBONES = {"resnet18": ResNet18, "mobilenetv3_small_100": MobileNetV3Small, "convnext_small": ConvNeXtSmall}


def build_backbone(name):
    return BACKBONES[name]()


# ---------------------------------------------------------------- classifier
class Classifier(Layer):
    """backbone -> dropout -> Dense(num_classes); logits always fp32 (mixed-precision safe)."""

    def __init__(self, backbone, num_classes, dropout_rate, name="classifier"):
        super().__init__(name)
        self.children_["backbone"] = backbone
        self.children_["head/dense"] = Dense(backbone.feat_dim, num_classes, name="head_dense")
        self.dropout_rate = float(dropout_rate)
        self.backbone = backbone

    def __call__(self, x, training, dtype, rng=None, backbone_training=None):
        bt = training if backbone_training is None else backbone_training
        f = self.backbone.features(tf.cast(x, dtype), bt, dtype)
        f = tf.cast(f, tf.float32)
        if training and self.dropout_rate > 0:
            f = dropout(f, self.dropout_rate, rng)
        return self.children_["head/dense"](f, tf.float32)


def build_model(name, num_classes, dropout_rate, pretrained_npz=None, trunk_npz=None, verbose=True):
    """Classifier with ImageNet (pretrained_npz) or Stage-1 (trunk_npz) backbone weights.
    Both loads are strict on the backbone key set (the torch load_trunk guarantee)."""
    backbone = build_backbone(name)
    model = Classifier(backbone, num_classes, dropout_rate)
    src = trunk_npz or pretrained_npz
    if src:
        flat = load_npz(src)
        expected = backbone.flat().keys()
        assert_same_keys(expected, flat.keys(), what=f"{src} vs {name} backbone")
        backbone.load_flat(flat)
        if verbose:
            print(f"loaded {'patch-pretrained trunk' if trunk_npz else 'ImageNet weights'} from {src} "
                  f"({len(flat)} arrays; only the head is random)")
    return model, backbone
