"""Convert timm ImageNet weights to the framework-neutral .npz used by BOTH bare ports, and dump
reference activations for the parity tests. Runs ONLY in the torch venv (torch + timm).

    python tools/convert_timm_to_npz.py --backbone resnet18 --out-dir ../../pretrained
    python tools/convert_timm_to_npz.py --all --out-dir ../../pretrained

Outputs per backbone:
    <out-dir>/<backbone>.npz      backbone weights, structural keys (see npzio.py), NHWC/HWIO
    <out-dir>/ref_<backbone>.npz  random inputs (2,224,224,3) and (2,250,250,3), every stage
                                  output (NHWC fp32), logits, the 1000-way fc as fc/kernel,fc/bias
    <out-dir>/PRETRAINED_MANIFEST.json   timm tags, versions, sha256 of each npz

Layout transforms (torch -> TF/JAX):  conv (O,I,kH,kW) -> (kH,kW,I,O); depthwise (C,1,k,k) ->
(k,k,C,1); linear (out,in) -> (in,out); BN/LN/layer-scale vectors copied; num_batches_tracked
dropped. Weight provenance = the torch runs' (timm) -- resnet18.a1_in1k is byte-identical to
the KerasHub preset the Keras runs used.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import timm
import torch

TIMM_TAGS = {
    "resnet18": "resnet18.a1_in1k",
    "mobilenetv3_small_100": "mobilenetv3_small_100.lamb_in1k",
    "convnext_small": "convnext_small.in12k_ft_in1k",
}
BN = {"weight": "scale", "bias": "bias", "running_mean": "mean", "running_var": "var"}
LN = {"weight": "scale", "bias": "bias"}
CONV = {"weight": "kernel", "bias": "bias"}


def _sub(rules, key):
    for pat, fn in rules:
        m = re.fullmatch(pat, key)
        if m:
            return fn(m)
    raise KeyError(f"no rename rule for {key}")


def rules_resnet18():
    return [
        (r"conv1\.weight", lambda m: "stem/conv/kernel"),
        (r"bn1\.(\w+)", lambda m: f"stem/bn/{BN[m[1]]}"),
        (r"layer(\d)\.(\d)\.conv(\d)\.weight", lambda m: f"layer{m[1]}/block{m[2]}/conv{m[3]}/kernel"),
        (r"layer(\d)\.(\d)\.bn(\d)\.(\w+)", lambda m: f"layer{m[1]}/block{m[2]}/bn{m[3]}/{BN[m[4]]}"),
        (r"layer(\d)\.0\.downsample\.0\.weight", lambda m: f"layer{m[1]}/block0/down/conv/kernel"),
        (r"layer(\d)\.0\.downsample\.1\.(\w+)", lambda m: f"layer{m[1]}/block0/down/bn/{BN[m[2]]}"),
        (r"fc\.(weight|bias)", lambda m: f"fc/{CONV[m[1]]}"),
    ]


def rules_mobilenetv3():
    return [
        (r"conv_stem\.weight", lambda m: "stem/conv/kernel"),
        (r"bn1\.(\w+)", lambda m: f"stem/bn/{BN[m[1]]}"),
        (r"blocks\.(\d)\.(\d)\.(conv_pw|conv_dw|conv_pwl|conv)\.weight",
         lambda m: f"blocks/{m[1]}/{m[2]}/{m[3]}/kernel"),
        (r"blocks\.(\d)\.(\d)\.bn(\d)\.(\w+)", lambda m: f"blocks/{m[1]}/{m[2]}/bn{m[3]}/{BN[m[4]]}"),
        (r"blocks\.(\d)\.(\d)\.se\.conv_(reduce|expand)\.(weight|bias)",
         lambda m: f"blocks/{m[1]}/{m[2]}/se/{m[3]}/{CONV[m[4]]}"),
        (r"conv_head\.(weight|bias)", lambda m: f"head/conv/{CONV[m[1]]}"),
        (r"classifier\.(weight|bias)", lambda m: f"fc/{CONV[m[1]]}"),
    ]


def rules_convnext():
    return [
        (r"stem\.0\.(weight|bias)", lambda m: f"stem/conv/{CONV[m[1]]}"),
        (r"stem\.1\.(weight|bias)", lambda m: f"stem/ln/{LN[m[1]]}"),
        (r"stages\.(\d)\.downsample\.0\.(weight|bias)", lambda m: f"stages/{m[1]}/down/ln/{LN[m[2]]}"),
        (r"stages\.(\d)\.downsample\.1\.(weight|bias)", lambda m: f"stages/{m[1]}/down/conv/{CONV[m[2]]}"),
        (r"stages\.(\d)\.blocks\.(\d+)\.gamma", lambda m: f"stages/{m[1]}/blocks/{m[2]}/gamma"),
        (r"stages\.(\d)\.blocks\.(\d+)\.conv_dw\.(weight|bias)",
         lambda m: f"stages/{m[1]}/blocks/{m[2]}/dwconv/{CONV[m[3]]}"),
        (r"stages\.(\d)\.blocks\.(\d+)\.norm\.(weight|bias)",
         lambda m: f"stages/{m[1]}/blocks/{m[2]}/ln/{LN[m[3]]}"),
        (r"stages\.(\d)\.blocks\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)",
         lambda m: f"stages/{m[1]}/blocks/{m[2]}/mlp/{m[3]}/{CONV[m[4]]}"),
        (r"head\.norm\.(weight|bias)", lambda m: f"head/ln/{LN[m[1]]}"),
        (r"head\.fc\.(weight|bias)", lambda m: f"fc/{CONV[m[1]]}"),
    ]


RULES = {"resnet18": rules_resnet18, "mobilenetv3_small_100": rules_mobilenetv3, "convnext_small": rules_convnext}


def transform(key, t, groups_of):
    """torch tensor -> numpy in the TF/JAX layout implied by the new key."""
    a = t.detach().cpu().numpy().astype(np.float32)
    if a.ndim == 4:
        if groups_of(key) > 1:  # depthwise (C,1,k,k) -> (k,k,C,1)
            assert a.shape[1] == 1
            return np.ascontiguousarray(a.transpose(2, 3, 0, 1))
        return np.ascontiguousarray(a.transpose(2, 3, 1, 0))  # (O,I,k,k) -> (k,k,I,O)
    if a.ndim == 2:
        return np.ascontiguousarray(a.T)  # linear (out,in) -> (in,out)
    return a


def convert(name, out_dir):
    tag = TIMM_TAGS[name]
    model = timm.create_model(tag, pretrained=True).eval()
    # groups per conv module, keyed by parameter name prefix
    conv_groups = {n + ".weight": m.groups for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)}
    rules = RULES[name]()
    flat, fc = {}, {}
    n_src = 0
    for k, t in model.state_dict().items():
        if k.endswith("num_batches_tracked"):
            continue
        n_src += 1
        nk = _sub(rules, k)
        arr = transform(k, t, lambda kk: conv_groups.get(kk, 1))
        (fc if nk.startswith("fc/") else flat)[nk] = arr
    assert len(flat) + len(fc) == n_src, "every source key must be consumed exactly once"
    size_mb = sum(v.nbytes for v in flat.values()) / 1e6
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez(path, **flat)
    print(f"{name}: {tag} -> {path}: {len(flat)} backbone arrays ({size_mb:.6f} MB), fc {len(fc)} arrays")
    return model, tag, flat, fc, path, size_mb


def dump_reference(name, model, fc, out_dir, tag, npz_sha):
    """Stage-wise activations on two fixed random inputs, in NHWC fp32."""
    hooks = {
        "resnet18": {"stem_act": "act1", "maxpool": "maxpool", "layer1": "layer1", "layer2": "layer2",
                     "layer3": "layer3", "layer4": "layer4", "pooled": "global_pool"},
        "mobilenetv3_small_100": {"stem_act": "bn1", **{f"blocks{i}": f"blocks.{i}" for i in range(6)},
                                  "pre_logits": "act2"},
        "convnext_small": {"stem": "stem", **{f"stages{i}": f"stages.{i}" for i in range(4)},
                           "pre_logits": "head.norm"},
    }[name]
    mods = dict(model.named_modules())
    captured = {}

    def mk(label):
        def hook(_m, _i, out):
            o = out.detach().float().cpu().numpy()
            if o.ndim == 4:
                o = o.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            captured[label] = np.ascontiguousarray(o)
        return hook

    handles = [mods[m].register_forward_hook(mk(label)) for label, m in hooks.items()]
    out = {"fc/kernel": fc["fc/kernel"], "fc/bias": fc["fc/bias"]}
    rng = np.random.default_rng(0)
    for s in (224, 250):
        x1 = rng.standard_normal((2, s, s, 1)).astype(np.float32)
        x = np.repeat(x1, 3, axis=3)  # z-score-like, 3 identical channels (like the real input)
        out[f"input_{s}"] = x
        captured.clear()
        with torch.no_grad():
            logits = model(torch.from_numpy(x.transpose(0, 3, 1, 2).copy()))
        for label, arr in captured.items():
            out[f"{s}/{label}"] = arr
        out[f"{s}/logits"] = logits.float().cpu().numpy()
    for h in handles:
        h.remove()
    meta = {"backbone": name, "timm_tag": tag, "timm": timm.__version__, "torch": torch.__version__,
            "pretrained_sha256": npz_sha, "stages": list(hooks)}
    out["meta"] = np.array(json.dumps(meta))
    path = os.path.join(out_dir, f"ref_{name}.npz")
    np.savez(path, **out)
    print(f"  reference -> {path}: {', '.join(k for k in out if k.startswith('224/'))}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=list(TIMM_TAGS), default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default="pretrained")
    a = ap.parse_args()
    names = list(TIMM_TAGS) if a.all else [a.backbone]
    if not names or names == [None]:
        sys.exit("pass --backbone <name> or --all")
    os.makedirs(a.out_dir, exist_ok=True)
    mpath = os.path.join(a.out_dir, "PRETRAINED_MANIFEST.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for name in names:
        model, tag, flat, fc, path, size_mb = convert(name, a.out_dir)
        sha = sha256(path)
        dump_reference(name, model, fc, a.out_dir, tag, sha)
        manifest[name] = {"timm_tag": tag, "file": os.path.basename(path), "sha256": sha,
                          "n_arrays": len(flat), "size_mb": size_mb,
                          "timm": timm.__version__, "torch": torch.__version__}
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
