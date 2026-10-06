# Pretrained weights for the plain-TensorFlow and plain-JAX ports

The Keras-free ports (`training_tf/`, `training_jax/`) do not download weights at run time. They read
framework-neutral `.npz` files converted once from the same timm checkpoints the PyTorch runs used:

| file | timm source | arrays |
|---|---|---|
| `resnet18.npz` | `resnet18.a1_in1k` | 120 |
| `mobilenetv3_small_100.npz` | `mobilenetv3_small_100.lamb_in1k` | 240 |
| `convnext_small.npz` | `convnext_small.in12k_ft_in1k` | 342 |

plus one `ref_<backbone>.npz` per model holding reference activations used by the parity tests
(`parity_test_tf.py`, `parity_test_jax.py`). `PRETRAINED_MANIFEST.json` lists the exact checkpoints and SHA-256
hashes of the conversion.

The `.npz` files are about 300 MB in total and are **not committed**. Either

1. download `vindr_mammo_pretrained_npz.tar` from the GitHub *Releases* page of this repository and extract it
   here (it unpacks to `pretrained/`), or
2. rebuild them in a PyTorch environment with timm installed:

```bash
python common/tools/convert_timm_to_npz.py --all --out-dir pretrained
```

Both ports look for the files under `<port dir>/pretrained/` (or `PRETRAINED_DIR`); the simplest is a symlink:
`ln -s ../pretrained training_tf/pretrained` and the same for `training_jax`.
