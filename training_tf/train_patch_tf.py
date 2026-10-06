"""Stage-1 (bare-TensorFlow port): train a 4-class patch classifier, export its conv trunk.

background / birads3 / birads4 / birads5 on 224px patches, so the backbone learns lesion
texture. Only the backbone weights (head discarded) are saved to `trunk_tf_<backbone>.npz`
(+ `.energy.json` sidecar), which s_main_tf.py loads via `--pretrained-trunk`. Selection on
val top-1, energy wraps the whole epoch loop -- as in train_patch.py / train_patch_keras.py.

    python train_patch_tf.py --folder ../training_pytorch/ --cache-ram
    python train_patch_tf.py --backbone convnext_small --folder ../training_pytorch/ --cache-ram
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from cli import add_xla_flag, stage1_parser
from energy import track_energy, write_sidecar
from lr import WarmupCosineSpec
from npzio import save_npz
from pipeline_np import (
    EvalLoader, TrainLoader, WorkerPool, assert_ram_fits, decode_all, limit_df, loader_buffer_bytes,
)


def parse_args():
    return stage1_parser(__doc__, "trunk_tf_<backbone>.npz", extra=add_xla_flag).parse_args()


def main():
    args = parse_args()
    if args.patch_index is None:
        args.patch_index = args.folder + "patch_index.csv"
    if args.trunk_out is None:
        args.trunk_out = f"trunk_tf_{args.backbone}.npz"
    pretrained = None if args.no_pretrained else os.path.join(args.pretrained_dir, f"{args.backbone}.npz")
    if pretrained and not os.path.exists(pretrained):
        raise SystemExit(f"missing {pretrained} -- see pretrained/README.md (release asset or common/tools/convert_timm_to_npz.py)")
    assert os.path.exists(args.patch_index), f"{args.patch_index} not found."

    df = pd.read_csv(args.patch_index)
    train_df = df[df.split == "patch_train"].reset_index(drop=True)
    val_df = df[df.split == "patch_val"].reset_index(drop=True)
    if args.limit:
        train_df, val_df = limit_df(train_df, args.limit), limit_df(val_df, args.limit)
        print(f"DEBUG --limit {args.limit}: subsetting both splits")
    print(f"patch_train: {len(train_df)}  patch_val: {len(val_df)}  classes: {sorted(df.label.unique().tolist())}")

    tr_cache = va_cache = None
    if args.cache_ram:
        total = (len(train_df) + len(val_df)) * args.img_size * args.img_size
        total += loader_buffer_bytes(args.workers, args.prefetch, args.batch, args.img_size, n_loaders=2)
        assert_ram_fits(total, force=args.force_cache)
        tr_cache = decode_all(train_df, args.img_size, args.workers, args.folder, "patch train cache")
        va_cache = decode_all(val_df, args.img_size, args.workers, args.folder, "patch val cache")

    pool = WorkerPool(args.workers)
    train_loader = TrainLoader(train_df, args.img_size, args.batch, args.seed, args.folder, tr_cache, pool, args.prefetch)
    val_loader = EvalLoader(val_df, args.img_size, args.batch, args.folder, va_cache, pool)
    pool.start()
    val_labels = val_df["label"].astype(int).to_numpy()
    steps_per_epoch = len(train_loader)

    import tensorflow as tf  # noqa: E402,PLC0415 (after the fork)
    import train_tf as T  # noqa: E402,PLC0415
    from backbones_tf import build_model  # noqa: E402,PLC0415

    gpu = T.setup_gpu()  # before any op creates the GPU context (memory growth must be set first)
    rng = T.set_seed(args.seed)
    amp_on = args.amp and gpu
    print(f"device: {'gpu' if gpu else 'cpu'}  backbone: {args.backbone}  optimizer: {args.optimizer}  "
          f"amp: {amp_on}  xla: {args.xla}  seed: {args.seed}  wd: {args.weight_decay}")

    model, backbone = build_model(args.backbone, 4, 0.0, pretrained, None)
    spec = WarmupCosineSpec(args.lr, args.warmup_epochs, args.epochs, steps_per_epoch)
    opt = T.Optimizer(T.trainable_variables(model), args.optimizer, spec, args.weight_decay, args.momentum)
    loss_scale = T.DynamicLossScale()
    step_fn = T.make_train_step(model, opt, loss_scale, args.batch, args.img_size, amp_on, args.xla, rng)
    predict_fn = T.make_predict_fn(model, tf.float16 if amp_on else tf.float32, args.batch, args.img_size, args.xla)

    best, first = 0.0, True
    with track_energy(f"stage1 train ({args.backbone})", enabled=args.energy) as energy:
        for epoch in range(args.epochs):
            t0 = time.time()
            for i, (x, y) in enumerate(train_loader):
                loss, _ = step_fn(tf.constant(x), tf.constant(y))
                if first:
                    print(f"  first step (incl. trace/compile): {time.time() - t0:.1f}s")
                    first = False
                if i % 100 == 0:
                    print(f"\r  epoch {epoch:02d}  batch {i:4d}/{steps_per_epoch}  loss={float(loss):.3f}",
                          end="", flush=True)
            logits = T.predict_logits(predict_fn, val_loader)
            preds = np.argmax(logits, axis=1)
            acc = accuracy_score(val_labels, preds) if len(val_labels) else 0.0
            print(f"\repoch {epoch:02d}  lr={float(opt.lr()):.2e}  val_top1={acc:.3f}  ({time.time() - t0:.0f}s)")
            if acc > best:
                best = acc
                save_npz(args.trunk_out, backbone.export_flat())  # backbone only, head discarded

    print(f"best val top-1: {best:.3f}  (chance=0.25)")
    print(f"wrote conv trunk -> {args.trunk_out}")
    write_sidecar(args.trunk_out, args.backbone, best, energy)
    pool.close()


if __name__ == "__main__":
    main()
