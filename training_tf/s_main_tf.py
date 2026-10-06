"""VinDr-Mammo — binary "suspicious" classifier (Stage-2 full-image), bare-TensorFlow port.

    label 0 = normal/benign   -> BI-RADS 1-2
    label 1 = suspicious      -> BI-RADS 3-5

Same CLI, run-tag format and 49-column results row as s_main.py / s_main_keras.py, so rows
concatenate with the PyTorch and Keras results for one Pareto chart. Differences to the Keras
port: no Keras/tf.data (shared numpy worker-pool loader, one tf.function train step), timm
ImageNet weights from pretrained/<backbone>.npz (the torch runs' provenance).

Run e.g.:
    python s_main_tf.py --folder ../training_pytorch/ --img-size 768 --batch 16 --cache-ram
    python s_main_tf.py --pretrained-trunk trunk_tf_resnet18.npz --folder ../training_pytorch/ \\
        --img-size 768 --batch 16 --quantize fp16 --cache-ram
"""
import os
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cli import add_xla_flag, stage2_parser
from energy import read_sidecar, track_energy
from evaluate import evaluate_stage2
from lr import WarmupCosineSpec
from npzio import cast_fp16, load_npz, save_npz, size_mb
from pipeline_np import (
    EvalLoader, TrainLoader, WorkerPool, assert_ram_fits, decode_all, limit_df, loader_buffer_bytes,
    split_trainval,
)
from results import append_results, build_row, run_tag


def parse_args():
    return stage2_parser(__doc__, "results_tf.csv", "trunk_tf_<backbone>.npz", extra=add_xla_flag).parse_args()


def train(args, tf, T, model, train_loader, val_loader, val_labels, ckpt, rng, amp_on):
    """Epoch loop: per-epoch val AUROC, save best (same selection as torch/Keras). Returns best."""
    steps_per_epoch = len(train_loader)
    spec = WarmupCosineSpec(args.lr, args.warmup_epochs, args.epochs, steps_per_epoch)
    head_only = args.train_mode == "head"
    opt = T.Optimizer(T.trainable_variables(model, head_only), args.optimizer, spec,
                      args.weight_decay, args.momentum)
    loss_scale = T.DynamicLossScale()
    step_fn = T.make_train_step(model, opt, loss_scale, args.batch, args.img_size, amp_on, args.xla,
                                rng, head_only)
    compute_dtype = tf.float16 if amp_on else tf.float32
    predict_fn = T.make_predict_fn(model, compute_dtype, args.batch, args.img_size, args.xla)
    best, first = 0.0, True
    for epoch in range(args.epochs):
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            loss, finite = step_fn(tf.constant(x), tf.constant(y))
            if first:
                print(f"  first step (incl. trace/compile): {time.time() - t0:.1f}s")
                first = False
            if i % 200 == 0:
                rate = (i + 1) * args.batch / max(time.time() - t0, 1e-9)
                print(f"\r  epoch {epoch:02d}  batch {i:4d}/{steps_per_epoch}  loss={float(loss):.3f}  "
                      f"{rate:.0f} img/s", end="", flush=True)
        train_secs = time.time() - t0
        probs = T.predict_probs(predict_fn, val_loader)
        acc = ((probs > 0.5) == val_labels).mean()
        auc = roc_auc_score(val_labels, probs)
        print(f"\repoch {epoch:02d}  lr={float(opt.lr()):.2e}  scale={float(loss_scale.scale):.0f}  "
              f"val_acc={acc:.3f}  val_auroc={auc:.3f}  ({train_secs:.0f}s train)")
        if auc > best:
            best = auc
            save_npz(ckpt, model.export_flat())
    print("best val AUROC:", round(best, 3))
    return best


def main():
    args = parse_args()
    if args.png_index is None:
        args.png_index = args.folder + "png_index.csv"
    if args.ckpt is None:
        args.ckpt = f"best_{run_tag(args)}.npz"  # per-TAG, written to the CURRENT dir
    pretrained = None if args.no_pretrained else os.path.join(args.pretrained_dir, f"{args.backbone}.npz")
    if pretrained and not os.path.exists(pretrained):
        raise SystemExit(f"missing {pretrained} -- see pretrained/README.md (release asset or common/tools/convert_timm_to_npz.py)")
    assert os.path.exists(args.png_index), f"{args.png_index} not found."

    df = pd.read_csv(args.png_index)
    train_df, val_df = split_trainval(df)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        train_df, val_df, test_df = (limit_df(d, args.limit) for d in (train_df, val_df, test_df))
        print(f"DEBUG --limit {args.limit}: subsetting all splits")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(d)} images | normal/benign={int((d.label == 0).sum())} "
              f"suspicious={int((d.label == 1).sum())}")

    tr_cache = va_cache = te_cache = None
    if args.cache_ram:
        cw = args.cache_workers or args.workers
        cache_dfs = [train_df, val_df] + ([test_df] if args.cache_test else [])
        cache_bytes = sum(len(d) for d in cache_dfs) * args.img_size * args.img_size
        buffer_bytes = loader_buffer_bytes(args.workers, args.prefetch, args.batch, args.img_size)
        assert_ram_fits(cache_bytes + buffer_bytes, force=args.force_cache)
        tr_cache = decode_all(train_df, args.img_size, cw, args.folder, "train cache")
        va_cache = decode_all(val_df, args.img_size, cw, args.folder, "val cache")
        if args.cache_test:
            te_cache = decode_all(test_df, args.img_size, cw, args.folder, "test cache")

    # loaders + worker pool BEFORE any TensorFlow GPU initialisation (fork safety)
    pool = WorkerPool(args.workers)
    train_loader = TrainLoader(train_df, args.img_size, args.batch, args.seed, args.folder, tr_cache, pool, args.prefetch)
    val_loader = EvalLoader(val_df, args.img_size, args.batch, args.folder, va_cache, pool)
    test_loader = EvalLoader(test_df, args.img_size, args.batch, args.folder, te_cache, pool)
    pool.start()
    val_labels = val_df["label"].astype(int).to_numpy()

    import tensorflow as tf  # noqa: E402,PLC0415 (after the fork)
    import train_tf as T  # noqa: E402,PLC0415
    from backbones_tf import build_model  # noqa: E402,PLC0415

    gpu = T.setup_gpu()
    amp_on = args.amp and gpu
    print(f"device: {'gpu' if gpu else 'cpu'}  backbone: {args.backbone}  img_size: {args.img_size}  "
          f"batch: {args.batch}  amp: {amp_on}  xla: {args.xla}  trunk: {args.pretrained_trunk or 'imagenet'}")
    print(f"optimizer: {args.optimizer}(mom={args.momentum})  train_mode: {args.train_mode}  "
          f"quantize: {args.quantize}  wd: {args.weight_decay}  dropout: {args.dropout}  "
          f"seed: {args.seed}  rounds: {args.rounds}  workers: {args.workers}")

    # Best-of-N rounds: train `--rounds` seeded models, keep best val AUROC. Energy wraps all rounds.
    stem = args.ckpt[:-len(".npz")] if args.ckpt.endswith(".npz") else args.ckpt
    best_round_auc, best_ckpt = -1.0, args.ckpt
    with track_energy(f"stage2 train ({args.backbone}, {args.rounds} round(s))", enabled=args.energy) as train_e:
        for r in range(args.rounds):
            rng = T.set_seed(args.seed + r)
            round_ckpt = args.ckpt if args.rounds == 1 else f"{stem}_r{r}.npz"
            if args.rounds > 1:
                print(f"\n=== round {r + 1}/{args.rounds}  seed={args.seed + r}  -> {round_ckpt} ===")
            model, _ = build_model(args.backbone, 2, args.dropout, pretrained, args.pretrained_trunk)
            auc = train(args, tf, T, model, train_loader, val_loader, val_labels, round_ckpt, rng, amp_on)
            if auc > best_round_auc:
                best_round_auc, best_ckpt = auc, round_ckpt
    if args.rounds > 1:
        print(f"\nbest round: val AUROC {best_round_auc:.3f} -> {best_ckpt}")
        shutil.copyfile(best_ckpt, args.ckpt)

    # evaluation: fresh model + best checkpoint; fp16 copy = fp16 variables AND fp16 compute
    model, _ = build_model(args.backbone, 2, args.dropout, None, None, verbose=False)
    flat = load_npz(best_ckpt)
    model.load_flat(flat)
    compute_dtype = tf.float16 if amp_on else tf.float32
    pred32 = T.make_predict_fn(model, compute_dtype, args.batch, args.img_size, args.xla)
    predict32 = lambda loader: T.predict_probs(pred32, loader)  # noqa: E731
    predict16, size16, fp16_ckpt = None, "", None
    if args.quantize == "fp16":
        flat16 = cast_fp16(flat)
        fp16_ckpt = f"{stem}_fp16.npz"
        save_npz(fp16_ckpt, flat16)
        qmodel, _ = build_model(args.backbone, 2, args.dropout, None, None, verbose=False)
        qmodel.cast_variables(tf.float16)
        qmodel.load_flat(flat16)
        pred16 = T.make_predict_fn(qmodel, tf.float16, args.batch, args.img_size, args.xla)
        predict16 = lambda loader: T.predict_probs(pred16, loader)  # noqa: E731
        size16 = size_mb(flat16)
    row, infer_e = evaluate_stage2(args, predict32, predict16, size_mb(flat), size16, val_loader, val_labels,
                                   test_loader, test_df, fp16_ckpt)
    row = build_row(args, amp_on, best_round_auc, row, train_e, infer_e, read_sidecar(args.pretrained_trunk))
    append_results(args.results_csv, row)
    pool.close()


if __name__ == "__main__":
    main()
