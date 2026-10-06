"""Stage-1 (Keras 3 / TF port): train a patch classifier, then export its conv trunk.

4-class classification (background / birads3 / birads4 / birads5) on 224px patches, so the
backbone learns mammographic lesion texture instead of relying on raw ImageNet features.
Only the trunk (the backbone sub-model, head discarded) is reused downstream.

The backbone MUST match the Stage-2 backbone for the trunk to transfer -- both stages build
it through pipeline_keras.build_backbone. Saves `trunk_keras_<backbone>.weights.h5` + a
`*.energy.json` sidecar, which s_main_keras.py loads via `--pretrained-trunk`.

Run (patches live in the original experiment dir):
    python train_patch_keras.py --folder ../training_pytorch/ --cache-ram
    python train_patch_keras.py --backbone convnext_small --folder ../training_pytorch/
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from pipeline_keras import (
    WarmupCosine, assert_ram_fits, build_model, build_optimizer, decode_all,
    energy_sidecar_path, make_eval_ds, make_train_ds, setup_gpu, track_energy,
)
import keras


def set_seed(s):
    keras.utils.set_random_seed(s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default="", help="prefix prepended to patch_index.csv and patch paths")
    p.add_argument("--patch-index", default=None, help="default: <folder>patch_index.csv")
    p.add_argument("--backbone", default="resnet18",
                   help="resnet18 / mobilenetv3_small_100 / convnext_small")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4, help="peak LR (cosine decays from here)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd", "rmsprop"])
    p.add_argument("--momentum", type=float, default=0.9, help="for sgd/rmsprop")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8, help="decode threads for --cache-ram")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-energy", dest="energy", action="store_false", help="disable CodeCarbon tracking")
    p.add_argument("--cache-ram", action="store_true", help="preload patches decoded into RAM")
    p.add_argument("--force-cache", action="store_true", help="bypass the RAM-fits guard")
    p.add_argument("--trunk-out", default=None, help="default: trunk_keras_<backbone>.weights.h5")
    p.add_argument("--limit", type=int, default=None,
                   help="DEBUG: stratified subset of N rows per split (smoke tests only)")
    return p.parse_args()


def limit_df(d, n):
    per = max(1, n // max(d["label"].nunique(), 1))
    return d.groupby("label", group_keys=False).head(per).head(n).reset_index(drop=True)


class Top1TrunkCheckpoint(keras.callbacks.Callback):
    """Per-epoch val top-1; on improvement, save the TRUNK (backbone weights only, head
    discarded) -- same selection + export as the torch train_patch.py."""

    def __init__(self, backbone, val_ds, val_labels, trunk_out, schedule=None):
        super().__init__()
        self.backbone, self.val_ds, self.val_labels = backbone, val_ds, val_labels
        self.trunk_out, self.schedule = trunk_out, schedule
        self.best, self.t0 = 0.0, time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logits = self.model.predict(self.val_ds, verbose=0)
        preds = np.argmax(np.asarray(logits), axis=1)
        acc = accuracy_score(self.val_labels, preds) if len(self.val_labels) else 0.0
        lr = ""
        if self.schedule is not None:
            try:
                lr_v = keras.ops.convert_to_numpy(self.schedule(self.model.optimizer.iterations))
                lr = f"lr={float(lr_v):.2e}  "
            except Exception:  # noqa: BLE001
                pass
        print(f"\repoch {epoch:02d}  {lr}val_top1={acc:.3f}  ({time.time() - self.t0:.0f}s)")
        if acc > self.best:
            self.best = acc
            self.backbone.save_weights(self.trunk_out)


class Heartbeat(keras.callbacks.Callback):
    def __init__(self, n_batches, every=100):
        super().__init__()
        self.n_batches, self.every, self.epoch = n_batches, every, 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.every == 0:
            loss = (logs or {}).get("loss", float("nan"))
            print(f"\r  epoch {self.epoch:02d}  batch {batch:4d}/{self.n_batches}  loss={loss:.3f}",
                  end="", flush=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.patch_index is None:
        args.patch_index = args.folder + "patch_index.csv"
    if args.trunk_out is None:
        args.trunk_out = f"trunk_keras_{args.backbone}.weights.h5"

    gpu = setup_gpu()
    amp_on = args.amp and gpu
    if amp_on:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"device: {'gpu' if gpu else 'cpu'}  backbone: {args.backbone}  optimizer: {args.optimizer}  "
          f"amp: {amp_on}  seed: {args.seed}  wd: {args.weight_decay}")
    assert os.path.exists(args.patch_index), f"{args.patch_index} not found."

    df = pd.read_csv(args.patch_index)
    train_df = df[df.split == "patch_train"].reset_index(drop=True)
    val_df = df[df.split == "patch_val"].reset_index(drop=True)
    if args.limit:
        train_df, val_df = limit_df(train_df, args.limit), limit_df(val_df, args.limit)
        print(f"DEBUG --limit {args.limit}: subsetting both splits")
    print(f"patch_train: {len(train_df)}  patch_val: {len(val_df)}  "
          f"classes: {sorted(df.label.unique().tolist())}")

    tr_cache = va_cache = None
    if args.cache_ram:
        total = (len(train_df) + len(val_df)) * args.img_size * args.img_size
        assert_ram_fits(total, force=args.force_cache)
        tr_cache = decode_all(train_df, args.img_size, args.workers, args.folder, "patch train cache")
        va_cache = decode_all(val_df, args.img_size, args.workers, args.folder, "patch val cache")

    train_ds = make_train_ds(train_df, args.img_size, args.batch, args.seed, args.folder, tr_cache)
    val_ds = make_eval_ds(val_df, args.img_size, args.batch, args.folder, va_cache)
    val_labels = val_df["label"].astype(int).to_numpy()
    steps_per_epoch = max(len(train_df) // args.batch, 1)

    model, backbone = build_model(args.backbone, 4, args.img_size, dropout=0.0)
    schedule = WarmupCosine(args.lr, args.warmup_epochs, args.epochs, steps_per_epoch)
    optimizer = build_optimizer(args.optimizer, schedule, args.weight_decay, args.momentum)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    ckpt_cb = Top1TrunkCheckpoint(backbone, val_ds, val_labels, args.trunk_out, schedule)
    with track_energy(f"stage1 train ({args.backbone})", enabled=args.energy) as energy:
        model.fit(train_ds, epochs=args.epochs, steps_per_epoch=steps_per_epoch, verbose=0,
                  callbacks=[Heartbeat(steps_per_epoch), ckpt_cb])

    print(f"best val top-1: {ckpt_cb.best:.3f}  (chance=0.25)")
    print(f"wrote conv trunk -> {args.trunk_out}")

    # sidecar so s_main_keras.py can fold Stage-1 energy into its results row
    sidecar = energy_sidecar_path(args.trunk_out)
    with open(sidecar, "w") as f:
        json.dump({"backbone": args.backbone, "val_top1": ckpt_cb.best,
                   "wh": energy["wh"], "co2_g": energy["co2_g"], "seconds": energy["seconds"]}, f)
    print(f"wrote energy sidecar -> {sidecar}")


if __name__ == "__main__":
    main()
