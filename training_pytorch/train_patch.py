"""Stage-1: train a patch classifier, then export its conv trunk for transfer.

4-class classification (background / birads3 / birads4 / birads5) on 224px patches, so the
backbone learns to localize mammographic lesion texture instead of relying on raw ImageNet
features. Only the conv trunk is reused downstream (the head is discarded).

The backbone is selectable (`--backbone`, torchvision set or any timm model) and MUST match the
Stage-2 backbone for the trunk to transfer -- both stages build it through pipeline.create_backbone.
Saves `trunk_{backbone}.pt` (= state_dict minus the classifier head) + a `*.energy.json` sidecar,
which s_main.py loads via `--pretrained-trunk`.

Run:
    python make_patches.py                        # once, builds patch_index.csv
    python train_patch.py                         # resnet18 (default)
    python train_patch.py --backbone convnext_tiny --optimizer sgd --lr 0.01
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from pipeline import (
    PngDataset, assert_ram_fits, balanced_sampler, build_optimizer, create_backbone,
    decode_all, make_lr_factor, track_energy, trunk_state_dict,
)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch-index", default="patch_index.csv")
    p.add_argument("--backbone", default="resnet18",
                   help="torchvision (resnet18/resnet50/efficientnet_b0) or any timm model name")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4, help="peak LR (cosine decays from here)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd", "rmsprop"])
    p.add_argument("--momentum", type=float, default=0.9, help="for sgd/rmsprop")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-energy", dest="energy", action="store_false", help="disable CodeCarbon tracking")
    # Patches are 224px (~50KB decoded) so the full cache is small (~2 GB) -- cheap decode skip.
    p.add_argument("--cache-ram", action="store_true", help="preload patches decoded into RAM")
    p.add_argument("--force-cache", action="store_true", help="bypass the RAM-fits guard")
    p.add_argument("--trunk-out", default=None, help="default: trunk_<backbone>.pt")
    return p.parse_args()


def collect_preds(model, loader, device, amp):
    preds, labels = [], []
    model.eval()
    with torch.no_grad(), torch.autocast(device, enabled=(amp and device == "cuda")):
        for x, y in loader:
            x = x.to(device).to(memory_format=torch.channels_last)
            preds += model(x).argmax(1).cpu().tolist()
            labels += y.tolist()
    return np.array(preds), np.array(labels)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.trunk_out is None:
        args.trunk_out = f"trunk_{args.backbone}.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  backbone: {args.backbone}  optimizer: {args.optimizer}  "
          f"amp: {args.amp and device == 'cuda'}  seed: {args.seed}  wd: {args.weight_decay}")

    df = pd.read_csv(args.patch_index)
    train_df = df[df.split == "patch_train"]
    val_df = df[df.split == "patch_val"]
    print(f"patch_train: {len(train_df)}  patch_val: {len(val_df)}  "
          f"classes: {sorted(df.label.unique().tolist())}")

    tr_cache = va_cache = None
    if args.cache_ram:
        total = (len(train_df) + len(val_df)) * args.img_size * args.img_size
        assert_ram_fits(total, force=args.force_cache)
        tr_cache = decode_all(train_df, args.img_size, args.workers, desc="patch train cache")
        va_cache = decode_all(val_df, args.img_size, args.workers, desc="patch val cache")

    kw = dict(num_workers=args.workers, pin_memory=True,
              persistent_workers=(args.workers > 0),
              prefetch_factor=(4 if args.workers > 0 else None))
    train_loader = DataLoader(PngDataset(train_df, img_size=args.img_size, train=True, cache=tr_cache),
                              args.batch, sampler=balanced_sampler(train_df.label.tolist()),
                              drop_last=True, **kw)
    val_loader = DataLoader(PngDataset(val_df, img_size=args.img_size, cache=va_cache),
                            args.batch, shuffle=False, **kw)

    model = create_backbone(args.backbone, num_classes=4).to(device).to(memory_format=torch.channels_last)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.momentum)
    factor = make_lr_factor(args.warmup_epochs, args.epochs, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler(device, enabled=(args.amp and device == "cuda"))

    best_acc, n_batches = 0.0, len(train_loader)
    with track_energy(f"stage1 train ({args.backbone})", enabled=args.energy) as energy:
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device).to(memory_format=torch.channels_last)
                y = y.to(device)
                optimizer.zero_grad()
                with torch.autocast(device, enabled=(args.amp and device == "cuda")):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= prev_scale:  # skip LR step when GradScaler skipped optimizer
                    scheduler.step()
                if i % 100 == 0:
                    print(f"\r  epoch {epoch:02d}  batch {i:4d}/{n_batches}  loss={loss.item():.3f}",
                          end="", flush=True)

            preds, labels = collect_preds(model, val_loader, device, args.amp)
            acc = accuracy_score(labels, preds) if len(labels) else 0.0
            print(f"\repoch {epoch:02d}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"val_top1={acc:.3f}  ({time.time() - t0:.0f}s)")
            if acc > best_acc:
                best_acc = acc
                torch.save(trunk_state_dict(model, args.backbone), args.trunk_out)

    print(f"best val top-1: {best_acc:.3f}  (chance=0.25)")
    print(f"wrote conv trunk -> {args.trunk_out}")

    # sidecar so s_main.py can fold Stage-1 energy into its results row
    sidecar = os.path.splitext(args.trunk_out)[0] + ".energy.json"
    with open(sidecar, "w") as f:
        json.dump({"backbone": args.backbone, "val_top1": best_acc,
                   "wh": energy["wh"], "co2_g": energy["co2_g"], "seconds": energy["seconds"]}, f)
    print(f"wrote energy sidecar -> {sidecar}")


if __name__ == "__main__":
    main()
