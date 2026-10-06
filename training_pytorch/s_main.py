"""VinDr-Mammo — binary "suspicious" classifier (Stage-2 full-image), parametrized for ablation.

    label 0 = normal/benign   -> BI-RADS 1-2
    label 1 = suspicious      -> BI-RADS 3-5

Ablation knobs (all CLI): --backbone (torchvision or any timm model), --optimizer,
--train-mode {full,head} (fine-tune all vs linear-probe the head), --quantize {none,fp16}.
Each run reports per-image + per-breast AUROC, confusion-matrix metrics (MCC/TPR/TNR/PPV/NPV) at
the 0.5 and val-selected thresholds, and energy/CO2/time (CodeCarbon) for Stage-2 training and for
inference; Stage-1 (patch) energy is folded in from train_patch.py's sidecar. Per-image predictions
go to predictions/pred_<tag>.csv and one summary row is appended to results.csv.

Run build_index.py once for png_index.csv. Then e.g.:
    python s_main.py --pretrained-trunk trunk_resnet18.pt --img-size 768 --batch 16
    python s_main.py --backbone convnext_tiny --optimizer sgd --lr 0.01 --train-mode head
    python s_main.py --pretrained-trunk trunk_resnet18.pt --quantize fp16
"""
import argparse
import csv
import datetime
import json
import math
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from pipeline import (
    PngDataset, assert_ram_fits, balanced_sampler, build_optimizer, create_backbone,
    decode_all, get_head, load_trunk, make_lr_factor, track_energy,
)

LABELS = ["normal/benign", "suspicious"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default="", help="prefix prepended to png_index.csv and png paths")
    p.add_argument("--png-index", default=None, help="path to png_index.csv (default: <folder>png_index.csv)")
    p.add_argument("--backbone", default="resnet18",
                   help="torchvision (resnet18/resnet50/efficientnet_b0) or any timm model name")
    p.add_argument("--pretrained-trunk", default=None,
                   help="trunk_<backbone>.pt from train_patch.py; loaded in place of the ImageNet trunk")
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--no-amp", dest="amp", action="store_false", help="train in fp32")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4, help="peak LR (cosine decays from here)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd", "rmsprop"],
                   help="SGD/RMSprop usually want a higher --lr than Adam")
    p.add_argument("--momentum", type=float, default=0.9, help="for sgd/rmsprop")
    p.add_argument("--train-mode", default="full", choices=["full", "head"],
                   help="full = fine-tune everything; head = freeze backbone, train only the classifier")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="enable gradient checkpointing (timm models) to fit heavy nets like ViT@768")
    p.add_argument("--quantize", default="none", choices=["none", "fp16"],
                   help="fp16 = also evaluate a half-precision copy and report the size/AUROC/latency tradeoff")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.2, help="dropout before the classifier (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=1, help="train N seeded rounds, keep best-on-val")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-ram", action="store_true", help="preload train+val decoded into RAM")
    p.add_argument("--cache-test", action="store_true", help="also cache the test set")
    p.add_argument("--cache-workers", type=int, default=None, help="decode threads (default: --workers)")
    p.add_argument("--force-cache", action="store_true", help="bypass the RAM-fits guard")
    p.add_argument("--no-energy", dest="energy", action="store_false", help="disable CodeCarbon tracking")
    p.add_argument("--ckpt", default=None, help="checkpoint path (default: best_<backbone>.pt)")
    p.add_argument("--pred-dir", default="predictions", help="dir for per-image prediction CSVs")
    p.add_argument("--results-csv", default="results.csv", help="append one summary row per run here")
    p.add_argument("--no-plot", action="store_true", help="skip confusion-matrix windows (headless)")
    return p.parse_args()


def split_trainval(df, val_frac=0.15, seed=42):
    train = df[df["split"] == "train"]
    tr_idx, va_idx = next(
        GroupShuffleSplit(1, test_size=val_frac, random_state=seed).split(train, groups=train["patient_id"]))
    return train.iloc[tr_idx], train.iloc[va_idx]


def seed_worker(_):
    np.random.seed(torch.initial_seed() % 2**32)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model(backbone, device, pretrained_trunk=None, dropout=0.0, grad_checkpoint=False):
    model = create_backbone(backbone, num_classes=2, dropout=dropout)
    if pretrained_trunk:
        load_trunk(model, pretrained_trunk, backbone)  # asserts backbone/trunk match
    if grad_checkpoint and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)  # timm: trade compute for memory (fits ViT@768)
    # channels_last is the memory layout tensor cores prefer; pairs with AMP for a free speedup.
    return model.to(device).to(memory_format=torch.channels_last)


def collect_probs(model, loader, device, amp=True, half=False):
    p, l = [], []
    model.eval()
    autocast = amp and device == "cuda" and not half  # half weights => no autocast
    with torch.no_grad(), torch.autocast(device, enabled=autocast):
        for x, y in loader:
            x = x.to(device).to(memory_format=torch.channels_last)
            if half:
                x = x.half()
            p += model(x).softmax(1)[:, 1].float().cpu().tolist()
            l += y.tolist()
    return np.array(p), np.array(l)


def confusion_metrics(labels, preds):
    """MCC / TPR(sens) / TNR(spec) / PPV(prec) / NPV / acc from the 2x2 confusion matrix.
    Positive class = suspicious (1)."""
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)

    def safe(num, den):
        return num / den if den else 0.0

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "acc": safe(tp + tn, tp + tn + fp + fn),
        "mcc": (tp * tn - fp * fn) / denom if denom else 0.0,
        "tpr": safe(tp, tp + fn),  # sensitivity / recall
        "tnr": safe(tn, tn + fp),  # specificity
        "ppv": safe(tp, tp + fp),  # precision
        "npv": safe(tn, tn + fn),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def print_cm_metrics(tag, m):
    print(f"  [{tag}] MCC={m['mcc']:.3f}  TPR/sens={m['tpr']:.3f}  TNR/spec={m['tnr']:.3f}  "
          f"PPV/prec={m['ppv']:.3f}  NPV={m['npv']:.3f}  acc={m['acc']:.3f}  "
          f"(TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']})")


def per_breast_auc(test_df, probs):
    """Aggregate per-image probs to per-breast by max and mean over CC+MLO, report AUROC."""
    d = test_df.copy()
    d["prob"] = probs
    g = d.groupby(["patient_id", "laterality"]).agg(
        prob_max=("prob", "max"), prob_mean=("prob", "mean"), label=("label", "max")).reset_index()
    return {"n_breasts": len(g), "max": roc_auc_score(g.label, g.prob_max),
            "mean": roc_auc_score(g.label, g.prob_mean)}


def show_confusion(labels, preds, title, no_plot):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(cmap="Blues", colorbar=False)
    disp.ax_.set_title(title)
    plt.tight_layout()
    if no_plot:
        out = title.split("—")[-1].strip().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        plt.savefig(out)
        print(f"saved confusion matrix -> {out}")
    else:
        plt.show()
    plt.close()


def run_tag(args):
    trunk = 1 if args.pretrained_trunk else 0
    return (f"{args.backbone}_{args.optimizer}_{args.train_mode}_q-{args.quantize}"
            f"_img{args.img_size}_bs{args.batch}_ep{args.epochs}_r{args.rounds}_s{args.seed}_trunk-{trunk}")


def state_dict_mb(model):
    return sum(t.numel() * t.element_size() for t in model.state_dict().values()) / 1e6


def train(args, model, loaders, device, ckpt):
    train_loader, val_loader, _ = loaders

    # --train-mode head: freeze the backbone (linear probe), train only the classifier head.
    head = get_head(model, args.backbone)
    if args.train_mode == "head":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in head.parameters():
            p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = list(model.parameters())

    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, params, args.lr, args.weight_decay, args.momentum)
    factor = make_lr_factor(args.warmup_epochs, args.epochs, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler(device, enabled=(args.amp and device == "cuda"))

    best_auc, n_batches = 0.0, len(train_loader)
    for epoch in range(args.epochs):
        if args.train_mode == "head":
            model.eval()   # keep frozen backbone (incl. BatchNorm) in eval...
            head.train()   # ...only the head trains
        else:
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
            if i % 200 == 0:  # heartbeat
                rate = (i + 1) * args.batch / (time.time() - t0)
                print(f"\r  epoch {epoch:02d}  batch {i:4d}/{n_batches}  "
                      f"loss={loss.item():.3f}  {rate:.0f} img/s", end="", flush=True)
        train_secs = time.time() - t0

        probs, labels = collect_probs(model, val_loader, device, args.amp)
        acc = ((probs > 0.5) == labels).mean()
        auc = roc_auc_score(labels, probs)
        print(f"\repoch {epoch:02d}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"val_acc={acc:.3f}  val_auroc={auc:.3f}  ({train_secs:.0f}s train)")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), ckpt)

    print("best val AUROC:", round(best_auc, 3))
    return best_auc


def evaluate(args, model, loaders, test_df, device, ckpt):
    """Load best ckpt, compute all metrics, write predictions, return (row_metrics, infer_energy)."""
    _, val_loader, test_loader = loaders
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    n_test = len(test_df)

    with track_energy("inference (test)", enabled=args.energy) as infer_e:
        test_probs, test_labels = collect_probs(model, test_loader, device, args.amp)
    lat32 = (infer_e["seconds"] or 0.0) / max(n_test, 1) * 1000
    fp32_mb = state_dict_mb(model)

    auc_img = roc_auc_score(test_labels, test_probs)
    pb = per_breast_auc(test_df, test_probs)
    print(f"\ntest images: {n_test}")
    print(f"per-image test AUROC: {auc_img:.3f}")
    print(f"per-breast test AUROC: max-agg {pb['max']:.3f}  mean-agg {pb['mean']:.3f}  ({pb['n_breasts']} breasts)")

    # threshold 0.5
    preds05 = (test_probs > 0.5).astype(int)
    m05 = confusion_metrics(test_labels, preds05)
    print(classification_report(test_labels, preds05, target_names=LABELS))
    print_cm_metrics("thr=0.50", m05)
    show_confusion(test_labels, preds05, "VinDr-Mammo — test confusion matrix (PNG pipeline)", args.no_plot)

    # val-selected Youden threshold, applied to TEST (no leakage)
    val_probs, val_labels = collect_probs(model, val_loader, device, args.amp)
    fpr, tpr, thr = roc_curve(val_labels, val_probs)
    best_thr = float(thr[np.argmax(tpr - fpr)])
    preds_opt = (test_probs > best_thr).astype(int)
    m_opt = confusion_metrics(test_labels, preds_opt)
    print(f"val-selected threshold: {best_thr:.3f}  (default 0.5)")
    print(classification_report(test_labels, preds_opt, target_names=LABELS))
    print_cm_metrics(f"thr={best_thr:.2f}", m_opt)
    show_confusion(test_labels, preds_opt, f"VinDr-Mammo — test confusion matrix (thr={best_thr:.2f})", args.no_plot)

    # optional FP16 quantization tradeoff
    q = {"size_mb_fp16": "", "test_auroc_img_fp16": "", "latency_ms_fp16": ""}
    probs16 = None
    if args.quantize == "fp16":
        qmodel = build_model(args.backbone, device, None, args.dropout)
        qmodel.load_state_dict(model.state_dict())
        qmodel = qmodel.half()
        fp16_ckpt = os.path.splitext(ckpt)[0] + "_fp16.pt"
        torch.save(qmodel.state_dict(), fp16_ckpt)
        t0 = time.time()
        probs16, labels16 = collect_probs(qmodel, test_loader, device, amp=False, half=True)
        lat16 = (time.time() - t0) / max(n_test, 1) * 1000
        auc16 = roc_auc_score(labels16, probs16)
        pb16 = per_breast_auc(test_df, probs16)
        q = {"size_mb_fp16": state_dict_mb(qmodel), "test_auroc_img_fp16": auc16, "latency_ms_fp16": lat16}
        print(f"\n[quantize fp16] size {fp32_mb:.1f} MB -> {q['size_mb_fp16']:.1f} MB | "
              f"per-image AUROC {auc_img:.3f} -> {auc16:.3f} | per-breast max {pb['max']:.3f} -> {pb16['max']:.3f} | "
              f"latency {lat32:.2f} -> {lat16:.2f} ms/img | ckpt -> {fp16_ckpt}")

    _write_predictions(args, test_df, test_probs, preds05, preds_opt, probs16)

    row = {
        "best_val_auroc": None, "thr_opt": best_thr,
        "test_auroc_img": auc_img, "test_auroc_breast_max": pb["max"], "test_auroc_breast_mean": pb["mean"],
        "size_mb_fp32": fp32_mb, "latency_ms_fp32": lat32, **q,
    }
    for pref, m in (("0.5", m05), ("opt", m_opt)):
        for k in ("acc", "mcc", "tpr", "tnr", "ppv", "npv"):
            row[f"{k}_{pref}"] = m[k]
    return row, infer_e


def _write_predictions(args, test_df, probs, preds05, preds_opt, probs16):
    os.makedirs(args.pred_dir, exist_ok=True)
    out = test_df[["patient_id", "laterality", "view", "label"]].copy()
    out["prob"] = probs
    out["pred_0.5"] = preds05
    out["pred_opt"] = preds_opt
    if probs16 is not None:
        out["prob_fp16"] = probs16
    path = os.path.join(args.pred_dir, f"pred_{run_tag(args)}.csv")
    out.to_csv(path, index=False)
    print(f"wrote predictions -> {path}")


RESULTS_FIELDS = [
    "timestamp", "tag", "backbone", "optimizer", "momentum", "train_mode", "quantize",
    "img_size", "batch", "epochs", "warmup_epochs", "lr", "weight_decay", "dropout", "seed",
    "rounds", "pretrained_trunk", "amp",
    "best_val_auroc", "thr_opt", "test_auroc_img", "test_auroc_breast_max", "test_auroc_breast_mean",
    "acc_0.5", "mcc_0.5", "tpr_0.5", "tnr_0.5", "ppv_0.5", "npv_0.5",
    "acc_opt", "mcc_opt", "tpr_opt", "tnr_opt", "ppv_opt", "npv_opt",
    "size_mb_fp32", "latency_ms_fp32", "size_mb_fp16", "test_auroc_img_fp16", "latency_ms_fp16",
    "stage1_wh", "stage1_co2_g", "stage1_s", "stage2_train_wh", "stage2_train_co2_g", "stage2_train_s",
    "infer_wh", "infer_co2_g", "infer_s",
]


def append_results(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended results row -> {path}")


def main():
    args = parse_args()
    if args.png_index is None:
        args.png_index = args.folder + "png_index.csv"
    if args.ckpt is None:
        # per-TAG (not per-backbone) so two runs of the same backbone at different
        # img-size/optimizer/... never clobber each other's checkpoint. The serving benchmark finds it by tag.
        args.ckpt = args.folder + f"best_{run_tag(args)}.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  backbone: {args.backbone}  img_size: {args.img_size}  batch: {args.batch}  "
          f"amp: {args.amp and device == 'cuda'}  trunk: {args.pretrained_trunk or 'imagenet'}")
    print(f"optimizer: {args.optimizer}(mom={args.momentum})  train_mode: {args.train_mode}  "
          f"quantize: {args.quantize}  grad_ckpt: {args.grad_checkpoint}  wd: {args.weight_decay}  "
          f"dropout: {args.dropout}  seed: {args.seed}  rounds: {args.rounds}")
    assert os.path.exists(args.png_index), f"{args.png_index} not found. Run build_index.py first."

    df = pd.read_csv(args.png_index)
    train_df, val_df = split_trainval(df)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(d)} images | normal/benign={int((d.label == 0).sum())} "
              f"suspicious={int((d.label == 1).sum())}")

    kw = dict(num_workers=args.workers, pin_memory=True,
              persistent_workers=(args.workers > 0),
              prefetch_factor=(4 if args.workers > 0 else None))

    tr_cache = va_cache = te_cache = None
    if args.cache_ram:
        cw = args.cache_workers or args.workers
        cache_dfs = [train_df, val_df] + ([test_df] if args.cache_test else [])
        cache_bytes = sum(len(d) for d in cache_dfs) * args.img_size * args.img_size
        loader_bytes = args.workers * 4 * args.batch * args.img_size * args.img_size * 3 * 4
        assert_ram_fits(cache_bytes + loader_bytes, force=args.force_cache)
        tr_cache = decode_all(train_df, args.img_size, cw, args.folder, "train cache")
        va_cache = decode_all(val_df, args.img_size, cw, args.folder, "val cache")
        if args.cache_test:
            te_cache = decode_all(test_df, args.img_size, cw, args.folder, "test cache")

    def ds(d, train=False, cache=None):
        return PngDataset(d, folder=args.folder, img_size=args.img_size, train=train, cache=cache)

    loaders = (
        DataLoader(ds(train_df, train=True, cache=tr_cache), args.batch,
                   sampler=balanced_sampler(train_df.label.tolist()),
                   drop_last=True, worker_init_fn=seed_worker, **kw),
        DataLoader(ds(val_df, cache=va_cache), args.batch, shuffle=False, **kw),
        DataLoader(ds(test_df, cache=te_cache), args.batch, shuffle=False, **kw),
    )

    # Best-of-N rounds: train `--rounds` seeded models, keep best val AUROC. Energy wraps all rounds.
    stem, ext = os.path.splitext(args.ckpt)
    best_round_auc, best_ckpt = -1.0, args.ckpt
    with track_energy(f"stage2 train ({args.backbone}, {args.rounds} round(s))", enabled=args.energy) as train_e:
        for r in range(args.rounds):
            set_seed(args.seed + r)
            round_ckpt = args.ckpt if args.rounds == 1 else f"{stem}_r{r}{ext}"
            if args.rounds > 1:
                print(f"\n=== round {r + 1}/{args.rounds}  seed={args.seed + r}  -> {round_ckpt} ===")
            model = build_model(args.backbone, device, args.pretrained_trunk, args.dropout, args.grad_checkpoint)
            auc = train(args, model, loaders, device, round_ckpt)
            if auc > best_round_auc:
                best_round_auc, best_ckpt = auc, round_ckpt
    if args.rounds > 1:
        print(f"\nbest round: val AUROC {best_round_auc:.3f} -> {best_ckpt}")
        shutil.copyfile(best_ckpt, args.ckpt)

    model = build_model(args.backbone, device, args.pretrained_trunk, args.dropout)
    row, infer_e = evaluate(args, model, loaders, test_df, device, best_ckpt)

    # assemble the results row (params + metrics + energy for all 3 regions)
    row.update({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "tag": run_tag(args),
        "backbone": args.backbone, "optimizer": args.optimizer, "momentum": args.momentum,
        "train_mode": args.train_mode, "quantize": args.quantize, "img_size": args.img_size,
        "batch": args.batch, "epochs": args.epochs, "warmup_epochs": args.warmup_epochs, "lr": args.lr,
        "weight_decay": args.weight_decay, "dropout": args.dropout, "seed": args.seed, "rounds": args.rounds,
        "pretrained_trunk": args.pretrained_trunk or "", "amp": args.amp and device == "cuda",
        "best_val_auroc": best_round_auc,
        "stage2_train_wh": train_e["wh"], "stage2_train_co2_g": train_e["co2_g"], "stage2_train_s": train_e["seconds"],
        "infer_wh": infer_e["wh"], "infer_co2_g": infer_e["co2_g"], "infer_s": infer_e["seconds"],
    })
    # Stage-1 energy from the trunk sidecar (if a trunk was used)
    if args.pretrained_trunk:
        sidecar = os.path.splitext(args.pretrained_trunk)[0] + ".energy.json"
        if os.path.exists(sidecar):
            s1 = json.load(open(sidecar))
            row.update({"stage1_wh": s1.get("wh"), "stage1_co2_g": s1.get("co2_g"), "stage1_s": s1.get("seconds")})
    append_results(args.results_csv, row)


if __name__ == "__main__":
    main()
