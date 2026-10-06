"""VinDr-Mammo — binary "suspicious" classifier (Stage-2 full-image), Keras 3 / TF port.

    label 0 = normal/benign   -> BI-RADS 1-2
    label 1 = suspicious      -> BI-RADS 3-5

Faithful port of ../training_pytorch/s_main.py restricted to the pareto-frontier
configs (backbones: resnet18 / mobilenetv3_small_100 / convnext_small). Same CLI, same
run-tag format, same 48-column results row -> plot_pareto.py works unchanged and rows can
be concatenated with the PyTorch results.csv for a combined frontier.

Run e.g.:
    python s_main_keras.py --folder ../training_pytorch/ --img-size 768 --batch 16
    python s_main_keras.py --pretrained-trunk trunk_keras_resnet18.weights.h5 \\
        --folder ../training_pytorch/ --img-size 768 --batch 16 --quantize fp16
"""
import argparse
import csv
import datetime
import json
import math
import os
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit

from pipeline_keras import (
    WarmupCosine, assert_ram_fits, build_model, build_optimizer, decode_all,
    energy_sidecar_path, make_eval_ds, make_train_ds, setup_gpu, softmax_np,
    track_energy, weights_mb,
)
import keras

LABELS = ["normal/benign", "suspicious"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default="", help="prefix prepended to png_index.csv and png paths")
    p.add_argument("--png-index", default=None, help="path to png_index.csv (default: <folder>png_index.csv)")
    p.add_argument("--backbone", default="resnet18",
                   help="resnet18 / mobilenetv3_small_100 / convnext_small")
    p.add_argument("--pretrained-trunk", default=None,
                   help="trunk_keras_<backbone>.weights.h5 from train_patch_keras.py")
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--no-amp", dest="amp", action="store_false", help="train in fp32")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4, help="peak LR (cosine decays from here)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd", "rmsprop"])
    p.add_argument("--momentum", type=float, default=0.9, help="for sgd/rmsprop")
    p.add_argument("--train-mode", default="full", choices=["full", "head"],
                   help="full = fine-tune everything; head = freeze backbone, train only the classifier")
    p.add_argument("--quantize", default="none", choices=["none", "fp16"],
                   help="fp16 = also evaluate a half-precision copy and report size/AUROC/latency")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.2, help="dropout before the classifier (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=1, help="train N seeded rounds, keep best-on-val")
    p.add_argument("--workers", type=int, default=8, help="decode threads for --cache-ram")
    p.add_argument("--cache-ram", action="store_true", help="preload train+val decoded into RAM")
    p.add_argument("--cache-test", action="store_true", help="also cache the test set")
    p.add_argument("--cache-workers", type=int, default=None, help="decode threads (default: --workers)")
    p.add_argument("--force-cache", action="store_true", help="bypass the RAM-fits guard")
    p.add_argument("--no-energy", dest="energy", action="store_false", help="disable CodeCarbon tracking")
    p.add_argument("--ckpt", default=None, help="checkpoint path (default: best_<tag>.weights.h5)")
    p.add_argument("--pred-dir", default="predictions", help="dir for per-image prediction CSVs")
    p.add_argument("--results-csv", default="results_keras.csv", help="append one summary row per run here")
    p.add_argument("--no-plot", action="store_true", help="save confusion-matrix PNGs (headless)")
    p.add_argument("--limit", type=int, default=None,
                   help="DEBUG: stratified subset of N rows per split (smoke tests only)")
    return p.parse_args()


def split_trainval(df, val_frac=0.15, seed=42):
    train = df[df["split"] == "train"]
    tr_idx, va_idx = next(
        GroupShuffleSplit(1, test_size=val_frac, random_state=seed).split(train, groups=train["patient_id"]))
    return train.iloc[tr_idx], train.iloc[va_idx]


def limit_df(d, n):
    """Stratified head(n) so tiny smoke-test splits still contain both classes."""
    per = max(1, n // max(d["label"].nunique(), 1))
    return d.groupby("label", group_keys=False).head(per).head(n).reset_index(drop=True)


def set_seed(s):
    keras.utils.set_random_seed(s)  # python + numpy + backend


def collect_probs(model, ds):
    logits = model.predict(ds, verbose=0)
    return softmax_np(np.asarray(logits, dtype=np.float32))[:, 1]


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


class Heartbeat(keras.callbacks.Callback):
    """Batch heartbeat with img/s, same cadence as the torch loop."""

    def __init__(self, batch_size, n_batches, every=200):
        super().__init__()
        self.batch_size, self.n_batches, self.every = batch_size, n_batches, every
        self.t0, self.epoch = time.time(), 0

    def on_epoch_begin(self, epoch, logs=None):
        self.t0, self.epoch = time.time(), epoch

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.every == 0:
            rate = (batch + 1) * self.batch_size / max(time.time() - self.t0, 1e-9)
            loss = (logs or {}).get("loss", float("nan"))
            print(f"\r  epoch {self.epoch:02d}  batch {batch:4d}/{self.n_batches}  "
                  f"loss={loss:.3f}  {rate:.0f} img/s", end="", flush=True)


class AurocCheckpoint(keras.callbacks.Callback):
    """Per-epoch val AUROC; save_weights on improvement (same selection as the torch loop)."""

    def __init__(self, val_ds, val_labels, ckpt, schedule=None):
        super().__init__()
        self.val_ds, self.val_labels, self.ckpt, self.schedule = val_ds, val_labels, ckpt, schedule
        self.best, self.t0 = 0.0, time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        train_secs = time.time() - self.t0
        probs = collect_probs(self.model, self.val_ds)
        acc = ((probs > 0.5) == self.val_labels).mean()
        auc = roc_auc_score(self.val_labels, probs)
        lr = ""
        if self.schedule is not None:
            try:
                lr_v = keras.ops.convert_to_numpy(self.schedule(self.model.optimizer.iterations))
                lr = f"lr={float(lr_v):.2e}  "
            except Exception:  # noqa: BLE001
                pass
        print(f"\repoch {epoch:02d}  {lr}val_acc={acc:.3f}  val_auroc={auc:.3f}  "
              f"({train_secs:.0f}s train)")
        if auc > self.best:
            self.best = auc
            self.model.save_weights(self.ckpt)


def train(args, model, backbone, train_ds, steps_per_epoch, val_ds, val_labels, ckpt):
    # --train-mode head: freeze the backbone (linear probe). trainable=False also puts
    # BatchNorm layers in inference mode -- the keras counterpart of model.eval()+head.train().
    if args.train_mode == "head":
        backbone.trainable = False

    schedule = WarmupCosine(args.lr, args.warmup_epochs, args.epochs, steps_per_epoch)
    optimizer = build_optimizer(args.optimizer, schedule, args.weight_decay, args.momentum)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    ckpt_cb = AurocCheckpoint(val_ds, val_labels, ckpt, schedule)
    model.fit(train_ds, epochs=args.epochs, steps_per_epoch=steps_per_epoch, verbose=0,
              callbacks=[Heartbeat(args.batch, steps_per_epoch), ckpt_cb])

    print("best val AUROC:", round(ckpt_cb.best, 3))
    return ckpt_cb.best


def evaluate(args, model, val_ds, val_labels, test_ds, test_df, ckpt):
    """Load best ckpt, compute all metrics, write predictions, return (row_metrics, infer_energy)."""
    model.load_weights(ckpt)
    test_labels = test_df["label"].astype(int).to_numpy()
    n_test = len(test_df)

    with track_energy("inference (test)", enabled=args.energy) as infer_e:
        test_probs = collect_probs(model, test_ds)
    lat32 = (infer_e["seconds"] or 0.0) / max(n_test, 1) * 1000
    fp32_mb = weights_mb(model)  # mixed_float16 keeps fp32 variables -> true fp32 size

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
    val_probs = collect_probs(model, val_ds)
    fpr, tpr, thr = roc_curve(val_labels, val_probs)
    best_thr = float(thr[np.argmax(tpr - fpr)])
    preds_opt = (test_probs > best_thr).astype(int)
    m_opt = confusion_metrics(test_labels, preds_opt)
    print(f"val-selected threshold: {best_thr:.3f}  (default 0.5)")
    print(classification_report(test_labels, preds_opt, target_names=LABELS))
    print_cm_metrics(f"thr={best_thr:.2f}", m_opt)
    show_confusion(test_labels, preds_opt, f"VinDr-Mammo — test confusion matrix (thr={best_thr:.2f})", args.no_plot)

    # optional FP16 tradeoff: rebuild under a float16 policy (fp16 variables + compute --
    # the keras counterpart of torch model.half()) and copy the trained weights in.
    q = {"size_mb_fp16": "", "test_auroc_img_fp16": "", "latency_ms_fp16": ""}
    probs16 = None
    if args.quantize == "fp16":
        prev_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("float16")
            qmodel, _ = build_model(args.backbone, 2, args.img_size, args.dropout)
            keras.mixed_precision.set_global_policy(prev_policy)
            qmodel.set_weights(model.get_weights())
            fp16_ckpt = ckpt.replace(".weights.h5", "_fp16.weights.h5")
            qmodel.save_weights(fp16_ckpt)
            t0 = time.time()
            probs16 = collect_probs(qmodel, test_ds)
            lat16 = (time.time() - t0) / max(n_test, 1) * 1000
            auc16 = roc_auc_score(test_labels, probs16)
            pb16 = per_breast_auc(test_df, probs16)
            q = {"size_mb_fp16": weights_mb(qmodel), "test_auroc_img_fp16": auc16, "latency_ms_fp16": lat16}
            print(f"\n[quantize fp16] size {fp32_mb:.1f} MB -> {q['size_mb_fp16']:.1f} MB | "
                  f"per-image AUROC {auc_img:.3f} -> {auc16:.3f} | per-breast max {pb['max']:.3f} -> {pb16['max']:.3f} | "
                  f"latency {lat32:.2f} -> {lat16:.2f} ms/img | ckpt -> {fp16_ckpt}")
        except Exception as e:  # noqa: BLE001 (fp16 can overflow on some nets; don't kill the run)
            keras.mixed_precision.set_global_policy(prev_policy)
            probs16 = None
            print(f"[quantize fp16] FAILED ({type(e).__name__}: {e}); leaving fp16 cells empty")

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
        # per-TAG so two runs of the same backbone never clobber each other's checkpoint.
        # Written to the CURRENT dir (not --folder) so keras artifacts stay out of the data dir.
        args.ckpt = f"best_{run_tag(args)}.weights.h5"

    gpu = setup_gpu()
    amp_on = args.amp and gpu
    if amp_on:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"device: {'gpu' if gpu else 'cpu'}  backbone: {args.backbone}  img_size: {args.img_size}  "
          f"batch: {args.batch}  amp: {amp_on} (policy={keras.mixed_precision.global_policy().name})  "
          f"trunk: {args.pretrained_trunk or 'imagenet'}")
    print(f"optimizer: {args.optimizer}(mom={args.momentum})  train_mode: {args.train_mode}  "
          f"quantize: {args.quantize}  wd: {args.weight_decay}  "
          f"dropout: {args.dropout}  seed: {args.seed}  rounds: {args.rounds}")
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
        buffer_bytes = args.workers * 4 * args.batch * args.img_size * args.img_size * 3 * 4
        assert_ram_fits(cache_bytes + buffer_bytes, force=args.force_cache)
        tr_cache = decode_all(train_df, args.img_size, cw, args.folder, "train cache")
        va_cache = decode_all(val_df, args.img_size, cw, args.folder, "val cache")
        if args.cache_test:
            te_cache = decode_all(test_df, args.img_size, cw, args.folder, "test cache")

    train_ds = make_train_ds(train_df, args.img_size, args.batch, args.seed, args.folder, tr_cache)
    val_ds = make_eval_ds(val_df, args.img_size, args.batch, args.folder, va_cache)
    test_ds = make_eval_ds(test_df, args.img_size, args.batch, args.folder, te_cache)
    val_labels = val_df["label"].astype(int).to_numpy()
    steps_per_epoch = max(len(train_df) // args.batch, 1)

    # Best-of-N rounds: train `--rounds` seeded models, keep best val AUROC. Energy wraps all rounds.
    stem = args.ckpt.replace(".weights.h5", "")
    best_round_auc, best_ckpt = -1.0, args.ckpt
    with track_energy(f"stage2 train ({args.backbone}, {args.rounds} round(s))", enabled=args.energy) as train_e:
        for r in range(args.rounds):
            set_seed(args.seed + r)
            round_ckpt = args.ckpt if args.rounds == 1 else f"{stem}_r{r}.weights.h5"
            if args.rounds > 1:
                print(f"\n=== round {r + 1}/{args.rounds}  seed={args.seed + r}  -> {round_ckpt} ===")
            model, backbone = build_model(args.backbone, 2, args.img_size, args.dropout,
                                          args.pretrained_trunk)
            auc = train(args, model, backbone, train_ds, steps_per_epoch, val_ds, val_labels, round_ckpt)
            if auc > best_round_auc:
                best_round_auc, best_ckpt = auc, round_ckpt
    if args.rounds > 1:
        print(f"\nbest round: val AUROC {best_round_auc:.3f} -> {best_ckpt}")
        shutil.copyfile(best_ckpt, args.ckpt)

    model, _ = build_model(args.backbone, 2, args.img_size, args.dropout)
    row, infer_e = evaluate(args, model, val_ds, val_labels, test_ds, test_df, best_ckpt)

    # assemble the results row (params + metrics + energy for all 3 regions)
    row.update({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "tag": run_tag(args),
        "backbone": args.backbone, "optimizer": args.optimizer, "momentum": args.momentum,
        "train_mode": args.train_mode, "quantize": args.quantize, "img_size": args.img_size,
        "batch": args.batch, "epochs": args.epochs, "warmup_epochs": args.warmup_epochs, "lr": args.lr,
        "weight_decay": args.weight_decay, "dropout": args.dropout, "seed": args.seed, "rounds": args.rounds,
        "pretrained_trunk": args.pretrained_trunk or "", "amp": amp_on,
        "best_val_auroc": best_round_auc,
        "stage2_train_wh": train_e["wh"], "stage2_train_co2_g": train_e["co2_g"], "stage2_train_s": train_e["seconds"],
        "infer_wh": infer_e["wh"], "infer_co2_g": infer_e["co2_g"], "infer_s": infer_e["seconds"],
    })
    # Stage-1 energy from the trunk sidecar (if a trunk was used)
    if args.pretrained_trunk:
        sidecar = energy_sidecar_path(args.pretrained_trunk)
        if os.path.exists(sidecar):
            s1 = json.load(open(sidecar))
            row.update({"stage1_wh": s1.get("wh"), "stage1_co2_g": s1.get("co2_g"), "stage1_s": s1.get("seconds")})
    append_results(args.results_csv, row)


if __name__ == "__main__":
    main()
