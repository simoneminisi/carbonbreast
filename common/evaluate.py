"""Stage-2 evaluation shared by both ports -- a line-by-line port of s_main_keras.evaluate.

The framework supplies two callables:
    predict32(loader) -> np.ndarray probs of class 1, row-aligned to the loader's df
    predict16(loader) -> same, from the fp16 copy of the model (or None if unsupported)
and the byte sizes of the fp32 / fp16 parameter sets.
"""
import os
import time

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from metrics import LABELS, confusion_metrics, per_breast_auc, print_cm_metrics, show_confusion, youden_threshold
from energy import track_energy
from results import run_tag


def evaluate_stage2(args, predict32, predict16, size32_mb, size16_mb, val_loader, val_labels,
                    test_loader, test_df, fp16_ckpt=None):
    """Compute all metrics, write predictions, return (row_metrics, infer_energy)."""
    test_labels = test_df["label"].astype(int).to_numpy()
    n_test = len(test_df)

    with track_energy("inference (test)", enabled=args.energy) as infer_e:
        test_probs = predict32(test_loader)
    lat32 = (infer_e["seconds"] or 0.0) / max(n_test, 1) * 1000
    fp32_mb = size32_mb

    auc_img = roc_auc_score(test_labels, test_probs)
    pb = per_breast_auc(test_df, test_probs)
    print(f"\ntest images: {n_test}")
    print(f"per-image test AUROC: {auc_img:.3f}")
    print(f"per-breast test AUROC: max-agg {pb['max']:.3f}  mean-agg {pb['mean']:.3f}  ({pb['n_breasts']} breasts)")

    # threshold 0.5
    preds05 = (test_probs > 0.5).astype(int)
    m05 = confusion_metrics(test_labels, preds05)
    print(classification_report(test_labels, preds05, target_names=LABELS, zero_division=0))
    print_cm_metrics("thr=0.50", m05)
    show_confusion(test_labels, preds05, "VinDr-Mammo — test confusion matrix (PNG pipeline)", args.no_plot)

    # val-selected Youden threshold, applied to TEST (no leakage)
    val_probs = predict32(val_loader)
    best_thr = youden_threshold(val_labels, val_probs)
    preds_opt = (test_probs > best_thr).astype(int)
    m_opt = confusion_metrics(test_labels, preds_opt)
    print(f"val-selected threshold: {best_thr:.3f}  (default 0.5)")
    print(classification_report(test_labels, preds_opt, target_names=LABELS, zero_division=0))
    print_cm_metrics(f"thr={best_thr:.2f}", m_opt)
    show_confusion(test_labels, preds_opt, f"VinDr-Mammo — test confusion matrix (thr={best_thr:.2f})", args.no_plot)

    # optional FP16 tradeoff (fp16 params + fp16 compute -- the counterpart of torch model.half())
    q = {"size_mb_fp16": "", "test_auroc_img_fp16": "", "latency_ms_fp16": ""}
    probs16 = None
    if args.quantize == "fp16" and predict16 is not None:
        try:
            t0 = time.time()
            probs16 = predict16(test_loader)
            lat16 = (time.time() - t0) / max(n_test, 1) * 1000
            auc16 = roc_auc_score(test_labels, probs16)
            pb16 = per_breast_auc(test_df, probs16)
            q = {"size_mb_fp16": size16_mb, "test_auroc_img_fp16": auc16, "latency_ms_fp16": lat16}
            print(f"\n[quantize fp16] size {fp32_mb:.1f} MB -> {size16_mb:.1f} MB | "
                  f"per-image AUROC {auc_img:.3f} -> {auc16:.3f} | per-breast max {pb['max']:.3f} -> {pb16['max']:.3f} | "
                  f"latency {lat32:.2f} -> {lat16:.2f} ms/img | ckpt -> {fp16_ckpt}")
        except Exception as e:  # noqa: BLE001 (fp16 can overflow on some nets; don't kill the run)
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
