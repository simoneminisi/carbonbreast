"""Metric block shared by both bare ports (verbatim from s_main_keras.py / s_main.py)."""
import math

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

LABELS = ["normal/benign", "suspicious"]


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


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
    """Aggregate per-image probs to per-breast by max and mean over CC+MLO, report AUROC.
    (In png_index.csv `laterality` is the standardised constant 'R', so this groups per STUDY --
    kept as-is for comparability with the torch and Keras rows.)"""
    d = test_df.copy()
    d["prob"] = probs
    g = d.groupby(["patient_id", "laterality"]).agg(
        prob_max=("prob", "max"), prob_mean=("prob", "mean"), label=("label", "max")).reset_index()
    return {"n_breasts": len(g), "max": roc_auc_score(g.label, g.prob_max),
            "mean": roc_auc_score(g.label, g.prob_mean)}


def youden_threshold(val_labels, val_probs):
    """Youden-J threshold selected on VALIDATION probabilities (applied to test downstream)."""
    fpr, tpr, thr = roc_curve(val_labels, val_probs)
    return float(thr[np.argmax(tpr - fpr)])


def show_confusion(labels, preds, title, no_plot):
    import matplotlib  # noqa: PLC0415 (lazy: not a dependency of the deploy benchmark)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
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
