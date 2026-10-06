"""Results-row contract shared with the torch and Keras runs: the 49-column header, the run
tag, and the CSV appender. Rows from every framework concatenate into one Pareto CSV."""
import csv
import datetime
import os

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
assert len(RESULTS_FIELDS) == 49


def run_tag(args):
    trunk = 1 if args.pretrained_trunk else 0
    return (f"{args.backbone}_{args.optimizer}_{args.train_mode}_q-{args.quantize}"
            f"_img{args.img_size}_bs{args.batch}_ep{args.epochs}_r{args.rounds}_s{args.seed}_trunk-{trunk}")


def append_results(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended results row -> {path}")


def build_row(args, amp_on, best_val_auroc, eval_row, train_e, infer_e, stage1=None):
    """Assemble the results row: params + metrics + energy for all 3 regions (same field
    semantics as s_main.py / s_main_keras.py; stage1_* stay absent -> empty when no trunk)."""
    row = dict(eval_row)
    row.update({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "tag": run_tag(args),
        "backbone": args.backbone, "optimizer": args.optimizer, "momentum": args.momentum,
        "train_mode": args.train_mode, "quantize": args.quantize, "img_size": args.img_size,
        "batch": args.batch, "epochs": args.epochs, "warmup_epochs": args.warmup_epochs, "lr": args.lr,
        "weight_decay": args.weight_decay, "dropout": args.dropout, "seed": args.seed, "rounds": args.rounds,
        "pretrained_trunk": args.pretrained_trunk or "", "amp": amp_on,
        "best_val_auroc": best_val_auroc,
        "stage2_train_wh": train_e["wh"], "stage2_train_co2_g": train_e["co2_g"], "stage2_train_s": train_e["seconds"],
        "infer_wh": infer_e["wh"], "infer_co2_g": infer_e["co2_g"], "infer_s": infer_e["seconds"],
    })
    if stage1:
        row.update({"stage1_wh": stage1.get("wh"), "stage1_co2_g": stage1.get("co2_g"),
                    "stage1_s": stage1.get("seconds")})
    return row
