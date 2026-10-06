"""Build an accuracy-vs-carbon Pareto curve from a results CSV.

x = carbon, in gCO2 (falls back to energy Wh, then wall-seconds, if CodeCarbon produced no CO2
    offline). --carbon selects which phase: train (Stage-1 + Stage-2, the default), infer
    (inference only — use with results_quant.csv), or total (train + infer).
y = a chosen accuracy metric column (default: per-breast max-agg AUROC).
Marks the Pareto frontier (max metric for min carbon) and saves an image.

    python plot_pareto.py [results.csv] [metric_column] [--logx] [--carbon train|infer|total]
    python plot_pareto.py results.csv mcc_opt --logx                  # training carbon
    python plot_pareto.py results_quant.csv mcc_opt --carbon infer    # inference carbon (quant)
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

FLAGS = {"--logx", "--logy", "--rowid"}
logx = "--logx" in sys.argv  # log-scale the carbon (x) axis
logy = "--logy" in sys.argv  # log-scale the metric (y) axis
rowid = "--rowid" in sys.argv  # prefix each point label with its CSV row index (for manual lookup)
# --carbon {train,infer,total}: which phase's carbon goes on the x-axis
carbon_phase = "train"
if "--carbon" in sys.argv:
    carbon_phase = sys.argv[sys.argv.index("--carbon") + 1]
# --unit {auto,co2,wh,s}: force the carbon unit. Default 'auto' prefers gCO2>Wh>s, but CodeCarbon's
# gCO2 depends on the (sometimes wildly inconsistent) logged grid intensity, so --unit wh plots the
# energy the model actually determines — the honest cross-run axis when gCO2 readings vary.
carbon_unit_opt = "auto"
if "--unit" in sys.argv:
    carbon_unit_opt = sys.argv[sys.argv.index("--unit") + 1]
# --co2-factor <g/kWh>: derive carbon from ENERGY with one fixed grid intensity (gCO2 = Wh * f/1000),
# so the carbon axis is consistent across runs (CodeCarbon's per-run gCO2 is not). Forces the Wh sum.
co2_factor = co2_factor_raw = None
if "--co2-factor" in sys.argv:
    co2_factor_raw = sys.argv[sys.argv.index("--co2-factor") + 1]
    co2_factor = float(co2_factor_raw)
    carbon_unit_opt = "wh"
_exclude = set(FLAGS) | {"--carbon", carbon_phase, "--unit", carbon_unit_opt}
if co2_factor_raw is not None:
    _exclude |= {"--co2-factor", co2_factor_raw}
argv = [a for a in sys.argv[1:] if a not in _exclude]
path = argv[0] if len(argv) > 0 else "results.csv"
metric = argv[1] if len(argv) > 1 else "test_auroc_breast_max"
d = pd.read_csv(path)
d["_row"] = range(len(d))  # 0-based CSV data-row index (line _row+2 in the file; header is line 1)

# nice y-axis labels + output name per metric
YLABELS = {
    "test_auroc_breast_max": "per-breast max-agg test AUROC",
    "test_auroc_breast_mean": "per-breast mean-agg test AUROC",
    "test_auroc_img": "per-image test AUROC",
    "mcc_opt": "MCC (val-selected threshold)",
    "mcc_0.5": "MCC (threshold 0.5)",
}
ylabel = YLABELS.get(metric, metric)
_stem = "pareto" if metric == "test_auroc_breast_max" else f"pareto_{metric}"
outfile = f"{_stem}.png" if carbon_phase == "train" else f"{_stem}_{carbon_phase}.png"

# which columns sum into the carbon axis, per phase (in gCO2 / Wh / s preference order)
PHASE_COLS = {
    "train": {"co2": ["stage1_co2_g", "stage2_train_co2_g"], "wh": ["stage1_wh", "stage2_train_wh"],
              "s": ["stage1_s", "stage2_train_s"]},
    "infer": {"co2": ["infer_co2_g"], "wh": ["infer_wh"], "s": ["infer_s"]},
    "total": {"co2": ["stage1_co2_g", "stage2_train_co2_g", "infer_co2_g"],
              "wh": ["stage1_wh", "stage2_train_wh", "infer_wh"],
              "s": ["stage1_s", "stage2_train_s", "infer_s"]},
}
if carbon_phase not in PHASE_COLS:
    sys.exit(f"--carbon must be one of {list(PHASE_COLS)}")
PHASE_LABEL = {"train": "training carbon (Stage-1 + Stage-2)", "infer": "inference carbon",
               "total": "total carbon (train + inference)"}


def col(name):
    return pd.to_numeric(d[name], errors="coerce") if name in d else pd.Series(0.0, index=d.index)


def _has_trunk(v):  # empty cells read back as NaN -> str(NaN)=='nan' (truthy), so check explicitly
    return 0 if (pd.isna(v) or str(v).strip() in ("", "nan")) else 1


# carbon axis: prefer gCO2, else Wh, else seconds — summed over the selected phase's columns.
# --unit restricts the search to one unit (e.g. wh) when gCO2 readings aren't comparable across runs.
_units = (("co2", "gCO2"), ("wh", "Wh"), ("s", "s"))
if carbon_unit_opt != "auto":
    _units = [(k, u) for k, u in _units if k == carbon_unit_opt]
    if not _units:
        sys.exit("--unit must be one of auto,co2,wh,s")
for key, unit in _units:
    x = sum((col(c).fillna(0) for c in PHASE_COLS[carbon_phase][key]), pd.Series(0.0, index=d.index))
    if x.sum() > 0:
        carbon, carbon_unit = x, unit
        break
else:
    sys.exit(f"no {carbon_phase} {carbon_unit_opt} energy/time columns with data in {path}")

if co2_factor is not None:  # Wh * (g/kWh) / 1000 = gCO2, one fixed intensity for every run
    carbon = carbon * co2_factor / 1000.0
    carbon_unit = f"gCO2 @{co2_factor:g} g/kWh"

if metric not in d:
    sys.exit(f"metric column '{metric}' not in {path}")
d = d.assign(_carbon=carbon, _acc=col(metric))
d = d.dropna(subset=["_acc"]).sort_values("_carbon").reset_index(drop=True)

# Pareto frontier: lowest carbon first; keep a point if it beats the best metric so far.
best, frontier = -1e9, []
for i, r in d.iterrows():
    if r["_acc"] > best:
        best = r["_acc"]
        frontier.append(i)

plt.figure(figsize=(8, 6))
plt.scatter(d["_carbon"], d["_acc"], c="tab:blue", zorder=3)
plt.plot(d.loc[frontier, "_carbon"], d.loc[frontier, "_acc"], "-o", c="tab:red", label="Pareto frontier", zorder=4)
def _label(r):
    # --rowid: show ONLY the CSV row index (declutter; look everything else up in the file by #N).
    if rowid:
        return f"#{int(r['_row'])}"
    # otherwise the descriptive label: results.csv rows carry train_mode + trunk;
    # results_quant.csv rows carry a quantize mode.
    bb = r.get("backbone", "?")
    if "quantize" in d and pd.notna(r.get("quantize")) and str(r.get("quantize")) not in ("", "none", "fp16"):
        return f"{bb}/{r.get('quantize')}"
    return f"{bb}/{r.get('train_mode', '?')}/trunk{_has_trunk(r.get('pretrained_trunk'))}"


for _, r in d.iterrows():
    plt.annotate(_label(r), (r["_carbon"], r["_acc"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
if logx:
    plt.xscale("log")
if logy:
    plt.yscale("log")
plt.xlabel(f"{PHASE_LABEL[carbon_phase]}, {carbon_unit}" + ("  [log]" if logx else ""))
plt.ylabel(ylabel + ("  [log]" if logy else ""))
plt.title(f"{ylabel} vs carbon — Pareto")
plt.legend()
plt.tight_layout()
plt.savefig(outfile, dpi=130)
print(f"{len(d)} runs | metric={metric} | carbon={carbon_phase} | frontier: "
      + ", ".join(f"{_label(d.loc[i])} "
                  f"({d.loc[i, '_carbon']:.2f} {carbon_unit}, {d.loc[i, '_acc']:.3f})" for i in frontier))
print(f"wrote {outfile}")
