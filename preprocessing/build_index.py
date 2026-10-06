"""Run ONCE: build png_index.csv for the VinDr-Mammo notebook from the already-baked PNGs.

The 512/1024px PNGs (breast crop + CLAHE) already exist under
  experiments/processed_dataset_1024/images/<study>_<lat>_<view>.png
with a manifest at processed_dataset_1024/processed_index.csv. We only reshape that
manifest into the column layout the notebook expects and derive the binary label.

Two quirks handled here (see project memory):
  * `png_path` in processed_index.csv may miss a path segment on the original machine -> fixed.
  * `breast_birads` is 0-indexed (0 = BI-RADS 1, ..., 4 = BI-RADS 5).

Binary "suspicious" label (matches the binary_suspicious project convention):
  label 0 = normal/benign   -> BI-RADS 1-2  (breast_birads in {0, 1})
  label 1 = suspicious      -> BI-RADS 3-5  (breast_birads >= 2)

Output: png_index.csv with columns png_path,patient_id,laterality,view,label,split
(same schema as cbis_mass_resnet18/png_index.csv, so the notebook code is identical).
"""
import os

import pandas as pd

DATA_ROOT = os.environ.get("CARBONBREAST_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
SRC = os.path.join(DATA_ROOT, "processed_dataset_1024", "processed_index.csv")   # written by preprocess.py
SUSPICIOUS_BIRADS = 2  # 0-indexed: >= 2 means BI-RADS 3, 4 or 5


def fix_path(p):
    # processed_index paths drop the `experiments/` segment; put it back.
    if "/CarbonBreast/experiments/" not in p:
        p = p.replace("/CarbonBreast/", "/CarbonBreast/experiments/")
    return p


def main():
    df = pd.read_csv(SRC)
    df["png_path"] = df["png_path"].map(fix_path)

    missing = (~df["png_path"].map(os.path.exists)).sum()
    if missing:
        print(f"warning: {missing} PNGs referenced but not found on disk")

    out = pd.DataFrame({
        "png_path": df["png_path"],
        "patient_id": df["study_id"],                       # group key for the split
        "laterality": df["laterality_std"],
        "view": df["view_position"],
        "label": (df["breast_birads"] >= SUSPICIOUS_BIRADS).astype(int),
        # VinDr calls the train fold "training"; the notebook expects "train"
        "split": df["split"].replace({"training": "train"}),
    })
    out.to_csv("png_index.csv", index=False)

    print(f"wrote {len(out)} rows to png_index.csv")
    print("by split:", out.groupby("split").size().to_dict())
    print("by label:", out.groupby("label").size().to_dict())
    for s in ["train", "test"]:
        d = out[out.split == s]
        frac = d.label.mean() if len(d) else 0.0
        print(f"  {s}: {len(d)} images | suspicious frac={frac:.3f}")


if __name__ == "__main__":
    main()
