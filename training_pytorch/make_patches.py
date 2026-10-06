"""Stage-1 patch data: 224x224 patches for the lesion-vs-background patch classifier.

Ported from vindr_multiview/preprocess/make_patches.py, using EXACT per-finding BI-RADS labels
(joined from finding_annotations.csv) rather than the image-level breast_birads.

Why we re-derive the boxes instead of reusing processed_index.csv's `bboxes_xyxy_json`: that
column drops boxes that fall outside the Otsu crop and mixes in BI-RADS 1/2 boxed findings, so it
cannot be zipped back to finding rows to recover per-box BI-RADS. Instead we recompute the exact
geometric transform (Otsu crop -> flip -> resize -> pad) with the SAME functions that produced
processed_dataset_1024 (preprocessing/preprocess.py) and carry each finding's
BI-RADS through it. Verified: re-derived boxes match the stored boxes to 0.0px, so they align with
the already-baked PNGs -- which is what we crop the patches from (no re-baking of images).

Classes: background / birads3 / birads4 / birads5 (per-finding). Patch-val is carved from the
*training* studies, so the full-image TEST set is never touched here.

Output: patches/<patch_train|patch_val>/<class>/<image_id>_*.png  and  patch_index.csv

Run:
    python make_patches.py            # full
    python make_patches.py --limit 50 # quick smoke
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import GroupShuffleSplit

# reuse the exact geometric pipeline that baked processed_dataset_1024
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))
from preprocess import (  # noqa: E402
    Geom, normalize_dicom, otsu_breast_crop, resize_pad_square, transform_bboxes,
)

# Data locations: set CARBONBREAST_DATA to the folder that holds VINDR-MAMMO/ (the PhysioNet download) and
# processed_dataset_1024/ (output of preprocessing/preprocess.py). Default: <repo>/data.
DATA_ROOT = os.environ.get("CARBONBREAST_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
PROC_INDEX = os.path.join(DATA_ROOT, "processed_dataset_1024", "processed_index.csv")
FINDINGS = os.path.join(DATA_ROOT, "VINDR-MAMMO", "finding_annotations.csv")
DICOM_ROOT = os.path.join(DATA_ROOT, "VINDR-MAMMO", "images")
TARGET = 1024

OUT_ROOT = "patches"
SIZE = 224
JITTER = 0.10
N_BG = 10
NONZERO_THRESH = 0.10
N_LESION_PER_CLASS = {"birads3": 10, "birads4": 10, "birads5": 16}  # oversample rare birads5
CLASS_OF = {3: "birads3", 4: "birads4", 5: "birads5"}
CLASS_LABEL = {"background": 0, "birads3": 1, "birads4": 2, "birads5": 3}


def fix_path(p):
    if "/CarbonBreast/experiments/" not in p:
        p = p.replace("/CarbonBreast/", "/CarbonBreast/experiments/")
    return p


def parse_finding_birads(s):
    """'BI-RADS 4' -> 4; None if not a 3/4/5 lesion."""
    try:
        v = int(str(s).strip().split()[-1])
        return v if v in (3, 4, 5) else None
    except Exception:  # noqa: BLE001
        return None


def rederive_geom(study_id, image_id, laterality_orig):
    """Recompute the exact crop->flip->resize->pad Geom used to bake processed_dataset_1024."""
    ds = pydicom.dcmread(f"{DICOM_ROOT}/{study_id}/{image_id}.dicom")
    norm = normalize_dicom(ds)
    cropped, (cx, cy, cw, ch) = otsu_breast_crop(norm)
    flipped = (laterality_orig == "L")
    std = np.fliplr(cropped) if flipped else cropped
    _, scale, pad_x, pad_y = resize_pad_square(std, TARGET)
    return Geom(cx, cy, cw, ch, flipped, scale, pad_x, pad_y, TARGET)


def crop_patch(img, cx, cy, size):
    h, w = img.shape[:2]
    half = size // 2
    x0 = int(round(min(max(cx - half, 0), w - size)))
    y0 = int(round(min(max(cy - half, 0), h - size)))
    return img[y0:y0 + size, x0:x0 + size], x0, y0


def overlaps(box, x0, y0, size):
    ax1, ay1, ax2, ay2 = box
    return not (ax2 <= x0 or x0 + size <= ax1 or ay2 <= y0 or y0 + size <= ay1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    proc = pd.read_csv(PROC_INDEX)
    proc["png_path"] = proc["png_path"].map(fix_path)
    proc = proc.set_index("image_id")

    # per-image list of (finding_birads, raw DICOM box), BI-RADS 3/4/5 boxed findings only
    fdf = pd.read_csv(FINDINGS)
    lesions = {}
    for r in fdf.itertuples():
        bir = parse_finding_birads(r.finding_birads)
        if bir is None or any(pd.isna(getattr(r, k)) for k in ("xmin", "ymin", "xmax", "ymax")):
            continue
        lesions.setdefault(r.image_id, []).append(
            (bir, (float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax))))

    # training-split lesion images that exist in the processed index
    image_ids = [i for i in lesions if i in proc.index and proc.loc[i, "split"] == "training"]
    if args.limit:
        image_ids = image_ids[:args.limit]

    # study-level patch train/val carve
    studies = np.array([proc.loc[i, "study_id"] for i in image_ids])
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=args.val_frac, random_state=args.seed)
                      .split(image_ids, groups=studies))
    split_of = {image_ids[i]: "patch_train" for i in tr_i}
    split_of.update({image_ids[i]: "patch_val" for i in va_i})

    rows, counts, n_err = [], {}, 0
    for n, image_id in enumerate(image_ids):
        prow = proc.loc[image_id]
        try:
            g = rederive_geom(prow.study_id, image_id, prow.laterality_orig)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            if n_err <= 3:
                print(f"  skip {image_id[:8]}: {type(e).__name__}: {e}")
            continue
        img = cv2.imread(prow.png_path, cv2.IMREAD_GRAYSCALE)  # crop patches from the BAKED PNG
        if img is None:
            n_err += 1
            continue
        h, w = img.shape[:2]
        split = split_of[image_id]

        # transform each finding box individually so BI-RADS stays aligned through drops
        boxes = []  # (birads, [x1,y1,x2,y2]) in 1024 frame
        for bir, raw in lesions[image_id]:
            tb = transform_bboxes([raw], g)
            if tb:
                boxes.append((bir, list(tb[0])))
        if not boxes:
            continue

        # ---- lesion patches ----
        for li, (bir, (bx1, by1, bx2, by2)) in enumerate(boxes):
            cls = CLASS_OF[bir]
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            bw, bh = max(bx2 - bx1, 1), max(by2 - by1, 1)
            for k in range(N_LESION_PER_CLASS[cls]):
                dx = rng.uniform(-JITTER, JITTER) * bw
                dy = rng.uniform(-JITTER, JITTER) * bh
                patch, _, _ = crop_patch(img, bcx + dx, bcy + dy, SIZE)
                if patch.shape[:2] != (SIZE, SIZE):
                    continue
                d = os.path.join(OUT_ROOT, split, cls)
                os.makedirs(d, exist_ok=True)
                p = os.path.join(d, f"{image_id}_les{li}_{k}.png")
                cv2.imwrite(p, patch)
                rows.append((p, CLASS_LABEL[cls], cls, split, prow.study_id))
                counts[(split, cls)] = counts.get((split, cls), 0) + 1

        # ---- background patches (no overlap with any lesion box) ----
        boxes_only = [b for _, b in boxes]
        got, tries = 0, 0
        while got < N_BG and tries < N_BG * 40:
            tries += 1
            cxp = rng.uniform(SIZE / 2, w - SIZE / 2)
            cyp = rng.uniform(SIZE / 2, h - SIZE / 2)
            patch, x0, y0 = crop_patch(img, cxp, cyp, SIZE)
            if patch.shape[:2] != (SIZE, SIZE):
                continue
            if any(overlaps(b, x0, y0, SIZE) for b in boxes_only):
                continue
            if (patch > 0.05 * 255).mean() < NONZERO_THRESH:
                continue  # mostly-black crop
            d = os.path.join(OUT_ROOT, split, "background")
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f"{image_id}_bg{got}.png")
            cv2.imwrite(p, patch)
            rows.append((p, CLASS_LABEL["background"], "background", split, prow.study_id))
            counts[(split, "background")] = counts.get((split, "background"), 0) + 1
            got += 1

        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(image_ids)} images  ({len(rows)} patches)", flush=True)

    out = pd.DataFrame(rows, columns=["png_path", "label", "class_name", "split", "study_id"])
    out.to_csv("patch_index.csv", index=False)
    print(f"\nwrote {len(out)} patches from {len(image_ids) - n_err} images "
          f"({n_err} skipped) -> patch_index.csv")
    for split in ("patch_train", "patch_val"):
        line = "  ".join(f"{c}={counts.get((split, c), 0)}"
                         for c in ("background", "birads3", "birads4", "birads5"))
        print(f"  {split}: {line}")


if __name__ == "__main__":
    main()
