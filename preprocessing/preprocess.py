"""DICOM → PNG preprocessing for VinDr-Mammo two-view BI-RADS classifier.

For each DICOM:
  1. Load with pydicom; apply RescaleSlope/Intercept.
  2. Apply window/level from DICOM tags. Handle MONOCHROME1 inversion.
  3. Normalize to uint8 [0,255].
  4. Otsu breast crop: largest connected component, tight bbox.
  5. CLAHE.
  6. MLO pectoral removal (after laterality standardization, simple line fit).
  7. Laterality flip if 'L' so the breast always points right.
  8. Resize long-side to target_size, pad to square.
  9. Transform finding-annotation bboxes through every geometric step.
 10. Write PNG to processed_dataset_<size>/images/, append row to index.

Run:
    python preprocessing/preprocess.py --target-size 1024 --workers 8
    python preprocessing/preprocess.py --target-size 1024 --limit 20 --visualize 5
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "original_dataset"


# ---------------------------------------------------------------------------
# DICOM normalization
# ---------------------------------------------------------------------------
def _first_scalar(v):
    """DICOM multi-value tags can be a single value or an iterable."""
    if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
        return float(list(v)[0])
    return float(v)


def normalize_dicom(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """Returns a uint8 image with tissue bright on a black background."""
    px = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    px = px * slope + intercept

    is_mono1 = getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1"
    if is_mono1:
        px = px.max() - px

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        wc = _first_scalar(wc)
        ww = _first_scalar(ww)
        if is_mono1:
            wc = px.max() - wc
        if ww <= 0:
            ww = float(px.max() - px.min() + 1)
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
    else:
        lo = float(np.percentile(px[px > px.min()], 0.5)) if (px > px.min()).any() else float(px.min())
        hi = float(np.percentile(px[px > px.min()], 99.5)) if (px > px.min()).any() else float(px.max())
        if hi <= lo:
            hi = lo + 1.0

    out = np.clip(px, lo, hi)
    out = (out - lo) / (hi - lo) * 255.0
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometric pipeline
# ---------------------------------------------------------------------------
@dataclass
class Geom:
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    flipped: bool
    scale: float
    pad_x: int
    pad_y: int
    final_size: int


def otsu_breast_crop(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Returns (cropped_image, (x, y, w, h))."""
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return img, (0, 0, img.shape[1], img.shape[0])
    # skip background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h = int(stats[largest, cv2.CC_STAT_HEIGHT])
    cropped = img[y:y + h, x:x + w]
    # zero out anything outside the largest-component mask within the crop box
    mask = (labels[y:y + h, x:x + w] == largest).astype(np.uint8)
    cropped = cropped * mask
    return cropped, (x, y, w, h)


def apply_clahe(img: np.ndarray, clip: float, tile: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(img)


def remove_pectoral_mlo(img_std: np.ndarray) -> np.ndarray:
    """Pectoral removal on STANDARDIZED MLO (breast points right, pectoral upper-left).

    Simple straight-line approach: threshold the upper-left ROI, find the
    component touching (0, 0), fit a line to its right edge, zero the wedge.

    TODO: replace with a U-Net or learned mask for better accuracy on dense breasts.
    """
    H, W = img_std.shape
    roi_h = int(H * 0.55)
    roi_w = int(W * 0.45)
    roi = img_std[:roi_h, :roi_w]
    if roi.size == 0:
        return img_std

    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num, labels = cv2.connectedComponents(bw, connectivity=8)
    corner_lbl = int(labels[0, 0])
    if corner_lbl == 0:
        return img_std  # corner is background, no pectoral wedge

    pec = (labels == corner_lbl)
    if pec.sum() < 0.005 * roi.size:
        return img_std  # too small to be the pectoral

    # rightmost pectoral pixel in each row (the diagonal edge)
    rows, cols = [], []
    for r in range(roi_h):
        idx = np.where(pec[r])[0]
        if len(idx) > 0:
            rows.append(r)
            cols.append(int(idx.max()))
    if len(rows) < 8:
        return img_std

    a, b = np.polyfit(rows, cols, 1)  # col = a*row + b
    out = img_std.copy()
    # zero out the wedge over the FULL image (not just the ROI) to be safe
    for r in range(H):
        c_edge = int(a * r + b)
        if c_edge <= 0:
            continue
        c_edge = min(c_edge, W)
        out[r, :c_edge] = 0
    return out


def resize_pad_square(img: np.ndarray, target: int) -> tuple[np.ndarray, float, int, int]:
    """Resize so the long side = target, then pad with zeros to (target, target)."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=img.dtype)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def transform_bboxes(bboxes: list[tuple[float, float, float, float]], g: Geom):
    """Apply crop -> flip -> resize -> pad to a list of (xmin, ymin, xmax, ymax)."""
    out = []
    for x1, y1, x2, y2 in bboxes:
        # crop
        x1 -= g.crop_x; x2 -= g.crop_x
        y1 -= g.crop_y; y2 -= g.crop_y
        # clip to crop
        x1 = max(0, min(x1, g.crop_w))
        x2 = max(0, min(x2, g.crop_w))
        y1 = max(0, min(y1, g.crop_h))
        y2 = max(0, min(y2, g.crop_h))
        if x2 <= x1 or y2 <= y1:
            continue
        # flip
        if g.flipped:
            x1, x2 = g.crop_w - x2, g.crop_w - x1
        # resize
        x1 *= g.scale; x2 *= g.scale
        y1 *= g.scale; y2 *= g.scale
        # pad
        x1 += g.pad_x; x2 += g.pad_x
        y1 += g.pad_y; y2 += g.pad_y
        # clamp to canvas
        x1 = max(0.0, min(x1, g.final_size))
        x2 = max(0.0, min(x2, g.final_size))
        y1 = max(0.0, min(y1, g.final_size))
        y2 = max(0.0, min(y2, g.final_size))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((float(x1), float(y1), float(x2), float(y2)))
    return out


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
def parse_birads(s: str) -> int:
    # "BI-RADS 3" -> 2 (0-indexed)
    return int(str(s).strip().split()[-1]) - 1


def process_one(args_tuple):
    """Worker function. Returns a dict (one row of processed_index.csv) or None on error."""
    (
        study_id, image_id, laterality, view_position, split, birads_str,
        density, raw_bboxes, dicom_path, out_png_path, target_size, clahe_clip,
        clahe_tile, skip_existing,
    ) = args_tuple

    try:
        if skip_existing and os.path.exists(out_png_path):
            # still need to recompute bbox transform (cheap) to keep index in sync.
            # For simplicity, only skip the heavy DICOM step but recompute geom by
            # reading the existing PNG dims. The existing PNG is already target_size.
            # We DO need the geom to transform bboxes, so we re-run the pipeline.
            pass

        ds = pydicom.dcmread(dicom_path)
        norm = normalize_dicom(ds)

        # Otsu crop
        cropped, (cx, cy, cw, ch) = otsu_breast_crop(norm)
        if cropped.size == 0 or cw < 16 or ch < 16:
            return {"error": "empty crop", "study_id": study_id, "image_id": image_id}

        # CLAHE
        cropped = apply_clahe(cropped, clahe_clip, clahe_tile)

        # Laterality standardization: flip if L so breast points right.
        flipped = (laterality == "L")
        std_img = cv2.flip(cropped, 1) if flipped else cropped

        # MLO pectoral removal (after standardization)
        if view_position == "MLO":
            std_img = remove_pectoral_mlo(std_img)

        # Resize + pad to square
        canvas, scale, pad_x, pad_y = resize_pad_square(std_img, target_size)

        g = Geom(
            crop_x=cx, crop_y=cy, crop_w=cw, crop_h=ch,
            flipped=flipped, scale=scale, pad_x=pad_x, pad_y=pad_y,
            final_size=target_size,
        )
        bboxes_out = transform_bboxes(raw_bboxes, g)

        Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_png_path, canvas)

        return {
            "study_id": study_id,
            "image_id": image_id,
            "laterality_orig": laterality,
            "laterality_std": "R",   # all standardized to right-pointing
            "view_position": view_position,
            "split": split,
            "breast_birads": parse_birads(birads_str),
            "breast_density": density,
            "has_bbox": int(len(bboxes_out) > 0),
            "bboxes_xyxy_json": json.dumps(bboxes_out),
            "png_path": str(out_png_path),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "study_id": study_id, "image_id": image_id}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(rows: list[dict], output_dir: Path, n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    bboxed = [r for r in rows if r.get("has_bbox")]
    no_bbox = [r for r in rows if not r.get("has_bbox") and "png_path" in r]
    take_bb = list(rng.choice(bboxed, size=min(n, len(bboxed)), replace=False)) if bboxed else []
    take_nb = list(rng.choice(no_bbox, size=min(n - len(take_bb), len(no_bbox)),
                              replace=False)) if no_bbox else []
    samples = take_bb + take_nb

    for i, r in enumerate(samples):
        img = cv2.imread(r["png_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for b in json.loads(r.get("bboxes_xyxy_json", "[]")):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = (f"{r['study_id'][:8]}  {r['laterality_orig']}-{r['view_position']}  "
                 f"BIRADS {r['breast_birads'] + 1}  bbox={r['has_bbox']}")
        cv2.putText(rgb, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.imwrite(str(output_dir / f"sanity_{i:02d}.png"), rgb)
    print(f"Wrote {len(samples)} sanity images to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_jobs(breast_csv: Path, finding_csv: Path, images_dir: Path,
               out_root: Path, target_size: int, clahe_clip: float,
               clahe_tile: int, skip_existing: bool, limit: int | None):
    bdf = pd.read_csv(breast_csv)
    fdf = pd.read_csv(finding_csv)

    # group bboxes by image_id
    bb_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
    for _, row in fdf.iterrows():
        if pd.isna(row.get("xmin")) or pd.isna(row.get("ymin")) \
                or pd.isna(row.get("xmax")) or pd.isna(row.get("ymax")):
            continue
        bb_by_image.setdefault(row["image_id"], []).append(
            (float(row["xmin"]), float(row["ymin"]),
             float(row["xmax"]), float(row["ymax"]))
        )

    rows = bdf.to_dict("records")
    if limit:
        # take whole studies (not arbitrary rows) to keep breasts complete
        keep_studies = set(pd.unique(bdf["study_id"])[:limit])
        rows = [r for r in rows if r["study_id"] in keep_studies]

    images_subdir = out_root / "images"
    images_subdir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for r in rows:
        sid = r["study_id"]; img_id = r["image_id"]
        dicom_path = images_dir / sid / f"{img_id}.dicom"
        out_png = images_subdir / f"{sid}_{r['laterality']}_{r['view_position']}.png"
        jobs.append((
            sid, img_id, r["laterality"], r["view_position"], r["split"],
            r["breast_birads"], r["breast_density"], bb_by_image.get(img_id, []),
            str(dicom_path), str(out_png), target_size, clahe_clip, clahe_tile,
            skip_existing,
        ))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--out-root", type=Path, default=None,
                   help="Defaults to processed_dataset_<target_size>/")
    p.add_argument("--target-size", type=int, default=1024)
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-tile", type=int, default=8)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N study_ids (for smoke test)")
    p.add_argument("--visualize", type=int, default=0,
                   help="After processing, save N sanity-check overlays")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    images_dir = args.dataset_dir / "images"
    breast_csv = args.dataset_dir / "breast-level_annotations.csv"
    finding_csv = args.dataset_dir / "finding_annotations.csv"
    out_root = args.out_root or (PROJECT_ROOT / f"processed_dataset_{args.target_size}")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:      {args.dataset_dir}")
    print(f"Output root:  {out_root}")
    print(f"Target size:  {args.target_size}")
    print(f"Workers:      {args.workers}")

    jobs = build_jobs(breast_csv, finding_csv, images_dir, out_root,
                      args.target_size, args.clahe_clip, args.clahe_tile,
                      args.skip_existing, args.limit)
    print(f"Jobs:         {len(jobs)}")

    rows: list[dict] = []
    errors: list[dict] = []

    if args.workers <= 1:
        for j in tqdm(jobs, desc="preprocess"):
            r = process_one(j)
            if r is None:
                continue
            (errors if "error" in r else rows).append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocess"):
                r = fut.result()
                if r is None:
                    continue
                (errors if "error" in r else rows).append(r)

    print(f"OK: {len(rows)}   Errors: {len(errors)}")
    if errors[:5]:
        print("First few errors:")
        for e in errors[:5]:
            print(" ", e)

    rows_df = pd.DataFrame(rows).sort_values(["study_id", "laterality_orig", "view_position"])
    out_csv = out_root / "processed_index.csv"
    rows_df.to_csv(out_csv, index=False)
    print(f"Wrote index: {out_csv} ({len(rows_df)} rows)")

    if args.visualize > 0:
        visualize(rows, out_root / "sanity", args.visualize)


if __name__ == "__main__":
    main()
