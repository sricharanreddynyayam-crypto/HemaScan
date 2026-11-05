"""
Fits the colour-pallor model on real photos with lab Hb values, and reports accuracy.

    python train/calibrate.py                 # uses train/labels.csv
    python train/calibrate.py my_labels.csv --threshold 11.5

labels.csv columns: image, hb   (optional: cropped=1 if the image is already
just the inner eyelid; otherwise the centre of the image is used).

Writes:
  model/calibration.json  (used by the app automatically on next start)
  model/metrics.json      (put these numbers in your write-up and video)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import analysis  # noqa: E402

FULL = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
CENTRE = {"x": 0.15, "y": 0.3, "w": 0.7, "h": 0.45}


def features_for(path: str, cropped: bool) -> tuple[float, float] | None:
    try:
        img = analysis.load_image(Path(path).read_bytes())
    except Exception as exc:
        print(f"  skip {path}: {exc}")
        return None
    roi = np.asarray(analysis.crop_roi(img, FULL if cropped else CENTRE))
    mask, _ = analysis.tissue_mask(roi)
    f = analysis.extract_features(roi, mask)
    return f["erythema_index"], f["a_star"]


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.c_[np.ones(len(X)), X]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ coef


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> dict:
    t, p = y_true < thr, y_pred < thr
    tp, tn = int((t & p).sum()), int((~t & ~p).sum())
    fp, fn = int((~t & p).sum()), int((t & ~p).sum())
    return {
        "threshold_g_dl": thr,
        "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
        "accuracy": round((tp + tn) / len(t), 3),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(Path(__file__).parent / "labels.csv"))
    ap.add_argument("--threshold", type=float, default=11.5, help="Hb cut-off for anaemia (g/dL)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--two-feature", action="store_true", help="also use CIELAB a* (may overfit on small data)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="")))
    X, y = [], []
    for r in rows:
        f = features_for(r["image"], r.get("cropped", "0") == "1")
        if f:
            X.append(f if args.two_feature else f[:1])
            y.append(float(r["hb"]))
    X, y = np.array(X), np.array(y)
    n = len(y)
    if n < 10:
        sys.exit(f"Only {n} usable images; need at least 10.")
    print(f"Using {n} images. Hb range {y.min():.1f}-{y.max():.1f}, {int((y < args.threshold).sum())} anaemic.")

    # k-fold cross-validation, so the reported numbers are on images the model did not see.
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, args.folds)
    oof = np.zeros(n)
    for k in range(args.folds):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        oof[test] = predict(fit(X[train], y[train]), X[test])
    oof = np.clip(oof, 4, 18)

    mae = float(np.mean(np.abs(oof - y)))
    r = float(np.corrcoef(oof, y)[0, 1])
    metrics = {
        "n_images": n,
        "cross_validation_folds": args.folds,
        "mae_g_dl": round(mae, 2),
        "pearson_r": round(r, 3),
        **binary_metrics(y, oof, args.threshold),
    }

    coef = fit(X, y)  # final model on all data
    calib = {"intercept": float(coef[0]), "slope": float(coef[1]), "source": f"calibrated on {n} images"}
    if args.two_feature:
        calib["slope_a"] = float(coef[2])

    out = ROOT / "model"
    out.mkdir(exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(calib, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved {out / 'calibration.json'}. Restart the app to use it.")


if __name__ == "__main__":
    main()
