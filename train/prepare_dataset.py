"""
Builds train/labels.csv (columns: image,hb) from the Kaggle "Eyes Defy Anemia" dataset.

1. Download: search kaggle.com for "Eyes Defy Anemia" and unzip it anywhere.
2. Run:  python train/prepare_dataset.py "C:/path/to/unzipped/folder"

The dataset has one folder per patient (India/ and Italy/), each with eye photos,
plus an Excel sheet with the lab haemoglobin. Folder layouts differ slightly between
versions of the download, so this script searches flexibly. If it can't match
your copy, make labels.csv by hand: one row per image, with the full image path
and the lab Hb value.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
OUT = Path(__file__).resolve().parent / "labels.csv"


def read_sheets(root: Path) -> dict[tuple[str, str], float]:
    """Returns {(country, patient_number): hb}."""
    try:
        import pandas as pd
    except ImportError:
        sys.exit("Please install pandas and openpyxl:  pip install pandas openpyxl")

    labels: dict[tuple[str, str], float] = {}
    for xl in list(root.rglob("*.xlsx")) + list(root.rglob("*.xls")) + list(root.rglob("*.csv")):
        df = pd.read_csv(xl) if xl.suffix == ".csv" else pd.read_excel(xl)
        cols = {c.lower().strip(): c for c in df.columns.astype(str)}
        num_col = next((cols[c] for c in cols if c in ("number", "no", "id", "patient", "n")), df.columns[0])
        hb_col = next((cols[c] for c in cols if "hgb" in c or "hb" in c or "hemoglobin" in c or "haemoglobin" in c), None)
        if hb_col is None:
            print(f"  skipping {xl.name}: no Hb column found ({list(df.columns)})")
            continue
        country = "india" if "india" in str(xl).lower() else "italy" if "italy" in str(xl).lower() else ""
        for _, row in df.iterrows():
            try:
                hb = float(str(row[hb_col]).replace(",", "."))
                num = str(int(float(row[num_col])))
            except (ValueError, TypeError):
                continue
            labels[(country, num)] = hb
        print(f"  read {xl.name}: Hb column '{hb_col}'")
    return labels


def pick_image(folder: Path) -> Path | None:
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT]
    if not imgs:
        return None
    # Prefer the cropped palpebral (inner eyelid) image if the dataset provides one.
    for key in ("palpebral", "conjunctiva", "crop"):
        hit = [p for p in imgs if key in p.name.lower()]
        if hit:
            return hit[0]
    return sorted(imgs)[0]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    labels = read_sheets(root)
    if not labels:
        sys.exit("No haemoglobin labels found. Make train/labels.csv by hand (columns: image,hb).")

    rows = []
    for folder in root.rglob("*"):
        if not folder.is_dir() or not re.fullmatch(r"\d+", folder.name):
            continue
        path_l = str(folder).lower()
        country = "india" if "india" in path_l else "italy" if "italy" in path_l else ""
        hb = labels.get((country, folder.name)) or labels.get(("", folder.name))
        img = pick_image(folder)
        if hb is not None and img is not None:
            rows.append({"image": str(img.resolve()), "hb": hb, "cropped": int("palpebral" in img.name.lower())})

    if not rows:
        sys.exit("Found labels but couldn't match them to image folders. Make train/labels.csv by hand.")
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "hb", "cropped"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
