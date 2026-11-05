"""
Fine-tunes MobileNetV3-Small to predict haemoglobin from eyelid photos.

Easiest on Google Colab (free GPU):
    !pip install torch torchvision pillow numpy
    upload this file + labels.csv + images, then:
    !python train_cnn.py labels.csv --epochs 40

Locally:  pip install torch torchvision
          python train/train_cnn.py

Saves model/hemascan_cnn.pt. Copy it into hemascan/model/ and restart the app.
The app then blends the CNN with the colour method and shows a Grad-CAM heatmap.

Note: no colour augmentation. Colour IS the signal (pale vs red),
so changing hue or saturation would destroy the label.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class EyelidDataset(Dataset):
    def __init__(self, rows, train: bool):
        self.rows = rows
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
               transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.8, 1.6))] if train else [transforms.Resize((224, 224))]
        self.tf = transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = ImageOps.exif_transpose(Image.open(r["image"])).convert("RGB")
        if r.get("cropped", "0") != "1":  # use the centre band if not pre-cropped
            W, H = img.size
            img = img.crop((int(.15 * W), int(.3 * H), int(.85 * W), int(.75 * H)))
        return self.tf(img), torch.tensor([float(r["hb"])], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(Path(__file__).parent / "labels.csv"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threshold", type=float, default=11.5)
    ap.add_argument("--out", default=str(ROOT / "model" / "hemascan_cnn.pt"))
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    rows = list(csv.DictReader(open(args.csv, newline="")))
    random.shuffle(rows)
    split = int(0.8 * len(rows))
    train_rows, val_rows = rows[:split], rows[split:]
    print(f"train {len(train_rows)}  val {len(val_rows)}")

    dl_train = DataLoader(EyelidDataset(train_rows, True), batch_size=args.batch, shuffle=True, num_workers=2)
    dl_val = DataLoader(EyelidDataset(val_rows, False), batch_size=args.batch, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    nn.init.constant_(model.classifier[-1].bias, float(np.mean([float(r["hb"]) for r in train_rows])))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.SmoothL1Loss()
    best, best_metrics = 1e9, {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in dl_val:
                preds += model(x.to(device)).cpu().flatten().tolist()
                gts += y.flatten().tolist()
        p, t = np.array(preds), np.array(gts)
        mae = float(np.abs(p - t).mean())
        ta, pa = t < args.threshold, p < args.threshold
        sens = float((ta & pa).sum() / max(1, ta.sum()))
        spec = float((~ta & ~pa).sum() / max(1, (~ta).sum()))
        print(f"epoch {epoch:3d}  val MAE {mae:.2f} g/dL  sensitivity {sens:.2f}  specificity {spec:.2f}")
        if mae < best:
            best = mae
            best_metrics = {"val_mae_g_dl": round(mae, 2), "sensitivity": round(sens, 3), "specificity": round(spec, 3),
                            "threshold_g_dl": args.threshold, "epoch": epoch, "n_train": len(train_rows), "n_val": len(val_rows)}
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "metrics": best_metrics}, args.out)

    Path(args.out).with_name("cnn_metrics.json").write_text(json.dumps(best_metrics, indent=2))
    print("best:", best_metrics, "\nsaved", args.out)


if __name__ == "__main__":
    main()
