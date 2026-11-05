"""
Optional deep-learning model (MobileNetV3 regression of Hb + Grad-CAM).

Only used if PyTorch is installed AND model/hemascan_cnn.pt exists
(created by train/train_cnn.py). Otherwise the app uses the colour method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_FILE = Path(__file__).resolve().parent.parent / "model" / "hemascan_cnn.pt"
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model = None
_available: bool | None = None


def build_model():
    import torch.nn as nn
    from torchvision import models

    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    return m


def available() -> bool:
    global _available, _model
    if _available is not None:
        return _available
    try:
        import torch

        if not MODEL_FILE.exists():
            _available = False
            return False
        m = build_model()
        state = torch.load(MODEL_FILE, map_location="cpu")
        m.load_state_dict(state["model"] if "model" in state else state)
        m.eval()
        _model = m
        _available = True
    except Exception as exc:
        print(f"[cnn] not available: {exc}")
        _available = False
    return _available


def preprocess(roi_rgb: np.ndarray):
    import torch
    from PIL import Image

    im = Image.fromarray(roi_rgb).resize((IMG_SIZE, IMG_SIZE))
    x = (np.asarray(im).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def predict_with_gradcam(roi_rgb: np.ndarray) -> tuple[float, np.ndarray]:
    """Returns (hb_estimate, cam 0..1 same size as roi)."""
    import torch
    from PIL import Image

    x = preprocess(roi_rgb).requires_grad_(False)
    feats = {}

    target = _model.features[-1]

    def fwd_hook(module, inputs, output):
        feats["a"] = output
        output.register_hook(lambda grad: feats.__setitem__("g", grad))

    h1 = target.register_forward_hook(fwd_hook)
    try:
        out = _model(x)
        hb = float(out.item())
        _model.zero_grad()
        # Grad-CAM on the *negative* output: highlights areas pushing Hb down (pale areas).
        (-out).sum().backward()
        a, g = feats["a"][0], feats["g"][0]
        weights = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * a).sum(0)).detach().numpy()
    finally:
        h1.remove()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (roi_rgb.shape[1], roi_rgb.shape[0]), Image.BILINEAR
    )
    return float(np.clip(hb, 4.0, 18.0)), np.asarray(cam_img).astype(np.float32) / 255.0
