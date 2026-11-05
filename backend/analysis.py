"""
Image analysis for HemaScan.

Pipeline:
  1. Load photo, fix rotation, crop to the eyelid box the user drew.
  2. Quality checks: brightness, blur, glare, how much tissue is visible.
  3. Tissue mask: keep pink/red conjunctiva pixels, drop glare, eyelashes and skin-dark areas.
  4. Pallor features: erythema index (log10 R/G) and CIELAB a* (redness).
     Pale conjunctiva = less red = lower haemoglobin. This is the same
     signal doctors look for by eye ("conjunctival pallor").
  5. Haemoglobin estimate from a linear calibration (fit with train/calibrate.py).
  6. Heatmap showing which areas look pale.

If a trained CNN exists (model/hemascan_cnn.pt) and PyTorch is installed,
app.py uses it for the estimate instead and blends both results.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "model" / "calibration.json"

# Default calibration (used until you run train/calibrate.py on real data).
# Maps erythema index -> Hb in g/dL. Rough values from the pallor literature,
# NOT clinically validated.
DEFAULT_CALIBRATION = {"slope": 28.0, "intercept": 4.5, "source": "default (uncalibrated)"}

MAX_SIDE = 900


@dataclass
class QualityReport:
    ok: bool
    score: float  # 0..1
    issues: list[str] = field(default_factory=list)
    brightness: float = 0.0
    sharpness: float = 0.0
    glare_fraction: float = 0.0
    tissue_fraction: float = 0.0


def load_calibration() -> dict:
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text())
        except Exception:
            pass
    return DEFAULT_CALIBRATION


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def crop_roi(img: Image.Image, roi: dict | None) -> Image.Image:
    """roi = {x, y, w, h} as fractions (0..1) of the image size."""
    W, H = img.size
    if not roi:
        roi = {"x": 0.2, "y": 0.35, "w": 0.6, "h": 0.3}
    x = max(0.0, min(1.0, float(roi["x"])))
    y = max(0.0, min(1.0, float(roi["y"])))
    w = max(0.02, min(1.0 - x, float(roi["w"])))
    h = max(0.02, min(1.0 - y, float(roi["h"])))
    box = (int(x * W), int(y * H), int((x + w) * W), int((y + h) * H))
    return img.crop(box)


def _laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(lap.var())


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) uint8 -> Lab float (D65)."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]],
        dtype=np.float32,
    )
    xyz = c @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def tissue_mask(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (tissue_mask, glare_mask)."""
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    mx = rgb.max(axis=-1).astype(np.int32)
    mn = rgb.min(axis=-1).astype(np.int32)
    glare = mn > 242  # specular highlights (sclera is ~230, so not counted)
    dark = mx < 45
    reddish = (r > g + 8) & (r > b + 8)
    # Conjunctiva is pink-red (hue about 340-15 degrees); skin is orange (about 20-30 degrees).
    # When red is the largest channel, hue = 60 * (G - B) / (max - min).
    hue = 60.0 * (g - b) / np.maximum(mx - mn, 1)
    pink_red = (hue > -20) & (hue < 15)
    return reddish & pink_red & ~glare & ~dark, glare


def pallor_map(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel erythema index log10(R/G)."""
    r = rgb[..., 0].astype(np.float32) + 1.0
    g = rgb[..., 1].astype(np.float32) + 1.0
    return np.log10(r / g)


def check_quality(roi_rgb: np.ndarray, mask: np.ndarray, glare: np.ndarray) -> QualityReport:
    gray = roi_rgb.mean(axis=-1)
    brightness = float(gray.mean())
    sharp_src = Image.fromarray(gray.astype(np.uint8))
    sharp_src.thumbnail((320, 320))
    sharpness = _laplacian_variance(np.asarray(sharp_src))
    glare_frac = float(glare.mean())
    tissue_frac = float(mask.mean())

    issues: list[str] = []
    score = 1.0
    if brightness < 55:
        issues.append("Photo is too dark. Turn on the flash or move to better light.")
        score -= 0.35
    elif brightness > 215:
        issues.append("Photo is overexposed. Move away from direct light.")
        score -= 0.3
    if sharpness < 12:
        issues.append("Photo looks blurry. Hold the phone steady and tap to focus.")
        score -= 0.3
    if glare_frac > 0.15:
        issues.append("Strong reflections on the eyelid. Tilt the phone slightly.")
        score -= 0.2
    if tissue_frac < 0.15:
        issues.append("Inner eyelid not clearly visible. Gently pull the lower eyelid down and draw the box around the pink area.")
        score -= 0.45
    if min(roi_rgb.shape[:2]) < 40:
        issues.append("Selected area is very small. Take the photo closer.")
        score -= 0.2

    score = max(0.0, min(1.0, score))
    return QualityReport(
        ok=score >= 0.5 and tissue_frac >= 0.08,
        score=round(score, 2),
        issues=issues,
        brightness=round(brightness, 1),
        sharpness=round(sharpness, 1),
        glare_fraction=round(glare_frac, 3),
        tissue_fraction=round(tissue_frac, 3),
    )


def extract_features(roi_rgb: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() < 20:
        pix = roi_rgb.reshape(-1, 3)
    else:
        pix = roi_rgb[mask]
    ei = pallor_map(pix[None, ...])[0]
    lab = rgb_to_lab(pix)
    return {
        "erythema_index": float(np.median(ei)),
        "a_star": float(np.median(lab[:, 1])),
        "lightness": float(np.median(lab[:, 0])),
        "mean_rgb": [float(v) for v in pix.mean(axis=0)],
        "pixels_used": int(len(pix)),
    }


def estimate_hb(features: dict, calib: dict | None = None) -> float:
    calib = calib or load_calibration()
    hb = calib["intercept"] + calib["slope"] * features["erythema_index"]
    if "slope_a" in calib:  # optional 2-feature calibration
        hb += calib["slope_a"] * features["a_star"]
    return float(np.clip(hb, 4.0, 18.0))


def _colormap(t: np.ndarray) -> np.ndarray:
    """t in 0..1 (1 = pale). Green -> yellow -> red."""
    t = np.clip(t, 0, 1)[..., None]
    green = np.array([34, 197, 94], dtype=np.float32)
    yellow = np.array([250, 204, 21], dtype=np.float32)
    red = np.array([239, 68, 68], dtype=np.float32)
    lo = green + (yellow - green) * np.clip(t * 2, 0, 1)
    hi = yellow + (red - yellow) * np.clip(t * 2 - 1, 0, 1)
    return np.where(t < 0.5, lo, hi)


def make_heatmap(roi_rgb: np.ndarray, mask: np.ndarray) -> str:
    """Returns PNG data-URL of the eyelid crop with a pallor overlay."""
    ei = pallor_map(roi_rgb)
    # 0.38+ = well perfused (not pale), 0.08 or less = very pale
    pale = 1.0 - (ei - 0.08) / (0.38 - 0.08)
    # light smoothing so it looks like a heatmap, not noise
    img = Image.fromarray((np.clip(pale, 0, 1) * 255).astype(np.uint8))
    small = img.resize((max(1, img.width // 8), max(1, img.height // 8)), Image.BILINEAR)
    pale_s = np.asarray(small.resize(img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    colors = _colormap(pale_s)
    alpha = np.where(mask, 0.55, 0.0)[..., None]
    out = roi_rgb.astype(np.float32) * (1 - alpha) + colors * alpha
    return to_data_url(out.astype(np.uint8))


def to_data_url(arr: np.ndarray) -> str:
    im = Image.fromarray(arr)
    im.thumbnail((480, 480))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# WHO haemoglobin cut-offs (g/dL), WHO 2024 "Guideline on haemoglobin cutoffs to define anaemia"
# https://www.who.int/publications/i/item/9789240088542
# Pregnant uses the 1st/3rd-trimester cut-off (11.0); WHO uses 10.5 in the 2nd trimester.
# Values are for sea level; WHO adjusts for altitude and smoking (not applied here).
def thresholds(group: str) -> dict:
    """Returns cut-offs: below 'normal' = anaemia; mild/moderate/severe bands."""
    table = {
        "child_6_23m": {"normal": 10.5, "mild": 9.5, "moderate": 7.0},
        "child_24_59m": {"normal": 11.0, "mild": 10.0, "moderate": 7.0},
        "child_5_11y": {"normal": 11.5, "mild": 11.0, "moderate": 8.0},
        "child_12_14y": {"normal": 12.0, "mild": 11.0, "moderate": 8.0},
        "woman": {"normal": 12.0, "mild": 11.0, "moderate": 8.0},
        "pregnant": {"normal": 11.0, "mild": 10.0, "moderate": 7.0},
        "man": {"normal": 13.0, "mild": 11.0, "moderate": 8.0},
    }
    return table.get(group, table["woman"])


def classify(hb: float, group: str) -> str:
    t = thresholds(group)
    if hb >= t["normal"]:
        return "none"
    if hb >= t["mild"]:
        return "mild"
    if hb >= t["moderate"]:
        return "moderate"
    return "severe"


def confidence(quality: QualityReport, hb: float, group: str, calibrated: bool) -> str:
    margin = abs(hb - thresholds(group)["normal"])
    score = quality.score
    if margin < 0.7:
        score -= 0.25
    if not calibrated:
        score -= 0.15
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def analyze(image_bytes: bytes, roi: dict | None, group: str) -> dict:
    img = load_image(image_bytes)
    roi_img = crop_roi(img, roi)
    roi_rgb = np.asarray(roi_img)
    mask, glare = tissue_mask(roi_rgb)
    quality = check_quality(roi_rgb, mask, glare)
    feats = extract_features(roi_rgb, mask)
    calib = load_calibration()
    hb = estimate_hb(feats, calib)
    calibrated = calib.get("source", "").startswith("default") is False
    risk = classify(hb, group)
    return {
        "quality": quality.__dict__,
        "features": feats,
        "hb_estimate": round(hb, 1),
        "hb_range": [round(max(4.0, hb - 1.2), 1), round(min(18.0, hb + 1.2), 1)],
        "risk": risk,
        "thresholds": thresholds(group),
        "confidence": confidence(quality, hb, group, calibrated),
        "method": "colour-pallor (erythema index)",
        "calibration": calib.get("source", "unknown"),
        "heatmap": make_heatmap(roi_rgb, mask),
        "crop": to_data_url(roi_rgb),
        "_roi_rgb": roi_rgb,  # removed before sending to client
    }
