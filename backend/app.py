"""
HemaScan API server.

Run from the hemascan folder:
    python -m uvicorn app:app --app-dir backend --reload --port 8000
Then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import analysis
import db
import explain
import model_cnn

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MAX_UPLOAD = 12 * 1024 * 1024

app = FastAPI(title="HemaScan", version="1.0")


@app.get("/api/health")
def health():
    calib = analysis.load_calibration()
    return {
        "status": "ok",
        "cnn_model": model_cnn.available(),
        "calibration": calib.get("source"),
        "llm": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    roi: str = Form(""),
    group: str = Form("woman"),
    age: str = Form(""),
    sex: str = Form(""),
    village: str = Form(""),
    lat: str = Form(""),
    lng: str = Form(""),
    lang: str = Form("en"),
    patient_ref: str = Form(""),
    save: str = Form("true"),
):
    data = await image.read()
    if not data:
        raise HTTPException(400, "No image uploaded.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is too large (max 12 MB).")
    try:
        roi_box = json.loads(roi) if roi else None
    except json.JSONDecodeError:
        roi_box = None

    try:
        result = analysis.analyze(data, roi_box, group)
    except Exception as exc:
        raise HTTPException(400, f"Could not read this image: {exc}")

    roi_rgb = result.pop("_roi_rgb")

    # Optional CNN: blend with the colour method and use Grad-CAM for the heatmap.
    if model_cnn.available():
        try:
            hb_cnn, cam = model_cnn.predict_with_gradcam(roi_rgb)
            hb = round(0.7 * hb_cnn + 0.3 * result["hb_estimate"], 1)
            result.update(
                hb_estimate=hb,
                hb_range=[round(max(4.0, hb - 1.0), 1), round(min(18.0, hb + 1.0), 1)],
                risk=analysis.classify(hb, group),
                method="CNN (MobileNetV3) + colour-pallor",
                hb_cnn=round(hb_cnn, 1),
            )
            colors = analysis._colormap(cam)
            alpha = (cam * 0.6)[..., None]
            blended = roi_rgb.astype(np.float32) * (1 - alpha) + colors * alpha
            result["heatmap"] = analysis.to_data_url(blended.astype(np.uint8))
        except Exception as exc:
            print(f"[cnn] prediction failed, using colour method: {exc}")

    age_i = int(age) if age.strip().isdigit() else None
    lat_f = _float(lat)
    lng_f = _float(lng)
    if (lat_f is None or lng_f is None) and village:
        coords = db.village_coords(village)
        if coords:
            lat_f, lng_f = coords

    result["explanation"] = explain.explain(result["risk"], result["hb_estimate"], lang, age_i, group)
    result["referral"] = db.referral(result["risk"], lat_f, lng_f)

    if save == "true" and result["quality"]["ok"]:
        result["screening_id"] = db.save_screening(
            {
                "patient_ref": patient_ref[:40] or None,
                "age": age_i,
                "sex": sex[:1] or None,
                "grp": group,
                "village": village or None,
                "lat": lat_f,
                "lng": lng_f,
                "hb_estimate": result["hb_estimate"],
                "risk": result["risk"],
                "confidence": result["confidence"],
                "erythema_index": result["features"]["erythema_index"],
                "referred_to": result["referral"]["name"] if result["referral"] else None,
            }
        )
    return result


def _float(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


@app.get("/api/screenings")
def screenings(limit: int = 100):
    return db.list_screenings(limit)


@app.get("/api/stats")
def stats():
    return db.stats()


@app.get("/api/villages")
def villages():
    return db.VILLAGES


@app.post("/api/demo/seed")
def seed():
    return {"added": db.seed_demo()}


@app.delete("/api/demo")
def clear():
    return {"removed": db.clear_demo()}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="static")
