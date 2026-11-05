"""
Automated checks for HemaScan.   Run:  python -m pytest -q
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import analysis  # noqa: E402
import db  # noqa: E402
import explain  # noqa: E402

SAMPLES = ROOT / "frontend" / "samples"
SAMPLE_ROI = {"x": 0.3, "y": 0.58, "w": 0.4, "h": 0.14}


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# ---------- WHO 2024 cut-offs ----------
@pytest.mark.parametrize(
    "group,hb,expected",
    [
        ("woman", 12.0, "none"), ("woman", 11.9, "mild"), ("woman", 11.0, "mild"),
        ("woman", 10.9, "moderate"), ("woman", 8.0, "moderate"), ("woman", 7.9, "severe"),
        ("pregnant", 11.0, "none"), ("pregnant", 10.9, "mild"), ("pregnant", 9.9, "moderate"), ("pregnant", 6.9, "severe"),
        ("man", 13.0, "none"), ("man", 12.9, "mild"), ("man", 10.9, "moderate"), ("man", 7.9, "severe"),
        ("child_6_23m", 10.5, "none"), ("child_6_23m", 10.4, "mild"), ("child_6_23m", 9.4, "moderate"), ("child_6_23m", 6.9, "severe"),
        ("child_24_59m", 11.0, "none"), ("child_24_59m", 10.9, "mild"), ("child_24_59m", 9.9, "moderate"), ("child_24_59m", 6.9, "severe"),
        ("child_5_11y", 11.5, "none"), ("child_5_11y", 11.4, "mild"), ("child_5_11y", 10.9, "moderate"), ("child_5_11y", 7.9, "severe"),
        ("child_12_14y", 12.0, "none"), ("child_12_14y", 11.9, "mild"), ("child_12_14y", 10.9, "moderate"), ("child_12_14y", 7.9, "severe"),
    ],
)
def test_who_classification(group, hb, expected):
    assert analysis.classify(hb, group) == expected


# ---------- Image analysis ----------
def test_samples_ranked_by_pallor():
    hb = {}
    for name in ("healthy", "mild", "pale"):
        r = analysis.analyze((SAMPLES / f"sample_{name}.jpg").read_bytes(), SAMPLE_ROI, "woman")
        assert r["quality"]["ok"], r["quality"]["issues"]
        hb[name] = r["hb_estimate"]
    assert hb["healthy"] > hb["mild"] > hb["pale"]
    assert analysis.classify(hb["healthy"], "woman") == "none"
    assert analysis.classify(hb["pale"], "woman") in ("moderate", "severe")


def test_redder_tissue_gives_higher_hb():
    def hb_for(rgb):
        arr = np.full((120, 160, 3), rgb, dtype=np.uint8)
        arr = arr + np.random.default_rng(0).integers(-15, 15, arr.shape)  # texture so blur check passes
        return analysis.analyze(png_bytes(np.clip(arr, 0, 255)), {"x": 0, "y": 0, "w": 1, "h": 1}, "woman")["hb_estimate"]
    assert hb_for((210, 90, 95)) > hb_for((215, 130, 130)) > hb_for((225, 170, 168))


def test_dark_photo_rejected():
    r = analysis.analyze(png_bytes(np.full((300, 400, 3), 15)), None, "woman")
    assert not r["quality"]["ok"]
    assert any("dark" in i for i in r["quality"]["issues"])


@pytest.mark.parametrize("base", [(128, 128, 128), (196, 150, 122), (141, 85, 50)])  # grey wall, light skin, dark skin
def test_photo_without_eyelid_rejected(base):
    rng = np.random.default_rng(1)
    texture = rng.normal(0, 18, (300, 400, 1))  # brightness texture, like a real surface
    img = np.clip(np.array(base, dtype=np.float32) + texture, 0, 255)
    r = analysis.analyze(png_bytes(img), None, "woman")
    assert not r["quality"]["ok"]
    assert r["quality"]["tissue_fraction"] < 0.08


def test_lab_conversion_reference_values():
    lab = analysis.rgb_to_lab(np.array([[255, 255, 255], [255, 0, 0]], dtype=np.uint8))
    assert lab[0, 0] == pytest.approx(100, abs=0.5) and abs(lab[0, 1]) < 0.5
    assert lab[1, 0] == pytest.approx(53.2, abs=0.5) and lab[1, 1] == pytest.approx(80.1, abs=1.0)


# ---------- Referral, explanations ----------
def test_referral_rules():
    assert db.referral("none", 18.0, 79.5) is None
    severe = db.referral("severe", 17.99, 79.43)
    assert severe["type"] in ("Hospital", "CHC")
    mild = db.referral("mild", 17.99, 79.43)
    assert mild["name"].startswith("PHC Dharmasagar")
    assert mild["distance_km"] < 2


def test_haversine_known_distance():
    # Hyderabad -> Warangal is about 140 km in a straight line
    assert db.haversine_km(17.385, 78.4867, 17.9689, 79.5941) == pytest.approx(137, abs=10)


@pytest.mark.parametrize("lang", ["en", "hi", "te"])
@pytest.mark.parametrize("risk", ["none", "mild", "moderate", "severe"])
def test_templates_complete(lang, risk, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = explain.explain(risk, 10.2, lang)
    assert out["source"] == "template"
    assert "10.2" in out["text"] and out["disclaimer"]


# ---------- API ----------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(db, "DB_FILE", tmp_path / "test.db")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import app

    return TestClient(app.app)


def test_api_full_flow(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200
    assert client.get("/sw.js").status_code == 200

    img = (SAMPLES / "sample_pale.jpg").read_bytes()
    r = client.post(
        "/api/analyze",
        files={"image": ("e.jpg", img, "image/jpeg")},
        data={"roi": json.dumps(SAMPLE_ROI), "group": "woman", "village": "Parkal", "lang": "te"},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["risk"] in ("moderate", "severe")
    assert d["referral"] and d["screening_id"]
    assert d["heatmap"].startswith("data:image/png;base64,")
    assert "_roi_rgb" not in d

    rows = client.get("/api/screenings").json()
    assert len(rows) == 1 and rows[0]["village"] == "Parkal"

    assert client.post("/api/demo/seed").json()["added"] > 0
    s = client.get("/api/stats").json()
    assert s["total"] > 100 and s["villages"]
    client.delete("/api/demo")
    assert client.get("/api/stats").json()["total"] == 1  # real screening kept


def test_api_rejects_bad_input(client):
    r = client.post("/api/analyze", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    dark = png_bytes(np.full((200, 200, 3), 10))
    r = client.post("/api/analyze", files={"image": ("d.png", dark, "image/png")})
    assert r.status_code == 200 and "screening_id" not in r.json()  # bad photos are not saved
