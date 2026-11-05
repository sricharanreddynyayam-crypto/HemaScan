"""SQLite storage for screenings, plus demo villages and health centres."""

from __future__ import annotations

import math
import random
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent.parent / "data" / "hemascan.db"

# Demo locations (Warangal district, Telangana). Coordinates are approximate.
# Replace with your own area for a real pilot.
VILLAGES = [
    {"name": "Dharmasagar", "lat": 17.99, "lng": 79.43},
    {"name": "Hasanparthy", "lat": 18.06, "lng": 79.52},
    {"name": "Atmakur", "lat": 18.08, "lng": 79.67},
    {"name": "Geesugonda", "lat": 17.92, "lng": 79.68},
    {"name": "Parkal", "lat": 18.20, "lng": 79.70},
    {"name": "Narsampet", "lat": 17.93, "lng": 79.89},
    {"name": "Wardhannapet", "lat": 17.77, "lng": 79.57},
    {"name": "Raghunathpally", "lat": 17.79, "lng": 79.33},
]

HEALTH_CENTRES = [
    {"name": "PHC Dharmasagar (demo)", "type": "PHC", "lat": 17.995, "lng": 79.435},
    {"name": "PHC Atmakur (demo)", "type": "PHC", "lat": 18.085, "lng": 79.665},
    {"name": "PHC Narsampet (demo)", "type": "PHC", "lat": 17.935, "lng": 79.885},
    {"name": "PHC Wardhannapet (demo)", "type": "PHC", "lat": 17.775, "lng": 79.575},
    {"name": "CHC Parkal (demo)", "type": "CHC", "lat": 18.205, "lng": 79.705},
    {"name": "District Hospital Warangal (demo)", "type": "Hospital", "lat": 17.968, "lng": 79.594},
]


@contextmanager
def connect():
    """Opens the database, commits on success, and always closes the file."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                patient_ref TEXT,
                age INTEGER,
                sex TEXT,
                grp TEXT,
                village TEXT,
                lat REAL,
                lng REAL,
                hb_estimate REAL,
                risk TEXT,
                confidence TEXT,
                erythema_index REAL,
                referred_to TEXT,
                is_demo INTEGER DEFAULT 0
            )"""
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def village_coords(name: str | None) -> tuple[float, float] | None:
    for v in VILLAGES:
        if name and v["name"].lower() == name.lower():
            return v["lat"], v["lng"]
    return None


def referral(risk: str, lat: float | None, lng: float | None) -> dict | None:
    """Nearest suitable centre. Severe cases go to a hospital/CHC."""
    if risk == "none":
        return None
    if lat is None or lng is None:
        lat, lng = 17.968, 79.594
    options = HEALTH_CENTRES
    if risk == "severe":
        options = [c for c in HEALTH_CENTRES if c["type"] in ("Hospital", "CHC")]
    best = min(options, key=lambda c: haversine_km(lat, lng, c["lat"], c["lng"]))
    urgency = {"mild": "within 2 weeks", "moderate": "this week", "severe": "within 1-2 days"}[risk]
    return {
        **best,
        "distance_km": round(haversine_km(lat, lng, best["lat"], best["lng"]), 1),
        "urgency": urgency,
    }


def save_screening(row: dict) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO screenings (created_at, patient_ref, age, sex, grp, village, lat, lng,
               hb_estimate, risk, confidence, erythema_index, referred_to, is_demo)
               VALUES (:created_at, :patient_ref, :age, :sex, :grp, :village, :lat, :lng,
               :hb_estimate, :risk, :confidence, :erythema_index, :referred_to, :is_demo)""",
            {"is_demo": 0, "created_at": datetime.now().isoformat(timespec="seconds"), **row},
        )
        return int(cur.lastrowid)


def list_screenings(limit: int = 200) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM screenings ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def stats() -> dict:
    with connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM screenings").fetchall()]
    total = len(rows)
    by_risk = {k: sum(1 for r in rows if r["risk"] == k) for k in ("none", "mild", "moderate", "severe")}
    villages: dict[str, dict] = {}
    for r in rows:
        name = r["village"] or "Unknown"
        v = villages.setdefault(
            name, {"village": name, "lat": r["lat"], "lng": r["lng"], "total": 0, "anaemic": 0, "severe": 0, "hb_sum": 0.0}
        )
        v["total"] += 1
        v["hb_sum"] += r["hb_estimate"] or 0
        if r["risk"] != "none":
            v["anaemic"] += 1
        if r["risk"] == "severe":
            v["severe"] += 1
    village_list = []
    for v in villages.values():
        v["rate"] = round(100 * v["anaemic"] / v["total"], 1) if v["total"] else 0
        v["avg_hb"] = round(v.pop("hb_sum") / v["total"], 1) if v["total"] else 0
        village_list.append(v)
    village_list.sort(key=lambda v: -v["rate"])
    return {
        "total": total,
        "by_risk": by_risk,
        "anaemia_rate": round(100 * (total - by_risk["none"]) / total, 1) if total else 0,
        "villages": village_list,
        "health_centres": HEALTH_CENTRES,
    }


def seed_demo(n: int = 160) -> int:
    """Adds clearly-marked fake screenings so the dashboard has something to show."""
    from analysis import classify  # local import to avoid a cycle

    rng = random.Random(42)
    # Some villages get lower average Hb so hotspots show up on the map.
    village_mean = {v["name"]: rng.uniform(10.2, 13.2) for v in VILLAGES}
    now = datetime.now()
    rows = []
    for _ in range(n):
        v = rng.choice(VILLAGES)
        grp = rng.choices(["woman", "pregnant", "child_6_23m", "child_24_59m", "man", "child_5_11y"], [45, 12, 8, 10, 15, 10])[0]
        hb = round(max(5.5, min(16.5, rng.gauss(village_mean[v["name"]] - (0.6 if grp == "pregnant" else 0), 1.5))), 1)
        risk = classify(hb, grp)
        ref = referral(risk, v["lat"], v["lng"])
        rows.append(
            {
                "created_at": (now - timedelta(days=rng.randint(0, 60), hours=rng.randint(0, 23))).isoformat(timespec="seconds"),
                "patient_ref": f"DEMO-{rng.randint(1000, 9999)}",
                "age": rng.randint(18, 45) if grp in ("woman", "pregnant", "man") else rng.randint(1, 11),
                "sex": "F" if grp in ("woman", "pregnant") else rng.choice(["M", "F"]) if grp.startswith("child") else "M",
                "grp": grp,
                "village": v["name"],
                "lat": v["lat"] + rng.uniform(-0.01, 0.01),
                "lng": v["lng"] + rng.uniform(-0.01, 0.01),
                "hb_estimate": hb,
                "risk": risk,
                "confidence": rng.choice(["high", "medium"]),
                "erythema_index": None,
                "referred_to": ref["name"] if ref else None,
                "is_demo": 1,
            }
        )
    with connect() as conn:
        conn.executemany(
            """INSERT INTO screenings (created_at, patient_ref, age, sex, grp, village, lat, lng,
               hb_estimate, risk, confidence, erythema_index, referred_to, is_demo)
               VALUES (:created_at, :patient_ref, :age, :sex, :grp, :village, :lat, :lng,
               :hb_estimate, :risk, :confidence, :erythema_index, :referred_to, :is_demo)""",
            rows,
        )
    return n


def clear_demo() -> int:
    with connect() as conn:
        return conn.execute("DELETE FROM screenings WHERE is_demo = 1").rowcount
