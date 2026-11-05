"""
Builds the one-page project description PDF for the submission.

    python tools/make_description_pdf.py

If model/metrics.json exists (from train/calibrate.py), your real accuracy
numbers are included automatically. Re-run after training.
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "HemaScan_Project_Description.pdf"
RED = colors.HexColor("#b91c3c")
INK = colors.HexColor("#1f1720")
MUTED = colors.HexColor("#6b5d66")

h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, textColor=RED, leading=24)
sub = ParagraphStyle("sub", fontName="Helvetica", fontSize=10.5, textColor=MUTED, leading=14)
h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, textColor=RED, leading=14, spaceBefore=7, spaceAfter=3)
body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, textColor=INK, leading=12.4)
cell = ParagraphStyle("cell", parent=body, fontSize=8.8, leading=11.5)
small = ParagraphStyle("small", parent=body, fontSize=7.8, textColor=MUTED, leading=10)

metrics_file = ROOT / "model" / "metrics.json"
if metrics_file.exists():
    m = json.loads(metrics_file.read_text())
    results = (
        f"On <b>{m['n_images']}</b> real eyelid photos with lab haemoglobin (Eyes Defy Anemia dataset, "
        f"{m['cross_validation_folds']}-fold cross-validation): mean absolute error <b>{m['mae_g_dl']} g/dL</b>, "
        f"correlation r = <b>{m['pearson_r']}</b>, sensitivity <b>{round(100 * (m['sensitivity'] or 0))}%</b> and "
        f"specificity <b>{round(100 * (m['specificity'] or 0))}%</b> for anaemia (Hb &lt; {m['threshold_g_dl']} g/dL)."
    )
else:
    results = (
        "Calibration and validation pipeline included (train/calibrate.py): k-fold cross-validated MAE, "
        "sensitivity and specificity on the public Eyes Defy Anemia dataset (real eyelid photos with lab Hb). "
        "<i>Run it and re-generate this PDF to insert the measured numbers.</i>"
    )

story = [
    Paragraph("HemaScan", h1),
    Paragraph("Needle-free anaemia screening from a phone photo of the eyelid · DSH Hacks V2 · AI x Healthcare", sub),
    Spacer(1, 4 * mm),
    Paragraph("The problem", h2),
    Paragraph(
        "Anaemia affects roughly <b>1 in 4 people worldwide</b> and more than <b>half of women in India</b> (NFHS-5). "
        "It causes fatigue, poor school performance, and dangerous pregnancies, yet most cases are never diagnosed. "
        "The standard test needs a needle, a lab, trained staff and money, which rural communities often lack. "
        "Community health workers (such as India's ASHA workers) meet these patients every day but have no tool to check them.",
        body,
    ),
    Paragraph("Our solution", h2),
    Paragraph(
        "HemaScan turns any smartphone into an anaemia screener. A health worker photographs the inner lower eyelid "
        "(palpebral conjunctiva). Pale conjunctiva is a well-known clinical sign of low haemoglobin. The AI measures "
        "that pallor, estimates haemoglobin, grades the risk using <b>WHO cut-offs</b>, explains the result aloud in the "
        "patient's own language, and refers high-risk patients to the nearest health centre. Every screening feeds a "
        "village-level hotspot map, so officials can see where to send iron supplements and follow-up camps.",
        body,
    ),
    Paragraph("Key features", h2),
]

features = [
    ("Photo quality check", "Detects blur, darkness, glare and a missing eyelid; tells the user how to fix it."),
    ("AI pallor analysis", "Tissue mask, erythema index log10(R/G) and CIELAB a*, calibrated Hb regression; optional fine-tuned MobileNetV3 CNN."),
    ("Explainable AI", "Heatmap of pale regions (Grad-CAM with the CNN), so the result is not a black box."),
    ("Clinical logic", "WHO 2024 haemoglobin thresholds for women, pregnancy, men and four child age bands; mild / moderate / severe grading."),
    ("Multilingual voice", "English, Hindi and Telugu explanations read aloud for low-literacy users (Claude API, offline templates as fallback)."),
    ("Smart referral", "Nearest PHC / hospital with distance and urgency; severe cases routed to a hospital."),
    ("Hotspot dashboard", "Map of anaemia rate by village, risk breakdown and recent screenings."),
]
t = Table(
    [[Paragraph(f"<b>{a}</b>", cell), Paragraph(b, cell)] for a, b in features],
    colWidths=[38 * mm, 136 * mm],
)
t.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#fdf2f4"), colors.white]),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story += [
    t,
    Paragraph("How it works", h2),
    Paragraph(
        "Photo &rarr; quality check &rarr; eyelid crop &rarr; tissue mask (removes glare, lashes, skin) &rarr; pallor features "
        "&rarr; haemoglobin estimate &rarr; WHO risk grade &rarr; heatmap + voice explanation + referral &rarr; dashboard. "
        "<b>Stack:</b> Python, FastAPI, NumPy, Pillow, SQLite, optional PyTorch (MobileNetV3), Claude API, JavaScript, Leaflet/OpenStreetMap, Web Speech API. "
        "Runs in any phone browser and installs as a web app (PWA), so no app store is needed.",
        body,
    ),
    Paragraph("Results", h2),
    Paragraph(results, body),
    Paragraph("Impact and next steps", h2),
    Paragraph(
        "A test that costs nothing per use and takes about 30 seconds lets health workers screen whole villages, catch severe anaemia in "
        "pregnancy early, and target supplements where they are needed most. Next: fully offline mode (on-device "
        "model), a printed colour-calibration card for consistent lighting, fingernail-bed photos as a second signal, SMS "
        "referrals, and a pilot with ASHA workers.",
        body,
    ),
    Spacer(1, 3 * mm),
    Paragraph(
        "HemaScan is a screening aid, not a diagnostic device. Positive results should always be confirmed with a blood test.",
        small,
    ),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm, bottomMargin=12 * mm,
                        title="HemaScan - Project Description")
doc.build(story)
print("wrote", OUT)
