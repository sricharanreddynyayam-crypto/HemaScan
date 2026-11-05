# HemaScan: needle-free anaemia screening from a phone photo

**DSH Hacks V2 · AI x Healthcare**

About 1 in 4 people in the world have anaemia, and more than half of women in India do (NFHS-5). Most never get tested, because the test needs a needle, a lab and money.

HemaScan lets a community health worker screen someone in about 30 seconds. They photograph the inside of the lower eyelid with an ordinary phone. The AI measures how pale the eyelid is, estimates haemoglobin, explains the result aloud in the patient's language, and refers high-risk patients to the nearest health centre. A dashboard shows which villages are hotspots.

## Features

| Feature | What it does |
|---|---|
| 📷 Guided capture | Step-by-step photo guide. You drag a box over the inner eyelid. |
| ✅ Photo quality check | Detects blur, darkness, glare and a missing eyelid, and tells the user how to fix it. Bad photos are not saved. |
| 🧠 AI pallor analysis | Tissue mask, then erythema index log₁₀(R/G) and CIELAB a*, then a calibrated haemoglobin estimate. Optional MobileNetV3 CNN. |
| 🔥 Explainable heatmap | Shows which parts of the eyelid look pale (Grad-CAM when the CNN is loaded). |
| 🩺 WHO 2024 cut-offs | Separate thresholds for women, pregnant women, men and four child age groups. Grades mild, moderate and severe. |
| 🗣️ Multilingual voice | Explanation in English, Hindi or Telugu, read aloud. Written by Claude if an API key is set, otherwise built-in templates. |
| 🏥 Smart referral | Nearest health centre with distance and urgency. Severe cases go to a hospital or CHC. |
| 🗺️ Hotspot dashboard | Map of anaemia rate by village, risk breakdown and recent screenings. |
| 📱 Mobile-first web app | Works in any phone browser and can be installed as a PWA. No app store needed. |

## Run it (Windows, macOS, Linux)

Requires Python 3.10+.

```bash
cd hemascan
pip install -r requirements.txt
python -m uvicorn app:app --app-dir backend --port 8000
```

Open **http://localhost:8000**.

- **Screen tab:** fill in the patient, take or upload a photo (or click *Try a sample image*), then click Analyse.
- **Dashboard tab:** click **Load demo data** to fill the map with sample screenings.

**Using it on your phone:** connect the phone and laptop to the same Wi-Fi. Run the server with `--host 0.0.0.0`, then open `http://<laptop-IP>:8000` on the phone. Phone browsers only allow the camera on `https` or `localhost`. Over plain Wi-Fi, *upload* a photo taken with the camera app instead.

**Optional AI explanations with Claude:**

```bash
set ANTHROPIC_API_KEY=your-key        # Windows (cmd)
export ANTHROPIC_API_KEY=your-key     # macOS / Linux
```

Without a key, the app uses its built-in templates.

## Make it accurate with real data (important for judging)

The app starts with an **uncalibrated** default formula. Calibrate it on real photos:

1. Download the **Eyes Defy Anemia** dataset from Kaggle: real eyelid photos with lab Hb values, from India and Italy.
2. Build the label file:
   ```bash
   pip install pandas openpyxl
   python train/prepare_dataset.py "C:/path/to/eyes-defy-anemia"
   ```
3. Fit the colour model and get accuracy numbers (5-fold cross-validation):
   ```bash
   python train/calibrate.py
   ```
   This writes `model/calibration.json` (the app uses it automatically after a restart) and `model/metrics.json`, which holds the **MAE, sensitivity and specificity** for your write-up.
4. *(Optional, more technical depth)* Train the CNN. Google Colab's free GPU is easiest:
   ```bash
   pip install torch torchvision
   python train/train_cnn.py --epochs 40
   ```
   Put `model/hemascan_cnn.pt` in `model/` and restart the app. The app then blends CNN and colour predictions and shows a Grad-CAM heatmap.

Report your real numbers honestly, even if modest. Judges trust honest numbers more than perfect ones.

5. Re-generate the one-page submission PDF. It picks up `model/metrics.json` automatically:
   ```bash
   python tools/make_description_pdf.py
   ```

## Tests

```bash
pip install pytest httpx
python -m pytest -q
```

The 53 tests cover the WHO 2024 cut-offs for every group, colour maths, rejection of bad photos (dark, plain skin, no eyelid), referral rules, all explanation templates, and the full API flow.

## How it works

```
Photo ─► Quality check ─► Eyelid crop ─► Tissue mask ─► Pallor features ─► Hb estimate ─► WHO risk grade
  (blur, light, glare)      (user box)   (drop glare,    (erythema index,   (calibrated      │
                                          lashes, skin)    CIELAB a*)        regression      ├─► Heatmap
                                                                             ± CNN)          ├─► Voice explanation (EN/HI/TE)
                                                                                             ├─► Referral to nearest centre
                                                                                             └─► Saved to village dashboard
```

**Stack:** Python · FastAPI · NumPy · Pillow · SQLite · optional PyTorch (MobileNetV3) · Claude API · vanilla JS · Leaflet / OpenStreetMap · Web Speech API.

## Project layout

```
backend/   app.py (API) · analysis.py (image AI) · explain.py (language) · db.py (storage, referral) · model_cnn.py (optional CNN)
frontend/  index.html · app.js · style.css · samples/ (synthetic test images)
train/     prepare_dataset.py · calibrate.py · train_cnn.py
model/     calibration.json, metrics.json, hemascan_cnn.pt (created by training)
data/      hemascan.db (created on first run)
```

## Limitations and safety

- **Screening, not diagnosis.** Positive results must be confirmed with a blood test.
- WHO cut-offs are used at sea level, without the altitude or smoking adjustments. For pregnancy the app uses the 11.0 g/dL cut-off (1st and 3rd trimester); WHO uses 10.5 in the 2nd trimester.
- Colour depends on lighting and camera. Always use the flash. A colour-reference card is a planned improvement.
- The sample images in `frontend/samples/` are **synthetic illustrations** for testing the flow. Use real photos in your demo.
- Village and health-centre locations are demo data (Warangal district). Replace them in `backend/db.py` for a real pilot.
- Patient data stays on the server's local SQLite file. Only an optional ID is stored, not names.

## Next steps

Offline mode (run the model in the browser), a colour-calibration card, a fingernail-bed photo as a second signal, SMS referrals, and a pilot with ASHA workers.
