"""Generates synthetic eyelid illustrations for testing the app flow.
These are NOT real medical images; use real photos for your demo."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "frontend" / "samples"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = {"healthy": (205, 92, 100), "mild": (214, 124, 124), "pale": (226, 168, 165)}

def make(name, color, seed):
    rng = np.random.default_rng(seed)
    W, H = 800, 600
    img = Image.new("RGB", (W, H), (196, 150, 122))
    d = ImageDraw.Draw(img)
    d.ellipse([140, 120, 660, 360], fill=(240, 236, 230))            # sclera
    d.ellipse([320, 150, 480, 310], fill=(92, 60, 40))               # iris
    d.ellipse([370, 200, 430, 260], fill=(15, 12, 10))               # pupil
    d.ellipse([388, 212, 404, 228], fill=(255, 255, 255))            # catchlight
    d.chord([180, 230, 620, 450], 0, 180, fill=color)                 # conjunctiva
    for _ in range(14):                                                # small vessels
        x = rng.integers(220, 580); y = rng.integers(345, 420)
        dark = tuple(int(c * 0.8) for c in color)
        d.line([(x, y), (x + rng.integers(-60, 60), y + rng.integers(-15, 15))], fill=dark, width=2)
    arr = np.asarray(img.filter(ImageFilter.GaussianBlur(1.2))).astype(np.int16)
    arr = np.clip(arr + rng.normal(0, 4, arr.shape), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(OUT / f"sample_{name}.jpg", quality=92)

for i, (n, c) in enumerate(SAMPLES.items()):
    make(n, c, i)
print("samples written to", OUT)
