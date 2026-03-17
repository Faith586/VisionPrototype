# Shape Vision — Real-Time Shape & Character Detection

A real-time vision system that detects paper cutout shapes and reads characters printed on them.

## Three Versions Available

| Version | Folder | Requires | Best for |
|---------|--------|----------|----------|
| **Browser** | `docs/` | Just a browser | Locked-down laptops, quick demos |
| **OpenCV** | `shape_vision/` | Python + OpenCV | Classical CV, no training needed |
| **YOLO** | `yolo_vision/` | Python + Ultralytics | Highest accuracy, custom trained model |

---

## YOLO Version (Recommended)

Uses a custom-trained YOLOv8 model for shape detection + EasyOCR for character recognition.

### Setup (macOS)

```bash
cd yolo_vision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Generate Training Data

```bash
python scripts/generate_data.py
```

Creates 2,000 training + 400 validation synthetic images with randomly placed shapes and characters.

### Step 2: Train the Model

```bash
python scripts/train.py
```

Trains YOLOv8-nano for 80 epochs. Takes ~10-30 minutes depending on your Mac. Results saved to `runs/detect/train/`.

### Step 3: Run the Dashboard

```bash
python main.py
```

Open http://127.0.0.1:5000 — live webcam feed with YOLO-powered detection.

### Optional: Collect Real Data

To improve accuracy with real-world images:

```bash
python scripts/collect_data.py
```

Use your webcam to capture and label real shape images (keys 1-5 to select class, SPACE to capture, draw a box, press 's' to save). Then retrain with `python scripts/train.py`.

---

## Browser Version (No Install)

Open `docs/index.html` in Chrome or Edge. Everything runs client-side via OpenCV.js and Tesseract.js.

```bash
cd docs
python3 -m http.server 8000
# Open http://localhost:8000
```

---

## OpenCV Version (Classical CV)

Uses contour analysis instead of machine learning. No training required.

```bash
cd shape_vision
pip install -r requirements.txt
python main.py
```

---

## Detected Shapes

- Triangle (cyan)
- Square (orange)
- Rectangle (magenta)
- Circle (green)
- Pentagon (yellow)

Each shape can contain any letter (A-Z) or digit (0-9).
