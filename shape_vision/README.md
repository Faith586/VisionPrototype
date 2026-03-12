# Shape Vision — Real-Time Shape & Character Detection (Web Dashboard)

A real-time vision classification system that detects paper cutout shapes (triangle, circle, rectangle, square, pentagon) and reads a character/letter printed on each shape using a standard USB webcam. **Runs entirely in your web browser.**

## What It Does

Hold a paper shape with a letter written on it in front of your webcam. The system will:
1. **Detect the shape** using OpenCV contour analysis (triangle, circle, rectangle, square, pentagon)
2. **Read the character** printed on it using EasyOCR
3. **Display results** on a live web dashboard with bounding boxes, detection log, and statistics

## Requirements

- Python 3.10 or newer
- A USB webcam (built-in laptop cameras work too)
- A web browser (Chrome, Firefox, Edge — any modern browser)
- Windows 10/11, macOS, or Linux

## Setup Instructions

### Step 1: Open a terminal

On Windows, open **Command Prompt** or **PowerShell** and navigate to the `shape_vision` folder:
```
cd path\to\shape_vision
```

### Step 2: Create a virtual environment

```
python -m venv venv
```

Activate it:
- **Windows (Command Prompt):** `venv\Scripts\activate`
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source venv/bin/activate`

You should see `(venv)` at the beginning of your terminal prompt.

> **No admin access?** The Python installer has an "Install just for me" option that
> doesn't require admin. Alternatively, download [WinPython](https://winpython.github.io)
> (portable, no install needed) and use its built-in terminal.

### Step 3: Install dependencies

```
pip install -r requirements.txt
```

This installs OpenCV, EasyOCR, Flask, and NumPy.

### Step 4: Run the program

```
python main.py
```

You'll see output like:
```
[INFO] Camera opened — Resolution: 1280x720  FPS: 30.0
[INFO] Open your browser to:  http://localhost:5000
```

### Step 5: Open your browser

Go to **http://localhost:5000** — the dashboard will appear with your live camera feed, bounding boxes, and the detection sidebar.

## Usage

- **Hold a paper shape** in front of the camera against a plain background
- **Write a large letter** on each shape with a dark marker for best OCR results
- The sidebar updates live with detections, a log, and statistics

### Command Line Options

```
python main.py --camera 1    # Use a different camera (default is 0)
python main.py --port 8080   # Use a different port (default is 5000)
```

### First Run Note

EasyOCR will download its text recognition model (~100 MB) on the first launch. This only happens once. You'll see a message in the terminal while it downloads.

## Dashboard Layout

| Left Panel (~70%) | Right Panel (~30%) |
|---|---|
| Live camera feed with bounding boxes and labels | Current detection (shape + character + confidence) |
| FPS counter in top-left corner | Detection log (last 20 entries, color-coded) |
| Color-coded by shape type | Statistics (total count, per-shape counts, avg confidence) |

### Shape Colors
- **Triangle:** Cyan
- **Circle:** Green
- **Rectangle:** Purple
- **Square:** Orange
- **Pentagon:** Yellow

## Adding New Shapes

Open `config.py` and add an entry to `SHAPE_CONFIG`:

```python
"Hexagon": {
    "min_vertices": 6,
    "max_vertices": 6,
    "color": (128, 0, 128),       # BGR for OpenCV
    "css_color": "#800080",        # For the web dashboard
},
```

No other code changes needed.

## Tips for Best Results

1. Use a **plain background** — white paper or a solid-color desk
2. Write letters **large and dark** (thick marker on white paper)
3. Ensure **good lighting** — normal office lighting is fine, avoid harsh shadows
4. Hold shapes **fully within the camera frame** — partially visible shapes are ignored
5. Keep shapes **at least a few inches** from the camera so the full shape is visible

## Troubleshooting

| Problem | Solution |
|---|---|
| "Cannot open camera" error | Check that your webcam is connected and not in use by another app (Zoom, Teams, etc.) |
| Browser shows broken image | Make sure the Python server is running in the terminal |
| Shapes not detected | Use a plain background with good contrast |
| OCR not reading characters | Write larger, darker letters; ensure good lighting |
| Slow performance | Close other applications; try reducing the browser tab count |
| EasyOCR download fails | Check your internet connection; the model only needs to download once |

## File Structure

```
shape_vision/
  main.py              — Flask web server with MJPEG streaming and detection API
  detector.py          — Shape detection (OpenCV contour analysis)
  ocr_engine.py        — Character recognition (EasyOCR wrapper)
  detection_log.py     — Detection history and statistics
  config.py            — Shape definitions, colors, thresholds
  requirements.txt     — Python dependencies
  templates/
    index.html         — Web dashboard (HTML/CSS/JS)
  README.md            — This file
```
