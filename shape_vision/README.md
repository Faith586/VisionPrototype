# Shape Vision — Real-Time Shape & Character Detection

A real-time vision classification system that detects paper cutout shapes (triangle, circle, rectangle, square, pentagon) and reads a character/letter printed on each shape using a standard USB webcam.

## What It Does

Hold a paper shape with a letter written on it in front of your webcam. The system will:
1. **Detect the shape** using OpenCV contour analysis (triangle, circle, rectangle, square, pentagon)
2. **Read the character** printed on it using EasyOCR
3. **Display results** on a live dashboard with bounding boxes, labels, and a detection log

## Requirements

- Python 3.10 or newer
- A USB webcam (built-in laptop cameras work too)
- Windows 10/11, macOS, or Linux

## Setup Instructions

### Step 1: Clone or download this project

Place the `shape_vision` folder somewhere on your computer.

### Step 2: Open a terminal

On Windows, open **Command Prompt** or **PowerShell** and navigate to the `shape_vision` folder:
```
cd path\to\shape_vision
```

### Step 3: Create a virtual environment

```
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

You should see `(venv)` at the beginning of your terminal prompt.

### Step 4: Install dependencies

```
pip install -r requirements.txt
```

This installs OpenCV, EasyOCR, and NumPy. The first time you run the program, EasyOCR will download its text recognition model (~100 MB). This only happens once.

### Step 5: Run the program

```
python main.py
```

A window titled **"Shape Vision Dashboard"** will appear showing your webcam feed.

## Usage

- **Hold a paper shape** in front of the camera against a plain background (white table or dark desk works best)
- **Write a large letter** on each shape with a dark marker for best OCR results
- Press **'s'** to save a screenshot
- Press **'q'** to quit

### Command Line Options

```
python main.py --camera 1    # Use a different camera (default is 0)
```

## Dashboard Layout

| Left Panel (70%) | Right Panel (30%) |
|---|---|
| Live camera feed with bounding boxes and labels | Current detection (shape + character + confidence) |
| FPS counter in top-left | Detection log (last 20 entries) |
| Color-coded by shape type | Statistics (counts, average confidence) |

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
    "color": (128, 0, 128),
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
| Shapes not detected | Make sure the shape is against a plain background with good contrast |
| OCR not reading characters | Write larger, darker letters; ensure good lighting |
| Low FPS | Close other applications; the system is optimized for CPU but heavy background processes can slow it down |
| EasyOCR download fails | Check your internet connection; the model only needs to download once |

## File Structure

```
shape_vision/
  main.py           — Entry point, starts camera and dashboard loop
  detector.py       — Shape detection (OpenCV contour analysis)
  ocr_engine.py     — Character recognition (EasyOCR wrapper)
  dashboard.py      — UI rendering (live feed + sidebar)
  config.py         — Shape definitions, colors, thresholds
  detection_log.py  — Detection history and statistics
  requirements.txt  — Python dependencies
  README.md         — This file
```
