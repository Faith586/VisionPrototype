# Shape Vision — Real-Time Shape & Character Detection

A real-time vision system that detects paper cutout shapes and reads characters printed on them.

## Two Versions Available

### Browser Version (No Install Required)
Open `docs/index.html` in Chrome or Edge — that's it. No Python, no downloads, no admin access needed.

Everything runs in your browser:
- **Webcam** via browser's getUserMedia API
- **Shape detection** via OpenCV.js
- **Character recognition** via Tesseract.js

### Python Version (Desktop)
If you have Python installed, the `shape_vision/` folder contains a Flask-based version with server-side processing. See `shape_vision/README.md` for setup.

---

## Quick Start (Browser Version)

1. Open `docs/index.html` in **Chrome** or **Edge**
2. Allow camera access when prompted
3. Hold paper shapes with letters in front of your webcam
4. The dashboard shows live detection with bounding boxes and a sidebar

### Or deploy as a GitHub Pages site:
Enable GitHub Pages on the `docs/` folder in your repo settings, then share the URL with anyone.

## Features

- Detects **Triangle, Circle, Rectangle, Square, Pentagon**
- Reads **any letter or number** on the shape
- Live dashboard with bounding boxes, detection log, and statistics
- Color-coded by shape type
- FPS counter and screenshot button
- Works on any laptop with a webcam and a modern browser
