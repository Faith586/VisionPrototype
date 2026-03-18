"""Configuration for the YOLO-based shape + character tag detection system."""

import os

# ---------------------------------------------------------------------------
# Tag characters and their appearance
# 3D-printed black & white tags: white shape with black letter on it.
# The model distinguishes identical shapes purely by the letter printed on them.
# ---------------------------------------------------------------------------
TAG_LETTERS = ["X", "Y", "Z"]

# ---------------------------------------------------------------------------
# Base shapes (only the ones we're printing)
# ---------------------------------------------------------------------------
SHAPES = ["Triangle", "Square", "Circle"]

# ---------------------------------------------------------------------------
# Compound classes — order matters (index = class id in YOLO labels)
# Format: "Shape-Letter"  e.g. "Triangle-X", "Square-Y", "Circle-Z"
# ---------------------------------------------------------------------------
CLASSES = []
for _shape in SHAPES:
    for _letter in TAG_LETTERS:
        CLASSES.append(f"{_shape}-{_letter}")
# Result: ["Triangle-X", "Triangle-Y", "Triangle-Z",
#          "Square-X",   "Square-Y",   "Square-Z",
#          "Circle-X",   "Circle-Y",   "Circle-Z"]

# Colors for bounding boxes on the dashboard (just for visual distinction)
_SHAPE_BASE_BGR = {
    "Triangle":  (255, 255, 0),    # cyan
    "Square":    (0, 165, 255),    # orange
    "Circle":    (0, 255, 0),      # green
}

_SHAPE_BASE_CSS = {
    "Triangle":  "#00FFFF",
    "Square":    "#FFA500",
    "Circle":    "#00FF00",
}

# Per-letter accent colors for dashboard badges
TAG_COLORS_CSS = {
    "X": "#FF6B6B",
    "Y": "#51CF66",
    "Z": "#339AF0",
}

CLASS_COLORS = {}
CLASS_COLORS_CSS = {}
for _cls in CLASSES:
    _shape = _cls.rsplit("-", 1)[0]
    CLASS_COLORS[_cls] = _SHAPE_BASE_BGR[_shape]
    CLASS_COLORS_CSS[_cls] = _SHAPE_BASE_CSS[_shape]

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
SYNTH_IMG_SIZE = 640
SYNTH_TRAIN_COUNT = 2000           # 9 classes, plenty of data
SYNTH_VAL_COUNT = 400
SYNTH_MIN_SHAPES = 1
SYNTH_MAX_SHAPES = 4
SYNTH_MIN_SIZE = 60
SYNTH_MAX_SIZE = 200

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
YOLO_MODEL_BASE = "yolov8n.pt"         # nano model for fast training
TRAIN_EPOCHS = 80
TRAIN_BATCH = 16
TRAIN_IMG_SIZE = 640

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MAX_LOG_ENTRIES = 20
DETECTION_COOLDOWN = 2.0   # seconds
CENTROID_SHIFT_THRESHOLD = 50  # pixels

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "shapes_dataset")
BEST_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
