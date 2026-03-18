"""Configuration for the YOLO-based shape + character tag detection system."""

import os

# ---------------------------------------------------------------------------
# Tag characters and their associated tag colors (BGR for OpenCV)
# Each 3D-printed tag has a shape, a color, and a letter.
# The model learns to distinguish identical shapes by letter AND color.
# ---------------------------------------------------------------------------
TAG_LETTERS = ["X", "Y", "Z"]

TAG_COLORS_BGR = {
    "X": (60, 60, 220),    # red tag
    "Y": (60, 180, 60),    # green tag
    "Z": (220, 140, 50),   # blue tag
}

TAG_COLORS_CSS = {
    "X": "#DC3C3C",
    "Y": "#3CB43C",
    "Z": "#328CDC",
}

# ---------------------------------------------------------------------------
# Base shapes
# ---------------------------------------------------------------------------
SHAPES = ["Triangle", "Square", "Rectangle", "Circle", "Pentagon"]

# ---------------------------------------------------------------------------
# Compound classes — order matters (index = class id in YOLO labels)
# Format: "Shape-Letter"  e.g. "Triangle-X", "Square-Y", "Circle-Z"
# ---------------------------------------------------------------------------
CLASSES = []
for _shape in SHAPES:
    for _letter in TAG_LETTERS:
        CLASSES.append(f"{_shape}-{_letter}")
# Result: ["Triangle-X", "Triangle-Y", "Triangle-Z", "Square-X", ..., "Pentagon-Z"]

# Colors for bounding boxes — based on shape, tinted by letter
_SHAPE_BASE_BGR = {
    "Triangle":  (255, 255, 0),    # cyan
    "Square":    (0, 165, 255),    # orange
    "Rectangle": (255, 0, 255),    # magenta
    "Circle":    (0, 255, 0),      # green
    "Pentagon":  (0, 255, 255),    # yellow
}

_SHAPE_BASE_CSS = {
    "Triangle":  "#00FFFF",
    "Square":    "#FFA500",
    "Rectangle": "#FF00FF",
    "Circle":    "#00FF00",
    "Pentagon":  "#FFFF00",
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
SYNTH_TRAIN_COUNT = 3000           # more images for 15 classes
SYNTH_VAL_COUNT = 600
SYNTH_MIN_SHAPES = 1
SYNTH_MAX_SHAPES = 4
SYNTH_MIN_SIZE = 60
SYNTH_MAX_SIZE = 200

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
YOLO_MODEL_BASE = "yolov8n.pt"         # nano model for fast training
TRAIN_EPOCHS = 100                      # more epochs for 15 classes
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
