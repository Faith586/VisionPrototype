"""Configuration for the YOLO-based shape detection system."""

import os

# ---------------------------------------------------------------------------
# Class definitions — order matters (index = class id in YOLO labels)
# ---------------------------------------------------------------------------
CLASSES = ["Triangle", "Square", "Rectangle", "Circle", "Pentagon"]

CLASS_COLORS = {
    "Triangle":  (255, 255, 0),    # cyan  (BGR)
    "Square":    (0, 165, 255),    # orange
    "Rectangle": (255, 0, 255),    # magenta
    "Circle":    (0, 255, 0),      # green
    "Pentagon":  (0, 255, 255),    # yellow
}

CLASS_COLORS_CSS = {
    "Triangle":  "#00FFFF",
    "Square":    "#FFA500",
    "Rectangle": "#FF00FF",
    "Circle":    "#00FF00",
    "Pentagon":  "#FFFF00",
}

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
SYNTH_IMG_SIZE = 640
SYNTH_TRAIN_COUNT = 2000
SYNTH_VAL_COUNT = 400
SYNTH_MIN_SHAPES = 1
SYNTH_MAX_SHAPES = 4
SYNTH_MIN_SIZE = 60
SYNTH_MAX_SIZE = 200
SYNTH_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

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
# OCR
# ---------------------------------------------------------------------------
OCR_LANGUAGES = ["en"]
OCR_CONFIDENCE_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MAX_LOG_ENTRIES = 20
DETECTION_COOLDOWN = 2.0   # seconds
CENTROID_SHIFT_THRESHOLD = 50  # pixels
OCR_FRAME_SKIP = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "shapes_dataset")
BEST_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
