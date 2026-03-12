"""
Shape definitions, colors, thresholds, and application configuration.

To add a new shape, simply add an entry to SHAPE_CONFIG.
No other code changes are needed.
"""

# Shape configuration dictionary.
# Each entry defines how to detect a shape by vertex count, aspect ratio, or circularity.
# Colors are BGR tuples (OpenCV convention).
SHAPE_CONFIG = {
    "Triangle": {
        "min_vertices": 3,
        "max_vertices": 3,
        "color": (255, 255, 0),       # Cyan
    },
    "Square": {
        "min_vertices": 4,
        "max_vertices": 4,
        "aspect_range": (0.85, 1.15),
        "color": (0, 165, 255),        # Orange
    },
    "Rectangle": {
        "min_vertices": 4,
        "max_vertices": 4,
        "aspect_range": None,          # Any aspect ratio (non-square quads)
        "color": (255, 0, 255),        # Purple
    },
    "Circle": {
        "circularity_threshold": 0.80,
        "color": (0, 255, 0),          # Green
    },
    "Pentagon": {
        "min_vertices": 5,
        "max_vertices": 5,
        "color": (0, 255, 255),        # Yellow
    },
}

# --- Contour filtering ---
MIN_CONTOUR_AREA = 1500          # Minimum area in pixels to consider a contour
APPROX_POLY_EPSILON = 0.03      # Fraction of arc length for polygon approximation
EDGE_MARGIN = 5                  # Pixels from frame edge — reject contours touching edges

# --- Detection debounce ---
CENTROID_SHIFT_THRESHOLD = 50    # Pixels — new detection if centroid moves more than this
DETECTION_COOLDOWN = 2.0         # Seconds — minimum gap between identical detections

# --- OCR ---
OCR_LANGUAGES = ["en"]
OCR_CONFIDENCE_THRESHOLD = 0.20  # Minimum confidence to accept an OCR result
OCR_SKIP_FRAMES = True           # Only run OCR on new/moved shapes, not every frame

# --- Dashboard ---
SIDEBAR_WIDTH = 320              # Pixels
MAX_LOG_ENTRIES = 20             # Number of entries in the detection log
FONT_SCALE_LABEL = 0.6
FONT_THICKNESS = 2

# --- Camera ---
DEFAULT_CAMERA_INDEX = 0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
