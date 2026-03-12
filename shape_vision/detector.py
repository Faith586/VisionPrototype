"""
Shape detection using OpenCV classical computer vision.

Converts frames to grayscale, applies adaptive thresholding, finds contours,
and classifies shapes by vertex count and circularity using the config.
"""

import math
import cv2
import numpy as np

from config import (
    SHAPE_CONFIG,
    MIN_CONTOUR_AREA,
    APPROX_POLY_EPSILON,
    EDGE_MARGIN,
)


class Detection:
    """Represents a single detected shape in a frame."""

    __slots__ = ("shape_name", "color", "contour", "bounding_rect", "centroid", "approx")

    def __init__(self, shape_name, color, contour, bounding_rect, centroid, approx):
        self.shape_name = shape_name
        self.color = color
        self.contour = contour
        self.bounding_rect = bounding_rect  # (x, y, w, h)
        self.centroid = centroid              # (cx, cy)
        self.approx = approx


def _classify_shape(approx, contour):
    """Classify a contour into a shape name using SHAPE_CONFIG.

    Returns (shape_name, color) or (None, None) if no match.
    """
    num_vertices = len(approx)

    # Compute circularity once (used for circle detection).
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    # Check circle first — circularity is the primary signal.
    for name, cfg in SHAPE_CONFIG.items():
        if "circularity_threshold" in cfg:
            if num_vertices >= 8 or circularity > cfg["circularity_threshold"]:
                # Extra check: only accept if vertex count is high enough
                # (avoids false positives on pentagons etc.)
                if circularity > cfg["circularity_threshold"]:
                    return name, cfg["color"]

    # Check vertex-based shapes.
    # Build a list of candidates that match the vertex count.
    candidates = []
    for name, cfg in SHAPE_CONFIG.items():
        if "circularity_threshold" in cfg:
            continue  # Already handled above.
        min_v = cfg.get("min_vertices", 0)
        max_v = cfg.get("max_vertices", 999)
        if min_v <= num_vertices <= max_v:
            candidates.append((name, cfg))

    if not candidates:
        return None, None

    # If there's an aspect_range filter (Square vs Rectangle), resolve it.
    if len(candidates) > 1:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / h if h > 0 else 0
        for name, cfg in candidates:
            ar = cfg.get("aspect_range")
            if ar is not None:
                if ar[0] <= aspect <= ar[1]:
                    return name, cfg["color"]
        # None matched the narrow range — pick the one with aspect_range=None (Rectangle).
        for name, cfg in candidates:
            if cfg.get("aspect_range") is None:
                return name, cfg["color"]

    # Single candidate — return it.
    name, cfg = candidates[0]
    return name, cfg["color"]


def preprocess_frame(frame):
    """Convert to grayscale, blur, and apply adaptive threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Brightness/contrast normalization via CLAHE.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3
    )
    return thresh


def detect_shapes(frame):
    """Detect and classify shapes in *frame*.

    Returns a list of Detection objects.
    """
    h, w = frame.shape[:2]
    thresh = preprocess_frame(frame)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        # Reject contours touching the frame edges.
        x, y, bw, bh = cv2.boundingRect(cnt)
        if x <= EDGE_MARGIN or y <= EDGE_MARGIN:
            continue
        if (x + bw) >= (w - EDGE_MARGIN) or (y + bh) >= (h - EDGE_MARGIN):
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_POLY_EPSILON * peri, True)

        shape_name, color = _classify_shape(approx, cnt)
        if shape_name is None:
            continue

        # Compute centroid.
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        detections.append(
            Detection(
                shape_name=shape_name,
                color=color,
                contour=cnt,
                bounding_rect=(x, y, bw, bh),
                centroid=(cx, cy),
                approx=approx,
            )
        )

    return detections
