"""
Dashboard UI rendering using OpenCV.

Draws the live camera feed with bounding boxes on the left and a
detection sidebar on the right.
"""

import cv2
import numpy as np

from config import SIDEBAR_WIDTH, FONT_SCALE_LABEL, FONT_THICKNESS


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_GRAY = (180, 180, 180)
_DARK_BG = (30, 30, 30)
_SECTION_BG = (45, 45, 45)


def draw_detections_on_frame(frame, detections, ocr_results):
    """Draw bounding boxes and labels on the camera frame.

    Parameters
    ----------
    frame : np.ndarray
        The live camera frame (modified in place).
    detections : list[Detection]
        Shape detections from detector.detect_shapes().
    ocr_results : dict
        Mapping from detection index to (character, confidence).
    """
    for i, det in enumerate(detections):
        x, y, w, h = det.bounding_rect
        color = det.color

        # Bounding box.
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label.
        char, conf = ocr_results.get(i, ("???", 0.0))
        label = f"{det.shape_name} - {char} ({conf * 100:.1f}%)"

        # Background rectangle for label readability.
        (tw, th), baseline = cv2.getTextSize(label, _FONT, FONT_SCALE_LABEL, FONT_THICKNESS)
        label_y = max(y - 10, th + 4)
        cv2.rectangle(frame, (x, label_y - th - 4), (x + tw + 4, label_y + baseline), (0, 0, 0), -1)
        cv2.putText(frame, label, (x + 2, label_y - 2), _FONT, FONT_SCALE_LABEL, color, FONT_THICKNESS)


def draw_fps(frame, fps):
    """Draw the FPS counter in the top-left corner."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30), _FONT, 0.8, (0, 255, 0), 2)


def _draw_text(sidebar, text, y, color=_WHITE, scale=0.5, thickness=1):
    """Helper to draw text on the sidebar and return the new y position."""
    cv2.putText(sidebar, text, (10, y), _FONT_SMALL, scale, color, thickness)
    return y + int(22 * scale / 0.5)


def build_sidebar(height, detection_log, current_detection=None):
    """Build the sidebar image.

    Parameters
    ----------
    height : int
        Height in pixels (must match the camera frame).
    detection_log : DetectionLog
        The detection log with history and stats.
    current_detection : dict or None
        Dict with keys: shape_name, character, shape_conf, char_conf, color, timestamp_str.

    Returns
    -------
    np.ndarray
        BGR image for the sidebar.
    """
    sidebar = np.full((height, SIDEBAR_WIDTH, 3), _DARK_BG, dtype=np.uint8)
    y = 10

    # --- Section: Current Detection ---
    cv2.rectangle(sidebar, (5, y), (SIDEBAR_WIDTH - 5, y + 120), _SECTION_BG, -1)
    y += 5
    y = _draw_text(sidebar, "CURRENT DETECTION", y + 15, _GRAY, 0.45, 1)

    if current_detection:
        cd = current_detection
        y = _draw_text(sidebar, f"{cd['shape_name']} - {cd['character']}", y + 5, cd["color"], 0.7, 2)
        y = _draw_text(sidebar, f"Shape conf: {cd['shape_conf']:.1f}%", y, _WHITE, 0.4, 1)
        y = _draw_text(sidebar, f"Char conf:  {cd['char_conf']:.1f}%", y, _WHITE, 0.4, 1)
        y = _draw_text(sidebar, cd["timestamp_str"], y, _GRAY, 0.35, 1)
    else:
        y = _draw_text(sidebar, "No detection", y + 10, _GRAY, 0.5, 1)
        y += 40

    y = max(y, 140)

    # --- Section: Detection Log ---
    cv2.line(sidebar, (5, y), (SIDEBAR_WIDTH - 5, y), _GRAY, 1)
    y += 5
    y = _draw_text(sidebar, "DETECTION LOG", y + 12, _GRAY, 0.45, 1)
    y += 2

    log_entries = detection_log.entries[-15:]  # Show last 15 that fit
    for entry in reversed(log_entries):
        if y > height - 150:
            break
        text = f"[{entry.time_str}] {entry.label} ({entry.confidence * 100:.0f}%)"
        y = _draw_text(sidebar, text, y, entry.color, 0.35, 1)

    # --- Section: Statistics ---
    stat_y = height - 130
    cv2.line(sidebar, (5, stat_y), (SIDEBAR_WIDTH - 5, stat_y), _GRAY, 1)
    stat_y += 5
    stat_y = _draw_text(sidebar, "STATISTICS", stat_y + 12, _GRAY, 0.45, 1)
    stat_y = _draw_text(sidebar, f"Total detections: {detection_log.total_count}", stat_y + 2, _WHITE, 0.4, 1)

    for shape_name, count in sorted(detection_log.shape_counts.items()):
        stat_y = _draw_text(sidebar, f"  {shape_name}: {count}", stat_y, _WHITE, 0.38, 1)
        if stat_y > height - 20:
            break

    avg_conf = detection_log.average_confidence * 100
    _draw_text(sidebar, f"Avg confidence: {avg_conf:.1f}%", min(stat_y, height - 20), _WHITE, 0.4, 1)

    return sidebar


def compose_dashboard(frame, sidebar):
    """Horizontally concatenate the camera frame and sidebar."""
    fh = frame.shape[0]
    sh = sidebar.shape[0]
    if fh != sh:
        sidebar = cv2.resize(sidebar, (SIDEBAR_WIDTH, fh))
    return np.hstack([frame, sidebar])
