"""
Entry point — Flask web server with MJPEG video streaming and detection API.

Usage:
    python main.py              # Use default camera (index 0)
    python main.py --camera 1   # Use camera at index 1
    python main.py --port 8080  # Use a different port
"""

import argparse
import math
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

from config import (
    DEFAULT_CAMERA_INDEX,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    CENTROID_SHIFT_THRESHOLD,
    OCR_SKIP_FRAMES,
    SHAPE_CONFIG,
    FONT_SCALE_LABEL,
    FONT_THICKNESS,
    WEB_HOST,
    WEB_PORT,
)
from detector import detect_shapes
from ocr_engine import OCREngine
from detection_log import DetectionLog

app = Flask(__name__)

# --- Shared state (protected by lock) ---
_lock = threading.Lock()
_latest_frame = None          # Most recent annotated JPEG bytes
_current_detection = None     # Dict for the sidebar "current detection"
_detection_log = DetectionLog()
_fps = 0.0

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _centroid_distance(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def _bgr_to_css(bgr):
    """Convert a BGR tuple to a CSS hex color string."""
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def camera_loop(camera_index):
    """Background thread: capture frames, run detection, encode JPEG."""
    global _latest_frame, _current_detection, _fps

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {camera_index}.")
        print("        Make sure your webcam is connected and not in use by another app.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Camera opened — Resolution: {actual_w}x{actual_h}  FPS: {cam_fps:.1f}")

    ocr = OCREngine()
    ocr_cache: dict[str, tuple[tuple[int, int], str, float]] = {}

    frame_count = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        detections = detect_shapes(frame)

        ocr_results: dict[int, tuple[str, float]] = {}
        det_info = None

        for i, det in enumerate(detections):
            cache_key = det.shape_name
            run_ocr = True

            if OCR_SKIP_FRAMES and cache_key in ocr_cache:
                prev_centroid, prev_char, prev_conf = ocr_cache[cache_key]
                if _centroid_distance(det.centroid, prev_centroid) < CENTROID_SHIFT_THRESHOLD:
                    ocr_results[i] = (prev_char, prev_conf)
                    run_ocr = False

            if run_ocr:
                char, conf = ocr.recognize(frame, det.bounding_rect)
                ocr_results[i] = (char, conf)
                ocr_cache[cache_key] = (det.centroid, char, conf)

            char, conf = ocr_results[i]

            with _lock:
                _detection_log.try_log(
                    shape_name=det.shape_name,
                    character=char,
                    confidence=conf,
                    color=det.color,
                    centroid=det.centroid,
                )

            css_color = SHAPE_CONFIG.get(det.shape_name, {}).get("css_color", _bgr_to_css(det.color))
            det_info = {
                "shape_name": det.shape_name,
                "character": char,
                "shape_conf": 95.0,
                "char_conf": round(conf * 100, 1),
                "css_color": css_color,
                "timestamp_str": time.strftime("%H:%M:%S"),
            }

        # Draw bounding boxes + labels on frame.
        for i, det in enumerate(detections):
            x, y, w, h = det.bounding_rect
            color = det.color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            char, conf = ocr_results.get(i, ("???", 0.0))
            label = f"{det.shape_name} - {char} ({conf * 100:.1f}%)"
            (tw, th), baseline = cv2.getTextSize(label, _FONT, FONT_SCALE_LABEL, FONT_THICKNESS)
            label_y = max(y - 10, th + 4)
            cv2.rectangle(frame, (x, label_y - th - 4), (x + tw + 4, label_y + baseline), (0, 0, 0), -1)
            cv2.putText(frame, label, (x + 2, label_y - 2), _FONT, FONT_SCALE_LABEL, color, FONT_THICKNESS)

        # FPS.
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 0.5:
            local_fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()
            with _lock:
                _fps = local_fps

        # Draw FPS on frame.
        with _lock:
            fps_val = _fps
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30), _FONT, 0.8, (0, 255, 0), 2)

        # Encode to JPEG.
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        with _lock:
            _latest_frame = jpeg.tobytes()
            if det_info is not None:
                _current_detection = det_info


def generate_mjpeg():
    """Yield MJPEG frames for the video stream endpoint."""
    while True:
        with _lock:
            frame_bytes = _latest_frame
        if frame_bytes is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)  # ~30 FPS cap for the stream


# ----- Flask routes -----

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/detections")
def api_detections():
    """JSON endpoint returning current detection, log, and stats."""
    with _lock:
        current = _current_detection
        entries = []
        for e in _detection_log.entries[-20:]:
            css_color = SHAPE_CONFIG.get(e.shape_name, {}).get("css_color", _bgr_to_css(e.color))
            entries.append({
                "time": e.time_str,
                "shape": e.shape_name,
                "char": e.character,
                "conf": round(e.confidence * 100, 1),
                "css_color": css_color,
            })
        stats = {
            "total": _detection_log.total_count,
            "by_shape": dict(_detection_log.shape_counts),
            "avg_conf": round(_detection_log.average_confidence * 100, 1),
        }
        fps = round(_fps, 1)

    return jsonify({
        "current": current,
        "log": list(reversed(entries)),
        "stats": stats,
        "fps": fps,
    })


def main():
    parser = argparse.ArgumentParser(description="Shape + Character Vision System (Web)")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera device index")
    parser.add_argument("--port", type=int, default=WEB_PORT, help="Web server port")
    args = parser.parse_args()

    # Start camera processing in a background thread.
    cam_thread = threading.Thread(target=camera_loop, args=(args.camera,), daemon=True)
    cam_thread.start()

    print(f"[INFO] Open your browser to:  http://localhost:{args.port}")
    app.run(host=WEB_HOST, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
