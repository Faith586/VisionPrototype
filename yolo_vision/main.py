"""
Real-time YOLO shape + character tag detection dashboard.

The model detects compound classes like "Triangle-X", "Square-Z", etc.
No OCR needed — the letter/color is baked into the YOLO class itself.

Usage:
    python main.py                      # uses best.pt from training
    python main.py --weights path.pt    # custom weights
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

import config
from detection_log import DetectionLog

app = Flask(__name__)

# ── shared state (protected by lock) ─────────────────────────────────────────
_lock = threading.Lock()
_latest_frame = None   # type: Optional[bytes]
_current_detection = None  # type: Optional[dict]
_detection_log = DetectionLog()
_fps = 0.0


# ── camera + YOLO inference loop ────────────────────────────────────────────

def camera_loop(weights_path):
    global _latest_frame, _current_detection, _fps

    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        print("Run these first:")
        print("  python scripts/generate_data.py")
        print("  python scripts/train.py")
        sys.exit(1)

    print(f"Loading YOLO model: {weights_path}")
    model = YOLO(weights_path)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    print("Camera started. Open http://127.0.0.1:5000 in your browser.")

    frame_count = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False,
        )

        annotated = frame.copy()
        detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                if cls_id >= len(config.CLASSES):
                    continue

                class_name = config.CLASSES[cls_id]
                shape_name, letter = class_name.rsplit("-", 1)
                bgr_color = config.CLASS_COLORS[class_name]
                css_color = config.CLASS_COLORS_CSS[class_name]
                tag_css = config.TAG_COLORS_CSS[letter]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 2)

                # Label
                label = f"{shape_name}-{letter} ({conf*100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), bgr_color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                detections.append({
                    "shape": shape_name,
                    "letter": letter,
                    "class": class_name,
                    "conf": conf,
                    "color": css_color,
                    "tag_color": tag_css,
                    "centroid": (cx, cy),
                })

                # Log
                with _lock:
                    _detection_log.try_log(class_name, letter, conf, css_color, (cx, cy))

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 0.5:
            _fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        cv2.putText(annotated, f"FPS: {_fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

        with _lock:
            _latest_frame = jpeg.tobytes()
            if detections:
                d = detections[0]
                _current_detection = {
                    "shape": d["shape"],
                    "letter": d["letter"],
                    "class": d["class"],
                    "conf": d["conf"] * 100,
                    "color": d["color"],
                    "tag_color": d["tag_color"],
                    "time": time.strftime("%H:%M:%S"),
                }


# ── Flask routes ─────────────────────────────────────────────────────────────

def generate_mjpeg():
    while True:
        with _lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.016)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/detections")
def api_detections():
    with _lock:
        log_entries = [
            {
                "time": e.time_str,
                "label": e.label,
                "confidence": round(e.confidence * 100, 1),
                "color": e.color,
            }
            for e in reversed(_detection_log.entries)
        ]
        stats = {
            "total": _detection_log.total_count,
            "avg_confidence": round(_detection_log.average_confidence * 100, 1),
            "shape_counts": dict(_detection_log.shape_counts),
        }
        return jsonify(
            current=_current_detection,
            log=log_entries,
            stats=stats,
            fps=round(_fps, 1),
        )


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO Shape+Tag Vision Dashboard")
    parser.add_argument("--weights", default=config.BEST_WEIGHTS, help="Path to YOLO weights (.pt)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    t = threading.Thread(target=camera_loop, args=(args.weights,), daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
