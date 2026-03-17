"""
Real-time YOLO shape detection dashboard.

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
from ocr_engine import OCREngine

app = Flask(__name__)

# ── shared state (protected by lock) ─────────────────────────────────────────
_lock = threading.Lock()
_latest_frame: bytes | None = None
_current_detection: dict | None = None
_detection_log = DetectionLog()
_fps = 0.0


# ── camera + YOLO inference loop ────────────────────────────────────────────

def camera_loop(weights_path):
    global _latest_frame, _current_detection, _fps

    # Load YOLO model
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        print("Run these first:")
        print("  python scripts/generate_data.py")
        print("  python scripts/train.py")
        sys.exit(1)

    print(f"Loading YOLO model: {weights_path}")
    model = YOLO(weights_path)
    ocr = OCREngine()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    print("Camera started. Open http://127.0.0.1:5000 in your browser.")

    frame_count = 0
    fps_start = time.time()
    ocr_cache = {}  # class_name -> { cx, cy, char, conf }

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run YOLO inference
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

                shape_name = config.CLASSES[cls_id]
                bgr_color = config.CLASS_COLORS[shape_name]
                css_color = config.CLASS_COLORS_CSS[shape_name]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # OCR: use cache or run fresh
                char, char_conf = "???", 0.0
                cache_key = shape_name
                if cache_key in ocr_cache:
                    cached = ocr_cache[cache_key]
                    dist = ((cx - cached["cx"])**2 + (cy - cached["cy"])**2) ** 0.5
                    if dist < config.CENTROID_SHIFT_THRESHOLD:
                        char, char_conf = cached["char"], cached["conf"]

                if frame_count % config.OCR_FRAME_SKIP == 0 or char == "???":
                    new_char, new_conf = ocr.recognize(frame, (x1, y1, x2, y2))
                    if new_char != "???":
                        char, char_conf = new_char, new_conf
                        ocr_cache[cache_key] = {"cx": cx, "cy": cy, "char": char, "conf": char_conf}

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 2)

                # Label
                label = f"{shape_name} - {char} ({conf*100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), bgr_color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                detections.append({
                    "shape": shape_name,
                    "char": char,
                    "shape_conf": conf,
                    "char_conf": char_conf,
                    "color": css_color,
                    "centroid": (cx, cy),
                })

                # Log
                with _lock:
                    _detection_log.try_log(shape_name, char, char_conf, css_color, (cx, cy))

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 0.5:
            _fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Draw FPS
        cv2.putText(annotated, f"FPS: {_fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode and store
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

        with _lock:
            _latest_frame = jpeg.tobytes()
            if detections:
                d = detections[0]
                _current_detection = {
                    "shape": d["shape"],
                    "character": d["char"],
                    "shape_conf": d["shape_conf"] * 100,
                    "char_conf": d["char_conf"] * 100,
                    "color": d["color"],
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
        time.sleep(0.016)  # ~60 fps cap


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
    parser = argparse.ArgumentParser(description="YOLO Shape Vision Dashboard")
    parser.add_argument("--weights", default=config.BEST_WEIGHTS, help="Path to YOLO weights (.pt)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    t = threading.Thread(target=camera_loop, args=(args.weights,), daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
