"""
Entry point — starts camera capture and runs the detection + dashboard loop.

Usage:
    python main.py              # Use default camera (index 0)
    python main.py --camera 1   # Use camera at index 1
"""

import argparse
import math
import sys
import time

import cv2

from config import (
    DEFAULT_CAMERA_INDEX,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    CENTROID_SHIFT_THRESHOLD,
    OCR_SKIP_FRAMES,
)
from detector import detect_shapes
from ocr_engine import OCREngine
from detection_log import DetectionLog
from dashboard import (
    draw_detections_on_frame,
    draw_fps,
    build_sidebar,
    compose_dashboard,
)


def _centroid_distance(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def main():
    parser = argparse.ArgumentParser(description="Shape + Character Vision System")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera device index")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {args.camera}.")
        print("        Make sure your USB webcam is connected and not in use by another application.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    cam_name = cap.getBackendName()

    print(f"[INFO] Camera opened: {cam_name}")
    print(f"[INFO] Resolution: {actual_w}x{actual_h}  Reported FPS: {cam_fps:.1f}")
    print("[INFO] Press 'q' to quit, 's' to save a screenshot.")

    ocr = OCREngine()
    detection_log = DetectionLog()

    # Cache for OCR results to avoid re-running every frame.
    # Maps shape_name -> (centroid, character, confidence)
    ocr_cache: dict[str, tuple[tuple[int, int], str, float]] = {}

    fps = 0.0
    frame_count = 0
    fps_start = time.time()
    current_detection = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame. Retrying...")
            continue

        detections = detect_shapes(frame)

        # Build OCR results for this frame.
        ocr_results: dict[int, tuple[str, float]] = {}

        for i, det in enumerate(detections):
            cache_key = det.shape_name

            # Decide whether to run OCR or use cached result.
            run_ocr = True
            if OCR_SKIP_FRAMES and cache_key in ocr_cache:
                prev_centroid, prev_char, prev_conf = ocr_cache[cache_key]
                if _centroid_distance(det.centroid, prev_centroid) < CENTROID_SHIFT_THRESHOLD:
                    # Shape hasn't moved significantly — reuse cached OCR.
                    ocr_results[i] = (prev_char, prev_conf)
                    run_ocr = False

            if run_ocr:
                char, conf = ocr.recognize(frame, det.bounding_rect)
                ocr_results[i] = (char, conf)
                ocr_cache[cache_key] = (det.centroid, char, conf)

            # Log (debounce handled internally).
            char, conf = ocr_results[i]
            logged = detection_log.try_log(
                shape_name=det.shape_name,
                character=char,
                confidence=conf,
                color=det.color,
                centroid=det.centroid,
            )

            # Update current detection display.
            current_detection = {
                "shape_name": det.shape_name,
                "character": char,
                "shape_conf": 95.0,  # Contour-based — high confidence by definition.
                "char_conf": conf * 100,
                "color": det.color,
                "timestamp_str": time.strftime("%H:%M:%S"),
            }

        # Draw on frame.
        draw_detections_on_frame(frame, detections, ocr_results)

        # FPS calculation (rolling).
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()
        draw_fps(frame, fps)

        # Build sidebar and compose dashboard.
        sidebar = build_sidebar(frame.shape[0], detection_log, current_detection)
        dashboard = compose_dashboard(frame, sidebar)

        cv2.imshow("Shape Vision Dashboard", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, dashboard)
            print(f"[INFO] Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Shut down cleanly.")


if __name__ == "__main__":
    main()
