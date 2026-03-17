"""
Webcam-based data collection tool for capturing real shape images.

Controls:
    1-5   : Select class (1=Triangle, 2=Square, 3=Rectangle, 4=Circle, 5=Pentagon)
    SPACE : Capture current frame and draw bounding box
    s     : Save annotation for current capture
    r     : Reset current capture (re-draw box)
    q     : Quit

After pressing SPACE, click and drag on the image to draw a bounding box,
then press 's' to save.

Usage:
    python scripts/collect_data.py
    python scripts/collect_data.py --split val   # save to validation set
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── globals for mouse callback ───────────────────────────────────────────────
drawing = False
ix, iy = -1, -1
box = None
display_img = None


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, box, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        box = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = display_img.copy()
        cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Collect Data", temp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        temp = display_img.copy()
        cv2.rectangle(temp, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("Collect Data", temp)


def main():
    global display_img, box

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    img_dir = os.path.join(config.DATASET_DIR, args.split, "images")
    lbl_dir = os.path.join(config.DATASET_DIR, args.split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Count existing files to continue numbering
    existing = [f for f in os.listdir(img_dir) if f.startswith("real_")]
    start_idx = len(existing)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    cv2.namedWindow("Collect Data")
    cv2.setMouseCallback("Collect Data", mouse_callback)

    current_class = 0
    captured_frame = None
    count = start_idx
    mode = "live"  # "live" or "annotate"

    print("=== Shape Data Collection Tool ===")
    print("Keys: 1-5 select class | SPACE capture | s save | r reset | q quit")
    print(f"Current class: {config.CLASSES[current_class]}")
    print(f"Saving to: {args.split}/")

    while True:
        if mode == "live":
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            # Show class and instructions
            label = f"Class: {config.CLASSES[current_class]} | SPACE to capture | q to quit"
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Saved: {count - start_idx}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow("Collect Data", display)
        # else: we're in annotate mode, display is managed by mouse callback

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            current_class = key - ord('1')
            print(f"Selected class: {config.CLASSES[current_class]}")
        elif key == ord(' ') and mode == "live":
            ret, captured_frame = cap.read()
            if ret:
                mode = "annotate"
                box = None
                display_img = captured_frame.copy()
                info = f"Draw box for {config.CLASSES[current_class]} | s=save r=reset"
                cv2.putText(display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Collect Data", display_img)
                print("Draw a bounding box around the shape, then press 's' to save.")
        elif key == ord('s') and mode == "annotate" and box is not None:
            h, w = captured_frame.shape[:2]
            x1, y1, x2, y2 = box
            # Convert to YOLO format
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            fname = f"real_{count:05d}"
            cv2.imwrite(os.path.join(img_dir, fname + ".jpg"), captured_frame)
            with open(os.path.join(lbl_dir, fname + ".txt"), "w") as f:
                f.write(f"{current_class} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

            count += 1
            print(f"Saved {fname} as {config.CLASSES[current_class]} ({count - start_idx} total)")
            mode = "live"
            box = None
        elif key == ord('r') and mode == "annotate":
            box = None
            display_img = captured_frame.copy()
            info = f"Draw box for {config.CLASSES[current_class]} | s=save r=reset"
            cv2.putText(display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Collect Data", display_img)
            print("Box reset. Draw again.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Saved {count - start_idx} new images to {args.split}/")


if __name__ == "__main__":
    main()
