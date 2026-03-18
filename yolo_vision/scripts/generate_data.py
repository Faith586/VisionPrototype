"""
Generate synthetic training data for YOLO shape + character tag detection.

Creates images with randomly placed/rotated colored shape tags, each bearing
a specific letter (X, Y, or Z).  The compound class encodes both the shape
and the letter so YOLO learns to distinguish identical shapes that carry
different letters/colors.

Usage:
    python scripts/generate_data.py
"""

import os
import sys
import random
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── helpers ──────────────────────────────────────────────────────────────────

def _random_bg(size):
    """Generate a random background (solid, gradient, or noisy)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    choice = random.random()
    if choice < 0.35:
        color = [random.randint(100, 240) for _ in range(3)]
        img[:] = color
    elif choice < 0.7:
        c1 = np.array([random.randint(80, 230) for _ in range(3)], dtype=np.float64)
        c2 = np.array([random.randint(80, 230) for _ in range(3)], dtype=np.float64)
        for y in range(size):
            t = y / size
            img[y, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    else:
        base = random.randint(120, 220)
        img[:] = base
        noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _tag_fill_color():
    """White-ish fill to mimic white 3D-printed filament under varied lighting."""
    v = random.randint(210, 255)
    return (v, v - random.randint(0, 10), v - random.randint(0, 10))


def _draw_character(img, cx, cy, size, char):
    """Draw a dark character on a white shape (black filament letter on white tag)."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_size = int(size * 0.5)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = cx - tw // 2
    ty = cy - th // 2

    # Dark text (black filament) with slight variation for robustness
    v = random.randint(0, 50)
    text_color = (v, v, v)

    draw.text((tx, ty), char, fill=text_color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _make_triangle(cx, cy, size):
    angle_offset = random.uniform(0, 2 * math.pi)
    r = size // 2
    pts = []
    for i in range(3):
        a = angle_offset + i * (2 * math.pi / 3)
        px = int(cx + r * math.cos(a))
        py = int(cy + r * math.sin(a))
        pts.append([px, py])
    return np.array(pts, dtype=np.int32)


def _make_square(cx, cy, size):
    half = size // 2
    angle = random.uniform(-0.3, 0.3)
    pts = []
    for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half)]:
        rx = int(cx + dx * math.cos(angle) - dy * math.sin(angle))
        ry = int(cy + dx * math.sin(angle) + dy * math.cos(angle))
        pts.append([rx, ry])
    return np.array(pts, dtype=np.int32)


SHAPE_GENERATORS = {
    "Triangle":  _make_triangle,
    "Square":    _make_square,
    "Circle":    None,
}


def _draw_shape(img, class_id, cx, cy, size):
    """Draw a white shape tag with a dark letter.  Returns bounding box."""
    class_name = config.CLASSES[class_id]
    shape_name, letter = class_name.rsplit("-", 1)

    fill_color = _tag_fill_color()
    border_color = tuple(max(0, c - random.randint(30, 60)) for c in fill_color)

    if shape_name == "Circle":
        r = size // 2
        cv2.circle(img, (cx, cy), r, fill_color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, border_color, 2, cv2.LINE_AA)
        img[:] = _draw_character(img, cx, cy, size, letter)
        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r
    else:
        gen_fn = SHAPE_GENERATORS[shape_name]
        pts = gen_fn(cx, cy, size)
        cv2.fillPoly(img, [pts], fill_color, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, border_color, 2, cv2.LINE_AA)
        img[:] = _draw_character(img, cx, cy, size, letter)
        x1 = pts[:, 0].min()
        y1 = pts[:, 1].min()
        x2 = pts[:, 0].max()
        y2 = pts[:, 1].max()

    return x1, y1, x2, y2


def _add_augmentation(img):
    """Apply random augmentations to make the model robust."""
    if random.random() > 0.5:
        delta = random.randint(-40, 40)
        img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    if random.random() > 0.6:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() > 0.7:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ── main generation ──────────────────────────────────────────────────────────

def generate_image(img_size):
    """Generate a single training image with random shape tags, return (img, labels)."""
    img = _random_bg(img_size)
    num_shapes = random.randint(config.SYNTH_MIN_SHAPES, config.SYNTH_MAX_SHAPES)
    labels = []
    placed_boxes = []

    for _ in range(num_shapes):
        class_id = random.randint(0, len(config.CLASSES) - 1)
        size = random.randint(config.SYNTH_MIN_SIZE, config.SYNTH_MAX_SIZE)
        margin = size // 2 + 10

        # Try to place without overlapping
        placed = False
        for _attempt in range(20):
            cx = random.randint(margin, img_size - margin)
            cy = random.randint(margin, img_size - margin)

            ok = True
            for bx1, by1, bx2, by2 in placed_boxes:
                if not (cx - size // 2 > bx2 or cx + size // 2 < bx1 or
                        cy - size // 2 > by2 or cy + size // 2 < by1):
                    ok = False
                    break
            if ok:
                placed = True
                break

        if not placed:
            continue

        x1, y1, x2, y2 = _draw_shape(img, class_id, cx, cy, size)

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_size, x2)
        y2 = min(img_size, y2)
        placed_boxes.append((x1, y1, x2, y2))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 5 or bh < 5:
            continue
        x_center = (x1 + x2) / 2 / img_size
        y_center = (y1 + y2) / 2 / img_size
        w_norm = bw / img_size
        h_norm = bh / img_size

        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    img = _add_augmentation(img)
    return img, labels


def generate_dataset():
    """Generate the full train/val dataset."""
    dataset_dir = config.DATASET_DIR

    for split, count in [("train", config.SYNTH_TRAIN_COUNT), ("val", config.SYNTH_VAL_COUNT)]:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        print(f"Generating {count} {split} images...")
        for i in range(count):
            img, labels = generate_image(config.SYNTH_IMG_SIZE)

            fname = f"{split}_{i:05d}"
            cv2.imwrite(os.path.join(img_dir, fname + ".jpg"), img)
            with open(os.path.join(lbl_dir, fname + ".txt"), "w") as f:
                f.write("\n".join(labels))

            if (i + 1) % 200 == 0:
                print(f"  {split}: {i + 1}/{count}")

    # Write dataset YAML
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n\n")
        f.write(f"nc: {len(config.CLASSES)}\n")
        f.write(f"names: {config.CLASSES}\n")

    print(f"\nDataset saved to {dataset_dir}")
    print(f"  Classes ({len(config.CLASSES)}): {config.CLASSES}")
    print(f"YAML config: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    generate_dataset()
