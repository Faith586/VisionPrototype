"""
Train a YOLOv8 model on the synthetic shapes dataset.

Usage:
    python scripts/train.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from ultralytics import YOLO


def train():
    yaml_path = os.path.join(config.DATASET_DIR, "dataset.yaml")

    if not os.path.exists(yaml_path):
        print("Dataset not found. Run 'python scripts/generate_data.py' first.")
        sys.exit(1)

    print(f"Loading base model: {config.YOLO_MODEL_BASE}")
    model = YOLO(config.YOLO_MODEL_BASE)

    print(f"Training for {config.TRAIN_EPOCHS} epochs...")
    model.train(
        data=yaml_path,
        epochs=config.TRAIN_EPOCHS,
        imgsz=config.TRAIN_IMG_SIZE,
        batch=config.TRAIN_BATCH,
        project=os.path.join(config.BASE_DIR, "runs", "detect"),
        name="train",
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
        verbose=True,
    )

    best = os.path.join(config.BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
    if os.path.exists(best):
        print(f"\nTraining complete! Best weights: {best}")
    else:
        print("\nTraining complete. Check runs/detect/train/ for results.")


if __name__ == "__main__":
    train()
