"""OCR engine for reading characters inside detected shapes."""

import cv2
import numpy as np
import config


class OCREngine:
    """Lazy-loading EasyOCR wrapper for single-character recognition."""

    def __init__(self):
        self._reader = None

    def _ensure_loaded(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=False)

    def _preprocess_roi(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        h, w = gray.shape[:2]
        if h < 64 or w < 64:
            scale = max(64 / h, 64 / w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray

    def recognize(self, frame, bbox):
        """Run OCR on a bounding box region.

        Args:
            frame: Full camera frame (BGR).
            bbox: (x1, y1, x2, y2) pixel coordinates.

        Returns:
            (character, confidence) or ("???", 0.0)
        """
        self._ensure_loaded()
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "???", 0.0

        processed = self._preprocess_roi(roi)
        results = self._reader.readtext(processed, detail=1)

        if not results:
            return "???", 0.0

        best_text, best_conf = "", 0.0
        for _, text, conf in results:
            if conf > best_conf:
                best_text = text.strip()
                best_conf = conf

        if best_conf < config.OCR_CONFIDENCE_THRESHOLD or not best_text:
            return "???", 0.0

        # Return first alphanumeric character
        for ch in best_text.upper():
            if ch.isalnum():
                return ch, best_conf
        return "???", 0.0
