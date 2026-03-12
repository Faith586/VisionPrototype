"""
Character recognition using EasyOCR.

Wraps EasyOCR to run on cropped shape ROIs. Handles model loading,
preprocessing, and graceful degradation when no character is detected.
"""

import cv2
import numpy as np
import easyocr

from config import OCR_LANGUAGES, OCR_CONFIDENCE_THRESHOLD


class OCREngine:
    """Lazy-loading EasyOCR wrapper optimised for single-character recognition."""

    def __init__(self):
        self._reader = None

    def _ensure_loaded(self):
        if self._reader is None:
            print("[OCR] Loading EasyOCR model (first run may download weights)...")
            self._reader = easyocr.Reader(OCR_LANGUAGES, gpu=False, verbose=False)
            print("[OCR] Model loaded.")

    @staticmethod
    def _preprocess_roi(roi):
        """Prepare a cropped shape ROI for OCR."""
        # Convert to grayscale if needed.
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Resize small ROIs up so OCR has more pixels to work with.
        h, w = gray.shape[:2]
        if h < 64 or w < 64:
            scale = max(64 / h, 64 / w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Denoise and threshold.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return gray

    def recognize(self, frame, bounding_rect):
        """Run OCR on the region defined by *bounding_rect* in *frame*.

        Parameters
        ----------
        frame : np.ndarray
            The full BGR camera frame.
        bounding_rect : tuple
            (x, y, w, h) bounding rectangle of the detected shape.

        Returns
        -------
        (character, confidence) : tuple
            *character* is a string (e.g. ``"A"``) or ``"???"`` if nothing was
            recognised.  *confidence* is a float between 0 and 1.
        """
        self._ensure_loaded()

        x, y, w, h = bounding_rect
        # Add a small padding to avoid clipping the character.
        pad = 10
        fh, fw = frame.shape[:2]
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, fw)
        y2 = min(y + h + pad, fh)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "???", 0.0

        processed = self._preprocess_roi(roi)

        try:
            results = self._reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            )
        except Exception:
            return "???", 0.0

        if not results:
            return "???", 0.0

        # Pick the result with the highest confidence.
        best = max(results, key=lambda r: r[2])
        text = best[1].strip().upper()
        conf = float(best[2])

        if conf < OCR_CONFIDENCE_THRESHOLD or not text:
            return "???", conf

        # Return only the first character (we expect single chars).
        return text[0], conf
