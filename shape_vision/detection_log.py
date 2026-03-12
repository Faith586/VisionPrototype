"""
Detection history tracking and statistics.

Manages the log of recent detections with debounce logic and
per-shape-type statistics.
"""

import math
import time

from config import MAX_LOG_ENTRIES, CENTROID_SHIFT_THRESHOLD, DETECTION_COOLDOWN


class LogEntry:
    """A single logged detection event."""

    __slots__ = ("timestamp", "shape_name", "character", "confidence", "color")

    def __init__(self, timestamp, shape_name, character, confidence, color):
        self.timestamp = timestamp
        self.shape_name = shape_name
        self.character = character
        self.confidence = confidence
        self.color = color

    @property
    def label(self):
        return f"{self.shape_name} — {self.character}"

    @property
    def time_str(self):
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class DetectionLog:
    """Maintains detection history with debounce and statistics."""

    def __init__(self):
        self.entries: list[LogEntry] = []
        self.total_count = 0
        self.shape_counts: dict[str, int] = {}
        self.confidence_sum = 0.0

        # Debounce state: keyed by shape_name, stores (timestamp, centroid).
        self._last_seen: dict[str, tuple[float, tuple[int, int]]] = {}

    def try_log(self, shape_name, character, confidence, color, centroid):
        """Log a detection if it passes the debounce filter.

        Returns True if the detection was logged, False if suppressed.
        """
        now = time.time()
        key = f"{shape_name}-{character}"

        if key in self._last_seen:
            prev_time, prev_centroid = self._last_seen[key]
            dist = math.hypot(centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1])
            elapsed = now - prev_time
            if dist < CENTROID_SHIFT_THRESHOLD and elapsed < DETECTION_COOLDOWN:
                return False

        self._last_seen[key] = (now, centroid)

        entry = LogEntry(
            timestamp=now,
            shape_name=shape_name,
            character=character,
            confidence=confidence,
            color=color,
        )
        self.entries.append(entry)
        if len(self.entries) > MAX_LOG_ENTRIES:
            self.entries = self.entries[-MAX_LOG_ENTRIES:]

        self.total_count += 1
        self.shape_counts[shape_name] = self.shape_counts.get(shape_name, 0) + 1
        self.confidence_sum += confidence

        return True

    @property
    def average_confidence(self):
        if self.total_count == 0:
            return 0.0
        return self.confidence_sum / self.total_count
