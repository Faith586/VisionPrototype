"""Detection history with debounce logic and statistics."""

import math
import time
from dataclasses import dataclass
import config


@dataclass
class LogEntry:
    timestamp: float
    class_name: str      # e.g. "Triangle-X"
    letter: str           # e.g. "X"
    confidence: float
    color: str

    @property
    def label(self):
        return self.class_name

    @property
    def time_str(self):
        t = time.localtime(self.timestamp)
        return time.strftime("%H:%M:%S", t)


class DetectionLog:
    def __init__(self):
        self.entries: list[LogEntry] = []
        self.total_count = 0
        self.shape_counts: dict[str, int] = {}
        self.confidence_sum = 0.0
        self._last_seen: dict[str, dict] = {}

    @property
    def average_confidence(self):
        return self.confidence_sum / self.total_count if self.total_count else 0.0

    def try_log(self, class_name, letter, confidence, color, centroid):
        now = time.time()
        key = class_name
        if key in self._last_seen:
            prev = self._last_seen[key]
            dist = math.hypot(centroid[0] - prev["cx"], centroid[1] - prev["cy"])
            if dist < config.CENTROID_SHIFT_THRESHOLD and (now - prev["time"]) < config.DETECTION_COOLDOWN:
                return False

        self._last_seen[key] = {"cx": centroid[0], "cy": centroid[1], "time": now}

        entry = LogEntry(now, class_name, letter, confidence, color)
        self.entries.append(entry)
        if len(self.entries) > config.MAX_LOG_ENTRIES:
            self.entries.pop(0)

        self.total_count += 1
        self.shape_counts[class_name] = self.shape_counts.get(class_name, 0) + 1
        self.confidence_sum += confidence
        return True
