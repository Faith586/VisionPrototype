"""Detection history with debounce logic and statistics."""

import math
import time
from dataclasses import dataclass, field
import config


@dataclass
class LogEntry:
    timestamp: float
    shape_name: str
    character: str
    confidence: float
    color: str

    @property
    def label(self):
        return f"{self.shape_name} — {self.character}"

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

    def try_log(self, shape_name, character, confidence, color, centroid):
        now = time.time()
        key = f"{shape_name}-{character}"
        if key in self._last_seen:
            prev = self._last_seen[key]
            dist = math.hypot(centroid[0] - prev["cx"], centroid[1] - prev["cy"])
            if dist < config.CENTROID_SHIFT_THRESHOLD and (now - prev["time"]) < config.DETECTION_COOLDOWN:
                return False

        self._last_seen[key] = {"cx": centroid[0], "cy": centroid[1], "time": now}

        entry = LogEntry(now, shape_name, character, confidence, color)
        self.entries.append(entry)
        if len(self.entries) > config.MAX_LOG_ENTRIES:
            self.entries.pop(0)

        self.total_count += 1
        self.shape_counts[shape_name] = self.shape_counts.get(shape_name, 0) + 1
        self.confidence_sum += confidence
        return True
