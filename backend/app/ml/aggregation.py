from collections import deque
from statistics import median
from ..config import MAX_WINDOWS, MIN_WINDOWS

def aggregate_scores(scores: list[float]) -> tuple[float, float, int]:
    if not scores:
        return 0.0, 0.0, 0
    stability = max(0.0, 100.0 - (max(scores) - min(scores)))
    return round(float(median(scores)), 1), round(stability, 1), len(scores)

class RollingAggregator:
    def __init__(self, max_windows: int = MAX_WINDOWS) -> None:
        self.scores: deque[float] = deque(maxlen=max_windows)
    def add(self, score: float) -> tuple[float, float, int]:
        self.scores.append(score)
        return aggregate_scores(list(self.scores))
    @property
    def ready(self) -> bool:
        return len(self.scores) >= MIN_WINDOWS
