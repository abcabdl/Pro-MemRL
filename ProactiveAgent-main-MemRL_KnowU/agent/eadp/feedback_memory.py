"""Feedback memory for recency-weighted user rejection signal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Sequence

from .types import clamp_01


@dataclass(slots=True)
class FeedbackMemoryConfig:
    """Configuration for exponentially decayed rejection accumulation."""

    decay_lambda: float = 0.5
    horizon: int = 6


class FeedbackMemory:
    """Track quick-rejection history and output delta_rej in [0, 1]."""

    def __init__(self, config: FeedbackMemoryConfig | None = None) -> None:
        self.config = config or FeedbackMemoryConfig()
        self._recent_quick_rejects: deque[int] = deque(maxlen=max(1, self.config.horizon))
        self._lock = threading.Lock()

    def update(self, quick_reject: bool) -> float:
        """Append one feedback event and return updated rejection signal."""
        with self._lock:
            self._recent_quick_rejects.appendleft(1 if quick_reject else 0)
            flags = list(self._recent_quick_rejects)
        return self.compute_from_flags(flags)

    def value(self) -> float:
        """Return decayed rejection signal from internal memory."""
        with self._lock:
            flags = list(self._recent_quick_rejects)
        return self.compute_from_flags(flags)

    def recent_flags(self) -> tuple[int, ...]:
        """Return newest-to-oldest rejection flags used by delta_rej."""
        with self._lock:
            return tuple(self._recent_quick_rejects)

    def compute_from_flags(self, recent_quick_rejects: Sequence[int]) -> float:
        """Compute decayed rejection signal from explicit newest-to-oldest flags."""
        decay_lambda = clamp_01(self.config.decay_lambda)
        total = 0.0
        for idx, flag in enumerate(recent_quick_rejects[: max(1, self.config.horizon)]):
            if int(flag) == 1:
                total += decay_lambda ** (idx + 1)
        return clamp_01(total)
