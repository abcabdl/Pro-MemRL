"""Thread-safe runtime feedback bus for online rejection-memory closure."""

from __future__ import annotations

from collections import deque
import threading


class RuntimeFeedbackMemory:
    """Keep newest-to-oldest user feedback flags (reject=1, accept=0)."""

    def __init__(self, horizon: int = 6) -> None:
        self._horizon = max(1, int(horizon))
        self._flags: deque[int] = deque(maxlen=self._horizon)
        self._lock = threading.Lock()

    @property
    def horizon(self) -> int:
        return self._horizon

    def record_accept(self) -> None:
        with self._lock:
            self._flags.appendleft(0)

    def record_reject(self) -> None:
        with self._lock:
            self._flags.appendleft(1)

    def recent_flags(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._flags)

    def reset(self) -> None:
        with self._lock:
            self._flags.clear()


RUNTIME_FEEDBACK = RuntimeFeedbackMemory(horizon=6)

