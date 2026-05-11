from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_q_value(memory: dict, reward: float, *, alpha: float = 0.12) -> dict:
    old_q = float(memory.get("q_value", memory.get("reward", 0.0)) or 0.0)
    new_q = (1.0 - alpha) * old_q + alpha * float(reward)
    memory["q_value"] = new_q
    memory["reward"] = float(reward)
    memory["q_visits"] = int(memory.get("q_visits", 0) or 0) + 1
    memory["updated_at"] = utc_now()
    return memory
