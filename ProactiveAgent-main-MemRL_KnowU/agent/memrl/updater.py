from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_q_value(memory: dict, reward: float, *, alpha: float = 0.12) -> dict:
    old_q = float(memory.get("q_value", memory.get("reward", 0.0)) or 0.0)
    new_q = (1.0 - alpha) * old_q + alpha * float(reward)
    memory["q_value"] = max(new_q, 0.0)
    memory["reward"] = float(reward)
    memory["q_visits"] = int(memory.get("q_visits", 0) or 0) + 1
    memory["updated_at"] = utc_now()
    return memory


def update_transfer_value(
    memory: dict,
    target_key: str,
    reward: float,
    *,
    alpha: float = 0.12,
) -> dict:
    stats = memory.setdefault("transfer_stats", {})
    item = stats.setdefault(target_key, {"gate": 0.35, "visits": 0})
    old_gate = float(item.get("gate", 0.35) or 0.35)
    target_gate = max(0.0, min(1.0, (float(reward) + 1.0) / 2.0))
    item["gate"] = (1.0 - alpha) * old_gate + alpha * target_gate
    item["visits"] = int(item.get("visits", 0) or 0) + 1
    item["last_reward"] = float(reward)
    item["updated_at"] = utc_now()
    return memory
