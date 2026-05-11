from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def default_state_path() -> Path:
    return Path(__file__).resolve().parent / "runtime_user_state.json"


def load_runtime_state(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else default_state_path()
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def save_runtime_state(payload: dict[str, Any], path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else default_state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def record_runtime_feedback(reaction: str, path: str | Path | None = None) -> None:
    payload = load_runtime_state(path)
    pending = payload.get("pending_intervention")
    if not isinstance(pending, dict):
        payload["last_feedback"] = {"reaction": reaction, "timestamp": time.time()}
        save_runtime_state(payload, path)
        return
    pending["reaction"] = str(reaction)
    pending["feedback_timestamp"] = time.time()
    payload["last_feedback"] = dict(pending)
    payload.pop("pending_intervention", None)
    save_runtime_state(payload, path)
