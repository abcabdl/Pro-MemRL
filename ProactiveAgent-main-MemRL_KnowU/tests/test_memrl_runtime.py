from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from agent.memrl.runtime import ProactiveMemRLRuntime, record_feedback_payload  # noqa: E402


def test_runtime_record_outcome_and_feedback_payload(tmp_path: Path) -> None:
    episode = {
        "memory_id": "m1",
        "sample_id": "m1",
        "source": "test",
        "domain": "coding",
        "observations": [{"event": "The user debugs code."}],
        "intent_text": "The user debugs code.",
        "candidate": {"purpose": "Fix bug", "proactive_task": "Inspect traceback", "response": "I can help.", "operation": None},
        "simulation": {"acceptance": "accept", "acceptance_confidence": 0.8, "flow_impact": "low", "relevance": "high", "timing": "good", "reasoning": ""},
        "decision": {"should_intervene": True, "commitment_level": 2, "risk": "low", "reason": ""},
        "labels": {},
        "reward": 0.5,
        "q_value": 0.5,
        "q_visits": 0,
        "created_at": "",
        "updated_at": "",
    }
    bootstrap = tmp_path / "episodes.jsonl"
    bootstrap.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")
    runtime = ProactiveMemRLRuntime()
    runtime.warm_start(str(bootstrap))
    runtime.record_outcome(["m1"], 1.0, {})
    assert runtime.memory_by_id["m1"]["q_value"] > 0.5

    record_feedback_payload(
        {"memrl": {"used_memory_ids": ["m1"], "episode": episode}},
        reaction="dismiss",
        state_dir=str(tmp_path / "state"),
        bootstrap_path=str(bootstrap),
    )
    snapshot = tmp_path / "state" / "memrl_snapshot.jsonl"
    assert snapshot.exists()
