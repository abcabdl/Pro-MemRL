from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from eval.run_memrl_eval import run_eval  # noqa: E402


def test_run_memrl_eval_smoke(tmp_path: Path) -> None:
    bootstrap_episode = {
        "memory_id": "b1",
        "sample_id": "b1",
        "source": "judge_stage2",
        "domain": "coding",
        "observations": [{"event": "The user debugs a traceback."}],
        "intent_text": "The user debugs a traceback.",
        "candidate": {"purpose": "Fix bug", "proactive_task": "Inspect traceback", "response": "I can help.", "operation": None},
        "simulation": {"acceptance": "accept", "acceptance_confidence": 0.8, "flow_impact": "low", "relevance": "high", "timing": "good", "reasoning": ""},
        "decision": {"should_intervene": True, "commitment_level": 1, "risk": "low", "reason": ""},
        "labels": {},
        "reward": 0.8,
        "q_value": 0.8,
        "q_visits": 1,
        "created_at": "",
        "updated_at": "",
    }
    bootstrap = tmp_path / "bootstrap.jsonl"
    bootstrap.write_text(json.dumps(bootstrap_episode, ensure_ascii=False) + "\n", encoding="utf-8")

    test_root = tmp_path / "test"
    gold_root = tmp_path / "gold"
    output_root = tmp_path / "pred"
    snapshot_root = tmp_path / "snapshot"
    test_root.mkdir()
    gold_root.mkdir()
    events = [
        {
            "observation": {"time": "1", "event": "The user debugs a traceback in VS Code."},
            "agent_response": {"candidate_task": ["Inspect the traceback."]},
            "task_status": True,
            "gold_decision": {"should_intervene": 1, "commitment_level": 1},
        }
    ]
    (test_root / "sample.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    (gold_root / "sample.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")

    run_eval(
        bootstrap_path=bootstrap,
        test_root=test_root,
        gold_root=gold_root,
        output_root=output_root,
        snapshot_root=snapshot_root,
    )
    pred = json.loads((output_root / "sample.json").read_text(encoding="utf-8"))
    assert pred[0]["other_infomation"]["Decision"]["commitment_level"] >= 0
    assert (snapshot_root / "memrl_snapshot.jsonl").exists()
