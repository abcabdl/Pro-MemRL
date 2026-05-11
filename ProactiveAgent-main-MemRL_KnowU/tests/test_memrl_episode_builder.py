from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from dataset.build_memrl_episode_dataset import (  # noqa: E402
    build_episode_from_judge_row,
    build_episode_from_teacher_row,
    build_generation_only_from_agent_trainset,
)


def test_build_episode_from_judge_row() -> None:
    row = {
        "sample_id": 1,
        "obs": [{"Time": "10:00", "Event": "The user debugs a Python traceback in VS Code."}],
        "teacher": {
            "purpose": "Fix the bug.",
            "thoughts": "The user is stuck on an error.",
            "proactive_task": "Offer to inspect the traceback.",
            "response": "I can inspect the traceback with you.",
            "q_need": 0.9,
            "q_accept": 0.8,
        },
        "judge_votes": {"m": {"thought_need": "user stuck", "thought_accept": "helpful"}},
        "labels": {"y_need": 1, "y_accept": 1},
        "gold_decision": {"should_intervene": 1, "commitment_level": 2},
    }
    episode = build_episode_from_judge_row(row)
    assert episode is not None
    assert episode["decision"]["should_intervene"] is True
    assert episode["decision"]["commitment_level"] == 2
    assert episode["reward"] == 1.0


def test_build_episode_from_teacher_row() -> None:
    row = {
        "sample_id": 2,
        "obs": [{"Time": "10:00", "Event": "The user writes a draft in a document editor."}],
        "teacher": {
            "purpose": "Draft a section.",
            "thoughts": "The user is already in flow.",
            "proactive_task": None,
            "response": None,
            "q_need": 0.1,
            "q_accept": 0.2,
            "y_need_pred": 0,
        },
    }
    episode = build_episode_from_teacher_row(row)
    assert episode is not None
    assert episode["decision"]["should_intervene"] is False
    assert episode["candidate"]["proactive_task"] is None
    assert episode["reward"] == 0.25


def test_build_generation_only_from_agent_trainset() -> None:
    row = {
        "conversations": [
            {"role": "user", "content": '{"Observations":[{"Time":"10:00","Event":"The user edits a file in VS Code."}]}'},
            {"role": "assistant", "content": '{"Purpose":"Code editing","Thoughts":"Offer a quick review.","Proactive Task":"Review the current diff.","Response":"I can review the diff."}'},
        ]
    }
    sample = build_generation_only_from_agent_trainset(row)
    assert sample is not None
    assert sample["candidate"]["proactive_task"] == "Review the current diff."
