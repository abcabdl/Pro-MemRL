from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from agent.memrl.retriever import MemRLRetriever  # noqa: E402
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


def test_runtime_updates_cross_user_transfer_gate_with_q_formula(tmp_path: Path) -> None:
    source_memory = {
        "memory_id": "knowu-profile-task-student-battery_saver-v0",
        "sample_id": "profile_task::student::battery_saver::v0",
        "source": "test",
        "domain": "mobile_routine",
        "observations": [{"event": "profile:student task_family:battery_saver"}],
        "intent_text": "student battery saver",
        "context_family": "knowu_profile_task_battery_saver",
        "action_family": "knowu_profile_task_battery_saver",
        "candidate": {"purpose": "Battery saver", "proactive_task": "Enable battery saver", "response": "OK", "operation": None},
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 2},
        "labels": {"profile_id": "student", "task_family": "battery_saver"},
        "reward": 0.2,
        "q_value": 0.2,
        "q_visits": 0,
        "created_at": "",
        "updated_at": "",
    }
    bootstrap = tmp_path / "episodes.jsonl"
    bootstrap.write_text(json.dumps(source_memory, ensure_ascii=False) + "\n", encoding="utf-8")

    runtime = ProactiveMemRLRuntime(alpha=0.12)
    runtime.warm_start(str(bootstrap))
    runtime.record_outcome(
        ["knowu-profile-task-student-battery_saver-v0"],
        1.0,
        {"labels": {"profile_id": "developer", "task_family": "battery_saver"}},
    )

    updated = runtime.memory_by_id["knowu-profile-task-student-battery_saver-v0"]
    transfer_item = updated["transfer_stats"]["developer::battery_saver"]
    assert transfer_item["gate"] == 0.428
    assert transfer_item["visits"] == 1
    assert updated["q_value"] == 0.2


def test_runtime_updates_same_user_q_without_transfer_gate(tmp_path: Path) -> None:
    source_memory = {
        "memory_id": "knowu-profile-task-grandma-battery_saver-v0",
        "sample_id": "profile_task::grandma::battery_saver::v0",
        "source": "test",
        "domain": "mobile_routine",
        "observations": [{"event": "profile:grandma task_family:battery_saver"}],
        "intent_text": "grandma battery saver",
        "context_family": "knowu_profile_task_battery_saver",
        "action_family": "knowu_profile_task_battery_saver",
        "candidate": {"purpose": "Battery saver", "proactive_task": "Enable battery saver", "response": "OK", "operation": None},
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 2},
        "labels": {"profile_id": "grandma", "task_family": "battery_saver"},
        "reward": 0.2,
        "q_value": 0.2,
        "q_visits": 0,
        "created_at": "",
        "updated_at": "",
    }
    bootstrap = tmp_path / "episodes.jsonl"
    bootstrap.write_text(json.dumps(source_memory, ensure_ascii=False) + "\n", encoding="utf-8")

    runtime = ProactiveMemRLRuntime(alpha=0.12)
    runtime.warm_start(str(bootstrap))
    runtime.record_outcome(
        ["knowu-profile-task-grandma-battery_saver-v0"],
        1.0,
        {"labels": {"profile_id": "grandma", "task_family": "battery_saver"}},
    )

    updated = runtime.memory_by_id["knowu-profile-task-grandma-battery_saver-v0"]
    assert updated["q_value"] > 0.2
    assert updated["q_visits"] == 1
    assert "transfer_stats" not in updated


def test_feedback_payload_uses_chosen_memory_ids_for_q_credit(tmp_path: Path) -> None:
    chosen_memory = {
        "memory_id": "chosen",
        "sample_id": "chosen",
        "source": "test",
        "domain": "mobile_routine",
        "observations": [{"event": "profile:developer task_family:battery_saver"}],
        "intent_text": "chosen same-user evidence",
        "context_family": "knowu_profile_task_battery_saver",
        "action_family": "knowu_profile_task_battery_saver",
        "candidate": {"operation": "battery_saver"},
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 2},
        "labels": {"profile_id": "developer", "task_family": "battery_saver"},
        "reward": 0.2,
        "q_value": 0.2,
        "q_visits": 0,
        "created_at": "",
        "updated_at": "",
    }
    contrast_memory = {
        **chosen_memory,
        "memory_id": "contrast",
        "sample_id": "contrast",
        "intent_text": "contrast evidence",
    }
    bootstrap = tmp_path / "episodes.jsonl"
    bootstrap.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in [chosen_memory, contrast_memory]) + "\n",
        encoding="utf-8",
    )

    record_feedback_payload(
        {
            "memrl": {
                "chosen_memory_ids": ["chosen"],
                "used_memory_ids": ["chosen", "contrast"],
                "episode": {"labels": {"profile_id": "developer", "task_family": "battery_saver"}},
            }
        },
        reaction="accept",
        state_dir=str(tmp_path / "state"),
        bootstrap_path=str(bootstrap),
    )

    memories = {}
    for line in (tmp_path / "state" / "memrl_snapshot.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        memories[item["memory_id"]] = item
    assert memories["chosen"]["q_value"] > 0.2
    assert memories["chosen"]["q_visits"] == 1
    assert memories["contrast"]["q_value"] == 0.2
    assert memories["contrast"]["q_visits"] == 0


def test_feedback_payload_uses_chosen_memory_ids_for_transfer_credit(tmp_path: Path) -> None:
    chosen_memory = {
        "memory_id": "chosen-transfer",
        "sample_id": "profile_task::student::battery_saver::v0",
        "source": "test",
        "domain": "mobile_routine",
        "observations": [{"event": "profile:student task_family:battery_saver"}],
        "intent_text": "chosen cross-user evidence",
        "context_family": "knowu_profile_task_battery_saver",
        "action_family": "knowu_profile_task_battery_saver",
        "candidate": {"operation": "battery_saver"},
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 2},
        "labels": {"profile_id": "student", "task_family": "battery_saver"},
        "reward": 0.2,
        "q_value": 0.2,
        "q_visits": 0,
        "created_at": "",
        "updated_at": "",
    }
    contrast_memory = {
        **chosen_memory,
        "memory_id": "contrast-transfer",
        "sample_id": "profile_task::grandma::battery_saver::v0",
        "labels": {"profile_id": "grandma", "task_family": "battery_saver"},
        "intent_text": "contrast cross-user evidence",
    }
    bootstrap = tmp_path / "episodes.jsonl"
    bootstrap.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in [chosen_memory, contrast_memory]) + "\n",
        encoding="utf-8",
    )

    record_feedback_payload(
        {
            "memrl": {
                "chosen_memory_ids": ["chosen-transfer"],
                "used_memory_ids": ["chosen-transfer", "contrast-transfer"],
                "episode": {"labels": {"profile_id": "developer", "task_family": "battery_saver"}},
            }
        },
        reaction="dismiss",
        state_dir=str(tmp_path / "state"),
        bootstrap_path=str(bootstrap),
    )

    memories = {}
    for line in (tmp_path / "state" / "memrl_snapshot.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        memories[item["memory_id"]] = item
    chosen_gate = memories["chosen-transfer"]["transfer_stats"]["developer::battery_saver"]
    assert chosen_gate["gate"] < 0.35
    assert chosen_gate["visits"] == 1
    assert "transfer_stats" not in memories["contrast-transfer"]
    assert memories["contrast-transfer"]["q_value"] == 0.2


def test_directional_credit_updates_only_supporting_wrong_decision_memories(tmp_path: Path) -> None:
    intervene_memory = {
        "memory_id": "intervene-evidence",
        "sample_id": "profile_task::developer::battery_saver::v0",
        "source": "test",
        "domain": "mobile_routine",
        "observations": [{"event": "profile:developer task_family:battery_saver"}],
        "intent_text": "intervene evidence",
        "context_family": "knowu_profile_task_battery_saver",
        "action_family": "knowu_profile_task_battery_saver",
        "candidate": {"operation": "battery_saver"},
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 2},
        "labels": {"profile_id": "developer", "task_family": "battery_saver"},
        "reward": 0.8,
        "q_value": 0.8,
        "q_visits": 0,
        "created_at": "",
        "updated_at": "",
    }
    abstain_memory = {
        **intervene_memory,
        "memory_id": "abstain-evidence",
        "sample_id": "profile_task::developer::battery_saver::v1",
        "intent_text": "abstain evidence",
        "decision": {"should_intervene": False, "commitment_level": 0},
        "reward": 0.8,
        "q_value": 0.8,
    }
    bootstrap = tmp_path / "episodes.jsonl"
    bootstrap.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in [intervene_memory, abstain_memory]) + "\n",
        encoding="utf-8",
    )

    runtime = ProactiveMemRLRuntime(alpha=0.5)
    runtime.warm_start(str(bootstrap))
    runtime.record_outcome(
        ["abstain-evidence"],
        -1.0,
        {"labels": {"profile_id": "developer", "task_family": "battery_saver"}},
    )

    assert runtime.memory_by_id["intervene-evidence"]["q_value"] == 0.8
    assert runtime.memory_by_id["intervene-evidence"]["q_visits"] == 0
    assert runtime.memory_by_id["abstain-evidence"]["q_value"] == 0.0
    assert runtime.memory_by_id["abstain-evidence"]["q_visits"] == 1


def test_explicit_transfer_target_updates_cross_family_gate(tmp_path: Path) -> None:
    source_memory = {
        "memory_id": "cross-family-evidence",
        "sample_id": "profile_task::student::stress_focus_block_dnd::v0",
        "source": "test",
        "domain": "mobile_routine",
        "observations": [{"event": "profile:student task_family:stress_focus_block_dnd"}],
        "intent_text": "student focus block",
        "context_family": "knowu_profile_task_stress_focus_block_dnd",
        "action_family": "knowu_profile_task_stress_focus_block_dnd",
        "candidate": {"operation": "dnd"},
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 2},
        "labels": {"profile_id": "student", "task_family": "stress_focus_block_dnd"},
        "reward": 0.8,
        "q_value": 0.8,
        "q_visits": 0,
        "created_at": "",
        "updated_at": "",
    }
    bootstrap = tmp_path / "episodes.jsonl"
    bootstrap.write_text(json.dumps(source_memory, ensure_ascii=False) + "\n", encoding="utf-8")

    runtime = ProactiveMemRLRuntime(alpha=0.5)
    runtime.warm_start(str(bootstrap))
    runtime.record_outcome(
        ["cross-family-evidence"],
        1.0,
        {
            "labels": {
                "profile_id": "developer",
                "task_family": "execution_dnd_only_day_focus",
                "transfer_target_families": {
                    "cross-family-evidence": "execution_dnd_only_day_focus",
                },
            }
        },
    )

    updated = runtime.memory_by_id["cross-family-evidence"]
    transfer_item = updated["transfer_stats"]["developer::execution_dnd_only_day_focus"]
    assert transfer_item["gate"] == 0.675
    assert transfer_item["visits"] == 1
    assert updated["q_value"] == 0.8


def test_retriever_keeps_cross_profile_same_task_transfer_for_generation() -> None:
    wrong_task = {
        "memory_id": "wrong-task",
        "sample_id": "transfer_stress::student::stress_imminent_meeting_open_doc::v0",
        "source": "test",
        "domain": "mobile_routine",
        "observations": [{"event": "profile:student task_family:stress_imminent_meeting_open_doc"}],
        "intent_text": "meeting starts in 3 minutes open pdf",
        "context_family": "knowu_profile_task_stress_imminent_meeting_open_doc",
        "action_family": "knowu_profile_task_stress_imminent_meeting_open_doc",
        "candidate": {
            "purpose": "Open meeting PDF",
            "proactive_task": "Open the meeting PDF.",
            "response": "I can open it now.",
            "operation": "knowu.routine.stress_imminent_meeting_open_doc",
        },
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 2},
        "labels": {"profile_id": "student", "task_family": "stress_imminent_meeting_open_doc"},
        "q_value": 0.9,
    }
    current_task = {
        **wrong_task,
        "memory_id": "current-task",
        "sample_id": "transfer_stress::student::stress_focus_block_dnd::v0",
        "intent_text": "protected implementation block enable dnd",
        "context_family": "knowu_profile_task_stress_focus_block_dnd",
        "action_family": "knowu_profile_task_stress_focus_block_dnd",
        "candidate": {
            "purpose": "Protect focus block",
            "proactive_task": "Enable DND for the focus block.",
            "response": "I can enable DND now.",
            "operation": "knowu.routine.stress_focus_block_dnd",
        },
        "labels": {"profile_id": "student", "task_family": "stress_focus_block_dnd"},
    }
    retriever = MemRLRetriever(topk=8, sim_threshold=0.0)
    retriever.build([wrong_task, current_task])

    prior = retriever.retrieve_for_generation(
        [
            {
                "source": "knowu_memrl_retrieval_hint",
                "event": (
                    "profile:developer task_family:stress_focus_block_dnd "
                    "context_family:knowu_profile_task_stress_focus_block_dnd"
                ),
            },
            {
                "source": "current_context",
                "event": "A protected 90-minute implementation block has started.",
            },
        ],
        {},
    )

    assert prior["used_memory_ids"] == ["current-task"]
    assert prior["candidate_positive_examples"][0]["memory_id"] == "current-task"
    assert prior["positive_patterns"] == ["Enable DND for the focus block."]

    prior_without_profile = retriever.retrieve_for_generation(
        [
            {
                "source": "knowu_memrl_retrieval_hint",
                "event": (
                    "task_family:stress_focus_block_dnd "
                    "context_family:knowu_profile_task_stress_focus_block_dnd"
                ),
            }
        ],
        {},
    )
    assert prior_without_profile["used_memory_ids"] == ["current-task"]


def test_generation_candidates_use_only_same_context_positive_examples() -> None:
    wrong_context = {
        "memory_id": "wrong-context",
        "sample_id": "unscoped-wrong-context",
        "source": "test",
        "domain": "mobile_routine",
        "observations": [{"event": "open pdf"}],
        "intent_text": "meeting starts in 3 minutes open pdf",
        "context_family": "knowu_profile_task_stress_imminent_meeting_open_doc",
        "action_family": "knowu_profile_task_stress_imminent_meeting_open_doc",
        "candidate": {
            "purpose": "Open meeting PDF",
            "proactive_task": "Open the meeting PDF.",
            "response": "I can open it now.",
            "operation": "knowu.routine.stress_imminent_meeting_open_doc",
        },
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 2},
        "labels": {},
        "q_value": 0.9,
    }
    same_context = {
        **wrong_context,
        "memory_id": "same-context",
        "sample_id": "unscoped-same-context",
        "intent_text": "protected implementation block enable dnd",
        "context_family": "knowu_profile_task_stress_focus_block_dnd",
        "action_family": "knowu_profile_task_stress_focus_block_dnd",
        "candidate": {
            "purpose": "Protect focus block",
            "proactive_task": "Enable DND for the focus block.",
            "response": "I can enable DND now.",
            "operation": "knowu.routine.stress_focus_block_dnd",
        },
    }
    retriever = MemRLRetriever(topk=8, sim_threshold=0.0)
    retriever.build([wrong_context, same_context])

    prior = retriever.retrieve_for_generation(
        [
            {
                "source": "knowu_memrl_retrieval_hint",
                "event": "context_family:knowu_profile_task_stress_focus_block_dnd",
            },
            {
                "source": "current_context",
                "event": "A protected 90-minute implementation block has started.",
            },
        ],
        {},
    )

    assert [item["memory_id"] for item in prior["candidate_positive_examples"]] == ["same-context"]
    assert prior["preferred_action_families"] == ["knowu_profile_task_stress_focus_block_dnd"]
    assert prior["positive_patterns"] == ["Enable DND for the focus block."]
