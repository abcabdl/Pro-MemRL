from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from agent.memrl.retriever import MemRLRetriever  # noqa: E402


def _memory(memory_id: str, *, task: str | None, should: bool, level: int, acceptance: str, q_value: float) -> dict:
    return {
        "memory_id": memory_id,
        "sample_id": memory_id,
        "source": "test",
        "domain": "coding",
        "observations": [{"event": "The user debugs Python code in VS Code."}],
        "intent_text": "The user debugs Python code in VS Code.",
        "context_family": "stuck_debug",
        "action_family": "debug_help" if task else "no_intervention",
        "outcome_family": "helpful_intervention" if should else "correct_abstain",
        "candidate": {
            "purpose": "Fix bug",
            "proactive_task": task,
            "response": "I can help.",
            "operation": None,
        },
        "simulation": {
            "acceptance": acceptance,
            "acceptance_confidence": 0.8,
            "flow_impact": "low",
            "relevance": "high",
            "timing": "good",
            "reasoning": "historical signal",
        },
        "decision": {
            "should_intervene": should,
            "commitment_level": level,
            "risk": "low",
            "reason": "historical outcome",
        },
        "labels": {},
        "reward": q_value,
        "q_value": q_value,
        "q_visits": 1,
        "created_at": "",
        "updated_at": "",
    }


def test_retrieve_generation_and_decision_priors() -> None:
    retriever = MemRLRetriever(topk=4, sim_threshold=0.0)
    retriever.build(
        [
            _memory("p1", task="Inspect the traceback.", should=True, level=2, acceptance="accept", q_value=0.9),
            _memory("p2", task="Suggest a quick diff review.", should=True, level=1, acceptance="accept", q_value=0.7),
            _memory("n1", task=None, should=False, level=0, acceptance="dismiss", q_value=0.2),
        ]
    )
    observations = [{"event": "The user debugs Python code in VS Code."}]
    signals = {"need": 0.8, "risk": 0.2}
    gen = retriever.retrieve_for_generation(observations, signals)
    sim = retriever.retrieve_for_simulation(observations, {"proactive_task": "Inspect the traceback."}, signals)
    dec = retriever.retrieve_for_decision(observations, {"proactive_task": "Inspect the traceback."}, sim, signals)
    assert gen["preferred_level"] in {1, 2}
    assert gen["preferred_action_families"]
    assert gen["candidate_positive_examples"]
    assert sim["historical_accept_rate"] > 0.0
    assert sim["support_cases"]
    assert "confidence" in dec["memory_recommendation"]
    assert dec["memory_recommendation"]["context_family"] == "stuck_debug"
    assert dec["memory_recommendation"]["should_intervene"] is True
    assert dec["memory_recommendation"]["positive_intervene_ratio"] > dec["memory_recommendation"]["positive_abstain_ratio"]


def test_active_search_close_memory_defaults_to_abstain() -> None:
    retriever = MemRLRetriever(topk=6, sim_threshold=0.0)
    memories = [
        {
            **_memory("a1", task="Suggest a better search keyword.", should=False, level=0, acceptance="dismiss", q_value=0.6),
            "context_family": "active_search",
            "action_family": "search_refine",
        },
        {
            **_memory("a2", task="Suggest a better search keyword.", should=False, level=0, acceptance="ignore", q_value=0.4),
            "context_family": "active_search",
            "action_family": "search_refine",
        },
        {
            **_memory("a3", task="Suggest a better search keyword.", should=True, level=1, acceptance="accept", q_value=0.55),
            "context_family": "active_search",
            "action_family": "search_refine",
        },
    ]
    retriever.build(memories)
    observations = [{"event": "The user is browsing search results and refining a search query in the browser."}]
    signals = {"need": 0.5, "risk": 0.2, "flow": 0.4}
    sim = retriever.retrieve_for_simulation(
        observations,
        {"proactive_task": "Suggest a better search keyword."},
        signals,
    )
    dec = retriever.retrieve_for_decision(
        observations,
        {"proactive_task": "Suggest a better search keyword."},
        sim,
        signals,
    )
    assert dec["current_context_family"] == "active_search"
    assert dec["candidate_action_family"] == "search_refine"
    assert dec["memory_recommendation"]["should_intervene"] is False
    assert dec["memory_recommendation"]["balanced_intervene_count"] == dec["memory_recommendation"]["balanced_abstain_count"]
    assert dec["memory_recommendation"]["reason"] == "balanced memory evidence does not clearly favor intervention, so defaulting to abstain"


def test_gate_prior_positive_memory_lowers_thresholds() -> None:
    retriever = MemRLRetriever(topk=4, sim_threshold=0.0)
    retriever.build(
        [
            _memory("p1", task="Inspect the traceback.", should=True, level=2, acceptance="accept", q_value=1.0),
            _memory("p2", task="Suggest a quick fix.", should=True, level=1, acceptance="accept", q_value=0.8),
        ]
    )
    prior = retriever.retrieve_for_gate(
        [{"event": "The user debugs Python code in VS Code and sees a traceback."}],
        {"need": 0.5, "risk": 0.2, "flow": 0.3},
    )
    assert prior["current_context_family"] == "stuck_debug"
    assert prior["confidence"] > 0.0
    assert prior["margin"] > 0.0
    assert prior["recommended_threshold_delta"]["need"] < 0.0
    assert prior["recommended_threshold_delta"]["evidence"] < 0.0
    assert "recommended_threshold_delta" in prior["gate_context"]


def test_gate_prior_rejection_memory_raises_thresholds() -> None:
    retriever = MemRLRetriever(topk=4, sim_threshold=0.0)
    retriever.build(
        [
            {
                **_memory("n1", task="Suggest a better search keyword.", should=True, level=1, acceptance="dismiss", q_value=-0.8),
                "context_family": "active_search",
                "action_family": "search_refine",
                "outcome_family": "false_positive",
            },
            {
                **_memory("n2", task=None, should=False, level=0, acceptance="ignore", q_value=0.7),
                "context_family": "active_search",
                "action_family": "no_intervention",
                "outcome_family": "correct_abstain",
            },
        ]
    )
    prior = retriever.retrieve_for_gate(
        [{"event": "The user searches cloud storage reviews and reads search results."}],
        {"need": 0.35, "risk": 0.3, "flow": 0.5},
    )
    assert prior["current_context_family"] == "active_search"
    assert prior["historical_reject_risk"] > 0.0
    assert prior["recommended_threshold_delta"]["need"] > 0.0
    assert prior["recommended_threshold_delta"]["evidence"] > 0.0


def test_gate_prior_downweights_teacher_stage1_negative_abstain() -> None:
    retriever = MemRLRetriever(topk=8, sim_threshold=0.0)
    memories = [
        {
            **_memory("teacher-n1", task=None, should=False, level=0, acceptance="ignore", q_value=0.25),
            "source": "teacher_stage1_weak",
        },
        {
            **_memory("teacher-n2", task=None, should=False, level=0, acceptance="ignore", q_value=0.25),
            "source": "teacher_stage1_weak",
        },
        _memory("judge-p1", task="Inspect the traceback.", should=True, level=2, acceptance="accept", q_value=1.0),
    ]
    retriever.build(memories)
    prior = retriever.retrieve_for_gate(
        [{"event": "The user debugs Python code in VS Code and sees a traceback."}],
        {"need": 0.5, "risk": 0.2, "flow": 0.3},
    )
    assert prior["margin"] > 0.0
    assert prior["recommended_threshold_delta"]["need"] < 0.0


def test_generation_prior_uses_value_aware_buckets() -> None:
    retriever = MemRLRetriever(topk=8, sim_threshold=0.0)
    retriever.build(
        [
            {**_memory("hp", task="Inspect the traceback.", should=True, level=2, acceptance="accept", q_value=0.9), "value_bucket": "helpful_positive"},
            {**_memory("ca", task=None, should=False, level=0, acceptance="ignore", q_value=0.8), "value_bucket": "correct_abstain"},
            {**_memory("bi", task="Interrupt with generic advice.", should=True, level=1, acceptance="dismiss", q_value=-0.9), "value_bucket": "bad_intervention"},
            {**_memory("mh", task=None, should=False, level=0, acceptance="accept", q_value=-0.7), "value_bucket": "missed_help"},
        ]
    )
    prior = retriever.retrieve_for_generation(
        [{"event": "The user debugs Python code in VS Code and sees a traceback."}],
        {"need": 0.7, "risk": 0.2, "flow": 0.3},
    )
    buckets = prior["value_aware_examples"]
    recommendation = prior["generation_recommendation"]
    assert buckets["helpful_positive"]
    assert buckets["correct_abstain"]
    assert buckets["bad_intervention"]
    assert buckets["missed_help"]
    assert "margin_ratio" in recommendation
    assert recommendation["balanced_intervene_count"] == recommendation["balanced_abstain_count"]
    assert "generation_recommendation" in prior["generation_context"]
    assert "value_aware_examples" in prior["generation_context"]


def test_execution_target_can_cold_start_from_stress_source_memory(monkeypatch) -> None:
    monkeypatch.setenv("KNOWU_MEMRL_TRANSFER_GATE_MODE", "default")
    monkeypatch.setenv("KNOWU_MEMRL_DISABLE_TRANSFER_GATE", "false")
    retriever = MemRLRetriever(topk=4, sim_threshold=0.0)
    memory = {
        **_memory(
            "stress-dark",
            task="Enable Dark Mode for late document reading.",
            should=True,
            level=2,
            acceptance="accept",
            q_value=0.9,
        ),
        "context_family": "knowu_profile_task_stress_late_reading_dark_mode",
        "action_family": "knowu_profile_task_stress_late_reading_dark_mode",
        "labels": {
            "profile_id": "night_creator",
            "task_family": "stress_late_reading_dark_mode",
        },
    }
    retriever.build([memory])

    observations = [
        {
            "event": (
                "profile:developer task_family:execution_battery_dark_late_doc "
                "Battery is low and the user is reading API docs in a dim room."
            )
        }
    ]
    signals = {"need": 0.7, "risk": 0.2, "flow": 0.3}
    generation = retriever.retrieve_for_generation(observations, signals)
    simulation = retriever.retrieve_for_simulation(
        observations,
        {"proactive_task": "Enable Dark Mode for late document reading."},
        signals,
    )
    decision = retriever.retrieve_for_decision(
        observations,
        {"proactive_task": "Enable Dark Mode for late document reading."},
        simulation,
        signals,
    )

    assert generation["candidate_positive_examples"]
    assert generation["generation_recommendation"]["confidence"] > 0.0
    assert decision["intervene_memories"]
    assert decision["memory_recommendation"]["should_intervene"] is True
    assert decision["transfer_target_families"]["stress-dark"] == "execution_battery_dark_late_doc"
