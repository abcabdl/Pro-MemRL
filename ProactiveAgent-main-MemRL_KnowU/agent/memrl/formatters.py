from __future__ import annotations

import json
from typing import Any

from .schema import summarize_memory_case


def _compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return " ".join(str(value).split())


def build_observation_text(observations: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in observations or []:
        source = _compact_text(item.get("source") or item.get("app") or item.get("role"))
        event = _compact_text(item.get("event") or item.get("content") or item.get("text") or item)
        if source:
            parts.append(f"{source}: {event}")
        elif event:
            parts.append(event)
    return " | ".join(part for part in parts if part)


def build_memory_document(memory: dict[str, Any]) -> str:
    candidate = memory.get("candidate", {}) or {}
    decision = memory.get("decision", {}) or {}
    simulation = memory.get("simulation", {}) or {}
    gate_features = memory.get("gate_features", {}) or {}
    bits = [
        _compact_text(memory.get("domain")),
        _compact_text(memory.get("context_family")),
        _compact_text(memory.get("action_family")),
        _compact_text(memory.get("outcome_family")),
        _compact_text(gate_features.get("gate_reason")),
        _compact_text(memory.get("intent_text")),
        _compact_text(candidate.get("purpose")),
        _compact_text(candidate.get("proactive_task")),
        _compact_text(candidate.get("response")),
        _compact_text(decision.get("reason")),
        _compact_text(simulation.get("reasoning")),
        _compact_text(simulation.get("acceptance")),
        _compact_text(decision.get("risk")),
    ]
    return " | ".join(bit for bit in bits if bit)


def build_generation_context(prior: dict[str, Any]) -> str:
    value_aware = prior.get("value_aware_examples", {}) if isinstance(prior.get("value_aware_examples"), dict) else {}
    payload = {
        "current_context_family": prior.get("current_context_family", "general"),
        "preferred_level": prior.get("preferred_level", 0),
        "preferred_action_families": prior.get("preferred_action_families", []),
        "disallowed_action_families": prior.get("disallowed_action_families", []),
        "generation_recommendation": prior.get("generation_recommendation", {}),
        "intervene_memory_value": prior.get("intervene_memory_value", 0.0),
        "abstain_memory_value": prior.get("abstain_memory_value", 0.0),
        "positive_patterns": prior.get("positive_patterns", []),
        "negative_patterns": prior.get("negative_patterns", []),
        "avoid_patterns": prior.get("avoid_patterns", []),
        "value_aware_examples": {
            "helpful_positive": [summarize_memory_case(item) for item in value_aware.get("helpful_positive", [])],
            "correct_abstain": [summarize_memory_case(item) for item in value_aware.get("correct_abstain", [])],
            "bad_intervention": [summarize_memory_case(item) for item in value_aware.get("bad_intervention", [])],
            "missed_help": [summarize_memory_case(item) for item in value_aware.get("missed_help", [])],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def build_simulation_context(prior: dict[str, Any]) -> str:
    payload = {
        "current_context_family": prior.get("current_context_family", "general"),
        "candidate_action_family": prior.get("candidate_action_family", "no_intervention"),
        "historical_accept_rate": prior.get("historical_accept_rate", 0.0),
        "historical_dismiss_rate": prior.get("historical_dismiss_rate", 0.0),
        "historical_annoy_rate": prior.get("historical_annoy_rate", 0.0),
        "support_cases": [summarize_memory_case(item) for item in prior.get("support_cases", [])],
        "risk_cases": [summarize_memory_case(item) for item in prior.get("risk_cases", [])],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_decision_context(prior: dict[str, Any]) -> str:
    payload = {
        "current_context_family": prior.get("current_context_family", "general"),
        "candidate_action_family": prior.get("candidate_action_family", "no_intervention"),
        "intervene_memory_value": prior.get("intervene_memory_value", 0.0),
        "abstain_memory_value": prior.get("abstain_memory_value", 0.0),
        "memory_level_mode": prior.get("memory_level_mode", 0),
        "historical_reject_risk": prior.get("historical_reject_risk", 0.0),
        "memory_recommendation": prior.get("memory_recommendation", {}),
        "support_cases": [summarize_memory_case(item) for item in prior.get("intervene_memories", [])],
        "risk_cases": [summarize_memory_case(item) for item in prior.get("abstain_memories", [])],
    }
    return json.dumps(payload, ensure_ascii=False)
