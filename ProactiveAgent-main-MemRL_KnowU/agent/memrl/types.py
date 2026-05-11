from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _default_candidate() -> dict[str, Any]:
    return {
        "purpose": None,
        "proactive_task": None,
        "response": None,
        "operation": None,
    }


def _default_simulation() -> dict[str, Any]:
    return {
        "acceptance": "ignore",
        "acceptance_confidence": 0.0,
        "flow_impact": "unknown",
        "relevance": "unknown",
        "timing": "unknown",
        "reasoning": "",
    }


def _default_decision() -> dict[str, Any]:
    return {
        "should_intervene": False,
        "commitment_level": 0,
        "risk": "unknown",
        "reason": "",
    }


def _default_labels() -> dict[str, Any]:
    return {
        "y_need": None,
        "y_accept": None,
        "gold_should": None,
        "gold_level": None,
        "q_need": None,
        "q_accept": None,
    }


@dataclass
class EpisodeMemory:
    memory_id: str
    sample_id: str
    source: str
    domain: str
    observations: list[dict[str, Any]]
    intent_text: str
    context_family: str = "general"
    action_family: str = "no_intervention"
    outcome_family: str = "uncertain"
    candidate: dict[str, Any] = field(default_factory=_default_candidate)
    simulation: dict[str, Any] = field(default_factory=_default_simulation)
    decision: dict[str, Any] = field(default_factory=_default_decision)
    labels: dict[str, Any] = field(default_factory=_default_labels)
    reward: float = 0.0
    q_value: float = 0.0
    q_visits: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeMemory":
        return cls(
            memory_id=str(payload.get("memory_id", "")),
            sample_id=str(payload.get("sample_id", "")),
            source=str(payload.get("source", "")),
            domain=str(payload.get("domain", "general")),
            observations=list(payload.get("observations", []) or []),
            intent_text=str(payload.get("intent_text", "")),
            context_family=str(payload.get("context_family", "general")),
            action_family=str(payload.get("action_family", "no_intervention")),
            outcome_family=str(payload.get("outcome_family", "uncertain")),
            candidate=dict(_default_candidate() | dict(payload.get("candidate", {}) or {})),
            simulation=dict(_default_simulation() | dict(payload.get("simulation", {}) or {})),
            decision=dict(_default_decision() | dict(payload.get("decision", {}) or {})),
            labels=dict(_default_labels() | dict(payload.get("labels", {}) or {})),
            reward=float(payload.get("reward", 0.0) or 0.0),
            q_value=float(payload.get("q_value", payload.get("reward", 0.0)) or 0.0),
            q_visits=int(payload.get("q_visits", 0) or 0),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )
