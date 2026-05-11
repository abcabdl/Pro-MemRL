from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PERSONA_VECTOR_DIM = 23
SUPPORTED_DOMAINS = ("coding", "writing", "other")
ACCEPTANCE_LABELS = ("accept", "ignore", "dismiss", "annoyed")


@dataclass(slots=True)
class PersonaProfile:
    persona_id: str
    personality: dict[str, str]
    coding_profile: dict[str, Any]
    writing_profile: dict[str, Any]
    behavioral_patterns: dict[str, Any]
    intervention_preference: dict[str, Any]
    vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PersonalizedRubric:
    persona_id: str
    domain: str
    personal_preference: str
    frequency: str
    timing: str
    communication_safety: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RubricScore:
    personal_preference: int
    frequency: int
    timing: int
    communication: int
    total_score: float
    acceptance: str
    acceptance_confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SimulationVote:
    persona_id: str
    weight: float
    rubric_scores: RubricScore

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rubric_scores"] = self.rubric_scores.to_dict()
        return payload


@dataclass(slots=True)
class SimulationResult:
    persona_votes: list[SimulationVote]
    aggregated_scores: RubricScore
    expected_score: float
    acceptance_confidence: float
    intervention_recommendation: dict[str, Any]
    risk_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_votes": [vote.to_dict() for vote in self.persona_votes],
            "aggregated_scores": self.aggregated_scores.to_dict(),
            "expected_score": self.expected_score,
            "acceptance_confidence": self.acceptance_confidence,
            "intervention_recommendation": dict(self.intervention_recommendation),
            "risk_flags": list(self.risk_flags),
        }


@dataclass(slots=True)
class InteractionRecord:
    timestamp: float
    domain: str
    context_signals: dict[str, float]
    candidate: dict[str, Any]
    reaction: str
    rubric_scores: dict[str, int]
    total_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UserModelState:
    persona_id: str
    rubric_by_domain: dict[str, dict[str, Any]]
    intervention_history: list[dict[str, Any]] = field(default_factory=list)
    recent_interactions: list[dict[str, Any]] = field(default_factory=list)
    neighbor_reliability: dict[str, float] = field(default_factory=dict)
    preferred_frequency: float = 2.0
    preferred_timing_signals: dict[str, Any] = field(default_factory=dict)
    receptive_flow_threshold: float = 0.4
    session_intervention_count: int = 0
    last_intervention_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_personas(path: str | Path) -> dict[str, PersonaProfile]:
    payload = _read_json(Path(path))
    out: dict[str, PersonaProfile] = {}
    for item in payload:
        persona = PersonaProfile(
            persona_id=str(item["persona_id"]),
            personality=dict(item["personality"]),
            coding_profile=dict(item["coding_profile"]),
            writing_profile=dict(item["writing_profile"]),
            behavioral_patterns=dict(item["behavioral_patterns"]),
            intervention_preference=dict(item["intervention_preference"]),
            vector=[float(v) for v in item.get("vector", [])],
        )
        out[persona.persona_id] = persona
    return out


def load_rubrics(path: str | Path) -> dict[str, dict[str, PersonalizedRubric]]:
    payload = _read_json(Path(path))
    out: dict[str, dict[str, PersonalizedRubric]] = {}
    for persona_id, by_domain in payload.items():
        out[str(persona_id)] = {}
        for domain, item in by_domain.items():
            out[str(persona_id)][str(domain)] = PersonalizedRubric(
                persona_id=str(persona_id),
                domain=str(domain),
                personal_preference=str(item["personal_preference"]),
                frequency=str(item["frequency"]),
                timing=str(item["timing"]),
                communication_safety=str(item["communication_safety"]),
            )
    return out


def load_persona_vectors(path: str | Path) -> dict[str, list[float]]:
    payload = _read_json(Path(path))
    out: dict[str, list[float]] = {}
    for persona_id, values in payload.items():
        vector = [float(v) for v in values]
        if len(vector) != PERSONA_VECTOR_DIM:
            raise ValueError(f"persona_id={persona_id} vector_dim={len(vector)} expected={PERSONA_VECTOR_DIM}")
        out[str(persona_id)] = vector
    return out
