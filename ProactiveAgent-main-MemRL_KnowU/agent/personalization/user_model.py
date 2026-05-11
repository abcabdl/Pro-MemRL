from __future__ import annotations

import re
import time
from dataclasses import asdict
from typing import Any

from .history import summarize_recent_interactions
from .types import InteractionRecord, PersonaProfile, PersonalizedRubric, UserModelState


def parse_frequency(rubric: PersonalizedRubric) -> float:
    text = rubric.frequency
    for pattern in (r"(\d+(?:\.\d+)?)\s*times per hour", r"每小时不超过\s*(\d+(?:\.\d+)?)\s*次"):
        match = re.search(pattern, text, flags=re.I)
        if match:
            return max(0.5, float(match.group(1)))
    return 2.0


def parse_timing(rubric: PersonalizedRubric) -> dict[str, Any]:
    text = rubric.timing
    threshold = 8
    for pattern in (r"stuck.*?(\d+)\s*minutes", r"卡住超过\s*(\d+)\s*分钟"):
        match = re.search(pattern, text, flags=re.I)
        if match:
            threshold = int(match.group(1))
            break
    return {"stuck_minutes": threshold, "requires_error_signal": "error" in text.lower() or "报错" in text}


class UserModel:
    def __init__(self, persona: PersonaProfile, rubric_by_domain: dict[str, PersonalizedRubric]) -> None:
        self.persona = persona
        self.rubric_by_domain = rubric_by_domain
        coding_rubric = rubric_by_domain.get("coding") or next(iter(rubric_by_domain.values()))
        self.intervention_history: list[dict[str, Any]] = []
        self.recent_interactions: list[dict[str, Any]] = []
        self.neighbor_reliability: dict[str, float] = {persona.persona_id: 0.5}
        self.preferred_frequency = parse_frequency(coding_rubric)
        self.initial_preferred_frequency = self.preferred_frequency
        self.preferred_timing_signals = parse_timing(coding_rubric)
        self.receptive_flow_threshold = 0.4
        self.session_intervention_count = 0
        self.last_intervention_time: float | None = None

    def update_neighbor_reliability(self, persona_id: str, prediction: str, actual: str, alpha: float = 0.2) -> None:
        current = float(self.neighbor_reliability.get(persona_id, 0.5))
        target = 1.0 if prediction == actual else 0.0
        self.neighbor_reliability[persona_id] = ((1.0 - alpha) * current) + (alpha * target)

    def update(
        self,
        intervention: dict[str, Any],
        reaction: str,
        rubric_scores: dict[str, int],
        domain: str,
    ) -> None:
        timestamp = float(intervention.get("timestamp", time.time()))
        total_score = float(sum(int(rubric_scores.get(key, 0)) for key in ("personal_preference", "frequency", "timing", "communication")))
        record = InteractionRecord(
            timestamp=timestamp,
            domain=domain,
            context_signals=dict(intervention.get("signals", {})),
            candidate=dict(intervention.get("candidate", {})),
            reaction=str(reaction),
            rubric_scores={key: int(rubric_scores.get(key, 0)) for key in ("personal_preference", "frequency", "timing", "communication")},
            total_score=total_score,
        )
        self.intervention_history.append(record.to_dict())
        self.recent_interactions = self.intervention_history[-10:]
        self.session_intervention_count += 1
        self.last_intervention_time = timestamp

        if reaction == "annoyed" and int(rubric_scores.get("frequency", 0)) == 0:
            self.preferred_frequency *= 0.85
        elif reaction == "accept" and int(rubric_scores.get("frequency", 0)) == 1:
            self.preferred_frequency = min(self.preferred_frequency * 1.05, self.initial_preferred_frequency)

        recent_three = self.recent_interactions[-3:]
        if len(recent_three) == 3 and all(item.get("reaction") in {"dismiss", "annoyed"} for item in recent_three):
            self.receptive_flow_threshold = max(0.25, self.receptive_flow_threshold - 0.05)
        elif reaction == "accept" and int(rubric_scores.get("timing", 0)) == 1:
            self.receptive_flow_threshold = min(0.55, self.receptive_flow_threshold + 0.02)

    def get_context_summary(self, domain: str) -> str:
        domain_records = [item for item in self.recent_interactions if item.get("domain") == domain]
        if not domain_records:
            domain_records = self.recent_interactions
        return summarize_recent_interactions(domain_records, limit=10)

    def to_dict(self) -> dict[str, Any]:
        rubric_by_domain = {key: value.to_dict() for key, value in self.rubric_by_domain.items()}
        state = UserModelState(
            persona_id=self.persona.persona_id,
            rubric_by_domain=rubric_by_domain,
            intervention_history=list(self.intervention_history),
            recent_interactions=list(self.recent_interactions),
            neighbor_reliability=dict(self.neighbor_reliability),
            preferred_frequency=float(self.preferred_frequency),
            preferred_timing_signals=dict(self.preferred_timing_signals),
            receptive_flow_threshold=float(self.receptive_flow_threshold),
            session_intervention_count=int(self.session_intervention_count),
            last_intervention_time=self.last_intervention_time,
        )
        return asdict(state)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        persona: PersonaProfile,
        rubric_by_domain: dict[str, PersonalizedRubric],
    ) -> "UserModel":
        model = cls(persona=persona, rubric_by_domain=rubric_by_domain)
        model.intervention_history = list(payload.get("intervention_history", []))
        model.recent_interactions = list(payload.get("recent_interactions", model.intervention_history[-10:]))
        model.neighbor_reliability = {str(key): float(value) for key, value in payload.get("neighbor_reliability", {}).items()}
        model.preferred_frequency = float(payload.get("preferred_frequency", model.preferred_frequency))
        model.initial_preferred_frequency = max(model.initial_preferred_frequency, model.preferred_frequency)
        model.preferred_timing_signals = dict(payload.get("preferred_timing_signals", model.preferred_timing_signals))
        model.receptive_flow_threshold = float(payload.get("receptive_flow_threshold", model.receptive_flow_threshold))
        model.session_intervention_count = int(payload.get("session_intervention_count", 0))
        model.last_intervention_time = payload.get("last_intervention_time")
        return model
