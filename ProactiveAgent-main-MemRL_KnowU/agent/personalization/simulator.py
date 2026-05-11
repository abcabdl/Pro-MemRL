from __future__ import annotations

import math
from typing import Any, Callable

from .persona_registry import PersonaRegistry
from .types import ACCEPTANCE_LABELS, RubricScore, SimulationResult, SimulationVote


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    raw = [math.exp(item - max_value) for item in values]
    total = sum(raw) or 1.0
    return [item / total for item in raw]


def _heuristic_vote(*, signals: dict[str, Any], candidate: dict[str, Any], rubric: Any) -> RubricScore:
    flow = float(signals.get("flow", signals.get("f_flow", 0.0)))
    stuck = float(signals.get("stuck", signals.get("d_stuck", 0.0)))
    need = float(signals.get("need", signals.get("p_need", 0.0)))
    response = str(candidate.get("response") or "")
    operation = str(candidate.get("operation") or "")

    personal_preference = int(bool(candidate.get("proactive_task")) and need >= 0.35)
    frequency = int(flow < 0.75)
    timing = int(stuck >= 0.20 or need >= 0.55)
    communication = int(len(response) <= 280 and "must" not in response.lower() and operation.lower() != "force")
    total_score = float(personal_preference + frequency + timing + communication)

    if total_score >= 3.0:
        acceptance = "accept"
    elif total_score >= 2.0:
        acceptance = "ignore"
    elif communication == 0 or flow >= 0.8:
        acceptance = "annoyed"
    else:
        acceptance = "dismiss"

    confidence = min(1.0, max(0.0, total_score / 4.0))
    reasoning = (
        f"Rubric-guided heuristic vote. preference={personal_preference}, frequency={frequency}, "
        f"timing={timing}, communication={communication}. rubric={rubric.domain}."
    )
    return RubricScore(
        personal_preference=personal_preference,
        frequency=frequency,
        timing=timing,
        communication=communication,
        total_score=total_score,
        acceptance=acceptance,
        acceptance_confidence=confidence,
        reasoning=reasoning,
    )


class PersonaAwareSimulator:
    def __init__(self, registry: PersonaRegistry, *, judge_fn: Callable[..., dict[str, Any]] | None = None) -> None:
        self.registry = registry
        self.judge_fn = judge_fn

    def _run_vote(
        self,
        *,
        persona_id: str,
        observations: list[dict[str, Any]],
        signals: dict[str, Any],
        domain: str,
        candidate: dict[str, Any],
        user_model: Any,
    ) -> RubricScore:
        rubric = self.registry.get_rubric(persona_id, domain)
        if rubric is None:
            return RubricScore(0, 0, 0, 0, 0.0, "ignore", 0.0, "Missing rubric.")
        if self.judge_fn is not None:
            payload = self.judge_fn(
                observations=observations,
                signals=signals,
                domain=domain,
                persona=self.registry.get_persona(persona_id).to_dict(),
                rubric=rubric.to_dict(),
                history_summary=user_model.get_context_summary(domain),
                candidate=candidate,
            )
            scores = payload.get("rubric_scores", {})
            total_score = float(payload.get("total_score", sum(int(scores.get(key, 0)) for key in ("personal_preference", "frequency", "timing", "communication"))))
            return RubricScore(
                personal_preference=int(scores.get("personal_preference", 0)),
                frequency=int(scores.get("frequency", 0)),
                timing=int(scores.get("timing", 0)),
                communication=int(scores.get("communication", 0)),
                total_score=total_score,
                acceptance=str(payload.get("acceptance", "ignore")),
                acceptance_confidence=float(payload.get("acceptance_confidence", total_score / 4.0 if total_score else 0.0)),
                reasoning=str(payload.get("reasoning", "")),
            )
        return _heuristic_vote(signals=signals, candidate=candidate, rubric=rubric)

    def simulate(
        self,
        observations: list[dict[str, Any]],
        signals: dict[str, Any],
        domain: str,
        persona_id: str,
        candidate: dict[str, Any],
        user_model: Any,
    ) -> SimulationResult:
        neighbor_ids = self.registry.nearest_personas(persona_id, k=3)
        logits: list[float] = []
        votes: list[RubricScore] = []
        for neighbor_id in neighbor_ids:
            similarity = self.registry.similarity(persona_id, neighbor_id)
            reliability = float(user_model.neighbor_reliability.get(neighbor_id, 0.5))
            logits.append((2.0 * similarity) + (1.5 * reliability))
            votes.append(
                self._run_vote(
                    persona_id=neighbor_id,
                    observations=observations,
                    signals=signals,
                    domain=domain,
                    candidate=candidate,
                    user_model=user_model,
                )
            )
        weights = _softmax(logits)

        acceptance_probs = {label: 0.0 for label in ACCEPTANCE_LABELS}
        total_preference = 0.0
        total_frequency = 0.0
        total_timing = 0.0
        total_communication = 0.0
        persona_votes: list[SimulationVote] = []
        all_reasonings: list[str] = []

        for neighbor_id, weight, vote in zip(neighbor_ids, weights, votes):
            persona_votes.append(SimulationVote(persona_id=neighbor_id, weight=weight, rubric_scores=vote))
            acceptance_probs[vote.acceptance] += weight
            total_preference += weight * vote.personal_preference
            total_frequency += weight * vote.frequency
            total_timing += weight * vote.timing
            total_communication += weight * vote.communication
            if vote.reasoning:
                all_reasonings.append(vote.reasoning)

        aggregated_total = total_preference + total_frequency + total_timing + total_communication
        aggregated_acceptance = max(ACCEPTANCE_LABELS, key=lambda label: (acceptance_probs[label], label))
        confidence = (0.7 * acceptance_probs.get("accept", 0.0)) + (0.3 * (aggregated_total / 4.0))
        risk_flags: list[str] = []
        if float(signals.get("flow", signals.get("f_flow", 0.0))) > float(user_model.receptive_flow_threshold):
            risk_flags.append("high_flow")
        if total_frequency < 0.5:
            risk_flags.append("frequency_mismatch")
        if total_communication < 0.5:
            risk_flags.append("communication_mismatch")
        if risk_flags:
            risk = "high"
        elif aggregated_total < 3.0:
            risk = "medium"
        else:
            risk = "low"

        aggregated_scores = RubricScore(
            personal_preference=int(round(total_preference)),
            frequency=int(round(total_frequency)),
            timing=int(round(total_timing)),
            communication=int(round(total_communication)),
            total_score=aggregated_total,
            acceptance=aggregated_acceptance,
            acceptance_confidence=confidence,
            reasoning=all_reasonings[0] if all_reasonings else "",
        )
        recommendation = {
            "should_intervene": aggregated_total >= 2.0 and aggregated_acceptance in {"accept", "ignore"},
            "level": 2 if aggregated_total >= 3.25 else (1 if aggregated_total >= 2.0 else 0),
            "risk": risk,
            "reason": "persona_weighted_simulation",
            "adjustment_hint": "Reduce interruption pressure when frequency or communication scores are weak.",
        }
        return SimulationResult(
            persona_votes=persona_votes,
            aggregated_scores=aggregated_scores,
            expected_score=aggregated_total,
            acceptance_confidence=confidence,
            intervention_recommendation=recommendation,
            risk_flags=risk_flags,
        )
