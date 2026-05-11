from __future__ import annotations

from collections import Counter
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_level(values: list[int]) -> int:
    filtered = [int(value) for value in values if int(value) >= 0]
    if not filtered:
        return 0
    counter = Counter(filtered)
    return min(counter.most_common(1)[0][0], max(filtered))


def fuse_decision(
    *,
    signal_score: float,
    backbone_level: int,
    generation_prior: dict[str, Any],
    simulation_result: dict[str, Any],
    decision_prior: dict[str, Any],
) -> dict[str, Any]:
    intervene_memory_value = _to_float(decision_prior.get("intervene_memory_value", 0.0))
    abstain_memory_value = _to_float(decision_prior.get("abstain_memory_value", 0.0))
    acceptance = str(simulation_result.get("acceptance", "ignore")).lower()
    sim_score = _to_float(
        simulation_result.get("total_score", simulation_result.get("acceptance_confidence", 0.0)),
        0.0,
    )
    reject_risk = _to_float(
        decision_prior.get(
            "historical_reject_risk",
            simulation_result.get("historical_reject_risk", 0.0),
        ),
        0.0,
    )
    memory_level_mode = int(decision_prior.get("memory_level_mode", 0) or 0)
    preferred_level = int(generation_prior.get("preferred_level", 0) or 0)

    should_intervene = backbone_level > 0
    reason = "backbone"
    if abstain_memory_value - intervene_memory_value >= 0.15:
        should_intervene = False
        reason = "abstain_memory_margin"
    elif acceptance in {"dismiss", "dismissed", "annoyed"} and reject_risk > 0.45:
        should_intervene = False
        reason = "simulation_reject_risk"
    elif acceptance in {"accept", "accepted", "helpful"} and intervene_memory_value > 0.60:
        should_intervene = True
        reason = "positive_memory_support"
    elif signal_score + sim_score + intervene_memory_value - abstain_memory_value <= 0.15:
        should_intervene = False
        reason = "weak_combined_signal"

    candidate_level = _candidate_level([backbone_level, preferred_level, memory_level_mode])
    if not should_intervene:
        candidate_level = 0
    elif acceptance in {"dismiss", "dismissed", "annoyed"}:
        candidate_level = min(candidate_level, 1)

    return {
        "should_intervene": should_intervene,
        "level": int(candidate_level),
        "reason": reason,
        "signal_score": signal_score,
        "simulation_score": sim_score,
        "intervene_memory_value": intervene_memory_value,
        "abstain_memory_value": abstain_memory_value,
        "historical_reject_risk": reject_risk,
    }
