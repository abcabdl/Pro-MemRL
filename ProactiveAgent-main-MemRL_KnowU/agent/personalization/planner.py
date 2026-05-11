from __future__ import annotations

import time
from typing import Any

from policy_rules import build_observation_context


class PersonalizedPlanner:
    def decide(
        self,
        signals: dict[str, Any],
        simulation_result: dict[str, Any],
        user_model: Any,
        domain: str,
        *,
        observations: list[dict[str, Any]] | None = None,
        proactive_task: str | None = None,
    ) -> dict[str, Any]:
        now = float(signals.get("timestamp", time.time()))
        time_since_last = now - float(user_model.last_intervention_time or 0.0)
        min_interval = 3600.0 / max(0.5, float(user_model.preferred_frequency))

        if time_since_last < min_interval * 0.7:
            return {"should_intervene": False, "level": 0, "reason": "frequency_constraint", "confidence": 0.0}

        flow_value = float(signals.get("flow", signals.get("f_flow", 0.0)))
        if flow_value > float(user_model.receptive_flow_threshold):
            return {"should_intervene": False, "level": 0, "reason": "flow_state", "confidence": 0.0}

        score = float(simulation_result.get("total_score", 0.0))
        context = build_observation_context(observations or [], proactive_task=proactive_task)
        if score < 2.0:
            return {"should_intervene": False, "level": 0, "reason": "low_simulation_score", "confidence": simulation_result.get("acceptance_confidence", 0.0)}

        level1_threshold = 2.0
        level2_threshold = 3.25
        recent_three = user_model.recent_interactions[-3:]
        if len(recent_three) == 3 and all(item.get("reaction") in {"dismiss", "annoyed"} for item in recent_three):
            level1_threshold = 2.5
            level2_threshold = 3.5
            if score < level1_threshold:
                return {"should_intervene": False, "level": 0, "reason": "recent_rejection_pattern", "confidence": simulation_result.get("acceptance_confidence", 0.0)}

        stuck_value = float(signals.get("stuck", signals.get("d_stuck", 0.0)))
        if score >= level2_threshold and stuck_value >= 0.45 and bool(context.get("task_clarity")):
            level = 2
        else:
            level = 1

        return {
            "should_intervene": True,
            "level": level,
            "expected_score": score,
            "confidence": float(simulation_result.get("acceptance_confidence", 0.0)),
            "reason": f"personalized_{domain}_score",
        }
