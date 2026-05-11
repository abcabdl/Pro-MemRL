from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from typing import Any, Iterable, Sequence


PHASE_A_SYSTEM = """<Role>
You are a proactive assistant.
</Role>

<Task>
Infer only the user's latent state from time-ascending observations.
</Task>

<Format>
Return strict JSON:
{
  "signals": {
    "flow": 0.0,
    "stuck": 0.0,
    "need": 0.0,
    "accept": 0.0,
    "risk": 0.0,
    "uncertainty": 0.0,
    "progress": 0.0,
    "rejection_memory": 0.0
  }
}
</Format>

<Rules>
- Keep all scores in [0, 1] with two decimals.
- High flow means the user is making smooth progress and should rarely be interrupted.
- High stuck means progress has stalled and direct help may be useful.
- High uncertainty means you are not confident the user's current task or needs are clear.
</Rules>
"""


PHASE_G_SYSTEM = """<Role>
You are a proactive assistant proposing one candidate intervention for the current moment.
</Role>

<Task>
Generate the single best candidate intervention from the current observations, inferred signals, and personalization context.
</Task>

<Format>
Return strict JSON:
{
  "candidate": {
    "purpose": "text or null",
    "proactive_task": "text or null",
    "response": "text or null",
    "operation": "text or null"
  }
}
</Format>

<Rules>
- Return one candidate only.
- If no proactive help is warranted yet, return null for every candidate field.
- Use a strict interruption standard. Only generate a candidate when interrupting the user right now is clearly worth the context switch and attention cost.
- Use `current_signals` and `interruption_gate` to decide whether interrupting is justified at all.
- If persona, rubric, or history_summary are provided, use them as supporting evidence for whether help would be timely and low-interruption.
- Return only timely, concrete, low-interruption help. Prefer the empty candidate over speculative, generic, repetitive, or merely nice-to-have suggestions.
- Generate a candidate only when the observations show a timely opportunity to reduce current friction, coordination overhead, uncertainty, or avoidable repeated cost.
- Prefer intervention only at moments where the user first exposes a concrete problem, a clearly new sub-problem appears, or there is an obvious blocker, mistake, or repeated failed attempt.
- If the user is actively progressing, generate a candidate only when the help is clearly timely and immediately useful rather than merely plausible or generally relevant.
- Level 1 style candidates should be short probes or clarifications and should be the default when help may be useful but the stronger step is not clearly justified.
- Level 2 style candidates should be one direct actionable suggestion only when the current situation supports that stronger intervention with clear immediate payoff.
- Avoid heavy-handed content that mostly replaces the user's ongoing work unless it resolves immediate friction in a way that clearly justifies interruption.
- During one continuous local flow, avoid generating a candidate that merely restates help that would have been equally plausible one moment earlier unless the situation has meaningfully changed.
</Rules>
"""


PHASE_B_SYSTEM = """<Role>
You are a proactive assistant that evaluates one already-proposed candidate intervention.
</Role>

<Task>
Given the provided candidate intervention, do two things in one response:
1. Simulate how the user would react if they saw it right now.
2. Decide whether the assistant should actually intervene with that same candidate.
</Task>

<Format>
Return strict JSON:
{
  "simulated_reaction": {
    "rubric_scores": {
      "personal_preference": 0,
      "frequency": 0,
      "timing": 0,
      "communication": 0
    },
    "total_score": 0.0,
    "acceptance": "accept | ignore | dismiss | annoyed",
    "acceptance_confidence": 0.0,
    "flow_impact": "improved | unchanged | disrupted",
    "relevance": "highly_relevant | somewhat_relevant | irrelevant",
    "timing": "good_timing | neutral | bad_timing",
    "reasoning": "text",
    "persona_vote_summary": {
      "persona_ids": [],
      "weights": []
    }
  },
  "intervention_recommendation": {
    "should_intervene": false,
    "level": 0,
    "risk": "low | medium | high",
    "reason": "text",
    "adjustment_hint": "text"
  }
}
</Format>

<Rules>
- Treat `candidate_intervention` as fixed input. Do not rewrite it.
- Use the provided persona, rubric, and history summary when available.
- Respect `interruption_gate` when it is provided. If the interruption case is weak or uncertain, prefer silence.
- Apply a strict interruption standard. Intervene only when the immediate benefit clearly outweighs the disruption of breaking the user's current flow.
- Prefer intervention only at moments where the user first exposes a concrete problem, a clearly new sub-problem appears, or there is an obvious blocker, mistake, or repeated failed attempt.
- If you are unsure whether intervention is necessary now, prefer silence.
- Reject help that is premature, generic, speculative, redundant, replaceable by the user's normal progress, or only abstractly useful.
- Prefer lighter-touch help while the user is still progressing.
- Level must be 0, 1, or 2.
- Level 0 means silence.
- Level 1 means a short probe or clarification only.
- Level 2 means one concrete action-oriented suggestion only when the task is clear, the intervention is likely to be welcomed, and the stronger step has clear immediate payoff.
- If the likely reaction is dismiss or annoyed, prefer silence.
- Avoid content that mainly replaces the user's ongoing work unless it resolves immediate friction that clearly justifies interruption.
</Rules>
"""


PHASE_B_DATASET_SYSTEM = PHASE_B_SYSTEM


PHASE_C_SYSTEM = """<Role>
You are a proactive assistant that writes the final user-facing intervention after reviewing a simulated user reaction.
</Role>

<Task>
Generate the final intervention content.
</Task>

<Format>
Return strict JSON:
{
  "Purpose": "text",
  "Thoughts": "text",
  "Proactive Task": "text or null",
  "Response": "text or null",
  "Operation": "text or null"
}
</Format>

<Rules>
- Respect the fixed decision level.
- Level 0 must output null task/response/operation.
- Level 1 must ask a short probe or clarification.
- Level 2 must provide one concrete actionable suggestion.
- Avoid over-helping when the simulated reaction is negative.
</Rules>
"""


USER_SIMULATION_SYSTEM = """[System]
You are the user who is currently working on a computer.
You will see your recent activity history and one proactive suggestion from an AI assistant.
Respond from the user's perspective and judge the likely reaction honestly.

Return strict JSON with fields:
- acceptance: accept | ignore | dismiss | annoyed
- flow_impact: improved | unchanged | disrupted
- relevance: highly_relevant | somewhat_relevant | irrelevant
- timing: good_timing | neutral | bad_timing
- reasoning: 2-3 sentences
"""


ACCEPTANCE_VALUES = ("accept", "ignore", "dismiss", "annoyed")
FLOW_IMPACT_VALUES = ("improved", "unchanged", "disrupted")
RELEVANCE_VALUES = ("highly_relevant", "somewhat_relevant", "irrelevant")
TIMING_VALUES = ("good_timing", "neutral", "bad_timing")
RISK_VALUES = ("low", "medium", "high")


def clamp_01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def clamp_range(value: Any, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def round_probability(value: Any) -> float:
    return round(clamp_01(value) + 1e-8, 2)


def normalize_nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nil", "n/a", "na", "not needed", "no task"}:
        return None
    return text


def parse_json_payload(raw_text: Any) -> dict[str, Any]:
    if isinstance(raw_text, dict):
        return raw_text
    if not isinstance(raw_text, str):
        raise ValueError("Expected JSON string or dict payload.")
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.I)
    if fence_match:
        text = fence_match.group(1).strip()
    obj_match = re.search(r"(\{[\s\S]*\})", text)
    if obj_match:
        text = obj_match.group(1)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")
    return payload


def _to_float_time(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except Exception:
        pass
    match = re.search(r"day\s*(\d+)\s*,\s*(\d{1,2}):(\d{2})\s*([ap]m)", text, flags=re.I)
    if not match:
        return fallback
    day = int(match.group(1))
    hour = int(match.group(2))
    minute = int(match.group(3))
    ampm = match.group(4).lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return float((day - 1) * 24 * 60 + hour * 60 + minute)


def normalize_observations(observations: Sequence[dict[str, Any]] | Any) -> list[dict[str, Any]]:
    if observations is None:
        return []
    if isinstance(observations, dict):
        observations = [observations]
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(observations):
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "time": _to_float_time(item.get("time", item.get("Time")), float(idx)),
                "event": str(item.get("event", item.get("Event", ""))),
            }
        )
    return out


def observations_to_text(observations: Sequence[dict[str, Any]]) -> str:
    rows = []
    for item in normalize_observations(observations):
        rows.append(f'- [{item["time"]:.2f}] {item["event"]}')
    return "\n".join(rows)


def _keyword_ratio(events: Sequence[dict[str, Any]], keywords: Sequence[str]) -> float:
    if not events:
        return 0.0
    hit = 0
    for event in events:
        text = str(event.get("event", "")).lower()
        if any(token in text for token in keywords):
            hit += 1
    return float(hit) / float(len(events))


def _avg_gap_norm(events: Sequence[dict[str, Any]]) -> float:
    if len(events) <= 1:
        return 5.0 / 15.0
    pairs = sorted(
        [(float(item.get("time", idx)), str(item.get("event", ""))) for idx, item in enumerate(events)],
        key=lambda pair: pair[0],
    )
    gaps = [max(0.0, pairs[idx + 1][0] - pairs[idx][0]) for idx in range(len(pairs) - 1)]
    if not gaps:
        return 5.0 / 15.0
    return clamp_01((sum(gaps) / len(gaps)) / 15.0)


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    values = [max(1e-6, float(score)) for score in scores]
    total = sum(values)
    return [value / total for value in values]


def _normalized_entropy(probs: Sequence[float]) -> float:
    values = _normalize_scores(probs)
    entropy = -sum(prob * math.log(prob) for prob in values)
    max_entropy = math.log(max(2, len(values)))
    return clamp_01(entropy / max_entropy)


def _domain_probs(observations: Sequence[dict[str, Any]]) -> list[float]:
    coding_keywords = (
        "code",
        "coding",
        "python",
        "java",
        "javascript",
        "typescript",
        "bug",
        "debug",
        "traceback",
        "pytest",
        "terminal",
        "api",
        "sql",
        "vscode",
        "visual studio code",
    )
    writing_keywords = (
        "document",
        "article",
        "blog",
        "report",
        "email",
        "outlook",
        "notes",
        "markdown",
        "essay",
        "draft",
        "summary",
        "research",
        "writing",
    )
    other_keywords = (
        "browser",
        "search",
        "google",
        "bing",
        "youtube",
        "amazon",
        "calendar",
        "shopping",
        "appointment",
        "notification",
    )
    coding_score = 1.0
    writing_score = 1.0
    other_score = 1.0
    for item in observations:
        text = str(item.get("event", "")).lower()
        coding_score += sum(1.0 for token in coding_keywords if token in text)
        writing_score += sum(1.0 for token in writing_keywords if token in text)
        other_score += sum(1.0 for token in other_keywords if token in text)
    return _normalize_scores((coding_score, writing_score, other_score))


def infer_domain(observations: Sequence[dict[str, Any]]) -> str:
    probs = _domain_probs(observations)
    return ("coding", "writing", "other")[max(range(len(probs)), key=probs.__getitem__)]


def _scene_entropy_from_distributions(
    observations: Sequence[dict[str, Any]],
    *,
    need: float,
    accept: float,
    risk: float,
) -> float:
    domain_entropy = _normalized_entropy(_domain_probs(observations))
    need_entropy = _normalized_entropy((clamp_01(need), clamp_01(1.0 - need)))
    accept_entropy = _normalized_entropy((clamp_01(accept), clamp_01(1.0 - accept)))
    joint_accept = clamp_01(need) * clamp_01(accept)
    level_entropy = _normalized_entropy(
        (
            max(1e-6, 1.0 - joint_accept),
            max(1e-6, joint_accept * (0.55 + 0.45 * clamp_01(risk))),
            max(1e-6, joint_accept * max(0.10, 1.0 - clamp_01(risk))),
        )
    )
    return clamp_01((domain_entropy + need_entropy + accept_entropy + level_entropy) / 4.0)


def _is_progress_event(text: str) -> bool:
    lower = text.lower()
    progress_keywords = (
        "implemented",
        "fixed",
        "resolved",
        "success",
        "saved",
        "completed",
        "finished",
        "run passed",
        "build passed",
        "committed",
        "merged",
    )
    weak_progress = ("type", "typing", "write", "coding", "code", "edit", "implement", "debug")
    negative = ("error", "fail", "exception", "traceback", "stuck", "blocked")
    if any(token in lower for token in progress_keywords):
        return True
    return any(token in lower for token in weak_progress) and not any(token in lower for token in negative)


def _infer_rejection_memory(
    observations: Sequence[dict[str, Any]],
    *,
    switch_ratio: float,
    idle_ratio: float,
) -> float:
    rejection_ratio = _keyword_ratio(
        observations,
        (
            "dismiss",
            "ignored",
            "ignore",
            "decline",
            "declined",
            "not now",
            "no thanks",
            "annoyed",
            "interrupt",
            "interruption",
            "later",
        ),
    )
    acceptance_ratio = _keyword_ratio(
        observations,
        (
            "accepted",
            "is accepted: true",
            "thanks assistant",
            "helped",
            "resolved by assistant",
        ),
    )
    assistant_ratio = _keyword_ratio(observations, ("assistant", "agent", "proactiveagent"))
    return clamp_01(
        0.12
        + 0.48 * rejection_ratio
        + 0.16 * assistant_ratio
        + 0.10 * switch_ratio
        + 0.10 * idle_ratio
        - 0.28 * acceptance_ratio
    )


def infer_signals(
    observations: Sequence[dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    observations = normalize_observations(observations)
    overrides = overrides or {}

    typing_ratio = _keyword_ratio(
        observations,
        ("type", "typing", "write", "coding", "code", "edit", "implement", "debug"),
    )
    switch_ratio = _keyword_ratio(
        observations,
        ("switch", "tab", "window", "navigate", "open", "alt-tab"),
    )
    idle_ratio = _keyword_ratio(
        observations,
        ("idle", "inactive", "pause", "waiting", "no specific action", "afk"),
    )
    error_ratio = _keyword_ratio(
        observations,
        ("error", "fail", "exception", "traceback", "stuck", "issue", "warning"),
    )
    search_ratio = _keyword_ratio(
        observations,
        ("search", "google", "bing", "docs", "documentation", "stackoverflow", "github issue"),
    )
    gap_norm = _avg_gap_norm(observations)

    flow = clamp_01(0.32 + 0.48 * typing_ratio - 0.22 * switch_ratio - 0.18 * idle_ratio - 0.08 * error_ratio)
    risk = clamp_01(0.16 + 0.34 * error_ratio + 0.18 * switch_ratio + 0.12 * idle_ratio)
    progress = clamp_01(0.22 + 0.55 * typing_ratio - 0.28 * error_ratio - 0.12 * idle_ratio)
    rejection_memory = clamp_01(
        overrides.get(
            "rejection_memory",
            overrides.get(
                "lambda_rej",
                _infer_rejection_memory(
                    observations,
                    switch_ratio=switch_ratio,
                    idle_ratio=idle_ratio,
                ),
            ),
        )
    )

    if not observations:
        stuck = 0.50
    else:
        sorted_obs = sorted(observations, key=lambda item: float(item.get("time", 0.0)))
        latest_time = float(sorted_obs[-1]["time"])
        first_time = float(sorted_obs[0]["time"])
        span = max(1.0, latest_time - first_time, float(len(sorted_obs) - 1))
        last_progress = None
        for item in sorted_obs:
            if _is_progress_event(str(item.get("event", ""))):
                last_progress = float(item["time"])
        time_since_progress = span if last_progress is None else max(0.0, latest_time - last_progress)
        stall_ratio = clamp_01(time_since_progress / max(2.0, 0.55 * span + 1.0))
        stuck = clamp_01(
            0.08 + 0.52 * stall_ratio + 0.22 * error_ratio + 0.12 * idle_ratio + 0.08 * switch_ratio - 0.22 * progress
        )

    need = clamp_01(
        overrides.get(
            "need",
            overrides.get(
                "p_need",
                0.10 + 0.45 * stuck + 0.15 * error_ratio + 0.15 * search_ratio + 0.10 * (1.0 - flow) + 0.05 * gap_norm,
            ),
        )
    )
    base_accept = clamp_01(
        overrides.get(
            "accept",
            overrides.get(
                "p_accept",
                0.12 + 0.52 * need + 0.12 * stuck - 0.20 * flow - 0.14 * risk - 0.18 * rejection_memory,
            ),
        )
    )
    uncertainty = clamp_01(
        overrides.get(
            "uncertainty",
            overrides.get(
                "epsilon_agent",
                _scene_entropy_from_distributions(observations, need=need, accept=base_accept, risk=risk),
            ),
        )
    )
    accept = clamp_01(base_accept * (1.0 - 0.18 * uncertainty))

    return {
        "flow": round_probability(overrides.get("flow", overrides.get("f_flow", flow))),
        "stuck": round_probability(overrides.get("stuck", overrides.get("d_stuck", stuck))),
        "need": round_probability(need),
        "accept": round_probability(accept),
        "risk": round_probability(overrides.get("risk", overrides.get("r_risk", risk))),
        "uncertainty": round_probability(uncertainty),
        "progress": round_probability(overrides.get("progress", overrides.get("progress_score", progress))),
        "rejection_memory": round_probability(rejection_memory),
    }


def _explicit_help_request_signal(observations: Sequence[dict[str, Any]]) -> float:
    return _keyword_ratio(
        observations,
        (
            "help",
            "how to",
            "how do",
            "how can",
            "why does",
            "why is",
            "fix",
            "traceback",
            "error",
            "exception",
            "issue",
            "problem",
            "question",
            "?",
        ),
    )


def _concrete_blocker_signal(observations: Sequence[dict[str, Any]]) -> float:
    return _keyword_ratio(
        observations,
        (
            "traceback",
            "exception",
            "error",
            "failed",
            "fail",
            "not working",
            "does not work",
            "doesn't work",
            "blocked",
            "cannot",
            "can't",
            "missing import",
            "module not found",
            "syntax error",
            "undefined",
            "warning",
        ),
    )


def strict_interruption_gate(
    *,
    observations: Sequence[dict[str, Any]],
    signals: dict[str, Any] | None = None,
    min_need: float = 0.30,
    gate_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_observations = normalize_observations(observations)
    normalized_signals = infer_signals(normalized_observations, signals or {})

    flow = clamp_01(normalized_signals.get("flow", 0.0))
    need = clamp_01(normalized_signals.get("need", 0.0))
    risk = clamp_01(normalized_signals.get("risk", 0.0))
    explicit_request_signal = clamp_01(_explicit_help_request_signal(normalized_observations))
    concrete_blocker_signal = clamp_01(_concrete_blocker_signal(normalized_observations))
    evidence = clamp_01(max(explicit_request_signal, concrete_blocker_signal, normalized_signals.get("evidence", 0.0)))

    base_inputs = {
        "need": need,
        "flow": flow,
        "risk": risk,
        "evidence": evidence,
    }
    thresholds = {
        "need": float(min_need),
        "flow": 0.68,
        "risk": 0.72,
        "evidence": 0.35,
        "evidence_high": 0.60,
    }
    memory_signal_delta: dict[str, float] = {}
    memory_threshold_delta: dict[str, float] = {}
    if isinstance(gate_prior, dict):
        raw_signal_delta = gate_prior.get("recommended_signal_delta", {})
        raw_threshold_delta = gate_prior.get("recommended_threshold_delta", {})
        if isinstance(raw_signal_delta, dict):
            memory_signal_delta = {
                key: float(raw_signal_delta.get(key, 0.0) or 0.0)
                for key in ("need", "flow", "risk", "evidence")
            }
        if isinstance(raw_threshold_delta, dict):
            memory_threshold_delta = {
                key: float(raw_threshold_delta.get(key, 0.0) or 0.0)
                for key in ("need", "flow", "risk", "evidence")
            }

    need = clamp_01(need + memory_signal_delta.get("need", 0.0))
    # Flow is a direct estimate of the current user's state, so memory is allowed
    # to tune its threshold but not to rewrite the observed flow aggressively.
    flow = clamp_01(flow + 0.25 * memory_signal_delta.get("flow", 0.0))
    risk = clamp_01(risk + memory_signal_delta.get("risk", 0.0))
    evidence = clamp_01(evidence + memory_signal_delta.get("evidence", 0.0))

    thresholds["need"] = max(0.12, min(0.50, thresholds["need"] + memory_threshold_delta.get("need", 0.0)))
    thresholds["flow"] = max(0.52, min(0.86, thresholds["flow"] + memory_threshold_delta.get("flow", 0.0)))
    thresholds["risk"] = max(0.50, min(0.90, thresholds["risk"] + memory_threshold_delta.get("risk", 0.0)))
    thresholds["evidence"] = max(0.20, min(0.60, thresholds["evidence"] + memory_threshold_delta.get("evidence", 0.0)))
    thresholds["evidence_high"] = max(thresholds["evidence"], min(0.85, thresholds["evidence"] + 0.25))

    immediate_benefit = clamp_01(
        max(
            0.65 * need + 0.45 * evidence,
            evidence,
        )
    )
    interruption_cost = clamp_01(0.58 * flow + 0.42 * risk)
    benefit_margin = round(immediate_benefit - interruption_cost, 4)
    strong_flow = flow >= thresholds["flow"]

    if need < thresholds["need"] and evidence < thresholds["evidence"]:
        allow_interruption = False
        reason = "prefilter_low_need"
    elif strong_flow and evidence < thresholds["evidence"]:
        allow_interruption = False
        reason = "strong_flow_no_request"
    elif risk >= thresholds["risk"] and evidence < thresholds["evidence_high"]:
        allow_interruption = False
        reason = "high_interrupt_risk"
    else:
        allow_interruption = True
        reason = "gate_pass"

    recommended_level = 0
    if allow_interruption:
        if evidence >= 0.75 and benefit_margin >= 0.12 and risk <= 0.45:
            recommended_level = 2
        else:
            recommended_level = 1

    return {
        "allow_interruption": allow_interruption,
        "reason": reason,
        "recommended_level": recommended_level,
        "strong_flow": strong_flow,
        "explicit_request_signal": round_probability(explicit_request_signal),
        "concrete_blocker_signal": round_probability(concrete_blocker_signal),
        "evidence": round_probability(evidence),
        "immediate_benefit": round_probability(immediate_benefit),
        "interruption_cost": round_probability(interruption_cost),
        "benefit_margin": benefit_margin,
        "gate_inputs": {key: round_probability(value) for key, value in base_inputs.items()},
        "calibrated_gate_inputs": {
            "need": round_probability(need),
            "flow": round_probability(flow),
            "risk": round_probability(risk),
            "evidence": round_probability(evidence),
        },
        "gate_thresholds": {key: round(value, 4) for key, value in thresholds.items()},
        "thresholds": {key: round(value, 4) for key, value in thresholds.items()},
        "memory_signal_delta": {key: round(value, 4) for key, value in memory_signal_delta.items()},
        "memory_threshold_delta": {key: round(value, 4) for key, value in memory_threshold_delta.items()},
    }


def infer_commitment_level(final_output: dict[str, Any]) -> int:
    task = normalize_nullable_text(final_output.get("Proactive Task") or final_output.get("proactive_task"))
    response = normalize_nullable_text(final_output.get("Response") or final_output.get("response"))
    operation = normalize_nullable_text(final_output.get("Operation") or final_output.get("operation"))
    if task is None and response is None and operation is None:
        return 0
    combined = " ".join(filter(None, [task, response])).lower()
    if "?" in (response or ""):
        return 1
    if any(token in combined for token in ("clarify", "confirm", "would you like", "do you want", "can you share")):
        return 1
    return 2


def normalize_candidate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    return {
        "purpose": normalize_nullable_text(payload.get("purpose", payload.get("Purpose"))),
        "proactive_task": normalize_nullable_text(
            payload.get("proactive_task", payload.get("Proactive Task", payload.get("Proactive_Task")))
        ),
        "response": normalize_nullable_text(payload.get("response", payload.get("Response"))),
        "operation": normalize_nullable_text(payload.get("operation", payload.get("Operation"))),
    }


def normalize_simulation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    rubric_payload = payload.get("rubric_scores", {}) if isinstance(payload.get("rubric_scores"), dict) else {}
    acceptance = str(payload.get("acceptance", "ignore")).strip().lower()
    if acceptance not in ACCEPTANCE_VALUES:
        acceptance = "ignore"
    flow_impact = str(payload.get("flow_impact", "unchanged")).strip().lower()
    if flow_impact not in FLOW_IMPACT_VALUES:
        flow_impact = "unchanged"
    relevance = str(payload.get("relevance", "somewhat_relevant")).strip().lower()
    if relevance not in RELEVANCE_VALUES:
        relevance = "somewhat_relevant"
    timing = str(payload.get("timing", "neutral")).strip().lower()
    if timing not in TIMING_VALUES:
        timing = "neutral"
    personal_preference_score = int(clamp_range(rubric_payload.get("personal_preference", payload.get("personal_preference_score", 0)), 0, 1))
    frequency_score = int(clamp_range(rubric_payload.get("frequency", payload.get("frequency_score", 0)), 0, 1))
    timing_score = int(clamp_range(rubric_payload.get("timing", payload.get("timing_score", 0)), 0, 1))
    communication_score = int(clamp_range(rubric_payload.get("communication", payload.get("communication_score", 0)), 0, 1))
    if not rubric_payload:
        personal_preference_score = 1 if relevance == "highly_relevant" else 0
        frequency_score = 0 if acceptance in {"dismiss", "annoyed"} and timing == "bad_timing" else 1
        timing_score = 1 if timing in {"good_timing", "neutral"} and flow_impact != "disrupted" else 0
        communication_score = 0 if acceptance == "annoyed" else 1
    total_score = float(payload.get("total_score", personal_preference_score + frequency_score + timing_score + communication_score))
    persona_vote_summary = payload.get("persona_vote_summary", {}) if isinstance(payload.get("persona_vote_summary"), dict) else {}
    return {
        "rubric_scores": {
            "personal_preference": personal_preference_score,
            "frequency": frequency_score,
            "timing": timing_score,
            "communication": communication_score,
        },
        "total_score": total_score,
        "acceptance": acceptance,
        "acceptance_confidence": round_probability(payload.get("acceptance_confidence", 0.50)),
        "flow_impact": flow_impact,
        "relevance": relevance,
        "timing": timing,
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "persona_vote_summary": {
            "persona_ids": list(persona_vote_summary.get("persona_ids", [])),
            "weights": [float(value) for value in persona_vote_summary.get("weights", [])],
        },
    }


def normalize_decision(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    should_intervene = payload.get("should_intervene", False)
    if isinstance(should_intervene, str):
        should_intervene = should_intervene.strip().lower() in {"1", "true", "yes"}
    else:
        should_intervene = bool(should_intervene)
    level = int(clamp_range(payload.get("level", 0), 0, 2))
    risk = str(payload.get("risk", "medium")).strip().lower()
    if risk not in RISK_VALUES:
        risk = "medium"
    if not should_intervene:
        level = 0
    if level == 0:
        should_intervene = False
    return {
        "should_intervene": bool(should_intervene),
        "level": level,
        "risk": risk,
        "reason": str(payload.get("reason", "")).strip(),
    }


def _strip_trailing_punctuation(text: str) -> str:
    return text.rstrip(" .!?")


def _lowercase_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _sanitize_thoughts(text: str) -> str:
    if not text:
        return ""
    sanitized = re.sub(r"\s*\(\d{1,2}:\d{2}:\d{2}\)", "", text)
    sanitized = re.sub(r"\bat\s+\d{1,2}:\d{2}:\d{2}\b", "", sanitized, flags=re.I)
    sanitized = re.sub(r"\b\d{1,2}:\d{2}:\d{2}\b", "", sanitized)
    sanitized = re.sub(r"\s+([,.;:])", r"\1", sanitized)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    return sanitized.strip(" ,")


def _looks_like_probe(text: str | None) -> bool:
    if not text:
        return False
    lower = text.lower()
    return "?" in text or any(token in lower for token in ("would you like", "do you want", "can you share", "could you"))


def _make_probe_task(candidate: dict[str, Any]) -> str:
    candidate_task = normalize_nullable_text(candidate.get("proactive_task"))
    if not candidate_task:
        return "Clarify whether the user wants help right now."
    return f"Clarify whether the user wants help with {_lowercase_first(_strip_trailing_punctuation(candidate_task))}."


def _make_probe_response(candidate: dict[str, Any]) -> str:
    candidate_response = normalize_nullable_text(candidate.get("response"))
    if _looks_like_probe(candidate_response):
        return candidate_response or "Would a quick clarification question help right now?"
    candidate_task = normalize_nullable_text(candidate.get("proactive_task"))
    if candidate_task:
        return f"Would it help if I {_lowercase_first(_strip_trailing_punctuation(candidate_task))}?"
    return "Would a quick clarification question help right now?"


def _make_level2_response(candidate: dict[str, Any], response: str | None, task: str | None) -> str:
    for text in (response, normalize_nullable_text(candidate.get("response"))):
        if text and "?" not in text:
            return text
    action = normalize_nullable_text(task) or normalize_nullable_text(candidate.get("proactive_task"))
    if action:
        return f"I can {_lowercase_first(_strip_trailing_punctuation(action))} now."
    return "I have one concrete next step suggestion."


def build_final_output(
    *,
    decision: dict[str, Any],
    candidate: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload or {}
    candidate = normalize_candidate(candidate)
    decision = normalize_decision(decision)
    should = bool(decision["should_intervene"])
    level = int(decision["level"])
    purpose = normalize_nullable_text(payload.get("Purpose", payload.get("purpose"))) or candidate.get("purpose")
    thoughts = _sanitize_thoughts(normalize_nullable_text(payload.get("Thoughts", payload.get("thoughts"))) or "")
    task = normalize_nullable_text(
        payload.get("Proactive Task", payload.get("Proactive_Task", payload.get("proactive_task")))
    )
    response = normalize_nullable_text(payload.get("Response", payload.get("response")))
    operation = normalize_nullable_text(payload.get("Operation", payload.get("operation")))

    if not should or level == 0:
        task = None
        response = None
        operation = None
    elif level == 1:
        task = _make_probe_task(candidate)
        response = response if _looks_like_probe(response) else _make_probe_response(candidate)
        operation = None
    else:
        if task is None:
            task = candidate.get("proactive_task") or "Provide one concrete next step"
        response = _make_level2_response(candidate, response, task)
        if operation is None:
            operation = candidate.get("operation")

    return {
        "Purpose": purpose or ("No intervention needed now." if level == 0 else "Support the user's current task."),
        "Thoughts": thoughts,
        "Proactive Task": task,
        "Response": response,
        "Operation": operation,
    }


def make_tuple_record(
    *,
    sample_id: str,
    observations: Sequence[dict[str, Any]],
    signals: dict[str, Any],
    candidate: dict[str, Any],
    reaction: dict[str, Any],
    final_output: dict[str, Any],
    decision: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    persona_id: str | None = None,
    domain: str | None = None,
    rubric: dict[str, Any] | None = None,
    history_summary: str | None = None,
) -> dict[str, Any]:
    normalized_reaction = normalize_simulation(reaction)
    decision = normalize_decision(
        decision
        if decision is not None
        else reaction_to_decision(normalized_reaction, signals=signals, observations=observations)
    )
    final_output = build_final_output(
        decision=decision,
        candidate=candidate,
        payload=final_output,
    )
    return {
        "sample_id": str(sample_id),
        "persona_id": persona_id,
        "domain": domain,
        "rubric": rubric or {},
        "history_summary": history_summary or "",
        "observations": normalize_observations(observations),
        "signals": infer_signals(observations, signals),
        "candidate_intervention": normalize_candidate(candidate),
        "simulated_reaction": normalized_reaction,
        "legacy_labels": {
            "y_accept": int(normalized_reaction["total_score"] >= 3.0),
            "y_need": int(decision["should_intervene"]),
            "y_star": int(decision["should_intervene"]),
        },
        "final_output": {
            **final_output,
            "Decision": normalize_decision(decision),
        },
        "metadata": metadata or {},
    }


def _distribution(votes: Iterable[str], allowed: Sequence[str]) -> dict[str, float]:
    counts = Counter(vote for vote in votes if vote in allowed)
    total = max(1, sum(counts.values()))
    return {label: round(counts.get(label, 0) / total, 2) for label in allowed}


def aggregate_simulation_votes(votes: Sequence[dict[str, Any]], rng: random.Random | None = None) -> dict[str, Any]:
    rng = rng or random.Random(0)
    normalized_votes = [normalize_simulation(vote) for vote in votes if isinstance(vote, dict)]
    if not normalized_votes:
        return {
            "reaction": normalize_simulation({}),
            "vote_distributions": {
                "acceptance": _distribution([], ACCEPTANCE_VALUES),
                "flow_impact": _distribution([], FLOW_IMPACT_VALUES),
                "relevance": _distribution([], RELEVANCE_VALUES),
                "timing": _distribution([], TIMING_VALUES),
            },
            "all_reasonings": [],
        }
    acceptance_dist = _distribution((vote["acceptance"] for vote in normalized_votes), ACCEPTANCE_VALUES)
    flow_dist = _distribution((vote["flow_impact"] for vote in normalized_votes), FLOW_IMPACT_VALUES)
    relevance_dist = _distribution((vote["relevance"] for vote in normalized_votes), RELEVANCE_VALUES)
    timing_dist = _distribution((vote["timing"] for vote in normalized_votes), TIMING_VALUES)

    reasonings = [vote["reasoning"] for vote in normalized_votes if vote["reasoning"]]
    reaction = {
        "acceptance": max(ACCEPTANCE_VALUES, key=lambda key: (acceptance_dist[key], key)),
        "acceptance_confidence": max(acceptance_dist.values()) if acceptance_dist else 0.0,
        "flow_impact": max(FLOW_IMPACT_VALUES, key=lambda key: (flow_dist[key], key)),
        "relevance": max(RELEVANCE_VALUES, key=lambda key: (relevance_dist[key], key)),
        "timing": max(TIMING_VALUES, key=lambda key: (timing_dist[key], key)),
        "reasoning": rng.choice(reasonings) if reasonings else "",
    }
    return {
        "reaction": normalize_simulation(reaction),
        "vote_distributions": {
            "acceptance": acceptance_dist,
            "flow_impact": flow_dist,
            "relevance": relevance_dist,
            "timing": timing_dist,
        },
        "all_reasonings": reasonings,
    }


def reaction_to_decision(
    reaction: dict[str, Any],
    *,
    signals: dict[str, Any] | None = None,
    observations: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reaction = normalize_simulation(reaction)
    if float(reaction.get("total_score", 0.0)) < 2.0:
        return {"should_intervene": False, "level": 0, "risk": "high", "reason": "low_rubric_score"}
    acceptance = reaction["acceptance"]
    flow_impact = reaction["flow_impact"]
    timing = reaction["timing"]
    observations = normalize_observations(observations or [])
    domain = infer_domain(observations) if observations else "other"
    normalized_signals = infer_signals(observations, signals or {})
    interruption_gate = strict_interruption_gate(observations=observations, signals=normalized_signals)
    accept_score = clamp_01(normalized_signals.get("accept", normalized_signals.get("p_accept", 0.0)))
    risk_score = clamp_01(normalized_signals.get("risk", normalized_signals.get("r_risk", 0.0)))
    uncertainty = clamp_01(normalized_signals.get("uncertainty", normalized_signals.get("epsilon_agent", 1.0)))
    acceptance_confidence = clamp_01(reaction["acceptance_confidence"])

    if acceptance in {"dismiss", "annoyed"}:
        return {"should_intervene": False, "level": 0, "risk": "high", "reason": "negative_user_reaction"}
    if acceptance == "ignore" and flow_impact == "disrupted":
        return {"should_intervene": False, "level": 0, "risk": "high", "reason": "likely_disruption"}
    if acceptance != "accept":
        return {"should_intervene": False, "level": 0, "risk": "medium", "reason": "fallback_non_accepting_reaction"}
    if accept_score < 0.30:
        return {"should_intervene": False, "level": 0, "risk": "medium", "reason": "low_accept_signal"}
    if acceptance_confidence < 0.50:
        return {"should_intervene": False, "level": 0, "risk": "medium", "reason": "low_simulation_confidence"}
    if timing == "bad_timing" and flow_impact != "improved":
        return {"should_intervene": False, "level": 0, "risk": "medium", "reason": "bad_timing_signal"}
    if not interruption_gate["allow_interruption"]:
        return {
            "should_intervene": False,
            "level": 0,
            "risk": "high" if interruption_gate["strong_flow"] else "medium",
            "reason": str(interruption_gate["reason"]),
        }

    strong_positive_reaction = timing == "good_timing" or flow_impact == "improved"
    low_action_risk = risk_score <= 0.45 and uncertainty <= 0.45
    moderate_action_risk = risk_score <= 0.60 and uncertainty <= 0.60

    if interruption_gate["recommended_level"] == 2 and strong_positive_reaction and low_action_risk:
        level = 2
        risk = "low"
        reason = "strict_gate_concrete_blocker"
    else:
        level = 1
        risk = "medium"
        if not strong_positive_reaction:
            reason = "strict_gate_light_probe_only"
        elif not moderate_action_risk:
            reason = "helpful_but_risky_or_uncertain"
        elif domain in {"code", "coding"}:
            reason = "strict_gate_code_probe"
        else:
            reason = "strict_gate_light_probe"

    return {"should_intervene": True, "level": level, "risk": risk, "reason": reason}


def build_phase_a_conversation(record: dict[str, Any]) -> dict[str, Any]:
    content = {
        "signals": record["signals"],
    }
    return {
        "conversations": [
            {"role": "system", "content": PHASE_A_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "signal_prediction",
                        "observations": observations_to_text(record["observations"]),
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "content": json.dumps(content, ensure_ascii=False)},
        ]
    }


def build_phase_g_conversation(record: dict[str, Any]) -> dict[str, Any]:
    target = {
        "candidate": record["candidate_intervention"],
    }
    payload = {
        "task": "candidate_generation",
        "observations": observations_to_text(record["observations"]),
        "current_signals": record["signals"],
    }
    if record.get("domain") is not None:
        payload["domain"] = record.get("domain")
    if record.get("persona_id") is not None:
        payload["persona"] = record.get("persona_id")
    if record.get("rubric") is not None:
        payload["rubric"] = record.get("rubric")
    if record.get("history_summary"):
        payload["history_summary"] = record.get("history_summary", "")
    if record.get("interruption_gate") is not None:
        payload["interruption_gate"] = record.get("interruption_gate")
    return {
        "conversations": [
            {"role": "system", "content": PHASE_G_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def build_phase_b_conversation(record: dict[str, Any]) -> dict[str, Any]:
    reaction = record["simulated_reaction"]
    fixed_decision = normalize_decision(
        ((record.get("final_output", {}) or {}).get("Decision"))
        or reaction_to_decision(
            reaction,
            signals=record.get("signals"),
            observations=record.get("observations"),
        )
    )
    target = {
        "simulated_reaction": {
            "rubric_scores": reaction["rubric_scores"],
            "total_score": reaction["total_score"],
            "acceptance": reaction["acceptance"],
            "acceptance_confidence": reaction["acceptance_confidence"],
            "flow_impact": reaction["flow_impact"],
            "relevance": reaction["relevance"],
            "timing": reaction["timing"],
            "reasoning": reaction["reasoning"],
            "persona_vote_summary": reaction.get("persona_vote_summary", {"persona_ids": [], "weights": []}),
        },
        "intervention_recommendation": fixed_decision,
    }
    return {
        "conversations": [
            {"role": "system", "content": PHASE_B_DATASET_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "mental_simulation",
                        "observations": observations_to_text(record["observations"]),
                        "current_signals": record["signals"],
                        "domain": record.get("domain"),
                        "persona": record.get("persona_id"),
                        "rubric": record.get("rubric"),
                        "history_summary": record.get("history_summary", ""),
                        "interruption_gate": record.get("interruption_gate"),
                        "candidate_intervention": record["candidate_intervention"],
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def build_phase_c_conversation(record: dict[str, Any]) -> dict[str, Any]:
    final_output = dict(record["final_output"])
    final_output.pop("Decision", None)
    return {
        "conversations": [
            {"role": "system", "content": PHASE_C_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "final_generation",
                        "observations": observations_to_text(record["observations"]),
                        "current_signals": record["signals"],
                        "candidate_intervention": record["candidate_intervention"],
                        "simulated_reaction": record["simulated_reaction"],
                        "fixed_decision": record["final_output"]["Decision"],
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "content": json.dumps(final_output, ensure_ascii=False)},
        ]
    }


def build_dataset_info_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def simulation_prompt_user_text(observations: Sequence[dict[str, Any]], candidate: dict[str, Any]) -> str:
    normalized_candidate = normalize_candidate(candidate)
    return (
        "[User]\n"
        "## Recent activity:\n"
        f"{observations_to_text(observations)}\n\n"
        "## AI assistant suggestion right now:\n"
        f"Purpose: {normalized_candidate['purpose']}\n"
        f"Task: {normalized_candidate['proactive_task']}\n"
        f"Response: {normalized_candidate['response']}\n"
        f"Operation: {normalized_candidate['operation']}\n\n"
        "Answer as the user."
    )
