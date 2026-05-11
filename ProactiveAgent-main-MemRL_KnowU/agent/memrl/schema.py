from __future__ import annotations

import re
from typing import Any


def _compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return " ".join(str(value).split())


def _joined_observation_text(observations: list[dict[str, Any]]) -> str:
    return " | ".join(
        _compact_text(item.get("event") or item.get("content") or item.get("text") or item)
        for item in observations or []
        if _compact_text(item.get("event") or item.get("content") or item.get("text") or item)
    ).lower()


def _extract_knowu_task_family(text: str) -> str | None:
    for pattern in (
        r"context_family:knowu_profile_task_([a-z0-9_]+)",
        r"context_family:knowu_([a-z0-9_]+)",
        r"action_family:knowu_profile_task_([a-z0-9_]+)",
        r"task_family:([a-z0-9_]+)",
        r"knowu\.routine\.([a-z0-9_]+)",
        r"knowu\.profile_task\.([a-z0-9_]+)",
        r"knowu\.profile_habit\.([a-z0-9_]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def infer_context_family(observations: list[dict[str, Any]], signals: dict[str, Any] | None = None) -> str:
    text = _joined_observation_text(observations)
    signals = signals or {}
    flow = float(signals.get("flow", signals.get("p_flow", 0.0)) or 0.0)
    stuck = float(signals.get("stuck", signals.get("p_stuck", 0.0)) or 0.0)
    need = float(signals.get("need", signals.get("p_need", 0.0)) or 0.0)

    knowu_family = _extract_knowu_task_family(text)
    if knowu_family:
        return f"knowu_profile_task_{knowu_family}"
    if any(token in text for token in ("traceback", "stack trace", "exception", "debug", "error")):
        return "stuck_debug"
    if any(token in text for token in ("cmd+v", "cmd+c", "copy", "paste", "copied", "pasted")) and any(
        token in text for token in ("tab", "browser", "deepseek", "kimi")
    ):
        return "copy_paste_multitab"
    if any(token in text for token in ("search", "google", "bing", "search results", "query")):
        return "active_search"
    if any(token in text for token in ("markdown", "draft", "article", "essay", "document", "summary", "write")):
        return "drafting"
    if flow >= 0.7 and stuck <= 0.3 and need <= 0.35:
        return "smooth_flow"
    if any(token in text for token in ("visual studio code", "vs code", "python", "javascript", "react", "terminal")):
        return "focused_coding"
    if any(token in text for token in ("tab", "browser", "switch")):
        return "tab_switching"
    return "general"


def infer_action_family(candidate: dict[str, Any] | None) -> str:
    payload = candidate or {}
    task = _compact_text(payload.get("proactive_task"))
    purpose = _compact_text(payload.get("purpose"))
    response = _compact_text(payload.get("response"))
    operation = _compact_text(payload.get("operation"))
    text = f"{purpose} | {task} | {response} | {operation}".lower()

    if not task and not response:
        return "no_intervention"
    knowu_family = _extract_knowu_task_family(text)
    if knowu_family:
        return f"knowu_profile_task_{knowu_family}"
    if any(token in text for token in ("shortcut", "keyboard", "tab", "browser extension", "navigate", "workflow")):
        return "workflow_tip"
    if any(token in text for token in ("debug", "traceback", "error", "bug", "fix", "validation", "stack trace")):
        return "debug_help"
    if any(token in text for token in ("search", "query", "keyword", "documentation", "tutorial", "look up")):
        return "search_refine"
    if any(token in text for token in ("rewrite", "grammar", "paragraph", "draft", "outline", "wording", "summary")):
        return "writing_help"
    if any(token in text for token in ("deadline", "schedule", "reminder", "plan", "checklist")):
        return "planning_help"
    return "direct_suggestion"


def infer_outcome_family(memory: dict[str, Any]) -> str:
    decision = memory.get("decision", {}) or {}
    labels = memory.get("labels", {}) or {}
    simulation = memory.get("simulation", {}) or {}
    should_intervene = bool(decision.get("should_intervene"))
    gold_should = labels.get("gold_should")
    reward = float(memory.get("q_value", memory.get("reward", 0.0)) or 0.0)
    acceptance = str(simulation.get("acceptance", "")).lower()

    if gold_should is not None:
        if should_intervene and not bool(gold_should):
            return "false_positive"
        if (not should_intervene) and bool(gold_should):
            return "false_negative"
    if should_intervene and reward > 0.4:
        return "helpful_intervention"
    if (not should_intervene) and reward > 0.4:
        return "correct_abstain"
    if should_intervene and acceptance in {"dismiss", "annoyed", "reject", "rejected"}:
        return "disruptive_intervention"
    if should_intervene and reward <= 0.0:
        return "low_value_intervention"
    if (not should_intervene) and reward <= 0.0:
        return "missed_help"
    return "uncertain"


def summarize_memory_case(memory: dict[str, Any]) -> dict[str, Any]:
    candidate = memory.get("candidate", {}) or {}
    decision = memory.get("decision", {}) or {}
    simulation = memory.get("simulation", {}) or {}
    return {
        "memory_id": str(memory.get("memory_id", "")),
        "context_family": str(memory.get("context_family", "general")),
        "action_family": str(memory.get("action_family", "no_intervention")),
        "outcome_family": str(memory.get("outcome_family", "uncertain")),
        "value_bucket": str(memory.get("value_bucket", "weak_uncertain")),
        "should_intervene": bool(decision.get("should_intervene", False)),
        "commitment_level": int(decision.get("commitment_level", 0) or 0),
        "acceptance": str(simulation.get("acceptance", "ignore")),
        "q_value": float(memory.get("q_value", memory.get("reward", 0.0)) or 0.0),
        "proactive_task": candidate.get("proactive_task"),
        "reason": decision.get("reason") or simulation.get("reasoning") or "",
    }


def enrich_memory_schema(memory: dict[str, Any], *, signals: dict[str, Any] | None = None) -> dict[str, Any]:
    observations = list(memory.get("observations", []) or [])
    candidate = dict(memory.get("candidate", {}) or {})
    if not memory.get("context_family"):
        memory["context_family"] = infer_context_family(observations, signals)
    if not memory.get("action_family"):
        memory["action_family"] = infer_action_family(candidate)
    if not memory.get("outcome_family"):
        memory["outcome_family"] = infer_outcome_family(memory)
    return memory
