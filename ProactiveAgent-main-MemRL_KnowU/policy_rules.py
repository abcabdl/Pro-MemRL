from __future__ import annotations

from typing import Any, Sequence


WRITE_ARTIFACT_KEYWORDS = (
    "markdown",
    "notes",
    "draft",
    "document",
    "report",
    "essay",
    "article",
    "summary",
    "outline",
    "blog",
    "proposal",
    "brand",
    "research",
    "write",
    "writing",
)

RESEARCH_BROWSE_KEYWORDS = (
    "website",
    "page",
    "scroll",
    "click",
    "clicked",
    "search",
    "query",
    "google",
    "bing",
    "forum",
    "article",
    "review",
    "blog",
    "report",
)

HARD_BLOCKER_KEYWORDS = (
    "error",
    "exception",
    "traceback",
    "fail",
    "failed",
    "failing",
    "stuck",
    "blocked",
    "cannot",
    "can't",
    "unable",
    "syntax error",
    "compile error",
    "save error",
    "permission denied",
    "not found",
    "crash",
    "bug",
    "warning",
    "conflict",
)

HELP_REQUEST_KEYWORDS = (
    "how to",
    "help",
    "guide",
    "tutorial",
    "what does",
    "why is",
    "why does",
    "fix",
    "debug",
    "troubleshoot",
)

CODING_KEYWORDS = (
    "code",
    "coding",
    "python",
    "java",
    "ruby",
    "javascript",
    "typescript",
    "bug",
    "debug",
    "traceback",
    "github",
    "visual studio code",
    "vscode",
    "terminal",
    "api",
    "sql",
    ".py",
    ".js",
    ".ts",
    ".java",
)

WRITING_KEYWORDS = (
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
    "write",
    "writing",
)

PLACEHOLDER_TASKS = {
    "clarify the user's current need",
    "provide a concrete next-step suggestion",
}


def clamp_01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def round_probability(value: Any) -> float:
    return round(clamp_01(value) + 1e-8, 2)


def _observation_texts(observations: Sequence[dict[str, Any]]) -> list[str]:
    return [str(item.get("event", item.get("Event", ""))).lower() for item in observations]


def infer_domain(observations: Sequence[dict[str, Any]]) -> str:
    text = " ".join(_observation_texts(observations))
    coding_hits = sum(1 for token in CODING_KEYWORDS if token in text)
    writing_hits = sum(1 for token in WRITING_KEYWORDS if token in text)
    if coding_hits > writing_hits and coding_hits >= 1:
        return "coding"
    if writing_hits > coding_hits and writing_hits >= 1:
        return "writing"
    return "other"


def has_hard_blocker(observations: Sequence[dict[str, Any]]) -> bool:
    return any(any(token in text for token in HARD_BLOCKER_KEYWORDS) for text in _observation_texts(observations))


def has_explicit_help_request(observations: Sequence[dict[str, Any]]) -> bool:
    return any(any(token in text for token in HELP_REQUEST_KEYWORDS) for text in _observation_texts(observations))


def has_recent_write_artifact(observations: Sequence[dict[str, Any]], horizon: int = 6) -> bool:
    recent = list(observations[-max(1, int(horizon)) :])
    return any(any(token in text for token in WRITE_ARTIFACT_KEYWORDS) for text in _observation_texts(recent))


def has_research_browse(observations: Sequence[dict[str, Any]], horizon: int = 6) -> bool:
    recent = list(observations[-max(1, int(horizon)) :])
    texts = _observation_texts(recent)
    has_browse = any(any(token in text for token in RESEARCH_BROWSE_KEYWORDS) for text in texts)
    has_write = any(any(token in text for token in WRITE_ARTIFACT_KEYWORDS) for text in texts)
    return bool(has_browse and has_write)


def infer_task_clarity(
    observations: Sequence[dict[str, Any]],
    *,
    proactive_task: str | None = None,
) -> bool:
    task = (proactive_task or "").strip().lower()
    if task and task not in PLACEHOLDER_TASKS and len(task) >= 12:
        return True
    domain = infer_domain(observations)
    if has_hard_blocker(observations) or has_explicit_help_request(observations):
        return True
    if domain == "coding":
        return any(any(token in text for token in (".py", ".js", ".ts", ".java", "sql", "api")) for text in _observation_texts(observations))
    return has_recent_write_artifact(observations) and has_research_browse(observations)


def build_observation_context(
    observations: Sequence[dict[str, Any]],
    *,
    proactive_task: str | None = None,
) -> dict[str, Any]:
    domain = infer_domain(observations)
    return {
        "domain": domain,
        "has_hard_blocker": has_hard_blocker(observations),
        "has_explicit_help_request": has_explicit_help_request(observations),
        "has_recent_write_artifact": has_recent_write_artifact(observations),
        "has_research_browse": has_research_browse(observations),
        "task_clarity": infer_task_clarity(observations, proactive_task=proactive_task),
    }


def decide_should_intervene(signals: dict[str, float], tau: float) -> bool:
    return bool(clamp_01(signals["p_accept"]) >= float(tau) and clamp_01(signals["p_need"]) >= 0.35)


def compute_epsilon_for_level(
    signals: dict[str, float],
    *,
    scene_uncertainty: dict[str, Any] | None = None,
) -> float:
    learned_epsilon = clamp_01(signals.get("epsilon_agent", 0.0))
    if not isinstance(scene_uncertainty, dict):
        return learned_epsilon
    scene_epsilon = clamp_01(scene_uncertainty.get("epsilon_scene", 0.0))
    return max(learned_epsilon, scene_epsilon)


def decide_commitment_level(
    should_intervene: bool,
    signals: dict[str, float],
    *,
    observations: Sequence[dict[str, Any]],
    proactive_task: str | None = None,
    scene_uncertainty: dict[str, Any] | None = None,
) -> tuple[int, str, dict[str, Any]]:
    context = build_observation_context(observations, proactive_task=proactive_task)
    epsilon_for_level = compute_epsilon_for_level(signals, scene_uncertainty=scene_uncertainty)
    context["epsilon_for_level"] = float(epsilon_for_level)

    if not should_intervene:
        return 0, "gate_blocked", context

    if epsilon_for_level >= 0.60:
        return 1, "high_epsilon", context
    if clamp_01(signals["r_risk"]) > 0.70:
        return 1, "risk_high", context

    allow_level2 = bool(
        clamp_01(signals["d_stuck"]) >= 0.35
        and epsilon_for_level <= 0.35
        and clamp_01(signals["r_risk"]) <= 0.55
        and context["task_clarity"]
    )
    if allow_level2:
        return 2, "clear_blocker_direct_action", context
    if context["domain"] == "writing" and context["has_research_browse"]:
        return 1, "writing_research_probe", context
    if context["has_explicit_help_request"]:
        return 1, "explicit_help_probe", context
    return 1, "conservative_probe", context
