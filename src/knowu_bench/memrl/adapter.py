from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knowu_bench.memrl.paths import ensure_embedded_memrl_importable, repo_root
from knowu_bench.runtime.utils.loader import UserProfileLoader
from knowu_bench.runtime.utils.user_log_context import load_logs, resolve_user_log_path
from knowu_bench.tasks.registry import TaskRegistry

DEFAULT_BUNDLE_DIR = "artifacts/memrl_knowu_routine_bundle"
HIGH_IMPACT_ACTION_TOKENS = (
    "send",
    "message",
    "sms",
    "mail",
    "email",
    "call",
    "post",
    "share",
    "delete",
    "remove",
    "checkout",
    "buy",
    "order",
    "payment",
    "contact",
    "report",
    "block",
)
ROUTINE_TASK_SPECS: tuple[dict[str, Any], ...] = (
    {
        "task_family": "battery_saver",
        "display": "battery saver",
        "habit_keys": ("low_battery_saver",),
        "trigger_variants": (
            "System Status: Battery Level 15%. System low battery warning is visible.",
            "Current phone battery is critically low and the user needs to stay connected.",
            "Battery has dropped below the user's low-power threshold.",
        ),
        "negative_variants": (
            "Battery level is normal and no power warning is present.",
            "The user is charging the phone; no low-battery routine trigger is active.",
        ),
    },
    {
        "task_family": "birthday_wish",
        "display": "birthday wish",
        "habit_keys": ("birthday_wish_routine",),
        "trigger_variants": (
            "Calendar reminder indicates a close contact has a birthday today.",
            "A birthday reminder is due and the user usually sends a greeting.",
            "The phone calendar shows today's birthday reminder for a saved contact.",
        ),
        "negative_variants": (
            "There is no birthday reminder today.",
            "The calendar has ordinary events only; no greeting routine is due.",
        ),
    },
    {
        "task_family": "bluetooth_cleanup",
        "display": "bluetooth cleanup",
        "habit_keys": ("bluetooth_cleanup",),
        "trigger_variants": (
            "Bluetooth headphones disconnected while media may still play aloud.",
            "The user was playing media and Bluetooth just disconnected.",
            "A Bluetooth audio device disconnected in a public context.",
        ),
        "negative_variants": (
            "Bluetooth remains connected and no media leak risk is present.",
            "No active media playback or Bluetooth disconnect event is present.",
        ),
    },
    {
        "task_family": "clock_out",
        "display": "clock-out handoff",
        "habit_keys": ("clock_out_routine",),
        "trigger_variants": (
            "It is weekday 18:00 and the user's workday handoff window has arrived.",
            "End-of-workday time boundary reached; team handoff message is expected.",
            "The current time matches the user's clock-out routine.",
        ),
        "negative_variants": (
            "It is not the end of the workday.",
            "No shift handoff or clock-out time boundary is active.",
        ),
    },
    {
        "task_family": "contact_saver",
        "display": "contact saver",
        "habit_keys": ("contact_saver",),
        "trigger_variants": (
            "A new trusted phone number appears in a recent message and should be saved.",
            "The user received contact details from a recurring relation.",
            "A message contains a new contact the user normally saves.",
        ),
        "negative_variants": (
            "No new trusted contact details are present.",
            "Recent messages do not contain a number that should be saved.",
        ),
    },
    {
        "task_family": "daily_family_call",
        "display": "daily family call",
        "habit_keys": ("daily_family_call",),
        "trigger_variants": (
            "It is the user's routine family-call time window.",
            "The phone is idle during the scheduled daily family call window.",
            "A daily call reminder for a family member is due.",
        ),
        "negative_variants": (
            "No family-call reminder or time window is active.",
            "The current time is outside the user's daily call routine.",
        ),
    },
    {
        "task_family": "deep_work",
        "display": "deep work",
        "habit_keys": ("deep_work_block",),
        "trigger_variants": (
            "The user's calendar marks a deep-work block starting now.",
            "Focus time is active and distractions should be reduced.",
            "The current schedule matches the user's deep-work routine.",
        ),
        "negative_variants": (
            "No focus block is active.",
            "The user is in normal activity, not a deep-work window.",
        ),
    },
    {
        "task_family": "gallery_cleanup",
        "display": "gallery cleanup",
        "habit_keys": ("gallery_cleanup",),
        "trigger_variants": (
            "The user's scheduled gallery cleanup window has arrived.",
            "Old screenshots/photos match the user's recurring cleanup policy.",
            "Storage/gallery maintenance is due according to the user's routine.",
        ),
        "negative_variants": (
            "No gallery cleanup schedule is active.",
            "Recent photos do not match the user's cleanup trigger.",
        ),
    },
    {
        "task_family": "mattermost_response",
        "display": "Mattermost alert response",
        "habit_keys": ("on_call_response",),
        "trigger_variants": (
            "Mattermost shows a critical off-hours alert requiring acknowledgement.",
            "A critical incident message arrived in the user's team channel.",
            "An on-call alert from the monitoring system is unread.",
        ),
        "negative_variants": (
            "Mattermost has no critical alert requiring acknowledgement.",
            "Recent messages are ordinary team chatter, not an on-call incident.",
        ),
    },
    {
        "task_family": "morning_paper_reading",
        "display": "morning paper reading",
        "habit_keys": ("morning_routine",),
        "trigger_variants": (
            "It is the user's morning reading window.",
            "The current morning context matches the user's news/paper routine.",
            "Morning routine context suggests opening the user's preferred reading source.",
        ),
        "negative_variants": (
            "It is not the user's morning reading window.",
            "No morning reading trigger is present.",
        ),
    },
    {
        "task_family": "morning_weather_check",
        "display": "morning weather check",
        "habit_keys": ("morning_weather",),
        "trigger_variants": (
            "It is morning and the user usually checks the weather before going out.",
            "The user's morning weather-check time window is active.",
            "Current context suggests preparing for today's weather.",
        ),
        "negative_variants": (
            "No morning weather-check trigger is active.",
            "The user is not in a commute-preparation context.",
        ),
    },
    {
        "task_family": "night_eye_care",
        "display": "night eye care",
        "habit_keys": ("night_eye_care",),
        "trigger_variants": (
            "It is late night and the user is actively using the phone.",
            "The current time falls inside the user's eye-care routine window.",
            "Night phone usage indicates dark-theme eye protection should be considered.",
        ),
        "negative_variants": (
            "It is daytime; no night eye-care trigger is present.",
            "The user is not using the phone during a late-night window.",
        ),
    },
    {
        "task_family": "pre_meeting_prep",
        "display": "pre-meeting prep",
        "habit_keys": ("pre_meeting_prep",),
        "trigger_variants": (
            "A meeting starts soon and the user's prep routine should open relevant material.",
            "Calendar reminder indicates a near-term meeting preparation window.",
            "The current time is just before a scheduled meeting.",
        ),
        "negative_variants": (
            "No meeting starts soon.",
            "There is no calendar reminder requiring meeting preparation.",
        ),
    },
    {
        "task_family": "scam_sms_intercept",
        "display": "scam SMS intercept",
        "habit_keys": ("scam_sms_intercept", "scam_sms_block"),
        "trigger_variants": (
            "A suspicious SMS with scam-like language arrived from an unknown sender.",
            "Messages contains a likely phishing/scam text.",
            "A spam SMS asks the user to click a risky link.",
        ),
        "negative_variants": (
            "Recent SMS messages are from trusted contacts.",
            "No suspicious or scam-like message is present.",
        ),
    },
    {
        "task_family": "weekly_report",
        "display": "weekly report",
        "habit_keys": ("weekly_report",),
        "trigger_variants": (
            "It is Friday near the user's weekly report sending window.",
            "The weekly report deadline window is active.",
            "The user's recurring weekly report routine is due.",
        ),
        "negative_variants": (
            "It is not the weekly report window.",
            "No report deadline or weekly sending routine is active.",
        ),
    },
    {
        "task_family": "weekend_sleeper",
        "display": "weekend sleeper",
        "habit_keys": ("weekend_sleeper", "weekend_alarm"),
        "trigger_variants": (
            "Tomorrow is a weekend and the user usually disables early alarms.",
            "The weekend sleep routine is active for tomorrow morning.",
            "The user's alarm settings conflict with their weekend sleep habit.",
        ),
        "negative_variants": (
            "Tomorrow is a workday, not a weekend sleep-in day.",
            "No weekend alarm adjustment trigger is present.",
        ),
    },
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_bundle_path() -> Path:
    return repo_root() / DEFAULT_BUNDLE_DIR / "memrl_episodes.jsonl"


def _compact(value: Any, *, max_len: int = 800) -> str:
    text = " ".join(str(value or "").split())
    if max_len > 0 and len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _task_family(task_name: str) -> str:
    stem = task_name.split("@", 1)[0]
    words = re.sub(r"(?<!^)([A-Z])", r" \1", stem).lower()
    for suffix in (" routine task", " task", " routine"):
        words = words.replace(suffix, "")
    return re.sub(r"[^a-z0-9]+", "_", words).strip("_") or "routine"


def _profile_id(task_name: str) -> str:
    return task_name.split("@", 1)[1] if "@" in task_name else "default"


def _actions_from_task(task: Any) -> list[str]:
    expectation = getattr(task, "expectation", {}) or {}
    actions = expectation.get("actions", [])
    if isinstance(actions, str):
        return [actions]
    if isinstance(actions, list):
        return [str(item) for item in actions if item]
    if isinstance(actions, dict):
        return [f"{key}:{value}" for key, value in actions.items()]
    return []


def _action_items(action: Any) -> list[str]:
    if isinstance(action, str):
        return [action]
    if isinstance(action, list):
        return [_compact(item, max_len=160) for item in action if item]
    if isinstance(action, dict):
        items = []
        for key, value in action.items():
            if isinstance(value, list):
                value_text = ", ".join(_compact(item, max_len=80) for item in value)
            elif isinstance(value, dict):
                value_text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                value_text = _compact(value, max_len=160)
            items.append(f"{key}: {value_text}")
        return items
    return []


def _commitment_level_from_text(text: str) -> int:
    lowered = text.lower()
    if any(token in lowered for token in HIGH_IMPACT_ACTION_TOKENS):
        return 1
    return 2


def _commitment_level(task: Any) -> int:
    actions = " ".join(_actions_from_task(task)).lower()
    name = getattr(task, "name", task.__class__.__name__).lower()
    app_names = " ".join(sorted(getattr(task, "app_names", set()) or set())).lower()
    joined = f"{name} {app_names} {actions}"
    return _commitment_level_from_text(joined)


def _initial_q_value(scenario_type: str, should_act: bool) -> float:
    values = {
        "strong_positive_habit": 0.18,
        "positive_habit_evidence": 0.12,
        "confirmation_boundary": 0.08,
        "near_miss_abstain": 0.10,
        "strong_abstain": 0.15,
        "confirmation_boundary_abstain": 0.08,
        "ambiguous_abstain": 0.05,
    }
    if scenario_type in values:
        return values[scenario_type]
    return 0.12 if should_act else 0.10


def _scenario_type(*, has_habit: bool, variant_index: int) -> str:
    idx = variant_index % 4
    if has_habit:
        return (
            "strong_positive_habit"
            if idx == 0
            else "positive_habit_evidence"
            if idx == 1
            else "confirmation_boundary"
            if idx == 2
            else "near_miss_abstain"
        )
    return (
        "strong_abstain"
        if idx == 0
        else "near_miss_abstain"
        if idx == 1
        else "confirmation_boundary_abstain"
        if idx == 2
        else "ambiguous_abstain"
    )


def _scenario_note(*, scenario_type: str, display: str) -> str:
    notes = {
        "strong_positive_habit": (
            f"Independent habit evidence for {display}: comparable contexts repeatedly show this preference."
        ),
        "positive_habit_evidence": (
            f"Alternate habit evidence for {display}: a different setting supports the same routine."
        ),
        "confirmation_boundary": (
            f"Boundary evidence for {display}: the habit is plausible, but concrete action requires confirmation."
        ),
        "near_miss_abstain": (
            f"Near-miss evidence for {display}: the context is related, but a required condition is absent or already handled."
        ),
        "strong_abstain": (
            f"No-routine evidence for {display}: the profile has no established matching habit."
        ),
        "confirmation_boundary_abstain": (
            f"Ambiguous boundary for {display}: do not proactively execute without stronger evidence."
        ),
        "ambiguous_abstain": (
            f"Weak abstain evidence for {display}: routine-like background activity is not enough."
        ),
    }
    return notes.get(scenario_type, f"Habit evidence for {display}.")


def _scenario_context(*, spec: dict[str, Any], scenario_type: str) -> str:
    display = str(spec["display"])
    positive = tuple(spec.get("trigger_variants", ())) or (f"A {display} routine cue is present.",)
    positive_0 = str(positive[0])
    positive_1 = str(positive[1 % len(positive)])
    positive_2 = str(positive[2 % len(positive)])
    contexts = {
        "strong_positive_habit": positive_0,
        "positive_habit_evidence": (
            f"Different supporting context for {display}: {positive_1}"
        ),
        "confirmation_boundary": (
            f"Boundary context for {display}: {positive_2} A relevant cue is visible, "
            "but the exact timing or user intent should be confirmed before any action."
        ),
        "near_miss_abstain": (
            f"Near-miss for {display}: the screen resembles the routine context, "
            "but the required condition is absent or the user has already handled it."
        ),
        "strong_abstain": (
            f"No {display} trigger is active; the phone shows ordinary activity unrelated to this routine."
        ),
        "confirmation_boundary_abstain": (
            f"Ambiguous {display} cue: a related app or notification is visible, "
            "but there is not enough evidence to infer a routine trigger."
        ),
        "ambiguous_abstain": (
            f"Weak background context near {display}: routine-like activity exists, "
            "but timing, recipient, or required content is missing."
        ),
    }
    return contexts.get(scenario_type, positive_0)


def _habit_evidence_candidate(
    *,
    profile_id: str,
    display: str,
    should_act: bool,
) -> dict[str, Any]:
    if not should_act:
        return {"purpose": None, "proactive_task": None, "response": None, "operation": "nop"}
    return {
        "purpose": f"Remember habit evidence for {profile_id}'s {display} routine.",
        "proactive_task": (
            f"Use this as evidence that {profile_id} may expect {display} support in a matching context; "
            "ask for confirmation before any concrete action."
        ),
        "response": f"This is habit evidence for {display}; do not execute a concrete action from memory alone.",
        "operation": None,
    }


def _candidate_for_task(task: Any, should_act: bool, level: int) -> dict[str, Any]:
    if not should_act:
        return {
            "purpose": None,
            "proactive_task": None,
            "response": None,
            "operation": "nop",
        }

    name = getattr(task, "name", task.__class__.__name__)
    family = _task_family(name).replace("_", " ")
    actions = _actions_from_task(task)
    action_text = ", ".join(actions) if actions else family
    if level <= 1:
        response = f"I noticed this matches your {family} routine. Would you like me to handle it now?"
    else:
        response = f"I can handle your {family} routine now."
    return {
        "purpose": f"Support the user's established {family} routine.",
        "proactive_task": f"Execute the {family} routine: {action_text}.",
        "response": response,
        "operation": f"knowu.routine.{_task_family(name)}",
    }


def _load_recent_log_observations(task: Any, *, max_items: int) -> list[dict[str, Any]]:
    user_profile = getattr(task, "user_profile", None)
    profile_path = getattr(task, "profile_path", None)
    try:
        log_path = resolve_user_log_path(user_profile, profile_path=profile_path)
    except Exception:
        log_path = None
    if not log_path:
        return []
    try:
        entries = load_logs(log_path)
    except Exception:
        return []
    observations = []
    for entry in entries[-max_items:]:
        action = entry.get("action")
        if not action:
            continue
        observations.append(
            {
                "time": entry.get("time"),
                "event": _compact(action, max_len=500),
                "source": "user_log",
                "category": entry.get("category"),
                "location": entry.get("location"),
            }
        )
    return observations


def _log_observations_for_profile(profile_id: str, *, max_items: int) -> list[dict[str, Any]]:
    log_path = repo_root() / "src" / "knowu_bench" / "user_logs" / f"{profile_id}.json"
    if not log_path.exists():
        return []
    try:
        entries = load_logs(str(log_path))
    except Exception:
        return []
    observations = []
    for entry in entries[-max_items:]:
        action = entry.get("action")
        if not action:
            continue
        observations.append(
            {
                "time": entry.get("time"),
                "event": _compact(action, max_len=500),
                "source": "user_log",
                "category": entry.get("category"),
                "location": entry.get("location"),
            }
        )
    return observations


def _log_window_observations_for_profile(
    profile_id: str,
    *,
    max_items: int,
    variant_index: int,
) -> list[dict[str, Any]]:
    log_path = repo_root() / "src" / "knowu_bench" / "user_logs" / f"{profile_id}.json"
    if not log_path.exists() or max_items <= 0:
        return []
    try:
        entries = [entry for entry in load_logs(str(log_path)) if entry.get("action")]
    except Exception:
        return []
    if not entries:
        return []
    if len(entries) <= max_items:
        window = entries
    else:
        max_start = len(entries) - max_items
        start = (variant_index * max(1, max_items // 2)) % (max_start + 1)
        window = entries[start : start + max_items]
    observations = []
    for entry in window:
        observations.append(
            {
                "time": entry.get("time"),
                "event": _compact(entry.get("action"), max_len=500),
                "source": "user_log",
                "category": entry.get("category"),
                "location": entry.get("location"),
            }
        )
    return observations


def _task_observations(task: Any, *, max_log_items: int) -> list[dict[str, Any]]:
    name = getattr(task, "name", task.__class__.__name__)
    observations = _load_recent_log_observations(task, max_items=max_log_items)
    observations.append(
        {
            "time": "current",
            "source": "knowu_task",
            "event": _compact(getattr(task, "goal", ""), max_len=1600),
            "task_name": name,
            "profile_id": _profile_id(name),
        }
    )
    expectation = getattr(task, "expectation", {}) or {}
    observations.append(
        {
            "time": "current",
            "source": "knowu_oracle_hidden_for_labels",
            "event": (
                f"task_name:{name} profile:{_profile_id(name)} "
                f"expected_should_act:{bool(expectation.get('should_act', False))} "
                f"expected_actions:{', '.join(_actions_from_task(task)) or 'none'}"
            ),
        }
    )
    return observations


def _habit_observations(
    *,
    profile_id: str,
    habit_name: str,
    habit: dict[str, Any],
    profile: dict[str, Any],
    max_log_items: int,
) -> list[dict[str, Any]]:
    observations = _log_observations_for_profile(profile_id, max_items=max_log_items)
    identity = profile.get("identity", {}) if isinstance(profile.get("identity"), dict) else {}
    trigger = habit.get("trigger", {}) if isinstance(habit, dict) else {}
    action = habit.get("action", {}) if isinstance(habit, dict) else {}
    observations.extend(
        [
            {
                "time": "long_term",
                "source": "profile_habit",
                "event": _compact(
                    f"profile:{profile_id} habit:{habit_name} "
                    f"description:{habit.get('description', '')} "
                    f"trigger:{json.dumps(trigger, ensure_ascii=False, sort_keys=True)}",
                    max_len=1600,
                ),
                "profile_id": profile_id,
                "habit_name": habit_name,
            },
            {
                "time": "long_term",
                "source": "profile_action_preference",
                "event": _compact(
                    f"profile:{profile_id} occupation:{identity.get('occupation', '')} "
                    f"preferred_action:{json.dumps(action, ensure_ascii=False, sort_keys=True)}",
                    max_len=1600,
                ),
                "profile_id": profile_id,
                "habit_name": habit_name,
            },
        ]
    )
    return observations


def _background_observations_for_profile(
    *,
    profile_id: str,
    profile: dict[str, Any],
    max_log_items: int,
    offset: int,
) -> list[dict[str, Any]]:
    observations = _log_observations_for_profile(profile_id, max_items=max_log_items + offset)
    if offset:
        observations = observations[:max_log_items]
    else:
        observations = observations[-max_log_items:]
    identity = profile.get("identity", {}) if isinstance(profile.get("identity"), dict) else {}
    observations.append(
        {
            "time": "current",
            "source": "synthetic_background_context",
            "event": _compact(
                f"profile:{profile_id} occupation:{identity.get('occupation', '')}. "
                "The user is in ordinary background activity; no explicit routine trigger, "
                "urgent alert, schedule boundary, low-battery state, or communication request is present.",
                max_len=1200,
            ),
            "profile_id": profile_id,
        }
    )
    return observations


def _simulation_for_task(task: Any, should_act: bool) -> dict[str, Any]:
    if should_act:
        return {
            "acceptance": "accept",
            "acceptance_confidence": 0.9,
            "flow_impact": "unchanged",
            "relevance": "high",
            "timing": "good",
            "reasoning": "The proposed intervention matches the user's established routine and current trigger.",
        }
    return {
        "acceptance": "dismiss",
        "acceptance_confidence": 0.85,
        "flow_impact": "disrupted",
        "relevance": "low",
        "timing": "bad",
        "reasoning": "The user profile does not establish this routine, so proactive execution would be unwanted.",
    }


def _simulation_for_decision(should_act: bool) -> dict[str, Any]:
    if should_act:
        return {
            "acceptance": "accept",
            "acceptance_confidence": 0.88,
            "flow_impact": "unchanged",
            "relevance": "high",
            "timing": "good",
            "reasoning": "The intervention follows a long-term profile habit and matching trigger/action schema.",
        }
    return {
        "acceptance": "ignore",
        "acceptance_confidence": 0.8,
        "flow_impact": "disrupted",
        "relevance": "low",
        "timing": "bad",
        "reasoning": "No profile habit trigger is active, so background monitoring should remain silent.",
    }


def _phase_a_signals_for_episode(
    observations: list[dict[str, Any]],
    *,
    should_act: bool,
    level: int,
) -> dict[str, float]:
    ensure_embedded_memrl_importable()
    try:
        import proactive_pipeline

        inferred = proactive_pipeline.infer_signals(observations)
    except Exception:
        inferred = {}

    if should_act:
        defaults = {
            "flow": 0.25 if level == 2 else 0.35,
            "stuck": 0.2,
            "need": 0.82,
            "accept": 0.82 if level == 2 else 0.68,
            "risk": 0.25 if level == 2 else 0.62,
            "uncertainty": 0.18 if level == 2 else 0.35,
            "progress": 0.25,
            "rejection_memory": 0.05,
        }
    else:
        defaults = {
            "flow": 0.75,
            "stuck": 0.1,
            "need": 0.18,
            "accept": 0.2,
            "risk": 0.2,
            "uncertainty": 0.25,
            "progress": 0.65,
            "rejection_memory": 0.1,
        }
    signals = {}
    for key, value in defaults.items():
        if key in inferred:
            blended = 0.35 * float(inferred.get(key, value) or value) + 0.65 * value
        else:
            blended = value
        signals[key] = round(max(0.0, min(1.0, blended)), 4)
    return signals


def _gate_features_for_episode(
    observations: list[dict[str, Any]],
    signals: dict[str, float],
) -> dict[str, Any]:
    ensure_embedded_memrl_importable()
    try:
        import proactive_pipeline

        return {
            "phase_a_signals": signals,
            "interruption_gate": proactive_pipeline.strict_interruption_gate(
                observations=observations,
                signals=signals,
            ),
        }
    except Exception:
        allow = signals.get("need", 0.0) >= 0.5 and signals.get("risk", 0.0) < 0.8
        return {
            "phase_a_signals": signals,
            "interruption_gate": {
                "allow_interruption": allow,
                "recommended_level": 2 if allow and signals.get("risk", 0.0) < 0.45 else 1 if allow else 0,
                "reason": "synthetic_phase_a_gate_fallback",
            },
        }


def _attach_phase_a_fields(episode: dict[str, Any]) -> dict[str, Any]:
    decision = episode.get("decision", {}) or {}
    should_act = bool(decision.get("should_intervene", False))
    level = int(decision.get("commitment_level", 0) or 0)
    observations = list(episode.get("observations", []) or [])
    signals = _phase_a_signals_for_episode(observations, should_act=should_act, level=level)
    episode["phase_a_signals"] = signals
    episode["gate_features"] = _gate_features_for_episode(observations, signals)
    return episode


def _episode_from_task(task: Any, *, max_log_items: int) -> dict[str, Any]:
    ensure_embedded_memrl_importable()
    from agent.memrl.schema import enrich_memory_schema

    name = getattr(task, "name", task.__class__.__name__)
    expectation = getattr(task, "expectation", {}) or {}
    should_act = bool(expectation.get("should_act", False))
    level = _commitment_level(task) if should_act else 0
    candidate = _candidate_for_task(task, should_act, level)
    simulation = _simulation_for_task(task, should_act)
    decision = {
        "should_intervene": should_act,
        "commitment_level": level,
        "risk": "medium" if level == 1 else "low",
        "reason": (
            "Known KnowU routine trigger with supporting user history."
            if should_act
            else "No matching hidden routine in KnowU profile; abstain."
        ),
    }
    now = utc_now()
    episode = {
        "memory_id": f"knowu-routine-{name}",
        "sample_id": name,
        "source": "knowu_routine",
        "domain": "mobile_routine",
        "observations": _task_observations(task, max_log_items=max_log_items),
        "intent_text": _compact(getattr(task, "goal", ""), max_len=2400),
        "context_family": f"knowu_{_task_family(name)}",
        "action_family": "no_intervention" if not should_act else f"knowu_{_task_family(name)}",
        "outcome_family": "correct_abstain" if not should_act else "helpful_intervention",
        "candidate": candidate,
        "simulation": simulation,
        "decision": decision,
        "labels": {
            "y_need": int(should_act),
            "y_accept": int(should_act),
            "gold_should": should_act,
            "gold_level": level,
            "profile_id": _profile_id(name),
            "task_name": name,
        },
        "reward": 1.0,
        "q_value": 1.0,
        "q_visits": 1,
        "created_at": now,
        "updated_at": now,
    }
    return _attach_phase_a_fields(enrich_memory_schema(episode))


def _episode_from_profile_habit(
    *,
    profile_id: str,
    habit_name: str,
    habit: dict[str, Any],
    profile: dict[str, Any],
    max_log_items: int,
) -> dict[str, Any]:
    ensure_embedded_memrl_importable()
    from agent.memrl.schema import enrich_memory_schema

    action = habit.get("action", {}) if isinstance(habit, dict) else {}
    action_text = ", ".join(_action_items(action)) or _compact(action, max_len=600)
    decision_text = f"{habit_name} {habit.get('description', '')} {action_text}"
    level = _commitment_level_from_text(decision_text)
    family = re.sub(r"[^a-z0-9]+", "_", habit_name.lower()).strip("_") or "habit"
    response = (
        f"I noticed this matches your {habit_name.replace('_', ' ')} routine. "
        "Would you like me to handle it now?"
        if level == 1
        else f"I can handle your {habit_name.replace('_', ' ')} routine now."
    )
    candidate = {
        "purpose": f"Support {profile_id}'s long-term routine: {habit.get('description', habit_name)}",
        "proactive_task": f"Execute profile habit '{habit_name}': {action_text or habit_name}.",
        "response": response,
        "operation": f"knowu.profile_habit.{family}",
    }
    simulation = _simulation_for_decision(True)
    decision = {
        "should_intervene": True,
        "commitment_level": level,
        "risk": "medium" if level == 1 else "low",
        "reason": "Profile-derived habit memory indicates a matching trigger should be handled proactively.",
    }
    now = utc_now()
    episode = {
        "memory_id": f"knowu-profile-habit-{profile_id}-{family}",
        "sample_id": f"profile_habit::{profile_id}::{family}",
        "source": "knowu_profile_habit_synthetic",
        "domain": "mobile_routine",
        "observations": _habit_observations(
            profile_id=profile_id,
            habit_name=habit_name,
            habit=habit,
            profile=profile,
            max_log_items=max_log_items,
        ),
        "intent_text": _compact(
            f"{profile_id} habit {habit_name}: {habit.get('description', '')}. "
            f"Action: {action_text}.",
            max_len=2400,
        ),
        "context_family": f"knowu_profile_{family}",
        "action_family": f"knowu_profile_{family}",
        "outcome_family": "helpful_intervention",
        "candidate": candidate,
        "simulation": simulation,
        "decision": decision,
        "labels": {
            "y_need": 1,
            "y_accept": 1,
            "gold_should": True,
            "gold_level": level,
            "profile_id": profile_id,
            "habit_name": habit_name,
            "task_name": None,
            "label_source": "profile_habit_not_task_expectation",
        },
        "reward": 1.0,
        "q_value": 1.0,
        "q_visits": 1,
        "created_at": now,
        "updated_at": now,
    }
    return _attach_phase_a_fields(enrich_memory_schema(episode))


def _background_episode_from_profile(
    *,
    profile_id: str,
    profile: dict[str, Any],
    max_log_items: int,
    index: int,
) -> dict[str, Any]:
    ensure_embedded_memrl_importable()
    from agent.memrl.schema import enrich_memory_schema

    simulation = _simulation_for_decision(False)
    decision = {
        "should_intervene": False,
        "commitment_level": 0,
        "risk": "low",
        "reason": "Synthetic background monitoring case with no active profile-habit trigger.",
    }
    now = utc_now()
    episode = {
        "memory_id": f"knowu-profile-background-{profile_id}-{index}",
        "sample_id": f"profile_background::{profile_id}::{index}",
        "source": "knowu_profile_background_synthetic",
        "domain": "mobile_routine",
        "observations": _background_observations_for_profile(
            profile_id=profile_id,
            profile=profile,
            max_log_items=max_log_items,
            offset=index * max_log_items,
        ),
        "intent_text": (
            "Background monitoring example: no explicit routine trigger is active; "
            "the correct proactive behavior is to stay silent."
        ),
        "context_family": "knowu_profile_background",
        "action_family": "no_intervention",
        "outcome_family": "correct_abstain",
        "candidate": {"purpose": None, "proactive_task": None, "response": None, "operation": "nop"},
        "simulation": simulation,
        "decision": decision,
        "labels": {
            "y_need": 0,
            "y_accept": 0,
            "gold_should": False,
            "gold_level": 0,
            "profile_id": profile_id,
            "habit_name": None,
            "task_name": None,
            "label_source": "synthetic_background_not_task_expectation",
        },
        "reward": 1.0,
        "q_value": 1.0,
        "q_visits": 1,
        "created_at": now,
        "updated_at": now,
    }
    return _attach_phase_a_fields(enrich_memory_schema(episode))


def _first_matching_habit(profile: dict[str, Any], spec: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    habits = profile.get("habits", {}) if isinstance(profile.get("habits"), dict) else {}
    for habit_key in spec.get("habit_keys", ()):
        habit = habits.get(habit_key)
        if isinstance(habit, dict) and habit:
            return str(habit_key), habit
    return None, None


def _task_matrix_episode(
    *,
    profile_id: str,
    profile: dict[str, Any],
    spec: dict[str, Any],
    variant_index: int,
    max_log_items: int,
) -> dict[str, Any]:
    ensure_embedded_memrl_importable()
    from agent.memrl.schema import enrich_memory_schema

    habit_name, habit = _first_matching_habit(profile, spec)
    has_habit = habit is not None
    family = str(spec["task_family"])
    scenario_type = _scenario_type(has_habit=has_habit, variant_index=variant_index)
    should_act = has_habit and scenario_type != "near_miss_abstain"
    trigger_text = _scenario_context(spec=spec, scenario_type=scenario_type)
    initial_q = _initial_q_value(scenario_type, should_act)
    observations = _log_window_observations_for_profile(
        profile_id,
        max_items=max_log_items,
        variant_index=variant_index,
    )
    observations.append(
        {
            "time": "current",
            "source": "synthetic_task_context",
            "event": _compact(
                f"profile:{profile_id} task_family:{family} scenario:{scenario_type} "
                f"context:{trigger_text} note:{_scenario_note(scenario_type=scenario_type, display=str(spec['display']))}",
                max_len=1200,
            ),
            "profile_id": profile_id,
            "task_family": family,
            "variant_index": variant_index,
            "scenario_type": scenario_type,
        }
    )

    if should_act and habit is not None and habit_name is not None:
        action = habit.get("action", {}) if isinstance(habit, dict) else {}
        action_text = ", ".join(_action_items(action)) or _compact(action, max_len=600)
        observations.append(
            {
                "time": "long_term",
                "source": "profile_task_habit",
                "event": _compact(
                    f"profile:{profile_id} task_family:{family} habit:{habit_name} "
                    f"description:{habit.get('description', '')} "
                    f"trigger:{json.dumps(habit.get('trigger', {}), ensure_ascii=False, sort_keys=True)} "
                    f"action:{json.dumps(action, ensure_ascii=False, sort_keys=True)}",
                    max_len=1800,
                ),
                "profile_id": profile_id,
                "task_family": family,
                "habit_name": habit_name,
            }
        )
        level = _commitment_level_from_text(f"{family} {habit_name} {action_text}")
        candidate = _habit_evidence_candidate(
            profile_id=profile_id,
            display=str(spec["display"]),
            should_act=True,
        )
        simulation = _simulation_for_decision(True)
        decision = {
            "should_intervene": True,
            "commitment_level": level,
            "risk": "medium" if level == 1 else "low",
            "reason": f"Profile-task matrix memory stores non-executable habit evidence: {scenario_type}.",
        }
        action_family = f"knowu_profile_task_{family}"
        outcome_family = "helpful_intervention"
    else:
        observations.append(
            {
                "time": "long_term",
                "source": "profile_task_absence",
                "event": _compact(
                    f"profile:{profile_id} task_family:{family} scenario:{scenario_type}. "
                    f"{_scenario_note(scenario_type=scenario_type, display=str(spec['display']))} "
                    f"Matching habit keys considered: {', '.join(spec.get('habit_keys', ()))}.",
                    max_len=1000,
                ),
                "profile_id": profile_id,
                "task_family": family,
                "scenario_type": scenario_type,
            }
        )
        level = 0
        habit_name = None
        candidate = {"purpose": None, "proactive_task": None, "response": None, "operation": "nop"}
        simulation = _simulation_for_decision(False)
        decision = {
            "should_intervene": False,
            "commitment_level": 0,
            "risk": "low",
            "reason": f"Profile-task matrix memory supports abstention: {scenario_type}.",
        }
        action_family = "no_intervention"
        outcome_family = "correct_abstain"

    now = utc_now()
    episode = {
        "memory_id": f"knowu-profile-task-{profile_id}-{family}-v{variant_index}",
        "sample_id": f"profile_task::{profile_id}::{family}::v{variant_index}",
        "source": "knowu_profile_task_matrix_synthetic",
        "domain": "mobile_routine",
        "observations": observations,
        "intent_text": _compact(
            f"{profile_id} / {family} / {scenario_type} / variant {variant_index}: {trigger_text}",
            max_len=2400,
        ),
        "context_family": f"knowu_profile_task_{family}",
        "action_family": action_family,
        "outcome_family": outcome_family,
        "candidate": candidate,
        "simulation": simulation,
        "decision": decision,
        "labels": {
            "y_need": int(should_act),
            "y_accept": int(should_act),
            "gold_should": should_act,
            "gold_level": level,
            "profile_id": profile_id,
            "habit_name": habit_name,
            "task_family": family,
            "task_name": None,
            "label_source": "profile_task_matrix_not_task_expectation",
            "scenario_type": scenario_type,
            "initial_q_value": initial_q,
        },
        "reward": initial_q,
        "q_value": initial_q,
        "q_visits": 0,
        "created_at": now,
        "updated_at": now,
    }
    return _attach_phase_a_fields(enrich_memory_schema(episode))


def _make_generation_row(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": episode["sample_id"],
        "source": episode["source"],
        "domain": episode["domain"],
        "observations": episode["observations"],
        "intent_text": episode["intent_text"],
        "memory_generation_prior": {
            "preferred_level": episode["decision"]["commitment_level"],
            "positive_patterns": [episode["candidate"].get("proactive_task")]
            if episode["candidate"].get("proactive_task")
            else [],
            "negative_patterns": [episode["decision"].get("reason")]
            if not episode["decision"]["should_intervene"]
            else [],
            "avoid_patterns": [episode["candidate"].get("response")]
            if not episode["decision"]["should_intervene"] and episode["candidate"].get("response")
            else [],
        },
        "target_candidate": episode["candidate"],
        "target_decision": episode["decision"],
    }


def _make_phase_a_row(episode: dict[str, Any]) -> dict[str, Any]:
    signals = episode.get("phase_a_signals", {}) or {}
    return {
        "sample_id": episode["sample_id"],
        "source": episode["source"],
        "domain": episode["domain"],
        "observations": episode["observations"],
        "phase_a_reference": {
            "signals": signals,
            "interruption_gate": (episode.get("gate_features", {}) or {}).get("interruption_gate", {}),
        },
        "target_signals": signals,
    }


def _make_simulation_row(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": episode["sample_id"],
        "source": episode["source"],
        "domain": episode["domain"],
        "observations": episode["observations"],
        "candidate": episode["candidate"],
        "memory_simulation_prior": {
            "historical_accept_rate": 1.0 if episode["simulation"]["acceptance"] == "accept" else 0.0,
            "historical_dismiss_rate": 1.0 if episode["simulation"]["acceptance"] == "dismiss" else 0.0,
            "historical_annoy_rate": 0.0,
        },
        "target_simulation": episode["simulation"],
    }


def _make_decision_row(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": episode["sample_id"],
        "source": episode["source"],
        "domain": episode["domain"],
        "observations": episode["observations"],
        "candidate": episode["candidate"],
        "simulation": episode["simulation"],
        "memory_decision_prior": {
            "intervene_memory_value": episode["q_value"]
            if episode["decision"]["should_intervene"]
            else 0.0,
            "abstain_memory_value": episode["q_value"]
            if not episode["decision"]["should_intervene"]
            else 0.0,
            "memory_level_mode": episode["decision"]["commitment_level"],
        },
        "target_decision": episode["decision"],
        "reward": episode["reward"],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_episode_bundle(
    *,
    out_dir: Path,
    episodes: list[dict[str, Any]],
    source: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generation_rows = [_make_generation_row(episode) for episode in episodes]
    phase_a_rows = [_make_phase_a_row(episode) for episode in episodes]
    simulation_rows = [_make_simulation_row(episode) for episode in episodes]
    decision_rows = [_make_decision_row(episode) for episode in episodes]
    snapshot_dir = out_dir / "snapshot"
    snapshot_dir.mkdir(exist_ok=True)

    (out_dir / "memrl_episodes.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in episodes),
        encoding="utf-8",
    )
    (snapshot_dir / "memrl_snapshot.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in episodes),
        encoding="utf-8",
    )
    _write_json(out_dir / "memrl_generation_train.json", generation_rows)
    _write_json(out_dir / "memrl_phase_a_train.json", phase_a_rows)
    _write_json(out_dir / "memrl_simulation_train.json", simulation_rows)
    _write_json(out_dir / "memrl_decision_train.json", decision_rows)

    by_profile: dict[str, int] = {}
    positive = 0
    abstain = 0
    by_source: dict[str, int] = {}
    for episode in episodes:
        profile = str(episode["labels"].get("profile_id", "default"))
        episode_source = str(episode.get("source", "unknown"))
        by_profile[profile] = by_profile.get(profile, 0) + 1
        by_source[episode_source] = by_source.get(episode_source, 0) + 1
        if episode["decision"]["should_intervene"]:
            positive += 1
        else:
            abstain += 1
    info = {
        "episode_count": len(episodes),
        "generation_count": len(generation_rows),
        "phase_a_count": len(phase_a_rows),
        "simulation_count": len(simulation_rows),
        "decision_count": len(decision_rows),
        "source": source,
        "domain": "mobile_routine",
        "profiles": by_profile,
        "sources": by_source,
        "positive_episodes": positive,
        "abstain_episodes": abstain,
        "created_at": utc_now(),
    }
    if extra_info:
        info.update(extra_info)
    _write_json(out_dir / "dataset_info.json", info)
    _write_json(out_dir / "memrl_summary.json", info)
    _write_json(
        snapshot_dir / "memrl_meta.json",
        {
            "source": source,
            "count": len(episodes),
            "created_at": info["created_at"],
        },
    )
    return info


def build_knowu_routine_bundle(
    *,
    output_dir: str | Path | None = None,
    task_set_path: str | None = None,
    max_log_items: int = 24,
    users: set[str] | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir) if output_dir is not None else repo_root() / DEFAULT_BUNDLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = TaskRegistry(task_set_path=task_set_path)
    episodes = []
    for task_name in sorted(registry.list_tasks()):
        task = registry.get_task(task_name)
        if "routine" not in set(getattr(task, "task_tags", set()) or set()):
            continue
        if users and _profile_id(task_name) not in users:
            continue
        episodes.append(_episode_from_task(task, max_log_items=max_log_items))

    return _write_episode_bundle(
        out_dir=out_dir,
        episodes=episodes,
        source="knowu_routine_task_oracle",
        extra_info={"uses_task_expectation": True},
    )


def _profile_paths(profile_dir: Path, users: set[str] | None) -> list[Path]:
    paths = []
    for path in sorted(profile_dir.glob("*.yaml")):
        if path.stem == "template":
            continue
        if users and path.stem not in users:
            continue
        paths.append(path)
    return paths


def build_knowu_profile_memory_bundle(
    *,
    output_dir: str | Path | None = None,
    profile_dir: str | Path | None = None,
    max_log_items: int = 24,
    users: set[str] | None = None,
    negatives_per_profile: int = 4,
) -> dict[str, Any]:
    """Build non-test MemRL memories from KnowU profiles and logs.

    This mode deliberately does not instantiate KnowU tasks and does not read
    task.expectation, so it can be used as a training bootstrap for held-out
    KnowU benchmark tasks.
    """

    out_dir = Path(output_dir) if output_dir is not None else repo_root() / DEFAULT_BUNDLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_profile_dir = (
        Path(profile_dir)
        if profile_dir is not None
        else repo_root() / "src" / "knowu_bench" / "user_profile"
    )
    episodes: list[dict[str, Any]] = []
    habit_count = 0
    for profile_path in _profile_paths(resolved_profile_dir, users):
        profile_id = profile_path.stem
        loader = UserProfileLoader(str(profile_path))
        profile = loader.user_profile
        habits = profile.get("habits", {}) if isinstance(profile.get("habits"), dict) else {}
        for habit_name, habit in sorted(habits.items()):
            if not isinstance(habit, dict):
                continue
            habit_count += 1
            episodes.append(
                _episode_from_profile_habit(
                    profile_id=profile_id,
                    habit_name=habit_name,
                    habit=habit,
                    profile=profile,
                    max_log_items=max_log_items,
                )
            )
        for index in range(max(0, negatives_per_profile)):
            episodes.append(
                _background_episode_from_profile(
                    profile_id=profile_id,
                    profile=profile,
                    max_log_items=max_log_items,
                    index=index,
                )
            )

    return _write_episode_bundle(
        out_dir=out_dir,
        episodes=episodes,
        source="knowu_profile_habit_synthetic",
        extra_info={
            "uses_task_expectation": False,
            "profile_dir": str(resolved_profile_dir),
            "habit_positive_episodes": habit_count,
            "background_negative_episodes": len(episodes) - habit_count,
        },
    )


def build_knowu_profile_task_matrix_bundle(
    *,
    output_dir: str | Path | None = None,
    profile_dir: str | Path | None = None,
    max_log_items: int = 24,
    users: set[str] | None = None,
    target_count: int = 256,
) -> dict[str, Any]:
    """Build a larger profile x routine-family memory matrix.

    Labels come from whether a profile contains the habit key associated with a
    routine family. The generator never instantiates KnowU tasks and never reads
    task.expectation.
    """

    out_dir = Path(output_dir) if output_dir is not None else repo_root() / DEFAULT_BUNDLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_profile_dir = (
        Path(profile_dir)
        if profile_dir is not None
        else repo_root() / "src" / "knowu_bench" / "user_profile"
    )
    profiles: list[tuple[str, dict[str, Any]]] = []
    for profile_path in _profile_paths(resolved_profile_dir, users):
        loader = UserProfileLoader(str(profile_path))
        profiles.append((profile_path.stem, loader.user_profile))

    base_pairs = [
        (profile_id, profile, spec)
        for spec in ROUTINE_TASK_SPECS
        for profile_id, profile in profiles
    ]
    episodes: list[dict[str, Any]] = []
    variant_index = 0
    scenarios_per_pair = 3 if target_count <= len(base_pairs) * 3 else 4
    target_count = len(base_pairs) * scenarios_per_pair
    while variant_index < scenarios_per_pair and base_pairs:
        for profile_id, profile, spec in base_pairs:
            episodes.append(
                _task_matrix_episode(
                    profile_id=profile_id,
                    profile=profile,
                    spec=spec,
                    variant_index=variant_index,
                    max_log_items=max_log_items,
                )
            )
        variant_index += 1

    by_task_family: dict[str, int] = {}
    for episode in episodes:
        family = str(episode["labels"].get("task_family", "unknown"))
        by_task_family[family] = by_task_family.get(family, 0) + 1

    return _write_episode_bundle(
        out_dir=out_dir,
        episodes=episodes,
        source="knowu_profile_task_matrix_synthetic",
        extra_info={
            "uses_task_expectation": False,
            "profile_dir": str(resolved_profile_dir),
            "target_count": target_count,
            "routine_task_families": len(ROUTINE_TASK_SPECS),
            "base_profile_task_pairs": len(base_pairs),
            "scenarios_per_profile_task_pair": scenarios_per_pair,
            "variants_generated": variant_index,
            "task_families": by_task_family,
        },
    )
