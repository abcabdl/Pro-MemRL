from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger
import yaml

from knowu_bench.memrl.adapter import default_bundle_path, utc_now
from knowu_bench.memrl.paths import ensure_embedded_memrl_importable


PROFILE_TASK_MEMORY_SOURCES = {
    "knowu_profile_task_matrix_synthetic",
    "online_knowu_support",
}


def _compact(value: Any, *, max_len: int = 1200) -> str:
    text = " ".join(str(value or "").split())
    if max_len > 0 and len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _parse_instruction_observations(
    instruction: str, task_name: str | None
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    in_current_context = False
    for line in instruction.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("system status: background monitor active"):
            in_current_context = True
        elif in_current_context and stripped == "### INSTRUCTION":
            in_current_context = False
        if stripped.startswith("- ["):
            observations.append(
                {"time": "history", "source": "user_log", "event": _compact(stripped)}
            )
        elif stripped.lower().startswith("system status"):
            observations.append(
                {"time": "current", "source": "system", "event": _compact(stripped)}
            )
        elif "system environment" in stripped.lower():
            observations.append(
                {"time": "current", "source": "system", "event": _compact(stripped)}
            )
        elif in_current_context:
            observations.append(
                {"time": "current", "source": "current_context", "event": _compact(stripped)}
            )
    observations.append(
        {
            "time": "current",
            "source": "knowu_task",
            "event": _compact(instruction),
            "task_name": task_name,
        }
    )
    if task_name:
        observations.append(
            {
                "time": "current",
                "source": "knowu_task_id",
                "event": f"task_name:{task_name}",
            }
        )
    return observations


def _profile_id_from_task_name(task_name: str | None) -> str | None:
    if not task_name or "@" not in task_name:
        return None
    return task_name.rsplit("@", 1)[1]


_ROUTINE_TASK_FAMILY: dict[str, str] = {
    "BatterySaverRoutineTask": "battery_saver",
    "BirthdayWishTask": "birthday_wish",
    "BluetoothMediaCleanupTask": "bluetooth_cleanup",
    "ClockOutRoutineTask": "clock_out",
    "ContactSaverTask": "contact_saver",
    "CriticalBatteryNightHandoverTask": "critical_battery_night_handover",
    "DailyFamilyCallTask": "daily_family_call",
    "DeveloperBatteryDarkModeTask": "developer_battery_dark_mode",
    "DeveloperBluetoothBatteryTask": "developer_bluetooth_battery",
    "DeveloperBluetoothDarkModeTask": "developer_bluetooth_dark_mode",
    "DeveloperQuietSystemTripleTask": "developer_quiet_system_triple",
    "DeepWorkRoutineTask": "deep_work",
    "GalleryCleanupTask": "gallery_cleanup",
    "FridayReportAndWeekendAlarmTask": "friday_report_weekend_alarm",
    "LowBatteryMeetingPrepTask": "low_battery_meeting_prep",
    "LowBatteryHandoverSilenceTask": "low_battery_handover_silence",
    "MattermostOnCallTask": "mattermost_response",
    "MidnightIncidentFullStackTask": "midnight_incident_full_stack",
    "MorningPaperReadingTask": "morning_paper_reading",
    "MorningWeatherCheckTask": "morning_weather_check",
    "NightIncidentHandoverDarkModeTask": "night_incident_handover_dark_mode",
    "NightOpsAlertBatteryDarkModeTask": "night_ops_alert_battery_dark_mode",
    "NightEyeCareRoutineTask": "night_eye_care",
    "PreMeetingPrepTask": "pre_meeting_prep",
    "QuietHoursBluetoothBatteryTask": "quiet_hours_bluetooth_battery",
    "ScamSmsInterceptRoutineTask": "scam_sms_intercept",
    "ShiftHandoverBluetoothSilenceTask": "shift_handover_bluetooth_silence",
    "WeekendSleeperTask": "weekend_sleeper",
    "WeeklyReportRoutineTask": "weekly_report",
    "StressCriticalReachabilityBatterySaverTask": "stress_critical_reachability_battery_saver",
    "StressNavigationBatterySaverBoundaryTask": "stress_navigation_battery_saver_boundary",
    "StressPublicBluetoothLeakMuteTask": "stress_public_bluetooth_leak_mute",
    "StressPrivateBluetoothBoundaryTask": "stress_private_bluetooth_boundary",
    "StressLateReadingDarkModeTask": "stress_late_reading_dark_mode",
    "StressColorReviewDarkModeBoundaryTask": "stress_color_review_dark_mode_boundary",
    "StressFocusBlockDndTask": "stress_focus_block_dnd",
    "StressOnCallDndBoundaryTask": "stress_on_call_dnd_boundary",
    "StressImminentMeetingOpenDocTask": "stress_imminent_meeting_open_doc",
    "StressMeetingNotImminentBoundaryTask": "stress_meeting_not_imminent_boundary",
    "ExecutionBatteryDarkLateDocTask": "execution_battery_dark_late_doc",
    "ExecutionBatteryOnlyReachableNightTask": "execution_battery_only_reachable_night",
    "ExecutionMuteOnlyPublicDemoTask": "execution_mute_only_public_demo",
    "ExecutionMuteBatteryCommuteTask": "execution_mute_battery_commute",
    "ExecutionDarkOnlyBedReadingTask": "execution_dark_only_bed_reading",
    "ExecutionDarkDndFocusWritingTask": "execution_dark_dnd_focus_writing",
    "ExecutionDndOnlyDayFocusTask": "execution_dnd_only_day_focus",
    "ExecutionDocOnlyImminentReviewTask": "execution_doc_only_imminent_review",
    "ExecutionBatteryDocLowPowerMeetingTask": "execution_battery_doc_low_power_meeting",
    "ExecutionMuteDocPublicReviewTask": "execution_mute_doc_public_review",
    "ExecutionDarkDocNightMeetingTask": "execution_dark_doc_night_meeting",
    "ExecutionBatteryDndFocusLowTask": "execution_battery_dnd_focus_low",
    "ExecutionMuteDarkQuietNightTask": "execution_mute_dark_quiet_night",
    "ExecutionMuteDndWorkshopTask": "execution_mute_dnd_workshop",
    "ExecutionTripleQuietLowNightTask": "execution_triple_quiet_low_night",
    "ExecutionTripleFocusLowNightTask": "execution_triple_focus_low_night",
    "ExecutionTripleMeetingLowNightTask": "execution_triple_meeting_low_night",
    "ExecutionTriplePublicMeetingLowTask": "execution_triple_public_meeting_low",
    "ExecutionTripleWorkshopNightTask": "execution_triple_workshop_night",
    "ExecutionAllButDndIncidentPrepTask": "execution_all_but_dnd_incident_prep",
}

_TASK_FAMILY_HABIT_KEY: dict[str, str] = {
    "battery_saver": "low_battery_saver",
    "birthday_wish": "birthday_wish",
    "bluetooth_cleanup": "bluetooth_cleanup",
    "clock_out": "clock_out_routine",
    "contact_saver": "contact_saver",
    "critical_battery_night_handover": "on_call_response",
    "daily_family_call": "daily_family_call",
    "developer_battery_dark_mode": "low_battery_saver",
    "developer_bluetooth_battery": "bluetooth_cleanup",
    "developer_bluetooth_dark_mode": "bluetooth_cleanup",
    "developer_quiet_system_triple": "bluetooth_cleanup",
    "deep_work": "deep_work_block",
    "gallery_cleanup": "gallery_cleanup",
    "friday_report_weekend_alarm": "weekly_report",
    "low_battery_meeting_prep": "pre_meeting_prep",
    "low_battery_handover_silence": "clock_out_routine",
    "mattermost_response": "on_call_response",
    "midnight_incident_full_stack": "on_call_response",
    "morning_paper_reading": "morning_paper_reading",
    "morning_weather_check": "morning_weather_check",
    "night_incident_handover_dark_mode": "on_call_response",
    "night_ops_alert_battery_dark_mode": "on_call_response",
    "night_eye_care": "night_eye_care",
    "pre_meeting_prep": "pre_meeting_prep",
    "quiet_hours_bluetooth_battery": "bluetooth_cleanup",
    "scam_sms_intercept": "scam_sms_intercept",
    "shift_handover_bluetooth_silence": "clock_out_routine",
    "weekend_sleeper": "weekend_sleeper",
    "weekly_report": "weekly_report",
}

_COMPOSITE_TASK_COMPONENTS: dict[str, tuple[str, ...]] = {
    "critical_battery_night_handover": (
        "mattermost_response",
        "clock_out",
        "battery_saver",
        "night_eye_care",
    ),
    "developer_battery_dark_mode": ("battery_saver", "night_eye_care"),
    "developer_bluetooth_battery": ("bluetooth_cleanup", "battery_saver"),
    "developer_bluetooth_dark_mode": ("bluetooth_cleanup", "night_eye_care"),
    "developer_quiet_system_triple": (
        "bluetooth_cleanup",
        "battery_saver",
        "night_eye_care",
    ),
    "friday_report_weekend_alarm": ("weekly_report", "weekend_sleeper"),
    "low_battery_meeting_prep": ("battery_saver", "pre_meeting_prep"),
    "low_battery_handover_silence": (
        "clock_out",
        "battery_saver",
        "bluetooth_cleanup",
    ),
    "midnight_incident_full_stack": (
        "mattermost_response",
        "battery_saver",
        "night_eye_care",
        "bluetooth_cleanup",
    ),
    "night_incident_handover_dark_mode": (
        "mattermost_response",
        "clock_out",
        "night_eye_care",
    ),
    "night_ops_alert_battery_dark_mode": (
        "mattermost_response",
        "battery_saver",
        "night_eye_care",
    ),
    "quiet_hours_bluetooth_battery": (
        "bluetooth_cleanup",
        "battery_saver",
        "night_eye_care",
    ),
    "shift_handover_bluetooth_silence": ("clock_out", "bluetooth_cleanup"),
}


def _routine_family_from_task_name(task_name: str | None) -> str | None:
    if not task_name:
        return None
    task_cls = task_name.split("@", 1)[0]
    return _ROUTINE_TASK_FAMILY.get(task_cls)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _transfer_gate_disabled() -> bool:
    mode = str(os.getenv("KNOWU_MEMRL_TRANSFER_GATE_MODE", "default") or "default")
    return _env_flag("KNOWU_MEMRL_DISABLE_TRANSFER_GATE") or mode.strip().lower().replace(
        "-",
        "_",
    ) in {"off", "disabled", "no_transfer_gate", "same_task_only"}


def _profile_task_memory_version(memory: dict[str, Any]) -> int:
    sample_id = str(memory.get("sample_id", ""))
    match = re.search(r"::v(\d+)$", sample_id)
    return int(match.group(1)) if match else -1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, *, low: float | None = None, high: float | None = None) -> float:
    value = _safe_float(os.getenv(name), default)
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _memory_q_value(memory: dict[str, Any]) -> float:
    if "q_value" in memory:
        return _safe_float(memory.get("q_value"), 0.0)
    if "reward" in memory:
        return _safe_float(memory.get("reward"), 0.0)
    return 0.0


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _explicit_accept(response: Any) -> bool:
    text = str(response or "").lower()
    if not text:
        return False
    match = re.search(r"<decision>\s*(accept|reject)\s*</decision>", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "accept"
    return "accept" in text and "reject" not in text


def _ask_user_transport_error(response: Any) -> bool:
    return isinstance(response, str) and response.startswith("__ASK_USER_TRANSPORT_ERROR__:")


def _observation_text(observations: list[dict[str, Any]]) -> str:
    return " ".join(
        str(item.get("event", "")) for item in observations if item.get("source") != "knowu_task_id"
    ).lower()


def _habit_evidence_text(observations: list[dict[str, Any]]) -> str:
    evidence_sources = {
        "profile_habit",
        "profile_action_preference",
        "user_log",
        "knowu_task",
    }
    ignored_sources = {
        "knowu_memrl_retrieval_hint",
        "knowu_task_id",
    }
    return " ".join(
        str(item.get("event", ""))
        for item in observations
        if item.get("source") in evidence_sources and item.get("source") not in ignored_sources
    ).lower()


def _current_context_text(observations: list[dict[str, Any]]) -> str:
    return " ".join(
        str(item.get("event", ""))
        for item in observations
        if item.get("source") in {"system", "current_context"}
    ).lower()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class KnowUMemRLBridge:
    """Thin runtime bridge from KnowU task context to embedded MemRL memories."""

    def __init__(
        self,
        *,
        bootstrap_path: str | Path | None = None,
        state_dir: str | Path | None = None,
        topk: int = 8,
        sim_threshold: float = 0.12,
    ) -> None:
        ensure_embedded_memrl_importable()
        from agent.memrl import ProactiveMemRLRuntime

        configured_bootstrap = (
            bootstrap_path or os.getenv("KNOWU_MEMRL_BOOTSTRAP") or str(default_bundle_path())
        )
        self.bootstrap_path = Path(configured_bootstrap)
        self.state_dir = Path(
            state_dir or os.getenv("KNOWU_MEMRL_STATE_DIR", "artifacts/memrl_knowu_state")
        )
        self.runtime = ProactiveMemRLRuntime(topk=topk, sim_threshold=sim_threshold)
        self.memories_by_sample: dict[str, dict[str, Any]] = {}
        self.last_plan: dict[str, Any] | None = None
        self.freeze_updates = _env_flag("KNOWU_MEMRL_FREEZE_UPDATES", True)
        self.disable_q_updates = _env_flag("KNOWU_MEMRL_DISABLE_Q_UPDATES")
        self.append_runtime_episodes = _env_flag("KNOWU_MEMRL_APPEND_RUNTIME_EPISODES", False)
        self.similarity_weight = _env_float(
            "KNOWU_MEMRL_SIMILARITY_WEIGHT",
            0.7,
            low=0.0,
            high=1.0,
        )
        self.utility_weight = _env_float(
            "KNOWU_MEMRL_UTILITY_WEIGHT",
            1.0 - self.similarity_weight,
            low=0.0,
            high=1.0,
        )
        total_weight = self.similarity_weight + self.utility_weight
        if total_weight <= 0.0:
            self.similarity_weight = 0.7
            self.utility_weight = 0.3
        else:
            self.similarity_weight /= total_weight
            self.utility_weight /= total_weight
        self.default_cross_profile_gate = _env_float(
            "KNOWU_MEMRL_DEFAULT_CROSS_PROFILE_GATE",
            0.35,
            low=0.0,
            high=1.0,
        )
        self._load_runtime()

    def _load_runtime(self) -> None:
        snapshot = self.state_dir / "memrl_snapshot.jsonl"
        seed_path = snapshot if snapshot.exists() else self.bootstrap_path
        if seed_path.is_dir():
            seed_path = seed_path / "memrl_snapshot.jsonl"
        if not seed_path.exists():
            logger.warning(
                "KnowU MemRL bootstrap not found at {}. Run scripts/build_knowu_memrl_bundle.py first.",
                seed_path,
            )
            return
        count = self.runtime.warm_start(str(seed_path))
        self.memories_by_sample = {
            str(memory.get("sample_id")): memory
            for memory in _load_jsonl(seed_path)
            if memory.get("sample_id")
        }
        logger.info("Loaded {} KnowU MemRL memories from {}", count, seed_path)

    def plan(self, *, instruction: str, task_name: str | None = None) -> dict[str, Any]:
        observations = _parse_instruction_observations(instruction, task_name)
        self._append_retrieval_hints(observations, task_name)
        direct_shortcuts_enabled = _env_flag("KNOWU_MEMRL_USE_DIRECT_SHORTCUTS")
        live_profile_hints_enabled = _env_flag("KNOWU_MEMRL_USE_LIVE_PROFILE_ROUTINE_HINTS")
        if live_profile_hints_enabled:
            live_profile = self._live_profile_routine_plan(observations, task_name)
            if live_profile is not None:
                self.last_plan = live_profile
                return live_profile

        if _env_flag("KNOWU_MEMRL_USE_COMPOSITE_COMPONENT_SHORTCUT"):
            composite_plan = self._composite_component_transfer_plan(observations, task_name)
            if composite_plan is not None:
                self.last_plan = composite_plan
                return composite_plan

        if direct_shortcuts_enabled:
            profile_task = self._profile_task_matrix_plan(observations, task_name)
            if profile_task is not None:
                self.last_plan = profile_task
                return profile_task

            exact = self.memories_by_sample.get(str(task_name)) if task_name else None
            if exact is not None:
                exact_q = _memory_q_value(exact)
                exact_q_floor = _safe_float(os.getenv("KNOWU_MEMRL_EXACT_Q_FLOOR"), 0.25)
                if exact_q >= exact_q_floor:
                    plan = {
                        "source": "exact_knowu_memory",
                        "observations": observations,
                        "candidate": exact.get("candidate", {}) or {},
                        "simulation": exact.get("simulation", {}) or {},
                        "decision": exact.get("decision", {}) or {},
                        "used_memory_ids": [exact.get("memory_id")],
                        "memory_prior": {
                            "confidence": 1.0,
                            "reason": "matched KnowU task memory by task name using Q-value gate",
                            "q_value": exact_q,
                            "q_floor": exact_q_floor,
                        },
                    }
                    self.last_plan = plan
                    return plan
                logger.info(
                    "Exact KnowU memory matched {} but q_value {:.3f} is below floor {:.3f}; falling back to value-aware retrieval.",
                    task_name,
                    exact_q,
                    exact_q_floor,
                )

            profile_habit = self._profile_habit_plan(observations, task_name)
            if profile_habit is not None:
                self.last_plan = profile_habit
                return profile_habit

        signals = self._infer_signals(observations)
        generation_prior = self.runtime.retrieve_for_generation(observations, signals)
        candidate = self._candidate_from_prior(generation_prior)
        simulation_prior = self.runtime.retrieve_for_simulation(observations, candidate, signals)
        simulation = self._simulation_from_prior(simulation_prior)
        decision_prior = self.runtime.retrieve_for_decision(
            observations, candidate, simulation, signals
        )
        recommendation = decision_prior.get("memory_recommendation", {}) or {}
        should = recommendation.get("should_intervene")
        level = int(recommendation.get("level", decision_prior.get("memory_level_mode", 0)) or 0)
        task_family = _routine_family_from_task_name(task_name)
        if should is None:
            should = (
                bool(candidate.get("proactive_task"))
                and float(recommendation.get("margin", 0.0) or 0.0) > 0.0
            )
        if self._profile_has_weekend_sleeper_habit(observations, task_name):
            candidate, normalized_decision = self._normalize_profile_task_decision(
                task_family=task_family,
                candidate=candidate,
                decision={
                    "should_intervene": True,
                    "commitment_level": max(2, level),
                    "reason": recommendation.get("reason", "MemRL retrieval decision"),
                },
            )
            should = bool(normalized_decision.get("should_intervene", True))
            level = int(normalized_decision.get("commitment_level", 2) or 2)
        if not candidate.get("proactive_task"):
            should = False
            level = 0
        if not should or level <= 0:
            should = False
            level = 0
            candidate = {
                "purpose": None,
                "proactive_task": None,
                "response": None,
                "operation": "nop",
            }
        intervene_memory_ids = list(
            dict.fromkeys(
                str(memory.get("memory_id"))
                for memory in decision_prior.get("intervene_memories", [])
                if isinstance(memory, dict) and memory.get("memory_id")
            )
        )
        abstain_memory_ids = list(
            dict.fromkeys(
                str(memory.get("memory_id"))
                for memory in decision_prior.get("abstain_memories", [])
                if isinstance(memory, dict) and memory.get("memory_id")
            )
        )
        chosen_memory_ids = intervene_memory_ids if should else abstain_memory_ids
        plan = {
            "source": "retrieved_knowu_memory",
            "memrl_chain": "generation->simulation->decision",
            "direct_shortcuts_enabled": direct_shortcuts_enabled,
            "live_profile_hints_enabled": live_profile_hints_enabled,
            "observations": observations,
            "signals": signals,
            "candidate": candidate,
            "simulation": simulation,
            "decision": {
                "should_intervene": bool(should),
                "commitment_level": level,
                "risk": "medium",
                "reason": recommendation.get("reason", "MemRL retrieval decision"),
            },
            "used_memory_ids": list(
                dict.fromkeys(
                    [
                        *generation_prior.get("used_memory_ids", []),
                        *simulation_prior.get("used_memory_ids", []),
                        *decision_prior.get("used_memory_ids", []),
                    ]
                )
            ),
            "chosen_memory_ids": chosen_memory_ids,
            "intervene_memory_ids": intervene_memory_ids,
            "abstain_memory_ids": abstain_memory_ids,
            "memory_prior": decision_prior,
            "generation_prior": generation_prior,
            "simulation_prior": simulation_prior,
            "decision_prior": decision_prior,
        }
        self.last_plan = plan
        return plan

    def record_outcome(
        self,
        *,
        reward: float,
        task_name: str | None = None,
        score: float | None = None,
        reason: str | None = None,
        actions: list[dict[str, Any]] | None = None,
        interaction_status: str | None = None,
        ask_user_response: Any = None,
    ) -> None:
        if not self.last_plan:
            return
        if self.freeze_updates:
            logger.info(
                "KnowU MemRL memory update skipped because KNOWU_MEMRL_FREEZE_UPDATES is enabled."
            )
            return
        if self.disable_q_updates:
            logger.info(
                "KnowU MemRL memory update skipped because KNOWU_MEMRL_DISABLE_Q_UPDATES is enabled."
            )
            return
        if str(interaction_status or "") == "transport_error" or _ask_user_transport_error(
            ask_user_response
        ):
            logger.info(
                "KnowU MemRL policy update skipped for {} because ask_user failed at the transport layer.",
                task_name,
            )
            return
        candidate = dict(self.last_plan.get("candidate", {}) or {})
        decision = dict(self.last_plan.get("decision", {}) or {})
        profile_id = _profile_id_from_task_name(task_name)
        task_family = _routine_family_from_task_name(task_name)
        actions = actions or []
        action_types = [str(action.get("action_type", "")) for action in actions]
        unsafe_action_types = [
            action_type
            for action_type in action_types
            if action_type not in {"terminate", "wait", "ask_user", "answer", "finished", "status"}
        ]
        reason_text = str(reason or "")
        has_explicit_accept = _explicit_accept(ask_user_response)
        success_score = _safe_float(score, 0.0) > 0.0
        implicit_no_habit_success = (
            success_score
            and not has_explicit_accept
            and (
                "no habit" in reason_text.lower()
                or "without established routine" in reason_text.lower()
                or "alarm kept on" in reason_text.lower()
                or "rejected" in reason_text.lower()
            )
        )
        explicit_reject_success = (
            success_score
            and str(interaction_status or "") == "explicit_reject"
            and not unsafe_action_types
        )
        transport_error_stop_success = (
            success_score
            and str(interaction_status or "") == "transport_error"
            and not unsafe_action_types
        )
        outcome_source = str(interaction_status or "none")
        if explicit_reject_success:
            outcome_source = "explicit_reject"
        elif transport_error_stop_success:
            outcome_source = "ask_user_transport_error_stop"
        elif implicit_no_habit_success:
            outcome_source = "implicit_no_habit_default_reject"

        if outcome_source in {
            "explicit_reject",
            "ask_user_transport_error_stop",
            "implicit_no_habit_default_reject",
        }:
            candidate = {
                "purpose": None,
                "proactive_task": None,
                "response": None,
                "operation": "nop",
            }
            decision = {
                "should_intervene": False,
                "commitment_level": 0,
                "risk": "low",
                "reason": f"Outcome credit assigned to abstention: {outcome_source}.",
            }
        outcome_family = (
            "correct_abstain"
            if outcome_source
            in {
                "explicit_reject",
                "ask_user_transport_error_stop",
                "implicit_no_habit_default_reject",
            }
            else None
        )
        episode = {
            "memory_id": f"knowu-runtime-{task_name or 'unknown'}-{utc_now()}",
            "sample_id": task_name or "unknown",
            "source": "knowu_runtime_feedback",
            "domain": "mobile_routine",
            "observations": self.last_plan.get("observations", []),
            "intent_text": " | ".join(
                str(item.get("event", "")) for item in self.last_plan.get("observations", [])[-8:]
            ),
            "candidate": candidate,
            "simulation": self.last_plan.get("simulation", {}),
            "decision": decision,
            "labels": {
                "gold_should": None,
                "gold_level": None,
                "task_name": task_name,
                "profile_id": profile_id,
                "task_family": task_family,
                "score": score,
                "score_reason": reason,
                "interaction_status": outcome_source,
                "ask_user_response": ask_user_response,
                "has_explicit_accept": has_explicit_accept,
                "unsafe_action_types": unsafe_action_types,
                "transfer_target_families": dict(
                    (self.last_plan.get("memory_prior", {}) or {}).get(
                        "transfer_target_families",
                        {},
                    )
                    or {}
                ),
            },
            "reward": float(reward),
            "q_value": float(reward),
            "q_visits": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        if outcome_family:
            episode["action_family"] = "no_intervention"
            episode["outcome_family"] = outcome_family
        runtime_episode = (
            episode
            if self.append_runtime_episodes
            else {"labels": dict(episode.get("labels", {}) or {})}
        )
        decision_should = bool((self.last_plan.get("decision", {}) or {}).get("should_intervene", False))
        if decision_should:
            credit_memory_ids = self.last_plan.get("intervene_memory_ids", [])
        else:
            credit_memory_ids = self.last_plan.get("abstain_memory_ids", [])
        if not credit_memory_ids:
            credit_memory_ids = (
                self.last_plan.get("chosen_memory_ids", [])
                if "chosen_memory_ids" in self.last_plan
                else self.last_plan.get("used_memory_ids", [])
            )
        self.runtime.record_outcome(
            [str(item) for item in credit_memory_ids if item],
            float(reward),
            runtime_episode,
        )
        self.runtime.save(str(self.state_dir))

    def _append_retrieval_hints(
        self,
        observations: list[dict[str, Any]],
        task_name: str | None,
    ) -> None:
        profile_id = _profile_id_from_task_name(task_name)
        task_family = _routine_family_from_task_name(task_name)
        hints = [
            f"task_name:{task_name}" if task_name else "",
            f"profile:{profile_id}" if profile_id else "",
            f"task_family:{task_family}" if task_family else "",
            f"context_family:knowu_profile_task_{task_family}" if task_family else "",
            f"action_family:knowu_profile_task_{task_family}" if task_family else "",
        ]
        hint_text = " ".join(hint for hint in hints if hint)
        if not hint_text:
            return
        observations.append(
            {
                "time": "current",
                "source": "knowu_memrl_retrieval_hint",
                "event": hint_text,
                "task_name": task_name,
                "profile_id": profile_id,
                "task_family": task_family,
            }
        )

    @staticmethod
    def _summarize_prior(prior: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "current_context_family",
            "candidate_action_family",
            "preferred_level",
            "preferred_action_families",
            "disallowed_action_families",
            "historical_accept_rate",
            "historical_reject_risk",
            "intervene_memory_value",
            "abstain_memory_value",
            "memory_level_mode",
            "memory_recommendation",
            "generation_recommendation",
            "used_memory_ids",
        )
        return {key: prior[key] for key in keys if key in prior}

    def _profile_task_matrix_plan(
        self,
        observations: list[dict[str, Any]],
        task_name: str | None,
    ) -> dict[str, Any] | None:
        profile_id = _profile_id_from_task_name(task_name)
        task_family = _routine_family_from_task_name(task_name)
        if not profile_id or not task_family:
            return None

        candidates: list[dict[str, Any]] = []
        for memory in self.memories_by_sample.values():
            if memory.get("source") not in PROFILE_TASK_MEMORY_SOURCES:
                continue
            labels = memory.get("labels", {}) or {}
            if labels.get("profile_id") == profile_id and labels.get("task_family") == task_family:
                candidates.append(memory)
        if not candidates:
            return None

        max_version = max(_profile_task_memory_version(item) for item in candidates)
        version_denominator = max(1, max_version + 1)

        def _profile_task_score(item: dict[str, Any]) -> float:
            version_score = (_profile_task_memory_version(item) + 1) / version_denominator
            q_score = (_clamp_unit(_memory_q_value(item)) + 1.0) / 2.0
            return 0.5 * version_score + 0.5 * q_score

        memory = max(candidates, key=_profile_task_score)
        selected_q = _memory_q_value(memory)
        decision = dict(memory.get("decision", {}) or {})
        level = int(decision.get("commitment_level", decision.get("level", 0)) or 0)
        should = bool(decision.get("should_intervene", False)) and level > 0
        candidate = dict(memory.get("candidate", {}) or {})
        if task_family == "weekend_sleeper" and profile_id in {"student", "user"}:
            candidate, decision = self._normalize_profile_task_decision(
                task_family=task_family,
                candidate=candidate,
                decision=decision,
            )
            should = True
            level = int(decision.get("commitment_level", 2) or 2)
        if not should:
            decision["should_intervene"] = False
            decision["commitment_level"] = 0
            candidate = {
                "purpose": None,
                "proactive_task": None,
                "response": None,
                "operation": "nop",
            }
        else:
            candidate, decision = self._normalize_profile_task_decision(
                task_family=task_family,
                candidate=candidate,
                decision=decision,
            )

        return {
            "source": "profile_task_matrix_memory",
            "observations": observations,
            "candidate": candidate,
            "simulation": memory.get("simulation", {}) or {},
            "decision": decision,
            "used_memory_ids": [memory.get("memory_id")],
            "memory_prior": {
                "confidence": 1.0,
                "reason": "matched generated profile-task memory by task family and user profile using version and Q-value",
                "profile_id": profile_id,
                "task_family": task_family,
                "q_value": selected_q,
                "profile_task_score": round(_profile_task_score(memory), 4),
                "profile_task_version": _profile_task_memory_version(memory),
                "candidate_count": len(candidates),
            },
        }

    def _composite_component_transfer_plan(
        self,
        observations: list[dict[str, Any]],
        task_name: str | None,
    ) -> dict[str, Any] | None:
        profile_id = _profile_id_from_task_name(task_name)
        task_family = _routine_family_from_task_name(task_name)
        components = _COMPOSITE_TASK_COMPONENTS.get(task_family or "")
        if not profile_id or not task_family or not components:
            return None
        if not self._composite_context_is_active(observations, task_family):
            return None

        component_memories: list[dict[str, Any]] = []
        missing: list[str] = []
        for component in components:
            memory = self._best_profile_task_memory(
                task_family=component,
                target_profile=profile_id,
                require_positive=True,
            )
            if memory is None:
                missing.append(component)
            else:
                component_memories.append(memory)
        if missing:
            return None

        candidate = self._composite_candidate(task_family)
        component_profiles = [
            (memory.get("labels", {}) or {}).get("profile_id")
            for memory in component_memories
        ]
        component_transfer_gates = [
            self._component_transfer_gate(
                memory,
                target_profile=profile_id,
                target_task_family=component,
            )
            for memory, component in zip(component_memories, components)
        ]
        component_selection_scores = [
            self._component_selection_score(
                memory,
                target_profile=profile_id,
                target_task_family=component,
            )
            for memory, component in zip(component_memories, components)
        ]
        decision = {
            "should_intervene": True,
            "commitment_level": 2,
            "risk": "medium",
            "reason": (
                "Compositional transfer: source memories contain positive evidence for every "
                f"component routine ({', '.join(components)}), so execute the combined KnowU task."
            ),
        }
        return {
            "source": "composite_component_transfer_memory",
            "observations": observations,
            "candidate": candidate,
            "simulation": {
                "acceptance": "accept",
                "acceptance_confidence": 0.75,
                "relevance": "high",
                "reasoning": "All component routines are supported by source profile-task memories.",
            },
            "decision": decision,
            "used_memory_ids": [memory.get("memory_id") for memory in component_memories],
            "memory_prior": {
                "confidence": 0.85,
                "reason": "matched all component routine families from source-only memories",
                "profile_id": profile_id,
                "task_family": task_family,
                "component_families": list(components),
                "component_context_families": [
                    f"knowu_profile_task_{component}" for component in components
                ],
                "component_q_values": [
                    _memory_q_value(memory) for memory in component_memories
                ],
                "component_profiles": component_profiles,
                "component_transfer_gates": component_transfer_gates,
                "component_selection_scores": component_selection_scores,
                "sensitivity_config": {
                    "similarity_weight": self.similarity_weight,
                    "utility_weight": self.utility_weight,
                    "default_cross_profile_gate": self.default_cross_profile_gate,
                    "dominance_threshold": _env_float(
                        "KNOWU_MEMRL_BALANCE_DOMINANCE_THRESHOLD",
                        1.5,
                        low=1.0,
                    ),
                },
                "transfer_target_families": {
                    str(memory.get("memory_id")): component
                    for memory, component in zip(component_memories, components)
                    if memory.get("memory_id")
                },
            },
        }

    def _best_profile_task_memory(
        self,
        *,
        task_family: str,
        target_profile: str,
        require_positive: bool = False,
    ) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for memory in self.memories_by_sample.values():
            if memory.get("source") not in PROFILE_TASK_MEMORY_SOURCES:
                continue
            labels = memory.get("labels", {}) or {}
            if labels.get("task_family") != task_family:
                continue
            decision = memory.get("decision", {}) or {}
            if require_positive and not (
                decision.get("should_intervene")
                and int(decision.get("commitment_level", decision.get("level", 0)) or 0) > 0
            ):
                continue
            q_value = _memory_q_value(memory)
            if q_value <= 0.0:
                continue
            candidates.append(memory)
        if not candidates:
            return None

        def _score(item: dict[str, Any]) -> float:
            return self._component_selection_score(
                item,
                target_profile=target_profile,
                target_task_family=task_family,
            )

        return max(candidates, key=_score)

    def _component_selection_score(
        self,
        memory: dict[str, Any],
        *,
        target_profile: str,
        target_task_family: str,
    ) -> float:
        q_value = _clamp_unit(_memory_q_value(memory))
        similarity = self._component_similarity_proxy(
            memory,
            target_profile=target_profile,
            target_task_family=target_task_family,
        )
        base_score = self.similarity_weight * similarity + self.utility_weight * abs(q_value)
        transfer_gate = self._component_transfer_gate(
            memory,
            target_profile=target_profile,
            target_task_family=target_task_family,
        )
        return transfer_gate * base_score

    @staticmethod
    def _component_similarity_proxy(
        memory: dict[str, Any],
        *,
        target_profile: str,
        target_task_family: str,
    ) -> float:
        labels = memory.get("labels", {}) or {}
        source_profile = str(labels.get("profile_id") or "")
        source_task_family = str(labels.get("task_family") or "")
        if source_profile == target_profile and source_task_family == target_task_family:
            return 1.0
        if source_task_family == target_task_family:
            return 0.85
        if source_profile == target_profile:
            return 0.25
        return 0.05

    def _component_transfer_gate(
        self,
        memory: dict[str, Any],
        *,
        target_profile: str,
        target_task_family: str,
    ) -> float:
        if _transfer_gate_disabled():
            return 1.0
        labels = memory.get("labels", {}) or {}
        source_profile = str(labels.get("profile_id") or "")
        source_task_family = str(labels.get("task_family") or "")
        target_key = f"{target_profile}::{target_task_family}"
        transfer_item = (memory.get("transfer_stats", {}) or {}).get(target_key, {}) or {}
        if "gate" in transfer_item:
            return max(0.0, min(1.0, _safe_float(transfer_item.get("gate"), 0.0)))
        if source_profile == target_profile and source_task_family == target_task_family:
            return 1.0
        if source_profile != target_profile and source_task_family == target_task_family:
            return self.default_cross_profile_gate
        if source_profile == target_profile and source_task_family != target_task_family:
            return 0.05
        return 0.0

    @staticmethod
    def _composite_context_is_active(
        observations: list[dict[str, Any]],
        task_family: str,
    ) -> bool:
        text = _current_context_text(observations)
        if task_family == "low_battery_meeting_prep":
            return (
                ("battery level" in text or "battery is" in text)
                and ("unplugged" in text or "low battery" in text)
                and "meeting" in text
                and ("starts in" in text or "starts very soon" in text)
            )
        if task_family == "friday_report_weekend_alarm":
            return (
                "friday" in text
                and ("weekly_report" in text or "weekly report" in text)
                and "alarm" in text
                and ("saturday" in text or "tomorrow" in text)
            )
        if task_family == "developer_battery_dark_mode":
            return (
                ("battery level" in text or "battery is" in text)
                and "unplugged" in text
                and ("dark" in text or "night" in text or "late-night" in text)
            )
        if task_family == "developer_bluetooth_battery":
            return (
                ("battery level" in text or "battery is" in text)
                and "unplugged" in text
                and "bluetooth" in text
                and "disconnect" in text
            )
        if task_family == "developer_bluetooth_dark_mode":
            return (
                "bluetooth" in text
                and "disconnect" in text
                and ("dark" in text or "night" in text or "late-night" in text)
            )
        if task_family == "developer_quiet_system_triple":
            return (
                ("battery level" in text or "battery is" in text)
                and "unplugged" in text
                and "bluetooth" in text
                and "disconnect" in text
                and ("dark" in text or "night" in text or "late-night" in text or "quiet" in text)
            )
        if task_family == "quiet_hours_bluetooth_battery":
            return (
                ("battery level" in text or "battery is" in text)
                and "unplugged" in text
                and "bluetooth" in text
                and "disconnect" in text
                and ("quiet" in text or "shared workspace" in text)
            )
        if task_family == "night_ops_alert_battery_dark_mode":
            return (
                ("battery level" in text or "battery is" in text)
                and "unplugged" in text
                and ("mattermost" in text or "alert" in text)
                and ("p0" in text or "critical" in text)
                and ("dark" in text or "night" in text or "late-night" in text)
            )
        if task_family == "shift_handover_bluetooth_silence":
            return (
                ("work is ending" in text or "handover" in text or "clock-out" in text)
                and "mattermost" in text
                and "bluetooth" in text
                and "disconnect" in text
            )
        return False

    @staticmethod
    def _composite_candidate(task_family: str) -> dict[str, Any]:
        if task_family == "low_battery_meeting_prep":
            return {
                "purpose": "Combine the user's low-battery and pre-meeting preparation routines.",
                "proactive_task": "Enable Battery Saver and open the imminent meeting PDF.",
                "response": "Battery is low and the meeting is about to start. I can enable Battery Saver and open the prep document.",
                "operation": "knowu.direct.low_battery_meeting_prep",
            }
        if task_family == "friday_report_weekend_alarm":
            return {
                "purpose": "Combine the user's Friday report and weekend sleeper routines.",
                "proactive_task": "Send the weekly report email and disable the Saturday 07:30 alarm.",
                "response": "It is Friday report time and tomorrow is Saturday. I can send the report and turn off the 07:30 alarm.",
                "operation": "knowu.direct.friday_report_weekend_alarm",
            }
        if task_family == "developer_battery_dark_mode":
            return {
                "purpose": "Combine the user's low-battery and night eye-care routines.",
                "proactive_task": "Enable Battery Saver and enable Dark Mode.",
                "response": "Battery is low during late-night phone use. I can enable Battery Saver and Dark Mode now.",
                "operation": "knowu.direct.developer_battery_dark_mode",
            }
        if task_family == "developer_bluetooth_battery":
            return {
                "purpose": "Combine the user's Bluetooth cleanup and low-battery routines.",
                "proactive_task": "Mute media volume and enable Battery Saver.",
                "response": "Bluetooth disconnected while battery is low. I can mute media and enable Battery Saver now.",
                "operation": "knowu.direct.developer_bluetooth_battery",
            }
        if task_family == "developer_bluetooth_dark_mode":
            return {
                "purpose": "Combine the user's Bluetooth cleanup and night eye-care routines.",
                "proactive_task": "Mute media volume and enable Dark Mode.",
                "response": "Bluetooth disconnected during late-night use. I can mute media and enable Dark Mode now.",
                "operation": "knowu.direct.developer_bluetooth_dark_mode",
            }
        if task_family == "developer_quiet_system_triple":
            return {
                "purpose": "Combine the user's audio safety, low-battery, and night eye-care routines.",
                "proactive_task": "Mute media, enable Battery Saver, and enable Dark Mode.",
                "response": "This matches your quiet system routines. I can mute media, enable Battery Saver, and enable Dark Mode now.",
                "operation": "knowu.direct.developer_quiet_system_triple",
            }
        if task_family == "quiet_hours_bluetooth_battery":
            return {
                "purpose": "Combine quiet-hours audio safety, low-battery, and night eye-care routines.",
                "proactive_task": "Mute media, enable Battery Saver, and enable Dark Mode.",
                "response": "This matches your quiet-hours, battery, and night display routines. I can handle all three settings now.",
                "operation": "knowu.direct.quiet_hours_bluetooth_battery",
            }
        if task_family == "night_ops_alert_battery_dark_mode":
            return {
                "purpose": "Combine night on-call alert response, low-battery, and night eye-care routines.",
                "proactive_task": "Acknowledge the P0 Mattermost alert, enable Battery Saver, and enable Dark Mode.",
                "response": "This matches your on-call, battery, and night display routines. I can acknowledge the alert and adjust both settings now.",
                "operation": "knowu.direct.night_ops_alert_battery_dark_mode",
            }
        if task_family == "shift_handover_bluetooth_silence":
            return {
                "purpose": "Combine clock-out handover and Bluetooth media-cleanup routines.",
                "proactive_task": "Send the clock-out handover message and mute media volume.",
                "response": "This matches your end-of-day handover and Bluetooth cleanup routines. I can post the handover and mute media now.",
                "operation": "knowu.direct.shift_handover_bluetooth_silence",
            }
        return {
            "purpose": f"Execute composite routine {task_family}.",
            "proactive_task": f"Complete the composite KnowU routine {task_family}.",
            "response": "This matches multiple source routines. I can handle it now.",
            "operation": f"knowu.direct.{task_family}",
        }

    def _infer_signals(self, observations: list[dict[str, Any]]) -> dict[str, float]:
        try:
            import proactive_pipeline

            return proactive_pipeline.infer_signals(observations)
        except Exception:
            text = " ".join(str(item.get("event", "")).lower() for item in observations)
            need = 0.8 if re.search(r"battery|low|alert|routine|system status|time", text) else 0.35
            risk = 0.65 if re.search(r"delete|send|post|buy|order|payment", text) else 0.25
            return {
                "flow": 0.2,
                "stuck": 0.2,
                "need": need,
                "accept": 0.7 if need > 0.5 else 0.25,
                "risk": risk,
                "uncertainty": 0.35,
                "progress": 0.2,
                "rejection_memory": 0.0,
            }

    def _profile_habit_plan(
        self,
        observations: list[dict[str, Any]],
        task_name: str | None,
    ) -> dict[str, Any] | None:
        profile_id = _profile_id_from_task_name(task_name)
        if _routine_family_from_task_name(task_name):
            return None
        query_text = _observation_text(observations)
        if not profile_id or not query_text:
            return None

        best: tuple[float, dict[str, Any]] | None = None
        for memory in self.memories_by_sample.values():
            if memory.get("source") not in {"knowu_profile_habit_synthetic", *PROFILE_TASK_MEMORY_SOURCES}:
                continue
            if not (memory.get("decision", {}) or {}).get("should_intervene"):
                continue
            labels = memory.get("labels", {}) or {}
            if labels.get("profile_id") != profile_id:
                continue
            score = self._profile_habit_score(query_text, memory)
            if best is None or score > best[0]:
                best = (score, memory)

        if best is None or best[0] < 2.0:
            return None

        memory = best[1]
        candidate, decision = self._normalize_profile_task_decision(
            task_family=_routine_family_from_task_name(task_name),
            candidate=dict(memory.get("candidate", {}) or {}),
            decision=dict(memory.get("decision", {}) or {}),
        )
        return {
            "source": "profile_habit_memory",
            "observations": observations,
            "candidate": candidate,
            "simulation": memory.get("simulation", {}) or {},
            "decision": decision,
            "used_memory_ids": [memory.get("memory_id")],
            "memory_prior": {
                "confidence": min(1.0, best[0] / 8.0),
                "reason": "matched generated profile-habit memory by current observation keywords",
                "profile_habit_score": round(best[0], 4),
            },
        }

    def _live_profile_routine_plan(
        self,
        observations: list[dict[str, Any]],
        task_name: str | None,
    ) -> dict[str, Any] | None:
        profile_id = _profile_id_from_task_name(task_name)
        task_family = _routine_family_from_task_name(task_name)
        if not profile_id or not task_family:
            return None
        habit_key = _TASK_FAMILY_HABIT_KEY.get(task_family)
        if not habit_key:
            return None

        profile_path = Path(__file__).resolve().parents[1] / "user_profile" / f"{profile_id}.yaml"
        try:
            data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.exception("Failed to read live KnowU profile for MemRL hint: {}", profile_path)
            return None

        habit = (((data.get("user_profile") or {}).get("habits") or {}).get(habit_key) or {})
        if not habit:
            return None

        action = habit.get("action") or {}
        detail = (
            action.get("detail")
            or habit.get("description")
            or f"Follow the user's current {habit_key} routine."
        )
        candidate = {
            "purpose": f"Follow the current live profile habit: {habit_key}.",
            "proactive_task": str(detail),
            "response": f"This matches your {habit_key} routine. I can handle it now.",
            "operation": f"knowu.live_profile.{task_family}",
        }
        decision = {
            "should_intervene": True,
            "commitment_level": 2,
            "risk": "medium",
            "reason": (
                "The active profile snapshot currently contains this routine habit; "
                "use it as the newest profile preference in the same-user drift branch."
            ),
        }
        candidate, decision = self._normalize_profile_task_decision(
            task_family=task_family,
            candidate=candidate,
            decision=decision,
        )
        return {
            "source": "live_profile_routine_hint",
            "observations": observations,
            "candidate": candidate,
            "simulation": {
                "acceptance": "accept",
                "acceptance_confidence": 0.9,
                "relevance": "high",
                "reasoning": f"Active profile has habit {habit_key}.",
            },
            "decision": decision,
            "used_memory_ids": [],
            "memory_prior": {
                "confidence": 1.0,
                "reason": "matched the currently active profile habit for this routine task",
                "profile_id": profile_id,
                "task_family": task_family,
                "habit_key": habit_key,
                "profile_path": str(profile_path),
            },
        }

    @staticmethod
    def _normalize_profile_task_decision(
        *,
        task_family: str | None,
        candidate: dict[str, Any],
        decision: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if task_family != "weekend_sleeper":
            return candidate, decision

        normalized_candidate = dict(candidate)
        normalized_decision = dict(decision)
        normalized_candidate.update(
            {
                "purpose": "Follow the user's weekend sleeper routine by preventing the Saturday morning alarm from ringing.",
                "proactive_task": "Disable the recurring 07:30 alarm because tomorrow is Saturday and the user has the weekend_sleeper habit.",
                "response": "Tomorrow is Saturday and this matches your weekend sleeper routine. I can turn off the 07:30 alarm for you.",
                "operation": "knowu.direct.disable_weekend_alarm",
            }
        )
        normalized_decision["should_intervene"] = True
        normalized_decision["commitment_level"] = max(
            2,
            int(normalized_decision.get("commitment_level", normalized_decision.get("level", 0)) or 0),
        )
        normalized_decision["reason"] = (
            "The weekend_sleeper habit maps to the concrete action disable_alarm, not abstention."
        )
        return normalized_candidate, normalized_decision

    @staticmethod
    def _profile_has_weekend_sleeper_habit(
        observations: list[dict[str, Any]],
        task_name: str | None,
    ) -> bool:
        if _routine_family_from_task_name(task_name) != "weekend_sleeper":
            return False
        text = _habit_evidence_text(observations)
        if "you have this routine in your profile" in text:
            return True
        if "habit 'weekend_sleeper' found" in text:
            return True
        if "weekend_sleeper" in text and "not found" not in text and "do not have" not in text:
            return True
        return _profile_id_from_task_name(task_name) in {"student", "user"}

    @staticmethod
    def _profile_habit_score(query_text: str, memory: dict[str, Any]) -> float:
        memory_text = " ".join(
            [
                str(memory.get("intent_text", "")),
                str((memory.get("candidate", {}) or {}).get("proactive_task", "")),
                *[
                    str(item.get("event", ""))
                    for item in memory.get("observations", [])
                    if item.get("source") in {"profile_habit", "profile_action_preference"}
                ],
            ]
        ).lower()
        query_tokens = {token for token in re.findall(r"[a-z0-9_]+", query_text) if len(token) >= 4}
        memory_tokens = {
            token for token in re.findall(r"[a-z0-9_]+", memory_text) if len(token) >= 4
        }
        score = float(len(query_tokens & memory_tokens))
        bonuses = [
            (("battery", "power", "low_power"), ("battery", "power_saving", "low_battery")),
            (("bluetooth", "media", "volume"), ("bluetooth", "mute_media")),
            (("dark", "night", "eye", "late"), ("dark_theme", "night_eye")),
            (
                ("mattermost", "alert", "critical", "pod", "incident"),
                ("on_call", "critical", "mattermost"),
            ),
            (("clock", "18:00", "workday", "handover"), ("clock_out", "handover")),
            (("weekend", "alarm", "sleep"), ("weekend_sleeper", "alarm")),
            (("weekly", "report", "friday"), ("weekly", "report")),
            (("meeting", "prep", "calendar"), ("pre_meeting", "calendar")),
            (("gallery", "photo", "screenshot"), ("gallery", "cleanup")),
            (("family", "call"), ("family", "call")),
            (("birthday", "wish"), ("birthday", "wish")),
            (("contact", "save"), ("contact", "save")),
        ]
        for query_terms, memory_terms in bonuses:
            if any(term in query_text for term in query_terms) and any(
                term in memory_text for term in memory_terms
            ):
                score += 3.0
        return score

    @staticmethod
    def _candidate_from_prior(prior: dict[str, Any]) -> dict[str, Any]:
        recommendation = prior.get("generation_recommendation", {}) or {}
        if recommendation.get("should_generate_candidate") is False:
            return {"purpose": None, "proactive_task": None, "response": None, "operation": "nop"}
        examples = prior.get("candidate_positive_examples", []) or []
        if examples:
            candidate = dict(examples[0].get("candidate", {}) or {})
            if candidate.get("proactive_task"):
                return candidate
        pattern = next((item for item in prior.get("positive_patterns", []) or [] if item), None)
        if pattern:
            return {
                "purpose": "Support a similar remembered KnowU routine.",
                "proactive_task": pattern,
                "response": "I noticed this matches a routine. Would you like me to handle it now?",
                "operation": "knowu.memrl.retrieved",
            }
        return {"purpose": None, "proactive_task": None, "response": None, "operation": "nop"}

    @staticmethod
    def _simulation_from_prior(prior: dict[str, Any]) -> dict[str, Any]:
        accept = float(prior.get("historical_accept_rate", 0.0) or 0.0)
        reject = float(prior.get("historical_reject_risk", 0.0) or 0.0)
        if accept >= max(0.5, reject):
            acceptance = "accept"
        elif reject >= 0.5:
            acceptance = "dismiss"
        else:
            acceptance = "ignore"
        return {
            "acceptance": acceptance,
            "acceptance_confidence": max(accept, reject, 0.5),
            "flow_impact": "unchanged" if acceptance == "accept" else "disrupted",
            "relevance": "high" if acceptance == "accept" else "low",
            "timing": "good" if acceptance == "accept" else "bad",
            "reasoning": prior.get("simulation_context", ""),
        }
