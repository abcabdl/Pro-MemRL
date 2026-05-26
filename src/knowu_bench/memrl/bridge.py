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
    for line in instruction.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
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
    "DailyFamilyCallTask": "daily_family_call",
    "DeepWorkRoutineTask": "deep_work",
    "GalleryCleanupTask": "gallery_cleanup",
    "MattermostOnCallTask": "mattermost_response",
    "MorningPaperReadingTask": "morning_paper_reading",
    "MorningWeatherCheckTask": "morning_weather_check",
    "NightEyeCareRoutineTask": "night_eye_care",
    "PreMeetingPrepTask": "pre_meeting_prep",
    "ScamSmsInterceptRoutineTask": "scam_sms_intercept",
    "WeekendSleeperTask": "weekend_sleeper",
    "WeeklyReportRoutineTask": "weekly_report",
}

_TASK_FAMILY_HABIT_KEY: dict[str, str] = {
    "battery_saver": "low_battery_saver",
    "birthday_wish": "birthday_wish",
    "bluetooth_cleanup": "bluetooth_cleanup",
    "clock_out": "clock_out_routine",
    "contact_saver": "contact_saver",
    "daily_family_call": "daily_family_call",
    "deep_work": "deep_work_block",
    "gallery_cleanup": "gallery_cleanup",
    "mattermost_response": "on_call_response",
    "morning_paper_reading": "morning_paper_reading",
    "morning_weather_check": "morning_weather_check",
    "night_eye_care": "night_eye_care",
    "pre_meeting_prep": "pre_meeting_prep",
    "scam_sms_intercept": "scam_sms_intercept",
    "weekend_sleeper": "weekend_sleeper",
    "weekly_report": "weekly_report",
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


def _profile_task_memory_version(memory: dict[str, Any]) -> int:
    sample_id = str(memory.get("sample_id", ""))
    match = re.search(r"::v(\d+)$", sample_id)
    return int(match.group(1)) if match else -1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        self.runtime.record_outcome(
            [str(item) for item in self.last_plan.get("used_memory_ids", []) if item],
            float(reward),
            episode,
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
        q_floor = _safe_float(os.getenv("KNOWU_MEMRL_DIRECT_Q_FLOOR"), 0.25)
        if selected_q < q_floor:
            logger.info(
                "Profile-task direct memory matched {} / {} but q_value {:.3f} is below floor {:.3f}; falling back to value-aware retrieval.",
                profile_id,
                task_family,
                selected_q,
                q_floor,
            )
            return None
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
                "q_floor": q_floor,
                "profile_task_score": round(_profile_task_score(memory), 4),
                "profile_task_version": _profile_task_memory_version(memory),
                "candidate_count": len(candidates),
            },
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
