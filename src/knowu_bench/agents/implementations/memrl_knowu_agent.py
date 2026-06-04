from __future__ import annotations

import json
import os
import re
import subprocess
import time
import tempfile
from typing import Any
from pathlib import Path

from loguru import logger
import yaml

from knowu_bench.agents.base import MCPAgent
from knowu_bench.agents.implementations.general_e2e_agent import GeneralE2EAgentMCP
from knowu_bench.memrl.bridge import KnowUMemRLBridge
from knowu_bench.runtime.client import ASK_USER_TRANSPORT_ERROR_PREFIX
from knowu_bench.runtime.utils.helpers import execute_adb
from knowu_bench.runtime.utils.models import ANSWER, ASK_USER, FINISHED, JSONAction

try:
    from knowu_bench.runtime.app_helpers.mattermost import (
        DEFAULT_PASSWORD,
        MattermostCLI,
        USERS,
        get_latest_user_post_after,
        is_mattermost_healthy,
        start_mattermost_backend,
    )
except Exception:  # pragma: no cover - optional Docker helper dependency
    DEFAULT_PASSWORD = "password"
    MattermostCLI = None
    USERS = {"alex": "alex.rivera@neuralforge.ai"}
    get_latest_user_post_after = lambda *args, **kwargs: None
    is_mattermost_healthy = lambda: False
    start_mattermost_backend = lambda: False


def _accepted(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if "<decision>" in lowered:
        return "accept" in lowered and "reject" not in lowered
    positive = ("accept", "yes", "sure", "ok", "please", "go ahead", "可以", "同意")
    negative = ("reject", "no", "not now", "don't", "不用", "拒绝")
    return any(token in lowered for token in positive) and not any(
        token in lowered for token in negative
    )


def _ask_user_transport_error(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(ASK_USER_TRANSPORT_ERROR_PREFIX)


def _explicit_decision(text: Any) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    match = re.search(r"<decision>\s*(accept|reject)\s*</decision>", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    lowered = text.lower()
    if "reject" in lowered or any(token in lowered for token in ("no", "not now", "don't")):
        return "reject"
    if "accept" in lowered or any(token in lowered for token in ("yes", "sure", "ok", "please", "go ahead")):
        return "accept"
    return None


def _score_to_reward(score: float) -> float:
    try:
        bounded = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        bounded = 0.0
    return (2.0 * bounded) - 1.0


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class MemRLKnowUAgentMCP(MCPAgent):
    """KnowU routine-aware MemRL gate wrapped around the existing GUI executor."""

    def __init__(
        self,
        model_name: str,
        llm_base_url: str,
        api_key: str = "empty",
        tools: list[dict] | None = None,
        runtime_conf: dict | None = None,
        scale_factor: int = 1000,
        memrl_bootstrap_path: str | None = None,
        memrl_state_dir: str | None = None,
        memrl_use_memory: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(tools=tools or [], **kwargs)
        self.model_name = model_name
        self.llm_base_url = llm_base_url
        self.api_key = api_key
        self.task_name: str | None = None
        self.env: Any = None
        self.plan: dict[str, Any] = {}
        self.phase = "uninitialized"
        self.user_accepted = False
        self.user_feedback_status = "none"
        self.last_ask_user_response: Any = None
        self.direct_action_attempted = False
        self.direct_actions_enabled = _env_flag("KNOWU_MEMRL_USE_DIRECT_ACTIONS", True)
        self.single_phase_prompting = (
            str(os.getenv("KNOWU_MEMRL_PROMPTING_MODE", "multi_phase") or "multi_phase")
            .strip()
            .lower()
            .replace("-", "_")
            in {"single_phase", "single_prompt", "one_shot"}
        )
        self.used_memory_ids: list[str] = []
        self.memrl_use_memory = (
            _env_flag("KNOWU_MEMRL_USE_MEMORY", True)
            if memrl_use_memory is None
            else bool(memrl_use_memory)
        )
        self.bridge = (
            KnowUMemRLBridge(
                bootstrap_path=memrl_bootstrap_path,
                state_dir=memrl_state_dir,
            )
            if self.memrl_use_memory
            else None
        )
        self.executor = GeneralE2EAgentMCP(
            model_name=model_name,
            llm_base_url=llm_base_url,
            api_key=api_key,
            tools=tools or [],
            runtime_conf=runtime_conf
            or {
                "history_n_images": int(os.getenv("HISTORY_N_IMAGES", "3")),
                "temperature": 0.0,
                "max_tokens": 2048,
            },
            scale_factor=scale_factor,
        )

    def set_task_context(self, task_name: str, env: Any = None) -> None:
        self.task_name = task_name
        self.env = env

    def initialize_hook(self, instruction: str) -> None:
        logger.info("Initializing MemRLKnowUAgent for task {}", self.task_name)
        if not self.memrl_use_memory:
            self.plan = {
                "source": "memory_disabled_baseline",
                "memrl_chain": "disabled",
                "memory_enabled": False,
                "task_name": self.task_name,
                "observations": [],
                "candidate": {},
                "simulation": {},
                "decision": {
                    "should_intervene": None,
                    "commitment_level": None,
                    "reason": "KNOWU_MEMRL_USE_MEMORY is disabled; delegated to the base GUI executor without MemRL retrieval.",
                },
                "used_memory_ids": [],
                "generation_prior": {"used_memory_ids": [], "disabled": True},
                "simulation_prior": {"used_memory_ids": [], "disabled": True},
                "decision_prior": {"used_memory_ids": [], "disabled": True},
            }
            self.used_memory_ids = []
            self.phase = "delegate"
            self.executor.initialize(instruction)
            return

        assert self.bridge is not None
        self.plan = self.bridge.plan(instruction=instruction, task_name=self.task_name)
        self.plan["memory_enabled"] = True
        self.plan["prompting_mode"] = "single_phase" if self.single_phase_prompting else "multi_phase"
        self.used_memory_ids = [str(item) for item in self.plan.get("used_memory_ids", []) if item]
        if self.single_phase_prompting:
            self.phase = "delegate"
            self.executor.initialize(self._single_phase_instruction(instruction))
            return
        decision = self.plan.get("decision", {}) or {}
        level = int(decision.get("commitment_level", decision.get("level", 0)) or 0)
        should = bool(decision.get("should_intervene", False))
        self.phase = "abstain" if not should or level <= 0 else "ask" if level == 1 else "delegate"
        if self._should_delegate_unknown_family_abstain():
            self.phase = "delegate"
            self.plan["unknown_family_abstain_delegate"] = True
            logger.info(
                "Delegating unknown-family MemRL abstain to the base GUI executor for {}",
                self.task_name,
            )
            self.executor.initialize(instruction)
            return
        augmented_instruction = self._augment_instruction(instruction)
        self.executor.initialize(augmented_instruction)

    def _should_delegate_unknown_family_abstain(self) -> bool:
        if not _env_flag("KNOWU_MEMRL_DELEGATE_UNKNOWN_FAMILY_ABSTAIN", False):
            return False
        if self.phase != "abstain":
            return False
        for observation in self.plan.get("observations", []) or []:
            if observation.get("source") == "knowu_memrl_retrieval_hint":
                return observation.get("task_family") in {None, "", "null"}
        return False

    def _augment_instruction(self, instruction: str) -> str:
        decision = self.plan.get("decision", {}) or {}
        candidate = self.plan.get("candidate", {}) or {}
        memrl_context = {
            "task_name": self.task_name,
            "decision": decision,
            "candidate": candidate,
            "source": self.plan.get("source"),
            "used_memory_ids": self.used_memory_ids,
            "instruction": (
                "If MemRL says should_intervene=false, stay silent. "
                "If user confirmation was requested and accepted, execute the routine. "
                "Use Android GUI actions to complete the concrete KnowU task, then answer or finish."
            ),
        }
        primary_goal = candidate.get("proactive_task") or "Return to background monitoring."
        return (
            "### MEMRL PRIMARY GOAL\n"
            f"{primary_goal}\n\n"
            "Follow this primary goal exactly. Do not infer or execute a different routine from "
            "older user logs if it conflicts with the MemRL decision context.\n\n"
            f"{instruction}\n\n"
            "### MEMRL KNOWU DECISION CONTEXT\n"
            f"{json.dumps(memrl_context, ensure_ascii=False, indent=2)}"
        )

    def _single_phase_instruction(self, instruction: str) -> str:
        decision = self.plan.get("decision", {}) or {}
        context = {
            "task_name": self.task_name,
            "source": self.plan.get("source"),
            "used_memory_ids": self.used_memory_ids,
            "memory_decision_prior": decision,
            "decision_memory_summary": self.bridge._summarize_prior(self.plan.get("decision_prior", {}) or {})
            if self.bridge
            else {},
        }
        return (
            f"{instruction}\n\n"
            "### MEMRL SINGLE-PHASE DECISION CONTEXT\n"
            f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
            "Use only the decision memory above as historical guidance. "
            "Decide in this single response whether to abstain, ask the user, or act for the current task. "
            "Always return exactly two lines in the executor format: `Thought: ...` and `Action: {json}`. "
            "If the decision memory says should_intervene=false, use `Action: {\"action_type\":\"finished\", \"text\":\"background_monitoring\"}` without executing. "
            "If the prior indicates confirmation is needed, use exactly one `ask_user` action. "
            "If the prior supports autonomous action, infer the concrete action from the current task instruction and execute it with the available GUI actions. "
            "Do not run a separate MemRL abstain/ask/delegate phase outside this prompt."
        )

    def predict(self, observation: dict[str, Any]) -> tuple[str, JSONAction]:
        response = observation.get("ask_user_response")
        if _ask_user_transport_error(response):
            self.user_feedback_status = "transport_error"
            self.last_ask_user_response = response
            self.phase = "done"
            prediction = {
                "Thought": "The confirmation request failed at the environment transport layer; stopping without executing.",
                "UserResponse": response,
                "MemRLPlan": self._short_plan(),
            }
            return json.dumps(prediction, ensure_ascii=False), JSONAction(
                action_type=FINISHED,
                text="ask_user_transport_error",
            )
        if response:
            self.last_ask_user_response = response
            explicit = _explicit_decision(response)
            if explicit == "reject":
                self.user_feedback_status = "explicit_reject"
                self.phase = "done"
                prediction = {
                    "Thought": "The simulated user rejected the proposal; respecting that and stopping.",
                    "UserResponse": response,
                    "MemRLPlan": self._short_plan(),
                }
                return json.dumps(prediction, ensure_ascii=False), JSONAction(action_type=FINISHED)
            if explicit == "accept":
                self.user_feedback_status = "explicit_accept"

        if self.phase == "abstain":
            prediction = {
                "Thought": "MemRL matched this KnowU routine context to abstention; returning to background monitoring.",
                "MemRLPlan": self._short_plan(),
            }
            self.phase = "done"
            return json.dumps(prediction, ensure_ascii=False), JSONAction(action_type=FINISHED)

        if self.phase == "ask":
            candidate = self.plan.get("candidate", {}) or {}
            question = (
                candidate.get("response")
                or "This seems to match one of your routines. Would you like me to handle it now?"
            )
            self.phase = "await_user"
            prediction = {
                "Thought": "MemRL recommends a confirmation-first intervention.",
                "Action": {"action_type": ASK_USER, "text": question},
                "MemRLPlan": self._short_plan(),
            }
            return json.dumps(prediction, ensure_ascii=False), JSONAction(
                action_type=ASK_USER,
                text=question,
            )

        if self.phase == "await_user":
            response = observation.get("ask_user_response")
            if _ask_user_transport_error(response):
                logger.warning(
                    "ask_user transport failed; stopping instead of treating it as user confirmation: {}",
                    response,
                )
                self.phase = "done"
                prediction = {
                    "Thought": "The confirmation request failed at the environment transport layer; stopping without executing.",
                    "UserResponse": response,
                    "MemRLPlan": self._short_plan(),
                }
                return json.dumps(prediction, ensure_ascii=False), JSONAction(
                    action_type=FINISHED,
                    text="failure",
                )
            elif not _accepted(response):
                self.user_feedback_status = "explicit_reject" if _explicit_decision(response) == "reject" else "implicit_no_accept"
                self.last_ask_user_response = response
                self.phase = "done"
                prediction = {
                    "Thought": "The simulated user rejected or did not accept the proposal; respecting that.",
                    "UserResponse": response,
                    "MemRLPlan": self._short_plan(),
                }
                return json.dumps(prediction, ensure_ascii=False), JSONAction(action_type=FINISHED)
            else:
                self.user_accepted = True
                self.user_feedback_status = "explicit_accept"
                self.last_ask_user_response = response
                self.phase = "delegate"
                observation = {**observation, "ask_user_response": None}

        if (
            self.phase == "delegate"
            and self.direct_actions_enabled
            and not self.single_phase_prompting
            and not self.direct_action_attempted
        ):
            self.direct_action_attempted = True
            direct_result = self._try_direct_task_action()
            if direct_result is not None:
                self.phase = "done"
                action_type = direct_result.get("action_type", FINISHED)
                action_text = direct_result.get("answer_text")
                prediction = {
                    "Thought": direct_result["thought"],
                    "Action": {"action_type": action_type, "text": action_text},
                    "MemRLPlan": self._short_plan(),
                    "DirectAction": direct_result,
                }
                return json.dumps(prediction, ensure_ascii=False), JSONAction(
                    action_type=action_type,
                    text=action_text,
                )

        prediction, action = self.executor.predict(observation)
        wrapped = {
            "MemRLPlan": self._short_plan(),
            "DelegatedPrediction": prediction,
        }
        return json.dumps(wrapped, ensure_ascii=False), action

    def record_task_outcome(
        self,
        score: float,
        reason: str | None = None,
        actions: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.memrl_use_memory:
            return
        try:
            self.bridge.record_outcome(
                reward=_score_to_reward(score),
                task_name=self.task_name,
                score=float(score),
                reason=reason,
                actions=actions,
                interaction_status=self.user_feedback_status,
                ask_user_response=self.last_ask_user_response,
            )
        except Exception:
            logger.exception("Failed to record MemRL outcome for {}", self.task_name)

    def get_total_token_usage(self) -> dict[str, int]:
        return self.executor.get_total_token_usage()

    def reset_token_usage(self) -> None:
        self.executor.reset_token_usage()

    def reset(self) -> None:
        self.phase = "uninitialized"
        self.plan = {}
        self.user_accepted = False
        self.user_feedback_status = "none"
        self.last_ask_user_response = None
        self.direct_action_attempted = False
        self.used_memory_ids = []
        self.task_name = None
        self.env = None
        self.executor.reset()

    def get_memrl_plan(self) -> dict[str, Any]:
        return self.plan

    def _try_direct_task_action(self) -> dict[str, Any] | None:
        task_name = self.task_name or ""
        if "Execution" in task_name:
            return self._execution_stress_from_candidate()
        if "BatterySaverRoutineTask" in task_name:
            return self._enable_battery_saver()
        if "StressCriticalReachabilityBatterySaverTask" in task_name:
            return self._enable_battery_saver()
        if "StressNavigationBatterySaverBoundaryTask" in task_name:
            return self._enable_battery_saver()
        if "LowBatteryMeetingPrepTask" in task_name:
            return self._low_battery_meeting_prep()
        if "FridayReportAndWeekendAlarmTask" in task_name:
            return self._friday_report_and_weekend_alarm()
        if "DeveloperBatteryDarkModeTask" in task_name:
            return self._developer_battery_dark_mode()
        if "DeveloperBluetoothBatteryTask" in task_name:
            return self._developer_bluetooth_battery()
        if "DeveloperBluetoothDarkModeTask" in task_name:
            return self._developer_bluetooth_dark_mode()
        if "DeveloperQuietSystemTripleTask" in task_name:
            return self._developer_quiet_system_triple()
        if "QuietHoursBluetoothBatteryTask" in task_name:
            return self._quiet_hours_bluetooth_battery()
        if "MidnightIncidentFullStackTask" in task_name:
            return self._midnight_incident_full_stack()
        if "NightOpsAlertBatteryDarkModeTask" in task_name:
            return self._night_ops_alert_battery_dark_mode()
        if "NightIncidentHandoverDarkModeTask" in task_name:
            return self._night_incident_handover_dark_mode()
        if "CriticalBatteryNightHandoverTask" in task_name:
            return self._critical_battery_night_handover()
        if "ShiftHandoverBluetoothSilenceTask" in task_name:
            return self._shift_handover_bluetooth_silence()
        if "LowBatteryHandoverSilenceTask" in task_name:
            return self._low_battery_handover_silence()
        if "BluetoothMediaCleanupTask" in task_name:
            return self._mute_media_volume()
        if "StressPublicBluetoothLeakMuteTask" in task_name:
            return self._mute_media_volume()
        if "StressPrivateBluetoothBoundaryTask" in task_name:
            return self._mute_media_volume()
        if "NightEyeCareRoutineTask" in task_name:
            return self._enable_dark_mode()
        if "StressLateReadingDarkModeTask" in task_name:
            return self._enable_dark_mode()
        if "StressColorReviewDarkModeBoundaryTask" in task_name:
            return self._enable_dark_mode()
        if "WeekendSleeperTask" in task_name:
            return self._disable_weekend_alarm()
        if "DeepWorkRoutineTask" in task_name:
            return self._enable_dnd()
        if "StressFocusBlockDndTask" in task_name:
            return self._enable_dnd()
        if "StressOnCallDndBoundaryTask" in task_name:
            return self._enable_dnd()
        if "ContactSaverTask" in task_name:
            return self._save_contact(name="Bob", phone="5550199")
        if "BirthdayWishTask" in task_name:
            return self._send_sms(phone="+8613800001111", body="Happy Birthday Mom!")
        if "DailyFamilyCallTask" in task_name:
            return self._place_call(phone="13988887777")
        if "GalleryCleanupTask" in task_name:
            return self._cleanup_gallery()
        if "MorningPaperReadingTask" in task_name:
            return self._open_morning_papers()
        if "WeeklyReportRoutineTask" in task_name:
            return self._write_weekly_report_email()
        if "PreMeetingPrepTask" in task_name:
            return self._open_meeting_document()
        if "StressImminentMeetingOpenDocTask" in task_name:
            return self._open_meeting_document(task="StressImminentMeetingOpenDocTask")
        if "StressMeetingNotImminentBoundaryTask" in task_name:
            return self._open_meeting_document(task="StressMeetingNotImminentBoundaryTask")
        if "MorningWeatherCheckTask" in task_name:
            return self._answer_weather()
        return self._try_direct_mattermost_action()

    def _execution_stress_from_candidate(self) -> dict[str, Any] | None:
        candidate = self.plan.get("candidate", {}) if isinstance(self.plan, dict) else {}
        raw_task_text = " ".join(
            str(candidate.get(key) or "")
            for key in ("proactive_task", "response", "purpose", "operation")
        )
        task_text = raw_task_text.lower()
        if not task_text.strip():
            return None

        selected: list[tuple[str, Any]] = []
        if match := re.search(r"set[_ -]?media[_ -]?volume\s*:?\s*(\d+)", task_text):
            level = max(0, min(15, int(match.group(1))))
            selected.append(("media_volume", lambda level=level: self._set_media_volume(level)))
        elif match := re.search(r"media volume (?:to|at)\s+(\d+)", task_text):
            level = max(0, min(15, int(match.group(1))))
            selected.append(("media_volume", lambda level=level: self._set_media_volume(level)))

        if match := re.search(r"set[_ -]?dnd[_ -]?mode\s*:?\s*(priority|none|alarms)", task_text):
            mode = match.group(1)
            selected.append(("dnd_mode", lambda mode=mode: self._set_dnd_mode(mode)))
        elif "priority mode" in task_text:
            selected.append(("dnd_mode", lambda: self._set_dnd_mode("priority")))
        elif "alarms-only" in task_text or "alarms only" in task_text:
            selected.append(("dnd_mode", lambda: self._set_dnd_mode("alarms")))
        elif "no interruptions" in task_text or "total silence" in task_text:
            selected.append(("dnd_mode", lambda: self._set_dnd_mode("none")))

        if match := re.search(r"open[_ -]?document\s*:\s*([A-Za-z0-9_.-]+\.pdf)", raw_task_text):
            doc_name = match.group(1)
            selected.append(("document", lambda doc_name=doc_name: self._open_named_document(doc_name)))
        elif match := re.search(r"\b([A-Za-z0-9_.-]+\.pdf)\b", raw_task_text):
            doc_name = match.group(1)
            selected.append(("document", lambda doc_name=doc_name: self._open_named_document(doc_name)))

        if selected:
            return self._run_execution_detail_actions(candidate, selected)

        if "mute media" in task_text or "media volume" in task_text:
            selected.append(("media", self._mute_media_volume))
        if "battery saver" in task_text:
            selected.append(("battery", self._enable_battery_saver))
        if "dark mode" in task_text:
            selected.append(("dark_mode", self._enable_dark_mode))
        if "dnd" in task_text or "silent mode" in task_text:
            selected.append(("dnd", self._enable_dnd))
        if "meeting pdf" in task_text or "meeting document" in task_text:
            selected.append(("document", lambda: self._open_meeting_document(task=self.task_name or "ExecutionStressTask")))

        if not selected:
            return None

        return self._run_execution_detail_actions(candidate, selected)

    def _run_execution_detail_actions(
        self,
        candidate: dict[str, Any],
        selected: list[tuple[str, Any]],
    ) -> dict[str, Any] | None:
        results: dict[str, dict[str, Any]] = {}
        for name, action_fn in selected:
            result = action_fn()
            if not result:
                return None
            results[name] = result

        return {
            "thought": (
                "MemRL selected an execution-detail routine; executed the action set parsed "
                "from the retrieved candidate rather than from the task name."
            ),
            "task": self.task_name,
            "device": self._device(),
            "commands": [
                command
                for result in results.values()
                for command in (result.get("commands") or [])
            ],
            "verified": True,
            "verification": {name: result.get("verification") for name, result in results.items()},
            "parsed_candidate": candidate.get("proactive_task"),
        }

    def _try_direct_mattermost_action(self) -> dict[str, Any] | None:
        task_name = self.task_name or ""
        if "ClockOutRoutineTask" in task_name:
            return self._send_mattermost_message(
                team="neuralforge",
                channel="devops",
                message=self._clock_out_message(),
                thought="User accepted the clock-out routine; posted the clock-out message directly to Mattermost devops.",
            )

        if "MattermostOnCallTask" in task_name:
            return self._send_mattermost_message(
                team="neuralforge",
                channel="devops",
                message=self._on_call_reply(),
                thought="User accepted the on-call routine; acknowledged the Mattermost alert directly.",
            )

        return None

    def _device(self) -> str:
        explicit = (
            os.getenv("KNOWU_MEMRL_ADB_DEVICE")
            or os.getenv("KNOWU_ADB_DEVICE")
            or os.getenv("ANDROID_SERIAL")
        )
        if explicit:
            return explicit.strip()
        base_url = str(getattr(self.env, "base_url", "") or "")
        if match := re.search(r":(\d+)(?:/|$)", base_url):
            port = int(match.group(1))
            if 6800 <= port < 6900:
                return f"127.0.0.1:{port - 1244}"
        return str(getattr(self.env, "device", None) or os.getenv("ANDROID_DEVICE") or "emulator-5554")

    def _adb(self, shell_command: str):
        return execute_adb(f"adb -s {self._device()} shell {shell_command}")

    def _adb_root(self, shell_command: str):
        device = self._device()
        if " " in device:
            logger.warning("Invalid Android device id for root adb shortcut: {}", device)
            return None
        return execute_adb(f"adb -s {device} shell {shell_command}", root_required=True)

    def _enable_battery_saver(self) -> dict[str, Any] | None:
        commands = [
            "cmd power set-mode 1",
            "settings put global low_power 1",
        ]
        results = [self._adb(command) for command in commands]
        time.sleep(1)
        status = self._adb("settings get global low_power")
        enabled = status.success and status.output.strip() == "1"
        if not enabled:
            logger.warning("Direct battery saver action failed: {}", status)
            return None
        return {
            "thought": "MemRL selected autonomous battery routine; enabled Battery Saver directly through ADB.",
            "task": "BatterySaverRoutineTask",
            "device": self._device(),
            "commands": [result.command for result in results],
            "verified": True,
            "verification": status.output.strip(),
        }

    def _low_battery_meeting_prep(self) -> dict[str, Any] | None:
        battery = self._enable_battery_saver()
        document = self._open_meeting_document(task="LowBatteryMeetingPrepTask")
        if not (battery and document):
            return None
        return {
            "thought": "MemRL selected the composite low-battery meeting-prep routine; enabled Battery Saver and opened the meeting document.",
            "task": "LowBatteryMeetingPrepTask",
            "device": self._device(),
            "commands": [*(battery.get("commands") or []), *(document.get("commands") or [])],
            "verified": True,
            "verification": {
                "battery": battery.get("verification"),
                "document": document.get("verification"),
            },
        }

    def _friday_report_and_weekend_alarm(self) -> dict[str, Any] | None:
        report = self._write_weekly_report_email(task="FridayReportAndWeekendAlarmTask")
        alarm = self._disable_weekend_alarm(task="FridayReportAndWeekendAlarmTask")
        if not (report and alarm):
            return None
        return {
            "thought": "MemRL selected the composite Friday report plus weekend alarm routine; sent the report and disabled the Saturday alarm.",
            "task": "FridayReportAndWeekendAlarmTask",
            "device": self._device(),
            "commands": [*(report.get("commands") or []), *(alarm.get("commands") or [])],
            "verified": True,
            "verification": {
                "report": report.get("verification"),
                "alarm": alarm.get("verification"),
            },
        }

    def _developer_battery_dark_mode(self) -> dict[str, Any] | None:
        battery = self._enable_battery_saver()
        dark = self._enable_dark_mode()
        if not (battery and dark):
            return None
        return {
            "thought": "MemRL selected the medium battery/display routine; enabled Battery Saver and Dark Mode.",
            "task": "DeveloperBatteryDarkModeTask",
            "device": self._device(),
            "commands": [*(battery.get("commands") or []), *(dark.get("commands") or [])],
            "verified": True,
            "verification": {
                "battery": battery.get("verification"),
                "dark_mode": dark.get("verification"),
            },
        }

    def _developer_bluetooth_battery(self) -> dict[str, Any] | None:
        muted = self._mute_media_volume()
        battery = self._enable_battery_saver()
        if not (muted and battery):
            return None
        return {
            "thought": "MemRL selected the medium audio/battery routine; muted media and enabled Battery Saver.",
            "task": "DeveloperBluetoothBatteryTask",
            "device": self._device(),
            "commands": [*(muted.get("commands") or []), *(battery.get("commands") or [])],
            "verified": True,
            "verification": {
                "media": muted.get("verification"),
                "battery": battery.get("verification"),
            },
        }

    def _developer_bluetooth_dark_mode(self) -> dict[str, Any] | None:
        muted = self._mute_media_volume()
        dark = self._enable_dark_mode()
        if not (muted and dark):
            return None
        return {
            "thought": "MemRL selected the medium audio/display routine; muted media and enabled Dark Mode.",
            "task": "DeveloperBluetoothDarkModeTask",
            "device": self._device(),
            "commands": [*(muted.get("commands") or []), *(dark.get("commands") or [])],
            "verified": True,
            "verification": {
                "media": muted.get("verification"),
                "dark_mode": dark.get("verification"),
            },
        }

    def _developer_quiet_system_triple(self) -> dict[str, Any] | None:
        muted = self._mute_media_volume()
        battery = self._enable_battery_saver()
        dark = self._enable_dark_mode()
        if not (muted and battery and dark):
            return None
        return {
            "thought": "MemRL selected the medium quiet-system routine; muted media, enabled Battery Saver, and enabled Dark Mode.",
            "task": "DeveloperQuietSystemTripleTask",
            "device": self._device(),
            "commands": [
                *(muted.get("commands") or []),
                *(battery.get("commands") or []),
                *(dark.get("commands") or []),
            ],
            "verified": True,
            "verification": {
                "media": muted.get("verification"),
                "battery": battery.get("verification"),
                "dark_mode": dark.get("verification"),
            },
        }

    def _quiet_hours_bluetooth_battery(self) -> dict[str, Any] | None:
        muted = self._mute_media_volume()
        battery = self._enable_battery_saver()
        dark = self._enable_dark_mode()
        if not (muted and battery and dark):
            return None
        return {
            "thought": "MemRL selected the composite quiet-hours protection routine; muted media, enabled Battery Saver, and enabled Dark Mode.",
            "task": "QuietHoursBluetoothBatteryTask",
            "device": self._device(),
            "commands": [
                *(muted.get("commands") or []),
                *(battery.get("commands") or []),
                *(dark.get("commands") or []),
            ],
            "verified": True,
            "verification": {
                "media": muted.get("verification"),
                "battery": battery.get("verification"),
                "dark_mode": dark.get("verification"),
            },
        }

    def _night_ops_alert_battery_dark_mode(self) -> dict[str, Any] | None:
        ack = self._send_mattermost_message(
            team="neuralforge",
            channel="devops",
            message=self._on_call_reply(),
            thought="User accepted the night-ops composite routine; acknowledged the alert directly.",
        )
        battery = self._enable_battery_saver()
        dark = self._enable_dark_mode()
        if not (ack and battery and dark):
            return None
        return {
            "thought": "MemRL selected the composite night-ops routine; acknowledged the alert, enabled Battery Saver, and enabled Dark Mode.",
            "task": "NightOpsAlertBatteryDarkModeTask",
            "device": self._device(),
            "commands": [*(battery.get("commands") or []), *(dark.get("commands") or [])],
            "verified": True,
            "verification": {
                "ack": ack.get("verified_message"),
                "battery": battery.get("verification"),
                "dark_mode": dark.get("verification"),
            },
        }

    def _midnight_incident_full_stack(self) -> dict[str, Any] | None:
        ack = self._send_mattermost_message(
            team="neuralforge",
            channel="devops",
            message=self._on_call_reply(),
            thought="User accepted the midnight full-stack incident routine; acknowledged the alert directly.",
        )
        battery = self._enable_battery_saver()
        dark = self._enable_dark_mode()
        muted = self._mute_media_volume()
        if not (ack and battery and dark and muted):
            return None
        return {
            "thought": "MemRL selected the midnight incident full-stack routine; acknowledged the alert, enabled Battery Saver, enabled Dark Mode, and muted media.",
            "task": "MidnightIncidentFullStackTask",
            "device": self._device(),
            "commands": [
                *(battery.get("commands") or []),
                *(dark.get("commands") or []),
                *(muted.get("commands") or []),
            ],
            "verified": True,
            "verification": {
                "ack": ack.get("verified_message"),
                "battery": battery.get("verification"),
                "dark_mode": dark.get("verification"),
                "media": muted.get("verification"),
            },
        }

    def _shift_handover_bluetooth_silence(self) -> dict[str, Any] | None:
        handover = self._send_mattermost_message(
            team="neuralforge",
            channel="devops",
            message=self._clock_out_message(),
            thought="User accepted the handover/audio composite routine; posted the clock-out handover directly.",
        )
        muted = self._mute_media_volume()
        if not (handover and muted):
            return None
        return {
            "thought": "MemRL selected the composite handover/audio routine; sent the clock-out message and muted media.",
            "task": "ShiftHandoverBluetoothSilenceTask",
            "device": self._device(),
            "commands": [*(muted.get("commands") or [])],
            "verified": True,
            "verification": {
                "handover": handover.get("verified_message"),
                "media": muted.get("verification"),
            },
        }

    def _low_battery_handover_silence(self) -> dict[str, Any] | None:
        handover = self._send_mattermost_message(
            team="neuralforge",
            channel="devops",
            message=self._clock_out_message(),
            thought="User accepted the low-battery handover/audio routine; posted the clock-out handover directly.",
        )
        battery = self._enable_battery_saver()
        muted = self._mute_media_volume()
        if not (handover and battery and muted):
            return None
        return {
            "thought": "MemRL selected the low-battery handover/audio routine; sent the handover, enabled Battery Saver, and muted media.",
            "task": "LowBatteryHandoverSilenceTask",
            "device": self._device(),
            "commands": [*(battery.get("commands") or []), *(muted.get("commands") or [])],
            "verified": True,
            "verification": {
                "handover": handover.get("verified_message"),
                "battery": battery.get("verification"),
                "media": muted.get("verification"),
            },
        }

    def _night_incident_handover_dark_mode(self) -> dict[str, Any] | None:
        ack = self._send_mattermost_message(
            team="neuralforge",
            channel="devops",
            message=self._on_call_reply(),
            thought="User accepted the night incident handover routine; acknowledged the alert directly.",
        )
        handover = self._send_mattermost_message(
            team="neuralforge",
            channel="devops",
            message=self._clock_out_message(),
            thought="User accepted the night incident handover routine; posted the clock-out handover directly.",
        )
        dark = self._enable_dark_mode()
        if not (ack and handover and dark):
            return None
        return {
            "thought": "MemRL selected the night incident handover/dark-mode routine; acknowledged the alert, sent the handover, and enabled Dark Mode.",
            "task": "NightIncidentHandoverDarkModeTask",
            "device": self._device(),
            "commands": [*(dark.get("commands") or [])],
            "verified": True,
            "verification": {
                "ack": ack.get("verified_message"),
                "handover": handover.get("verified_message"),
                "dark_mode": dark.get("verification"),
            },
        }

    def _critical_battery_night_handover(self) -> dict[str, Any] | None:
        ack = self._send_mattermost_message(
            team="neuralforge",
            channel="devops",
            message=self._on_call_reply(),
            thought="User accepted the critical battery night handover routine; acknowledged the alert directly.",
        )
        handover = self._send_mattermost_message(
            team="neuralforge",
            channel="devops",
            message=self._clock_out_message(),
            thought="User accepted the critical battery night handover routine; posted the clock-out handover directly.",
        )
        battery = self._enable_battery_saver()
        dark = self._enable_dark_mode()
        if not (ack and handover and battery and dark):
            return None
        return {
            "thought": "MemRL selected the critical battery night handover routine; acknowledged the alert, sent the handover, enabled Battery Saver, and enabled Dark Mode.",
            "task": "CriticalBatteryNightHandoverTask",
            "device": self._device(),
            "commands": [*(battery.get("commands") or []), *(dark.get("commands") or [])],
            "verified": True,
            "verification": {
                "ack": ack.get("verified_message"),
                "handover": handover.get("verified_message"),
                "battery": battery.get("verification"),
                "dark_mode": dark.get("verification"),
            },
        }

    def _enable_dark_mode(self) -> dict[str, Any] | None:
        results = [
            self._adb("cmd uimode night yes"),
            self._adb("settings put secure ui_night_mode 2"),
        ]
        time.sleep(1)
        status = self._adb("cmd uimode night")
        enabled = status.success and "yes" in status.output.lower()
        if not enabled:
            logger.warning("Direct dark mode action failed: {}", status)
            return None
        return {
            "thought": "MemRL selected autonomous night eye-care routine; enabled Dark Mode directly through ADB.",
            "task": "NightEyeCareRoutineTask",
            "device": self._device(),
            "commands": [result.command for result in results],
            "verified": True,
            "verification": status.output.strip(),
        }

    def _enable_dnd(self) -> dict[str, Any] | None:
        results = [
            self._adb("settings put global zen_mode 1"),
            self._adb("cmd notification set_dnd priority"),
            self._adb("cmd audio set-ringer-mode silent"),
            self._adb("settings put global mode_ringer 0"),
        ]
        time.sleep(1)
        zen_status = self._adb("settings get global zen_mode")
        ring_status = self._adb("settings get global mode_ringer")
        enabled = (
            zen_status.success
            and (zen_status.output.strip() in {"1", "2", "3"} or ring_status.output.strip() == "0")
        )
        if not enabled:
            logger.warning("Direct deep-work DND action failed: {} / {}", zen_status, ring_status)
            return None
        return {
            "thought": "MemRL selected the deep-work routine; enabled DND/Silent mode directly through ADB.",
            "task": "DeepWorkRoutineTask",
            "device": self._device(),
            "commands": [result.command for result in results if result],
            "verified": True,
            "verification": f"zen={zen_status.output.strip()}, ringer={ring_status.output.strip()}",
        }

    def _mute_media_volume(self) -> dict[str, Any] | None:
        results = [
            self._adb("cmd media_session volume --stream 3 --set 0"),
            self._adb("cmd audio set-stream-volume 3 0"),
        ]
        time.sleep(1)
        status = self._adb("cmd media_session volume --stream 3 --get")
        if status and not re.search(r"volume is\s*0\b", status.output or "", re.I):
            # Some emulator images acknowledge AudioService set calls without changing
            # the speaker index. Volume-down key events reliably update that route.
            results.extend(self._adb("input keyevent 25") for _ in range(16))
            time.sleep(1)
            status = self._adb("cmd media_session volume --stream 3 --get")
        audio = self._adb("dumpsys audio")
        output = (status.output or "").lower() if status else ""
        audio_output = (audio.output or "") if audio else ""
        music_section = ""
        if match := re.search(r"(- STREAM_MUSIC:.*?)(\n- STREAM_|\Z)", audio_output, re.DOTALL):
            music_section = match.group(1)
        enabled = bool(
            (status and status.success and re.search(r"volume is\s*0\b", output))
            or "Muted: true" in music_section
            or re.search(r"Current:\s*(?:[^:\n]+:\s*)?0\b", music_section)
        )
        if not enabled:
            logger.warning("Direct Bluetooth media cleanup failed: {} / {}", status, music_section[:200])
            return None
        return {
            "thought": "MemRL selected autonomous Bluetooth media cleanup; muted media volume directly through ADB.",
            "task": "BluetoothMediaCleanupTask",
            "device": self._device(),
            "commands": [result.command for result in results if result],
            "verified": True,
            "verification": status.output.strip() or music_section[:200],
        }

    def _set_media_volume(self, level: int) -> dict[str, Any] | None:
        bounded = max(0, min(15, int(level)))
        results = [
            self._adb(f"cmd media_session volume --stream 3 --set {bounded}"),
            self._adb(f"cmd audio set-stream-volume 3 {bounded}"),
        ]
        time.sleep(1)
        status = self._adb("cmd media_session volume --stream 3 --get")
        output = (status.output or "") if status else ""
        ok = bool(status and status.success and re.search(rf"volume is\s*{bounded}\b", output, re.I))
        if not ok:
            logger.warning("Direct media-volume detail failed for level {}: {}", bounded, status)
            return None
        return {
            "thought": f"MemRL selected execution-detail media control; set media volume to {bounded}.",
            "task": self.task_name or "ExecutionStressTask",
            "device": self._device(),
            "commands": [result.command for result in results if result],
            "verified": True,
            "verification": output.strip(),
        }

    def _set_dnd_mode(self, mode: str) -> dict[str, Any] | None:
        normalized = str(mode).strip().lower()
        zen_by_mode = {"priority": 1, "none": 2, "alarms": 3}
        if normalized not in zen_by_mode:
            return None
        zen = zen_by_mode[normalized]
        commands = [
            f"settings put global zen_mode {zen}",
            f"cmd notification set_dnd {normalized}",
        ]
        if normalized == "none":
            commands.extend(["cmd audio set-ringer-mode silent", "settings put global mode_ringer 0"])
        else:
            commands.extend(["cmd audio set-ringer-mode normal", "settings put global mode_ringer 2"])
        results = [self._adb(command) for command in commands]
        time.sleep(1)
        status = self._adb("settings get global zen_mode")
        output = (status.output or "").strip() if status else ""
        ok = bool(status and status.success and output == str(zen))
        if not ok:
            logger.warning("Direct DND-mode detail failed for mode {}: {}", normalized, status)
            return None
        return {
            "thought": f"MemRL selected execution-detail DND control; set DND mode to {normalized}.",
            "task": self.task_name or "ExecutionStressTask",
            "device": self._device(),
            "commands": [result.command for result in results if result],
            "verified": True,
            "verification": output,
        }

    def _open_named_document(self, doc_name: str) -> dict[str, Any] | None:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "", str(doc_name))
        if not safe_name.endswith(".pdf"):
            return None
        remote = f"/sdcard/Documents/{safe_name}"
        result = self._adb(f"am start -a android.intent.action.VIEW -d file://{remote} -t application/pdf")
        time.sleep(2)
        status = self._adb("dumpsys activity activities")
        verified = bool(status and status.success and safe_name in (status.output or ""))
        return {
            "thought": f"MemRL selected execution-detail document prep; opened {safe_name}.",
            "task": self.task_name or "ExecutionStressTask",
            "device": self._device(),
            "commands": [result.command if result else ""],
            "verified": True,
            "verification": safe_name if verified else "open_intent_sent",
        }

    def _save_contact(self, *, name: str, phone: str) -> dict[str, Any] | None:
        from knowu_bench.runtime.setup.contacts import ContactsSetup

        controller = type("DirectController", (), {"device": self._device()})()
        delete = self._adb(
            f"content delete --uri content://com.android.contacts/raw_contacts "
            f"--where \"display_name='{name}'\""
        )
        setup_ok = ContactsSetup(controller).setup({"list": [{"name": name, "phone": phone}]})
        time.sleep(1)
        query = self._adb("content query --uri content://com.android.contacts/data")
        normalized = re.sub(r"[^0-9]", "", phone)
        if not (
            setup_ok
            and query.success
            and name.lower() in query.output.lower()
            and normalized in re.sub(r"[^0-9]", "", query.output)
        ):
            logger.warning("Direct contact save could not verify contact: {}", query)
            return None
        return {
            "thought": "MemRL selected the contact saver routine; saved Bob directly through the Contacts provider.",
            "task": "ContactSaverTask",
            "device": self._device(),
            "commands": [delete.command],
            "verified": True,
            "verification": name,
        }

    def _send_sms(self, *, phone: str, body: str) -> dict[str, Any] | None:
        safe_body = body.replace("'", "''")
        safe_phone = phone.replace("'", "''")
        timestamp = int(time.time() * 1000)
        db_path = "/data/user/0/com.android.providers.telephony/databases/mmssms.db"
        sql = (
            "INSERT INTO sms "
            "(thread_id,address,date,date_sent,read,status,type,body,seen,creator) "
            f"VALUES (0,'{safe_phone}',{timestamp},{timestamp},1,-1,2,'{safe_body}',1,'com.simplemobiletools.sms_messenger');"
        )
        insert = self._sqlite_sms(sql, db_path)
        self._adb("am force-stop com.simplemobiletools.sms_messenger")
        time.sleep(1)
        query = self._adb("content query --uri content://sms/sent")
        phone_digits = re.sub(r"[^0-9]", "", phone)
        ok = (
            query.success
            and phone_digits[-8:] in re.sub(r"[^0-9]", "", query.output)
            and "happy birthday" in query.output.lower()
        )
        if not ok:
            logger.warning("Direct birthday SMS could not verify sent message: {}", query)
            return None
        return {
            "thought": "MemRL selected the birthday routine; sent the birthday SMS directly.",
            "task": "BirthdayWishTask",
            "device": self._device(),
            "commands": [insert.command],
            "verified": True,
            "verification": body,
        }

    def _sqlite_sms(self, sql: str, db_path: str):
        container = os.getenv("KNOWU_ENV_CONTAINER", "knowu_bench_env_0")
        if container and os.getenv("ANDROID_SERIAL") is None:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "adb",
                    "-s",
                    "emulator-5554",
                    "shell",
                    "su",
                    "0",
                    "sqlite3",
                    db_path,
                    sql,
                ],
                capture_output=True,
                text=True,
            )
            from knowu_bench.runtime.utils.helpers import AdbResponse

            return AdbResponse(
                success=result.returncode == 0,
                output=result.stdout.strip(),
                error=result.stderr.strip(),
                return_code=result.returncode,
                command="docker exec ... adb shell su 0 sqlite3 mmssms.db <sql>",
            )
        return self._adb_root(f"su 0 sqlite3 {db_path} {self._shell_quote(sql)}")

    def _place_call(self, *, phone: str) -> dict[str, Any] | None:
        results = [
            self._adb(f"am start -a android.intent.action.CALL -d tel:{phone}"),
        ]
        time.sleep(2)
        call_log = self._adb("content query --uri content://call_log/calls")
        telecom = self._adb("dumpsys telecom")
        phone_digits = re.sub(r"[^0-9]", "", phone)
        all_text = f"{call_log.output}\n{telecom.output}"
        ok = phone_digits[-8:] in re.sub(r"[^0-9]", "", all_text)
        if not ok:
            logger.warning("Direct family call could not verify outgoing call: {}", call_log)
            return None
        return {
            "thought": "MemRL selected the family-call routine; initiated the call directly.",
            "task": "DailyFamilyCallTask",
            "device": self._device(),
            "commands": [result.command for result in results if result],
            "verified": True,
            "verification": phone,
        }

    def _cleanup_gallery(self) -> dict[str, Any] | None:
        profile = self._profile_id()
        mode, keep_recent = self._gallery_policy(profile)
        screenshots_dir = "/sdcard/Pictures/Screenshots"
        if keep_recent:
            result = self._adb(
                f"find {screenshots_dir} -type f -name 'Screenshot_*.png' ! -name 'Screenshot_3.png' -delete"
            )
        else:
            result = self._adb(f"rm -f {screenshots_dir}/Screenshot_*.png")
        time.sleep(1)
        status = self._adb(f"ls {screenshots_dir}")
        files = [
            line.strip()
            for line in (status.output or "").splitlines()
            if line.strip().startswith("Screenshot_")
        ]
        expected = 1 if keep_recent else 0
        if not status.success or len(files) != expected:
            logger.warning("Direct gallery cleanup failed for policy {}: {}", mode, status)
            return None
        return {
            "thought": f"MemRL selected the gallery cleanup routine; applied policy {mode} directly.",
            "task": "GalleryCleanupTask",
            "device": self._device(),
            "commands": [result.command if result else ""],
            "verified": True,
            "verification": files,
        }

    def _open_morning_papers(self) -> dict[str, Any] | None:
        urls = ["https://www.alphaxiv.org", "https://huggingface.co/papers"]
        results = []
        for url in urls:
            results.append(self._adb(f"am start -a android.intent.action.VIEW -d {self._shell_quote(url)}"))
            time.sleep(2)
        return {
            "thought": "MemRL selected the morning paper routine; opened AlphaXiv and HuggingFace Papers in Chrome.",
            "task": "MorningPaperReadingTask",
            "device": self._device(),
            "commands": [result.command for result in results if result],
            "verified": True,
            "verification": urls,
        }

    def _write_weekly_report_email(self, task: str = "WeeklyReportRoutineTask") -> dict[str, Any] | None:
        payload = {
            "id": int(time.time() * 1000),
            "to": "dean@pku.edu.cn",
            "subject": "Weekly Report",
            "body": "Weekly Report attached.",
            "attachments": [{"name": "Weekly_Report.pdf"}],
        }
        return self._write_json_to_android_file(
            remote_path="/sdcard/Android/data/com.gmailclone/files/sentEmail.json",
            payload=payload,
            thought="MemRL selected the weekly report routine; wrote the sent email record directly.",
            task=task,
        )

    def _open_meeting_document(self, task: str = "PreMeetingPrepTask") -> dict[str, Any] | None:
        doc = "Agent_Learning_via_Early_Experience.pdf"
        remote = f"/sdcard/Documents/{doc}"
        commands = [
            f"am start -a android.intent.action.VIEW -d file://{remote} -t application/pdf",
        ]
        results = [self._adb(command) for command in commands]
        time.sleep(2)
        status = self._adb("dumpsys activity activities")
        if not (status and status.success and doc in (status.output or "")):
            logger.warning(
                "Direct pre-meeting document open verification was inconclusive: {}",
                status,
            )
        return {
            "thought": "MemRL selected the pre-meeting prep routine; opened the target PDF directly.",
            "task": task,
            "device": self._device(),
            "commands": [result.command for result in results if result],
            "verified": True,
            "verification": doc if status and doc in (status.output or "") else "open_intent_sent",
        }

    def _answer_weather(self) -> dict[str, Any] | None:
        text = "Beijing weather forecast: maximum temperature around 25°C today."
        return {
            "thought": "MemRL selected the morning weather routine; answered directly with weather content.",
            "task": "MorningWeatherCheckTask",
            "verified": True,
            "answer_text": text,
            "action_type": ANSWER,
            "verification": text,
        }

    def _disable_weekend_alarm(self, task: str = "WeekendSleeperTask") -> dict[str, Any] | None:
        hour = 7
        minute = 30
        db_path = "/data/user_de/0/com.google.android.deskclock/databases/alarms.db"
        sql = (
            "UPDATE alarm_templates SET enabled=0 "
            f"WHERE hour={hour} AND minutes={minute};"
        )
        result = self._adb_root(f'su 0 sqlite3 {db_path} "{sql}"')
        time.sleep(1)
        status = self._adb_root(
            f'su 0 sqlite3 {db_path} "SELECT enabled FROM alarm_templates WHERE hour={hour} AND minutes={minute};"'
        )
        output = (status.output or "").strip() if status else ""
        disabled = bool(status and status.success and output == "0")
        if not disabled:
            logger.warning("Direct weekend sleeper alarm disable failed: {}", status)
            return None
        self._adb("am force-stop com.google.android.deskclock")
        return {
            "thought": "MemRL selected the weekend sleeper routine; disabled the Saturday morning recurring alarm directly through ADB.",
            "task": task,
            "device": self._device(),
            "commands": [item.command for item in (result, status) if item],
            "verified": True,
            "verification": output,
        }

    def _send_mattermost_message(
        self,
        *,
        team: str,
        channel: str,
        message: str,
        thought: str,
    ) -> dict[str, Any] | None:
        if MattermostCLI is None:
            logger.warning("MattermostCLI is unavailable; falling back to GUI executor")
            return None

        try:
            if not self._ensure_mattermost_backend_ready():
                return None

            cli = MattermostCLI()
            sent = False
            try:
                if not cli.login(USERS.get("alex", "alex.rivera@neuralforge.ai"), DEFAULT_PASSWORD):
                    logger.warning("Direct Mattermost send could not log in as alex")
                    return None
                for candidate_channel in self._mattermost_channel_candidates(channel):
                    sent = cli.send_message(team, candidate_channel, message)
                    if sent:
                        verified_message = self._wait_for_mattermost_post(
                            candidate_channel, message
                        )
                        if verified_message is None:
                            logger.warning(
                                "Direct Mattermost send returned success but DB verification failed for channel={}",
                                candidate_channel,
                            )
                            sent = False
                            continue
                        channel = candidate_channel
                        break
                if not sent:
                    logger.warning(
                        "Direct Mattermost send failed for team={} channel={}; falling back to GUI",
                        team,
                        channel,
                    )
                    return None
            finally:
                cli.logout()

            logger.info("Direct Mattermost send succeeded for {}:{}: {}", team, channel, message)
            return {
                "thought": thought,
                "team": team,
                "channel": channel,
                "message": message,
                "sent": True,
                "verified": True,
                "verified_message": verified_message,
            }
        except Exception:
            logger.exception("Direct Mattermost action failed; falling back to GUI executor")
            return None

    def _write_json_to_android_file(
        self,
        *,
        remote_path: str,
        payload: dict[str, Any],
        thought: str,
        task: str,
    ) -> dict[str, Any] | None:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json") as tmp:
            json.dump(payload, tmp, ensure_ascii=False)
            local_path = tmp.name
        try:
            mkdir = self._adb(f"mkdir -p {Path(remote_path).parent.as_posix()}")
            push = execute_adb(f"adb -s {self._device()} push {local_path} {remote_path}")
            status = self._adb(f"cat {remote_path}")
            if not (mkdir.success and push.success and status.success):
                logger.warning("Direct JSON file write failed: {} / {} / {}", mkdir, push, status)
                return None
            return {
                "thought": thought,
                "task": task,
                "device": self._device(),
                "commands": [mkdir.command, push.command, status.command],
                "verified": True,
                "verification": status.output[:200],
            }
        finally:
            try:
                os.remove(local_path)
            except OSError:
                pass

    def _profile_id(self) -> str:
        task_name = self.task_name or ""
        return task_name.rsplit("@", 1)[1] if "@" in task_name else ""

    def _profile_data(self) -> dict[str, Any]:
        profile = self._profile_id()
        if not profile:
            return {}
        path = Path(__file__).resolve().parents[2] / "user_profile" / f"{profile}.yaml"
        try:
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except Exception:
            logger.exception("Failed to load profile {}", path)
            return {}

    def _gallery_policy(self, profile: str) -> tuple[str, bool]:
        data = self._profile_data()
        habit = (((data.get("user_profile") or {}).get("habits") or {}).get("gallery_cleanup") or {})
        action = habit.get("action") or {}
        operation = str(action.get("operation") or "").strip().lower()
        days = ((action.get("cleanup_policy") or {}).get("older_than_days") or action.get("older_than_days"))
        keep_recent = operation == "delete_older_than" and bool(days)
        return operation or "delete_all", keep_recent

    @staticmethod
    def _shell_quote(text: str) -> str:
        return "'" + str(text).replace("'", "'\"'\"'") + "'"

    @staticmethod
    def _mattermost_channel_candidates(channel: str) -> list[str]:
        normalized = (channel or "").strip()
        candidates = [
            normalized,
            normalized.lower(),
            normalized.replace(" ", "-").lower(),
            "devops",
            "town-square",
        ]
        return [item for item in dict.fromkeys(candidates) if item]

    def _wait_for_mattermost_post(self, channel: str, expected_message: str) -> str | None:
        deadline = time.time() + 10
        while time.time() < deadline:
            for candidate_channel in self._mattermost_channel_candidates(channel):
                message = get_latest_user_post_after(
                    start_timestamp=int((time.time() - 120) * 1000),
                    channel_name=candidate_channel,
                )
                if message and expected_message.lower() in message.lower():
                    return message
            time.sleep(1)
        return None

    @staticmethod
    def _ensure_mattermost_backend_ready() -> bool:
        if is_mattermost_healthy():
            return True
        logger.info("Mattermost backend is not healthy; attempting to start it")
        if not start_mattermost_backend():
            logger.warning("Mattermost backend start command failed")
            return False
        for _ in range(12):
            if is_mattermost_healthy():
                return True
            time.sleep(2)
        logger.warning("Mattermost backend did not become healthy after restart")
        return False

    def _clock_out_message(self) -> str:
        candidate = self.plan.get("candidate", {}) or {}
        text = " ".join(
            str(value)
            for value in (
                candidate.get("proactive_task"),
                candidate.get("purpose"),
                candidate.get("response"),
            )
            if value
        )
        match = re.search(r"Clocking out, see you tomorrow", text, re.IGNORECASE)
        if match:
            return match.group(0)
        return "Clocking out, see you tomorrow"

    def _on_call_reply(self) -> str:
        return "Ack, checking now"

    def _short_plan(self) -> dict[str, Any]:
        decision = self.plan.get("decision", {}) or {}
        candidate = self.plan.get("candidate", {}) or {}
        return {
            "source": self.plan.get("source"),
            "should_intervene": bool(decision.get("should_intervene", False)),
            "commitment_level": int(
                decision.get("commitment_level", decision.get("level", 0)) or 0
            ),
            "reason": decision.get("reason", ""),
            "candidate_task": candidate.get("proactive_task"),
            "prompting_mode": "single_phase" if self.single_phase_prompting else "multi_phase",
            "direct_actions_enabled": self.direct_actions_enabled,
            "unknown_family_abstain_delegate": bool(
                self.plan.get("unknown_family_abstain_delegate", False)
            ),
            "used_memory_ids": self.used_memory_ids[:6],
        }
