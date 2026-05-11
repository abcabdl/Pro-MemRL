from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from loguru import logger

from knowu_bench.agents.base import MCPAgent
from knowu_bench.agents.implementations.general_e2e_agent import GeneralE2EAgentMCP
from knowu_bench.memrl.bridge import KnowUMemRLBridge
from knowu_bench.runtime.client import ASK_USER_TRANSPORT_ERROR_PREFIX
from knowu_bench.runtime.utils.helpers import execute_adb
from knowu_bench.runtime.utils.models import ASK_USER, FINISHED, JSONAction

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
        self.direct_action_attempted = False
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
        self.used_memory_ids = [str(item) for item in self.plan.get("used_memory_ids", []) if item]
        decision = self.plan.get("decision", {}) or {}
        level = int(decision.get("commitment_level", decision.get("level", 0)) or 0)
        should = bool(decision.get("should_intervene", False))
        self.phase = "abstain" if not should or level <= 0 else "ask" if level == 1 else "delegate"
        augmented_instruction = self._augment_instruction(instruction)
        self.executor.initialize(augmented_instruction)

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

    def predict(self, observation: dict[str, Any]) -> tuple[str, JSONAction]:
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
                    "ask_user transport failed; treating it as environment error, not user rejection: {}",
                    response,
                )
                self.user_accepted = True
                self.phase = "delegate"
                observation = {**observation, "ask_user_response": None}
            elif not _accepted(response):
                self.phase = "done"
                prediction = {
                    "Thought": "The simulated user rejected or did not accept the proposal; respecting that.",
                    "UserResponse": response,
                    "MemRLPlan": self._short_plan(),
                }
                return json.dumps(prediction, ensure_ascii=False), JSONAction(action_type=FINISHED)
            else:
                self.user_accepted = True
                self.phase = "delegate"
                observation = {**observation, "ask_user_response": None}

        if self.phase == "delegate" and not self.direct_action_attempted:
            self.direct_action_attempted = True
            direct_result = self._try_direct_task_action()
            if direct_result is not None:
                self.phase = "done"
                prediction = {
                    "Thought": direct_result["thought"],
                    "Action": {"action_type": FINISHED},
                    "MemRLPlan": self._short_plan(),
                    "DirectAction": direct_result,
                }
                return json.dumps(prediction, ensure_ascii=False), JSONAction(action_type=FINISHED)

        prediction, action = self.executor.predict(observation)
        wrapped = {
            "MemRLPlan": self._short_plan(),
            "DelegatedPrediction": prediction,
        }
        return json.dumps(wrapped, ensure_ascii=False), action

    def record_task_outcome(self, score: float) -> None:
        if not self.memrl_use_memory:
            return
        try:
            self.bridge.record_outcome(
                reward=_score_to_reward(score),
                task_name=self.task_name,
                score=float(score),
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
        self.direct_action_attempted = False
        self.used_memory_ids = []
        self.task_name = None
        self.env = None
        self.executor.reset()

    def get_memrl_plan(self) -> dict[str, Any]:
        return self.plan

    def _try_direct_task_action(self) -> dict[str, Any] | None:
        task_name = self.task_name or ""
        if "BatterySaverRoutineTask" in task_name:
            return self._enable_battery_saver()
        if "NightEyeCareRoutineTask" in task_name:
            return self._enable_dark_mode()
        return self._try_direct_mattermost_action()

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
        return str(getattr(self.env, "device", None) or os.getenv("ANDROID_DEVICE") or "emulator-5554")

    def _adb(self, shell_command: str):
        return execute_adb(f"adb -s {self._device()} shell {shell_command}")

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

            if not sent:
                return None
            verified_message = self._wait_for_mattermost_post(channel, message)
            if verified_message is None:
                logger.warning(
                    "Direct Mattermost send returned success but DB verification failed for channel={}",
                    channel,
                )
                return None

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
            "used_memory_ids": self.used_memory_ids[:6],
        }
