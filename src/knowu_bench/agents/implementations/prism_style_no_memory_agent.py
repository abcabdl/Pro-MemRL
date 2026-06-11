from __future__ import annotations

import json
import os
import re
from typing import Any

from loguru import logger

from knowu_bench.agents.implementations.memrl_knowu_agent import MemRLKnowUAgentMCP
from knowu_bench.runtime.utils.models import FINISHED, JSONAction


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _json_from_response(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _bounded_prob(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _commitment_level_from_text(text: str) -> int:
    lowered = text.lower()
    if re.search(r"delete|send|post|buy|order|payment|call|message|sms", lowered):
        return 1
    return 2


class PRISMStyleNoMemoryAgentMCP(MemRLKnowUAgentMCP):
    """PRISM-style risk gate for KnowU tasks without memory retrieval or updates.

    This is an adapted baseline, not the released PRISM model. It keeps the KnowU
    executor interface and direct stress-action helpers, but replaces MemRL
    retrieval with a no-memory cost-sensitive intervention gate.
    """

    def __init__(
        self,
        model_name: str,
        llm_base_url: str,
        api_key: str = "empty",
        tools: list[dict] | None = None,
        runtime_conf: dict | None = None,
        scale_factor: int = 1000,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("memrl_use_memory", None)
        super().__init__(
            model_name=model_name,
            llm_base_url=llm_base_url,
            api_key=api_key,
            tools=tools or [],
            runtime_conf=runtime_conf,
            scale_factor=scale_factor,
            memrl_use_memory=False,
            **kwargs,
        )
        self.c_false_alarm = _env_float("KNOWU_PRISM_C_FALSE_ALARM", 1.0)
        self.c_false_negative = _env_float("KNOWU_PRISM_C_FALSE_NEGATIVE", 1.0)
        self.default_threshold = _env_float("KNOWU_PRISM_DEFAULT_THRESHOLD", 0.5)

    def initialize_hook(self, instruction: str) -> None:
        logger.info("Initializing PRISM-style no-memory gate for task {}", self.task_name)
        gate = self._run_prism_gate(instruction)
        candidate = gate.get("candidate", {}) if isinstance(gate.get("candidate"), dict) else {}
        decision = gate.get("decision", {}) if isinstance(gate.get("decision"), dict) else {}
        level = int(decision.get("commitment_level", 0) or 0)
        should = bool(decision.get("should_intervene", False))

        self.plan = {
            "source": "prism_style_no_memory_gate",
            "memory_enabled": False,
            "memrl_chain": "disabled",
            "task_name": self.task_name,
            "observations": [],
            "candidate": candidate,
            "simulation": gate.get("simulation", {}),
            "decision": decision,
            "gate": gate.get("gate", {}),
            "used_memory_ids": [],
            "generation_prior": {"used_memory_ids": [], "disabled": True},
            "simulation_prior": {"used_memory_ids": [], "disabled": True},
            "decision_prior": {"used_memory_ids": [], "disabled": True},
        }
        self.used_memory_ids = []
        self.phase = "abstain" if not should or level <= 0 else "ask" if level == 1 else "delegate"
        self.executor.initialize(self._augment_instruction(instruction))

    def _run_prism_gate(self, instruction: str) -> dict[str, Any]:
        prompt = (
            "You are an adapted PRISM-style proactive intervention gate for KnowU-Bench.\n"
            "Do not use memory. Decide from only the current task instruction, current system "
            "environment, and visible user logs.\n\n"
            "Estimate probabilities in [0,1]:\n"
            "- p_need: probability the user currently needs a proactive intervention.\n"
            "- p_accept: probability the user would accept this intervention if offered.\n"
            "- p_risk: probability the intervention is intrusive, harmful, or wrong.\n\n"
            "Use cost-sensitive abstention: intervene only when the expected acceptance is "
            "high enough for the need and risk. If intervening, propose exactly one concrete "
            "Android task. For execution-detail stress tasks, include exact details such as "
            "`set_media_volume:<level>`, `set_dnd_mode:<priority|none|alarms>`, or "
            "`open_document:<file.pdf>` when the context specifies them.\n\n"
            "Return strict JSON only with keys: p_need, p_accept, p_risk, should_intervene, "
            "commitment_level, proactive_task, response, reason."
        )
        response = self.openai_chat_completions_create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": instruction},
            ],
            retry_times=1,
            temperature=0.0,
            max_tokens=900,
        )
        data = _json_from_response(response)
        p_need = _bounded_prob(data.get("p_need"), 0.0)
        p_accept = _bounded_prob(data.get("p_accept"), 0.0)
        p_risk = _bounded_prob(data.get("p_risk"), 0.0)
        threshold = self._threshold(p_need, p_risk)
        model_should = bool(data.get("should_intervene", False))
        should = model_should and p_accept >= threshold
        task = str(data.get("proactive_task") or "").strip()
        if should and not task:
            task = self._fallback_candidate_from_instruction(instruction)
        if not should:
            task = ""
        level = int(data.get("commitment_level") or 0)
        if should and level <= 0:
            level = _commitment_level_from_text(task)
        if not should:
            level = 0

        return {
            "candidate": {
                "purpose": str(data.get("reason") or ""),
                "proactive_task": task or None,
                "response": str(data.get("response") or "") or None,
                "operation": "execute" if should else "nop",
            },
            "simulation": {
                "acceptance": "accept" if should else "ignore",
                "acceptance_confidence": p_accept,
                "risk": p_risk,
            },
            "decision": {
                "should_intervene": should,
                "commitment_level": level,
                "risk": "high" if p_risk >= 0.7 else "medium" if p_risk >= 0.35 else "low",
                "reason": str(data.get("reason") or "PRISM-style no-memory cost gate."),
            },
            "gate": {
                "p_need": p_need,
                "p_accept": p_accept,
                "p_risk": p_risk,
                "threshold": threshold,
                "model_should_intervene": model_should,
                "raw_response": response,
            },
        }

    def _threshold(self, p_need: float, p_risk: float) -> float:
        denom = self.c_false_alarm + max(0.0, p_need) * self.c_false_negative
        if denom <= 0:
            threshold = self.default_threshold
        else:
            threshold = self.c_false_alarm / denom
        return max(0.05, min(0.95, threshold + 0.25 * p_risk))

    def _fallback_candidate_from_instruction(self, instruction: str) -> str:
        text = instruction.lower()
        if "media volume" in text:
            if match := re.search(r"volume (?:to|should be|is)\s*(\d+)", text):
                return f"set_media_volume:{match.group(1)}"
            return "set_media_volume:0"
        if "dnd" in text or "do not disturb" in text:
            for mode in ("priority", "alarms", "none"):
                if mode in text:
                    return f"set_dnd_mode:{mode}"
            return "set_dnd_mode:priority"
        if match := re.search(r"([A-Za-z0-9_.-]+\.pdf)", instruction):
            return f"open_document:{match.group(1)}"
        if "battery" in text:
            return "enable_battery_saver"
        if "dark mode" in text:
            return "enable_dark_mode"
        return "perform the warranted proactive routine"
