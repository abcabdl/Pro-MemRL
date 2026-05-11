from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = ROOT.parent / "ProactiveAgent-main"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from agent.memrl import ProactiveMemRLRuntime
from agent.memrl.schema import enrich_memory_schema, summarize_memory_case
from eval.memrl_eval_prompts import (
    MEMRL_DECISION_SYSTEM,
    MEMRL_PHASE_B_SYSTEM,
    MEMRL_PHASE_G_SYSTEM,
    PHASE_A_SYSTEM,
    PHASE_C_SYSTEM,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _load_original_pipeline() -> Any:
    pipeline_path = ORIGINAL_ROOT / "proactive_pipeline.py"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Original proactive pipeline not found: {pipeline_path}")
    spec = importlib.util.spec_from_file_location("original_proactive_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load original proactive pipeline: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIPELINE = _load_original_pipeline()


class SimpleProgressBar:
    def __init__(self, total: int, *, desc: str) -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.current = 0
        self.start = time.time()
        self._last_print = 0.0
        self._render(force=True)

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + int(step))
        now = time.time()
        if now - self._last_print >= 0.2 or self.current >= self.total:
            self._render(force=self.current >= self.total)

    def _render(self, *, force: bool = False) -> None:
        self._last_print = time.time()
        ratio = min(max(self.current / self.total, 0.0), 1.0)
        width = 28
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = self._last_print - self.start
        message = f"\r{self.desc} [{bar}] {self.current}/{self.total} {ratio * 100:5.1f}% elapsed={elapsed:6.1f}s"
        end = "\n" if force and self.current >= self.total else ""
        print(message, end=end, file=sys.stderr, flush=True)

    def close(self) -> None:
        self._render(force=True)


def make_progress(total: int, *, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, dynamic_ncols=True)
    return SimpleProgressBar(total, desc=desc)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_seed_path(path: Path) -> Path:
    if path.is_dir():
        snapshot = path / "memrl_snapshot.jsonl"
        if snapshot.exists():
            return snapshot
        raise FileNotFoundError(f"Could not find memrl_snapshot.jsonl under {path}")
    return path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_decision_trace(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = PIPELINE.normalize_decision(payload)
    return {**normalized, "commitment_level": int(normalized.get("level", 0) or 0)}


def _normalize_simulation_trace(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = PIPELINE.normalize_simulation(payload)
    normalized.pop("total_score", None)
    return normalized


def _memory_generation_payload(prior: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_context_family": str(prior.get("current_context_family", "general")),
        "preferred_level": int(prior.get("preferred_level", 0) or 0),
        "preferred_action_families": list(prior.get("preferred_action_families", []) or []),
        "disallowed_action_families": list(prior.get("disallowed_action_families", []) or []),
        "generation_recommendation": dict(prior.get("generation_recommendation", {}) or {}),
        "intervene_memory_value": _safe_float(prior.get("intervene_memory_value", 0.0)),
        "abstain_memory_value": _safe_float(prior.get("abstain_memory_value", 0.0)),
        "positive_patterns": list(prior.get("positive_patterns", []) or []),
        "negative_patterns": list(prior.get("negative_patterns", []) or []),
        "avoid_patterns": list(prior.get("avoid_patterns", []) or []),
        "used_memory_ids": list(prior.get("used_memory_ids", []) or []),
    }


def _memory_simulation_payload(prior: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_context_family": str(prior.get("current_context_family", "general")),
        "candidate_action_family": str(prior.get("candidate_action_family", "no_intervention")),
        "historical_accept_rate": _safe_float(prior.get("historical_accept_rate", 0.0)),
        "historical_dismiss_rate": _safe_float(prior.get("historical_dismiss_rate", 0.0)),
        "historical_annoy_rate": _safe_float(prior.get("historical_annoy_rate", 0.0)),
        "historical_reject_risk": _safe_float(prior.get("historical_reject_risk", 0.0)),
        "support_cases": [summarize_memory_case(item) for item in list(prior.get("support_cases", []) or [])],
        "risk_cases": [summarize_memory_case(item) for item in list(prior.get("risk_cases", []) or [])],
        "used_memory_ids": list(prior.get("used_memory_ids", []) or []),
    }


def _memory_decision_payload(prior: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_context_family": str(prior.get("current_context_family", "general")),
        "candidate_action_family": str(prior.get("candidate_action_family", "no_intervention")),
        "intervene_memory_value": _safe_float(prior.get("intervene_memory_value", 0.0)),
        "abstain_memory_value": _safe_float(prior.get("abstain_memory_value", 0.0)),
        "memory_level_mode": int(prior.get("memory_level_mode", 0) or 0),
        "historical_reject_risk": _safe_float(prior.get("historical_reject_risk", 0.0)),
        "memory_recommendation": dict(prior.get("memory_recommendation", {}) or {}),
        "support_cases": [summarize_memory_case(item) for item in list(prior.get("intervene_memories", []) or [])],
        "risk_cases": [summarize_memory_case(item) for item in list(prior.get("abstain_memories", []) or [])],
        "used_memory_ids": list(prior.get("used_memory_ids", []) or []),
    }


def _memory_trace_payload(
    *,
    gate_prior: dict[str, Any],
    generation_prior: dict[str, Any],
    simulation_prior: dict[str, Any],
    decision_prior: dict[str, Any],
) -> dict[str, Any]:
    return {
        "gate": list(gate_prior.get("used_memory_ids", []) or []),
        "generation": list(generation_prior.get("used_memory_ids", []) or []),
        "simulation": list(simulation_prior.get("used_memory_ids", []) or []),
        "decision": list(decision_prior.get("used_memory_ids", []) or []),
    }


def _build_messages(system: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


async def _call_openai_compatible(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for _ in range(max(1, int(max_retries))):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
            )
            raw = response.choices[0].message.content or "{}"
            return PIPELINE.parse_json_payload(raw)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            await asyncio.sleep(0.5)
    print(
        f"[warn] API call failed after {max_retries} attempts; falling back to empty payload: {last_exc}",
        file=sys.stderr,
        flush=True,
    )
    return {}


def _decision_from_payload(payload: dict[str, Any], *, candidate: dict[str, Any]) -> dict[str, Any]:
    raw_decision = payload.get("decision", payload.get("intervention_recommendation", payload))
    if not isinstance(raw_decision, dict):
        raw_decision = {}
    normalized = _normalize_decision_trace(raw_decision)
    if not any(candidate.values()):
        return _normalize_decision_trace(
            {"should_intervene": False, "level": 0, "risk": "low", "reason": "no_candidate_generated"}
        )
    if not candidate.get("proactive_task"):
        return _normalize_decision_trace(
            {"should_intervene": False, "level": 0, "risk": "low", "reason": "candidate_missing_task"}
        )
    return normalized


def _reward_from_gold(pred_decision: dict[str, Any], gold_decision: dict[str, Any]) -> float:
    pred_should = int(bool(pred_decision.get("should_intervene", False)))
    pred_level = int(pred_decision.get("commitment_level", pred_decision.get("level", 0)) or 0)
    gold_should = int(gold_decision.get("should_intervene", 0) or 0)
    gold_level = int(gold_decision.get("commitment_level", 0) or 0)
    if pred_should == gold_should and pred_level == gold_level:
        return 1.0
    if pred_should == gold_should:
        return 0.4
    if pred_should == 1 and gold_should == 0:
        return -1.0
    return -0.8


def _empty_generation_prior() -> dict[str, Any]:
    return {
        "current_context_family": "general",
        "preferred_level": 0,
        "preferred_action_families": [],
        "disallowed_action_families": [],
        "positive_patterns": [],
        "negative_patterns": [],
        "avoid_patterns": [],
        "generation_context": "{}",
        "used_memory_ids": [],
    }


def _empty_gate_prior() -> dict[str, Any]:
    return {
        "current_context_family": "general",
        "similar_case_count": 0,
        "historical_intervene_value": 0.0,
        "historical_abstain_value": 0.0,
        "historical_reject_risk": 0.0,
        "missed_help_risk": 0.0,
        "confidence": 0.0,
        "margin": 0.0,
        "recommended_signal_delta": {},
        "recommended_threshold_delta": {},
        "gate_context": "{}",
        "used_memory_ids": [],
    }


def _empty_simulation_prior() -> dict[str, Any]:
    return {
        "current_context_family": "general",
        "candidate_action_family": "no_intervention",
        "historical_accept_rate": 0.0,
        "historical_dismiss_rate": 0.0,
        "historical_annoy_rate": 0.0,
        "historical_reject_risk": 0.0,
        "support_cases": [],
        "risk_cases": [],
        "simulation_context": "{}",
        "used_memory_ids": [],
    }


def _empty_decision_prior() -> dict[str, Any]:
    return {
        "current_context_family": "general",
        "candidate_action_family": "no_intervention",
        "intervene_memory_value": 0.0,
        "abstain_memory_value": 0.0,
        "memory_level_mode": 0,
        "historical_reject_risk": 0.0,
        "memory_recommendation": {
            "should_intervene": None,
            "level": 0,
            "confidence": 0.0,
            "margin": 0.0,
            "reason": "no_memory_used",
            "context_family": "general",
            "action_family": "no_intervention",
        },
        "decision_context": "{}",
        "used_memory_ids": [],
    }


def _gate_memory_augmented_result(
    *,
    base_result: dict[str, Any],
    memory_result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_decision = dict(base_result.get("decision", {}) or {})
    memory_decision = dict(memory_result.get("decision", {}) or {})
    base_candidate = dict(base_result.get("candidate", {}) or {})
    memory_candidate = dict(memory_result.get("candidate", {}) or {})
    recommendation = (
        dict((memory_result.get("decision_prior", {}) or {}).get("memory_recommendation", {}) or {})
        if isinstance(memory_result.get("decision_prior", {}), dict)
        else {}
    )
    confidence = _safe_float(recommendation.get("confidence", 0.0))
    margin = _safe_float(recommendation.get("margin", 0.0))
    positive_intervene_ratio = _safe_float(recommendation.get("positive_intervene_ratio", 0.0))
    rec_should = recommendation.get("should_intervene")

    base_should = bool(base_decision.get("should_intervene", False))
    memory_should = bool(memory_decision.get("should_intervene", False))
    base_level = int(base_decision.get("commitment_level", base_decision.get("level", 0)) or 0)
    memory_level = int(memory_decision.get("commitment_level", memory_decision.get("level", 0)) or 0)

    same_decision = (base_should == memory_should) and (base_level == memory_level)
    same_candidate = base_candidate == memory_candidate
    same_output = (base_result.get("final_output", {}) or {}) == (memory_result.get("final_output", {}) or {})
    if same_decision and same_candidate and same_output:
        return memory_result, {
            "applied": False,
            "kept_memory_result": True,
            "reason": "memory_result_matches_baseline",
            "confidence": confidence,
            "recommendation_should_intervene": rec_should,
        }

    positive_conf_threshold = 0.42
    positive_margin_threshold = 0.30
    positive_ratio_threshold = 0.62
    positive_refine_conf_threshold = 0.48
    positive_refine_margin_threshold = 0.22
    positive_refine_ratio_threshold = 0.58
    negative_conf_threshold = 0.60
    strong_positive_support = (
        (confidence >= positive_conf_threshold and margin >= positive_margin_threshold)
        or (confidence >= 0.36 and margin >= 0.50 and positive_intervene_ratio >= 0.72)
    )
    strong_positive_refine_support = (
        (confidence >= positive_refine_conf_threshold and margin >= positive_refine_margin_threshold)
        or (confidence >= 0.40 and margin >= 0.40 and positive_intervene_ratio >= 0.68)
    )
    if not same_decision and (not memory_should) and base_should:
        allow = (rec_should is False) and confidence >= negative_conf_threshold
        reason = "high_conf_negative_recommendation_veto" if allow else "blocked_low_conf_negative_recommendation"
    elif not same_decision and memory_should and (not base_should):
        allow = (
            (rec_should is True)
            and strong_positive_support
            and positive_intervene_ratio >= positive_ratio_threshold
        )
        reason = "high_conf_positive_recommendation_lift" if allow else "blocked_low_conf_positive_recommendation"
    elif base_should and memory_should and ((base_level != memory_level) or (not same_candidate) or (not same_output)):
        allow = (
            (rec_should is True)
            and strong_positive_refine_support
            and positive_intervene_ratio >= positive_refine_ratio_threshold
        )
        reason = "high_conf_positive_recommendation_refine" if allow else "blocked_low_conf_refinement"
    else:
        allow = False
        reason = "blocked_unmatched_memory_change"

    if allow:
        return memory_result, {
            "applied": True,
            "kept_memory_result": True,
            "reason": reason,
            "confidence": confidence,
            "margin": margin,
            "positive_intervene_ratio": positive_intervene_ratio,
            "recommendation_should_intervene": rec_should,
        }
    return base_result, {
        "applied": False,
        "kept_memory_result": False,
        "reason": reason,
        "confidence": confidence,
        "margin": margin,
        "positive_intervene_ratio": positive_intervene_ratio,
        "recommendation_should_intervene": rec_should,
    }


async def run_original_chain_eval(
    *,
    bootstrap_path: Path,
    test_root: Path,
    output_root: Path,
    snapshot_root: Path,
    gold_root: Path | None = None,
    api_base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    history_window: int = 32,
    temperature: float = 0.0,
    max_retries: int = 5,
    allow_gold_update: bool = False,
    run_baseline: bool = False,
    baseline_output_root: Path | None = None,
    save_updated_memory: bool = True,
    responder: Callable[[list[dict[str, str]]], Awaitable[dict[str, Any]]] | None = None,
) -> None:
    runtime = ProactiveMemRLRuntime()
    runtime.warm_start(str(_runtime_seed_path(bootstrap_path)))
    output_root.mkdir(parents=True, exist_ok=True)
    if run_baseline:
        if baseline_output_root is None:
            baseline_output_root = output_root.with_name(f"{output_root.name}_baseline")
        baseline_output_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    if gold_root is not None:
        splits_path = gold_root / "splits.json"
        if splits_path.exists():
            shutil.copy2(splits_path, output_root / "splits.json")
            if run_baseline and baseline_output_root is not None:
                shutil.copy2(splits_path, baseline_output_root / "splits.json")
    else:
        splits_path = test_root / "splits.json"
        if splits_path.exists():
            shutil.copy2(splits_path, output_root / "splits.json")
            if run_baseline and baseline_output_root is not None:
                shutil.copy2(splits_path, baseline_output_root / "splits.json")

    client: AsyncOpenAI | None = None
    if responder is None:
        if not api_base_url or not api_key or not model:
            raise ValueError("api_base_url, api_key, and model are required when responder is not provided.")
        client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)

    test_files = [path for path in sorted(test_root.rglob("*.json")) if path.name != "splits.json"]
    file_progress = make_progress(len(test_files), desc="OrigChain+MemRL files")
    for test_file in test_files:
        rel_path = test_file.relative_to(test_root)
        gold_file = gold_root / rel_path if gold_root is not None else None
        events = _load_json(test_file)
        gold_events = _load_json(gold_file) if gold_file is not None and gold_file.exists() else events
        predictions: list[dict[str, Any]] = []
        baseline_predictions: list[dict[str, Any]] = []
        event_progress = make_progress(len(events), desc=f"Events {rel_path.name}")

        for idx, event in enumerate(events):
            history_start = max(0, idx - max(1, int(history_window)) + 1)
            observations = []
            for row in events[history_start : idx + 1]:
                raw = row.get("observation", {}) if isinstance(row.get("observation"), dict) else {}
                observations.append({"time": raw.get("time"), "event": raw.get("event", "")})
            normalized_obs = PIPELINE.normalize_observations(observations)

            phase_a_messages = _build_messages(
                PHASE_A_SYSTEM,
                {
                    "task": "signal_prediction",
                    "observations": PIPELINE.observations_to_text(normalized_obs),
                },
            )
            if responder is not None:
                phase_a_raw = await responder(phase_a_messages)
            else:
                phase_a_raw = await _call_openai_compatible(
                    client=client,
                    model=str(model),
                    messages=phase_a_messages,
                    temperature=0.0,
                    max_retries=max_retries,
                )
            predicted_signals = phase_a_raw.get("signals", phase_a_raw.get("Signals", {}))
            signals = PIPELINE.infer_signals(
                normalized_obs,
                predicted_signals if isinstance(predicted_signals, dict) else {},
            )
            async def execute_chain(*, memory_enabled: bool, memory_mode: str) -> dict[str, Any]:
                gate_prior = runtime.retrieve_for_gate(normalized_obs[-16:], signals) if memory_enabled else _empty_gate_prior()
                interruption_gate = PIPELINE.strict_interruption_gate(
                    observations=normalized_obs,
                    signals=signals,
                    gate_prior=gate_prior if memory_enabled else None,
                )
                generation_prior = (
                    runtime.retrieve_for_generation(normalized_obs[-16:], signals)
                    if memory_enabled
                    else _empty_generation_prior()
                )
                raw_phase_g: dict[str, Any] = {}
                raw_phase_b: dict[str, Any] = {}
                raw_phase_d: dict[str, Any] = {}
                raw_phase_c: dict[str, Any] = {}
                simulation = _normalize_simulation_trace({})
                candidate = PIPELINE.normalize_candidate({})
                final_decision = _normalize_decision_trace(
                    {
                        "should_intervene": False,
                        "level": 0,
                        "risk": "high" if interruption_gate.get("strong_flow") else "low",
                        "reason": str(interruption_gate.get("reason", "gate_blocked")),
                    }
                )
                simulation_prior = _empty_simulation_prior()
                decision_prior = _empty_decision_prior()

                if interruption_gate.get("allow_interruption", False):
                    phase_g_messages = _build_messages(
                        MEMRL_PHASE_G_SYSTEM,
                        {
                            "task": "candidate_generation",
                            "memory_mode": memory_mode,
                            "observations": PIPELINE.observations_to_text(normalized_obs),
                            "current_signals": signals,
                            "interruption_gate": interruption_gate,
                            "memory_generation_prior": _memory_generation_payload(generation_prior),
                            "memory_generation_context": generation_prior.get("generation_context", "{}"),
                        },
                    )
                    if responder is not None:
                        raw_phase_g = await responder(phase_g_messages)
                    else:
                        raw_phase_g = await _call_openai_compatible(
                            client=client,
                            model=str(model),
                            messages=phase_g_messages,
                            temperature=temperature,
                            max_retries=max_retries,
                        )
                    candidate = PIPELINE.normalize_candidate(raw_phase_g.get("candidate", raw_phase_g))

                    if any(candidate.values()):
                        simulation_prior = (
                            runtime.retrieve_for_simulation(normalized_obs[-16:], candidate, signals)
                            if memory_enabled
                            else _empty_simulation_prior()
                        )
                        phase_b_messages = _build_messages(
                            MEMRL_PHASE_B_SYSTEM,
                            {
                                "task": "mental_simulation",
                                "memory_mode": memory_mode,
                                "observations": PIPELINE.observations_to_text(normalized_obs),
                                "current_signals": signals,
                                "interruption_gate": interruption_gate,
                                "candidate_intervention": candidate,
                                "memory_generation_prior": _memory_generation_payload(generation_prior),
                                "memory_simulation_prior": _memory_simulation_payload(simulation_prior),
                                "memory_simulation_context": simulation_prior.get("simulation_context", "{}"),
                            },
                        )
                        if responder is not None:
                            raw_phase_b = await responder(phase_b_messages)
                        else:
                            raw_phase_b = await _call_openai_compatible(
                                client=client,
                                model=str(model),
                                messages=phase_b_messages,
                                temperature=temperature,
                                max_retries=max_retries,
                            )
                        simulation = _normalize_simulation_trace(
                            raw_phase_b.get("simulated_reaction", raw_phase_b.get("simulation", raw_phase_b))
                        )
                        decision_prior = (
                            runtime.retrieve_for_decision(normalized_obs[-16:], candidate, simulation, signals)
                            if memory_enabled
                            else _empty_decision_prior()
                        )
                        phase_d_messages = _build_messages(
                            MEMRL_DECISION_SYSTEM,
                            {
                                "task": "integrated_decision",
                                "memory_mode": memory_mode,
                                "observations": PIPELINE.observations_to_text(normalized_obs),
                                "current_signals": signals,
                                "interruption_gate": interruption_gate,
                                "candidate_intervention": candidate,
                                "simulated_reaction": simulation,
                                "memory_generation_prior": _memory_generation_payload(generation_prior),
                                "memory_decision_prior": _memory_decision_payload(decision_prior),
                                "memory_decision_context": decision_prior.get("decision_context", "{}"),
                            },
                        )
                        if responder is not None:
                            raw_phase_d = await responder(phase_d_messages)
                        else:
                            raw_phase_d = await _call_openai_compatible(
                                client=client,
                                model=str(model),
                                messages=phase_d_messages,
                                temperature=temperature,
                                max_retries=max_retries,
                            )
                        final_decision = _decision_from_payload(raw_phase_d, candidate=candidate)

                        if final_decision["should_intervene"] and final_decision["commitment_level"] > 0:
                            phase_c_messages = _build_messages(
                                PHASE_C_SYSTEM,
                                {
                                    "task": "final_generation",
                                    "memory_mode": memory_mode,
                                    "observations": PIPELINE.observations_to_text(normalized_obs),
                                    "current_signals": signals,
                                    "candidate_intervention": candidate,
                                    "simulated_reaction": simulation,
                                    "fixed_decision": final_decision,
                                },
                            )
                            if responder is not None:
                                raw_phase_c = await responder(phase_c_messages)
                            else:
                                raw_phase_c = await _call_openai_compatible(
                                    client=client,
                                    model=str(model),
                                    messages=phase_c_messages,
                                    temperature=temperature,
                                    max_retries=max_retries,
                                )

                final_output = PIPELINE.build_final_output(decision=final_decision, candidate=candidate, payload=raw_phase_c)
                return {
                    "gate_prior": gate_prior,
                    "interruption_gate": interruption_gate,
                    "generation_prior": generation_prior,
                    "simulation_prior": simulation_prior,
                    "decision_prior": decision_prior,
                    "candidate": candidate,
                    "simulation": simulation,
                    "decision": final_decision,
                    "final_output": final_output,
                    "raw_phase_g": raw_phase_g,
                    "raw_phase_b": raw_phase_b,
                    "raw_phase_d": raw_phase_d,
                    "raw_phase_c": raw_phase_c,
                }

            memory_path = await execute_chain(memory_enabled=True, memory_mode="memory_augmented")
            if run_baseline:
                base_path = await execute_chain(memory_enabled=False, memory_mode="no_memory_baseline")
            else:
                base_path = {
                    "gate_prior": _empty_gate_prior(),
                    "interruption_gate": PIPELINE.strict_interruption_gate(
                        observations=normalized_obs,
                        signals=signals,
                        gate_prior=None,
                    ),
                    "generation_prior": _empty_generation_prior(),
                    "simulation_prior": _empty_simulation_prior(),
                    "decision_prior": _empty_decision_prior(),
                    "candidate": PIPELINE.normalize_candidate({}),
                    "simulation": _normalize_simulation_trace({}),
                    "decision": _normalize_decision_trace(
                        {
                            "should_intervene": False,
                            "level": 0,
                            "risk": "medium",
                            "reason": "baseline_not_run",
                        }
                    ),
                    "final_output": PIPELINE.build_final_output(
                        decision={"should_intervene": False, "commitment_level": 0, "risk": "medium", "reason": "baseline_not_run"},
                        candidate=PIPELINE.normalize_candidate({}),
                        payload={},
                    ),
                    "raw_phase_g": {},
                    "raw_phase_b": {},
                    "raw_phase_d": {},
                    "raw_phase_c": {},
                }
            base_generation_prior = base_path["generation_prior"]
            memory_generation_prior = memory_path["generation_prior"]
            base_candidate = base_path["candidate"]
            memory_candidate_proposal = memory_path["candidate"]
            base_simulation = base_path["simulation"]
            memory_simulation_proposal = memory_path["simulation"]
            base_decision = base_path["decision"]
            memory_proposed_decision = memory_path["decision"]
            base_simulation_prior = base_path["simulation_prior"]
            memory_simulation_prior = memory_path["simulation_prior"]
            base_decision_prior = base_path["decision_prior"]
            memory_decision_prior = memory_path["decision_prior"]
            base_output = base_path["final_output"]
            memory_proposed_output = memory_path["final_output"]

            def build_prediction_record(
                *,
                effective_path: dict[str, Any],
                decision_mode: str,
                memory_gate: dict[str, Any],
                memory_episode_id_suffix: str,
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                gate_prior = effective_path["gate_prior"]
                interruption_gate = effective_path["interruption_gate"]
                generation_prior = effective_path["generation_prior"]
                simulation_prior = effective_path["simulation_prior"]
                decision_prior = effective_path["decision_prior"]
                candidate = effective_path["candidate"]
                simulation = effective_path["simulation"]
                final_decision = effective_path["decision"]
                final_output = effective_path["final_output"]
                raw_phase_g = effective_path["raw_phase_g"]
                raw_phase_b = effective_path["raw_phase_b"]
                raw_phase_d = effective_path["raw_phase_d"]
                raw_phase_c = effective_path["raw_phase_c"]
                memory_candidate = {
                    "purpose": final_output.get("Purpose") or candidate.get("purpose"),
                    "proactive_task": final_output.get("Proactive Task"),
                    "response": final_output.get("Response"),
                    "operation": final_output.get("Operation"),
                }
                memory_episode = enrich_memory_schema(
                    {
                        "memory_id": f"offline-orig-chain-{rel_path.stem}-{idx}-{memory_episode_id_suffix}",
                        "sample_id": f"{rel_path.stem}-{idx}-{memory_episode_id_suffix}",
                        "source": "offline_eval",
                        "domain": PIPELINE.infer_domain(normalized_obs),
                        "observations": normalized_obs[-16:],
                        "intent_text": " | ".join(str(item.get("event", "")) for item in normalized_obs[-8:]),
                        "candidate": memory_candidate,
                        "simulation": simulation,
                        "decision": {
                            "should_intervene": bool(final_decision.get("should_intervene", False)),
                            "commitment_level": int(final_decision.get("commitment_level", 0) or 0),
                            "risk": final_decision.get("risk", "medium"),
                            "reason": final_decision.get("reason", ""),
                        },
                        "labels": {
                            "gold_should": None,
                            "gold_level": None,
                            "q_need": _safe_float(signals.get("need", signals.get("p_need", 0.0))),
                            "q_accept": _safe_float(signals.get("accept", signals.get("p_accept", 0.0))),
                        },
                        "gate_features": {
                            "need": _safe_float(interruption_gate.get("calibrated_gate_inputs", {}).get("need", signals.get("need", 0.0))),
                            "flow": _safe_float(interruption_gate.get("calibrated_gate_inputs", {}).get("flow", signals.get("flow", 0.0))),
                            "risk": _safe_float(interruption_gate.get("calibrated_gate_inputs", {}).get("risk", signals.get("risk", 0.0))),
                            "evidence": _safe_float(interruption_gate.get("calibrated_gate_inputs", {}).get("evidence", 0.0)),
                            "allow_interruption": bool(interruption_gate.get("allow_interruption", False)),
                            "gate_reason": str(interruption_gate.get("reason", "")),
                            "gate_thresholds": dict(interruption_gate.get("gate_thresholds", interruption_gate.get("thresholds", {})) or {}),
                            "memory_signal_delta": dict(interruption_gate.get("memory_signal_delta", {}) or {}),
                            "memory_threshold_delta": dict(interruption_gate.get("memory_threshold_delta", {}) or {}),
                            "memory_prior_confidence": _safe_float(gate_prior.get("confidence", 0.0)),
                        },
                        "reward": 0.0,
                        "q_value": 0.0,
                        "q_visits": 0,
                        "created_at": "",
                        "updated_at": "",
                    },
                    signals=signals,
                )
                return {
                    "observation": event.get("observation"),
                    "agent_response": [] if final_output.get("Proactive Task") is None else [final_output.get("Proactive Task")],
                    "task_status": bool(final_decision.get("should_intervene", False)),
                    "other_infomation": {
                        "Purpose": final_output.get("Purpose"),
                        "Thoughts": final_output.get("Thoughts"),
                        "Response": final_output.get("Response"),
                        "Operation": final_output.get("Operation"),
                        "PredictedSignals": predicted_signals if isinstance(predicted_signals, dict) else {},
                        "Signals": signals,
                        "MemoryTrace": _memory_trace_payload(
                            gate_prior=gate_prior,
                            generation_prior=generation_prior,
                            simulation_prior=simulation_prior,
                            decision_prior=decision_prior,
                        ),
                        "GatePrior": {
                            "current_context_family": gate_prior.get("current_context_family", "general"),
                            "similar_case_count": gate_prior.get("similar_case_count", 0),
                            "historical_intervene_value": gate_prior.get("historical_intervene_value", 0.0),
                            "historical_abstain_value": gate_prior.get("historical_abstain_value", 0.0),
                            "historical_reject_risk": gate_prior.get("historical_reject_risk", 0.0),
                            "missed_help_risk": gate_prior.get("missed_help_risk", 0.0),
                            "confidence": gate_prior.get("confidence", 0.0),
                            "margin": gate_prior.get("margin", 0.0),
                            "recommended_signal_delta": gate_prior.get("recommended_signal_delta", {}),
                            "recommended_threshold_delta": gate_prior.get("recommended_threshold_delta", {}),
                            "gate_context": gate_prior.get("gate_context", "{}"),
                            "used_memory_ids": gate_prior.get("used_memory_ids", []),
                        },
                        "GateCalibration": {
                            "gate_inputs": interruption_gate.get("gate_inputs", {}),
                            "calibrated_gate_inputs": interruption_gate.get("calibrated_gate_inputs", {}),
                            "gate_thresholds": interruption_gate.get("gate_thresholds", interruption_gate.get("thresholds", {})),
                            "memory_signal_delta": interruption_gate.get("memory_signal_delta", {}),
                            "memory_threshold_delta": interruption_gate.get("memory_threshold_delta", {}),
                        },
                        "InterruptionGate": interruption_gate,
                        "BaseCandidate": base_candidate,
                        "MemoryProposedCandidate": memory_candidate_proposal,
                        "Candidate": candidate,
                        "BaseSimulation": base_simulation,
                        "MemoryProposedSimulation": memory_simulation_proposal,
                        "Simulation": simulation,
                        "BaseDecision": base_decision,
                        "MemoryProposedDecision": memory_proposed_decision,
                        "Decision": final_decision,
                        "BaseGenerationPrior": _memory_generation_payload(base_generation_prior),
                        "MemoryProposedGenerationPrior": _memory_generation_payload(memory_generation_prior),
                        "GenerationPrior": _memory_generation_payload(generation_prior),
                        "BaseSimulationPrior": _memory_simulation_payload(base_simulation_prior),
                        "MemoryProposedSimulationPrior": _memory_simulation_payload(memory_simulation_prior),
                        "SimulationPrior": _memory_simulation_payload(simulation_prior),
                        "BaseDecisionPrior": _memory_decision_payload(base_decision_prior),
                        "MemoryProposedDecisionPrior": _memory_decision_payload(memory_decision_prior),
                        "DecisionPrior": _memory_decision_payload(decision_prior),
                        "DecisionMode": decision_mode,
                        "MemorySchema": {
                            "context_family": memory_episode.get("context_family", "general"),
                            "action_family": memory_episode.get("action_family", "no_intervention"),
                            "outcome_family": memory_episode.get("outcome_family", "uncertain"),
                        },
                        "MemoryGate": memory_gate,
                        "memory_changed_candidate": base_candidate != candidate,
                        "memory_changed_decision": base_decision != final_decision,
                        "raw_phase_a_output": phase_a_raw,
                        "raw_base_phase_g_output": base_path["raw_phase_g"],
                        "raw_memory_phase_g_output": memory_path["raw_phase_g"],
                        "raw_phase_g_output": raw_phase_g,
                        "raw_base_phase_b_output": base_path["raw_phase_b"],
                        "raw_memory_phase_b_output": memory_path["raw_phase_b"],
                        "raw_phase_b_output": raw_phase_b,
                        "raw_base_phase_d_output": base_path["raw_phase_d"],
                        "raw_memory_phase_d_output": memory_path["raw_phase_d"],
                        "raw_phase_d_output": raw_phase_d,
                        "raw_base_phase_c_output": base_path["raw_phase_c"],
                        "raw_memory_phase_c_output": memory_path["raw_phase_c"],
                        "raw_phase_c_output": raw_phase_c,
                        "BaseOutput": {
                            "Purpose": base_output.get("Purpose"),
                            "Thoughts": base_output.get("Thoughts"),
                            "Response": base_output.get("Response"),
                            "Operation": base_output.get("Operation"),
                            "Proactive Task": base_output.get("Proactive Task"),
                        },
                        "MemoryProposedOutput": {
                            "Purpose": memory_proposed_output.get("Purpose"),
                            "Thoughts": memory_proposed_output.get("Thoughts"),
                            "Response": memory_proposed_output.get("Response"),
                            "Operation": memory_proposed_output.get("Operation"),
                            "Proactive Task": memory_proposed_output.get("Proactive Task"),
                        },
                    },
                }, memory_episode

            memory_gate = {
                "applied": True,
                "kept_memory_result": True,
                "reason": "separate_memory_augmented_run" if run_baseline else "baseline_skipped_memory_only",
                "confidence": _safe_float(
                    (memory_path.get("decision_prior", {}) or {})
                    .get("memory_recommendation", {})
                    .get("confidence", 0.0)
                ),
            }
            memory_record, memory_episode = build_prediction_record(
                effective_path=memory_path,
                decision_mode="memory_augmented_separate" if run_baseline else "memory_only_no_baseline",
                memory_gate=memory_gate,
                memory_episode_id_suffix="memory",
            )
            predictions.append(memory_record)

            if run_baseline:
                baseline_gate = {
                    "applied": False,
                    "kept_memory_result": False,
                    "reason": "separate_no_memory_baseline_run",
                    "confidence": 0.0,
                }
                baseline_record, _ = build_prediction_record(
                    effective_path=base_path,
                    decision_mode="no_memory_baseline_separate",
                    memory_gate=baseline_gate,
                    memory_episode_id_suffix="baseline",
                )
                baseline_predictions.append(baseline_record)

            final_decision = memory_path["decision"]
            gate_prior = memory_path["gate_prior"]
            generation_prior = memory_path["generation_prior"]
            simulation_prior = memory_path["simulation_prior"]
            decision_prior = memory_path["decision_prior"]

            gold_event = gold_events[idx] if idx < len(gold_events) else {}
            gold_decision = gold_event.get("gold_decision", {})
            if allow_gold_update and isinstance(gold_decision, dict) and gold_decision:
                reward = _reward_from_gold(final_decision, gold_decision)
                used_memory_ids: list[str] = []
                for bucket in (
                    gate_prior.get("used_memory_ids", []),
                    generation_prior.get("used_memory_ids", []),
                    simulation_prior.get("used_memory_ids", []),
                    decision_prior.get("used_memory_ids", []),
                ):
                    for memory_id in bucket:
                        if memory_id not in used_memory_ids:
                            used_memory_ids.append(memory_id)
                runtime.record_outcome(
                    used_memory_ids,
                    reward,
                    enrich_memory_schema(
                        {
                            **memory_episode,
                            "labels": {
                                **memory_episode.get("labels", {}),
                                "gold_should": gold_decision.get("should_intervene"),
                                "gold_level": gold_decision.get("commitment_level"),
                            },
                            "reward": reward,
                            "q_value": reward,
                        },
                        signals=signals,
                    ),
                )
            event_progress.update(1)

        target = output_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        if run_baseline and baseline_output_root is not None:
            baseline_target = baseline_output_root / rel_path
            baseline_target.parent.mkdir(parents=True, exist_ok=True)
            baseline_target.write_text(json.dumps(baseline_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        event_progress.close()
        file_progress.update(1)

    file_progress.close()
    if save_updated_memory:
        runtime.save(str(snapshot_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Original proactive chain + external MemRL retrieval.")
    parser.add_argument("--bootstrap", type=Path, required=True, help="Bootstrap JSONL or snapshot directory.")
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-output-root",
        type=Path,
        default=None,
        help="Where to save no-memory baseline predictions when --run-baseline is set. Defaults to <output-root>_baseline.",
    )
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, default=None)
    parser.add_argument("--api-base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--history-window", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--allow-gold-update",
        action="store_true",
        help="Allow gold labels to update memory during evaluation. Disabled by default to avoid test leakage.",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Also run the no-memory baseline chain. Disabled by default for faster memory-only evaluation.",
    )
    parser.add_argument(
        "--no-save-updated-memory",
        action="store_true",
        help="Do not write the final runtime memory snapshot to --snapshot-root.",
    )
    args = parser.parse_args()
    asyncio.run(
        run_original_chain_eval(
            bootstrap_path=args.bootstrap,
            test_root=args.test_root,
            output_root=args.output_root,
            baseline_output_root=args.baseline_output_root,
            snapshot_root=args.snapshot_root,
            gold_root=args.gold_root,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            model=args.model,
            history_window=args.history_window,
            temperature=args.temperature,
            max_retries=args.max_retries,
            allow_gold_update=args.allow_gold_update,
            run_baseline=args.run_baseline,
            save_updated_memory=not args.no_save_updated_memory,
        )
    )


if __name__ == "__main__":
    main()
