from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from agent.memrl import ProactiveMemRLRuntime, fuse_decision
from proactive_pipeline import strict_interruption_gate

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SYSTEM_PROMPT = """<Role>
You are a helpful assistant that provides proactive suggestions to the user.
</Role>

<Task>
Understand what the user is doing and anticipate whether a proactive suggestion would help right now.
Use the observation history and memory priors. Prefer no interruption when the user is progressing normally.
</Task>

<Format>
Return strict JSON:
{
  "Purpose": "short text",
  "Thoughts": "short reasoning",
  "Proactive Task": "task text or null",
  "Response": "what you would say to the user or null"
}
</Format>

<Rules>
- Use only one proactive task.
- If interruption is not clearly worthwhile, set both `Proactive Task` and `Response` to null.
- Keep suggestions concrete and timely.
- Prefer lower interruption.
</Rules>"""


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


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    obj_match = re.search(r"(\{[\s\S]*\})", text)
    if obj_match:
        return obj_match.group(1)
    return text


def _normalize_nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in {"", "null", "none", "nil", "n/a", "na"}:
            return None
        return normalized
    return str(value)


def _parse_model_payload(raw_text: str) -> dict[str, Any]:
    payload = json.loads(_extract_json_text(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")
    proactive_task = payload.get("Proactive Task")
    if proactive_task is None:
        proactive_task = payload.get("Proactive_Task", payload.get("proactive_task"))
    response = payload.get("Response")
    if response is None:
        response = payload.get("response")
    purpose = payload.get("Purpose")
    if purpose is None:
        purpose = payload.get("purpose")
    thoughts = payload.get("Thoughts")
    if thoughts is None:
        thoughts = payload.get("thoughts")
    return {
        "Purpose": _normalize_nullable_text(purpose),
        "Thoughts": _normalize_nullable_text(thoughts),
        "Proactive Task": _normalize_nullable_text(proactive_task),
        "Response": _normalize_nullable_text(response),
        "raw": raw_text,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _observation_record(event: dict[str, Any]) -> dict[str, Any]:
    observation = event.get("observation", {}) if isinstance(event.get("observation"), dict) else {}
    return {
        "time": observation.get("time"),
        "event": observation.get("event"),
        "source": "offline_eval",
    }


def _infer_signals(history: list[dict[str, Any]], event: dict[str, Any]) -> dict[str, float]:
    text = " ".join(str(item.get("event", "")).lower() for item in history[-16:])
    stuck_hits = sum(token in text for token in ("error", "traceback", "fail", "stuck", "debug", "fix"))
    flow_hits = sum(token in text for token in ("edit", "working", "draft", "report", "code", "research"))
    need = min(1.0, 0.2 + 0.35 * stuck_hits)
    accept = 0.45
    risk = min(1.0, 0.2 + 0.1 * max(flow_hits - stuck_hits, 0))
    flow = min(1.0, 0.25 + 0.12 * flow_hits)
    return {
        "f_flow": flow,
        "d_stuck": min(1.0, 0.3 * stuck_hits),
        "epsilon_agent": 0.25,
        "r_risk": risk,
        "delta_rej": 0.0,
        "p_need": need,
        "p_accept": min(1.0, accept),
    }


def _infer_domain(history: list[dict[str, Any]]) -> str:
    text = " ".join(str(item.get("event", "")).lower() for item in history[-16:])
    coding_hits = sum(token in text for token in ("code", "vscode", "traceback", "debug", ".py", ".js", "terminal"))
    writing_hits = sum(token in text for token in ("draft", "document", "report", "outline", "essay", "paper"))
    finance_hits = sum(token in text for token in ("investment", "spreadsheet", "balance", "stock", "market", "account"))
    if coding_hits >= max(writing_hits, finance_hits) and coding_hits > 0:
        return "coding"
    if writing_hits >= max(coding_hits, finance_hits) and writing_hits > 0:
        return "writing"
    if finance_hits > 0:
        return "finance"
    return "general"


def _candidate_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "purpose": payload.get("Purpose"),
        "proactive_task": payload.get("Proactive Task"),
        "response": payload.get("Response"),
        "operation": None if payload.get("Proactive Task") is None else "notify_only",
    }


def _backbone_level(candidate: dict[str, Any]) -> int:
    if not candidate.get("proactive_task"):
        return 0
    return 2 if candidate.get("response") else 1


def _simulation_from_prior(prior: dict[str, Any]) -> dict[str, Any]:
    accept = float(prior.get("historical_accept_rate", 0.0))
    dismiss = float(prior.get("historical_dismiss_rate", 0.0))
    annoy = float(prior.get("historical_annoy_rate", 0.0))
    if accept >= max(dismiss, annoy) and accept >= 0.5:
        label = "accept"
    elif annoy >= max(accept, dismiss) and annoy >= 0.2:
        label = "annoyed"
    elif dismiss >= 0.25:
        label = "dismiss"
    else:
        label = "ignore"
    return {
        "acceptance": label,
        "acceptance_confidence": accept,
        "historical_accept_rate": accept,
        "historical_dismiss_rate": dismiss,
        "historical_annoy_rate": annoy,
        "historical_reject_risk": min(1.0, dismiss + annoy),
        "total_score": accept - dismiss - annoy,
    }


def _reward_from_gold(pred_decision: dict[str, Any], gold_decision: dict[str, Any]) -> float:
    pred_should = int(bool(pred_decision.get("should_intervene", False)))
    pred_level = int(pred_decision.get("commitment_level", 0) or 0)
    gold_should = int(gold_decision.get("should_intervene", 0) or 0)
    gold_level = int(gold_decision.get("commitment_level", 0) or 0)
    if pred_should == gold_should and pred_level == gold_level:
        return 1.0
    if pred_should == gold_should:
        return 0.4
    if pred_should == 1 and gold_should == 0:
        return -1.0
    return -0.8


def _runtime_seed_path(path: Path) -> Path:
    if path.is_dir():
        snapshot = path / "memrl_snapshot.jsonl"
        if snapshot.exists():
            return snapshot
        raise FileNotFoundError(f"Could not find memrl_snapshot.jsonl under {path}")
    return path


def _build_messages(
    *,
    history: list[dict[str, Any]],
    signals: dict[str, float],
    generation_prior: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "Instructions": "Analyze the recent observation history and decide whether a proactive task is worth proposing now.",
        "Observations": history[-16:],
        "Signals": signals,
        "MemoryGenerationPrior": {
            "preferred_level": generation_prior.get("preferred_level", 0),
            "positive_patterns": generation_prior.get("positive_patterns", []),
            "negative_patterns": generation_prior.get("negative_patterns", []),
            "avoid_patterns": generation_prior.get("avoid_patterns", []),
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
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
    for _ in range(max(1, max_retries)):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content or "{}"
            return _parse_model_payload(content)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            await asyncio.sleep(0.5)
    raise RuntimeError(f"API call failed after {max_retries} attempts: {last_exc}")


async def run_api_eval(
    *,
    bootstrap_path: Path,
    test_root: Path,
    output_root: Path,
    snapshot_root: Path,
    gold_root: Path | None = None,
    api_base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    responder: Callable[[list[dict[str, str]]], Awaitable[dict[str, Any]]] | None = None,
) -> None:
    runtime = ProactiveMemRLRuntime()
    runtime.warm_start(str(_runtime_seed_path(bootstrap_path)))

    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    if gold_root is not None:
        splits_path = gold_root / "splits.json"
        if splits_path.exists():
            shutil.copy2(splits_path, output_root / "splits.json")
    else:
        splits_path = test_root / "splits.json"
        if splits_path.exists():
            shutil.copy2(splits_path, output_root / "splits.json")

    client: AsyncOpenAI | None = None
    if responder is None:
        if not api_base_url or not api_key or not model:
            raise ValueError("api_base_url, api_key, and model are required when responder is not provided.")
        client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)

    test_files = [path for path in sorted(test_root.rglob("*.json")) if path.name != "splits.json"]
    file_progress = make_progress(len(test_files), desc="MemRL API files")

    for test_file in test_files:
        rel_path = test_file.relative_to(test_root)
        gold_file = gold_root / rel_path if gold_root is not None else None
        events = _load_json(test_file)
        gold_events = _load_json(gold_file) if gold_file is not None and gold_file.exists() else events
        history: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        event_progress = make_progress(len(events), desc=f"Events {rel_path.name}")

        for idx, event in enumerate(events):
            obs_record = _observation_record(event)
            history.append(obs_record)
            signals = _infer_signals(history, event)
            gate_prior = runtime.retrieve_for_gate(history[-16:], signals)
            interruption_gate = strict_interruption_gate(
                observations=history[-16:],
                signals=signals,
                gate_prior=gate_prior,
            )
            calibrated_signals = dict(signals)
            calibrated_inputs = interruption_gate.get("calibrated_gate_inputs", {})
            if isinstance(calibrated_inputs, dict):
                calibrated_signals.update(
                    {
                        "p_need": calibrated_inputs.get("need", signals.get("p_need", 0.0)),
                        "f_flow": calibrated_inputs.get("flow", signals.get("f_flow", 0.0)),
                        "r_risk": calibrated_inputs.get("risk", signals.get("r_risk", 0.0)),
                    }
                )
            generation_prior = runtime.retrieve_for_generation(history[-16:], signals)
            messages = _build_messages(history=history, signals=signals, generation_prior=generation_prior)

            if responder is not None:
                model_payload = await responder(messages)
            else:
                model_payload = await _call_openai_compatible(
                    client=client,
                    model=str(model),
                    messages=messages,
                    temperature=temperature,
                    max_retries=max_retries,
                )

            candidate = _candidate_from_payload(model_payload)
            simulation_prior = runtime.retrieve_for_simulation(history[-16:], candidate, signals)
            simulation = _simulation_from_prior(simulation_prior)
            decision_prior = runtime.retrieve_for_decision(history[-16:], candidate, simulation, signals)
            fused = fuse_decision(
                signal_score=float(calibrated_signals.get("p_need", 0.0)) - float(calibrated_signals.get("r_risk", 0.0)),
                backbone_level=_backbone_level(candidate),
                generation_prior=generation_prior,
                simulation_result=simulation,
                decision_prior=decision_prior,
            )
            decision = {
                "should_intervene": bool(fused["should_intervene"]),
                "commitment_level": int(fused["level"]),
                "reason": fused["reason"],
            }

            pred_event = {
                "observation": event.get("observation"),
                "agent_response": {
                    "candidate_task": [candidate["proactive_task"]]
                    if decision["should_intervene"] and candidate.get("proactive_task")
                    else [],
                },
                "task_status": bool(decision["should_intervene"]),
                "other_infomation": {
                    "Purpose": model_payload.get("Purpose"),
                    "Thoughts": model_payload.get("Thoughts"),
                    "Response": candidate.get("response"),
                    "Decision": decision,
                    "Simulation": simulation,
                    "GenerationPrior": {
                        "preferred_level": generation_prior.get("preferred_level", 0),
                        "positive_patterns": generation_prior.get("positive_patterns", []),
                        "negative_patterns": generation_prior.get("negative_patterns", []),
                        "avoid_patterns": generation_prior.get("avoid_patterns", []),
                    },
                    "DecisionPrior": {
                        "intervene_memory_value": decision_prior.get("intervene_memory_value", 0.0),
                        "abstain_memory_value": decision_prior.get("abstain_memory_value", 0.0),
                    },
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
                    "raw_model_output": model_payload.get("raw"),
                },
            }
            predictions.append(pred_event)

            gold_event = gold_events[idx] if idx < len(gold_events) else {}
            gold_decision = gold_event.get("gold_decision", {})
            if isinstance(gold_decision, dict) and gold_decision:
                reward = _reward_from_gold(decision, gold_decision)
                episode = {
                    "memory_id": f"offline-{rel_path.stem}-{idx}",
                    "sample_id": f"{rel_path.stem}-{idx}",
                    "source": "offline_eval",
                    "domain": _infer_domain(history),
                    "observations": history[-16:],
                    "intent_text": " | ".join(str(item.get("event", "")) for item in history[-8:]),
                    "candidate": candidate,
                    "simulation": simulation,
                    "decision": decision,
                    "labels": {
                        "gold_should": gold_decision.get("should_intervene"),
                        "gold_level": gold_decision.get("commitment_level"),
                        "q_need": _safe_float(signals.get("p_need")),
                        "q_accept": _safe_float(signals.get("p_accept")),
                    },
                    "reward": reward,
                    "q_value": reward,
                    "q_visits": 0,
                    "created_at": "",
                    "updated_at": "",
                }
                used_memory_ids: list[str] = []
                for bucket in (
                    generation_prior.get("used_memory_ids", []),
                    simulation_prior.get("used_memory_ids", []),
                    decision_prior.get("used_memory_ids", []),
                    gate_prior.get("used_memory_ids", []),
                ):
                    for memory_id in bucket:
                        if memory_id not in used_memory_ids:
                            used_memory_ids.append(memory_id)
                runtime.record_outcome(used_memory_ids, reward, episode)

            event_progress.update(1)

        target = output_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        event_progress.close()
        file_progress.update(1)

    file_progress.close()
    runtime.save(str(snapshot_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, required=True, help="Bootstrap JSONL or a snapshot directory.")
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, default=None)
    parser.add_argument("--api-base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(
        run_api_eval(
            bootstrap_path=args.bootstrap,
            test_root=args.test_root,
            output_root=args.output_root,
            snapshot_root=args.snapshot_root,
            gold_root=args.gold_root,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
    )


if __name__ == "__main__":
    main()
