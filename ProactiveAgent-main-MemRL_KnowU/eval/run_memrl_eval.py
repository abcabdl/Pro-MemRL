from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from agent.memrl import ProactiveMemRLRuntime, fuse_decision

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


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
    candidate_pool = event.get("agent_response", {}).get("candidate_task", []) if isinstance(event.get("agent_response"), dict) else []
    need = min(1.0, 0.2 + 0.35 * stuck_hits + (0.25 if candidate_pool else 0.0))
    accept = 0.45 + (0.15 if candidate_pool else 0.0)
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


def _candidate_from_event(event: dict[str, Any], generation_prior: dict[str, Any]) -> dict[str, Any]:
    candidate_pool = []
    if isinstance(event.get("agent_response"), dict):
        candidate_pool = list(event["agent_response"].get("candidate_task", []) or [])
    candidate_text = candidate_pool[0] if candidate_pool else None
    if candidate_text is None:
        positives = generation_prior.get("candidate_positive_examples", [])
        if positives:
            candidate_text = positives[0].get("candidate", {}).get("proactive_task")
    response = f"I can help with: {candidate_text}" if candidate_text else None
    return {
        "purpose": str(event.get("observation", {}).get("event", ""))[:200],
        "proactive_task": candidate_text,
        "response": response,
        "operation": None if candidate_text is None else "notify_only",
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


def run_eval(
    *,
    bootstrap_path: Path,
    test_root: Path,
    gold_root: Path,
    output_root: Path,
    snapshot_root: Path,
) -> None:
    runtime = ProactiveMemRLRuntime()
    runtime.warm_start(str(bootstrap_path))

    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    splits_path = gold_root / "splits.json"
    if splits_path.exists():
        shutil.copy2(splits_path, output_root / "splits.json")

    test_files = [path for path in sorted(test_root.rglob("*.json")) if path.name != "splits.json"]
    file_progress = make_progress(len(test_files), desc="MemRL files")
    for test_file in test_files:
        if test_file.name == "splits.json":
            continue
        rel_path = test_file.relative_to(test_root)
        gold_file = gold_root / rel_path
        events = _load_json(test_file)
        gold_events = _load_json(gold_file) if gold_file.exists() else events
        history: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        event_progress = make_progress(len(events), desc=f"Events {rel_path.name}")

        for idx, event in enumerate(events):
            obs_record = _observation_record(event)
            history.append(obs_record)
            signals = _infer_signals(history, event)
            generation_prior = runtime.retrieve_for_generation(history[-16:], signals)
            candidate = _candidate_from_event(event, generation_prior)
            simulation_prior = runtime.retrieve_for_simulation(history[-16:], candidate, signals)
            simulation = _simulation_from_prior(simulation_prior)
            decision_prior = runtime.retrieve_for_decision(history[-16:], candidate, simulation, signals)
            fused = fuse_decision(
                signal_score=float(signals.get("p_need", 0.0)) - float(signals.get("r_risk", 0.0)),
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
                    "candidate_task": [candidate["proactive_task"]] if decision["should_intervene"] and candidate.get("proactive_task") else [],
                },
                "task_status": bool(decision["should_intervene"]),
                "other_infomation": {
                    "Decision": decision,
                    "Simulation": simulation,
                    "GenerationPrior": {
                        "preferred_level": generation_prior.get("preferred_level", 0),
                    },
                    "DecisionPrior": {
                        "intervene_memory_value": decision_prior.get("intervene_memory_value", 0.0),
                        "abstain_memory_value": decision_prior.get("abstain_memory_value", 0.0),
                    },
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
                    "domain": "general",
                    "observations": history[-16:],
                    "intent_text": str(event.get("observation", {}).get("event", "")),
                    "candidate": candidate,
                    "simulation": simulation,
                    "decision": decision,
                    "labels": {
                        "gold_should": gold_decision.get("should_intervene"),
                        "gold_level": gold_decision.get("commitment_level"),
                    },
                    "reward": reward,
                    "q_value": reward,
                    "q_visits": 0,
                    "created_at": "",
                    "updated_at": "",
                }
                used_memory_ids = []
                for bucket in (
                    generation_prior.get("used_memory_ids", []),
                    simulation_prior.get("used_memory_ids", []),
                    decision_prior.get("used_memory_ids", []),
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
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    run_eval(
        bootstrap_path=args.bootstrap,
        test_root=args.test_root,
        gold_root=args.gold_root,
        output_root=args.output_root,
        snapshot_root=args.snapshot_root,
    )


if __name__ == "__main__":
    main()
