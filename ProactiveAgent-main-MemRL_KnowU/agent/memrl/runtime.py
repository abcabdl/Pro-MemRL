from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .retriever import MemRLRetriever
from .updater import update_q_value


class ProactiveMemRLRuntime:
    def __init__(
        self,
        *,
        alpha: float = 0.12,
        topk: int = 8,
        sim_threshold: float = 0.18,
    ) -> None:
        self.alpha = alpha
        self.retriever = MemRLRetriever(topk=topk, sim_threshold=sim_threshold)
        self.memories: list[dict[str, Any]] = []
        self.memory_by_id: dict[str, dict[str, Any]] = {}

    def _rebuild(self) -> None:
        self.memory_by_id = {str(item["memory_id"]): item for item in self.memories}
        self.retriever.build(self.memories)

    def warm_start(self, episode_path: str) -> int:
        path = Path(episode_path)
        if not path.exists():
            raise FileNotFoundError(f"MemRL episode file not found: {path}")
        self.memories = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._rebuild()
        return len(self.memories)

    def retrieve_for_generation(self, observations: list[dict], signals: dict) -> dict:
        if not self.memories:
            return {
                "current_context_family": "general",
                "candidate_positive_examples": [],
                "candidate_negative_examples": [],
                "preferred_level": 0,
                "preferred_action_families": [],
                "disallowed_action_families": [],
                "positive_patterns": [],
                "negative_patterns": [],
                "avoid_patterns": [],
                "generation_context": "{}",
                "used_memory_ids": [],
            }
        return self.retriever.retrieve_for_generation(observations, signals)

    def retrieve_for_gate(self, observations: list[dict], signals: dict) -> dict:
        if not self.memories:
            return {
                "current_context_family": "general",
                "similar_case_count": 0,
                "historical_intervene_value": 0.0,
                "historical_abstain_value": 0.0,
                "historical_reject_risk": 0.0,
                "missed_help_risk": 0.0,
                "confidence": 0.0,
                "recommended_signal_delta": {
                    "need": 0.0,
                    "flow": 0.0,
                    "risk": 0.0,
                    "evidence": 0.0,
                },
                "recommended_threshold_delta": {
                    "need": 0.0,
                    "flow": 0.0,
                    "risk": 0.0,
                    "evidence": 0.0,
                },
                "support_cases": [],
                "risk_cases": [],
                "used_memory_ids": [],
                "gate_context": "{}",
            }
        return self.retriever.retrieve_for_gate(observations, signals)

    def retrieve_for_simulation(self, observations: list[dict], candidate: dict, signals: dict) -> dict:
        if not self.memories:
            return {
                "current_context_family": "general",
                "candidate_action_family": "no_intervention",
                "historical_accept_rate": 0.0,
                "historical_dismiss_rate": 0.0,
                "historical_annoy_rate": 0.0,
                "support_cases": [],
                "risk_cases": [],
                "historical_reject_risk": 0.0,
                "simulation_context": "{}",
                "used_memory_ids": [],
            }
        return self.retriever.retrieve_for_simulation(observations, candidate, signals)

    def retrieve_for_decision(self, observations: list[dict], candidate: dict, simulation: dict, signals: dict) -> dict:
        if not self.memories:
            return {
                "current_context_family": "general",
                "candidate_action_family": "no_intervention",
                "intervene_memories": [],
                "abstain_memories": [],
                "intervene_memory_value": 0.0,
                "abstain_memory_value": 0.0,
                "memory_level_mode": 0,
                "historical_reject_risk": 0.0,
                "memory_recommendation": {
                    "should_intervene": None,
                    "level": 0,
                    "confidence": 0.0,
                    "margin": 0.0,
                    "reason": "no_memory_available",
                    "context_family": "general",
                    "action_family": "no_intervention",
                },
                "decision_context": "{}",
                "used_memory_ids": [],
            }
        return self.retriever.retrieve_for_decision(observations, candidate, simulation, signals)

    def record_outcome(self, used_memory_ids: list[str], reward: float, episode: dict) -> None:
        appended = False
        for memory_id in used_memory_ids:
            memory = self.memory_by_id.get(str(memory_id))
            if memory is None:
                continue
            update_q_value(memory, reward, alpha=self.alpha)
        if episode:
            candidate = dict(episode)
            if "memory_id" not in candidate and "sample_id" in candidate:
                candidate["memory_id"] = str(candidate["sample_id"])
            if candidate.get("memory_id") and str(candidate["memory_id"]) not in self.memory_by_id:
                self.memories.append(candidate)
                self.memory_by_id[str(candidate["memory_id"])] = candidate
                appended = True
        if appended:
            self._rebuild()

    def save(self, out_dir: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        snapshot = path / "memrl_snapshot.jsonl"
        snapshot.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in self.memories),
            encoding="utf-8",
        )
        meta = path / "memrl_meta.json"
        meta.write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "topk": self.retriever.topk,
                    "sim_threshold": self.retriever.sim_threshold,
                    "count": len(self.memories),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, in_dir: str) -> None:
        snapshot = Path(in_dir) / "memrl_snapshot.jsonl"
        self.warm_start(str(snapshot))


def record_feedback_payload(
    feedback_payload: dict[str, Any],
    *,
    reaction: str,
    state_dir: str,
    bootstrap_path: str | None = None,
    alpha: float = 0.12,
    topk: int = 8,
    sim_threshold: float = 0.18,
) -> None:
    memrl_payload = feedback_payload.get("memrl", {}) if isinstance(feedback_payload, dict) else {}
    if not isinstance(memrl_payload, dict):
        return
    used_memory_ids = list(memrl_payload.get("used_memory_ids", []) or [])
    episode = memrl_payload.get("episode", {}) if isinstance(memrl_payload.get("episode"), dict) else {}
    runtime = ProactiveMemRLRuntime(alpha=alpha, topk=topk, sim_threshold=sim_threshold)
    snapshot = Path(state_dir) / "memrl_snapshot.jsonl"
    if snapshot.exists():
        runtime.load(state_dir)
    elif bootstrap_path:
        boot = Path(bootstrap_path)
        if boot.exists():
            runtime.warm_start(str(boot))
    reward = {
        "accept": 1.0,
        "ignore": -0.2,
        "dismiss": -0.8,
        "annoyed": -1.0,
    }.get(str(reaction).lower(), -0.2)
    runtime.record_outcome(used_memory_ids, reward, episode)
    runtime.save(state_dir)
