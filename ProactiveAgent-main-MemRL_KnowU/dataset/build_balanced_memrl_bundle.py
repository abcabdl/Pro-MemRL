from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.memrl.schema import enrich_memory_schema  # noqa: E402
from dataset.build_memrl_episode_dataset import (  # noqa: E402
    _make_decision_row,
    _make_generation_row,
    _make_simulation_row,
    write_episode_bundle,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bucket_of(episode: dict[str, Any]) -> str:
    bucket = episode.get("value_bucket")
    if bucket:
        return str(bucket)
    decision = episode.get("decision", {}) or {}
    outcome = episode.get("outcome_family")
    if outcome == "false_negative":
        return "missed_help"
    if outcome == "false_positive":
        return "bad_intervention"
    return "helpful_positive" if decision.get("should_intervene") else "correct_abstain"


def has_candidate(episode: dict[str, Any]) -> bool:
    candidate = episode.get("candidate", {}) or {}
    return bool(candidate.get("proactive_task") or candidate.get("response"))


def domain_of(episode: dict[str, Any]) -> str:
    return str(episode.get("domain") or "general")


def desired_domain_counts(candidates: list[dict[str, Any]], quota: int) -> dict[str, int]:
    counts = Counter(domain_of(item) for item in candidates)
    if quota <= 0 or not counts:
        return {}
    total = sum(counts.values())
    raw = {domain: quota * count / total for domain, count in counts.items()}
    desired = {domain: min(counts[domain], int(raw[domain])) for domain in counts}
    remaining = quota - sum(desired.values())
    order = sorted(counts, key=lambda domain: (raw[domain] - int(raw[domain]), counts[domain]), reverse=True)
    while remaining > 0:
        changed = False
        for domain in order:
            if desired[domain] < counts[domain]:
                desired[domain] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return desired


def stratified_sample(
    rng: random.Random,
    candidates: list[dict[str, Any]],
    quota: int,
) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_domain[domain_of(item)].append(item)
    selected: list[dict[str, Any]] = []
    for domain, count in desired_domain_counts(candidates, quota).items():
        items = list(by_domain[domain])
        rng.shuffle(items)
        selected.extend(items[:count])
    if len(selected) < quota:
        selected_ids = {id(item) for item in selected}
        leftovers = [item for item in candidates if id(item) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: quota - len(selected)])
    rng.shuffle(selected)
    return selected[:quota]


def balanced_sample(
    episodes: list[dict[str, Any]],
    *,
    total: int,
    positive_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positive = [item for item in episodes if bucket_of(item) == "helpful_positive" and has_candidate(item)]
    negative = [item for item in episodes if bucket_of(item) == "correct_abstain"]
    target_positive = min(len(positive), round(total * positive_ratio))
    target_negative = min(len(negative), total - target_positive)
    if target_positive + target_negative < total:
        target_positive = min(len(positive), target_positive + (total - target_positive - target_negative))
    selected = stratified_sample(rng, positive, target_positive) + stratified_sample(rng, negative, target_negative)
    rng.shuffle(selected)
    return selected[:total]


def normalize_episode(episode: dict[str, Any]) -> dict[str, Any]:
    item = json.loads(json.dumps(episode, ensure_ascii=False))
    item["value_bucket"] = bucket_of(item)
    item = enrich_memory_schema(item)
    if item["value_bucket"] == "correct_abstain":
        item["q_value"] = min(float(item.get("q_value", item.get("reward", 0.0)) or 0.0), 0.45)
        item["reward"] = min(float(item.get("reward", item.get("q_value", 0.0)) or 0.0), 0.45)
    elif item["value_bucket"] == "helpful_positive":
        item["q_value"] = max(float(item.get("q_value", item.get("reward", 0.0)) or 0.0), 0.8)
        item["reward"] = max(float(item.get("reward", item.get("q_value", 0.0)) or 0.0), 0.8)
    return item


def write_snapshot(out_dir: Path, episodes: list[dict[str, Any]], *, alpha: float, topk: int, sim_threshold: float) -> None:
    snapshot = out_dir / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "memrl_episodes.jsonl", snapshot / "memrl_snapshot.jsonl")
    (snapshot / "memrl_meta.json").write_text(
        json.dumps(
            {
                "alpha": alpha,
                "topk": topk,
                "sim_threshold": sim_threshold,
                "count": len(episodes),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced MemRL bundle from a large episode JSONL.")
    parser.add_argument("--source", type=Path, required=True, help="Source memrl_episodes.jsonl or memrl_snapshot.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--sim-threshold", type=float, default=0.18)
    args = parser.parse_args()

    rows = load_jsonl(args.source)
    selected = [normalize_episode(item) for item in balanced_sample(
        rows,
        total=args.total,
        positive_ratio=args.positive_ratio,
        seed=args.seed,
    )]
    generation_rows = [_make_generation_row(item) for item in selected]
    simulation_rows = [_make_simulation_row(item) for item in selected]
    decision_rows = [_make_decision_row(item) for item in selected]
    write_episode_bundle(args.output_dir, selected, generation_rows, simulation_rows, decision_rows)
    write_snapshot(args.output_dir, selected, alpha=args.alpha, topk=args.topk, sim_threshold=args.sim_threshold)
    summary = Counter(bucket_of(item) for item in selected)
    (args.output_dir / "value_bucket_summary.json").write_text(
        json.dumps({"episode_count": len(selected), "value_buckets": dict(summary)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"episode_count": len(selected), "value_buckets": dict(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
