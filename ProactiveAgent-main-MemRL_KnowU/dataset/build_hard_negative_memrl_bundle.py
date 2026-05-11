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
from dataset.build_value_aware_memrl_from_judged import build_episode, load_jsonl  # noqa: E402


ERROR_WORDS = (
    "error",
    "failed",
    "fails",
    "traceback",
    "exception",
    "bug",
    "debug",
    "stuck",
    "not working",
    "issue",
    "problem",
)
DOC_WORDS = (
    "documentation",
    "docs",
    "stackoverflow",
    "stack overflow",
    "tutorial",
    "guide",
    "google search",
    "search result",
    "browser",
)
PROGRESS_WORDS = (
    "review",
    "scroll",
    "read",
    "opens",
    "clicks",
    "search",
    "takes notes",
    "copies",
    "saves",
    "writes",
    "continues",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def deep_copy_json(item: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(item, ensure_ascii=False))


def vote_strength(labels: dict[str, Any]) -> float:
    strengths: list[float] = []
    for key in ("need_vote", "accept_vote"):
        vote = labels.get(key, {}) if isinstance(labels.get(key), dict) else {}
        pos = int(vote.get("positive_votes", 0) or 0)
        neg = int(vote.get("negative_votes", 0) or 0)
        total = int(vote.get("total_votes", pos + neg) or 0)
        if total:
            strengths.append(abs(pos - neg) / total)
    return sum(strengths) / len(strengths) if strengths else 0.0


def signal(episode: dict[str, Any], name: str, default: float = 0.0) -> float:
    signals = episode.get("phase_a_signals", {}) or {}
    if not signals and isinstance(episode.get("gate_features"), dict):
        signals = episode["gate_features"].get("phase_a_signals", {}) or {}
    try:
        return float(signals.get(name, default))
    except (TypeError, ValueError):
        return default


def observations_text(episode: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in episode.get("observations", []) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("event", "")))
    parts.append(str(episode.get("intent_text", "")))
    candidate = episode.get("candidate", {}) or {}
    parts.append(str(candidate.get("purpose", "")))
    decision = episode.get("decision", {}) or {}
    parts.append(str(decision.get("reason", "")))
    return " ".join(parts).lower()


def bucket_of(episode: dict[str, Any]) -> str:
    bucket = episode.get("value_bucket")
    if bucket:
        return str(bucket)
    decision = episode.get("decision", {}) or {}
    return "helpful_positive" if decision.get("should_intervene") else "correct_abstain"


def has_candidate(episode: dict[str, Any]) -> bool:
    candidate = episode.get("candidate", {}) or {}
    return bool(candidate.get("proactive_task") or candidate.get("response"))


def normalize_episode(episode: dict[str, Any], *, source_kind: str, source_file: Path, source_line: int) -> dict[str, Any]:
    item = deep_copy_json(episode)
    item = enrich_memory_schema(item)
    metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
    metadata.update(
        {
            "hardneg_source_kind": source_kind,
            "hardneg_source_file": str(source_file),
            "hardneg_source_line": source_line,
        }
    )
    item["metadata"] = metadata
    return item


def load_candidate_pool(judged_sources: list[Path], episode_sources: list[Path]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for source in judged_sources:
        for line_no, row in enumerate(load_jsonl(source), 1):
            episode = build_episode(row)
            if episode is None:
                continue
            episodes.append(
                normalize_episode(
                    episode,
                    source_kind="judged_row",
                    source_file=source,
                    source_line=line_no,
                )
            )
    for source in episode_sources:
        for line_no, episode in enumerate(read_jsonl(source), 1):
            episodes.append(
                normalize_episode(
                    episode,
                    source_kind="episode",
                    source_file=source,
                    source_line=line_no,
                )
            )
    return episodes


def negative_score(episode: dict[str, Any]) -> tuple[float, tuple[str, ...]]:
    bucket = bucket_of(episode)
    context = str(episode.get("context_family") or "general")
    action = str(episode.get("action_family") or "no_intervention")
    text = observations_text(episode)
    labels = episode.get("labels", {}) if isinstance(episode.get("labels"), dict) else {}
    score = 0.0
    reasons: list[str] = []

    if bucket == "bad_intervention":
        score += 8.0
        reasons.append("bad_intervention")
    elif bucket == "correct_abstain":
        score += 3.0
        reasons.append("correct_abstain")

    if context == "active_search":
        score += 2.0
        reasons.append("active_search")
    if context == "stuck_debug":
        score += 1.8
        reasons.append("debug_like")
    if context == "drafting":
        score += 1.5
        reasons.append("drafting")
    if action == "no_intervention":
        score += 1.2
        reasons.append("no_intervention")

    flow = signal(episode, "flow")
    progress = signal(episode, "progress")
    stuck = signal(episode, "stuck")
    need = signal(episode, "need", float(labels.get("q_need", 0.0) or 0.0))
    if flow >= 0.55 or progress >= 0.55:
        score += 1.6
        reasons.append("smooth_progress")
    if stuck <= 0.35 and need <= 0.45:
        score += 1.4
        reasons.append("not_stuck_low_need")

    if any(word in text for word in DOC_WORDS):
        score += 1.0
        reasons.append("docs_or_search")
    if any(word in text for word in PROGRESS_WORDS):
        score += 0.8
        reasons.append("normal_progress")
    if any(word in text for word in ERROR_WORDS):
        score += 0.5
        reasons.append("tempting_error_or_debug_terms")

    return score, tuple(reasons)


def positive_score(episode: dict[str, Any]) -> float:
    labels = episode.get("labels", {}) if isinstance(episode.get("labels"), dict) else {}
    q_need = signal(episode, "need", float(labels.get("q_need", 0.0) or 0.0))
    q_accept = signal(episode, "accept", float(labels.get("q_accept", 0.0) or 0.0))
    strength = vote_strength(labels)
    score = q_need + q_accept + strength
    if has_candidate(episode):
        score += 0.5
    return score


def cap_sample(
    rng: random.Random,
    candidates: list[dict[str, Any]],
    *,
    quota: int,
    key_name: str,
    caps: dict[str, int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for item in candidates:
        key = str(item.get(key_name) or "general")
        if counts[key] >= caps.get(key, quota):
            continue
        selected.append(item)
        counts[key] += 1
        if len(selected) >= quota:
            break
    if len(selected) < quota:
        selected_ids = {id(item) for item in selected}
        leftovers = [item for item in candidates if id(item) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: quota - len(selected)])
    return selected[:quota]


def select_episodes(
    episodes: list[dict[str, Any]],
    *,
    positive: int,
    negative: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    positives = [item for item in episodes if bucket_of(item) == "helpful_positive" and has_candidate(item)]
    negatives = [item for item in episodes if bucket_of(item) in {"correct_abstain", "bad_intervention"}]

    positives.sort(key=positive_score, reverse=True)
    pos_caps = {
        "debug_help": max(1, round(positive * 0.45)),
        "search_refine": max(1, round(positive * 0.18)),
        "direct_suggestion": max(1, round(positive * 0.18)),
    }
    selected_pos = cap_sample(rng, positives, quota=positive, key_name="action_family", caps=pos_caps)

    scored_neg: list[tuple[float, tuple[str, ...], float, dict[str, Any]]] = []
    for item in negatives:
        score, reasons = negative_score(item)
        scored_neg.append((score, reasons, rng.random(), item))
    scored_neg.sort(key=lambda row: (row[0], row[2]), reverse=True)
    selected_neg = [item for _, _, _, item in scored_neg[:negative]]

    for score, reasons, _, item in scored_neg[:negative]:
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        metadata["hard_negative_score"] = round(score, 4)
        metadata["hard_negative_reasons"] = list(reasons)
        item["metadata"] = metadata

    selected = selected_pos + selected_neg
    rng.shuffle(selected)
    meta = {
        "available_positive": len(positives),
        "available_negative": len(negatives),
        "selected_positive": len(selected_pos),
        "selected_negative": len(selected_neg),
        "positive_action_caps": pos_caps,
        "negative_sampling": "top hard-negative score over correct_abstain and bad_intervention",
    }
    return selected, meta


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
                "source": "build_hard_negative_memrl_bundle.py",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_mix_jsonl(out_path: Path, selected: list[dict[str, Any]]) -> None:
    rows: list[str] = []
    for order, episode in enumerate(selected):
        item = deep_copy_json(episode)
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        item["_mix_order"] = order
        item["_mix_label"] = 1 if bucket_of(item) == "helpful_positive" else 0
        item["_mix_source_file"] = metadata.get("hardneg_source_file")
        item["_mix_source_line"] = metadata.get("hardneg_source_line")
        item["_mix_source_kind"] = metadata.get("hardneg_source_kind")
        item["_mix_sampling"] = "positive80_hard_negative220"
        rows.append(json.dumps(item, ensure_ascii=False))
    out_path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative MemRL bundle with fewer positives and hard negatives.")
    parser.add_argument("--judged-source", type=Path, action="append", default=[])
    parser.add_argument("--episode-source", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mix-output", type=Path)
    parser.add_argument("--positive", type=int, default=80)
    parser.add_argument("--negative", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--sim-threshold", type=float, default=0.18)
    args = parser.parse_args()

    episodes = load_candidate_pool(args.judged_source, args.episode_source)
    selected, meta = select_episodes(episodes, positive=args.positive, negative=args.negative, seed=args.seed)
    generation_rows = [_make_generation_row(item) for item in selected]
    simulation_rows = [_make_simulation_row(item) for item in selected]
    decision_rows = [_make_decision_row(item) for item in selected]

    write_episode_bundle(args.output_dir, selected, generation_rows, simulation_rows, decision_rows)
    write_snapshot(args.output_dir, selected, alpha=args.alpha, topk=args.topk, sim_threshold=args.sim_threshold)
    if args.mix_output:
        args.mix_output.parent.mkdir(parents=True, exist_ok=True)
        write_mix_jsonl(args.mix_output, selected)

    summary = {
        "episode_count": len(selected),
        "requested_positive": args.positive,
        "requested_negative": args.negative,
        "value_buckets": dict(Counter(bucket_of(item) for item in selected)),
        "decision_should_intervene_counts": dict(Counter(str(bool((item.get("decision") or {}).get("should_intervene"))) for item in selected)),
        "domains": dict(Counter(str(item.get("domain") or "general") for item in selected)),
        "context_families": dict(Counter(str(item.get("context_family") or "general") for item in selected)),
        "action_families": dict(Counter(str(item.get("action_family") or "no_intervention") for item in selected)),
        "source_kinds": dict(Counter(str((item.get("metadata") or {}).get("hardneg_source_kind")) for item in selected)),
        **meta,
    }
    (args.output_dir / "hard_negative_bundle_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
