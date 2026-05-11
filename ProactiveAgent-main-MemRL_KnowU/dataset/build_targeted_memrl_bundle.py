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


HELP_WORDS = (
    "stack overflow",
    "stackoverflow",
    "forum",
    "documentation",
    "docs",
    "tutorial",
    "guide",
    "community",
    "slack",
    "search result",
    "searches for",
)
FAIL_WORDS = (
    "error",
    "failed",
    "fails",
    "glitch",
    "bug",
    "issue",
    "problem",
    "traceback",
    "exception",
    "not working",
    "compile",
    "testing",
    "logs",
    "breakpoint",
    "debug",
)
REPEAT_WORDS = (
    "again",
    "multiple",
    "repeated",
    "continues",
    "another",
    "back to",
    "returns to",
    "reopen",
    "repeatedly",
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


def bucket_of(episode: dict[str, Any]) -> str:
    bucket = episode.get("value_bucket")
    if bucket:
        return str(bucket)
    decision = episode.get("decision", {}) or {}
    return "helpful_positive" if decision.get("should_intervene") else "correct_abstain"


def has_candidate(episode: dict[str, Any]) -> bool:
    candidate = episode.get("candidate", {}) or {}
    return bool(candidate.get("proactive_task") or candidate.get("response"))


def signal(episode: dict[str, Any], name: str, default: float = 0.0) -> float:
    signals = episode.get("phase_a_signals", {}) or {}
    if not signals and isinstance(episode.get("gate_features"), dict):
        signals = episode["gate_features"].get("phase_a_signals", {}) or {}
    try:
        return float(signals.get(name, default))
    except (TypeError, ValueError):
        return default


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


def observations_text(episode: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in episode.get("observations", []) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("event", "")))
    candidate = episode.get("candidate", {}) or {}
    decision = episode.get("decision", {}) or {}
    parts.extend(
        [
            str(episode.get("intent_text", "")),
            str(candidate.get("purpose", "")),
            str(decision.get("reason", "")),
        ]
    )
    return " ".join(parts).lower()


def has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def repeated_failure(episode: dict[str, Any]) -> bool:
    text = observations_text(episode)
    if not has_any(text, FAIL_WORDS):
        return False
    return (
        has_any(text, REPEAT_WORDS)
        or text.count("search") >= 2
        or text.count("error") >= 2
        or text.count("debug") >= 2
    )


def active_search_helpseek(episode: dict[str, Any]) -> bool:
    return (
        bucket_of(episode) == "helpful_positive"
        and str(episode.get("context_family") or "") == "active_search"
        and has_any(observations_text(episode), HELP_WORDS)
    )


def stuck_debug_repeated_fail(episode: dict[str, Any]) -> bool:
    return (
        bucket_of(episode) == "helpful_positive"
        and str(episode.get("context_family") or "") == "stuck_debug"
        and repeated_failure(episode)
    )


def active_search_helpseek_strong(episode: dict[str, Any]) -> bool:
    labels = episode.get("labels", {}) if isinstance(episode.get("labels"), dict) else {}
    need = signal(episode, "need", float(labels.get("q_need", 0.0) or 0.0))
    accept = signal(episode, "accept", float(labels.get("q_accept", 0.0) or 0.0))
    return active_search_helpseek(episode) and need >= 0.55 and accept >= 0.55


def stuck_debug_repeated_fail_strong(episode: dict[str, Any]) -> bool:
    labels = episode.get("labels", {}) if isinstance(episode.get("labels"), dict) else {}
    need = signal(episode, "need", float(labels.get("q_need", 0.0) or 0.0))
    accept = signal(episode, "accept", float(labels.get("q_accept", 0.0) or 0.0))
    stuck = signal(episode, "stuck")
    return stuck_debug_repeated_fail(episode) and need >= 0.6 and accept >= 0.55 and stuck >= 0.45


def active_search_correct_abstain_strong(episode: dict[str, Any]) -> bool:
    if bucket_of(episode) != "correct_abstain" or str(episode.get("context_family") or "") != "active_search":
        return False
    labels = episode.get("labels", {}) if isinstance(episode.get("labels"), dict) else {}
    flow = signal(episode, "flow")
    progress = signal(episode, "progress")
    need = signal(episode, "need", float(labels.get("q_need", 0.0) or 0.0))
    return flow >= 0.55 and progress >= 0.45 and need <= 0.45


def stuck_debug_smoothish_abstain(episode: dict[str, Any]) -> bool:
    if bucket_of(episode) != "correct_abstain" or str(episode.get("context_family") or "") != "stuck_debug":
        return False
    labels = episode.get("labels", {}) if isinstance(episode.get("labels"), dict) else {}
    need = signal(episode, "need", float(labels.get("q_need", 0.0) or 0.0))
    flow = signal(episode, "flow")
    progress = signal(episode, "progress")
    return need <= 0.5 and flow >= 0.45 and progress >= 0.35


def drafting_abstain_strong(episode: dict[str, Any]) -> bool:
    return (
        bucket_of(episode) == "correct_abstain"
        and str(episode.get("context_family") or "") == "drafting"
        and signal(episode, "flow") >= 0.55
    )


def positive_score(episode: dict[str, Any]) -> float:
    labels = episode.get("labels", {}) if isinstance(episode.get("labels"), dict) else {}
    need = signal(episode, "need", float(labels.get("q_need", 0.0) or 0.0))
    accept = signal(episode, "accept", float(labels.get("q_accept", 0.0) or 0.0))
    score = need + accept + vote_strength(labels)
    if active_search_helpseek_strong(episode):
        score += 2.0
    elif active_search_helpseek(episode):
        score += 1.0
    if stuck_debug_repeated_fail_strong(episode):
        score += 2.5
    elif stuck_debug_repeated_fail(episode):
        score += 1.2
    if has_candidate(episode):
        score += 0.5
    return score


def negative_score(episode: dict[str, Any]) -> float:
    score = 0.0
    if bucket_of(episode) == "bad_intervention":
        score += 6.0
    if active_search_correct_abstain_strong(episode):
        score += 2.5
    if stuck_debug_smoothish_abstain(episode):
        score += 2.3
    if drafting_abstain_strong(episode):
        score += 2.0
    score += signal(episode, "flow")
    score += signal(episode, "progress")
    score -= signal(episode, "need")
    score -= signal(episode, "stuck") * 0.5
    if has_any(observations_text(episode), HELP_WORDS):
        score += 0.6
    return score


def normalize_episode(episode: dict[str, Any], *, source_kind: str, source_file: Path, source_line: int) -> dict[str, Any]:
    item = deep_copy_json(episode)
    item = enrich_memory_schema(item)
    metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
    metadata.update(
        {
            "targeted_source_kind": source_kind,
            "targeted_source_file": str(source_file),
            "targeted_source_line": source_line,
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


def take_top(candidates: list[dict[str, Any]], quota: int, score_fn) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=score_fn, reverse=True)
    return ranked[:quota]


def extend_unique(selected: list[dict[str, Any]], candidates: list[dict[str, Any]], quota: int, score_fn) -> None:
    selected_ids = {id(item) for item in selected}
    ranked = sorted(candidates, key=score_fn, reverse=True)
    for item in ranked:
        if id(item) in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(id(item))
        if len(selected) >= quota:
            return


def select_episodes(
    episodes: list[dict[str, Any]],
    *,
    positive_total: int,
    negative_total: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)

    positive_pool = [item for item in episodes if bucket_of(item) == "helpful_positive" and has_candidate(item)]
    negative_pool = [item for item in episodes if bucket_of(item) in {"correct_abstain", "bad_intervention"}]

    pos_active_strong = [item for item in positive_pool if active_search_helpseek_strong(item)]
    pos_active_other = [item for item in positive_pool if active_search_helpseek(item) and not active_search_helpseek_strong(item)]
    pos_stuck_strong = [item for item in positive_pool if stuck_debug_repeated_fail_strong(item)]
    pos_stuck_other = [item for item in positive_pool if stuck_debug_repeated_fail(item) and not stuck_debug_repeated_fail_strong(item)]
    pos_remaining = [item for item in positive_pool if item not in pos_active_strong + pos_active_other + pos_stuck_strong + pos_stuck_other]

    neg_bad = [item for item in negative_pool if bucket_of(item) == "bad_intervention"]
    neg_active = [item for item in negative_pool if active_search_correct_abstain_strong(item)]
    neg_stuck = [item for item in negative_pool if stuck_debug_smoothish_abstain(item)]
    neg_drafting = [item for item in negative_pool if drafting_abstain_strong(item)]
    neg_remaining = [item for item in negative_pool if item not in neg_bad + neg_active + neg_stuck + neg_drafting]

    selected_pos: list[dict[str, Any]] = []
    extend_unique(selected_pos, pos_stuck_strong, 40, positive_score)
    extend_unique(selected_pos, pos_stuck_other, 65, positive_score)
    extend_unique(selected_pos, pos_active_strong, 79, positive_score)
    extend_unique(selected_pos, pos_active_other, 90, positive_score)
    extend_unique(selected_pos, pos_remaining, positive_total, positive_score)

    selected_neg: list[dict[str, Any]] = []
    extend_unique(selected_neg, neg_bad, min(22, negative_total), negative_score)
    extend_unique(selected_neg, neg_active, min(92, negative_total), negative_score)
    extend_unique(selected_neg, neg_stuck, min(137, negative_total), negative_score)
    extend_unique(selected_neg, neg_drafting, min(162, negative_total), negative_score)
    extend_unique(selected_neg, neg_remaining, negative_total, negative_score)

    selected = selected_pos[:positive_total] + selected_neg[:negative_total]
    rng.shuffle(selected)

    meta = {
        "available_positive": len(positive_pool),
        "available_negative": len(negative_pool),
        "selected_positive": len(selected_pos[:positive_total]),
        "selected_negative": len(selected_neg[:negative_total]),
        "positive_subgroups": {
            "active_search_helpseek_strong": len(pos_active_strong),
            "active_search_helpseek_other": len(pos_active_other),
            "stuck_debug_repeated_fail_strong": len(pos_stuck_strong),
            "stuck_debug_repeated_fail_other": len(pos_stuck_other),
        },
        "negative_subgroups": {
            "bad_intervention": len(neg_bad),
            "active_search_correct_abstain_strong": len(neg_active),
            "stuck_debug_smoothish_abstain": len(neg_stuck),
            "drafting_abstain_strong": len(neg_drafting),
        },
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
                "source": "build_targeted_memrl_bundle.py",
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
        item["_mix_source_file"] = metadata.get("targeted_source_file")
        item["_mix_source_line"] = metadata.get("targeted_source_line")
        item["_mix_source_kind"] = metadata.get("targeted_source_kind")
        item["_mix_sampling"] = "targeted_100p200n"
        rows.append(json.dumps(item, ensure_ascii=False))
    out_path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a targeted MemRL bundle that restores strong positives and prioritizes hard negatives.")
    parser.add_argument("--judged-source", type=Path, action="append", default=[])
    parser.add_argument("--episode-source", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mix-output", type=Path)
    parser.add_argument("--positive", type=int, default=100)
    parser.add_argument("--negative", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--sim-threshold", type=float, default=0.18)
    args = parser.parse_args()

    episodes = load_candidate_pool(args.judged_source, args.episode_source)
    selected, meta = select_episodes(
        episodes,
        positive_total=args.positive,
        negative_total=args.negative,
        seed=args.seed,
    )

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
        "source_kinds": dict(Counter(str((item.get("metadata") or {}).get("targeted_source_kind")) for item in selected)),
        **meta,
    }
    (args.output_dir / "targeted_bundle_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
