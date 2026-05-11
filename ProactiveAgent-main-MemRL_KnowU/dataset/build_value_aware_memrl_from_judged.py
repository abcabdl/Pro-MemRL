from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.memrl.schema import enrich_memory_schema  # noqa: E402
from dataset.build_memrl_episode_dataset import (  # noqa: E402
    _build_intent_text,
    _candidate_from_teacher,
    _event_to_observation,
    _infer_domain,
    _make_decision_row,
    _make_generation_row,
    _make_simulation_row,
    utc_now,
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_binary(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return 1 if int(value) else 0
    except (TypeError, ValueError):
        return 0


def candidate_exists(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("proactive_task") or candidate.get("response"))


def vote_counts(labels: dict[str, Any], key: str) -> tuple[int, int, int]:
    vote = labels.get(key, {}) if isinstance(labels.get(key), dict) else {}
    pos = int(vote.get("positive_votes", 0) or 0)
    neg = int(vote.get("negative_votes", 0) or 0)
    total = int(vote.get("total_votes", pos + neg) or 0)
    return pos, neg, total


def vote_strength(labels: dict[str, Any]) -> float:
    need_pos, need_neg, need_total = vote_counts(labels, "need_vote")
    accept_pos, accept_neg, accept_total = vote_counts(labels, "accept_vote")
    strengths: list[float] = []
    if need_total:
        strengths.append(abs(need_pos - need_neg) / need_total)
    if accept_total:
        strengths.append(abs(accept_pos - accept_neg) / accept_total)
    return sum(strengths) / len(strengths) if strengths else 0.0


def infer_value_bucket(row: dict[str, Any], candidate: dict[str, Any], labels: dict[str, Any]) -> str:
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    teacher_should = bool(as_binary(teacher.get("y_need_pred")) and candidate_exists(candidate))
    y_need = as_binary(labels.get("y_need"))
    y_accept = as_binary(labels.get("y_accept"))
    y_star = as_binary(labels.get("y_star", int(bool(y_need and y_accept and candidate_exists(candidate)))))
    if y_star and candidate_exists(candidate):
        return "helpful_positive"
    if teacher_should and not y_star:
        return "bad_intervention"
    if (not teacher_should) and y_need:
        return "missed_help"
    if not y_need:
        return "correct_abstain"
    return "weak_uncertain"


def q_from_bucket(bucket: str, labels: dict[str, Any]) -> float:
    strength = vote_strength(labels)
    if bucket == "helpful_positive":
        return round(0.4 + 0.6 * strength, 4)
    if bucket == "correct_abstain":
        return round(0.35 + 0.45 * strength, 4)
    if bucket == "bad_intervention":
        return round(-(0.4 + 0.6 * strength), 4)
    if bucket == "missed_help":
        return round(-(0.35 + 0.45 * strength), 4)
    return 0.1


def simulation_from_bucket(row: dict[str, Any], bucket: str, labels: dict[str, Any]) -> dict[str, Any]:
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    votes = row.get("judge_votes", {}) if isinstance(row.get("judge_votes"), dict) else {}
    reasons: list[str] = []
    for model, payload in votes.items():
        if not isinstance(payload, dict) or "error" in payload:
            continue
        thought_need = payload.get("thought_need")
        thought_accept = payload.get("thought_accept")
        if thought_need:
            reasons.append(f"{model}: {thought_need}")
        if thought_accept:
            reasons.append(f"{model}: {thought_accept}")
    if bucket == "helpful_positive":
        acceptance = "accept"
    elif bucket == "bad_intervention":
        acceptance = "dismiss"
    elif bucket == "missed_help":
        acceptance = "accept"
    else:
        acceptance = "ignore"
    return {
        "acceptance": acceptance,
        "acceptance_confidence": safe_float(teacher.get("q_accept"), 0.5),
        "flow_impact": "low" if bucket in {"helpful_positive", "missed_help"} else "high",
        "relevance": "high" if bucket in {"helpful_positive", "missed_help"} else "low",
        "timing": "good" if bucket in {"helpful_positive", "missed_help"} else "interruptive",
        "reasoning": " | ".join(reasons) or str(labels),
    }


def decision_from_bucket(candidate: dict[str, Any], bucket: str, labels: dict[str, Any], simulation: dict[str, Any]) -> dict[str, Any]:
    should = bucket == "helpful_positive"
    if should:
        level = 2 if candidate.get("response") and candidate.get("proactive_task") else 1
    else:
        level = 0
    return {
        "should_intervene": should,
        "commitment_level": level,
        "risk": "low" if should else "high" if bucket == "bad_intervention" else "medium",
        "reason": simulation.get("reasoning") or f"value_bucket={bucket}",
    }


def source_weight(row: dict[str, Any], bucket: str) -> float:
    source = str(row.get("source", row.get("source_pred_task", "judged_teacher_stage1")))
    if source == "teacher_stage1_weak" or "teacher" in source:
        if bucket == "correct_abstain":
            return 0.55
        if bucket in {"helpful_positive", "bad_intervention", "missed_help"}:
            return 0.85
    return 1.0


def build_episode(row: dict[str, Any]) -> dict[str, Any] | None:
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    if not teacher or not labels:
        return None
    observations = [_event_to_observation(item) for item in row.get("obs", []) if isinstance(item, dict)]
    candidate = _candidate_from_teacher(teacher)
    bucket = infer_value_bucket(row, candidate, labels)
    if bucket != "helpful_positive":
        if bucket in {"correct_abstain", "bad_intervention"}:
            candidate["operation"] = "nop" if not candidate_exists(candidate) else candidate.get("operation")
        if bucket == "correct_abstain":
            candidate["proactive_task"] = None
            candidate["response"] = None
            candidate["operation"] = "nop"
    simulation = simulation_from_bucket(row, bucket, labels)
    decision = decision_from_bucket(candidate, bucket, labels, simulation)
    q_value = q_from_bucket(bucket, labels)
    now = utc_now()
    episode = {
        "memory_id": f"judged-{row.get('sample_id')}",
        "sample_id": str(row.get("sample_id", "")),
        "source": "judged_teacher_stage1",
        "domain": _infer_domain(observations),
        "observations": observations,
        "intent_text": _build_intent_text(observations, teacher),
        "candidate": candidate,
        "simulation": simulation,
        "decision": decision,
        "labels": {
            "y_need": labels.get("y_need"),
            "y_accept": labels.get("y_accept"),
            "gold_should": labels.get("y_star", labels.get("y_need")),
            "gold_level": decision["commitment_level"],
            "q_need": teacher.get("q_need"),
            "q_accept": teacher.get("q_accept"),
            "need_vote": labels.get("need_vote"),
            "accept_vote": labels.get("accept_vote"),
        },
        "value_bucket": bucket,
        "phase_a_signals": (row.get("phase_a") or {}).get("signals", {}) if isinstance(row.get("phase_a"), dict) else {},
        "gate_features": {
            "phase_a_signals": (row.get("phase_a") or {}).get("signals", {}) if isinstance(row.get("phase_a"), dict) else {},
            "prompt_source": (row.get("phase_a") or {}).get("prompt_source") if isinstance(row.get("phase_a"), dict) else None,
        },
        "memory_quality": {
            "label_strength": "judged",
            "source_weight": source_weight(row, bucket),
            "vote_strength": round(vote_strength(labels), 4),
            "judge_models_used": labels.get("judge_models_used", []),
        },
        "reward": q_value,
        "q_value": q_value,
        "q_visits": 0,
        "created_at": now,
        "updated_at": now,
    }
    return enrich_memory_schema(episode)


def write_snapshot(out_dir: Path, episodes: list[dict[str, Any]], *, alpha: float, topk: int, sim_threshold: float) -> None:
    snapshot_dir = out_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "memrl_snapshot.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in episodes),
        encoding="utf-8",
    )
    (snapshot_dir / "memrl_meta.json").write_text(
        json.dumps(
            {
                "alpha": alpha,
                "topk": topk,
                "sim_threshold": sim_threshold,
                "count": len(episodes),
                "source": "build_value_aware_memrl_from_judged.py",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build value-aware MemRL memory from multi-LLM judged teacher rows.")
    parser.add_argument("--judged-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--sim-threshold", type=float, default=0.18)
    args = parser.parse_args()

    rows = load_jsonl(args.judged_source)
    episodes = [episode for row in rows if (episode := build_episode(row)) is not None]
    generation_rows = [_make_generation_row(episode) for episode in episodes]
    simulation_rows = [_make_simulation_row(episode) for episode in episodes]
    decision_rows = [_make_decision_row(episode) for episode in episodes]
    write_episode_bundle(args.output_dir, episodes, generation_rows, simulation_rows, decision_rows)
    write_snapshot(args.output_dir, episodes, alpha=args.alpha, topk=args.topk, sim_threshold=args.sim_threshold)
    bucket_counts: dict[str, int] = {}
    for episode in episodes:
        bucket = str(episode.get("value_bucket", "unknown"))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    (args.output_dir / "value_bucket_summary.json").write_text(
        json.dumps({"episode_count": len(episodes), "value_buckets": bucket_counts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"episodes={len(episodes)} output_dir={args.output_dir}")
    print(f"snapshot={args.output_dir / 'snapshot' / 'memrl_snapshot.jsonl'}")


if __name__ == "__main__":
    main()
