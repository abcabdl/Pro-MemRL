from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def has_candidate(row: dict[str, Any]) -> bool:
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    return bool(teacher.get("proactive_task") or teacher.get("response"))


def teacher_positive(row: dict[str, Any]) -> bool:
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    return bool(int(teacher.get("y_need_pred", 0) or 0)) and has_candidate(row)


def need_score(row: dict[str, Any]) -> float:
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    return safe_float(teacher.get("q_need", 0.0)) * 0.7 + safe_float(teacher.get("q_accept", 0.0)) * 0.3


def domain_of(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "").lower()
    text = " ".join(str(item.get("event", "")) for item in row.get("obs", []) if isinstance(item, dict)).lower()
    if any(token in text for token in ("code", "python", "javascript", "java", "ruby", "visual studio", "terminal", "github", "stackoverflow")):
        return "coding"
    if any(token in text for token in ("write", "document", "report", "essay", "article", "email", "draft")):
        return "writing"
    if any(token in text for token in ("bank", "payment", "price", "stock", "finance", "budget")):
        return "finance"
    if category:
        return category.split("/")[0][:32]
    return "general"


def row_rank(row: dict[str, Any]) -> tuple[float, float, int]:
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    return (
        need_score(row),
        safe_float(teacher.get("q_need", 0.0)),
        len(row.get("obs", []) or []),
    )


def stratified_take(rng: random.Random, rows: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[domain_of(row)].append(row)
    total = sum(len(items) for items in by_domain.values())
    selected: list[dict[str, Any]] = []
    if total <= 0 or quota <= 0:
        return selected
    desired: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for domain, items in by_domain.items():
        raw = quota * len(items) / total
        desired[domain] = min(len(items), int(raw))
        fractions.append((raw - int(raw), domain))
    remaining = quota - sum(desired.values())
    for _, domain in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        if desired[domain] < len(by_domain[domain]):
            desired[domain] += 1
            remaining -= 1
    for domain, count in desired.items():
        items = sorted(by_domain[domain], key=row_rank, reverse=True)
        top_pool = items[: max(count, min(len(items), count * 3))]
        rng.shuffle(top_pool)
        selected.extend(sorted(top_pool[:count], key=row_rank, reverse=True))
    if len(selected) < quota:
        selected_ids = {id(row) for row in selected}
        leftovers = [row for row in sorted(rows, key=row_rank, reverse=True) if id(row) not in selected_ids]
        selected.extend(leftovers[: quota - len(selected)])
    rng.shuffle(selected)
    return selected[:quota]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select likely-positive teacher rows before expensive multi-model judging.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=600)
    parser.add_argument("--positive-ratio", type=float, default=0.75)
    parser.add_argument("--min-q-need", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(args.input)
    positives = [row for row in rows if teacher_positive(row)]
    hard_candidates = [
        row
        for row in rows
        if row not in positives and has_candidate(row) and safe_float((row.get("teacher") or {}).get("q_need", 0.0)) >= args.min_q_need
    ]
    negatives = [row for row in rows if row not in positives and row not in hard_candidates]

    positive_quota = min(len(positives), round(args.total * args.positive_ratio))
    hard_quota = min(len(hard_candidates), max(0, round(args.total * 0.15)))
    negative_quota = max(0, args.total - positive_quota - hard_quota)

    selected = (
        stratified_take(rng, positives, positive_quota)
        + stratified_take(rng, hard_candidates, hard_quota)
        + stratified_take(rng, negatives, negative_quota)
    )
    if len(selected) < args.total:
        selected_ids = {id(row) for row in selected}
        leftovers = [row for row in sorted(rows, key=row_rank, reverse=True) if id(row) not in selected_ids]
        selected.extend(leftovers[: args.total - len(selected)])
    rng.shuffle(selected)
    selected = selected[: args.total]
    write_jsonl(args.output, selected)

    summary = {
        "input_count": len(rows),
        "output_count": len(selected),
        "teacher_positive_count": sum(1 for row in selected if teacher_positive(row)),
        "has_candidate_count": sum(1 for row in selected if has_candidate(row)),
        "avg_q_need": round(sum(safe_float((row.get("teacher") or {}).get("q_need", 0.0)) for row in selected) / max(len(selected), 1), 4),
        "domains": dict(Counter(domain_of(row) for row in selected)),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
