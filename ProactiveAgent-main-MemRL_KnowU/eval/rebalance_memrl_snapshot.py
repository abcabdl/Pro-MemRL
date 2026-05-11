from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
if str(ROOT / "agent") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT / "agent"))

from agent.memrl.schema import enrich_memory_schema


TARGET_POSITIVE_FAMILIES = {
    ("active_search", "search_refine"),
    ("focused_coding", "debug_help"),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score_memory(memory: dict[str, Any]) -> tuple[float, float, int, str]:
    context_family = str(memory.get("context_family", "general"))
    action_family = str(memory.get("action_family", "no_intervention"))
    outcome_family = str(memory.get("outcome_family", "uncertain"))
    q_value = float(memory.get("q_value", memory.get("reward", 0.0)) or 0.0)
    targeted = int((context_family, action_family) in TARGET_POSITIVE_FAMILIES)
    outcome_rank = {
        "helpful_intervention": 4,
        "correct_abstain": 3,
        "missed_help": 2,
        "low_value_intervention": 2,
        "disruptive_intervention": 2,
        "uncertain": 1,
    }.get(outcome_family, 1)
    return (float(targeted), abs(q_value), float(outcome_rank), str(memory.get("memory_id", "")))


def _sort_memories(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(memories, key=_score_memory, reverse=True)


def _rebalance_memories(
    memories: list[dict[str, Any]],
    *,
    abstain_multiplier: float,
    uncertain_multiplier: float,
    targeted_positive_boost: int,
) -> list[dict[str, Any]]:
    for memory in memories:
        enrich_memory_schema(memory)

    positives = [m for m in memories if m.get("outcome_family") == "helpful_intervention"]
    abstains = [m for m in memories if m.get("outcome_family") == "correct_abstain"]
    uncertain = [m for m in memories if m.get("outcome_family") == "uncertain"]
    hard_cases = [m for m in memories if m.get("outcome_family") not in {"helpful_intervention", "correct_abstain", "uncertain"}]

    positives = _sort_memories(positives)
    abstains = _sort_memories(abstains)
    uncertain = _sort_memories(uncertain)
    hard_cases = _sort_memories(hard_cases)

    target_abstains = min(len(abstains), max(int(len(positives) * abstain_multiplier), len(hard_cases)))
    target_uncertain = min(len(uncertain), int(len(positives) * uncertain_multiplier))

    selected = list(positives)
    selected.extend(abstains[:target_abstains])
    selected.extend(uncertain[:target_uncertain])
    selected.extend(hard_cases)

    boosted: list[dict[str, Any]] = list(selected)
    if targeted_positive_boost > 1:
        for memory in positives:
            pair = (str(memory.get("context_family", "general")), str(memory.get("action_family", "no_intervention")))
            if pair not in TARGET_POSITIVE_FAMILIES:
                continue
            for replica_idx in range(1, int(targeted_positive_boost)):
                cloned = copy.deepcopy(memory)
                cloned["memory_id"] = f"{memory.get('memory_id', 'memory')}-boost{replica_idx}"
                cloned["sample_id"] = f"{memory.get('sample_id', memory.get('memory_id', 'memory'))}-boost{replica_idx}"
                boosted.append(cloned)
    return boosted


def _summarize(memories: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_counter = Counter(str(memory.get("outcome_family", "uncertain")) for memory in memories)
    action_counter = Counter(str(memory.get("action_family", "no_intervention")) for memory in memories)
    targeted_counter = Counter(
        f"{memory.get('context_family', 'general')}::{memory.get('action_family', 'no_intervention')}"
        for memory in memories
        if (str(memory.get("context_family", "general")), str(memory.get("action_family", "no_intervention"))) in TARGET_POSITIVE_FAMILIES
    )
    should_true = sum(1 for memory in memories if bool((memory.get("decision") or {}).get("should_intervene", False)))
    return {
        "count": len(memories),
        "should_intervene_true": should_true,
        "should_intervene_false": len(memories) - should_true,
        "outcome_family": dict(outcome_counter),
        "action_family": dict(action_counter.most_common(10)),
        "target_positive_families": dict(targeted_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebalance a MemRL snapshot to reduce abstain dominance.")
    parser.add_argument("--input", type=Path, required=True, help="Input memrl_snapshot.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output rebalanced memrl_snapshot.jsonl")
    parser.add_argument("--abstain-multiplier", type=float, default=3.0)
    parser.add_argument("--uncertain-multiplier", type=float, default=1.5)
    parser.add_argument("--targeted-positive-boost", type=int, default=3)
    args = parser.parse_args()

    memories = _load_jsonl(args.input)
    balanced = _rebalance_memories(
        memories,
        abstain_multiplier=float(args.abstain_multiplier),
        uncertain_multiplier=float(args.uncertain_multiplier),
        targeted_positive_boost=int(args.targeted_positive_boost),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in balanced),
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "input_summary": _summarize(memories),
                "output_summary": _summarize(balanced),
                "abstain_multiplier": float(args.abstain_multiplier),
                "uncertain_multiplier": float(args.uncertain_multiplier),
                "targeted_positive_boost": int(args.targeted_positive_boost),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
