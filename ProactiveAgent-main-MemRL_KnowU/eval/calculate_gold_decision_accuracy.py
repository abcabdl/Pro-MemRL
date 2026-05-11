from __future__ import annotations

import csv
import json
import os
from typing import Any


def clamp_level(value: Any) -> int:
    return max(0, min(2, int(value or 0)))


def read_pred_decision(event: dict[str, Any]) -> tuple[int, int]:
    info = event.get("other_infomation", {})
    if isinstance(info, dict):
        decision = info.get("Decision", {})
        if isinstance(decision, dict):
            should = int(bool(decision.get("should_intervene", 0)))
            level = clamp_level(decision.get("commitment_level", 0))
            return should, level

    agent_response = event.get("agent_response")
    if isinstance(agent_response, list):
        should = int(bool(agent_response and agent_response[0] is not None))
        return should, should
    if isinstance(agent_response, dict):
        candidates = agent_response.get("candidate_task", [])
        should = int(bool(candidates))
        return should, should

    should = int(bool(event.get("task_status", False)))
    return should, should


def read_gold_decision(event: dict[str, Any]) -> tuple[int | None, int | None]:
    gold = event.get("gold_decision", {})
    if isinstance(gold, list):
        gold = gold[0] if gold else {}
    if isinstance(gold, dict) and gold:
        should = gold.get("should_intervene")
        if should is None:
            return None, None
        return int(bool(should)), clamp_level(gold.get("commitment_level", 0))

    if "task_status" in event:
        should = int(bool(event.get("task_status", False)))
        return should, should
    return None, None


def score_trace(
    pred_trace: list[dict[str, Any]],
    gold_trace: list[dict[str, Any]],
    eps: float = 1e-8,
) -> dict[str, float | int]:
    if len(pred_trace) != len(gold_trace):
        raise ValueError(f"Length mismatch: pred={len(pred_trace)} gold={len(gold_trace)}")

    total = 0
    should_correct = 0
    joint_correct = 0
    level_total = 0
    level_correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    proposed_total = 0
    accepted_total = 0

    for pred_event, gold_event in zip(pred_trace, gold_trace):
        gold_should, gold_level = read_gold_decision(gold_event)
        if gold_should is None:
            continue

        pred_should, pred_level = read_pred_decision(pred_event)
        total += 1
        should_ok = pred_should == gold_should
        should_correct += int(should_ok)
        exact_ok = should_ok and (gold_should == 0 or pred_level == gold_level)

        if pred_should == 1 and gold_should == 1:
            tp += 1
        elif pred_should == 0 and gold_should == 0:
            tn += 1
        elif pred_should == 1 and gold_should == 0:
            fp += 1
        else:
            fn += 1

        if pred_should == 1:
            proposed_total += 1
            accepted_total += int(exact_ok)

        if gold_should == 1 and gold_level is not None:
            level_total += 1
            level_ok = pred_level == gold_level
            level_correct += int(level_ok)
            joint_correct += int(should_ok and level_ok)
        else:
            joint_correct += int(should_ok)

    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    false_alarm = fp / (tp + fp + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    level_acc = level_correct / (level_total + eps)
    should_acc = should_correct / max(1, total)
    joint_acc = joint_correct / max(1, total)
    accept_rate = accepted_total / (proposed_total + eps)

    return {
        "should_acc": should_acc,
        "level_acc": level_acc,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "false_alarm": false_alarm,
        "f1_score": f1,
    }


def load_split_events(base_dir: str, file_names: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for file_name in file_names:
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            events.extend(json.load(f))
    return events


def list_json_files(base_dir: str) -> list[str]:
    rel_paths: list[str] = []
    for root, _, files in os.walk(base_dir):
        for file_name in files:
            if not file_name.endswith(".json"):
                continue
            if file_name == "splits.json":
                continue
            abs_path = os.path.join(root, file_name)
            rel_paths.append(os.path.relpath(abs_path, base_dir))
    rel_paths.sort()
    return rel_paths


def resolve_default_gold_dir(dir_path: str) -> str:
    candidates = [
        os.path.normpath(os.path.join(dir_path, "../dataset/test_data_with_level")),
        os.path.normpath(os.path.join(dir_path, "../dataset/test_data")),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "splits.json")):
            return candidate
    raise FileNotFoundError("Could not find dataset/test_data_with_level or dataset/test_data.")


def main(
    pred_dir: str,
    output: str | None = None,
    o: str | None = None,
    gold_dir: str | None = None,
    dir_path: str | None = None,
) -> None:
    output = output or o
    if dir_path is None:
        dir_path = os.path.dirname(__file__)
    if gold_dir is None:
        gold_dir = resolve_default_gold_dir(dir_path)

    split_path = os.path.join(gold_dir, "splits.json")
    rows: list[dict[str, float | int | str]] = []
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            splits: dict[str, dict[str, Any]] = json.load(f)

        all_files: set[str] = set()
        for split_name, split_cfg in splits.items():
            file_names = list(split_cfg["files"])
            pred_events = load_split_events(pred_dir, file_names)
            gold_events = load_split_events(gold_dir, file_names)
            rows.append({"Category": split_name, **score_trace(pred_events, gold_events)})
            all_files.update(file_names)

        overall_files = sorted(all_files)
        rows.append(
            {
                "Category": "overall",
                **score_trace(
                    load_split_events(pred_dir, overall_files),
                    load_split_events(gold_dir, overall_files),
                ),
            }
        )
    else:
        gold_files = set(list_json_files(gold_dir))
        pred_files = set(list_json_files(pred_dir))
        file_names = sorted(gold_files & pred_files)
        if not file_names:
            raise FileNotFoundError(
                f"No matching .json files found between pred_dir={pred_dir!r} and gold_dir={gold_dir!r}."
            )
        rows.append(
            {
                "Category": "overall",
                **score_trace(
                    load_split_events(pred_dir, file_names),
                    load_split_events(gold_dir, file_names),
                ),
            }
        )

    fieldnames = [
        "Category",
        "should_acc",
        "level_acc",
        "recall",
        "precision",
        "accuracy",
        "false_alarm",
        "f1_score",
    ]

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))

    if output is None:
        output = os.path.join(
            dir_path,
            "results",
            f"{os.path.basename(os.path.normpath(pred_dir))}_vs_gold_decision.csv",
        )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
