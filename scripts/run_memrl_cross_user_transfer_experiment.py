from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_CONDITIONS = (
    "same_user_only",
    "same_user_online",
    "cross_user_no_gate",
    "cross_user_no_gate_online",
    "cross_user_gate_frozen",
    "transfer_gate_online",
)

CONDITIONS = (
    *DEFAULT_CONDITIONS,
    "no_memory",
    "source_only_no_gate",
    "source_only_gate",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_extra_args(values: list[str] | None) -> list[str]:
    if not values:
        return []
    if values and values[0] == "--":
        return values[1:]
    return values


def _display_command(command: list[str]) -> str:
    displayed = list(command)
    for idx, value in enumerate(displayed[:-1]):
        if value == "--api-key":
            displayed[idx + 1] = "REDACTED"
    return " ".join(displayed)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Bootstrap memory file does not exist: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def _memory_profile(memory: dict[str, Any]) -> str | None:
    labels = memory.get("labels", {}) if isinstance(memory.get("labels"), dict) else {}
    if labels.get("profile_id"):
        return str(labels["profile_id"])

    for key in ("sample_id", "memory_id", "intent_text"):
        text = str(memory.get(key, "") or "")
        for pattern in (
            r"profile_task::([^:]+)::",
            r"profile_habit::([^:]+)::",
            r"profile_background::([^:]+)::",
            r"knowu-profile-task-([^-]+)-",
            r"knowu-profile-habit-([^-]+)-",
            r"knowu-profile-background-([^-]+)-",
        ):
            match = re.search(pattern, text)
            if match:
                return match.group(1)
    return None


def _memory_task_family(memory: dict[str, Any]) -> str:
    labels = memory.get("labels", {}) if isinstance(memory.get("labels"), dict) else {}
    if labels.get("task_family"):
        return str(labels["task_family"])
    sample_id = str(memory.get("sample_id", "") or "")
    match = re.search(r"profile_task::[^:]+::([^:]+)(?:::|$)", sample_id)
    if match:
        return match.group(1)
    memory_id = str(memory.get("memory_id", "") or "")
    match = re.search(r"knowu-profile-task-[^-]+-([a-z0-9_]+)-v\d+", memory_id)
    if match:
        return match.group(1)
    for key in ("context_family", "action_family"):
        text = str(memory.get(key, "") or "")
        for pattern in (r"knowu_profile_task_([a-z0-9_]+)", r"knowu_([a-z0-9_]+)"):
            match = re.search(pattern, text)
            if match:
                return match.group(1)
    return "__unscoped__"


def _select_limited_target_memories(
    memories: list[dict[str, Any]],
    *,
    target_user: str,
    per_family: int,
    include_unscoped: bool,
) -> list[dict[str, Any]]:
    target_memories = [
        item
        for item in memories
        if _memory_profile(item) == target_user
        and (include_unscoped or _memory_task_family(item) != "__unscoped__")
    ]
    if per_family < 0:
        return target_memories
    if per_family == 0:
        return []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in target_memories:
        grouped[_memory_task_family(item)].append(item)

    selected: list[dict[str, Any]] = []
    for family in sorted(grouped):
        selected.extend(
            sorted(
                grouped[family],
                key=lambda item: (
                    str(item.get("sample_id", "")),
                    str(item.get("memory_id", "")),
                ),
            )[:per_family]
        )
    return selected


def _select_source_memories(
    memories: list[dict[str, Any]],
    *,
    target_user: str,
    source_users: list[str],
    include_unscoped: bool,
) -> list[dict[str, Any]]:
    source_set = set(source_users)
    selected = []
    for item in memories:
        profile = _memory_profile(item)
        if profile is None:
            if include_unscoped:
                selected.append(item)
            continue
        if profile != target_user and (not source_set or profile in source_set):
            selected.append(item)
    return selected


def _limit_memories_per_family(
    memories: list[dict[str, Any]],
    *,
    per_family: int,
) -> list[dict[str, Any]]:
    if per_family < 0:
        return memories
    if per_family == 0:
        return []

    grouped: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for item in memories:
        grouped[(_memory_profile(item), _memory_task_family(item))].append(item)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: (str(value[0]), value[1])):
        selected.extend(
            sorted(
                grouped[key],
                key=lambda item: (
                    str(item.get("sample_id", "")),
                    str(item.get("memory_id", "")),
                ),
            )[:per_family]
        )
    return selected


def _available_profiles(memories: list[dict[str, Any]]) -> list[str]:
    return sorted({profile for item in memories if (profile := _memory_profile(item))})


def _build_condition_bootstrap(
    *,
    condition: str,
    all_memories: list[dict[str, Any]],
    target_user: str,
    source_users: list[str],
    target_memories_per_family: int,
    source_memories_per_family: int,
    include_unscoped: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_memories = _select_limited_target_memories(
        all_memories,
        target_user=target_user,
        per_family=target_memories_per_family,
        include_unscoped=include_unscoped,
    )
    source_memories = _select_source_memories(
        all_memories,
        target_user=target_user,
        source_users=source_users,
        include_unscoped=include_unscoped,
    )
    limited_source_memories = _limit_memories_per_family(
        source_memories,
        per_family=source_memories_per_family,
    )

    if condition == "no_memory":
        rows = []
        included_target_count = 0
        included_source_count = 0
    elif condition in {"same_user_only", "same_user_online"}:
        rows = target_memories
        included_target_count = len(target_memories)
        included_source_count = 0
    elif condition in {
        "cross_user_no_gate",
        "cross_user_no_gate_online",
        "cross_user_gate_frozen",
        "transfer_gate_online",
    }:
        rows = [*target_memories, *source_memories]
        included_target_count = len(target_memories)
        included_source_count = len(source_memories)
    elif condition in {"source_only_no_gate", "source_only_gate"}:
        rows = limited_source_memories
        included_target_count = 0
        included_source_count = len(limited_source_memories)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return rows, {
        "condition": condition,
        "target_user": target_user,
        "source_users": source_users,
        "target_memory_count": included_target_count,
        "available_source_memory_count": len(source_memories),
        "included_source_memory_count": included_source_count,
        "total_memory_count": len(rows),
        "target_memories_per_family": target_memories_per_family,
        "source_memories_per_family": source_memories_per_family,
        "include_unscoped": include_unscoped,
    }


def _latest_report(round_dir: Path) -> Path | None:
    reports = sorted(round_dir.glob("eval_report_*.json"), key=lambda path: path.stat().st_mtime)
    return reports[-1] if reports else None


def _load_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


STRESS_TASK_GOLD_SHOULD: dict[str, bool] = {
    "StressCriticalReachabilityBatterySaverTask": True,
    "StressNavigationBatterySaverBoundaryTask": False,
    "StressPublicBluetoothLeakMuteTask": True,
    "StressPrivateBluetoothBoundaryTask": False,
    "StressLateReadingDarkModeTask": True,
    "StressColorReviewDarkModeBoundaryTask": False,
    "StressFocusBlockDndTask": True,
    "StressOnCallDndBoundaryTask": False,
    "StressImminentMeetingOpenDocTask": True,
    "StressMeetingNotImminentBoundaryTask": False,
    "ExecutionBatteryDarkLateDocTask": True,
    "ExecutionBatteryOnlyReachableNightTask": True,
    "ExecutionMuteOnlyPublicDemoTask": True,
    "ExecutionMuteBatteryCommuteTask": True,
    "ExecutionDarkOnlyBedReadingTask": True,
    "ExecutionDarkDndFocusWritingTask": True,
    "ExecutionDndOnlyDayFocusTask": True,
    "ExecutionDocOnlyImminentReviewTask": True,
    "ExecutionBatteryDocLowPowerMeetingTask": True,
    "ExecutionMuteDocPublicReviewTask": True,
    "ExecutionDarkDocNightMeetingTask": True,
    "ExecutionBatteryDndFocusLowTask": True,
    "ExecutionMuteDarkQuietNightTask": True,
    "ExecutionMuteDndWorkshopTask": True,
    "ExecutionTripleQuietLowNightTask": True,
    "ExecutionTripleFocusLowNightTask": True,
    "ExecutionTripleMeetingLowNightTask": True,
    "ExecutionTriplePublicMeetingLowTask": True,
    "ExecutionTripleWorkshopNightTask": True,
    "ExecutionAllButDndIncidentPrepTask": True,
}


def _task_base_name(task_name: str | None) -> str:
    return str(task_name or "").split("@", 1)[0]


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return None


def _build_gold_should_map(
    memories: list[dict[str, Any]],
    *,
    target_user: str,
) -> dict[str, bool]:
    gold_by_task: dict[str, bool] = {}
    for memory in memories:
        labels = memory.get("labels", {}) if isinstance(memory.get("labels"), dict) else {}
        task_name = labels.get("task_name")
        if not task_name:
            continue
        task_base = _task_base_name(str(task_name))
        target_profile = labels.get("transfer_target_profile")
        if target_profile == target_user:
            gold_should = _bool_or_none(labels.get("transfer_target_should_act"))
        elif labels.get("profile_id") == target_user:
            gold_should = _bool_or_none(labels.get("gold_should"))
        else:
            continue
        if gold_should is not None:
            gold_by_task[task_base] = gold_should

    for task_name, gold_should in STRESS_TASK_GOLD_SHOULD.items():
        gold_by_task.setdefault(task_name, gold_should)
    return gold_by_task


def _load_memrl_plan(round_dir: Path, task_name: str) -> dict[str, Any] | None:
    plan_path = round_dir / task_name / "memrl_plan.json"
    if not plan_path.exists():
        return None
    try:
        return json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _annotate_decision_metrics(
    report: dict[str, Any] | None,
    *,
    round_dir: Path,
    gold_should_by_task: dict[str, bool],
    report_path: Path | None = None,
) -> dict[str, Any]:
    metrics = {
        "memrl_decision_tasks_with_labels": 0,
        "memrl_decision_correct": 0,
        "memrl_decision_accuracy": 0.0,
        "memrl_decision_score_mismatch_count": 0,
    }
    if not report:
        return metrics

    tasks = report.get("tasks_with_results") or []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task_name = str(item.get("task_name") or "")
        task_base = _task_base_name(task_name)
        gold_should = gold_should_by_task.get(task_base)
        plan = _load_memrl_plan(round_dir, task_name) if task_name else None
        decision = plan.get("decision", {}) if isinstance(plan, dict) else {}
        candidate = plan.get("candidate", {}) if isinstance(plan, dict) else {}
        predicted_should = _bool_or_none(decision.get("should_intervene"))

        item["memrl_gold_should"] = gold_should
        item["memrl_should_intervene"] = predicted_should
        item["memrl_decision_correct"] = (
            predicted_should == gold_should
            if predicted_should is not None and gold_should is not None
            else None
        )
        item["memrl_candidate_task"] = candidate.get("proactive_task") if isinstance(candidate, dict) else None

        if item["memrl_decision_correct"] is None:
            continue
        metrics["memrl_decision_tasks_with_labels"] += 1
        if item["memrl_decision_correct"]:
            metrics["memrl_decision_correct"] += 1
        score_success = float(item.get("score", 0.0) or 0.0) > 0.0
        if score_success != bool(item["memrl_decision_correct"]):
            metrics["memrl_decision_score_mismatch_count"] += 1

    labeled = metrics["memrl_decision_tasks_with_labels"]
    metrics["memrl_decision_accuracy"] = (
        metrics["memrl_decision_correct"] / labeled if labeled else 0.0
    )
    summary = report.setdefault("summary", {})
    summary.update(metrics)
    if report_path is not None:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _report_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {
            "assigned": 0,
            "with_results": 0,
            "successful": 0,
            "no_results": 0,
            "success_rate": 0.0,
            "decision_labeled": 0,
            "decision_correct": 0,
            "decision_accuracy": 0.0,
            "decision_score_mismatches": 0,
        }
    summary = report.get("summary", report)
    assigned = int(summary.get("total_tasks_assigned", 0) or 0)
    with_results = int(summary.get("total_tasks_with_results", 0) or 0)
    successful = int(summary.get("successful_tasks", 0) or 0)
    no_results = int(summary.get("total_tasks_with_no_results", 0) or 0)
    if "overall_success_rate" in summary:
        success_rate = float(summary.get("overall_success_rate") or 0.0)
    else:
        success_rate = successful / with_results if with_results else 0.0
    decision_labeled = int(summary.get("memrl_decision_tasks_with_labels", 0) or 0)
    decision_correct = int(summary.get("memrl_decision_correct", 0) or 0)
    decision_accuracy = float(summary.get("memrl_decision_accuracy", 0.0) or 0.0)
    decision_score_mismatches = int(summary.get("memrl_decision_score_mismatch_count", 0) or 0)
    return {
        "assigned": assigned,
        "with_results": with_results,
        "successful": successful,
        "no_results": no_results,
        "success_rate": success_rate,
        "decision_labeled": decision_labeled,
        "decision_correct": decision_correct,
        "decision_accuracy": decision_accuracy,
        "decision_score_mismatches": decision_score_mismatches,
    }


def _memory_count(state_dir: Path) -> int:
    snapshot = state_dir / "memrl_snapshot.jsonl"
    if not snapshot.exists():
        return 0
    return sum(1 for line in snapshot.read_text(encoding="utf-8").splitlines() if line.strip())


def _copy_state_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)


def _condition_env(condition: str) -> dict[str, str]:
    common_retrieval = {
        "KNOWU_MEMRL_USE_COMPOSITE_COMPONENT_SHORTCUT": "false",
        "KNOWU_MEMRL_USE_DIRECT_SHORTCUTS": "false",
        "KNOWU_ROUTINE_PENALIZE_UNNECESSARY_ASK": "true",
    }
    if condition == "no_memory":
        return {
            **common_retrieval,
            "KNOWU_MEMRL_USE_MEMORY": "false",
            "KNOWU_MEMRL_FREEZE_UPDATES": "true",
            "KNOWU_MEMRL_TRANSFER_GATE_MODE": "no_transfer_gate",
            "KNOWU_MEMRL_DISABLE_TRANSFER_GATE": "true",
            "KNOWU_MEMRL_APPEND_RUNTIME_EPISODES": "false",
        }
    if condition == "same_user_only":
        return {**common_retrieval, "KNOWU_MEMRL_FREEZE_UPDATES": "true"}
    if condition == "same_user_online":
        return {
            **common_retrieval,
            "KNOWU_MEMRL_FREEZE_UPDATES": "false",
            "KNOWU_MEMRL_TRANSFER_GATE_MODE": "no_transfer_gate",
            "KNOWU_MEMRL_DISABLE_TRANSFER_GATE": "true",
            "KNOWU_MEMRL_APPEND_RUNTIME_EPISODES": "false",
        }
    if condition == "cross_user_no_gate":
        return {
            **common_retrieval,
            "KNOWU_MEMRL_FREEZE_UPDATES": "true",
            "KNOWU_MEMRL_TRANSFER_GATE_MODE": "no_transfer_gate",
            "KNOWU_MEMRL_DISABLE_TRANSFER_GATE": "true",
        }
    if condition == "cross_user_no_gate_online":
        return {
            **common_retrieval,
            "KNOWU_MEMRL_FREEZE_UPDATES": "false",
            "KNOWU_MEMRL_TRANSFER_GATE_MODE": "no_transfer_gate",
            "KNOWU_MEMRL_DISABLE_TRANSFER_GATE": "true",
            "KNOWU_MEMRL_APPEND_RUNTIME_EPISODES": "false",
        }
    if condition == "cross_user_gate_frozen":
        return {
            **common_retrieval,
            "KNOWU_MEMRL_FREEZE_UPDATES": "true",
            "KNOWU_MEMRL_TRANSFER_GATE_MODE": "default",
            "KNOWU_MEMRL_DISABLE_TRANSFER_GATE": "false",
        }
    if condition == "transfer_gate_online":
        return {
            **common_retrieval,
            "KNOWU_MEMRL_FREEZE_UPDATES": "false",
            "KNOWU_MEMRL_TRANSFER_GATE_MODE": "default",
            "KNOWU_MEMRL_DISABLE_TRANSFER_GATE": "false",
            "KNOWU_MEMRL_APPEND_RUNTIME_EPISODES": "false",
        }
    if condition == "source_only_no_gate":
        return {
            **common_retrieval,
            "KNOWU_MEMRL_FREEZE_UPDATES": "false",
            "KNOWU_MEMRL_TRANSFER_GATE_MODE": "no_transfer_gate",
            "KNOWU_MEMRL_DISABLE_TRANSFER_GATE": "true",
            "KNOWU_MEMRL_APPEND_RUNTIME_EPISODES": "false",
        }
    if condition == "source_only_gate":
        return {
            **common_retrieval,
            "KNOWU_MEMRL_FREEZE_UPDATES": "false",
            "KNOWU_MEMRL_TRANSFER_GATE_MODE": "default",
            "KNOWU_MEMRL_DISABLE_TRANSFER_GATE": "false",
            "KNOWU_MEMRL_APPEND_RUNTIME_EPISODES": "false",
        }
    raise ValueError(f"Unknown condition: {condition}")


def _condition_memrl_use_memory(condition: str) -> bool:
    return condition != "no_memory"


def _request_json(url: str, *, method: str, timeout: float) -> Any:
    request = Request(url, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend request failed: {url} -> HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach backend: {url} -> {exc}") from exc
    return json.loads(body) if body else None


def _switch_url(
    aw_host: str,
    *,
    user_log_mode: str,
    rag_top_k: int,
    rag_backend: str,
    user_log_source: str,
) -> str:
    params = urlencode(
        {
            "target_family": "knowu_bench",
            "user_log_mode": user_log_mode,
            "rag_top_k": rag_top_k,
            "rag_backend": rag_backend,
            "user_log_source": user_log_source,
        }
    )
    return f"{aw_host.rstrip('/')}/suite_family/switch?{params}"


def _expected_tasks(task_arg: str) -> list[str]:
    tasks = _parse_csv(task_arg)
    if len(tasks) == 1 and tasks[0].upper() == "ALL":
        return []
    return tasks


def _ensure_backend_tasks(
    *,
    aw_host: str,
    expected_tasks: list[str],
    user_log_mode: str,
    user_log_source: str,
    rag_top_k: int,
    rag_backend: str,
    timeout: float,
    reload_backend: bool,
) -> None:
    if not expected_tasks:
        return

    def available_tasks() -> set[str]:
        task_list = _request_json(f"{aw_host.rstrip('/')}/task/list", method="GET", timeout=timeout)
        return {str(item.get("name")) for item in task_list or [] if isinstance(item, dict)}

    available = available_tasks()
    missing = [task for task in expected_tasks if task not in available]
    if missing and reload_backend:
        force_top_k = rag_top_k + 1 if rag_top_k < 10000 else rag_top_k - 1
        for top_k in (force_top_k, rag_top_k):
            _request_json(
                _switch_url(
                    aw_host,
                    user_log_mode=user_log_mode,
                    rag_top_k=top_k,
                    rag_backend=rag_backend,
                    user_log_source=user_log_source,
                ),
                method="POST",
                timeout=timeout,
            )
        available = available_tasks()
        missing = [task for task in expected_tasks if task not in available]

    if missing:
        visible = ", ".join(sorted(name for name in available if "Task@" in name)[:12])
        raise RuntimeError(
            "Backend is missing requested task(s): "
            f"{', '.join(missing)}. Restart/rebuild the backend container or copy the current "
            f"task definitions into it. Sample visible profile tasks: {visible or '<none>'}"
        )


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "round",
        "assigned",
        "with_results",
        "successful",
        "no_results",
        "success_rate",
        "decision_labeled",
        "decision_correct",
        "decision_accuracy",
        "decision_score_mismatches",
        "memory_count_before",
        "memory_count_after",
        "bootstrap_memory_count",
        "target_memory_count",
        "included_source_memory_count",
        "available_source_memory_count",
        "target_memories_per_family",
        "source_memories_per_family",
        "source_users",
        "memrl_use_memory",
        "transfer_gate",
        "freeze_updates",
        "report_path",
        "run_log",
    ]
    with (output_root / "cross_user_transfer_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "cross_user_transfer_summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the cross-user MemRL transfer experiment without changing the existing "
            "KnowU evaluation pipeline. The default conditions preserve the existing "
            "same-user, cross-user no-gate, and transfer-gate entries; optional source-only "
            "and no-memory entries can be requested with --conditions."
        )
    )
    parser.add_argument("--target-user", required=True, help="Current user/profile to evaluate.")
    parser.add_argument(
        "--source-users",
        default=None,
        help="Comma-separated source profiles. Defaults to every profile except --target-user.",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
        help=f"Comma-separated conditions to run. Choices: {', '.join(CONDITIONS)}.",
    )
    parser.add_argument(
        "--target-memories-per-family",
        type=int,
        default=1,
        help=(
            "How many same-profile memories to keep per routine family. Use 0 for pure "
            "cross-user cold start or -1 for all same-profile memories."
        ),
    )
    parser.add_argument(
        "--source-memories-per-family",
        type=int,
        default=-1,
        help=(
            "How many source-profile memories to keep per source profile and routine family "
            "for source-only conditions. Use -1 for all source memories. Existing mixed "
            "conditions keep their previous source-memory behavior."
        ),
    )
    parser.add_argument(
        "--include-unscoped-memories",
        action="store_true",
        help="Keep memories whose profile or routine family cannot be inferred.",
    )
    parser.add_argument("--rounds", type=int, default=1, help="Repeated eval rounds per condition.")
    parser.add_argument("--output-root", default=None, help="Root directory for generated bootstraps and runs.")
    parser.add_argument(
        "--bootstrap",
        default=None,
        help="Base MemRL bootstrap jsonl. Defaults to KNOWU_MEMRL_BOOTSTRAP or the profile-task bundle.",
    )
    parser.add_argument("--agent-type", default="memrl_knowu")
    parser.add_argument("--task", default="ALL")
    parser.add_argument("--task-tags", default="routine")
    parser.add_argument("--aw-host", default="http://localhost:6800")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "qwen3.5-397b-a17b"))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "https://az.gptplus5.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY"))
    parser.add_argument("--max-round", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--user-log-mode",
        choices=["all", "rag", "profile"],
        default="all",
        help=(
            "User context exposed to the agent. Defaults to all historical activity logs; "
            "use profile only for the explicit no-log ablation."
        ),
    )
    parser.add_argument("--user-log-source", choices=["clean", "noise"], default="clean")
    parser.add_argument("--rag-top-k", type=int, default=10)
    parser.add_argument("--rag-backend", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--backend-reload-timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-backend-reload",
        action="store_true",
        help="Skip /suite_family/switch registry reload when requested tasks are missing.",
    )
    parser.add_argument("--step-wait-time", type=float, default=1.0)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the output root before running.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed round directories and continue from the latest saved memory state.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write bootstraps and print commands only.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args passed to `uv run mw eval`.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")

    repo = _repo_root()
    bootstrap = Path(
        args.bootstrap
        or os.getenv("KNOWU_MEMRL_BOOTSTRAP", "")
        or repo / "artifacts" / "memrl_knowu_profile_task_train_bundle" / "memrl_episodes.jsonl"
    )
    if not bootstrap.is_absolute():
        bootstrap = repo / bootstrap

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(
        args.output_root
        or repo / "artifacts" / f"memrl_cross_user_transfer_{args.target_user}_{timestamp}"
    )
    if not output_root.is_absolute():
        output_root = repo / output_root
    if args.reset and output_root.exists() and not args.dry_run:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_memories = _load_jsonl(bootstrap)
    profiles = _available_profiles(all_memories)
    source_users = _parse_csv(args.source_users)
    if not source_users:
        source_users = [profile for profile in profiles if profile != args.target_user]
    conditions = _parse_csv(args.conditions)
    unknown = [condition for condition in conditions if condition not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown condition(s): {', '.join(unknown)}")

    extra_args = _parse_extra_args(args.extra_args)
    expected_tasks = _expected_tasks(args.task)
    if not args.dry_run:
        _ensure_backend_tasks(
            aw_host=args.aw_host,
            expected_tasks=expected_tasks,
            user_log_mode=args.user_log_mode,
            user_log_source=args.user_log_source,
            rag_top_k=args.rag_top_k,
            rag_backend=args.rag_backend,
            timeout=args.backend_reload_timeout,
            reload_backend=not args.no_backend_reload,
        )
    rows: list[dict[str, Any]] = []

    experiment_config = {
        "target_user": args.target_user,
        "source_users": source_users,
        "conditions": conditions,
        "target_memories_per_family": args.target_memories_per_family,
        "source_memories_per_family": args.source_memories_per_family,
        "include_unscoped_memories": args.include_unscoped_memories,
        "rounds": args.rounds,
        "base_bootstrap": str(bootstrap),
        "available_profiles": profiles,
    }
    (output_root / "cross_user_transfer_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[cross-user-transfer] output_root={output_root}")
    print(f"[cross-user-transfer] base_bootstrap={bootstrap}")
    print(f"[cross-user-transfer] target_user={args.target_user}")
    print(f"[cross-user-transfer] source_users={','.join(source_users) if source_users else 'none'}")
    print(f"[cross-user-transfer] conditions={','.join(conditions)}")

    for condition in conditions:
        condition_bootstrap, bootstrap_meta = _build_condition_bootstrap(
            condition=condition,
            all_memories=all_memories,
            target_user=args.target_user,
            source_users=source_users,
            target_memories_per_family=args.target_memories_per_family,
            source_memories_per_family=args.source_memories_per_family,
            include_unscoped=args.include_unscoped_memories,
        )
        bootstrap_path = output_root / "bootstraps" / f"{condition}.jsonl"
        _write_jsonl(bootstrap_path, condition_bootstrap)
        (bootstrap_path.with_suffix(".meta.json")).write_text(
            json.dumps(bootstrap_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        gold_should_by_task = _build_gold_should_map(
            condition_bootstrap,
            target_user=args.target_user,
        )
        print(
            "[cross-user-transfer] {condition}: bootstrap={total} target={target} source={source}".format(
                condition=condition,
                total=bootstrap_meta["total_memory_count"],
                target=bootstrap_meta["target_memory_count"],
                source=bootstrap_meta["included_source_memory_count"],
            )
        )

        condition_root = output_root / condition
        state_dir = condition_root / "memory_state"
        if state_dir.exists() and not args.dry_run and not args.resume:
            shutil.rmtree(state_dir)

        for round_idx in range(1, args.rounds + 1):
            round_dir = condition_root / f"round_{round_idx:02d}"
            run_log = round_dir / "run.log"
            memory_before = _memory_count(state_dir)
            resume_report_path = _latest_report(round_dir) if args.resume else None
            resume_state_dir = round_dir / "memory_state_after"
            if args.resume and resume_report_path is not None:
                report = _load_report(resume_report_path)
                _annotate_decision_metrics(
                    report,
                    round_dir=round_dir,
                    gold_should_by_task=gold_should_by_task,
                    report_path=resume_report_path if not args.dry_run else None,
                )
                return_code = 0
                if resume_state_dir.exists() and not args.dry_run:
                    _copy_state_dir(resume_state_dir, state_dir)
                summary = _report_summary(report)
                memory_after = _memory_count(state_dir)
                env_meta = _condition_env(condition)
                row = {
                    "condition": condition,
                    "round": round_idx,
                    **summary,
                    "memory_count_before": memory_before,
                    "memory_count_after": memory_after,
                    "bootstrap_memory_count": bootstrap_meta["total_memory_count"],
                    "target_memory_count": bootstrap_meta["target_memory_count"],
                    "included_source_memory_count": bootstrap_meta["included_source_memory_count"],
                    "available_source_memory_count": bootstrap_meta["available_source_memory_count"],
                    "target_memories_per_family": args.target_memories_per_family,
                    "source_memories_per_family": args.source_memories_per_family,
                    "source_users": ",".join(source_users),
                    "memrl_use_memory": str(_condition_memrl_use_memory(condition)).lower(),
                    "transfer_gate": env_meta.get("KNOWU_MEMRL_TRANSFER_GATE_MODE", "default"),
                    "freeze_updates": env_meta.get("KNOWU_MEMRL_FREEZE_UPDATES", "true"),
                    "report_path": str(resume_report_path),
                    "run_log": str(run_log),
                }
                if args.dry_run:
                    print(
                        "[cross-user-transfer] {condition} round {round}: would resume "
                        "success={successful}/{with_results} rate={rate:.1%} memory={before}->{after} rc={rc}".format(
                            condition=condition,
                            round=round_idx,
                            successful=summary["successful"],
                            with_results=summary["with_results"],
                            rate=summary["success_rate"],
                            before=memory_before,
                            after=memory_after,
                            rc=return_code,
                        )
                    )
                    continue
                rows.append(row)
                _write_summary(output_root, rows)
                print(
                    "[cross-user-transfer] {condition} round {round}: resumed "
                    "success={successful}/{with_results} rate={rate:.1%} memory={before}->{after} rc={rc}".format(
                        condition=condition,
                        round=round_idx,
                        successful=summary["successful"],
                        with_results=summary["with_results"],
                        rate=summary["success_rate"],
                        before=memory_before,
                        after=memory_after,
                        rc=return_code,
                    )
                )
                continue
            command = [
                "uv",
                "run",
                "mw",
                "eval",
                "--agent-type",
                args.agent_type,
                "--task",
                args.task,
                "--task-tags",
                args.task_tags,
                "--user",
                args.target_user,
                "--aw-host",
                args.aw_host,
                "--model-name",
                args.model_name,
                "--llm-base-url",
                args.llm_base_url,
                "--max-round",
                str(args.max_round),
                "--max-concurrency",
                str(args.max_concurrency),
                "--enable-user-interaction",
                "--user-log-mode",
                args.user_log_mode,
                "--user-log-source",
                args.user_log_source,
                "--rag-top-k",
                str(args.rag_top_k),
                "--rag-backend",
                args.rag_backend,
                "--step-wait-time",
                str(args.step_wait_time),
                "--output",
                str(round_dir),
            ]
            command.append(
                "--memrl-use-memory"
                if _condition_memrl_use_memory(condition)
                else "--no-memrl-use-memory"
            )
            if args.api_key:
                command.extend(["--api-key", args.api_key])
            command.extend(extra_args)

            env = os.environ.copy()
            env["KNOWU_MEMRL_BOOTSTRAP"] = str(bootstrap_path)
            env["KNOWU_MEMRL_STATE_DIR"] = str(state_dir)
            env.update(_condition_env(condition))

            print(
                f"[cross-user-transfer] condition={condition} round={round_idx}/{args.rounds}: "
                + _display_command(command)
            )
            if args.dry_run:
                continue

            round_dir.mkdir(parents=True, exist_ok=True)
            with run_log.open("w", encoding="utf-8", errors="replace") as log:
                proc = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            return_code = proc.returncode
            report_path = _latest_report(round_dir)
            report = _load_report(report_path)
            _annotate_decision_metrics(
                report,
                round_dir=round_dir,
                gold_should_by_task=gold_should_by_task,
                report_path=report_path,
            )

            summary = _report_summary(report)
            memory_after = _memory_count(state_dir)
            if not args.dry_run:
                _copy_state_dir(state_dir, round_dir / "memory_state_after")
            env_meta = _condition_env(condition)
            row = {
                "condition": condition,
                "round": round_idx,
                **summary,
                "memory_count_before": memory_before,
                "memory_count_after": memory_after,
                "bootstrap_memory_count": bootstrap_meta["total_memory_count"],
                "target_memory_count": bootstrap_meta["target_memory_count"],
                "included_source_memory_count": bootstrap_meta["included_source_memory_count"],
                "available_source_memory_count": bootstrap_meta["available_source_memory_count"],
                "target_memories_per_family": args.target_memories_per_family,
                "source_memories_per_family": args.source_memories_per_family,
                "source_users": ",".join(source_users),
                "memrl_use_memory": str(_condition_memrl_use_memory(condition)).lower(),
                "transfer_gate": env_meta.get("KNOWU_MEMRL_TRANSFER_GATE_MODE", "default"),
                "freeze_updates": env_meta.get("KNOWU_MEMRL_FREEZE_UPDATES", "true"),
                "report_path": str(report_path or ""),
                "run_log": str(run_log),
            }
            rows.append(row)
            _write_summary(output_root, rows)
            print(
                "[cross-user-transfer] {condition} round {round}: success={successful}/{with_results} "
                "rate={rate:.1%} memory={before}->{after} rc={rc}".format(
                    condition=condition,
                    round=round_idx,
                    successful=summary["successful"],
                    with_results=summary["with_results"],
                    rate=summary["success_rate"],
                    before=memory_before,
                    after=memory_after,
                    rc=return_code,
                )
            )
            if return_code != 0:
                print(f"[cross-user-transfer] failed; see {run_log}", file=sys.stderr)
                return return_code

    print(f"[cross-user-transfer] summary_csv={output_root / 'cross_user_transfer_summary.csv'}")
    print(f"[cross-user-transfer] summary_json={output_root / 'cross_user_transfer_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
