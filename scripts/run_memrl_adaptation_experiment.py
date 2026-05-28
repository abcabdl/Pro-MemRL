from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_extra_args(values: list[str] | None) -> list[str]:
    if not values:
        return []
    if values and values[0] == "--":
        return values[1:]
    return values


ABLATION_CHOICES = (
    "no_transfer_gate",
    "similarity_only",
    "no_q_update",
    "single_phase",
)


def _normalize_ablations(values: list[str] | None) -> list[str]:
    ablations: list[str] = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip().lower().replace("-", "_")
            if not item or item == "none":
                continue
            if item not in ABLATION_CHOICES:
                raise ValueError(
                    f"Unknown ablation {item!r}; expected one of: {', '.join(ABLATION_CHOICES)}"
                )
            if item not in ablations:
                ablations.append(item)
    return ablations


def _apply_ablation_env(env: dict[str, str], ablations: list[str]) -> None:
    if "no_transfer_gate" in ablations:
        env["KNOWU_MEMRL_TRANSFER_GATE_MODE"] = "no_transfer_gate"
        env["KNOWU_MEMRL_DISABLE_TRANSFER_GATE"] = "true"
    if "similarity_only" in ablations:
        env["KNOWU_MEMRL_RETRIEVAL_SCORING"] = "similarity_only"
    if "no_q_update" in ablations:
        env["KNOWU_MEMRL_DISABLE_Q_UPDATES"] = "true"
    if "single_phase" in ablations:
        env["KNOWU_MEMRL_PROMPTING_MODE"] = "single_phase"


def _copy_seed_memory(seed: Path, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    if seed.is_dir():
        snapshot = seed / "memrl_snapshot.jsonl"
        meta = seed / "memrl_meta.json"
    else:
        snapshot = seed
        meta = seed.with_name("memrl_meta.json")
    if not snapshot.exists():
        raise FileNotFoundError(f"Seed memory snapshot does not exist: {snapshot}")
    shutil.copy2(snapshot, state_dir / "memrl_snapshot.jsonl")
    if meta.exists():
        shutil.copy2(meta, state_dir / "memrl_meta.json")


def _memory_count(state_dir: Path) -> int:
    snapshot = state_dir / "memrl_snapshot.jsonl"
    if not snapshot.exists():
        return 0
    return sum(1 for line in snapshot.read_text(encoding="utf-8").splitlines() if line.strip())


def _tasks_arg(round_cfg: dict[str, Any], default_task: str) -> str:
    tasks = round_cfg.get("tasks", round_cfg.get("task", default_task))
    if isinstance(tasks, list):
        if not tasks:
            return default_task
        return ",".join(str(task) for task in tasks)
    return str(tasks or default_task)


def _scan_result_files(round_dir: Path) -> tuple[dict[str, float], dict[str, str]]:
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for result_path in sorted(round_dir.glob("*@*/result.txt")):
        task_name = result_path.parent.name
        text = result_path.read_text(encoding="utf-8", errors="replace").strip()
        match = re.search(r"score:\s*([0-9.]+)", text)
        scores[task_name] = float(match.group(1)) if match else 0.0
        reasons[task_name] = text
    return scores, reasons


def _write_outputs(output_root: Path, rows: list[dict[str, Any]], reports: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "round",
        "name",
        "phase",
        "user",
        "task",
        "assigned",
        "with_results",
        "successful",
        "success_rate",
        "delta_successful_vs_prev",
        "improved_vs_prev",
        "regressed_vs_prev",
        "same_vs_prev",
        "memory_count_before",
        "memory_count_after",
        "condition",
        "memrl_use_memory",
        "freeze_updates",
        "ablations",
        "run_log",
    ]
    with (output_root / "adaptation_rounds_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_at": datetime.now().isoformat(),
        "rounds": rows,
        "task_scores_by_round": [report["scores"] for report in reports],
        "task_reasons_by_round": [report["reasons"] for report in reports],
    }
    (output_root / "adaptation_rounds_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a scheduled MemRL online-adaptation experiment without changing the existing "
            "KnowU eval pipeline. Each schedule round can use different tasks/users."
        )
    )
    parser.add_argument("--schedule", required=True, help="JSON schedule file.")
    parser.add_argument("--output-root", default=None, help="Root directory for all round outputs.")
    parser.add_argument("--memory-state-dir", default=None, help="Shared MemRL state directory.")
    parser.add_argument("--seed-memory", default=None, help="Snapshot file or state dir copied before round 1.")
    parser.add_argument("--reset-memory", action="store_true", help="Delete memory-state dir before starting.")
    parser.add_argument(
        "--condition",
        choices=["online", "static_memory", "no_memory"],
        default="online",
        help="Baseline condition: online updates, frozen memory, or no memory retrieval.",
    )
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        help=(
            "Optional MemRL ablation(s), comma-separated or repeated: "
            "no_transfer_gate, similarity_only, no_q_update, single_phase."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        default=None,
        help="Initial MemRL bootstrap jsonl. Defaults to KNOWU_MEMRL_BOOTSTRAP or the profile-task bundle.",
    )
    parser.add_argument("--agent-type", default="memrl_knowu")
    parser.add_argument("--task", default="ALL", help="Default task argument when a round omits tasks.")
    parser.add_argument("--task-tags", default="routine")
    parser.add_argument("--aw-host", default="http://localhost:6800")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "qwen3.5-397b-a17b"))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "https://az.gptplus5.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY"))
    parser.add_argument("--max-round", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--user-log-mode", choices=["all", "rag", "profile"], default="profile")
    parser.add_argument("--user-log-source", choices=["clean", "noise"], default="clean")
    parser.add_argument("--rag-top-k", type=int, default=10)
    parser.add_argument("--rag-backend", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--step-wait-time", type=float, default=1.0)
    parser.add_argument(
        "--fail-on-missing-results",
        action="store_true",
        help="Return a non-zero status if an explicit task list produces fewer result.txt files than assigned tasks.",
    )
    parser.add_argument(
        "--delegate-unknown-family-abstain",
        action="store_true",
        help=(
            "When MemRL cannot map a task to a task_family and would abstain, delegate to "
            "the normal GUI executor instead of finishing immediately."
        ),
    )
    parser.add_argument(
        "--live-profile-routine-hints",
        action="store_true",
        help=(
            "Use the currently active profile snapshot as a routine hint before retrieval. "
            "This is intended for same-user profile-drift experiments."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args passed to `uv run mw eval`.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = _repo_root()
    schedule_path = Path(args.schedule)
    if not schedule_path.is_absolute():
        schedule_path = repo / schedule_path
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    rounds = schedule.get("rounds", [])
    if not rounds:
        raise ValueError(f"Schedule has no rounds: {schedule_path}")

    schedule_name = str(schedule.get("name") or schedule_path.stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or repo / "artifacts" / f"memrl_adaptation_{schedule_name}_{args.condition}_{timestamp}")
    if not output_root.is_absolute():
        output_root = repo / output_root
    state_dir = Path(args.memory_state_dir or output_root / "memory_state")
    if not state_dir.is_absolute():
        state_dir = repo / state_dir

    bootstrap = Path(
        args.bootstrap
        or os.getenv("KNOWU_MEMRL_BOOTSTRAP", "")
        or repo / "artifacts" / "memrl_knowu_profile_task_train_bundle" / "memrl_episodes.jsonl"
    )
    if not bootstrap.is_absolute():
        bootstrap = repo / bootstrap

    if args.reset_memory and state_dir.exists() and not args.dry_run:
        shutil.rmtree(state_dir)
    if args.seed_memory and not args.dry_run:
        seed = Path(args.seed_memory)
        if not seed.is_absolute():
            seed = repo / seed
        _copy_seed_memory(seed, state_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "schedule_used.json").write_text(json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    prev_scores: dict[str, float] | None = None
    prev_successful: int | None = None
    extra_args = _parse_extra_args(args.extra_args)
    ablations = _normalize_ablations(args.ablation)

    print(f"[adaptation] schedule={schedule_path}")
    print(f"[adaptation] output_root={output_root}")
    print(f"[adaptation] memory_state_dir={state_dir}")
    print(f"[adaptation] condition={args.condition}")
    print(f"[adaptation] ablations={','.join(ablations) if ablations else 'none'}")

    for idx, round_cfg in enumerate(rounds, start=1):
        round_name = str(round_cfg.get("name") or f"round_{idx:02d}")
        phase = str(round_cfg.get("phase") or "")
        user = str(round_cfg.get("user") or schedule.get("user") or "")
        task_arg = _tasks_arg(round_cfg, args.task)
        task_tags = str(round_cfg.get("task_tags", args.task_tags))
        round_dir = output_root / f"round_{idx:02d}_{round_name}"
        run_log = round_dir / "run.log"
        round_dir.mkdir(parents=True, exist_ok=True)

        if args.condition == "no_memory":
            memrl_use_memory = False
            freeze_updates = True
        elif args.condition == "static_memory":
            memrl_use_memory = True
            freeze_updates = True
        else:
            memrl_use_memory = bool(round_cfg.get("memrl_use_memory", True))
            freeze_updates = bool(round_cfg.get("freeze_updates", False))

        memory_before = _memory_count(state_dir)
        command = [
            "uv",
            "run",
            "mw",
            "eval",
            "--agent-type",
            args.agent_type,
            "--task",
            task_arg,
            "--task-tags",
            task_tags,
            "--aw-host",
            str(round_cfg.get("aw_host", args.aw_host)),
            "--model-name",
            str(round_cfg.get("model_name", args.model_name)),
            "--llm-base-url",
            str(round_cfg.get("llm_base_url", args.llm_base_url)),
            "--max-round",
            str(round_cfg.get("max_round", args.max_round)),
            "--max-concurrency",
            str(round_cfg.get("max_concurrency", args.max_concurrency)),
            "--enable-user-interaction",
            "--user-log-mode",
            str(round_cfg.get("user_log_mode", args.user_log_mode)),
            "--user-log-source",
            str(round_cfg.get("user_log_source", args.user_log_source)),
            "--rag-top-k",
            str(round_cfg.get("rag_top_k", args.rag_top_k)),
            "--rag-backend",
            str(round_cfg.get("rag_backend", args.rag_backend)),
            "--step-wait-time",
            str(round_cfg.get("step_wait_time", args.step_wait_time)),
            "--output",
            str(round_dir),
        ]
        if user:
            command.extend(["--user", user])
        if args.api_key:
            command.extend(["--api-key", args.api_key])
        command.append("--memrl-use-memory" if memrl_use_memory else "--no-memrl-use-memory")
        command.extend(extra_args)

        env = os.environ.copy()
        env["KNOWU_MEMRL_STATE_DIR"] = str(state_dir)
        env["KNOWU_MEMRL_BOOTSTRAP"] = str(bootstrap)
        env["KNOWU_MEMRL_FREEZE_UPDATES"] = "true" if freeze_updates else "false"
        _apply_ablation_env(env, ablations)
        if args.delegate_unknown_family_abstain:
            env["KNOWU_MEMRL_DELEGATE_UNKNOWN_FAMILY_ABSTAIN"] = "true"
        if args.live_profile_routine_hints:
            env["KNOWU_MEMRL_USE_LIVE_PROFILE_ROUTINE_HINTS"] = "true"

        print(f"[adaptation] round {idx}/{len(rounds)} phase={phase} user={user} tasks={task_arg}")
        print(f"[adaptation] command: {' '.join(command)}")
        if args.dry_run:
            return_code = 0
        else:
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

        scores, reasons = _scan_result_files(round_dir)
        assigned = len([task for task in task_arg.split(",") if task.strip()]) if task_arg.upper() != "ALL" else len(scores)
        with_results = len(scores)
        successful = sum(1 for score in scores.values() if score > 0.99)
        success_rate = successful / with_results if with_results else 0.0
        if prev_scores is None:
            improved = regressed = same = 0
        else:
            common = sorted(set(prev_scores) & set(scores))
            improved = sum(1 for task in common if scores[task] > prev_scores[task])
            regressed = sum(1 for task in common if scores[task] < prev_scores[task])
            same = sum(1 for task in common if scores[task] == prev_scores[task])

        memory_after = _memory_count(state_dir)
        row = {
            "round": idx,
            "name": round_name,
            "phase": phase,
            "user": user,
            "task": task_arg,
            "assigned": assigned,
            "with_results": with_results,
            "successful": successful,
            "success_rate": success_rate,
            "delta_successful_vs_prev": "" if prev_successful is None else successful - prev_successful,
            "improved_vs_prev": improved,
            "regressed_vs_prev": regressed,
            "same_vs_prev": same,
            "memory_count_before": memory_before,
            "memory_count_after": memory_after,
            "condition": args.condition,
            "memrl_use_memory": memrl_use_memory,
            "freeze_updates": freeze_updates,
            "ablations": ",".join(ablations),
            "run_log": str(run_log),
        }
        rows.append(row)
        reports.append({"scores": scores, "reasons": reasons})
        _write_outputs(output_root, rows, reports)
        print(
            "[adaptation] round {idx}: success={successful}/{with_results} rate={rate:.1%} "
            "memory={before}->{after} rc={rc}".format(
                idx=idx,
                successful=successful,
                with_results=with_results,
                rate=success_rate,
                before=memory_before,
                after=memory_after,
                rc=return_code,
            )
        )
        if return_code != 0:
            print(f"[adaptation] round {idx} failed; see {run_log}", file=sys.stderr)
            return return_code
        if (
            args.fail_on_missing_results
            and not args.dry_run
            and task_arg.upper() != "ALL"
            and with_results < assigned
        ):
            print(
                f"[adaptation] round {idx} missing results: {with_results}/{assigned}; see {run_log}",
                file=sys.stderr,
            )
            return 2
        prev_scores = scores
        prev_successful = successful

    print(f"[adaptation] summary_csv={output_root / 'adaptation_rounds_summary.csv'}")
    print(f"[adaptation] summary_json={output_root / 'adaptation_rounds_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
