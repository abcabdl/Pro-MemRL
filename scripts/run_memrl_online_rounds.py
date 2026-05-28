from __future__ import annotations

import argparse
import csv
import json
import os
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


def _latest_report(round_dir: Path) -> Path | None:
    reports = sorted(round_dir.glob("eval_report_*.json"), key=lambda path: path.stat().st_mtime)
    return reports[-1] if reports else None


def _load_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _report_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {
            "assigned": 0,
            "with_results": 0,
            "successful": 0,
            "no_results": 0,
            "success_rate": 0.0,
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
    return {
        "assigned": assigned,
        "with_results": with_results,
        "successful": successful,
        "no_results": no_results,
        "success_rate": success_rate,
    }


def _task_scores(report: dict[str, Any] | None) -> dict[str, float]:
    if not report:
        return {}
    return {
        str(item.get("task_name")): float(item.get("score", 0.0) or 0.0)
        for item in report.get("tasks_with_results", [])
        if item.get("task_name")
    }


def _memory_count(state_dir: Path) -> int:
    snapshot = state_dir / "memrl_snapshot.jsonl"
    if not snapshot.exists():
        return 0
    return sum(1 for line in snapshot.read_text(encoding="utf-8").splitlines() if line.strip())


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


def _write_outputs(
    *,
    output_root: Path,
    rows: list[dict[str, Any]],
    reports: list[dict[str, Any] | None],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "online_memory_rounds_summary.csv"
    fieldnames = [
        "round",
        "assigned",
        "with_results",
        "successful",
        "no_results",
        "success_rate",
        "delta_successful_vs_prev",
        "improved_vs_prev",
        "regressed_vs_prev",
        "same_vs_prev",
        "memory_count_before",
        "memory_count_after",
        "ablations",
        "report_path",
        "run_log",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_at": datetime.now().isoformat(),
        "rounds": rows,
        "task_scores_by_round": [_task_scores(report) for report in reports],
    }
    (output_root / "online_memory_rounds_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated single-user KnowU MemRL evaluations while reusing and updating a "
            "separate memory snapshot between rounds."
        )
    )
    parser.add_argument("--user", required=True, help="Profile to evaluate, e.g. developer/student/grandma/user.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of full eval rounds to run.")
    parser.add_argument("--output-root", default=None, help="Root directory for all round outputs.")
    parser.add_argument(
        "--memory-state-dir",
        default=None,
        help="Shared MemRL state directory. Defaults to <output-root>/memory_state.",
    )
    parser.add_argument(
        "--bootstrap",
        default=None,
        help=(
            "Initial MemRL bootstrap jsonl or snapshot directory. Defaults to KNOWU_MEMRL_BOOTSTRAP "
            "or artifacts/memrl_knowu_profile_task_train_bundle/memrl_episodes.jsonl."
        ),
    )
    parser.add_argument(
        "--seed-memory",
        default=None,
        help="Existing memrl_snapshot.jsonl or directory to copy into the shared state before round 1.",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete the shared memory state directory before starting.",
    )
    parser.add_argument("--agent-type", default="memrl_knowu")
    parser.add_argument("--task", default="ALL")
    parser.add_argument("--task-tags", default="routine")
    parser.add_argument("--aw-host", default="http://localhost:6800")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "qwen3.5-397b-a17b"))
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("LLM_BASE_URL", "https://az.gptplus5.com/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("API_KEY"))
    parser.add_argument("--max-round", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--user-log-mode", choices=["all", "rag", "profile"], default="profile")
    parser.add_argument("--user-log-source", choices=["clean", "noise"], default="clean")
    parser.add_argument("--rag-top-k", type=int, default=10)
    parser.add_argument("--rag-backend", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--step-wait-time", type=float, default=1.0)
    parser.add_argument(
        "--freeze-updates",
        action="store_true",
        help="Run with memory retrieval enabled but do not write online updates.",
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
        "--dry-run",
        action="store_true",
        help="Print commands and write no eval reports.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to `uv run mw eval` after `--`.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")

    repo = _repo_root()
    user_tag = args.user.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or repo / "artifacts" / f"memrl_online_rounds_{user_tag}_{timestamp}")
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
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any] | None] = []
    prev_scores: dict[str, float] | None = None
    prev_successful: int | None = None
    extra_args = _parse_extra_args(args.extra_args)
    ablations = _normalize_ablations(args.ablation)

    print(f"[online-rounds] output_root={output_root}")
    print(f"[online-rounds] memory_state_dir={state_dir}")
    print(f"[online-rounds] bootstrap={bootstrap}")
    print(f"[online-rounds] freeze_updates={args.freeze_updates}")
    print(f"[online-rounds] ablations={','.join(ablations) if ablations else 'none'}")

    for round_idx in range(1, args.rounds + 1):
        round_dir = output_root / f"round_{round_idx:02d}"
        run_log = round_dir / "run.log"
        round_dir.mkdir(parents=True, exist_ok=True)
        memory_before = _memory_count(state_dir)
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
            args.user,
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
            "--memrl-use-memory",
            "--output",
            str(round_dir),
        ]
        if args.api_key:
            command.extend(["--api-key", args.api_key])
        command.extend(extra_args)

        env = os.environ.copy()
        env["KNOWU_MEMRL_STATE_DIR"] = str(state_dir)
        env["KNOWU_MEMRL_BOOTSTRAP"] = str(bootstrap)
        env["KNOWU_MEMRL_FREEZE_UPDATES"] = "true" if args.freeze_updates else "false"
        _apply_ablation_env(env, ablations)

        print(f"[online-rounds] round {round_idx}/{args.rounds}: {' '.join(command)}")
        if args.dry_run:
            report = None
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
            report = _load_report(_latest_report(round_dir))

        summary = _report_summary(report)
        scores = _task_scores(report)
        if prev_scores is None:
            improved = regressed = same = 0
        else:
            common = sorted(set(prev_scores) & set(scores))
            improved = sum(1 for task in common if scores[task] > prev_scores[task])
            regressed = sum(1 for task in common if scores[task] < prev_scores[task])
            same = sum(1 for task in common if scores[task] == prev_scores[task])

        memory_after = _memory_count(state_dir)
        row = {
            "round": round_idx,
            **summary,
            "delta_successful_vs_prev": (
                "" if prev_successful is None else summary["successful"] - prev_successful
            ),
            "improved_vs_prev": improved,
            "regressed_vs_prev": regressed,
            "same_vs_prev": same,
            "memory_count_before": memory_before,
            "memory_count_after": memory_after,
            "ablations": ",".join(ablations),
            "report_path": str(_latest_report(round_dir) or ""),
            "run_log": str(run_log),
        }
        rows.append(row)
        reports.append(report)
        _write_outputs(output_root=output_root, rows=rows, reports=reports)
        print(
            "[online-rounds] round {round}: success={successful}/{with_results} "
            "rate={rate:.1%} memory={before}->{after} rc={rc}".format(
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
            print(f"[online-rounds] round {round_idx} failed; see {run_log}", file=sys.stderr)
            return return_code
        prev_scores = scores
        prev_successful = summary["successful"]

    print(f"[online-rounds] summary_csv={output_root / 'online_memory_rounds_summary.csv'}")
    print(f"[online-rounds] summary_json={output_root / 'online_memory_rounds_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
