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
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(repo: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _task_name(base: str, profile_id: str) -> str:
    return base if "@" in base else f"{base}@{profile_id}"


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(command)}\n{detail}")
    return proc


def _detect_docker_container(explicit: str | None) -> str | None:
    candidates = [
        explicit,
        os.getenv("KNOWU_MEMRL_DOCKER_CONTAINER"),
        "knowu_bench_env_0",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            _run_checked(["docker", "inspect", candidate])
            return candidate
        except Exception:
            pass

    try:
        proc = _run_checked(["docker", "ps", "--filter", "publish=6800", "--format", "{{.Names}}"])
    except Exception:
        return None
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return names[0] if names else None


def _container_file_exists(container: str, path: str) -> bool:
    proc = subprocess.run(
        ["docker", "exec", container, "sh", "-lc", f"test -f {path!r}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _container_cp_to(container: str, source: Path, target_path: str) -> None:
    _run_checked(["docker", "cp", str(source), f"{container}:{target_path}"])


def _container_cp_from(container: str, source_path: str, target: Path) -> None:
    _run_checked(["docker", "cp", f"{container}:{source_path}", str(target)])


def _container_rm(container: str, path: str) -> None:
    _run_checked(["docker", "exec", container, "sh", "-lc", f"rm -f {path!r}"])


def _read_child_row(child_output: Path) -> dict[str, Any]:
    summary_path = child_output / "adaptation_rounds_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds") or []
    return dict(rounds[0]) if rounds else {}


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


def _reload_backend_registry(
    *,
    aw_host: str,
    user_log_mode: str,
    user_log_source: str,
    rag_top_k: int,
    rag_backend: str,
    timeout: float,
    expected_tasks: list[str],
) -> None:
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

    task_list = _request_json(f"{aw_host.rstrip('/')}/task/list", method="GET", timeout=timeout)
    available = {str(item.get("name")) for item in task_list or [] if isinstance(item, dict)}
    missing = [task for task in expected_tasks if task not in available]
    if missing:
        visible = ", ".join(sorted(name for name in available if "developer_drift" in name)[:8])
        raise RuntimeError(
            "Backend registry reload finished, but expected task(s) are still missing: "
            f"{', '.join(missing)}. developer_drift tasks visible: {visible or '<none>'}"
        )


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "round",
        "name",
        "phase",
        "profile_overlay",
        "tasks",
        "assigned",
        "missing_results",
        "condition",
        "success_rate",
        "successful",
        "with_results",
        "memory_count_before",
        "memory_count_after",
        "child_output",
    ]
    with (output_root / "same_user_drift_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "same_user_drift_summary.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat(), "rounds": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a same-user MemRL drift experiment by swapping an experimental profile "
            "snapshot before each round. Existing task and agent logic are not modified."
        )
    )
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--profile-id",
        default=None,
        help="Profile id used for generated task names. Defaults to schedule.profile_id, then developer_drift.",
    )
    parser.add_argument(
        "--base-log-profile",
        default="developer",
        help="Existing user log profile copied when the drift profile has no explicit log_overlay.",
    )
    parser.add_argument("--condition", choices=["online", "static_memory", "no_memory"], default="online")
    parser.add_argument("--reset-memory", action="store_true")
    parser.add_argument("--user-log-mode", choices=["all", "rag", "profile"], default="profile")
    parser.add_argument("--user-log-source", choices=["clean", "noise"], default="clean")
    parser.add_argument("--rag-top-k", type=int, default=10)
    parser.add_argument("--rag-backend", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--aw-host", default=os.getenv("AW_HOST", "http://localhost:6800"))
    parser.add_argument("--backend-reload-timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-backend-reload",
        action="store_true",
        help="Skip /suite_family/switch registry reload after copying each profile overlay.",
    )
    parser.add_argument("--docker-container", default=None, help="Docker container that serves --aw-host.")
    parser.add_argument(
        "--container-profile-dir",
        default="/app/service/src/knowu_bench/user_profile",
        help="Profile directory inside the Docker backend container.",
    )
    parser.add_argument(
        "--container-log-dir",
        default="/app/service/src/knowu_bench/user_logs",
        help="User-log directory inside the Docker backend container.",
    )
    parser.add_argument("--no-docker-sync", action="store_true", help="Do not copy overlays into Docker.")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "qwen3.5-397b-a17b"))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "https://az.gptplus5.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY"))
    parser.add_argument("--max-round", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--step-wait-time", type=float, default=1.0)
    parser.add_argument(
        "--allow-missing-results",
        action="store_true",
        help="Continue even if a round produces fewer result.txt files than assigned tasks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-active-profile", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = _repo_root()
    schedule_path = _resolve(repo, args.schedule)
    if schedule_path is None:
        raise ValueError("Missing schedule")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    rounds = schedule.get("rounds") or []
    if not rounds:
        raise ValueError(f"Schedule has no rounds: {schedule_path}")
    profile_id = str(args.profile_id or schedule.get("profile_id") or "developer_drift")

    output_root = _resolve(repo, args.output_root)
    if output_root is None:
        raise ValueError("Missing output root")
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_copy = output_root / "schedule_used.json"
    schedule_copy.write_text(json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8")

    profile_target = repo / "src" / "knowu_bench" / "user_profile" / f"{profile_id}.yaml"
    log_target = repo / "src" / "knowu_bench" / "user_logs" / f"{profile_id}.json"
    profile_backup = output_root / "_backup_profile.yaml"
    log_backup = output_root / "_backup_log.json"
    docker_container = None if args.no_docker_sync or args.dry_run else _detect_docker_container(args.docker_container)
    container_profile = (
        f"{args.container_profile_dir.rstrip('/')}/{profile_id}.yaml" if docker_container else None
    )
    container_log = f"{args.container_log_dir.rstrip('/')}/{profile_id}.json" if docker_container else None
    container_profile_backup = output_root / "_backup_container_profile.yaml"
    container_log_backup = output_root / "_backup_container_log.json"
    had_profile = profile_target.exists()
    had_log = log_target.exists()
    had_container_profile = False
    had_container_log = False
    if had_profile:
        shutil.copy2(profile_target, profile_backup)
    if had_log:
        shutil.copy2(log_target, log_backup)
    if docker_container and container_profile:
        had_container_profile = _container_file_exists(docker_container, container_profile)
        if had_container_profile:
            _container_cp_from(docker_container, container_profile, container_profile_backup)
        if container_log:
            had_container_log = _container_file_exists(docker_container, container_log)
            if had_container_log:
                _container_cp_from(docker_container, container_log, container_log_backup)
        print(f"[same-user-drift] syncing profile overlays into Docker container {docker_container}")
    elif not args.no_docker_sync and not args.dry_run:
        print("[same-user-drift] Docker backend container not found; using host files only")

    state_dir = output_root / "memory_state"
    if args.reset_memory and state_dir.exists() and not args.dry_run:
        shutil.rmtree(state_dir)

    rows: list[dict[str, Any]] = []
    temp_dir = output_root / "_round_schedules"
    temp_dir.mkdir(parents=True, exist_ok=True)
    child_runner = repo / "scripts" / "run_memrl_adaptation_experiment.py"

    try:
        for idx, round_cfg in enumerate(rounds, start=1):
            profile_overlay = _resolve(repo, round_cfg.get("profile_overlay"))
            if profile_overlay is None or not profile_overlay.exists():
                raise FileNotFoundError(f"Missing profile_overlay in round {idx}: {profile_overlay}")
            if not args.dry_run:
                shutil.copy2(profile_overlay, profile_target)
                if docker_container and container_profile:
                    _container_cp_to(docker_container, profile_overlay, container_profile)

            log_overlay = _resolve(repo, round_cfg.get("log_overlay"))
            if log_overlay is not None and log_overlay.exists() and not args.dry_run:
                shutil.copy2(log_overlay, log_target)
                if docker_container and container_log:
                    _container_cp_to(docker_container, log_overlay, container_log)
            elif not args.dry_run:
                suffix = "_noise_25pct.json" if args.user_log_source == "noise" else ".json"
                fallback_log = log_target.with_name(f"{args.base_log_profile}{suffix}")
                if fallback_log.exists() and fallback_log.resolve() != log_target.resolve():
                    shutil.copy2(fallback_log, log_target)
                    if docker_container and container_log:
                        _container_cp_to(docker_container, fallback_log, container_log)

            tasks = [_task_name(str(task), profile_id) for task in round_cfg.get("tasks", [])]
            if not args.dry_run and not args.no_backend_reload:
                print(f"[same-user-drift] reloading backend registry via {args.aw_host}")
                _reload_backend_registry(
                    aw_host=args.aw_host,
                    user_log_mode=args.user_log_mode,
                    user_log_source=args.user_log_source,
                    rag_top_k=args.rag_top_k,
                    rag_backend=args.rag_backend,
                    timeout=args.backend_reload_timeout,
                    expected_tasks=tasks,
                )

            one_round = {
                "name": f"{schedule.get('name', schedule_path.stem)}_round_{idx:02d}",
                "user": profile_id,
                "rounds": [
                    {
                        "name": str(round_cfg.get("name") or f"round_{idx:02d}"),
                        "phase": str(round_cfg.get("phase") or ""),
                        "user": profile_id,
                        "tasks": tasks,
                        "task_tags": str(round_cfg.get("task_tags") or "routine"),
                    }
                ],
            }
            round_schedule = temp_dir / f"round_{idx:02d}.json"
            round_schedule.write_text(json.dumps(one_round, ensure_ascii=False, indent=2), encoding="utf-8")
            # The child runner already creates a descriptive round directory below
            # output_root. Keep this parent short so Windows paths do not exceed
            # MAX_PATH once task names and screenshot filenames are appended.
            child_output = output_root / f"r{idx:02d}"

            cmd = [
                sys.executable,
                str(child_runner),
                "--schedule",
                str(round_schedule),
                "--output-root",
                str(child_output),
                "--memory-state-dir",
                str(state_dir),
                "--condition",
                args.condition,
                "--user-log-mode",
                args.user_log_mode,
                "--user-log-source",
                args.user_log_source,
                "--rag-top-k",
                str(args.rag_top_k),
                "--rag-backend",
                args.rag_backend,
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
                "--step-wait-time",
                str(args.step_wait_time),
            ]
            if not args.allow_missing_results:
                cmd.append("--fail-on-missing-results")
            cmd.append("--delegate-unknown-family-abstain")
            if args.api_key:
                cmd.extend(["--api-key", args.api_key])
            if args.dry_run:
                cmd.append("--dry-run")

            print(f"[same-user-drift] round {idx}/{len(rounds)} profile={profile_overlay.name} tasks={','.join(tasks)}")
            rc = subprocess.run(cmd, cwd=repo, check=False).returncode
            child_row = _read_child_row(child_output)
            row = {
                "round": idx,
                "name": one_round["rounds"][0]["name"],
                "phase": one_round["rounds"][0]["phase"],
                "profile_overlay": str(profile_overlay),
                "tasks": ",".join(tasks),
                "assigned": child_row.get("assigned", len(tasks)),
                "missing_results": (
                    ""
                    if child_row.get("with_results", "") == ""
                    else max(0, len(tasks) - int(child_row.get("with_results", 0)))
                ),
                "condition": args.condition,
                "success_rate": child_row.get("success_rate", ""),
                "successful": child_row.get("successful", ""),
                "with_results": child_row.get("with_results", ""),
                "memory_count_before": child_row.get("memory_count_before", ""),
                "memory_count_after": child_row.get("memory_count_after", ""),
                "child_output": str(child_output),
            }
            rows.append(row)
            _write_summary(output_root, rows)
            if rc != 0:
                return rc
    finally:
        if not args.keep_active_profile and not args.dry_run:
            if had_profile:
                shutil.copy2(profile_backup, profile_target)
            elif profile_target.exists():
                profile_target.unlink()
            if had_log:
                shutil.copy2(log_backup, log_target)
            elif log_target.exists():
                log_target.unlink()
            if docker_container and container_profile:
                if had_container_profile:
                    _container_cp_to(docker_container, container_profile_backup, container_profile)
                else:
                    _container_rm(docker_container, container_profile)
            if docker_container and container_log:
                if had_container_log:
                    _container_cp_to(docker_container, container_log_backup, container_log)
                else:
                    _container_rm(docker_container, container_log)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
