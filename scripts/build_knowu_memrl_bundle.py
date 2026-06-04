from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from knowu_bench.memrl.adapter import (  # noqa: E402
    DEFAULT_BUNDLE_DIR,
    build_knowu_profile_memory_bundle,
    build_knowu_profile_task_matrix_bundle,
    build_knowu_routine_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MemRL bundle for KnowU routine tasks.")
    parser.add_argument(
        "--source",
        choices=("profile-task-matrix", "profile-habits", "task-oracle"),
        default="profile-task-matrix",
        help=(
            "profile-task-matrix builds non-test profile x routine-family memories. "
            "profile-habits builds one memory per profile habit. "
            "task-oracle builds from KnowU routine task expectations for sanity checks only."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / DEFAULT_BUNDLE_DIR,
        help="Output directory for memrl_episodes.jsonl and derived train files.",
    )
    parser.add_argument(
        "--task-set-path",
        type=str,
        default=None,
        help="Optional KnowU task definitions directory for --source task-oracle.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Optional user_profile directory for --source profile-habits.",
    )
    parser.add_argument(
        "--max-log-items",
        type=int,
        default=24,
        help="Recent user-log entries to keep in each memory observation.",
    )
    parser.add_argument(
        "--users",
        type=str,
        default="",
        help="Optional comma-separated profile filter, e.g. developer,student.",
    )
    parser.add_argument(
        "--negatives-per-profile",
        type=int,
        default=4,
        help="Synthetic background abstain examples per profile for --source profile-habits.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=256,
        help=(
            "Requested memory count for --source profile-task-matrix. "
            "The builder emits 3 or 4 diverse scenarios per profile x routine cell."
        ),
    )
    parser.add_argument(
        "--no-transfer-stress",
        action="store_true",
        help="Do not append the atomic cross-profile transfer stress memories to profile-task-matrix bundles.",
    )
    args = parser.parse_args()
    users = {item.strip() for item in args.users.split(",") if item.strip()} or None
    if args.source == "task-oracle":
        info = build_knowu_routine_bundle(
            output_dir=args.output_dir,
            task_set_path=args.task_set_path,
            max_log_items=args.max_log_items,
            users=users,
        )
    elif args.source == "profile-habits":
        info = build_knowu_profile_memory_bundle(
            output_dir=args.output_dir,
            profile_dir=args.profile_dir,
            max_log_items=args.max_log_items,
            users=users,
            negatives_per_profile=args.negatives_per_profile,
        )
    else:
        info = build_knowu_profile_task_matrix_bundle(
            output_dir=args.output_dir,
            profile_dir=args.profile_dir,
            max_log_items=args.max_log_items,
            users=users,
            target_count=args.target_count,
            include_transfer_stress=not args.no_transfer_stress,
        )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
