from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import tenacity
from openai import AsyncOpenAI

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

ROOT_DIR = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT_DIR.parent / "ProactiveAgent-main"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from eval.memrl_eval_prompts import PHASE_A_SYSTEM  # noqa: E402
from proactive_pipeline import (  # noqa: E402
    infer_signals,
    normalize_observations,
    observations_to_text,
    parse_json_payload,
)

try:
    from dataset.build_rdc_filtered_trainset import (  # type: ignore  # noqa: E402
        JUDGE_SYSTEM_PROMPT,
        build_judge_user_payload,
        majority_binary_vote,
        normalize_nullable_text,
        stable_json_dumps,
        to_binary,
    )
except Exception:  # noqa: BLE001
    JUDGE_SYSTEM_PROMPT = """<Role>
You are an impartial evaluator for proactive assistant data curation.
</Role>

<Task>
Given user observations and one proposed proactive intervention, output:
- y_need: whether proactive help is needed now.
- y_accept: if a proactive intervention is proposed, whether the user would
  accept this intervention now.
</Task>

<Format>
Respond strictly in JSON:
{
  "thought_need": "brief reasoning for y_need",
  "y_need": 0,
  "thought_accept": "brief reasoning for y_accept",
  "y_accept": 0
}
</Format>

<Rules>
- y_need and y_accept must be 0 or 1.
- If "Proposed Task" is null, set y_accept=0 because no intervention occurred.
- Evaluate y_accept only for non-null proposed interventions.
- Minimize unnecessary interruption and judge only with the given context.
</Rules>
"""

    def normalize_nullable_text(value: Any) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).split())
        return text or None

    def to_binary(value: Any, default: int = 0) -> int:
        try:
            return 1 if int(value) else 0
        except (TypeError, ValueError):
            return default

    def majority_binary_vote(votes: list[int], tie_break: int) -> tuple[int, dict[str, int]]:
        positives = sum(1 for vote in votes if int(vote) == 1)
        negatives = len(votes) - positives
        if positives > negatives:
            return 1, {"positive_votes": positives, "negative_votes": negatives, "tie_break_used": 0}
        if negatives > positives:
            return 0, {"positive_votes": positives, "negative_votes": negatives, "tie_break_used": 0}
        return int(tie_break), {"positive_votes": positives, "negative_votes": negatives, "tie_break_used": 1}

    def stable_json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def build_judge_user_payload(
        *,
        obs: list[dict[str, Any]],
        recent_observation_count: int,
        history_highlight_count: int,
        proposed_task: str | None,
        proposed_response: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {
            "Instruction": "Judge whether help is needed and whether this intervention will be accepted.",
            "Proposed Task": proposed_task,
            "Proposed Response": proposed_response,
            "Observations (Time Ascending)": obs[-recent_observation_count:],
        }, {
            "payload_mode": "fallback_recent_only",
            "total_observations": len(obs),
            "recent_observations": min(len(obs), recent_observation_count),
            "older_observations": max(0, len(obs) - recent_observation_count),
            "history_highlights": history_highlight_count,
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_model_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    return {
        "purpose": normalize_text(teacher.get("purpose")),
        "proactive_task": normalize_text(teacher.get("proactive_task")),
        "response": normalize_text(teacher.get("response")),
        "operation": normalize_text(teacher.get("operation")),
    }


def build_formal_judge_payload(
    row: dict[str, Any],
    *,
    recent_observation_count: int,
    history_highlight_count: int,
    models: list[str],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    obs = row.get("obs", []) if isinstance(row.get("obs"), list) else []
    payload, meta = build_judge_user_payload(
        obs=obs,
        recent_observation_count=recent_observation_count,
        history_highlight_count=history_highlight_count,
        proposed_task=normalize_nullable_text(teacher.get("proactive_task")),
        proposed_response=normalize_nullable_text(teacher.get("response")),
    )
    signature_material = {
        "version": 1,
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "user_payload": payload,
        "judge_models": list(models),
    }
    signature = hashlib.sha1(stable_json_dumps(signature_material).encode("utf-8")).hexdigest()
    return payload, meta, signature


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("judge response is not a JSON object")
    return payload


def normalize_vote(payload: dict[str, Any], *, raw_response: str) -> dict[str, Any]:
    return {
        "y_need": to_binary(payload.get("y_need"), default=0),
        "thought_need": normalize_text(payload.get("thought_need")) or "",
        "y_accept": to_binary(payload.get("y_accept"), default=0),
        "thought_accept": normalize_text(payload.get("thought_accept")) or "",
        "raw_response": raw_response,
    }


def vote_summary(votes: dict[str, dict[str, Any]], *, candidate_exists: bool, teacher: dict[str, Any], models: list[str]) -> dict[str, Any]:
    used = {name: vote for name, vote in votes.items() if isinstance(vote, dict) and "error" not in vote}
    need_votes = [int(vote.get("y_need", 0)) for vote in used.values()]
    accept_votes = [int(vote.get("y_accept", 0)) for vote in used.values()]
    need_tie_break = int(float(teacher.get("q_need", 0.0) or 0.0) >= 0.5)
    accept_tie_break = 0 if not candidate_exists else int(float(teacher.get("q_accept", 0.0) or 0.0) >= 0.5)
    y_need, need_vote_meta = majority_binary_vote(need_votes, tie_break=need_tie_break)
    y_accept, accept_vote_meta = majority_binary_vote(accept_votes, tie_break=accept_tie_break)
    if not candidate_exists:
        y_accept = 0
    return {
        "y_need": int(y_need),
        "y_accept": int(y_accept),
        "y_star": int(bool(y_need and y_accept and candidate_exists)),
        "category": "",
        "source": "llm_judge_majority_vote",
        "need_vote": need_vote_meta,
        "accept_vote": accept_vote_meta,
        "judge_models_requested": list(models),
        "judge_models_used": list(used.keys()),
    }


async def call_judge(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: int,
    retries: int,
) -> dict[str, Any]:
    async for attempt in tenacity.AsyncRetrying(
        stop=tenacity.stop_after_attempt(max(1, retries)),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    ):
        with attempt:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                ),
                timeout=max(30, timeout_seconds),
            )
            content = response.choices[0].message.content or "{}"
            return normalize_vote(extract_json(content), raw_response=content)
    raise RuntimeError("unreachable")


async def call_phase_a_signals(
    *,
    client: AsyncOpenAI,
    model: str,
    observations: list[dict[str, Any]],
    timeout_seconds: int,
    retries: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    normalized_obs = normalize_observations(observations)
    messages = [
        {"role": "system", "content": PHASE_A_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "signal_prediction",
                    "observations": observations_to_text(normalized_obs),
                },
                ensure_ascii=False,
            ),
        },
    ]
    async for attempt in tenacity.AsyncRetrying(
        stop=tenacity.stop_after_attempt(max(1, retries)),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    ):
        with attempt:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                ),
                timeout=max(30, timeout_seconds),
            )
            raw = response.choices[0].message.content or "{}"
            payload = parse_json_payload(raw)
            predicted = payload.get("signals", payload.get("Signals", {}))
            signals = infer_signals(normalized_obs, predicted if isinstance(predicted, dict) else {})
            return signals, {"raw_response": raw, "parsed": payload}
    raise RuntimeError("unreachable")


async def judge_row(
    *,
    row: dict[str, Any],
    client: AsyncOpenAI,
    models: list[str],
    semaphore: asyncio.Semaphore,
    timeout_seconds: int,
    retries: int,
    recent_observation_count: int,
    history_highlight_count: int,
) -> dict[str, Any]:
    payload, prompt_meta, signature = build_formal_judge_payload(
        row,
        recent_observation_count=recent_observation_count,
        history_highlight_count=history_highlight_count,
        models=models,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    async def one(model: str) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            try:
                signals, phase_a_output = await call_phase_a_signals(
                    client=client,
                    model=model,
                    observations=row.get("obs", []) if isinstance(row.get("obs"), list) else [],
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                )
                vote = await call_judge(
                    client=client,
                    model=model,
                    messages=messages,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                )
                vote["signals"] = signals
                vote["raw_phase_a_output"] = phase_a_output
            except Exception as exc:  # noqa: BLE001 - persist per-model failures for resume/debug
                vote = {"error": str(exc)}
            return model, vote

    pairs = await asyncio.gather(*(one(model) for model in models))
    votes = dict(pairs)
    candidate = candidate_from_row(row)
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    labels = vote_summary(
        votes,
        candidate_exists=bool(candidate.get("proactive_task") or candidate.get("response")),
        teacher=teacher,
        models=models,
    )
    labels["category"] = row.get("category", "")
    used_votes = [vote for vote in votes.values() if isinstance(vote, dict) and "error" not in vote]
    signal_keys = ("flow", "stuck", "need", "accept", "risk", "uncertainty", "progress", "rejection_memory")
    phase_a_signals = {
        key: round(
            sum(float((vote.get("signals") or {}).get(key, 0.0)) for vote in used_votes) / max(len(used_votes), 1),
            4,
        )
        for key in signal_keys
    }
    return {
        **row,
        "judge_votes": votes,
        "labels": labels,
        "phase_a": {
            "signals": phase_a_signals,
            "prompt_source": "eval.memrl_eval_prompts.PHASE_A_SYSTEM",
            "model_signals": {
                model: vote.get("signals", {})
                for model, vote in votes.items()
                if isinstance(vote, dict) and "error" not in vote
            },
        },
        "judge_prompt_meta": prompt_meta,
        "judge_input_signature": signature,
    }


async def amain(args: argparse.Namespace) -> None:
    models = parse_model_list(args.models)
    if not models:
        raise ValueError("--models must contain at least one model")
    if not args.api_key:
        raise ValueError("Missing API key. Use --api-key or RDC_API_KEY/OPENAI_API_KEY.")
    rows = load_jsonl(args.input)
    if args.sample_size is not None:
        rng = random.Random(int(args.sample_seed))
        sample_size = max(0, min(int(args.sample_size), len(rows)))
        rows = rng.sample(rows, sample_size)
        rows.sort(key=lambda item: str(item.get("sample_id", "")))
    if args.max_samples is not None:
        rows = rows[: max(0, int(args.max_samples))]
    completed: set[str] = set()
    if args.resume and args.output.exists():
        for row in load_jsonl(args.output):
            completed.add(str(row.get("sample_id")))
    pending = [row for row in rows if str(row.get("sample_id")) not in completed]
    print(f"input_rows={len(rows)} completed={len(completed)} pending={len(pending)} output={args.output}")
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    for row in pending:
        queue.put_nowait(row)
    worker_count = min(max(1, int(args.concurrency)), max(1, len(pending)))
    for _ in range(worker_count):
        queue.put_nowait(None)
    lock = asyncio.Lock()
    processed = 0
    progress = tqdm(total=len(pending), desc="judging teacher memory", unit="row", dynamic_ncols=True) if tqdm and pending else None

    async def worker() -> None:
        nonlocal processed
        while True:
            row = await queue.get()
            if row is None:
                queue.task_done()
                return
            judged = await judge_row(
                row=row,
                client=client,
                models=models,
                semaphore=semaphore,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
                recent_observation_count=args.recent_observations,
                history_highlight_count=args.history_highlights,
            )
            async with lock:
                append_jsonl(args.output, judged)
                processed += 1
                if progress is not None:
                    labels = judged.get("labels", {}) if isinstance(judged.get("labels"), dict) else {}
                    progress.update(1)
                    progress.set_postfix(
                        y_need=labels.get("y_need", "?"),
                        y_accept=labels.get("y_accept", "?"),
                    )
                if processed % max(1, int(args.log_every)) == 0 or processed == len(pending):
                    print(f"processed={processed}/{len(pending)}", flush=True)
            queue.task_done()

    try:
        if pending:
            await asyncio.gather(*(worker() for _ in range(worker_count)))
    finally:
        if progress is not None:
            progress.close()
    print(f"complete=true wrote_new={processed} output={args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge teacher-stage memory rows with multiple LLM votes.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default=os.environ.get("MEMRL_JUDGE_MODELS", "deepseek-r1,gpt-4o,gemini-2.5-pro"))
    parser.add_argument("--base-url", default=os.environ.get("RDC_API_BASE_URL", os.environ.get("OPENAI_BASE_URL")))
    parser.add_argument("--api-key", default=os.environ.get("RDC_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--recent-observations", type=int, default=24)
    parser.add_argument("--history-highlights", type=int, default=6)
    parser.add_argument("--sample-size", type=int, default=None, help="Randomly sample this many input rows before judging.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed used with --sample-size.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
