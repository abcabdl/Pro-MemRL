"""Runtime operation resolution for strict L0/L1/L2 intervention levels."""

from __future__ import annotations


def normalize_operation(value: str | None) -> str:
    if value is None:
        return "nop"
    text = str(value).strip().lower()
    if text in {"", "null", "none"}:
        return "nop"
    return text


def resolve_operation(level: int, candidate_operation: str | None) -> str:
    if int(level) <= 0:
        return "nop"
    if int(level) == 1:
        return "notify_only"
    op = normalize_operation(candidate_operation)
    return "notify_only" if op == "nop" else op

