from __future__ import annotations

from typing import Any


def summarize_recent_interactions(records: list[dict[str, Any]], limit: int = 10) -> str:
    if not records:
        return "No prior interaction history."

    recent = list(records[-max(1, int(limit)) :])
    accepted = [item for item in recent if item.get("reaction") == "accept"]
    ignored = [item for item in recent if item.get("reaction") == "ignore"]
    rejected = [item for item in recent if item.get("reaction") in {"dismiss", "annoyed"}]

    lines = [
        f"Recent interventions: {len(recent)}",
        f"Accepted: {len(accepted)}",
        f"Ignored: {len(ignored)}",
        f"Dismissed/Annoyed: {len(rejected)}",
    ]
    if accepted:
        lines.append(f"Most recent accepted context: {accepted[-1].get('context_signals', {})}")
    if rejected:
        lines.append(f"Most recent rejected context: {rejected[-1].get('context_signals', {})}")
    return "\n".join(lines)
