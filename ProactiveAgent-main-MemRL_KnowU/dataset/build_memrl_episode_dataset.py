from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _event_to_observation(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": item.get("Time"),
        "event": item.get("Event"),
    }


def _infer_domain(observations: list[dict[str, Any]]) -> str:
    text = " ".join(str(item.get("event", "")).lower() for item in observations)
    coding_tokens = ("code", "vscode", ".py", ".js", "terminal", "traceback", "debug")
    writing_tokens = ("document", "draft", "outline", "essay", "blog", "markdown", "paper")
    finance_tokens = ("investment", "balance", "spreadsheet", "stock", "market", "account")
    coding_hits = sum(token in text for token in coding_tokens)
    writing_hits = sum(token in text for token in writing_tokens)
    finance_hits = sum(token in text for token in finance_tokens)
    if coding_hits >= max(writing_hits, finance_hits) and coding_hits > 0:
        return "coding"
    if writing_hits >= max(coding_hits, finance_hits) and writing_hits > 0:
        return "writing"
    if finance_hits > 0:
        return "finance"
    return "general"


def _build_intent_text(observations: list[dict[str, Any]], teacher: dict[str, Any]) -> str:
    event_text = " | ".join(
        " ".join(
            part for part in [str(item.get("time") or ""), str(item.get("event") or "")]
            if part
        )
        for item in observations[-8:]
    )
    summary_bits = [
        _normalize_text(teacher.get("purpose")),
        _normalize_text(teacher.get("thoughts")),
    ]
    return " | ".join(part for part in [event_text, *summary_bits] if part)


def _candidate_from_teacher(teacher: dict[str, Any], *, allow_null: bool = True) -> dict[str, Any]:
    candidate = {
        "purpose": _normalize_text(teacher.get("purpose")),
        "proactive_task": _normalize_text(teacher.get("proactive_task")),
        "response": _normalize_text(teacher.get("response")),
        "operation": None,
    }
    if allow_null:
        return candidate
    if candidate["proactive_task"] is None and candidate["response"] is None:
        return {"purpose": candidate["purpose"], "proactive_task": None, "response": None, "operation": None}
    return candidate


def _judge_votes_to_text(judge_votes: dict[str, Any]) -> str:
    parts: list[str] = []
    for model_name, payload in (judge_votes or {}).items():
        if not isinstance(payload, dict):
            continue
        thought_need = _normalize_text(payload.get("thought_need"))
        thought_accept = _normalize_text(payload.get("thought_accept"))
        if thought_need:
            parts.append(f"{model_name}: {thought_need}")
        if thought_accept:
            parts.append(f"{model_name}: {thought_accept}")
    return " | ".join(parts)


def load_agent_trainset(path: Path) -> list[dict]:
    return list(json.loads(path.read_text(encoding="utf-8")))


def load_rdc_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def derive_simulation_target(row: dict, candidate: dict) -> dict:
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    judge_votes = row.get("judge_votes", {}) if isinstance(row.get("judge_votes"), dict) else {}
    y_accept = labels.get("y_accept")
    if y_accept == 1:
        acceptance = "accept"
    elif y_accept == 0:
        acceptance = "dismiss"
    elif candidate.get("proactive_task"):
        acceptance = "ignore"
    else:
        acceptance = "accept"
    confidence = teacher.get("q_accept")
    if confidence is None:
        confidence = 0.8 if acceptance == "accept" else 0.55
    return {
        "acceptance": acceptance,
        "acceptance_confidence": float(confidence),
        "flow_impact": "low" if labels.get("y_need") == 1 else "high",
        "relevance": "high" if labels.get("y_need") == 1 else "low",
        "timing": "good" if labels.get("y_need") == 1 else "interruptive",
        "reasoning": _judge_votes_to_text(judge_votes) or _normalize_text(teacher.get("thoughts")) or "",
    }


def derive_decision_target(row: dict, simulation: dict, candidate: dict) -> dict:
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    gold = row.get("gold_decision", {}) if isinstance(row.get("gold_decision"), dict) else {}
    gold_should = gold.get("should_intervene")
    if gold_should is None:
        gold_should = labels.get("y_need")
    gold_level = gold.get("commitment_level")
    if gold_level is None:
        if int(gold_should or 0) <= 0:
            gold_level = 0
        elif candidate.get("response") and candidate.get("proactive_task"):
            gold_level = 2
        else:
            gold_level = 1
    return {
        "should_intervene": bool(int(gold_should or 0)),
        "commitment_level": int(gold_level or 0),
        "risk": "high" if simulation.get("acceptance") in {"dismiss", "annoyed"} else "low",
        "reason": simulation.get("reasoning") or _normalize_text(row.get("category")) or "",
    }


def compute_initial_reward(row: dict, decision: dict) -> float:
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    gold = row.get("gold_decision", {}) if isinstance(row.get("gold_decision"), dict) else {}
    source = str(row.get("_source", ""))
    gold_should = gold.get("should_intervene")
    if gold_should is None:
        gold_should = labels.get("y_need")
    gold_level = gold.get("commitment_level")
    if gold_level is None:
        gold_level = 0 if int(gold_should or 0) <= 0 else int(decision.get("commitment_level", 1) or 1)

    pred_should = int(bool(decision.get("should_intervene")))
    pred_level = int(decision.get("commitment_level", 0) or 0)
    gold_should = int(gold_should or 0)
    gold_level = int(gold_level or 0)

    if source == "teacher_stage1_weak":
        return 0.25
    if gold_should == pred_should and gold_level == pred_level:
        return 1.0
    if gold_should == pred_should:
        return 0.4
    if pred_should == 1 and gold_should == 0:
        return -1.0
    if pred_should == 0 and gold_should == 1:
        return -0.8

    teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    if labels.get("y_need") is None and float(teacher.get("q_need", 0.0) or 0.0) >= 0.8:
        return 0.5
    if pred_should == 0 and float(teacher.get("q_need", 0.0) or 0.0) <= 0.2:
        return 0.7
    return 0.0


def build_episode_from_judge_row(row: dict) -> dict | None:
    teacher = row.get("teacher")
    if not isinstance(teacher, dict):
        return None
    observations = [_event_to_observation(item) for item in row.get("obs", []) if isinstance(item, dict)]
    candidate = _candidate_from_teacher(teacher)
    simulation = derive_simulation_target(row, candidate)
    decision = derive_decision_target(row, simulation, candidate)
    if not decision["should_intervene"]:
        candidate["proactive_task"] = None
        candidate["response"] = None
        candidate["operation"] = "nop"
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    gold = row.get("gold_decision", {}) if isinstance(row.get("gold_decision"), dict) else {}
    now = utc_now()
    built = {
        "memory_id": f"judge-{row.get('sample_id', uuid.uuid4().hex)}",
        "sample_id": str(row.get("sample_id", "")),
        "source": "judge_stage2",
        "domain": _infer_domain(observations),
        "observations": observations,
        "intent_text": _build_intent_text(observations, teacher),
        "candidate": candidate,
        "simulation": simulation,
        "decision": decision,
        "labels": {
            "y_need": labels.get("y_need"),
            "y_accept": labels.get("y_accept"),
            "gold_should": gold.get("should_intervene", labels.get("y_need")),
            "gold_level": gold.get("commitment_level", decision["commitment_level"]),
            "q_need": teacher.get("q_need"),
            "q_accept": teacher.get("q_accept"),
        },
        "reward": 0.0,
        "q_value": 0.0,
        "q_visits": 0,
        "created_at": now,
        "updated_at": now,
    }
    built["reward"] = compute_initial_reward(row, built["decision"])
    built["q_value"] = built["reward"]
    return built


def build_episode_from_teacher_row(row: dict) -> dict | None:
    teacher = row.get("teacher")
    if not isinstance(teacher, dict):
        return None
    y_need_pred = int(teacher.get("y_need_pred", 0) or 0)
    has_candidate = _normalize_text(teacher.get("proactive_task")) is not None
    if y_need_pred == 1 and not has_candidate:
        return None
    observations = [_event_to_observation(item) for item in row.get("obs", []) if isinstance(item, dict)]
    candidate = _candidate_from_teacher(teacher)
    if y_need_pred == 0:
        candidate["proactive_task"] = None
        candidate["response"] = None
        candidate["operation"] = "nop"
    simulation = {
        "acceptance": "accept" if y_need_pred == 1 else "ignore",
        "acceptance_confidence": float(teacher.get("q_accept", 0.25) or 0.25),
        "flow_impact": "low" if y_need_pred == 1 else "high",
        "relevance": "high" if y_need_pred == 1 else "low",
        "timing": "good" if y_need_pred == 1 else "interruptive",
        "reasoning": _normalize_text(teacher.get("thoughts")) or "",
    }
    decision = {
        "should_intervene": bool(y_need_pred),
        "commitment_level": 2 if y_need_pred == 1 else 0,
        "risk": "medium",
        "reason": _normalize_text(teacher.get("thoughts")) or "",
    }
    weak_row = dict(row)
    weak_row["_source"] = "teacher_stage1_weak"
    now = utc_now()
    built = {
        "memory_id": f"teacher-{row.get('sample_id', uuid.uuid4().hex)}",
        "sample_id": str(row.get("sample_id", "")),
        "source": "teacher_stage1_weak",
        "domain": _infer_domain(observations),
        "observations": observations,
        "intent_text": _build_intent_text(observations, teacher),
        "candidate": candidate,
        "simulation": simulation,
        "decision": decision,
        "labels": {
            "y_need": y_need_pred,
            "y_accept": 1 if y_need_pred == 1 else None,
            "gold_should": None,
            "gold_level": None,
            "q_need": teacher.get("q_need"),
            "q_accept": teacher.get("q_accept"),
        },
        "reward": compute_initial_reward(weak_row, decision),
        "q_value": 0.25,
        "q_visits": 0,
        "created_at": now,
        "updated_at": now,
    }
    return built


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def build_generation_only_from_agent_trainset(row: dict) -> dict | None:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None
    user_turn = next((item for item in conversations if item.get("role") == "user"), None)
    assistant_turn = next((item for item in reversed(conversations) if item.get("role") == "assistant"), None)
    if not isinstance(user_turn, dict) or not isinstance(assistant_turn, dict):
        return None
    user_payload = _extract_json_from_text(str(user_turn.get("content", "")))
    assistant_payload = _extract_json_from_text(str(assistant_turn.get("content", "")))
    if not isinstance(user_payload, dict) or not isinstance(assistant_payload, dict):
        return None
    observations = [
        _event_to_observation(item)
        for item in user_payload.get("Observations", [])
        if isinstance(item, dict)
    ]
    candidate = {
        "purpose": _normalize_text(assistant_payload.get("Purpose")),
        "proactive_task": _normalize_text(assistant_payload.get("Proactive Task")),
        "response": _normalize_text(assistant_payload.get("Response")),
        "operation": None,
    }
    if candidate["proactive_task"] is None and candidate["response"] is None:
        return None
    return {
        "sample_id": str(uuid.uuid4()),
        "source": "agent_trainset_coldstart",
        "domain": _infer_domain(observations),
        "observations": observations,
        "intent_text": _build_intent_text(observations, {"purpose": assistant_payload.get("Purpose"), "thoughts": assistant_payload.get("Thoughts")}),
        "candidate": candidate,
    }


def _make_generation_row(episode: dict) -> dict[str, Any]:
    return {
        "sample_id": episode["sample_id"],
        "source": episode["source"],
        "domain": episode["domain"],
        "observations": episode["observations"],
        "intent_text": episode["intent_text"],
        "memory_generation_prior": {
            "preferred_level": episode["decision"]["commitment_level"],
            "positive_patterns": [episode["candidate"].get("proactive_task")] if episode["candidate"].get("proactive_task") else [],
            "negative_patterns": [episode["decision"].get("reason")] if not episode["decision"]["should_intervene"] else [],
            "avoid_patterns": [episode["candidate"].get("response")] if not episode["decision"]["should_intervene"] and episode["candidate"].get("response") else [],
        },
        "target_candidate": episode["candidate"],
        "target_decision": episode["decision"],
    }


def _make_simulation_row(episode: dict) -> dict[str, Any]:
    return {
        "sample_id": episode["sample_id"],
        "source": episode["source"],
        "domain": episode["domain"],
        "observations": episode["observations"],
        "candidate": episode["candidate"],
        "memory_simulation_prior": {
            "historical_accept_rate": 1.0 if episode["simulation"]["acceptance"] == "accept" else 0.0,
            "historical_dismiss_rate": 1.0 if episode["simulation"]["acceptance"] == "dismiss" else 0.0,
            "historical_annoy_rate": 1.0 if episode["simulation"]["acceptance"] == "annoyed" else 0.0,
        },
        "target_simulation": episode["simulation"],
    }


def _make_decision_row(episode: dict) -> dict[str, Any]:
    return {
        "sample_id": episode["sample_id"],
        "source": episode["source"],
        "domain": episode["domain"],
        "observations": episode["observations"],
        "candidate": episode["candidate"],
        "simulation": episode["simulation"],
        "memory_decision_prior": {
            "intervene_memory_value": episode["q_value"] if episode["decision"]["should_intervene"] else 0.0,
            "abstain_memory_value": episode["q_value"] if not episode["decision"]["should_intervene"] else 0.0,
            "memory_level_mode": episode["decision"]["commitment_level"],
        },
        "target_decision": episode["decision"],
        "reward": episode["reward"],
    }


def write_episode_bundle(
    out_dir: Path,
    episodes: list[dict],
    generation_rows: list[dict],
    simulation_rows: list[dict],
    decision_rows: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memrl_episodes.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in episodes),
        encoding="utf-8",
    )
    (out_dir / "memrl_generation_train.json").write_text(
        json.dumps(generation_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "memrl_simulation_train.json").write_text(
        json.dumps(simulation_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "memrl_decision_train.json").write_text(
        json.dumps(decision_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    domains: dict[str, int] = {}
    sources: dict[str, int] = {}
    for episode in episodes:
        domains[episode["domain"]] = domains.get(episode["domain"], 0) + 1
        sources[episode["source"]] = sources.get(episode["source"], 0) + 1
    dataset_info = {
        "episode_count": len(episodes),
        "generation_count": len(generation_rows),
        "simulation_count": len(simulation_rows),
        "decision_count": len(decision_rows),
        "domains": domains,
        "sources": sources,
        "created_at": utc_now(),
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        **dataset_info,
        "positive_episodes": sum(1 for item in episodes if item["decision"]["should_intervene"]),
        "abstain_episodes": sum(1 for item in episodes if not item["decision"]["should_intervene"]),
        "avg_reward": round(sum(float(item["reward"]) for item in episodes) / max(len(episodes), 1), 6),
    }
    (out_dir / "memrl_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-source", type=Path, required=True)
    parser.add_argument("--teacher-source", type=Path, required=True)
    parser.add_argument("--agent-trainset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-judge-confidence", type=int, default=2)
    parser.add_argument("--include-agent-trainset-coldstart", default="true")
    args = parser.parse_args()

    judge_rows = load_rdc_jsonl(args.judge_source)
    teacher_rows = load_rdc_jsonl(args.teacher_source)
    agent_rows = load_agent_trainset(args.agent_trainset)

    episodes: list[dict] = []
    generation_rows: list[dict] = []
    simulation_rows: list[dict] = []
    decision_rows: list[dict] = []

    for row in judge_rows:
        labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
        vote = labels.get("need_vote", {}) if isinstance(labels.get("need_vote"), dict) else {}
        confidence = max(int(vote.get("positive_votes", 0) or 0), int(vote.get("negative_votes", 0) or 0))
        if confidence < args.min_judge_confidence:
            continue
        episode = build_episode_from_judge_row(row)
        if episode is None:
            continue
        episodes.append(episode)
        generation_rows.append(_make_generation_row(episode))
        simulation_rows.append(_make_simulation_row(episode))
        decision_rows.append(_make_decision_row(episode))

    existing_ids = {item["sample_id"] for item in episodes}
    for row in teacher_rows:
        if str(row.get("sample_id", "")) in existing_ids:
            continue
        episode = build_episode_from_teacher_row(row)
        if episode is None:
            continue
        episodes.append(episode)
        generation_rows.append(_make_generation_row(episode))
        simulation_rows.append(_make_simulation_row(episode))
        decision_rows.append(_make_decision_row(episode))

    if _as_bool(args.include_agent_trainset_coldstart):
        for row in agent_rows:
            generated = build_generation_only_from_agent_trainset(row)
            if generated is None:
                continue
            generation_rows.append(
                {
                    **generated,
                    "memory_generation_prior": {
                        "preferred_level": 2,
                        "positive_patterns": [generated["candidate"].get("proactive_task")],
                        "negative_patterns": [],
                        "avoid_patterns": [],
                    },
                    "target_candidate": generated["candidate"],
                }
            )

    write_episode_bundle(args.output_dir, episodes, generation_rows, simulation_rows, decision_rows)


if __name__ == "__main__":
    main()
