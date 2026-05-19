from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .formatters import (
    build_decision_context,
    build_generation_context,
    build_memory_document,
    build_observation_text,
    build_simulation_context,
)
from .index import EmbeddingIndex
from .schema import enrich_memory_schema, infer_action_family, infer_context_family


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_level(memory: dict[str, Any]) -> int:
    return int(memory.get("decision", {}).get("commitment_level", 0) or 0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _gate_trust_multiplier(memory: dict[str, Any]) -> float:
    source = str(memory.get("source", ""))
    decision = memory.get("decision", {}) or {}
    should_intervene = bool(decision.get("should_intervene", False))
    q_value = _clamp(_safe_float(memory.get("q_value", memory.get("reward", 0.0))), -1.0, 1.0)
    if source == "teacher_stage1_weak":
        if should_intervene:
            return 0.75 if q_value >= 0.3 else 0.55
        return 0.25 if q_value <= 0.3 else 0.45
    if source == "rdc_topk_scored":
        return 0.85
    return 1.0


def _value_bucket(memory: dict[str, Any]) -> str:
    existing = str(memory.get("value_bucket", "")).strip()
    if existing:
        return existing
    q_value = _clamp(_safe_float(memory.get("q_value", memory.get("reward", 0.0))), -1.0, 1.0)
    should_intervene = bool(memory.get("decision", {}).get("should_intervene", False))
    if should_intervene and q_value > 0.0:
        return "helpful_positive"
    if not should_intervene and q_value > 0.0:
        return "correct_abstain"
    if should_intervene and q_value < 0.0:
        return "bad_intervention"
    if not should_intervene and q_value < 0.0:
        return "missed_help"
    return "weak_uncertain"


def _evidence_side(memory: dict[str, Any]) -> str | None:
    bucket = _value_bucket(memory)
    if bucket in {"helpful_positive", "missed_help"}:
        return "intervene"
    if bucket in {"correct_abstain", "bad_intervention"}:
        return "abstain"
    decision = memory.get("decision", {}) or {}
    q_value = _clamp(_safe_float(memory.get("q_value", memory.get("reward", 0.0))), -1.0, 1.0)
    if q_value == 0.0:
        return None
    should_intervene = bool(decision.get("should_intervene", False))
    if should_intervene:
        return "intervene" if q_value > 0.0 else "abstain"
    return "abstain" if q_value > 0.0 else "intervene"


_PROFILE_RE = re.compile(r"(developer|student|grandma|user)")


def _memory_profile(memory: dict[str, Any]) -> str | None:
    labels = memory.get("labels", {}) if isinstance(memory.get("labels"), dict) else {}
    if labels.get("profile_id"):
        return str(labels["profile_id"])
    for key in ("sample_id", "memory_id", "intent_text"):
        match = _PROFILE_RE.search(str(memory.get(key, "") or ""))
        if match:
            return match.group(1)
    return None


def _memory_task_family(memory: dict[str, Any]) -> str | None:
    labels = memory.get("labels", {}) if isinstance(memory.get("labels"), dict) else {}
    if labels.get("task_family"):
        return str(labels["task_family"])
    for key in ("context_family", "action_family", "sample_id", "memory_id"):
        match = re.search(
            r"(?:knowu_profile_task_|knowu_|profile_task::|routine\.)([a-z0-9_]+)",
            str(memory.get(key, "") or ""),
        )
        if match:
            return match.group(1)
    return None


def _target_profile(query: str) -> str | None:
    match = re.search(r"\bprofile:(developer|student|grandma|user)\b", query)
    return match.group(1) if match else None


def _target_task_family(query: str) -> str | None:
    for pattern in (
        r"\btask_family:([a-z0-9_]+)\b",
        r"\bcontext_family:knowu_profile_task_([a-z0-9_]+)\b",
        r"\baction_family:knowu_profile_task_([a-z0-9_]+)\b",
    ):
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return None


def _transfer_gate(
    memory: dict[str, Any],
    *,
    target_profile: str | None,
    target_task_family: str | None,
) -> float:
    source_profile = _memory_profile(memory)
    source_task = _memory_task_family(memory)
    if not target_profile or not target_task_family or not source_profile or not source_task:
        return 0.25

    target_key = f"{target_profile}::{target_task_family}"
    transfer_item = (memory.get("transfer_stats", {}) or {}).get(target_key, {})

    if source_profile == target_profile and source_task == target_task_family:
        base_gate = 1.0
    elif source_profile != target_profile and source_task == target_task_family:
        base_gate = 0.35
    elif source_profile == target_profile and source_task != target_task_family:
        base_gate = 0.05
    else:
        base_gate = 0.0

    if "gate" in transfer_item:
        return _clamp(_safe_float(transfer_item.get("gate"), base_gate), 0.0, 1.0)
    return _clamp(base_gate, 0.0, 1.0)


class MemRLRetriever:
    def __init__(self, *, topk: int = 8, sim_threshold: float = 0.18) -> None:
        self.topk = topk
        self.sim_threshold = sim_threshold
        self.index = EmbeddingIndex()
        self.memories: dict[str, dict[str, Any]] = {}

    def build(self, memories: list[dict[str, Any]]) -> None:
        for item in memories:
            enrich_memory_schema(item)
        self.memories = {str(item["memory_id"]): item for item in memories}
        doc_pairs = [
            (str(item["memory_id"]), build_memory_document(item))
            for item in memories
        ]
        self.index.build(doc_pairs)

    def _base_query(self, observations: list[dict[str, Any]], signals: dict[str, Any]) -> str:
        context_family = infer_context_family(observations, signals)
        return " | ".join(
            part
            for part in [
                f"context_family:{context_family}",
                build_observation_text(observations),
                " ".join(f"{k}:{v}" for k, v in sorted((signals or {}).items())),
            ]
            if part
        )

    def _rank(self, query: str) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        target_profile = _target_profile(query)
        target_task_family = _target_task_family(query)
        for memory_id, sim in self.index.search(query, topk=len(self.memories)):
            if sim < self.sim_threshold:
                continue
            memory = self.memories[memory_id]
            q_value = _clamp(_safe_float(memory.get("q_value", 0.0)), -1.0, 1.0)
            base_score = 0.7 * sim + 0.3 * abs(q_value)
            transfer_gate = _transfer_gate(
                memory,
                target_profile=target_profile,
                target_task_family=target_task_family,
            )
            score = transfer_gate * base_score
            ranked.append(
                {
                    "memory": memory,
                    "similarity": sim,
                    "score": score,
                    "base_score": base_score,
                    "transfer_gate": transfer_gate,
                    "target_profile": target_profile,
                    "target_task_family": target_task_family,
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _rerank(
        self,
        ranked: list[dict[str, Any]],
        *,
        context_family: str,
        action_family: str | None = None,
    ) -> list[dict[str, Any]]:
        rescored: list[dict[str, Any]] = []
        for item in ranked:
            memory = item["memory"]
            adjusted_score = item["score"]
            q_value = _clamp(_safe_float(memory.get("q_value", memory.get("reward", 0.0))), -1.0, 1.0)
            match_weight = max(
                _safe_float(item.get("similarity", 0.0))
                * _safe_float(item.get("transfer_gate", 1.0), 1.0),
                1e-6,
            )
            rescored.append(
                {
                    **item,
                    "adjusted_score": adjusted_score,
                    "match_weight": match_weight,
                    "evidence_score": _safe_float(item.get("transfer_gate", 1.0), 1.0) * abs(q_value),
                    "evidence_side": _evidence_side(memory),
                }
            )
        rescored.sort(key=lambda item: item["adjusted_score"], reverse=True)
        return rescored

    def _balanced_evidence(
        self,
        ranked: list[dict[str, Any]],
        *,
        per_side: int | None = None,
        context_family: str | None = None,
        action_family: str | None = None,
    ) -> list[dict[str, Any]]:
        intervene = [
            item for item in ranked
            if item.get("evidence_side") == "intervene"
        ]
        abstain = [
            item for item in ranked
            if item.get("evidence_side") == "abstain"
        ]
        intervene.sort(key=lambda item: _safe_float(item.get("evidence_score", 0.0)), reverse=True)
        abstain.sort(key=lambda item: _safe_float(item.get("evidence_score", 0.0)), reverse=True)
        if intervene and abstain:
            count = min(self.topk, len(intervene), len(abstain))
            intervene_strength = sum(
                _safe_float(item.get("evidence_score", 0.0))
                for item in intervene[:count]
            )
            abstain_strength = sum(
                _safe_float(item.get("evidence_score", 0.0))
                for item in abstain[:count]
            )
            if intervene_strength > abstain_strength * 1.5:
                selected = intervene[: self.topk]
            elif abstain_strength > intervene_strength * 1.5:
                selected = abstain[: self.topk]
            else:
                limit = max(1, int(per_side or max(1, self.topk // 2)))
                count = min(limit, len(intervene), len(abstain))
                selected = intervene[:count] + abstain[:count]
        else:
            selected = (intervene or abstain)[: self.topk]
        selected.sort(key=lambda item: _safe_float(item.get("evidence_score", 0.0)), reverse=True)
        return selected

    def _weighted_rate(self, ranked: list[dict[str, Any]], predicate) -> float:
        weights = [max(_safe_float(item.get("adjusted_score", item["score"])), 1e-6) for item in ranked]
        total = sum(weights) or 1.0
        return sum(weight for weight, item in zip(weights, ranked) if predicate(item["memory"])) / total

    def _recommend_for_generation(
        self,
        ranked: list[dict[str, Any]],
        *,
        context_family: str,
    ) -> dict[str, Any]:
        intervene_value = 0.0
        abstain_value = 0.0
        intervene_weight = 0.0
        abstain_weight = 0.0
        total_weight = 0.0
        matched_weight = 0.0

        for item in ranked:
            memory = item["memory"]
            weight = max(_safe_float(item.get("match_weight", item.get("adjusted_score", item["score"]))), 1e-6)
            q_value = abs(_clamp(_safe_float(memory.get("q_value", memory.get("reward", 0.0))), -1.0, 1.0))
            strength = weight * q_value
            total_weight += weight
            if str(memory.get("context_family", "general")) == context_family:
                matched_weight += weight
            side = item.get("evidence_side") or _evidence_side(memory)
            if side == "intervene":
                intervene_value += strength
                intervene_weight += strength
            elif side == "abstain":
                abstain_value += strength
                abstain_weight += strength

        total_value = intervene_value + abstain_value
        margin = intervene_value - abstain_value
        margin_ratio = margin / total_value if total_value > 0.0 else 0.0
        confidence = min(1.0, matched_weight / max(total_weight, 1e-6))
        intervene_ratio = intervene_weight / total_value if total_value > 0.0 else 0.0
        abstain_ratio = abstain_weight / total_value if total_value > 0.0 else 0.0

        positive_conf_threshold = 0.32
        positive_margin_ratio_threshold = 0.08
        negative_conf_threshold = 0.32
        negative_margin_ratio_threshold = -0.08
        close_abstain_margin_ratio_threshold = 0.02
        if context_family == "active_search":
            positive_conf_threshold = 0.40
            positive_margin_ratio_threshold = 0.12
            negative_conf_threshold = 0.26
            negative_margin_ratio_threshold = -0.04

        should_generate: bool | None = None
        if intervene_weight > 0.0 and abstain_weight > 0.0 and confidence >= negative_conf_threshold and margin_ratio <= close_abstain_margin_ratio_threshold:
            should_generate = False
        elif confidence >= positive_conf_threshold and margin_ratio >= positive_margin_ratio_threshold and intervene_ratio >= 0.55:
            should_generate = True
        elif confidence >= negative_conf_threshold and margin_ratio <= negative_margin_ratio_threshold and abstain_ratio >= 0.52:
            should_generate = False

        if should_generate is True:
            reason = "memory evidence clearly favors generating a timely candidate"
        elif should_generate is False:
            reason = "memory evidence does not clearly favor intervention, so generation should be conservative"
        elif margin_ratio > 0.0:
            reason = "memory evidence weakly favors intervention but is below the generation threshold"
        elif margin_ratio < 0.0:
            reason = "memory evidence weakly favors abstaining but is below the abstention threshold"
        else:
            reason = "memory evidence is balanced or unavailable"

        if margin_ratio >= positive_margin_ratio_threshold:
            direction = "favors_intervention"
        elif margin_ratio > close_abstain_margin_ratio_threshold:
            direction = "weakly_favors_intervention"
        elif margin_ratio <= negative_margin_ratio_threshold:
            direction = "favors_abstain"
        elif margin_ratio < 0.0:
            direction = "weakly_favors_abstain"
        else:
            direction = "balanced_or_close"

        return {
            "should_generate_candidate": should_generate,
            "confidence": round(confidence, 4),
            "intervene_memory_value": round(intervene_value, 4),
            "abstain_memory_value": round(abstain_value, 4),
            "margin": round(margin, 4),
            "margin_ratio": round(margin_ratio, 4),
            "positive_intervene_ratio": round(intervene_ratio, 4),
            "positive_abstain_ratio": round(abstain_ratio, 4),
            "balanced_intervene_count": sum(1 for item in ranked if (item.get("evidence_side") or _evidence_side(item["memory"])) == "intervene"),
            "balanced_abstain_count": sum(1 for item in ranked if (item.get("evidence_side") or _evidence_side(item["memory"])) == "abstain"),
            "evidence_direction": direction,
            "reason": reason,
            "context_family": context_family,
        }

    def retrieve_for_gate(
        self,
        observations: list[dict[str, Any]],
        signals: dict[str, Any],
    ) -> dict[str, Any]:
        context_family = infer_context_family(observations, signals)
        query = " | ".join(
            part
            for part in [
                self._base_query(observations, signals),
                "gate_view need flow risk evidence interruption threshold",
            ]
            if part
        )
        ranked = self._balanced_evidence(
            self._rerank(self._rank(query), context_family=context_family),
            context_family=context_family,
        )

        total_weight = 0.0
        matched_weight = 0.0
        intervene_value = 0.0
        abstain_value = 0.0
        reject_weight = 0.0
        missed_help_weight = 0.0
        support_cases: list[dict[str, Any]] = []
        risk_cases: list[dict[str, Any]] = []

        for item in ranked:
            memory = item["memory"]
            memory_context = str(memory.get("context_family", "general"))
            context_boost = 1.35 if memory_context == context_family else 0.9
            if context_family == "general" or memory_context == "general":
                context_boost = max(context_boost, 1.0)
            weight = max(_safe_float(item.get("match_weight", item.get("similarity", 0.0))), 1e-6)
            weight *= _gate_trust_multiplier(memory)
            q_value = _clamp(_safe_float(memory.get("q_value", memory.get("reward", 0.0))), -1.0, 1.0)
            strength = weight * abs(q_value)
            total_weight += weight
            if memory_context == context_family:
                matched_weight += weight

            decision = memory.get("decision", {}) or {}
            simulation = memory.get("simulation", {}) or {}
            outcome_family = str(memory.get("outcome_family", "uncertain"))
            acceptance = str(simulation.get("acceptance", "")).lower()
            side = item.get("evidence_side") or _evidence_side(memory)

            if side == "intervene":
                intervene_value += strength
                if len(support_cases) < 3:
                    support_cases.append(memory)
                if outcome_family in {"false_negative", "missed_help"}:
                    missed_help_weight += weight
            elif side == "abstain":
                abstain_value += strength
                if len(risk_cases) < 3:
                    risk_cases.append(memory)
                if q_value < 0.0 or acceptance in {"dismiss", "dismissed", "reject", "rejected", "annoyed"} or outcome_family in {"false_positive", "bad_intervention"}:
                    reject_weight += weight

        total = max(total_weight, 1e-6)
        confidence = _clamp((matched_weight / total) * min(1.0, len(ranked) / 4.0), 0.0, 1.0)
        historical_intervene_value = intervene_value / total
        historical_abstain_value = abstain_value / total
        historical_reject_risk = reject_weight / total
        missed_help_risk = missed_help_weight / total
        margin = historical_intervene_value - historical_abstain_value

        signal_need_delta = _clamp(0.08 * margin + 0.06 * missed_help_risk - 0.08 * historical_reject_risk, -0.12, 0.12) * confidence
        signal_risk_delta = _clamp(0.10 * historical_reject_risk - 0.04 * margin, -0.10, 0.12) * confidence
        signal_evidence_delta = _clamp(0.06 * margin + 0.04 * missed_help_risk, -0.08, 0.10) * confidence

        threshold_need_delta = _clamp(-0.08 * margin - 0.06 * missed_help_risk + 0.10 * historical_reject_risk, -0.10, 0.12) * confidence
        threshold_flow_delta = _clamp(0.04 * margin - 0.04 * historical_reject_risk, -0.06, 0.08) * confidence
        threshold_risk_delta = _clamp(0.06 * margin - 0.08 * historical_reject_risk, -0.08, 0.08) * confidence
        threshold_evidence_delta = _clamp(-0.04 * margin + 0.04 * historical_reject_risk, -0.06, 0.08) * confidence

        prior = {
            "current_context_family": context_family,
            "similar_case_count": len(ranked),
            "historical_intervene_value": round(historical_intervene_value, 4),
            "historical_abstain_value": round(historical_abstain_value, 4),
            "historical_reject_risk": round(historical_reject_risk, 4),
            "missed_help_risk": round(missed_help_risk, 4),
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "recommended_signal_delta": {
                "need": round(signal_need_delta, 4),
                "flow": 0.0,
                "risk": round(signal_risk_delta, 4),
                "evidence": round(signal_evidence_delta, 4),
            },
            "recommended_threshold_delta": {
                "need": round(threshold_need_delta, 4),
                "flow": round(threshold_flow_delta, 4),
                "risk": round(threshold_risk_delta, 4),
                "evidence": round(threshold_evidence_delta, 4),
            },
            "support_cases": support_cases,
            "risk_cases": risk_cases,
            "used_memory_ids": [item["memory"]["memory_id"] for item in ranked[:6]],
        }
        prior["gate_context"] = json.dumps(
            {
                "context_family": prior["current_context_family"],
                "similar_case_count": prior["similar_case_count"],
                "confidence": prior["confidence"],
                "margin": prior["margin"],
                "historical_reject_risk": prior["historical_reject_risk"],
                "missed_help_risk": prior["missed_help_risk"],
                "recommended_signal_delta": prior["recommended_signal_delta"],
                "recommended_threshold_delta": prior["recommended_threshold_delta"],
            },
            ensure_ascii=False,
        )
        return prior

    def _recommend_from_ranked(
        self,
        ranked: list[dict[str, Any]],
        *,
        context_family: str,
        action_family: str,
        memory_level_mode: int,
    ) -> dict[str, Any]:
        intervene_support = 0.0
        abstain_support = 0.0
        intervene_penalty = 0.0
        abstain_penalty = 0.0
        positive_intervene_weight = 0.0
        positive_abstain_weight = 0.0
        total_weight = 0.0
        matched_weight = 0.0

        for item in ranked:
            memory = item["memory"]
            weight = max(_safe_float(item.get("match_weight", item.get("adjusted_score", item["score"]))), 1e-6)
            q_value = max(min(_safe_float(memory.get("q_value", memory.get("reward", 0.0))), 1.0), -1.0)
            total_weight += weight
            if str(memory.get("context_family", "general")) == context_family:
                matched_weight += 0.6 * weight
            if str(memory.get("action_family", "no_intervention")) == action_family:
                matched_weight += 0.4 * weight

            strength = weight * abs(q_value)
            side = item.get("evidence_side") or _evidence_side(memory)
            if side == "intervene":
                intervene_support += strength
                positive_intervene_weight += strength
            elif side == "abstain":
                abstain_support += strength
                positive_abstain_weight += strength

        confidence = min(1.0, matched_weight / max(total_weight, 1e-6))
        intervene_value = intervene_support - 0.35 * intervene_penalty
        abstain_value = abstain_support - 0.35 * abstain_penalty
        margin = intervene_value - abstain_value
        positive_total = positive_intervene_weight + positive_abstain_weight
        positive_intervene_ratio = positive_intervene_weight / positive_total if positive_total > 0.0 else 0.0
        positive_abstain_ratio = positive_abstain_weight / positive_total if positive_total > 0.0 else 0.0
        margin_ratio = margin / positive_total if positive_total > 0.0 else 0.0
        should_intervene: bool | None = None
        level = 0
        positive_conf_threshold = 0.32
        positive_margin_ratio_threshold = 0.08
        negative_conf_threshold = 0.32
        negative_margin_ratio_threshold = -0.08
        close_abstain_margin_ratio_threshold = 0.02

        if context_family == "active_search" and action_family == "search_refine":
            positive_conf_threshold = 0.45
            positive_margin_ratio_threshold = 0.12
            negative_conf_threshold = 0.26
            negative_margin_ratio_threshold = -0.04
            close_abstain_margin_ratio_threshold = 0.02

        has_two_sided_evidence = positive_intervene_weight > 0.0 and positive_abstain_weight > 0.0
        if has_two_sided_evidence and confidence >= negative_conf_threshold and margin_ratio <= close_abstain_margin_ratio_threshold:
            should_intervene = False
        elif confidence >= positive_conf_threshold and margin_ratio >= positive_margin_ratio_threshold and positive_intervene_ratio >= 0.55:
            should_intervene = True
            level = max(1, int(memory_level_mode or 1))
        elif confidence >= negative_conf_threshold and margin_ratio <= negative_margin_ratio_threshold and positive_abstain_ratio >= 0.52:
            should_intervene = False

        if should_intervene is True:
            reason = "matched cases with the same context/action family more often led to successful intervention"
        elif should_intervene is False:
            if has_two_sided_evidence and margin_ratio <= close_abstain_margin_ratio_threshold:
                reason = "balanced memory evidence does not clearly favor intervention, so defaulting to abstain"
            else:
                reason = "matched cases with the same context/action family more often rewarded abstaining"
        else:
            reason = "memory evidence is mixed or low-confidence for this context/action family"
        return {
            "should_intervene": should_intervene,
            "level": int(level),
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "margin_ratio": round(margin_ratio, 4),
            "positive_intervene_ratio": round(positive_intervene_ratio, 4),
            "positive_abstain_ratio": round(positive_abstain_ratio, 4),
            "balanced_intervene_count": sum(1 for item in ranked if (item.get("evidence_side") or _evidence_side(item["memory"])) == "intervene"),
            "balanced_abstain_count": sum(1 for item in ranked if (item.get("evidence_side") or _evidence_side(item["memory"])) == "abstain"),
            "reason": reason,
            "context_family": context_family,
            "action_family": action_family,
        }

    def retrieve_for_generation(
        self,
        observations: list[dict[str, Any]], 
        signals: dict[str, Any],
    ) -> dict[str, Any]:
        context_family = infer_context_family(observations, signals)
        query = self._base_query(observations, signals)
        ranked = self._balanced_evidence(
            self._rerank(self._rank(query), context_family=context_family),
            context_family=context_family,
        )
        helpful_positive = [item for item in ranked if _value_bucket(item["memory"]) == "helpful_positive"]
        correct_abstain = [item for item in ranked if _value_bucket(item["memory"]) == "correct_abstain"]
        bad_intervention = [item for item in ranked if _value_bucket(item["memory"]) == "bad_intervention"]
        missed_help = [item for item in ranked if _value_bucket(item["memory"]) == "missed_help"]
        weak_uncertain = [item for item in ranked if _value_bucket(item["memory"]) == "weak_uncertain"]
        positives = helpful_positive + missed_help
        negatives = correct_abstain + bad_intervention
        generation_recommendation = self._recommend_for_generation(ranked, context_family=context_family)
        level_counter = Counter(_candidate_level(item["memory"]) for item in positives)
        preferred_level = level_counter.most_common(1)[0][0] if level_counter else 0
        preferred_action_families = [
            action_family
            for action_family, _ in Counter(
                str(item["memory"].get("action_family", "no_intervention")) for item in positives
            ).most_common(2)
        ]
        disallowed_action_families = [
            action_family
            for action_family, _ in Counter(
                str(item["memory"].get("action_family", "no_intervention")) for item in negatives
            ).most_common(2)
        ]
        positive_patterns = [
            item["memory"].get("candidate", {}).get("proactive_task")
            or item["memory"].get("candidate", {}).get("purpose")
            for item in positives[:3]
        ]
        negative_patterns = [
            item["memory"].get("simulation", {}).get("reasoning")
            or item["memory"].get("decision", {}).get("reason")
            for item in negatives[:3]
        ]
        avoid_patterns = [
            item["memory"].get("candidate", {}).get("response")
            for item in (bad_intervention + correct_abstain)[:3]
            if item["memory"].get("candidate", {}).get("response")
        ]
        prior = {
            "current_context_family": context_family,
            "candidate_positive_examples": [item["memory"] for item in positives[:3]],
            "candidate_negative_examples": [item["memory"] for item in negatives[:3]],
            "generation_recommendation": generation_recommendation,
            "intervene_memory_value": generation_recommendation["intervene_memory_value"],
            "abstain_memory_value": generation_recommendation["abstain_memory_value"],
            "value_aware_examples": {
                "helpful_positive": [item["memory"] for item in helpful_positive[:3]],
                "correct_abstain": [item["memory"] for item in correct_abstain[:3]],
                "bad_intervention": [item["memory"] for item in bad_intervention[:3]],
                "missed_help": [item["memory"] for item in missed_help[:3]],
                "weak_uncertain": [item["memory"] for item in weak_uncertain[:3]],
            },
            "preferred_level": preferred_level,
            "preferred_action_families": preferred_action_families,
            "disallowed_action_families": disallowed_action_families,
            "positive_patterns": [item for item in positive_patterns if item],
            "negative_patterns": [item for item in negative_patterns if item],
            "avoid_patterns": avoid_patterns,
            "used_memory_ids": [item["memory"]["memory_id"] for item in ranked[:6]],
        }
        prior["generation_context"] = build_generation_context(prior)
        return prior

    def retrieve_for_simulation(
        self,
        observations: list[dict[str, Any]],
        candidate: dict[str, Any],
        signals: dict[str, Any],
    ) -> dict[str, Any]:
        context_family = infer_context_family(observations, signals)
        action_family = infer_action_family(candidate)
        query = " | ".join(
            part for part in [
                self._base_query(observations, signals),
                f"action_family:{action_family}",
                candidate.get("purpose") or "",
                candidate.get("proactive_task") or "",
                candidate.get("response") or "",
            ] if part
        )
        ranked = self._balanced_evidence(
            self._rerank(self._rank(query), context_family=context_family, action_family=action_family),
            context_family=context_family,
            action_family=action_family,
        )
        accept_like = {"accept", "accepted", "helpful"}
        dismiss_like = {"dismiss", "dismissed", "reject", "rejected"}
        annoy_like = {"annoyed", "interrupting"}
        historical_accept_rate = self._weighted_rate(
            ranked,
            lambda memory: str(memory.get("simulation", {}).get("acceptance", "")).lower() in accept_like,
        )
        historical_dismiss_rate = self._weighted_rate(
            ranked,
            lambda memory: str(memory.get("simulation", {}).get("acceptance", "")).lower() in dismiss_like,
        )
        historical_annoy_rate = self._weighted_rate(
            ranked,
            lambda memory: str(memory.get("simulation", {}).get("acceptance", "")).lower() in annoy_like,
        )
        support_cases = [
            item["memory"]
            for item in ranked
            if str(item["memory"].get("simulation", {}).get("acceptance", "")).lower() in accept_like
        ][:3]
        risk_cases = [
            item["memory"]
            for item in ranked
            if str(item["memory"].get("simulation", {}).get("acceptance", "")).lower() in dismiss_like | annoy_like
        ][:3]
        prior = {
            "current_context_family": context_family,
            "candidate_action_family": action_family,
            "historical_accept_rate": historical_accept_rate,
            "historical_dismiss_rate": historical_dismiss_rate,
            "historical_annoy_rate": historical_annoy_rate,
            "support_cases": support_cases,
            "risk_cases": risk_cases,
            "historical_reject_risk": min(1.0, historical_dismiss_rate + historical_annoy_rate),
            "used_memory_ids": [item["memory"]["memory_id"] for item in ranked[:6]],
        }
        prior["simulation_context"] = build_simulation_context(prior)
        return prior

    def retrieve_for_decision(
        self,
        observations: list[dict[str, Any]],
        candidate: dict[str, Any],
        simulation: dict[str, Any],
        signals: dict[str, Any],
    ) -> dict[str, Any]:
        context_family = infer_context_family(observations, signals)
        action_family = infer_action_family(candidate)
        query = " | ".join(
            part for part in [
                self._base_query(observations, signals),
                f"action_family:{action_family}",
                candidate.get("proactive_task") or "",
                str(simulation.get("acceptance", "")),
                str(simulation.get("reasoning", "")),
            ] if part
        )
        ranked = self._balanced_evidence(
            self._rerank(self._rank(query), context_family=context_family, action_family=action_family),
            context_family=context_family,
            action_family=action_family,
        )
        intervene = [
            item for item in ranked
            if bool(item["memory"].get("decision", {}).get("should_intervene"))
        ]
        abstain = [
            item for item in ranked
            if not bool(item["memory"].get("decision", {}).get("should_intervene"))
        ]
        level_counter = Counter(_candidate_level(item["memory"]) for item in intervene[:5])
        memory_level_mode = level_counter.most_common(1)[0][0] if level_counter else 0
        memory_recommendation = self._recommend_from_ranked(
            ranked,
            context_family=context_family,
            action_family=action_family,
            memory_level_mode=memory_level_mode,
        )
        intervene_value = max(memory_recommendation["margin"], 0.0)
        abstain_value = max(-memory_recommendation["margin"], 0.0)
        historical_reject_risk = self._weighted_rate(
            ranked,
            lambda memory: str(memory.get("simulation", {}).get("acceptance", "")).lower()
            in {"dismiss", "dismissed", "reject", "rejected", "annoyed"},
        )
        prior = {
            "current_context_family": context_family,
            "candidate_action_family": action_family,
            "intervene_memories": [item["memory"] for item in intervene[:3]],
            "abstain_memories": [item["memory"] for item in abstain[:3]],
            "intervene_memory_value": intervene_value,
            "abstain_memory_value": abstain_value,
            "memory_level_mode": memory_level_mode,
            "historical_reject_risk": historical_reject_risk,
            "memory_recommendation": memory_recommendation,
            "used_memory_ids": [item["memory"]["memory_id"] for item in ranked[:6]],
        }
        prior["decision_context"] = build_decision_context(prior)
        return prior
