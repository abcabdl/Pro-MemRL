from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

import tenacity
from codelinker import CodeLinker, CodeLinkerConfig

from eval.reward_model_template import format_personalized_rubric_instruction
from proactive_pipeline import normalize_decision, normalize_simulation, parse_json_payload


class PersonalizedLLMJudge:
    def __init__(
        self,
        *,
        model_aliases: list[str],
        cfg_path: str = "private.toml",
        concurrency: int = 4,
        temperature: float = 0.0,
    ) -> None:
        if not model_aliases:
            raise ValueError("model_aliases must not be empty.")
        config = CodeLinkerConfig.from_toml(cfg_path)
        config.request.use_cache = False
        self.client = CodeLinker(config=config)
        self.model_aliases = list(model_aliases)
        self._concurrency = max(1, int(concurrency))
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        self.temperature = float(temperature)

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self._concurrency)
            self._sem_loop = loop
        return self._sem

    async def _call_one(
        self,
        *,
        model: str,
        observations: list[dict[str, Any]],
        domain: str,
        persona: dict[str, Any],
        rubric: dict[str, Any],
        history_summary: str,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        messages = format_personalized_rubric_instruction(
            observations=observations,
            domain=domain,
            persona=persona,
            rubric=rubric,
            history_summary=history_summary,
            candidate=candidate,
        )
        async for attempt in tenacity.AsyncRetrying(stop=tenacity.stop_after_attempt(3), reraise=True):
            with attempt:
                async with self._get_semaphore():
                    raw = await self.client.exec(
                        model=model,
                        messages=messages,
                        completions_kwargs={"temperature": self.temperature},
                    )
                payload = parse_json_payload(raw)
                payload["_judge_model"] = model
                return payload
        raise RuntimeError("unreachable")

    @staticmethod
    def _aggregate_label(values: list[str], *, allowed: tuple[str, ...], default: str) -> str:
        filtered = [value for value in values if value in allowed]
        if not filtered:
            return default
        counts = Counter(filtered)
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    @staticmethod
    def _aggregate_reaction(payloads: list[dict[str, Any]], *, persona_id: str) -> dict[str, Any]:
        normalized = [normalize_simulation(payload) for payload in payloads]
        if not normalized:
            return normalize_simulation({})
        n = float(len(normalized))
        rubric_scores = {
            "personal_preference": round(sum(item["rubric_scores"]["personal_preference"] for item in normalized) / n),
            "frequency": round(sum(item["rubric_scores"]["frequency"] for item in normalized) / n),
            "timing": round(sum(item["rubric_scores"]["timing"] for item in normalized) / n),
            "communication": round(sum(item["rubric_scores"]["communication"] for item in normalized) / n),
        }
        total_score = sum(float(item["total_score"]) for item in normalized) / n
        acceptance = PersonalizedLLMJudge._aggregate_label(
            [str(item["acceptance"]) for item in normalized],
            allowed=("accept", "ignore", "dismiss", "annoyed"),
            default="ignore",
        )
        flow_impact = PersonalizedLLMJudge._aggregate_label(
            [str(item["flow_impact"]) for item in normalized],
            allowed=("improved", "unchanged", "disrupted"),
            default="unchanged",
        )
        relevance = PersonalizedLLMJudge._aggregate_label(
            [str(item["relevance"]) for item in normalized],
            allowed=("highly_relevant", "somewhat_relevant", "irrelevant"),
            default="somewhat_relevant",
        )
        timing = PersonalizedLLMJudge._aggregate_label(
            [str(item["timing"]) for item in normalized],
            allowed=("good_timing", "neutral", "bad_timing"),
            default="neutral",
        )
        reasoning = " ".join(filter(None, [str(item.get("reasoning", "")).strip() for item in normalized[:2]])).strip()
        return normalize_simulation(
            {
                "rubric_scores": rubric_scores,
                "total_score": total_score,
                "acceptance": acceptance,
                "acceptance_confidence": sum(float(item["acceptance_confidence"]) for item in normalized) / n,
                "flow_impact": flow_impact,
                "relevance": relevance,
                "timing": timing,
                "reasoning": reasoning,
                "persona_vote_summary": {
                    "persona_ids": [persona_id for _ in normalized],
                    "weights": [round(1.0 / n, 4) for _ in normalized],
                },
            }
        )

    @staticmethod
    def _aggregate_decision(payloads: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = [
            normalize_decision(payload.get("intervention_recommendation", {}))
            for payload in payloads
            if isinstance(payload, dict)
        ]
        if not normalized:
            return normalize_decision({})
        n = float(len(normalized))
        should_intervene = sum(1 for item in normalized if item["should_intervene"]) >= (len(normalized) / 2.0)
        if not should_intervene:
            return {
                "should_intervene": False,
                "level": 0,
                "risk": PersonalizedLLMJudge._aggregate_label(
                    [str(item["risk"]) for item in normalized],
                    allowed=("low", "medium", "high"),
                    default="medium",
                ),
                "reason": "llm_ensemble_majority_block",
            }
        level = round(sum(int(item["level"]) for item in normalized) / n)
        risk = PersonalizedLLMJudge._aggregate_label(
            [str(item["risk"]) for item in normalized],
            allowed=("low", "medium", "high"),
            default="medium",
        )
        reason = " | ".join(filter(None, [str(item.get("reason", "")).strip() for item in normalized[:2]])).strip()
        return normalize_decision(
            {
                "should_intervene": True,
                "level": level,
                "risk": risk,
                "reason": reason or "llm_ensemble_decision",
            }
        )

    async def annotate_async(
        self,
        *,
        observations: list[dict[str, Any]],
        domain: str,
        persona: dict[str, Any],
        rubric: dict[str, Any],
        history_summary: str,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        payloads = await asyncio.gather(
            *[
                self._call_one(
                    model=model,
                    observations=observations,
                    domain=domain,
                    persona=persona,
                    rubric=rubric,
                    history_summary=history_summary,
                    candidate=candidate,
                )
                for model in self.model_aliases
            ]
        )
        persona_id = str(persona.get("persona_id", "persona_unknown"))
        return {
            "reaction": self._aggregate_reaction(payloads, persona_id=persona_id),
            "decision": self._aggregate_decision(payloads),
            "raw_payloads": payloads,
        }

    def annotate(
        self,
        *,
        observations: list[dict[str, Any]],
        domain: str,
        persona: dict[str, Any],
        rubric: dict[str, Any],
        history_summary: str,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        return asyncio.run(
            self.annotate_async(
                observations=observations,
                domain=domain,
                persona=persona,
                rubric=rubric,
                history_summary=history_summary,
                candidate=candidate,
            )
        )
