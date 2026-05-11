from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from eval.run_memrl_original_chain_eval import _gate_memory_augmented_result, run_original_chain_eval  # noqa: E402


def test_run_memrl_original_chain_eval_smoke(tmp_path: Path) -> None:
    bootstrap_episode = {
        "memory_id": "b1",
        "sample_id": "b1",
        "source": "judge_stage2",
        "domain": "coding",
        "observations": [{"event": "The user debugs a traceback in VS Code."}],
        "intent_text": "The user debugs a traceback in VS Code.",
        "context_family": "stuck_debug",
        "action_family": "debug_help",
        "outcome_family": "helpful_intervention",
        "candidate": {
            "purpose": "Fix the bug.",
            "proactive_task": "Inspect the traceback and suggest the next fix.",
            "response": "I can inspect the traceback and suggest the next fix.",
            "operation": None,
        },
        "simulation": {
            "acceptance": "accept",
            "acceptance_confidence": 0.9,
            "flow_impact": "improved",
            "relevance": "highly_relevant",
            "timing": "good_timing",
            "reasoning": "The suggestion is timely.",
        },
        "decision": {"should_intervene": True, "commitment_level": 2, "risk": "low", "reason": "helpful"},
        "labels": {},
        "reward": 1.0,
        "q_value": 1.0,
        "q_visits": 1,
        "created_at": "",
        "updated_at": "",
    }
    bootstrap = tmp_path / "bootstrap.jsonl"
    bootstrap.write_text(json.dumps(bootstrap_episode, ensure_ascii=False) + "\n", encoding="utf-8")

    test_root = tmp_path / "test"
    gold_root = tmp_path / "gold"
    output_root = tmp_path / "pred"
    snapshot_root = tmp_path / "snapshot"
    test_root.mkdir()
    gold_root.mkdir()

    events = [
        {
            "observation": {"time": "1", "event": "The user debugs a traceback in VS Code."},
            "gold_decision": {"should_intervene": 1, "commitment_level": 2},
        }
    ]
    (test_root / "sample.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    (gold_root / "sample.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")

    async def fake_responder(messages: list[dict[str, str]]) -> dict[str, object]:
        system = messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        if "Infer only the user's latent state" in system:
            return {
                "signals": {
                    "flow": 0.2,
                    "stuck": 0.85,
                    "need": 0.9,
                    "accept": 0.8,
                    "risk": 0.2,
                    "uncertainty": 0.15,
                    "progress": 0.15,
                    "rejection_memory": 0.0,
                }
            }
        if "proposing one candidate intervention" in system:
            assert "memory_generation_prior" in payload
            return {
                "candidate": {
                    "purpose": "Help debug the traceback.",
                    "proactive_task": "Inspect the traceback and suggest the next fix.",
                    "response": "I can inspect the traceback and suggest the next fix.",
                    "operation": None,
                }
            }
        if "evaluates one already-proposed candidate intervention" in system:
            assert "memory_simulation_prior" in payload
            if payload.get("memory_mode") == "no_memory_baseline":
                return {
                    "simulated_reaction": {
                        "rubric_scores": {
                            "personal_preference": 0,
                            "frequency": 0,
                            "timing": 0,
                            "communication": 0,
                        },
                        "acceptance": "ignore",
                        "acceptance_confidence": 0.35,
                        "flow_impact": "unchanged",
                        "relevance": "somewhat_relevant",
                        "timing": "neutral",
                        "reasoning": "Without memory, this looks plausible but not clearly timely.",
                        "persona_vote_summary": {"persona_ids": [], "weights": []},
                    }
                }
            return {
                "simulated_reaction": {
                    "rubric_scores": {
                        "personal_preference": 1,
                        "frequency": 1,
                        "timing": 1,
                        "communication": 1,
                    },
                    "acceptance": "accept",
                    "acceptance_confidence": 0.92,
                    "flow_impact": "improved",
                    "relevance": "highly_relevant",
                    "timing": "good_timing",
                    "reasoning": "The user is blocked and the suggestion is timely.",
                    "persona_vote_summary": {"persona_ids": [], "weights": []},
                }
            }
        if "making the final intervention decision" in system:
            assert "memory_decision_prior" in payload
            if payload.get("memory_mode") == "no_memory_baseline":
                return {
                    "decision": {
                        "should_intervene": False,
                        "level": 0,
                        "risk": "medium",
                        "reason": "Baseline path abstains because the evidence is only weakly positive.",
                    }
                }
            return {
                "decision": {
                    "should_intervene": True,
                    "level": 2,
                    "risk": "low",
                    "reason": "The gate passes, the simulated reaction is positive, and similar memories support intervention.",
                }
            }
        if "writes the final user-facing intervention" in system:
            return {
                "Purpose": "Help debug the traceback.",
                "Thoughts": "The user appears blocked and the candidate is well timed.",
                "Proactive Task": "Inspect the traceback and suggest the next fix.",
                "Response": "I can inspect the traceback and suggest the next fix.",
                "Operation": None,
            }
        raise AssertionError(f"Unexpected system prompt: {system[:80]}")

    asyncio.run(
        run_original_chain_eval(
            bootstrap_path=bootstrap,
            test_root=test_root,
            gold_root=gold_root,
            output_root=output_root,
            snapshot_root=snapshot_root,
            run_baseline=True,
            responder=fake_responder,
        )
    )

    pred = json.loads((output_root / "sample.json").read_text(encoding="utf-8"))
    baseline_pred = json.loads((tmp_path / "pred_baseline" / "sample.json").read_text(encoding="utf-8"))
    assert pred[0]["task_status"] is True
    assert pred[0]["agent_response"] == ["Inspect the traceback and suggest the next fix."]
    assert pred[0]["other_infomation"]["BaseDecision"]["should_intervene"] is False
    assert pred[0]["other_infomation"]["Decision"]["commitment_level"] == 2
    assert pred[0]["other_infomation"]["DecisionMode"] == "memory_augmented_separate"
    assert pred[0]["other_infomation"]["memory_changed_decision"] is True
    assert pred[0]["other_infomation"]["MemoryGate"]["applied"] is True
    assert pred[0]["other_infomation"]["MemoryGate"]["reason"] == "separate_memory_augmented_run"
    assert baseline_pred[0]["task_status"] is False
    assert baseline_pred[0]["other_infomation"]["DecisionMode"] == "no_memory_baseline_separate"
    assert baseline_pred[0]["other_infomation"]["Decision"]["should_intervene"] is False
    assert "preferred_level" in pred[0]["other_infomation"]["GenerationPrior"]
    assert pred[0]["other_infomation"]["DecisionPrior"]["memory_recommendation"]["confidence"] > 0.0
    assert pred[0]["other_infomation"]["DecisionPrior"]["intervene_memory_value"] > 0.0
    assert (snapshot_root / "memrl_snapshot.jsonl").exists()


def test_low_confidence_memory_cannot_override_base_decision() -> None:
    base_result = {
        "candidate": {"proactive_task": None},
        "simulation": {"acceptance": "ignore"},
        "decision": {"should_intervene": False, "commitment_level": 0},
        "decision_prior": {"memory_recommendation": {"should_intervene": None, "confidence": 0.22}},
        "final_output": {"Proactive Task": None},
        "raw_phase_g": {},
        "raw_phase_b": {},
        "raw_phase_d": {},
        "raw_phase_c": {},
        "generation_prior": {},
        "simulation_prior": {},
    }
    memory_result = {
        "candidate": {"proactive_task": "Suggest a better search keyword."},
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 1},
        "decision_prior": {"memory_recommendation": {"should_intervene": None, "confidence": 0.22}},
        "final_output": {"Proactive Task": "Suggest a better search keyword."},
        "raw_phase_g": {},
        "raw_phase_b": {},
        "raw_phase_d": {},
        "raw_phase_c": {},
        "generation_prior": {},
        "simulation_prior": {},
    }
    effective, gate = _gate_memory_augmented_result(base_result=base_result, memory_result=memory_result)
    assert effective is base_result
    assert gate["applied"] is False


def test_strong_positive_memory_can_lift_baseline_with_moderate_confidence() -> None:
    base_result = {
        "candidate": {"proactive_task": None},
        "simulation": {"acceptance": "ignore"},
        "decision": {"should_intervene": False, "commitment_level": 0},
        "decision_prior": {"memory_recommendation": {"should_intervene": None, "confidence": 0.0}},
        "final_output": {"Proactive Task": None},
        "raw_phase_g": {},
        "raw_phase_b": {},
        "raw_phase_d": {},
        "raw_phase_c": {},
        "generation_prior": {},
        "simulation_prior": {},
    }
    memory_result = {
        "candidate": {"proactive_task": "Suggest a better search keyword."},
        "simulation": {"acceptance": "accept"},
        "decision": {"should_intervene": True, "commitment_level": 1},
        "decision_prior": {
            "memory_recommendation": {
                "should_intervene": True,
                "confidence": 0.43,
                "margin": 0.34,
                "positive_intervene_ratio": 0.7,
            }
        },
        "final_output": {"Proactive Task": "Suggest a better search keyword."},
        "raw_phase_g": {},
        "raw_phase_b": {},
        "raw_phase_d": {},
        "raw_phase_c": {},
        "generation_prior": {},
        "simulation_prior": {},
    }
    effective, gate = _gate_memory_augmented_result(base_result=base_result, memory_result=memory_result)
    assert effective is memory_result
    assert gate["applied"] is True
    assert gate["reason"] == "high_conf_positive_recommendation_lift"
