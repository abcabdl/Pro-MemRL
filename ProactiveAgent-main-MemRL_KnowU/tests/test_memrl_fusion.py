from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from agent.memrl.fusion import fuse_decision  # noqa: E402


def test_abstain_margin_blocks_intervention() -> None:
    fused = fuse_decision(
        signal_score=0.6,
        backbone_level=2,
        generation_prior={"preferred_level": 2},
        simulation_result={"acceptance": "accept", "acceptance_confidence": 0.7},
        decision_prior={
            "intervene_memory_value": 0.2,
            "abstain_memory_value": 0.5,
            "memory_level_mode": 1,
            "historical_reject_risk": 0.2,
        },
    )
    assert fused["should_intervene"] is False
    assert fused["level"] == 0


def test_positive_memory_support_keeps_intervention() -> None:
    fused = fuse_decision(
        signal_score=0.5,
        backbone_level=1,
        generation_prior={"preferred_level": 1},
        simulation_result={"acceptance": "accept", "acceptance_confidence": 0.8},
        decision_prior={
            "intervene_memory_value": 0.75,
            "abstain_memory_value": 0.1,
            "memory_level_mode": 2,
            "historical_reject_risk": 0.1,
        },
    )
    assert fused["should_intervene"] is True
    assert fused["level"] >= 1
