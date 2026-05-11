import unittest

from agent.eadp.dynamic_commitment_mapper import DynamicCommitmentConfig, DynamicCommitmentMapper
from agent.eadp.types import DecisionContext, DualState


class TestEADPGate(unittest.TestCase):
    def test_r_t_and_tau_formula(self) -> None:
        mapper = DynamicCommitmentMapper(
            config=DynamicCommitmentConfig(
                r0=1.0,
                alpha_flow=1.0,
                alpha_epistemic=1.0,
                alpha_reject=1.0,
                beta_stuck=1.0,
                beta_risk=1.0,
            )
        )
        state = DualState(
            flow_index=0.2,
            stuck_index=0.3,
            epistemic_confidence=0.6,
            need_probability=0.5,
        )
        ctx = DecisionContext(
            p_need=0.5,
            p_accept=0.8,
            r_risk=0.4,
            epsilon_agent=0.4,
            delta_rej=0.2,
        )
        decision = mapper.map_state(state, ctx)
        expected_r = ((1 + 0.2) * (1 + 0.4) * (1 + 0.2)) / ((1 + 0.3) * (1 + 0.4))
        expected_tau = expected_r / (expected_r + 0.5)
        self.assertAlmostEqual(decision.r_value, expected_r, places=8)
        self.assertAlmostEqual(decision.tau, expected_tau, places=8)

    def test_level_three_removed(self) -> None:
        mapper = DynamicCommitmentMapper()
        state = DualState(
            flow_index=0.1,
            stuck_index=0.8,
            epistemic_confidence=0.95,
            need_probability=0.9,
        )
        ctx = DecisionContext(
            p_need=0.9,
            p_accept=0.95,
            r_risk=0.1,
            epsilon_agent=0.05,
            delta_rej=0.0,
            auth_granted=True,
            can_rollback_or_preview=True,
            target_unambiguous=True,
        )
        decision = mapper.map_state(state, ctx)
        self.assertLessEqual(decision.commitment_level, 2)
        self.assertEqual(decision.commitment_level, 2)

    def test_gate_blocks_when_accept_below_tau(self) -> None:
        mapper = DynamicCommitmentMapper()
        state = DualState(
            flow_index=0.4,
            stuck_index=0.4,
            epistemic_confidence=0.5,
            need_probability=0.8,
        )
        ctx = DecisionContext(
            p_need=0.8,
            p_accept=0.05,
            r_risk=0.4,
            epsilon_agent=0.5,
            delta_rej=0.2,
        )
        decision = mapper.map_state(state, ctx)
        self.assertEqual(decision.commitment_level, 0)
        self.assertFalse(decision.should_intervene)


if __name__ == "__main__":
    unittest.main()

