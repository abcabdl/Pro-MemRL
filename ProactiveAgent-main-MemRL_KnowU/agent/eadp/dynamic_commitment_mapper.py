"""Two-stage intervention gate and commitment mapping with dynamic R(t)."""

from __future__ import annotations

from dataclasses import dataclass

from .feedback_memory import FeedbackMemory
from .types import DecisionContext, DualState, MappingDecision, clamp_01


@dataclass(slots=True)
class DynamicCommitmentConfig:
    """Coefficients and thresholds for dynamic gating and commitment levels."""

    # Dynamic R(t) coefficients.
    r0: float = 1.0
    alpha_flow: float = 1.0
    alpha_epistemic: float = 1.0
    alpha_reject: float = 1.0
    beta_stuck: float = 0.7
    beta_risk: float = 0.7
    min_term: float = 1e-6

    # Stage-2 commitment thresholds.
    epsilon_high_threshold: float = 0.70
    epsilon_low_threshold: float = 0.35
    risk_high_threshold: float = 0.70
    risk_low_threshold: float = 0.35


class DynamicCommitmentMapper:
    """Two-stage policy: intervene gate first, then level among {0,1,2}."""

    def __init__(
        self,
        config: DynamicCommitmentConfig | None = None,
        feedback_memory: FeedbackMemory | None = None,
    ) -> None:
        self.config = config or DynamicCommitmentConfig()
        self.feedback_memory = feedback_memory or FeedbackMemory()

    def map_state(
        self,
        state: DualState,
        context: DecisionContext | None = None,
    ) -> MappingDecision:
        """Run two-stage decision: hard-constraints -> dynamic gate -> level."""
        ctx = context or DecisionContext()
        if ctx.user_pref_reject or ctx.manual_suppressed:
            return MappingDecision(
                commitment_level=0,
                reason="Level 0 (Keep Silent): user preference/suppression blocks intervention.",
                should_intervene=False,
                diagnostics={"hard_constraint": 1.0},
            )

        f_flow = clamp_01(state.flow_index)
        d_stuck = clamp_01(state.stuck_index)
        epsilon_agent = clamp_01(
            ctx.epsilon_agent if ctx.epsilon_agent is not None else (1.0 - state.epistemic_confidence)
        )
        r_risk = clamp_01(ctx.r_risk)
        p_need = clamp_01(ctx.p_need if ctx.p_need is not None else state.need_probability)
        p_accept = clamp_01(
            ctx.p_accept if ctx.p_accept is not None else (p_need * (1.0 - 0.35 * epsilon_agent))
        )
        delta_rej = self._resolve_delta_rej(ctx)
        r_value = self._compute_r_t(
            f_flow=f_flow,
            epsilon_agent=epsilon_agent,
            delta_rej=delta_rej,
            d_stuck=d_stuck,
            r_risk=r_risk,
        )
        tau = r_value / (r_value + max(self.config.min_term, p_need))

        diagnostics = {
            "f_flow": f_flow,
            "d_stuck": d_stuck,
            "epsilon_agent": epsilon_agent,
            "r_risk": r_risk,
            "delta_rej": delta_rej,
            "p_need": p_need,
            "p_accept": p_accept,
            "r_value": r_value,
            "tau": tau,
        }

        if p_accept < tau:
            return MappingDecision(
                commitment_level=0,
                reason="Level 0 (Keep Silent): intervene gate blocked by p_accept < tau(t).",
                should_intervene=False,
                r_value=r_value,
                tau=tau,
                diagnostics=diagnostics,
            )

        if epsilon_agent >= self.config.epsilon_high_threshold:
            return MappingDecision(
                commitment_level=1,
                reason="Level 1 (Probe): high epistemic uncertainty, clarify before acting.",
                should_intervene=True,
                r_value=r_value,
                tau=tau,
                diagnostics=diagnostics,
            )

        if epsilon_agent < self.config.epsilon_high_threshold and r_risk >= self.config.risk_high_threshold:
            return MappingDecision(
                commitment_level=1,
                reason="Level 1 (Probe): model is usable but action risk is high.",
                should_intervene=True,
                r_value=r_value,
                tau=tau,
                diagnostics=diagnostics,
            )

        risk_is_medium = self.config.risk_low_threshold <= r_risk < self.config.risk_high_threshold
        if epsilon_agent <= self.config.epsilon_low_threshold and risk_is_medium:
            return MappingDecision(
                commitment_level=2,
                reason="Level 2 (Suggest): low uncertainty with medium action risk.",
                should_intervene=True,
                r_value=r_value,
                tau=tau,
                diagnostics=diagnostics,
            )

        return MappingDecision(
            commitment_level=2,
            reason="Level 2 (Suggest): gate passed and no probe condition hit.",
            should_intervene=True,
            r_value=r_value,
            tau=tau,
            diagnostics=diagnostics,
        )

    def observe_feedback(self, quick_reject: bool) -> float:
        """Update feedback memory with one rejection event and return delta_rej."""
        return self.feedback_memory.update(quick_reject=quick_reject)

    def _resolve_delta_rej(self, ctx: DecisionContext) -> float:
        if ctx.delta_rej is not None:
            return clamp_01(ctx.delta_rej)
        if ctx.recent_quick_rejects:
            return self.feedback_memory.compute_from_flags(ctx.recent_quick_rejects)
        return self.feedback_memory.value()

    def _compute_r_t(
        self,
        f_flow: float,
        epsilon_agent: float,
        delta_rej: float,
        d_stuck: float,
        r_risk: float,
    ) -> float:
        numerator = (
            self.config.r0
            * (1.0 + self.config.alpha_flow * f_flow)
            * (1.0 + self.config.alpha_epistemic * epsilon_agent)
            * (1.0 + self.config.alpha_reject * delta_rej)
        )
        denominator = (
            (1.0 + self.config.beta_stuck * d_stuck)
            * (1.0 + self.config.beta_risk * r_risk)
        )
        safe_numerator = max(self.config.min_term, numerator)
        safe_denominator = max(self.config.min_term, denominator)
        return safe_numerator / safe_denominator
