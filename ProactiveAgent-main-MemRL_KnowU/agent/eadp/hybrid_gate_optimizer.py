"""Hybrid optimizer for dynamic gate parameters and learnable signal estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .dynamic_commitment_mapper import DynamicCommitmentMapper
from .signal_estimation_layer import SignalEstimationLayer
from .types import DecisionContext, DualState, clamp_01


@dataclass(slots=True)
class GateTrainingExample:
    """Offline training tuple for gate and signal-estimator optimization."""

    f_flow: float
    d_stuck: float
    epsilon_agent: float
    delta_rej: float
    r_risk: float
    p_need: float
    p_accept: float
    y_need: int
    y_accept: int
    user_pref_reject: bool = False
    manual_suppressed: bool = False
    flow_features: list[float] = field(default_factory=list)
    risk_features: list[float] = field(default_factory=list)
    reward: float | None = None


@dataclass(slots=True)
class HybridGateOptimizerConfig:
    """Config for reward-driven optimization of alpha/beta/lambda and estimators."""

    meta_learning_rate: float = 0.08
    finite_diff_eps: float = 0.02
    steps_per_call: int = 1
    reward_tp: float = 1.0
    reward_tn: float = 1.0
    penalty_fp: float = 1.0
    penalty_fn: float = 1.0
    alpha_min: float = 0.0
    alpha_max: float = 3.0
    beta_min: float = 0.0
    beta_max: float = 3.0
    lambda_min: float = 0.05
    lambda_max: float = 0.95


class HybridGateOptimizer:
    """Optimize gate weights and partially learn signal estimators via reward feedback."""

    def __init__(
        self,
        mapper: DynamicCommitmentMapper,
        signal_layer: SignalEstimationLayer,
        config: HybridGateOptimizerConfig | None = None,
    ) -> None:
        self.mapper = mapper
        self.signal_layer = signal_layer
        self.config = config or HybridGateOptimizerConfig()

    def optimize(self, examples: Sequence[GateTrainingExample]) -> dict[str, float]:
        """Run reward-driven optimization updates."""
        if not examples:
            return {"objective_before": 0.0, "objective_after": 0.0}

        objective_before = self._objective(examples)
        for _ in range(max(1, self.config.steps_per_call)):
            self._finite_difference_update(examples)
            self._finetune_estimators(examples)
        objective_after = self._objective(examples)
        return {
            "objective_before": objective_before,
            "objective_after": objective_after,
            "alpha_flow": float(self.mapper.config.alpha_flow),
            "alpha_epistemic": float(self.mapper.config.alpha_epistemic),
            "alpha_reject": float(self.mapper.config.alpha_reject),
            "beta_stuck": float(self.mapper.config.beta_stuck),
            "beta_risk": float(self.mapper.config.beta_risk),
            "decay_lambda": float(self.mapper.feedback_memory.config.decay_lambda),
        }

    def _finite_difference_update(self, examples: Sequence[GateTrainingExample]) -> None:
        param_specs = [
            ("alpha_flow", self.config.alpha_min, self.config.alpha_max),
            ("alpha_epistemic", self.config.alpha_min, self.config.alpha_max),
            ("alpha_reject", self.config.alpha_min, self.config.alpha_max),
            ("beta_stuck", self.config.beta_min, self.config.beta_max),
            ("beta_risk", self.config.beta_min, self.config.beta_max),
        ]
        eps = self.config.finite_diff_eps

        for name, p_min, p_max in param_specs:
            base_val = float(getattr(self.mapper.config, name))
            plus_val = clamp_01((base_val + eps) / max(1e-8, p_max)) * p_max
            minus_val = clamp_01((base_val - eps) / max(1e-8, p_max)) * p_max
            setattr(self.mapper.config, name, plus_val)
            obj_plus = self._objective(examples)
            setattr(self.mapper.config, name, minus_val)
            obj_minus = self._objective(examples)
            setattr(self.mapper.config, name, base_val)

            grad = (obj_plus - obj_minus) / max(1e-8, 2.0 * eps)
            new_val = base_val + self.config.meta_learning_rate * grad
            setattr(self.mapper.config, name, max(p_min, min(p_max, new_val)))

        base_lambda = float(self.mapper.feedback_memory.config.decay_lambda)
        plus_lambda = max(self.config.lambda_min, min(self.config.lambda_max, base_lambda + eps))
        minus_lambda = max(self.config.lambda_min, min(self.config.lambda_max, base_lambda - eps))
        self.mapper.feedback_memory.config.decay_lambda = plus_lambda
        obj_plus = self._objective(examples)
        self.mapper.feedback_memory.config.decay_lambda = minus_lambda
        obj_minus = self._objective(examples)
        self.mapper.feedback_memory.config.decay_lambda = base_lambda

        grad_lambda = (obj_plus - obj_minus) / max(1e-8, 2.0 * eps)
        updated_lambda = base_lambda + self.config.meta_learning_rate * grad_lambda
        self.mapper.feedback_memory.config.decay_lambda = max(
            self.config.lambda_min,
            min(self.config.lambda_max, updated_lambda),
        )

    def _finetune_estimators(self, examples: Sequence[GateTrainingExample]) -> None:
        flow_feats: list[list[float]] = []
        risk_feats: list[list[float]] = []
        flow_rewards: list[float] = []
        risk_rewards: list[float] = []

        for ex in examples:
            decision = self._decide(ex)
            utility = ex.reward if ex.reward is not None else self._utility(ex, decision.should_intervene)
            if ex.flow_features:
                flow_feats.append(ex.flow_features)
                flow_rewards.append(utility)
            if ex.risk_features:
                risk_feats.append(ex.risk_features)
                risk_rewards.append(utility)

        if flow_feats and flow_rewards:
            self.signal_layer.flow_estimator.rl_finetune_step(flow_feats, flow_rewards)
        if risk_feats and risk_rewards:
            self.signal_layer.risk_estimator.rl_finetune_step(risk_feats, risk_rewards)

    def _objective(self, examples: Sequence[GateTrainingExample]) -> float:
        values = []
        for ex in examples:
            decision = self._decide(ex)
            reward = ex.reward if ex.reward is not None else self._utility(ex, decision.should_intervene)
            values.append(float(reward))
        return float(sum(values) / max(1, len(values)))

    def _decide(self, ex: GateTrainingExample):
        state = DualState(
            flow_index=clamp_01(ex.f_flow),
            stuck_index=clamp_01(ex.d_stuck),
            epistemic_confidence=clamp_01(1.0 - ex.epsilon_agent),
            need_probability=clamp_01(ex.p_need),
        )
        return self.mapper.map_state(
            state=state,
            context=DecisionContext(
                p_need=clamp_01(ex.p_need),
                p_accept=clamp_01(ex.p_accept),
                r_risk=clamp_01(ex.r_risk),
                epsilon_agent=clamp_01(ex.epsilon_agent),
                delta_rej=clamp_01(ex.delta_rej),
                user_pref_reject=bool(ex.user_pref_reject),
                manual_suppressed=bool(ex.manual_suppressed),
            ),
        )

    def _utility(self, ex: GateTrainingExample, intervene: bool) -> float:
        if intervene:
            if int(ex.y_accept) == 1:
                return self.config.reward_tp
            return -self.config.penalty_fp
        if int(ex.y_need) == 0:
            return self.config.reward_tn
        return -self.config.penalty_fn
