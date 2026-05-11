"""Layer-1 signal estimation with mixed rule-based and learnable components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .dual_state_estimator import DualStateEstimator
from .feedback_memory import FeedbackMemory
from .types import EventRecord, InternalGenerationSignal, SignalEstimate, clamp_01


@dataclass(slots=True)
class LearnableEstimatorConfig:
    """Configuration for lightweight learnable sigmoid estimators."""

    feature_dim: int
    learning_rate: float = 0.05
    l2_coeff: float = 1e-4
    init_scale: float = 0.05
    seed: int = 17


class LearnableSigmoidEstimator:
    """Small logistic estimator with supervised and reward-driven updates."""

    def __init__(self, config: LearnableEstimatorConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(self.config.seed)
        self.weights = self.rng.normal(
            0.0,
            self.config.init_scale,
            size=(self.config.feature_dim,),
        )
        self.bias = float(self.rng.normal(0.0, self.config.init_scale))

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def predict_proba(self, features: Sequence[float]) -> float:
        x = np.asarray(features, dtype=np.float64)
        logits = float(np.dot(self.weights, x) + self.bias)
        return clamp_01(self._sigmoid(logits))

    def supervised_fit_step(
        self,
        feature_batch: Sequence[Sequence[float]],
        labels: Sequence[float],
    ) -> dict[str, float]:
        """One SGD step with binary cross-entropy objective."""
        if not feature_batch:
            return {"loss": 0.0, "grad_norm": 0.0}

        grad_w = np.zeros_like(self.weights)
        grad_b = 0.0
        loss_acc = 0.0
        n = float(len(feature_batch))

        for feats, label in zip(feature_batch, labels):
            x = np.asarray(feats, dtype=np.float64)
            y = clamp_01(float(label))
            p = self.predict_proba(x)
            loss_acc += -(y * math.log(max(p, 1e-8)) + (1.0 - y) * math.log(max(1.0 - p, 1e-8)))
            error = p - y
            grad_w += error * x
            grad_b += error

        grad_w = grad_w / n + self.config.l2_coeff * self.weights
        grad_b = grad_b / n
        grad_norm = float(np.linalg.norm(grad_w))

        self.weights -= self.config.learning_rate * grad_w
        self.bias -= self.config.learning_rate * grad_b
        return {
            "loss": float(loss_acc / n),
            "grad_norm": grad_norm,
        }

    def rl_finetune_step(
        self,
        feature_batch: Sequence[Sequence[float]],
        rewards: Sequence[float],
    ) -> dict[str, float]:
        """REINFORCE-style update from reward signals.

        We sample a Bernoulli decision from the estimator and move parameters in
        the direction that increases expected reward.
        """
        if not feature_batch:
            return {"avg_advantage": 0.0, "grad_norm": 0.0}

        reward_arr = np.asarray(rewards, dtype=np.float64)
        baseline = float(np.mean(reward_arr))
        grad_w = np.zeros_like(self.weights)
        grad_b = 0.0

        for feats, reward in zip(feature_batch, rewards):
            x = np.asarray(feats, dtype=np.float64)
            p = self.predict_proba(x)
            sample = float(self.rng.random() < p)
            advantage = float(reward) - baseline
            score = sample - p  # grad(log Bernoulli(pi)) wrt logits
            grad_w += advantage * score * x
            grad_b += advantage * score

        n = float(len(feature_batch))
        grad_w = grad_w / n - self.config.l2_coeff * self.weights
        grad_b = grad_b / n
        grad_norm = float(np.linalg.norm(grad_w))

        self.weights += self.config.learning_rate * grad_w
        self.bias += self.config.learning_rate * grad_b
        return {
            "avg_advantage": float(np.mean(reward_arr) - baseline),
            "grad_norm": grad_norm,
        }


@dataclass(slots=True)
class SignalEstimationLayerConfig:
    """Configuration for mixed signal estimation."""

    flow_feature_dim: int = 7
    risk_feature_dim: int = 8
    flow_mix_ratio: float = 0.7  # learned-flow vs rule-flow mixing
    default_action_reversible: float = 0.5
    default_action_failure_cost: float = 0.5
    default_action_auth_cost: float = 0.5


class SignalEstimationLayer:
    """Layer-1 module: rules for simple signals + learned estimators for latent signals."""

    def __init__(
        self,
        config: SignalEstimationLayerConfig | None = None,
        dual_state_estimator: DualStateEstimator | None = None,
        feedback_memory: FeedbackMemory | None = None,
        flow_estimator: LearnableSigmoidEstimator | None = None,
        risk_estimator: LearnableSigmoidEstimator | None = None,
    ) -> None:
        self.config = config or SignalEstimationLayerConfig()
        self.dual_state_estimator = dual_state_estimator or DualStateEstimator()
        self.feedback_memory = feedback_memory or FeedbackMemory()
        self.flow_estimator = flow_estimator or LearnableSigmoidEstimator(
            LearnableEstimatorConfig(feature_dim=self.config.flow_feature_dim, seed=123)
        )
        self.risk_estimator = risk_estimator or LearnableSigmoidEstimator(
            LearnableEstimatorConfig(feature_dim=self.config.risk_feature_dim, seed=456)
        )

    def estimate(
        self,
        *,
        event_window: Sequence[EventRecord],
        internal_signal: InternalGenerationSignal,
        p_need: float | None = None,
        p_accept: float | None = None,
        action_features: Mapping[str, Any] | None = None,
        recent_quick_rejects: Sequence[int] = (),
        quick_reject_event: bool | None = None,
    ) -> SignalEstimate:
        """Estimate all gate signals for one decision step."""
        dual = self.dual_state_estimator.estimate(event_window=event_window, internal_signal=internal_signal)

        epsilon_agent = clamp_01(1.0 - dual.epistemic_confidence)
        d_stuck = clamp_01(dual.stuck_index)
        p_need_out = clamp_01(p_need if p_need is not None else dual.need_probability)

        if quick_reject_event is not None:
            delta_rej = self.feedback_memory.update(quick_reject=quick_reject_event)
        elif recent_quick_rejects:
            delta_rej = self.feedback_memory.compute_from_flags(recent_quick_rejects)
        else:
            delta_rej = self.feedback_memory.value()

        flow_features = self.build_flow_features(event_window=event_window, dual_flow=dual.flow_index, d_stuck=d_stuck)
        learned_flow = self.flow_estimator.predict_proba(flow_features)
        f_flow = clamp_01(
            self.config.flow_mix_ratio * learned_flow + (1.0 - self.config.flow_mix_ratio) * dual.flow_index
        )

        risk_features = self.build_risk_features(
            action_features=action_features,
            flow_features=flow_features,
            epsilon_agent=epsilon_agent,
            d_stuck=d_stuck,
        )
        r_risk = self.risk_estimator.predict_proba(risk_features)

        if p_accept is None:
            # Conservative default: uncertainty/risk/rejection lower acceptance.
            p_accept_out = clamp_01(
                p_need_out
                * (1.0 - 0.25 * epsilon_agent)
                * (1.0 - 0.20 * r_risk)
                * (1.0 - 0.10 * delta_rej)
            )
        else:
            p_accept_out = clamp_01(p_accept)

        return SignalEstimate(
            f_flow=f_flow,
            d_stuck=d_stuck,
            epsilon_agent=epsilon_agent,
            delta_rej=delta_rej,
            r_risk=clamp_01(r_risk),
            p_need=p_need_out,
            p_accept=p_accept_out,
            uncertainty={
                "flow_entropy": self._bernoulli_entropy(learned_flow),
                "risk_entropy": self._bernoulli_entropy(r_risk),
                "epistemic_uncertainty": epsilon_agent,
            },
        )

    def build_flow_features(
        self,
        *,
        event_window: Sequence[EventRecord],
        dual_flow: float,
        d_stuck: float,
    ) -> list[float]:
        typing_ratio, switch_ratio, idle_ratio, avg_gap_norm = self._event_statistics(event_window)
        return [
            clamp_01(dual_flow),
            clamp_01(d_stuck),
            typing_ratio,
            switch_ratio,
            idle_ratio,
            avg_gap_norm,
            1.0,
        ]

    def build_risk_features(
        self,
        *,
        action_features: Mapping[str, Any] | None,
        flow_features: Sequence[float],
        epsilon_agent: float,
        d_stuck: float,
    ) -> list[float]:
        af = action_features or {}
        reversible = clamp_01(float(af.get("reversible", self.config.default_action_reversible)))
        failure_cost = clamp_01(float(af.get("failure_cost", self.config.default_action_failure_cost)))
        auth_cost = clamp_01(float(af.get("auth_required", self.config.default_action_auth_cost)))
        return [
            clamp_01(flow_features[0]),  # flow
            clamp_01(flow_features[3]),  # switch ratio
            clamp_01(epsilon_agent),
            clamp_01(d_stuck),
            1.0 - reversible,
            failure_cost,
            auth_cost,
            1.0,
        ]

    @staticmethod
    def _bernoulli_entropy(prob: float) -> float:
        p = clamp_01(prob)
        return float(-(p * math.log(max(p, 1e-8)) + (1.0 - p) * math.log(max(1.0 - p, 1e-8))))

    @staticmethod
    def _event_statistics(event_window: Sequence[EventRecord]) -> tuple[float, float, float, float]:
        if not event_window:
            return (0.0, 0.0, 0.5, 5.0 / 15.0)

        typing_keywords = ("type", "typing", "write", "coding", "code", "edit", "implement", "debug")
        switch_keywords = ("switch", "tab", "window", "navigate", "open", "alt-tab")
        idle_keywords = ("idle", "no specific actions", "waiting", "pause", "inactive")

        typing_count = 0
        switch_count = 0
        idle_count = 0
        for event in event_window:
            text = event.event.lower()
            if any(token in text for token in typing_keywords):
                typing_count += 1
            if any(token in text for token in switch_keywords):
                switch_count += 1
            if any(token in text for token in idle_keywords):
                idle_count += 1

        total = float(len(event_window))
        typing_ratio = typing_count / total
        switch_ratio = switch_count / total
        idle_ratio = idle_count / total

        sorted_events = sorted(event_window, key=lambda e: e.time)
        if len(sorted_events) <= 1:
            avg_gap = 5.0
        else:
            gaps = [
                max(0.0, sorted_events[i + 1].time - sorted_events[i].time)
                for i in range(len(sorted_events) - 1)
            ]
            avg_gap = sum(gaps) / max(1, len(gaps))
        avg_gap_norm = clamp_01(avg_gap / 15.0)
        return (typing_ratio, switch_ratio, idle_ratio, avg_gap_norm)
