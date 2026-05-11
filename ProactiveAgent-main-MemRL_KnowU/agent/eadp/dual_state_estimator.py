"""Dual-state estimator for EADP.

This module jointly estimates:
1) user cognitive state (flow vs stuck),
2) user proactive-help need probability, and
3) agent epistemic confidence (hallucination risk proxy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .types import DualState, EventRecord, InternalGenerationSignal, clamp_01


@dataclass(slots=True)
class DualStateEstimatorConfig:
    """Configuration knobs for the heuristic estimator."""

    max_expected_entropy: float = 5.0
    typing_keywords: tuple[str, ...] = (
        "type",
        "typing",
        "write",
        "coding",
        "code",
        "editing",
        "implement",
        "debug",
    )
    switch_keywords: tuple[str, ...] = (
        "switch",
        "tab",
        "window",
        "navigate",
        "open",
        "alt-tab",
    )
    idle_keywords: tuple[str, ...] = (
        "idle",
        "no specific actions",
        "waiting",
        "pause",
        "inactive",
    )


@dataclass(slots=True)
class _BehaviorFeatures:
    """Intermediate feature container from event streams."""

    typing_ratio: float
    switch_ratio: float
    idle_ratio: float
    avg_inter_event_gap: float


class DualStateEstimator:
    """Estimate user and agent state from recent events and model signals."""

    def __init__(self, config: DualStateEstimatorConfig | None = None) -> None:
        self.config = config or DualStateEstimatorConfig()

    def estimate(
        self,
        event_window: Sequence[EventRecord],
        internal_signal: InternalGenerationSignal,
    ) -> DualState:
        """Estimate `flow_index`, `stuck_index`, `need_probability`, and epistemic confidence.

        Args:
            event_window: Sliding window from the ProactiveBench event stream.
            internal_signal: Entropy/confidence signals from the language model.

        Returns:
            DualState object with values clamped to [0, 1].
        """
        features = self._extract_behavior_features(event_window)
        flow_index = self._compute_flow_index(features)
        stuck_index = self._compute_stuck_index(features)
        need_probability = self._compute_need_probability(features, stuck_index)
        epistemic_confidence = self._compute_epistemic_confidence(internal_signal)
        return DualState(
            flow_index=clamp_01(flow_index),
            stuck_index=clamp_01(stuck_index),
            epistemic_confidence=clamp_01(epistemic_confidence),
            need_probability=clamp_01(need_probability),
        )

    def _extract_behavior_features(
        self,
        event_window: Sequence[EventRecord],
    ) -> _BehaviorFeatures:
        """Convert raw event text into coarse behavior statistics."""
        if not event_window:
            # Neutral default when no observations are available.
            return _BehaviorFeatures(
                typing_ratio=0.0,
                switch_ratio=0.0,
                idle_ratio=0.5,
                avg_inter_event_gap=5.0,
            )

        typing_count = 0
        switch_count = 0
        idle_count = 0
        for event in event_window:
            text = event.event.lower()
            if any(token in text for token in self.config.typing_keywords):
                typing_count += 1
            if any(token in text for token in self.config.switch_keywords):
                switch_count += 1
            if any(token in text for token in self.config.idle_keywords):
                idle_count += 1

        event_count = float(len(event_window))
        typing_ratio = typing_count / event_count
        switch_ratio = switch_count / event_count
        idle_ratio = idle_count / event_count

        sorted_events = sorted(event_window, key=lambda e: e.time)
        if len(sorted_events) <= 1:
            avg_gap = 5.0
        else:
            gaps = [
                max(0.0, sorted_events[i + 1].time - sorted_events[i].time)
                for i in range(len(sorted_events) - 1)
            ]
            avg_gap = sum(gaps) / max(1, len(gaps))

        return _BehaviorFeatures(
            typing_ratio=typing_ratio,
            switch_ratio=switch_ratio,
            idle_ratio=idle_ratio,
            avg_inter_event_gap=avg_gap,
        )

    def _compute_flow_index(self, features: _BehaviorFeatures) -> float:
        """High flow: frequent productive activity, low context switching."""
        gap_stability = 1.0 - clamp_01(features.avg_inter_event_gap / 15.0)
        raw_flow = (
            0.55 * features.typing_ratio
            + 0.30 * gap_stability
            - 0.15 * features.switch_ratio
            - 0.20 * features.idle_ratio
        )
        return raw_flow + 0.20

    def _compute_stuck_index(self, features: _BehaviorFeatures) -> float:
        """High stuck: frequent switches, inactivity, and sparse progress."""
        long_gap_pressure = clamp_01(features.avg_inter_event_gap / 15.0)
        raw_stuck = (
            0.45 * features.switch_ratio
            + 0.35 * features.idle_ratio
            + 0.30 * long_gap_pressure
            - 0.30 * features.typing_ratio
        )
        return raw_stuck + 0.10

    def _compute_epistemic_confidence(
        self,
        internal_signal: InternalGenerationSignal,
    ) -> float:
        """Convert generation entropy/confidence into epistemic confidence."""
        entropy_term = 1.0 - clamp_01(
            internal_signal.generation_entropy / self.config.max_expected_entropy
        )
        if internal_signal.generation_confidence is None:
            return entropy_term
        confidence_term = clamp_01(internal_signal.generation_confidence)
        return 0.5 * entropy_term + 0.5 * confidence_term

    def _compute_need_probability(
        self,
        features: _BehaviorFeatures,
        stuck_index: float,
    ) -> float:
        """Estimate intervention-need probability from behavior friction signals."""
        gap_pressure = clamp_01(features.avg_inter_event_gap / 15.0)
        raw_need = (
            0.60 * clamp_01(stuck_index)
            + 0.20 * features.idle_ratio
            + 0.10 * features.switch_ratio
            + 0.10 * gap_pressure
        )
        return raw_need
