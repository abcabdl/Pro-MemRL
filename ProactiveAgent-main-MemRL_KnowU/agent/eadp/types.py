"""Shared datatypes for Epistemic-Aware Dynamic Proactivity (EADP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


def clamp_01(value: float) -> float:
    """Clamp a numeric value into the [0, 1] interval."""
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class EventRecord:
    """Single event from a ProactiveBench-style event stream."""

    time: float
    event: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EventRecord":
        """Build an event record from a dict-like payload."""
        return cls(time=float(raw["time"]), event=str(raw["event"]))


@dataclass(slots=True)
class InternalGenerationSignal:
    """Model-side uncertainty signals used for epistemic confidence estimation."""

    generation_entropy: float
    generation_confidence: Optional[float] = None


@dataclass(slots=True)
class DualState:
    """Joint estimate of human state and agent epistemic state."""

    flow_index: float
    stuck_index: float
    epistemic_confidence: float
    need_probability: float = 0.5

    def as_tuple(self) -> tuple[float, float, float]:
        """Return state values in tuple form."""
        return (self.flow_index, self.stuck_index, self.epistemic_confidence)


@dataclass(slots=True)
class MappingDecision:
    """Policy output for behavioral commitment selection."""

    commitment_level: int
    reason: str
    should_intervene: bool = False
    r_value: float = 0.0
    tau: float = 1.0
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionContext:
    """Auxiliary signals for two-stage intervention and commitment decisions."""

    p_need: Optional[float] = None
    p_accept: Optional[float] = None
    r_risk: float = 0.5
    epsilon_agent: Optional[float] = None
    delta_rej: Optional[float] = None
    recent_quick_rejects: Sequence[int] = ()
    user_pref_reject: bool = False
    manual_suppressed: bool = False
    auth_granted: bool = False
    can_rollback_or_preview: bool = False
    target_unambiguous: bool = False
    action_features: dict[str, Any] = field(default_factory=dict)
    quick_reject_event: Optional[bool] = None


@dataclass(slots=True)
class SignalEstimate:
    """Layer-1 signal estimation outputs used by dynamic gating."""

    f_flow: float
    d_stuck: float
    epsilon_agent: float
    delta_rej: float
    r_risk: float
    p_need: float
    p_accept: float
    uncertainty: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RewardBreakdown:
    """Reward output with diagnostics to support RL debugging."""

    total_reward: float
    components: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MockScenario:
    """Synthetic scenario used by the mock training loop.

    RDC-related fields are optional and are consumed by RDC filtering when
    available.
    """

    name: str
    events: list[EventRecord]
    internal_signal: InternalGenerationSignal
    task_success: bool
    clarification_success: bool = False
    y_need: Optional[int] = None
    y_accept: Optional[int] = None
    q_need: Optional[float] = None
    q_accept: Optional[float] = None
    y_need_pred: Optional[int] = None
    teacher_response: str = ""
    # Stage-1/2 decision context fields for mock execution.
    r_risk: float = 0.5
    epsilon_agent: Optional[float] = None
    delta_rej: Optional[float] = None
    recent_quick_rejects: Sequence[int] = ()
    user_pref_reject: bool = False
    manual_suppressed: bool = False
    auth_granted: bool = False
    can_rollback_or_preview: bool = False
    target_unambiguous: bool = False
    action_features: dict[str, Any] = field(default_factory=dict)
    quick_reject_event: Optional[bool] = None


@dataclass(slots=True)
class ObjectiveVector:
    """Multi-objective reward vector before scalarization."""

    task_success: float = 0.0
    non_interruption: float = 0.0
    epistemic_safety: float = 0.0
    helpfulness: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """Return objective vector as a plain dict."""
        return {
            "task_success": self.task_success,
            "non_interruption": self.non_interruption,
            "epistemic_safety": self.epistemic_safety,
            "helpfulness": self.helpfulness,
        }


@dataclass(slots=True)
class PolicyStep:
    """One policy interaction step for policy-gradient training."""

    state: DualState
    action: int
    log_prob: float
    reward: float
    objective_vector: ObjectiveVector = field(default_factory=ObjectiveVector)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyBatch:
    """A mini-batch of policy steps."""

    steps: list[PolicyStep] = field(default_factory=list)

    def add(self, step: PolicyStep) -> None:
        """Append one trajectory step."""
        self.steps.append(step)

    def __len__(self) -> int:
        """Return number of stored steps."""
        return len(self.steps)
