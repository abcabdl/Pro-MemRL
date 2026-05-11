"""Epistemic-Aware Dynamic Proactivity (EADP) package."""

from .dual_state_estimator import DualStateEstimator, DualStateEstimatorConfig
from .dynamic_commitment_mapper import DynamicCommitmentConfig, DynamicCommitmentMapper
from .feedback_memory import FeedbackMemory, FeedbackMemoryConfig
from .hybrid_gate_optimizer import (
    GateTrainingExample,
    HybridGateOptimizer,
    HybridGateOptimizerConfig,
)
from .runtime_feedback import RUNTIME_FEEDBACK, RuntimeFeedbackMemory
from .runtime_policy import normalize_operation, resolve_operation
from .signal_estimation_layer import (
    LearnableEstimatorConfig,
    LearnableSigmoidEstimator,
    SignalEstimationLayer,
    SignalEstimationLayerConfig,
)
from .types import (
    DecisionContext,
    DualState,
    EventRecord,
    InternalGenerationSignal,
    MappingDecision,
    SignalEstimate,
)

__all__ = [
    "DecisionContext",
    "DualState",
    "DualStateEstimator",
    "DualStateEstimatorConfig",
    "DynamicCommitmentConfig",
    "DynamicCommitmentMapper",
    "FeedbackMemory",
    "FeedbackMemoryConfig",
    "EventRecord",
    "GateTrainingExample",
    "HybridGateOptimizer",
    "HybridGateOptimizerConfig",
    "InternalGenerationSignal",
    "LearnableEstimatorConfig",
    "LearnableSigmoidEstimator",
    "MappingDecision",
    "RUNTIME_FEEDBACK",
    "RuntimeFeedbackMemory",
    "normalize_operation",
    "resolve_operation",
    "SignalEstimate",
    "SignalEstimationLayer",
    "SignalEstimationLayerConfig",
]
