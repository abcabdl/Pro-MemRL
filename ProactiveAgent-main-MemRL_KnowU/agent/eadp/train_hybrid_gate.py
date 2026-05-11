"""Train hybrid proactive gate with pretraining + reward fine-tuning (RFT).

Pipeline:
1) Layer-1 weak-supervision pretraining for latent estimators:
   - flow estimator (f_flow)
   - risk estimator (r_risk)
2) Reward fine-tuning (RFT):
   - optimize alpha/beta/lambda for R(t) gate
   - jointly fine-tune flow/risk estimators by reward feedback
3) Evaluate gate metrics on validation set and export best checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jsonlines

from .dynamic_commitment_mapper import DynamicCommitmentConfig, DynamicCommitmentMapper
from .hybrid_gate_optimizer import (
    GateTrainingExample,
    HybridGateOptimizer,
    HybridGateOptimizerConfig,
)
from .signal_estimation_layer import (
    LearnableEstimatorConfig,
    LearnableSigmoidEstimator,
    SignalEstimationLayer,
    SignalEstimationLayerConfig,
)
from .types import DecisionContext, DualState, EventRecord, InternalGenerationSignal, clamp_01


@dataclass(slots=True)
class TrainSample:
    events: list[EventRecord]
    internal_signal: InternalGenerationSignal
    y_need: int
    y_accept: int
    p_need: float
    p_accept: float
    action_features: dict[str, float]
    pred_task: str | None
    category: str
    user_pref_reject: bool = False
    manual_suppressed: bool = False
    quick_reject_event: bool = False
    recent_quick_rejects: tuple[int, ...] = ()
    flow_proxy_label: int | None = None
    risk_proxy_label: int | None = None


def _pick(row: dict[str, Any], paths: list[list[str]]) -> Any:
    for path in paths:
        cur: Any = row
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def _as_binary(value: Any, *, default: int = 0) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) >= 0.5)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "accepted", "need", "needed"}:
            return 1
        if norm in {"0", "false", "no", "rejected", "none", "null", ""}:
            return 0
    return int(default)


def _to_float_time(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    try:
        return float(text)
    except Exception:
        pass
    # Fallback for natural language timestamps: Day 1, 11:00 PM
    match = re.search(r"day\s*(\d+)\s*,\s*(\d{1,2}):(\d{2})\s*([ap]m)", text, flags=re.I)
    if match:
        day = int(match.group(1))
        hour = int(match.group(2))
        minute = int(match.group(3))
        ampm = match.group(4).lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        return float((day - 1) * 24 * 60 + hour * 60 + minute)
    return fallback


def _to_events(obs: list[dict[str, Any]]) -> list[EventRecord]:
    events: list[EventRecord] = []
    for idx, item in enumerate(obs):
        events.append(
            EventRecord(
                time=_to_float_time(item.get("time"), fallback=float(idx)),
                event=str(item.get("event", "")),
            )
        )
    return events


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _derive_action_features(pred_task: str | None) -> dict[str, float]:
    if pred_task is None:
        return {
            "reversible": 1.0,
            "failure_cost": 0.1,
            "auth_required": 0.0,
        }
    text = pred_task.lower()
    high_risk_keywords = (
        "delete",
        "remove",
        "overwrite",
        "pay",
        "payment",
        "purchase",
        "transfer",
        "deploy",
        "execute",
        "run command",
        "authorize",
        "login",
        "credential",
    )
    medium_risk_keywords = ("install", "config", "change", "modify", "update", "refactor", "migrate")
    low_risk_keywords = ("summarize", "explain", "suggest", "clarify", "draft", "outline")

    if any(k in text for k in high_risk_keywords):
        return {"reversible": 0.2, "failure_cost": 0.85, "auth_required": 0.9}
    if any(k in text for k in medium_risk_keywords):
        return {"reversible": 0.5, "failure_cost": 0.6, "auth_required": 0.4}
    if any(k in text for k in low_risk_keywords):
        return {"reversible": 0.9, "failure_cost": 0.2, "auth_required": 0.1}
    return {"reversible": 0.7, "failure_cost": 0.4, "auth_required": 0.2}


def _derive_internal_signal(row: dict[str, Any], y_need: int, y_accept: int) -> InternalGenerationSignal:
    # Weak proxy for internal uncertainty when no model entropy/confidence is stored.
    entropy = _pick(row, [["generation_entropy"], ["internal_signal", "generation_entropy"]])
    confidence = _pick(row, [["generation_confidence"], ["internal_signal", "generation_confidence"]])
    if entropy is not None:
        return InternalGenerationSignal(
            generation_entropy=float(entropy),
            generation_confidence=float(confidence) if confidence is not None else None,
        )
    if y_need == 1 and y_accept == 1:
        return InternalGenerationSignal(generation_entropy=1.2, generation_confidence=0.82)
    if y_need == 1 and y_accept == 0:
        return InternalGenerationSignal(generation_entropy=3.0, generation_confidence=0.45)
    return InternalGenerationSignal(generation_entropy=1.8, generation_confidence=0.65)


def _derive_probabilities(row: dict[str, Any], y_need: int, y_accept: int) -> tuple[float, float]:
    q_need = _pick(row, [["signals", "p_need"], ["q_need"], ["teacher", "q_need"], ["scores", "q_need"]])
    q_accept = _pick(row, [["signals", "p_accept"], ["q_accept"], ["teacher", "q_accept"], ["scores", "q_accept"]])
    if q_need is not None and q_accept is not None:
        return clamp_01(float(q_need)), clamp_01(float(q_accept))

    # Fallback when only binary labels are available.
    p_need = 0.80 if y_need == 1 else 0.20
    p_accept = 0.80 if y_accept == 1 else 0.20
    return p_need, p_accept


def _derive_proxy_labels(
    *,
    events: Sequence[EventRecord],
    y_need: int,
    y_accept: int,
    pred_task: str | None,
) -> tuple[int | None, int | None]:
    # Shared event stats used by flow proxy.
    typing_keywords = ("type", "typing", "write", "coding", "code", "edit", "implement", "debug")
    switch_keywords = ("switch", "tab", "window", "navigate", "open", "alt-tab")
    idle_keywords = ("idle", "no specific actions", "waiting", "pause", "inactive")

    typing_count = 0
    switch_count = 0
    idle_count = 0
    error_count = 0
    for e in events:
        text = e.event.lower()
        if any(k in text for k in typing_keywords):
            typing_count += 1
        if any(k in text for k in switch_keywords):
            switch_count += 1
        if any(k in text for k in idle_keywords):
            idle_count += 1
        if any(k in text for k in ("error", "fail", "exception", "traceback")):
            error_count += 1
    total = float(max(1, len(events)))
    typing_ratio = typing_count / total
    switch_ratio = switch_count / total
    idle_ratio = idle_count / total

    flow_label: int | None = None
    if y_need == 0 and typing_ratio >= 0.18 and switch_ratio <= 0.45 and idle_ratio <= 0.35 and error_count == 0:
        flow_label = 1
    elif y_need == 1 and (switch_ratio >= 0.40 or idle_ratio >= 0.30 or error_count > 0):
        flow_label = 0

    risk_label: int | None = None
    if pred_task is not None:
        text = pred_task.lower()
        risky = any(
            k in text
            for k in (
                "delete",
                "overwrite",
                "pay",
                "purchase",
                "transfer",
                "deploy",
                "execute",
                "authorize",
                "login",
                "credential",
            )
        )
        if y_accept == 0 or risky:
            risk_label = 1
        elif y_accept == 1 and not risky:
            risk_label = 0

    return flow_label, risk_label


def load_samples(input_path: Path, max_samples: int | None = None) -> list[TrainSample]:
    rows = list(jsonlines.Reader(input_path.open("r", encoding="utf-8")))
    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]

    samples: list[TrainSample] = []
    for row in rows:
        y_need = _as_binary(_pick(row, [["y_need"], ["labels", "y_need"], ["help_needed"]]), default=0)
        y_accept = _as_binary(_pick(row, [["y_accept"], ["labels", "y_accept"], ["valid"]]), default=0)
        p_need, p_accept = _derive_probabilities(row, y_need, y_accept)

        obs = _pick(row, [["obs"], ["observations"], ["events"]])
        if not isinstance(obs, list):
            continue
        events = _to_events(obs)

        pred_task = _normalize_text(
            _pick(row, [["pred_task"], ["source_pred_task"], ["teacher", "proactive_task"]])
        )
        category = str(_pick(row, [["category"], ["labels", "category"]]) or "")
        internal_signal = _derive_internal_signal(row, y_need, y_accept)
        action_features = _derive_action_features(pred_task)
        flow_proxy, risk_proxy = _derive_proxy_labels(
            events=events,
            y_need=y_need,
            y_accept=y_accept,
            pred_task=pred_task,
        )
        signal_flow = _pick(row, [["signals", "f_flow"]])
        signal_risk = _pick(row, [["signals", "r_risk"]])
        signal_delta = _pick(row, [["signals", "delta_rej"]])
        if signal_flow is not None:
            flow_proxy = int(float(signal_flow) >= 0.5)
        if signal_risk is not None:
            risk_proxy = int(float(signal_risk) >= 0.5)
        quick_reject = bool((pred_task is not None and y_accept == 0) or (_as_binary(signal_delta, default=0) == 1))

        samples.append(
            TrainSample(
                events=events,
                internal_signal=internal_signal,
                y_need=y_need,
                y_accept=y_accept,
                p_need=p_need,
                p_accept=p_accept,
                action_features=action_features,
                pred_task=pred_task,
                category=category,
                quick_reject_event=quick_reject,
                recent_quick_rejects=(1,) if quick_reject else (0,),
                flow_proxy_label=flow_proxy,
                risk_proxy_label=risk_proxy,
            )
        )
    return samples


def split_train_val(
    samples: Sequence[TrainSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[TrainSample], list[TrainSample]]:
    items = list(samples)
    random.Random(seed).shuffle(items)
    val_count = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
    val = items[:val_count]
    train = items[val_count:]
    if not train and val:
        train, val = val, train
    return train, val


def batch_iter(items: Sequence[TrainSample], batch_size: int) -> Sequence[list[TrainSample]]:
    size = max(1, batch_size)
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def evaluate_gate_metrics(
    samples: Sequence[TrainSample],
    signal_layer: SignalEstimationLayer,
    mapper: DynamicCommitmentMapper,
) -> dict[str, float]:
    tp = fp = tn = fn = proposed = accepted = 0
    for sample in samples:
        signal = signal_layer.estimate(
            event_window=sample.events,
            internal_signal=sample.internal_signal,
            p_need=sample.p_need,
            p_accept=sample.p_accept,
            action_features=sample.action_features,
            recent_quick_rejects=sample.recent_quick_rejects,
        )
        state = DualState(
            flow_index=signal.f_flow,
            stuck_index=signal.d_stuck,
            epistemic_confidence=1.0 - signal.epsilon_agent,
            need_probability=signal.p_need,
        )
        decision = mapper.map_state(
            state=state,
            context=DecisionContext(
                p_need=signal.p_need,
                p_accept=signal.p_accept,
                r_risk=signal.r_risk,
                epsilon_agent=signal.epsilon_agent,
                delta_rej=signal.delta_rej,
                user_pref_reject=sample.user_pref_reject,
                manual_suppressed=sample.manual_suppressed,
            ),
        )
        intervene = bool(decision.should_intervene)
        if intervene:
            proposed += 1
            if sample.y_accept == 1:
                accepted += 1
                tp += 1
            else:
                fp += 1
        else:
            if sample.y_need == 0:
                tn += 1
            else:
                fn += 1

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    false_alarm = fp / (tp + fp + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    accept_rate = accepted / (proposed + eps)
    return {
        "Recall": recall,
        "Precision": precision,
        "Accuracy": accuracy,
        "False-Alarm": false_alarm,
        "F1-Score": f1,
        "Accept Rate": accept_rate,
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
        "Proposed": float(proposed),
        "Accepted": float(accepted),
    }


def pretrain_signal_estimators(
    *,
    train_samples: Sequence[TrainSample],
    val_samples: Sequence[TrainSample],
    signal_layer: SignalEstimationLayer,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    rng = random.Random(seed)
    for epoch in range(1, max(1, epochs) + 1):
        items = list(train_samples)
        rng.shuffle(items)
        train_flow_losses: list[float] = []
        train_risk_losses: list[float] = []

        for batch in batch_iter(items, batch_size):
            flow_feats: list[list[float]] = []
            flow_labels: list[float] = []
            risk_feats: list[list[float]] = []
            risk_labels: list[float] = []

            for sample in batch:
                dual = signal_layer.dual_state_estimator.estimate(
                    event_window=sample.events,
                    internal_signal=sample.internal_signal,
                )
                flow_features = signal_layer.build_flow_features(
                    event_window=sample.events,
                    dual_flow=dual.flow_index,
                    d_stuck=dual.stuck_index,
                )
                risk_features = signal_layer.build_risk_features(
                    action_features=sample.action_features,
                    flow_features=flow_features,
                    epsilon_agent=1.0 - dual.epistemic_confidence,
                    d_stuck=dual.stuck_index,
                )

                if sample.flow_proxy_label is not None:
                    flow_feats.append(flow_features)
                    flow_labels.append(float(sample.flow_proxy_label))
                if sample.risk_proxy_label is not None:
                    risk_feats.append(risk_features)
                    risk_labels.append(float(sample.risk_proxy_label))

            if flow_feats:
                met = signal_layer.flow_estimator.supervised_fit_step(flow_feats, flow_labels)
                train_flow_losses.append(met["loss"])
            if risk_feats:
                met = signal_layer.risk_estimator.supervised_fit_step(risk_feats, risk_labels)
                train_risk_losses.append(met["loss"])

        # Validation proxy accuracy.
        flow_correct = flow_total = 0
        risk_correct = risk_total = 0
        for sample in val_samples:
            dual = signal_layer.dual_state_estimator.estimate(
                event_window=sample.events,
                internal_signal=sample.internal_signal,
            )
            flow_features = signal_layer.build_flow_features(
                event_window=sample.events,
                dual_flow=dual.flow_index,
                d_stuck=dual.stuck_index,
            )
            risk_features = signal_layer.build_risk_features(
                action_features=sample.action_features,
                flow_features=flow_features,
                epsilon_agent=1.0 - dual.epistemic_confidence,
                d_stuck=dual.stuck_index,
            )
            if sample.flow_proxy_label is not None:
                pred = int(signal_layer.flow_estimator.predict_proba(flow_features) >= 0.5)
                flow_correct += int(pred == sample.flow_proxy_label)
                flow_total += 1
            if sample.risk_proxy_label is not None:
                pred = int(signal_layer.risk_estimator.predict_proba(risk_features) >= 0.5)
                risk_correct += int(pred == sample.risk_proxy_label)
                risk_total += 1

        row = {
            "epoch": float(epoch),
            "flow_train_loss": float(sum(train_flow_losses) / max(1, len(train_flow_losses))),
            "risk_train_loss": float(sum(train_risk_losses) / max(1, len(train_risk_losses))),
            "flow_val_acc": float(flow_correct / max(1, flow_total)),
            "risk_val_acc": float(risk_correct / max(1, risk_total)),
        }
        history.append(row)
        print(
            f"[Pretrain] epoch={epoch:02d} | "
            f"flow_loss={row['flow_train_loss']:.4f} | risk_loss={row['risk_train_loss']:.4f} | "
            f"flow_val_acc={row['flow_val_acc']:.4f} | risk_val_acc={row['risk_val_acc']:.4f}"
        )
    return history


def run_rft(
    *,
    train_samples: Sequence[TrainSample],
    val_samples: Sequence[TrainSample],
    signal_layer: SignalEstimationLayer,
    mapper: DynamicCommitmentMapper,
    optimizer: HybridGateOptimizer,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    rng = random.Random(seed)
    best_f1 = -1.0
    for epoch in range(1, max(1, epochs) + 1):
        items = list(train_samples)
        rng.shuffle(items)
        obj_before_list: list[float] = []
        obj_after_list: list[float] = []

        for batch in batch_iter(items, batch_size):
            examples: list[GateTrainingExample] = []
            for sample in batch:
                signal = signal_layer.estimate(
                    event_window=sample.events,
                    internal_signal=sample.internal_signal,
                    p_need=sample.p_need,
                    p_accept=sample.p_accept,
                    action_features=sample.action_features,
                    recent_quick_rejects=sample.recent_quick_rejects,
                    quick_reject_event=sample.quick_reject_event,
                )
                flow_features = signal_layer.build_flow_features(
                    event_window=sample.events,
                    dual_flow=signal.f_flow,
                    d_stuck=signal.d_stuck,
                )
                risk_features = signal_layer.build_risk_features(
                    action_features=sample.action_features,
                    flow_features=flow_features,
                    epsilon_agent=signal.epsilon_agent,
                    d_stuck=signal.d_stuck,
                )
                examples.append(
                    GateTrainingExample(
                        f_flow=signal.f_flow,
                        d_stuck=signal.d_stuck,
                        epsilon_agent=signal.epsilon_agent,
                        delta_rej=signal.delta_rej,
                        r_risk=signal.r_risk,
                        p_need=signal.p_need,
                        p_accept=signal.p_accept,
                        y_need=sample.y_need,
                        y_accept=sample.y_accept,
                        user_pref_reject=sample.user_pref_reject,
                        manual_suppressed=sample.manual_suppressed,
                        flow_features=flow_features,
                        risk_features=risk_features,
                    )
                )
            if examples:
                ret = optimizer.optimize(examples)
                obj_before_list.append(ret["objective_before"])
                obj_after_list.append(ret["objective_after"])

        train_metrics = evaluate_gate_metrics(train_samples, signal_layer, mapper)
        val_metrics = evaluate_gate_metrics(val_samples, signal_layer, mapper)
        best_f1 = max(best_f1, val_metrics["F1-Score"])
        row = {
            "epoch": float(epoch),
            "objective_before": float(sum(obj_before_list) / max(1, len(obj_before_list))),
            "objective_after": float(sum(obj_after_list) / max(1, len(obj_after_list))),
            "train_f1": float(train_metrics["F1-Score"]),
            "val_f1": float(val_metrics["F1-Score"]),
            "val_false_alarm": float(val_metrics["False-Alarm"]),
            "best_val_f1_so_far": best_f1,
        }
        history.append(row)
        print(
            f"[RFT] epoch={epoch:02d} | obj={row['objective_before']:.4f}->{row['objective_after']:.4f} | "
            f"train_f1={row['train_f1']:.4f} | val_f1={row['val_f1']:.4f} | "
            f"val_fa={row['val_false_alarm']:.4f}"
        )
    return history


def save_checkpoint(
    *,
    output_dir: Path,
    signal_layer: SignalEstimationLayer,
    mapper: DynamicCommitmentMapper,
    final_metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "mapper_config": {
            "r0": mapper.config.r0,
            "alpha_flow": mapper.config.alpha_flow,
            "alpha_epistemic": mapper.config.alpha_epistemic,
            "alpha_reject": mapper.config.alpha_reject,
            "beta_stuck": mapper.config.beta_stuck,
            "beta_risk": mapper.config.beta_risk,
            "epsilon_high_threshold": mapper.config.epsilon_high_threshold,
            "epsilon_low_threshold": mapper.config.epsilon_low_threshold,
            "risk_high_threshold": mapper.config.risk_high_threshold,
            "risk_low_threshold": mapper.config.risk_low_threshold,
        },
        "feedback_memory": {
            "decay_lambda": mapper.feedback_memory.config.decay_lambda,
            "horizon": mapper.feedback_memory.config.horizon,
        },
        "flow_estimator": {
            "weights": signal_layer.flow_estimator.weights.tolist(),
            "bias": signal_layer.flow_estimator.bias,
        },
        "risk_estimator": {
            "weights": signal_layer.risk_estimator.weights.tolist(),
            "bias": signal_layer.risk_estimator.bias,
        },
        "final_metrics": dict(final_metrics),
    }
    (output_dir / "hybrid_gate_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train hybrid proactive gate (pretrain + RFT).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/agent_data/rdc_topk_scored.jsonl"),
        help="Training JSONL path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=4)
    parser.add_argument("--rft-epochs", type=int, default=8)
    parser.add_argument("--rft-steps-per-call", type=int, default=1)

    parser.add_argument("--flow-lr", type=float, default=0.06)
    parser.add_argument("--risk-lr", type=float, default=0.06)
    parser.add_argument("--meta-lr", type=float, default=0.08)
    parser.add_argument("--finite-diff-eps", type=float, default=0.02)

    parser.add_argument("--alpha-flow", type=float, default=1.0)
    parser.add_argument("--alpha-epistemic", type=float, default=1.0)
    parser.add_argument("--alpha-reject", type=float, default=1.0)
    parser.add_argument("--beta-stuck", type=float, default=0.7)
    parser.add_argument("--beta-risk", type=float, default=0.7)
    parser.add_argument("--decay-lambda", type=float, default=0.5)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agent/eadp/checkpoints"),
        help="Directory to save histories and checkpoint.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=None,
        help="Optional held-out test JSONL path for final reporting.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    samples = load_samples(args.input, max_samples=args.max_samples)
    if not samples:
        raise ValueError(f"No valid samples loaded from {args.input}")
    train_samples, val_samples = split_train_val(samples, args.val_ratio, args.seed)
    print(
        f"samples_total={len(samples)} | train={len(train_samples)} | val={len(val_samples)} | "
        f"input={args.input}"
    )

    mapper = DynamicCommitmentMapper(
        config=DynamicCommitmentConfig(
            alpha_flow=float(args.alpha_flow),
            alpha_epistemic=float(args.alpha_epistemic),
            alpha_reject=float(args.alpha_reject),
            beta_stuck=float(args.beta_stuck),
            beta_risk=float(args.beta_risk),
        )
    )
    mapper.feedback_memory.config.decay_lambda = clamp_01(float(args.decay_lambda))

    flow_estimator = LearnableSigmoidEstimator(
        LearnableEstimatorConfig(feature_dim=7, learning_rate=float(args.flow_lr), seed=args.seed + 1)
    )
    risk_estimator = LearnableSigmoidEstimator(
        LearnableEstimatorConfig(feature_dim=8, learning_rate=float(args.risk_lr), seed=args.seed + 2)
    )
    signal_layer = SignalEstimationLayer(
        config=SignalEstimationLayerConfig(),
        feedback_memory=mapper.feedback_memory,
        flow_estimator=flow_estimator,
        risk_estimator=risk_estimator,
    )
    optimizer = HybridGateOptimizer(
        mapper=mapper,
        signal_layer=signal_layer,
        config=HybridGateOptimizerConfig(
            meta_learning_rate=float(args.meta_lr),
            finite_diff_eps=float(args.finite_diff_eps),
            steps_per_call=max(1, int(args.rft_steps_per_call)),
        ),
    )

    pretrain_history = pretrain_signal_estimators(
        train_samples=train_samples,
        val_samples=val_samples,
        signal_layer=signal_layer,
        epochs=max(1, int(args.pretrain_epochs)),
        batch_size=max(1, int(args.batch_size)),
        seed=args.seed,
    )

    rft_history = run_rft(
        train_samples=train_samples,
        val_samples=val_samples,
        signal_layer=signal_layer,
        mapper=mapper,
        optimizer=optimizer,
        epochs=max(1, int(args.rft_epochs)),
        batch_size=max(1, int(args.batch_size)),
        seed=args.seed + 9,
    )

    final_train_metrics = evaluate_gate_metrics(train_samples, signal_layer, mapper)
    final_val_metrics = evaluate_gate_metrics(val_samples, signal_layer, mapper)
    print("=" * 100)
    print("Final Gate Metrics")
    print("=" * 100)
    for name, m in (("Train", final_train_metrics), ("Val", final_val_metrics)):
        print(
            f"{name:<5} | Recall={m['Recall']:.4f} | Precision={m['Precision']:.4f} | "
            f"Accuracy={m['Accuracy']:.4f} | False-Alarm={m['False-Alarm']:.4f} | "
            f"F1={m['F1-Score']:.4f} | Accept={m['Accept Rate']:.4f}"
        )

    if args.test_input is not None:
        test_samples = load_samples(args.test_input)
        test_metrics = evaluate_gate_metrics(test_samples, signal_layer, mapper)
        print(
            f"Test  | Recall={test_metrics['Recall']:.4f} | Precision={test_metrics['Precision']:.4f} | "
            f"Accuracy={test_metrics['Accuracy']:.4f} | False-Alarm={test_metrics['False-Alarm']:.4f} | "
            f"F1={test_metrics['F1-Score']:.4f} | Accept={test_metrics['Accept Rate']:.4f}"
        )
    print("=" * 100)

    save_checkpoint(
        output_dir=args.output_dir,
        signal_layer=signal_layer,
        mapper=mapper,
        final_metrics={
            "train": final_train_metrics,
            "val": final_val_metrics,
            "pretrain_history": pretrain_history,
            "rft_history": rft_history,
        },
    )
    print(f"checkpoint_saved={args.output_dir / 'hybrid_gate_checkpoint.json'}")


if __name__ == "__main__":
    main()
