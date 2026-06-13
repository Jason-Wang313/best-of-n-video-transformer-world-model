"""Deployment gate for large-candidate-count video selection."""

from __future__ import annotations

from counterfactual_video_audit.config import GATE_LABELS


def deployment_gate(
    *,
    n: int,
    action_violation_rate: float,
    temporal_causal_violation_rate: float,
    occlusion_uncertainty: float,
    frame_to_state_consistency_error: float,
    pilot_label_count: int,
    calibration_mae: float | None = None,
) -> str:
    """Return exactly one scoped large-candidate-count deployment label."""

    n = int(n)
    if action_violation_rate >= 0.55 or frame_to_state_consistency_error >= 0.55:
        return "block_high_n"
    if action_violation_rate >= 0.18:
        return "require_action_consistency_check"
    if temporal_causal_violation_rate >= 0.18:
        return "stop_early"
    if n >= 16 and pilot_label_count < 16:
        return "collect_pilot_labels"
    if n >= 32 and occlusion_uncertainty >= 0.42 and (calibration_mae is None or calibration_mae > 0.45):
        return "collect_pilot_labels"
    if calibration_mae is not None and calibration_mae > 0.90:
        return "stop_early"
    return "allow_high_n"


def valid_gate_label(label: str) -> bool:
    return label in GATE_LABELS
