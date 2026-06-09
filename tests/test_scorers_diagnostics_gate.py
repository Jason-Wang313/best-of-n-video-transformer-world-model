from __future__ import annotations

from video_transformer_best_of_n.candidates import sample_candidate_pool
from video_transformer_best_of_n.config import GATE_LABELS
from video_transformer_best_of_n.envs import GridVideoWorld
from video_transformer_best_of_n.gate import deployment_gate
from video_transformer_best_of_n.scorers import select_best


def test_scorers_and_diagnostics_are_sane():
    world = GridVideoWorld()
    pool = sample_candidate_pool(16, seed=123, world=world)
    selected = select_best(pool)
    for key, value in selected.scores.items():
        assert 0.0 <= value <= 1.0, key
    for key, value in selected.diagnostics.items():
        assert 0.0 <= value <= 1.0, key
    assert selected.real_utility <= selected.predicted_utility or selected.trap_type == "valid_detour"


def test_deployment_gate_returns_exactly_one_valid_label():
    label = deployment_gate(
        n=64,
        action_violation_rate=0.25,
        temporal_causal_violation_rate=0.0,
        occlusion_uncertainty=0.4,
        frame_to_state_consistency_error=0.1,
        pilot_label_count=32,
        calibration_mae=0.2,
    )
    assert isinstance(label, str)
    assert label in GATE_LABELS
    assert label == "require_action_consistency_check"

    assert (
        deployment_gate(
            n=64,
            action_violation_rate=0.0,
            temporal_causal_violation_rate=0.0,
            occlusion_uncertainty=0.1,
            frame_to_state_consistency_error=0.0,
            pilot_label_count=32,
            calibration_mae=0.1,
        )
        == "allow_high_n"
    )
