from __future__ import annotations

from counterfactual_video_audit.envs import GridVideoWorld
from counterfactual_video_audit.evaluation import evaluate_score_selected, high_minus_low


def test_small_score_selected_evaluation_exposes_tail_pattern():
    _, summary = evaluate_score_selected(trials=6, seed=333, world=GridVideoWorld())
    assert set(summary) == {"1", "2", "4", "8", "16", "32", "64"}
    assert high_minus_low(summary, "visual_plausibility") >= 0.0
    assert high_minus_low(summary, "action_consistency_violation_rate") >= 0.0
