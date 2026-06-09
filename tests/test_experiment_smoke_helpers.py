from __future__ import annotations

from video_transformer_best_of_n.envs import GridVideoWorld
from video_transformer_best_of_n.evaluation import evaluate_best_of_n, high_minus_low


def test_small_best_of_n_evaluation_exposes_tail_pattern():
    _, summary = evaluate_best_of_n(trials=6, seed=333, world=GridVideoWorld())
    assert set(summary) == {"1", "2", "4", "8", "16", "32", "64"}
    assert high_minus_low(summary, "visual_plausibility") >= 0.0
    assert high_minus_low(summary, "action_consistency_violation_rate") >= 0.0
