from __future__ import annotations

from counterfactual_video_audit.candidates import make_candidate
from counterfactual_video_audit.envs import GridVideoWorld, direct_wall_actions, valid_detour_actions


def test_valid_detour_reaches_goal_and_direct_path_is_blocked():
    world = GridVideoWorld()
    valid = world.rollout(valid_detour_actions(world.horizon))
    direct = world.rollout(direct_wall_actions(world.horizon))
    assert valid[-1].agent == world.goal
    assert direct[-1].agent != world.goal
    assert world.utility_from_states(valid, valid_detour_actions(world.horizon)) > world.utility_from_states(
        direct, direct_wall_actions(world.horizon)
    )


def test_ghost_candidate_is_visually_goal_like_but_action_invalid():
    world = GridVideoWorld()
    candidate = make_candidate(world, "ghost_wall", direct_wall_actions(world.horizon))
    assert candidate.predicted_states[-1].agent == world.goal
    assert candidate.true_states[-1].agent != world.goal
    assert candidate.diagnostics["action_consistency_violation_rate"] > 0.0
    assert candidate.scores["goal_frame_similarity"] > 0.95
    assert candidate.predicted_frames.shape == candidate.true_frames.shape
    assert candidate.predicted_frames.shape[1:] == (world.frame_size, world.frame_size, 3)
