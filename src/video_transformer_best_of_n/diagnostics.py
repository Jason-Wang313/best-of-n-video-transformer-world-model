"""Video-specific diagnostics for generated futures."""

from __future__ import annotations

from typing import Any

import numpy as np

from video_transformer_best_of_n.envs import STAY, GridVideoWorld, State


def action_consistency_violations(world: GridVideoWorld, states: list[State], actions: np.ndarray) -> np.ndarray:
    violations: list[float] = []
    reachable = states[0]
    already_unreachable = False
    for observed_prev, observed_next, action in zip(states[:-1], states[1:], actions):
        expected_next = world.transition(reachable, int(action))
        local_bad = (
            already_unreachable
            or world.is_wall(observed_prev.agent)
            or world.is_wall(observed_next.agent)
            or expected_next.agent != observed_next.agent
        )
        violations.append(1.0 if local_bad else 0.0)
        if local_bad:
            already_unreachable = True
            reachable = expected_next
        else:
            reachable = observed_next
    return np.asarray(violations, dtype=float)


def action_consistency_violation_rate(world: GridVideoWorld, states: list[State], actions: np.ndarray) -> float:
    violations = action_consistency_violations(world, states, actions)
    if violations.size == 0:
        return 0.0
    return float(np.mean(violations))


def temporal_causal_violation_rate(states: list[State], actions: np.ndarray) -> float:
    bad = 0
    total = max(1, len(actions))
    for prev, nxt, action in zip(states[:-1], states[1:], actions):
        dx = nxt.agent[0] - prev.agent[0]
        dy = nxt.agent[1] - prev.agent[1]
        moved = abs(dx) + abs(dy)
        if moved > 1:
            bad += 1
        elif int(action) == STAY and moved > 0:
            bad += 1
    return float(bad / total)


def occlusion_uncertainty(world: GridVideoWorld, states: list[State]) -> float:
    lo, hi = min(world.occlusion_cols), max(world.occlusion_cols)
    uncertain = []
    for state in states:
        x, y = state.agent
        near_occluder = lo - 1 <= x <= hi + 1 and 2 <= y <= 6
        uncertain.append(1.0 if near_occluder else 0.0)
    return float(np.mean(uncertain)) if uncertain else 0.0


def estimate_agent_from_frame(frame: np.ndarray, world: GridVideoWorld) -> tuple[int, int] | None:
    blue = frame[:, :, 2] - 0.5 * (frame[:, :, 0] + frame[:, :, 1])
    mask = blue > 0.28
    if int(mask.sum()) < max(3, world.cell_size() // 2):
        return None
    ys, xs = np.nonzero(mask)
    cx = int(np.clip(np.round(xs.mean() / world.cell_size() - 0.5), 0, world.grid_size - 1))
    cy = int(np.clip(np.round(ys.mean() / world.cell_size() - 0.5), 0, world.grid_size - 1))
    return (cx, cy)


def frame_to_state_consistency_error(frames: np.ndarray, states: list[State], world: GridVideoWorld) -> float:
    if len(states) == 0:
        return 0.0
    errors = []
    denom = 2.0 * (world.grid_size - 1)
    for frame, state in zip(frames, states):
        estimate = estimate_agent_from_frame(frame, world)
        if estimate is None:
            errors.append(1.0)
        else:
            dist = abs(estimate[0] - state.agent[0]) + abs(estimate[1] - state.agent[1])
            errors.append(min(1.0, dist / denom))
    return float(np.mean(errors))


def candidate_diagnostics(candidate: Any, world: GridVideoWorld) -> dict[str, float]:
    return {
        "action_consistency_violation_rate": action_consistency_violation_rate(
            world, candidate.predicted_states, candidate.actions
        ),
        "temporal_causal_violation_rate": temporal_causal_violation_rate(candidate.predicted_states, candidate.actions),
        "occlusion_uncertainty": occlusion_uncertainty(world, candidate.predicted_states),
        "frame_to_state_consistency_error": frame_to_state_consistency_error(
            candidate.predicted_frames, candidate.predicted_states, world
        ),
    }


def pool_diagnostics(pool: list[Any], world: GridVideoWorld) -> dict[str, float]:
    if not pool:
        return {
            "action_consistency_violation_rate": 0.0,
            "temporal_causal_violation_rate": 0.0,
            "occlusion_uncertainty": 0.0,
            "frame_to_state_consistency_error": 0.0,
        }
    keys = list(candidate_diagnostics(pool[0], world).keys()) if not getattr(pool[0], "diagnostics", None) else list(pool[0].diagnostics.keys())
    return {key: float(np.mean([candidate.diagnostics[key] for candidate in pool])) for key in keys}
