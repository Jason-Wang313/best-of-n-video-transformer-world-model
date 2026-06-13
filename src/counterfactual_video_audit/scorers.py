"""Internal video scores and top-score selectors."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from counterfactual_video_audit.envs import GridVideoWorld


def _state_path_length(states: list[Any]) -> float:
    total = 0.0
    for prev, nxt in zip(states[:-1], states[1:]):
        total += abs(nxt.agent[0] - prev.agent[0]) + abs(nxt.agent[1] - prev.agent[1])
    return float(total)


def goal_frame_similarity(candidate: Any, world: GridVideoWorld) -> float:
    final = candidate.predicted_states[-1].agent
    dist = abs(final[0] - world.goal[0]) + abs(final[1] - world.goal[1])
    return float(1.0 - min(1.0, dist / (2.0 * (world.grid_size - 1))))


def temporal_smoothness(candidate: Any) -> float:
    speeds = []
    for prev, nxt in zip(candidate.predicted_states[:-1], candidate.predicted_states[1:]):
        speeds.append(abs(nxt.agent[0] - prev.agent[0]) + abs(nxt.agent[1] - prev.agent[1]))
    if not speeds:
        return 1.0
    speeds_arr = np.asarray(speeds, dtype=float)
    jerk = float(np.mean(np.abs(np.diff(speeds_arr)))) if len(speeds_arr) > 1 else 0.0
    overspeed = float(np.mean(np.maximum(0.0, speeds_arr - 1.0)))
    return float(np.clip(1.0 - 0.35 * jerk - 0.50 * overspeed, 0.0, 1.0))


def visual_plausibility(candidate: Any, world: GridVideoWorld) -> float:
    smooth = temporal_smoothness(candidate)
    path_length = _state_path_length(candidate.predicted_states)
    start = candidate.predicted_states[0].agent
    goal = world.goal
    shortest = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    directness = shortest / max(shortest, path_length, 1.0)
    frame_energy = float(np.mean(np.abs(np.diff(candidate.predicted_frames, axis=0))))
    low_flicker = 1.0 - min(1.0, 5.0 * frame_energy)
    return float(np.clip(0.55 * smooth + 0.30 * directness + 0.15 * low_flicker, 0.0, 1.0))


def learned_video_score(candidate: Any) -> float:
    frames = candidate.predicted_frames
    if len(frames) < 2:
        return 1.0
    frame_delta = float(np.mean(np.abs(np.diff(frames, axis=0))))
    color_range = float(np.mean(np.std(frames.reshape(len(frames), -1, 3), axis=1)))
    score = 1.0 - min(1.0, 4.0 * frame_delta) + 0.15 * min(1.0, color_range)
    return float(np.clip(score, 0.0, 1.0))


def score_candidate(candidate: Any, world: GridVideoWorld) -> dict[str, float]:
    scores = {
        "visual_plausibility": visual_plausibility(candidate, world),
        "temporal_smoothness": temporal_smoothness(candidate),
        "goal_frame_similarity": goal_frame_similarity(candidate, world),
        "learned_video_score": learned_video_score(candidate),
    }
    scores["combined_internal_score"] = combined_internal_score(scores)
    return scores


def combined_internal_score(scores: dict[str, float]) -> float:
    return float(
        0.34 * scores["visual_plausibility"]
        + 0.18 * scores["temporal_smoothness"]
        + 0.38 * scores["goal_frame_similarity"]
        + 0.10 * scores["learned_video_score"]
    )


def select_best(pool: list[Any], key: str = "combined_internal_score") -> Any:
    if not pool:
        raise ValueError("pool must be non-empty")
    return max(pool, key=lambda candidate: float(candidate.scores[key]))


def select_random(pool: list[Any], rng: np.random.Generator) -> Any:
    if not pool:
        raise ValueError("pool must be non-empty")
    return pool[int(rng.integers(0, len(pool)))]


def select_oracle(pool: list[Any]) -> Any:
    if not pool:
        raise ValueError("pool must be non-empty")
    return max(pool, key=lambda candidate: float(candidate.real_utility))


def select_with_filter(
    pool: list[Any],
    predicate: Callable[[Any], bool],
    key: str = "combined_internal_score",
) -> Any:
    kept = [candidate for candidate in pool if predicate(candidate)]
    if not kept:
        kept = pool
    return select_best(kept, key=key)
