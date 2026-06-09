"""Candidate future generation for the controlled video world."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from video_transformer_best_of_n.envs import (
    DOWN,
    RIGHT,
    STAY,
    GridVideoWorld,
    State,
    direct_wall_actions,
    near_miss_actions,
    random_actions,
    temporal_trap_actions,
    valid_detour_actions,
)

_CANDIDATE_CACHE: dict[tuple[object, ...], VideoCandidate] = {}


@dataclass
class VideoCandidate:
    """One generated future and its action-conditioned real execution."""

    actions: np.ndarray
    predicted_states: list[State]
    true_states: list[State]
    predicted_frames: np.ndarray
    true_frames: np.ndarray
    trap_type: str
    real_utility: float
    predicted_utility: float
    scores: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


def _world_cache_key(world: GridVideoWorld) -> tuple[object, ...]:
    return (
        world.grid_size,
        world.horizon,
        world.frame_size,
        world.wall_x,
        world.door_y,
        world.occlusion_cols,
    )


def _shifted_temporal_states(world: GridVideoWorld, actions: np.ndarray, start: State) -> list[State]:
    shifted = list(actions[1:]) + [STAY]
    states = world.rollout_ignore_walls(shifted, start=start)
    if len(states) > 4:
        states[2] = State(agent=(min(world.grid_size - 1, states[2].agent[0] + 1), states[2].agent[1]), goal=start.goal)
    return states


def predicted_rollout_for_trap(
    world: GridVideoWorld,
    actions: np.ndarray,
    trap_type: str,
    start: State | None = None,
) -> list[State]:
    start_state = start or world.initial_state()
    if trap_type in {"valid_detour", "near_miss", "random"}:
        return world.rollout(actions, start=start_state)
    if trap_type in {"ghost_wall", "occlusion_skip"}:
        return world.rollout_ignore_walls(actions, start=start_state)
    if trap_type == "temporal_causal":
        return _shifted_temporal_states(world, actions, start_state)
    raise ValueError(f"unknown trap_type {trap_type}")


def sample_candidate_type(rng: np.random.Generator) -> str:
    types = np.asarray(["valid_detour", "near_miss", "random", "ghost_wall", "occlusion_skip", "temporal_causal"])
    probs = np.asarray([0.32, 0.14, 0.04, 0.25, 0.18, 0.07], dtype=float)
    return str(rng.choice(types, p=probs))


def actions_for_type(world: GridVideoWorld, trap_type: str, rng: np.random.Generator) -> np.ndarray:
    if trap_type == "valid_detour":
        return valid_detour_actions(world.horizon)
    if trap_type in {"ghost_wall", "occlusion_skip"}:
        return direct_wall_actions(world.horizon)
    if trap_type == "temporal_causal":
        return temporal_trap_actions(world.horizon)
    if trap_type == "near_miss":
        return near_miss_actions(world.horizon)
    if trap_type == "random":
        return random_actions(rng, world.horizon)
    raise ValueError(f"unknown trap_type {trap_type}")


def make_candidate(world: GridVideoWorld, trap_type: str, actions: np.ndarray) -> VideoCandidate:
    from video_transformer_best_of_n.diagnostics import candidate_diagnostics
    from video_transformer_best_of_n.scorers import score_candidate

    actions = np.asarray(actions, dtype=int)
    key = (_world_cache_key(world), str(trap_type), tuple(int(a) for a in actions))
    cached = _CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached
    start = world.initial_state()
    true_states = world.rollout(actions, start=start)
    predicted_states = predicted_rollout_for_trap(world, actions, trap_type, start=start)
    true_frames = world.render_rollout(true_states, show_occluder=True)
    predicted_frames = world.render_rollout(predicted_states, show_occluder=True)
    candidate = VideoCandidate(
        actions=np.asarray(actions, dtype=int),
        predicted_states=predicted_states,
        true_states=true_states,
        predicted_frames=predicted_frames,
        true_frames=true_frames,
        trap_type=trap_type,
        real_utility=world.utility_from_states(true_states, actions),
        predicted_utility=world.utility_from_states(predicted_states, actions),
    )
    candidate.scores = score_candidate(candidate, world)
    candidate.diagnostics = candidate_diagnostics(candidate, world)
    _CANDIDATE_CACHE[key] = candidate
    return candidate


def sample_candidate(world: GridVideoWorld, rng: np.random.Generator) -> VideoCandidate:
    trap_type = sample_candidate_type(rng)
    return make_candidate(world, trap_type, actions_for_type(world, trap_type, rng))


def sample_candidate_pool(n: int, seed: int, world: GridVideoWorld | None = None) -> list[VideoCandidate]:
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    world = world or GridVideoWorld()
    rng = np.random.default_rng(seed)
    return [sample_candidate(world, rng) for _ in range(int(n))]


def candidate_validity_overlay(candidate: VideoCandidate, world: GridVideoWorld) -> str:
    symbols = []
    for prev, nxt, action in zip(candidate.predicted_states[:-1], candidate.predicted_states[1:], candidate.actions):
        expected = world.transition(prev, int(action))
        if expected.agent == nxt.agent:
            symbols.append(".")
        elif int(action) in {RIGHT, DOWN, STAY}:
            symbols.append("x")
        else:
            symbols.append("!")
    return "".join(symbols)
