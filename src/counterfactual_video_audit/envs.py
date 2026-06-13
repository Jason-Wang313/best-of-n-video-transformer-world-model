"""A tiny action-conditioned video world with occlusion and wall traps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

STAY, UP, DOWN, LEFT, RIGHT = range(5)
ACTION_NAMES = {
    STAY: "stay",
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
}
ACTION_DELTAS = {
    STAY: (0, 0),
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}


@dataclass(frozen=True)
class State:
    """Grid state rendered as a low-resolution RGB frame."""

    agent: tuple[int, int]
    goal: tuple[int, int] = (6, 4)


class GridVideoWorld:
    """A deterministic video world with a hidden action-consistency trap.

    The agent starts left of a vertical wall. The shortest-looking path is to
    move right, but the real dynamics block that route. A longer detour through
    the upper door is action-valid. Generated videos may hallucinate the direct
    route as smooth and goal-reaching, especially while the crossing is hidden
    by a grey occluder.
    """

    def __init__(
        self,
        grid_size: int = 8,
        horizon: int = 11,
        frame_size: int = 32,
        wall_x: int = 3,
        door_y: int = 1,
        occlusion_cols: tuple[int, int] = (3, 4),
    ) -> None:
        if grid_size < 7:
            raise ValueError("grid_size must be at least 7")
        if frame_size % grid_size != 0:
            raise ValueError("frame_size must be divisible by grid_size")
        self.grid_size = int(grid_size)
        self.horizon = int(horizon)
        self.frame_size = int(frame_size)
        self.wall_x = int(wall_x)
        self.door_y = int(door_y)
        self.occlusion_cols = tuple(int(c) for c in occlusion_cols)
        self.start = (1, 4)
        self.goal = (6, 4)

    def initial_state(self) -> State:
        return State(agent=self.start, goal=self.goal)

    def cell_size(self) -> int:
        return self.frame_size // self.grid_size

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_wall(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return x == self.wall_x and y != self.door_y and 1 <= y <= self.grid_size - 2

    def transition(self, state: State, action: int) -> State:
        if int(action) not in ACTION_DELTAS:
            raise ValueError(f"unknown action {action}")
        dx, dy = ACTION_DELTAS[int(action)]
        candidate = (state.agent[0] + dx, state.agent[1] + dy)
        if not self.in_bounds(candidate) or self.is_wall(candidate):
            candidate = state.agent
        return State(agent=candidate, goal=state.goal)

    def rollout(self, actions: Iterable[int], start: State | None = None) -> list[State]:
        state = start or self.initial_state()
        states = [state]
        for action in actions:
            state = self.transition(state, int(action))
            states.append(state)
        return states

    def rollout_ignore_walls(self, actions: Iterable[int], start: State | None = None) -> list[State]:
        state = start or self.initial_state()
        states = [state]
        for action in actions:
            dx, dy = ACTION_DELTAS[int(action)]
            candidate = (
                int(np.clip(state.agent[0] + dx, 0, self.grid_size - 1)),
                int(np.clip(state.agent[1] + dy, 0, self.grid_size - 1)),
            )
            state = State(agent=candidate, goal=state.goal)
            states.append(state)
        return states

    def real_utility(self, actions: Iterable[int], start: State | None = None) -> float:
        actions_arr = [int(a) for a in actions]
        states = self.rollout(actions_arr, start=start)
        return self.utility_from_states(states, actions_arr)

    def utility_from_states(self, states: list[State], actions: Iterable[int] | None = None) -> float:
        final = states[-1].agent
        dist = abs(final[0] - states[-1].goal[0]) + abs(final[1] - states[-1].goal[1])
        success_bonus = 3.0 if dist == 0 else 0.0
        progress = -0.35 * dist
        step_cost = -0.02 * (len(states) - 1)
        blocked_penalty = 0.0
        if actions is not None:
            actions_arr = [int(a) for a in actions]
            true_states = self.rollout(actions_arr, start=states[0])
            for prev, nxt, action in zip(true_states[:-1], true_states[1:], actions_arr):
                if action != STAY and prev.agent == nxt.agent:
                    blocked_penalty -= 0.05
        return float(success_bonus + progress + step_cost + blocked_penalty)

    def _fill_cell(self, frame: np.ndarray, pos: tuple[int, int], color: tuple[float, float, float], inset: int = 0) -> None:
        cs = self.cell_size()
        x, y = pos
        x0 = x * cs + inset
        y0 = y * cs + inset
        x1 = (x + 1) * cs - inset
        y1 = (y + 1) * cs - inset
        frame[y0:y1, x0:x1, :] = np.asarray(color, dtype=np.float32)

    def render_state(self, state: State, show_occluder: bool = True) -> np.ndarray:
        frame = np.ones((self.frame_size, self.frame_size, 3), dtype=np.float32)
        frame[:, :, :] = np.array([0.94, 0.95, 0.92], dtype=np.float32)
        cs = self.cell_size()
        for i in range(self.grid_size + 1):
            frame[i * cs - 1 if i else 0 : i * cs + 1 if i else 1, :, :] = 0.82
            frame[:, i * cs - 1 if i else 0 : i * cs + 1 if i else 1, :] = 0.82
        for y in range(1, self.grid_size - 1):
            if y != self.door_y:
                self._fill_cell(frame, (self.wall_x, y), (0.12, 0.13, 0.15), inset=1)
        self._fill_cell(frame, state.goal, (0.20, 0.70, 0.32), inset=1)
        self._fill_cell(frame, state.agent, (0.05, 0.28, 0.88), inset=max(1, cs // 5))
        if show_occluder:
            lo, hi = min(self.occlusion_cols), max(self.occlusion_cols)
            x0 = lo * cs
            x1 = (hi + 1) * cs
            y0 = 2 * cs
            y1 = 7 * cs
            frame[y0:y1, x0:x1, :] = 0.55 * frame[y0:y1, x0:x1, :] + 0.45 * np.array([0.58, 0.60, 0.62])
        return np.clip(frame, 0.0, 1.0)

    def render_rollout(self, states: list[State], show_occluder: bool = True) -> np.ndarray:
        return np.stack([self.render_state(state, show_occluder=show_occluder) for state in states], axis=0)

    def action_names(self, actions: Iterable[int]) -> list[str]:
        return [ACTION_NAMES[int(action)] for action in actions]


def valid_detour_actions(horizon: int = 11) -> np.ndarray:
    actions = [UP, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN]
    return _pad_actions(actions, horizon)


def direct_wall_actions(horizon: int = 11) -> np.ndarray:
    actions = [RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, STAY, STAY, STAY, STAY, STAY, STAY]
    return _pad_actions(actions, horizon)


def near_miss_actions(horizon: int = 11) -> np.ndarray:
    actions = [UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, DOWN, RIGHT, DOWN, STAY, STAY]
    return _pad_actions(actions, horizon)


def temporal_trap_actions(horizon: int = 11) -> np.ndarray:
    actions = [STAY, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, STAY, STAY, STAY, STAY, STAY]
    return _pad_actions(actions, horizon)


def random_actions(rng: np.random.Generator, horizon: int = 11) -> np.ndarray:
    probs = np.array([0.10, 0.16, 0.16, 0.08, 0.50], dtype=float)
    return rng.choice(np.arange(5), size=int(horizon), p=probs)


def _pad_actions(actions: list[int], horizon: int) -> np.ndarray:
    if len(actions) < horizon:
        actions = actions + [STAY] * (horizon - len(actions))
    return np.asarray(actions[:horizon], dtype=int)
