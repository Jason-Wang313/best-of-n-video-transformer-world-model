"""Exact finite tie-aware Best-of-N law for score/utility pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TieGroup:
    """One score tie group in ascending score order."""

    score: float
    start: int
    stop: int
    r_min: int
    r_max: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def _as_1d_float(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_n_values(n_values: Iterable[int]) -> list[int]:
    out = [int(n) for n in n_values]
    if not out:
        raise ValueError("n_values must be non-empty")
    if any(n < 1 for n in out):
        raise ValueError("all N values must be >= 1")
    return out


def sorted_tie_groups(scores: Iterable[float]) -> tuple[np.ndarray, list[TieGroup]]:
    scores_arr = _as_1d_float(scores, "scores")
    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    groups: list[TieGroup] = []
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        groups.append(TieGroup(float(sorted_scores[i]), i, j, i + 1, j))
        i = j
    return order, groups


def utility_best_of_n_finite(
    scores: Iterable[float],
    utilities: Iterable[float],
    n_values: Iterable[int],
) -> dict[int, float]:
    """Expected utility of top-score selection from a finite pool.

    Sampling is with replacement. When several sampled items share the maximum
    score, selection is uniform among the tied sampled items. By exchangeability,
    a tie group's contribution is its empirical mean utility multiplied by the
    probability that this group is the maximum-score group in the N draws.
    """

    scores_arr = _as_1d_float(scores, "scores")
    utilities_arr = _as_1d_float(utilities, "utilities")
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same length")
    ns = _as_n_values(n_values)
    m = len(scores_arr)
    order, groups = sorted_tie_groups(scores_arr)
    sorted_utilities = utilities_arr[order]
    out: dict[int, float] = {}
    for N in ns:
        expected = 0.0
        for group in groups:
            mass = (group.r_max / m) ** N - ((group.r_min - 1) / m) ** N
            expected += float(np.mean(sorted_utilities[group.start : group.stop])) * mass
        out[N] = float(expected)
    return out


def binary_best_of_n_finite(
    scores: Iterable[float],
    success: Iterable[bool | int | float],
    n_values: Iterable[int],
) -> dict[int, float]:
    success_arr = _as_1d_float(success, "success")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("success must be binary")
    return utility_best_of_n_finite(scores, success_arr, n_values)


def auc_kappa(scores: Iterable[float], success: Iterable[bool | int | float]) -> float:
    scores_arr = _as_1d_float(scores, "scores")
    success_arr = _as_1d_float(success, "success")
    if scores_arr.shape != success_arr.shape:
        raise ValueError("scores and success must have the same length")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("success must be binary")
    pos = scores_arr[success_arr == 1.0]
    neg = scores_arr[success_arr == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for score in pos:
        wins += float(np.sum(score > neg)) + 0.5 * float(np.sum(score == neg))
    return float(wins / (len(pos) * len(neg)))


def n2_auc_identity(p: float, kappa: float) -> float:
    p = float(p)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite for 0 < p < 1")
    return float(p * p + 2.0 * p * (1.0 - p) * float(kappa))


def tie_rate(scores: Iterable[float]) -> float:
    scores_arr = _as_1d_float(scores, "scores")
    n = len(scores_arr)
    if n < 2:
        return 0.0
    _, counts = np.unique(scores_arr, return_counts=True)
    tied_pairs = sum(int(count) * (int(count) - 1) / 2 for count in counts)
    return float(tied_pairs / (n * (n - 1) / 2))


def simulate_best_of_n(
    scores: Iterable[float],
    utilities: Iterable[float],
    N: int,
    n_trials: int = 10_000,
    seed: int | None = None,
) -> float:
    scores_arr = _as_1d_float(scores, "scores")
    utilities_arr = _as_1d_float(utilities, "utilities")
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same length")
    if int(N) < 1:
        raise ValueError("N must be >= 1")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scores_arr), size=(int(n_trials), int(N)))
    drawn_scores = scores_arr[idx]
    max_scores = np.max(drawn_scores, axis=1)
    selected = np.empty(int(n_trials), dtype=int)
    for row in range(int(n_trials)):
        tied_positions = np.flatnonzero(drawn_scores[row] == max_scores[row])
        selected[row] = idx[row, int(rng.choice(tied_positions))]
    return float(np.mean(utilities_arr[selected]))
