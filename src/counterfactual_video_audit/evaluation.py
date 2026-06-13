"""Reusable top-score video evaluation helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np

from counterfactual_video_audit.candidates import sample_candidate_pool
from counterfactual_video_audit.config import N_VALUES
from counterfactual_video_audit.envs import GridVideoWorld
from counterfactual_video_audit.scorers import select_best


Selector = Callable[[list[Any], np.random.Generator], Any]


def raw_selector(pool: list[Any], rng: np.random.Generator) -> Any:
    return select_best(pool)


def evaluate_score_selected(
    selector: Selector = raw_selector,
    *,
    n_values: tuple[int, ...] = N_VALUES,
    trials: int = 40,
    seed: int = 0,
    world: GridVideoWorld | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    world = world or GridVideoWorld()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for N in n_values:
        for trial in range(int(trials)):
            pool_seed = int(rng.integers(0, 2**31 - 1))
            pool = sample_candidate_pool(N, seed=pool_seed, world=world)
            selected = selector(pool, rng)
            row = {
                "N": int(N),
                "trial": int(trial),
                "trap_type": selected.trap_type,
                "real_utility": float(selected.real_utility),
                "predicted_utility": float(selected.predicted_utility),
                "combined_internal_score": float(selected.scores["combined_internal_score"]),
                "visual_plausibility": float(selected.scores["visual_plausibility"]),
                "temporal_smoothness": float(selected.scores["temporal_smoothness"]),
                "goal_frame_similarity": float(selected.scores["goal_frame_similarity"]),
                "learned_video_score": float(selected.scores["learned_video_score"]),
                "action_consistency_violation_rate": float(
                    selected.diagnostics["action_consistency_violation_rate"]
                ),
                "temporal_causal_violation_rate": float(selected.diagnostics["temporal_causal_violation_rate"]),
                "occlusion_uncertainty": float(selected.diagnostics["occlusion_uncertainty"]),
                "frame_to_state_consistency_error": float(
                    selected.diagnostics["frame_to_state_consistency_error"]
                ),
            }
            rows.append(row)
    return rows, summarize_rows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_n: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_n.setdefault(int(row["N"]), []).append(row)
    summary: dict[str, dict[str, float]] = {}
    numeric_keys = [
        "real_utility",
        "predicted_utility",
        "combined_internal_score",
        "visual_plausibility",
        "temporal_smoothness",
        "goal_frame_similarity",
        "learned_video_score",
        "action_consistency_violation_rate",
        "temporal_causal_violation_rate",
        "occlusion_uncertainty",
        "frame_to_state_consistency_error",
    ]
    for N, group in sorted(by_n.items()):
        stats = {key: float(np.mean([row[key] for row in group])) for key in numeric_keys}
        counts = Counter(str(row["trap_type"]) for row in group)
        total = max(1, len(group))
        for trap, count in counts.items():
            stats[f"trap_rate_{trap}"] = float(count / total)
        summary[str(N)] = stats
    return summary


def high_minus_low(summary: dict[str, dict[str, float]], metric: str, low: int = 1, high: int = 64) -> float:
    return float(summary[str(high)][metric] - summary[str(low)][metric])


def mean_ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "lo": mean, "hi": mean}
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return {"mean": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se}
