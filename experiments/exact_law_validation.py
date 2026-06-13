"""Validate the finite tie-aware top-score selection law."""

from __future__ import annotations

import itertools

import numpy as np

from experiments.common import FIGURES, RESULTS, TABLES, save_line_plot, write_csv, write_json
from counterfactual_video_audit.config import N_VALUES
from counterfactual_video_audit.theory import (
    auc_kappa,
    binary_score_selected_finite,
    n2_auc_identity,
    simulate_score_selected,
    tie_rate,
    utility_score_selected_finite,
)


def brute_force(scores: np.ndarray, utilities: np.ndarray, N: int) -> float:
    total = 0.0
    count = 0
    for tup in itertools.product(range(len(scores)), repeat=N):
        drawn_scores = scores[list(tup)]
        max_score = float(drawn_scores.max())
        tied = [tup[i] for i, score in enumerate(drawn_scores) if score == max_score]
        total += float(np.mean(utilities[tied]))
        count += 1
    return float(total / count)


def run(*, smoke: bool = False, seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    scores = np.asarray([-1.0, -0.2, -0.2, 0.5, 0.5, 0.5, 1.0], dtype=float)
    utilities = np.asarray([-0.5, 0.2, 0.6, 1.1, -0.8, 0.4, -1.2], dtype=float)
    n_values = N_VALUES
    exact = utility_score_selected_finite(scores, utilities, n_values)
    mc_trials = 5_000 if smoke else 25_000
    rows = []
    max_abs_error = 0.0
    for N in n_values:
        mc = simulate_score_selected(scores, utilities, N, n_trials=mc_trials, seed=seed + N)
        error = abs(exact[N] - mc)
        max_abs_error = max(max_abs_error, error)
        rows.append({"N": N, "exact": exact[N], "monte_carlo": mc, "abs_error": error})

    brute = {N: brute_force(scores, utilities, N) for N in (1, 2, 3)}
    oracle = utility_score_selected_finite(utilities, utilities, n_values)
    anti = utility_score_selected_finite(-utilities, utilities, n_values)
    success = np.asarray([0, 1, 1, 1, 0, 1, 0], dtype=float)
    binary = binary_score_selected_finite(scores, success, [1, 2])
    identity = n2_auc_identity(float(success.mean()), auc_kappa(scores, success))

    x = list(n_values)
    save_line_plot(
        FIGURES / "figure5_exact_law_validation.png",
        x,
        {
            "exact": [exact[N] for N in x],
            "Monte Carlo": [row["monte_carlo"] for row in rows],
            "oracle score": [oracle[N] for N in x],
            "anti-aligned score": [anti[N] for N in x],
        },
        title="Tie-aware finite-pool selection law",
        ylabel="Expected selected utility",
    )
    write_csv(TABLES / "exact_law_validation.csv", rows)
    payload = {
        "schema_version": 1,
        "smoke": bool(smoke),
        "seed": int(seed),
        "pool_size": int(len(scores)),
        "tie_rate": tie_rate(scores),
        "max_abs_error": float(max_abs_error),
        "rows": rows,
        "brute_force_n1_n3": {str(k): float(v) for k, v in brute.items()},
        "binary_n2_identity": {"finite_law": float(binary[2]), "auc_identity": float(identity)},
        "oracle_high_minus_low": float(oracle[64] - oracle[1]),
        "anti_aligned_high_minus_low": float(anti[64] - anti[1]),
    }
    write_json(RESULTS / "exact_law_validation.json", payload)
    return payload


if __name__ == "__main__":
    run()
