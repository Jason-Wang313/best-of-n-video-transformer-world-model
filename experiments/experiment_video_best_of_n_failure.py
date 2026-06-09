"""Raw Best-of-N failure in the controlled video world."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import FIGURES, RESULTS, TABLES, summary_rows, write_csv, write_json
from video_transformer_best_of_n.candidates import candidate_validity_overlay, sample_candidate_pool
from video_transformer_best_of_n.config import N_VALUES
from video_transformer_best_of_n.envs import GridVideoWorld
from video_transformer_best_of_n.evaluation import evaluate_best_of_n, high_minus_low
from video_transformer_best_of_n.scorers import select_best


def make_counterfactual_lineup(seed: int) -> None:
    world = GridVideoWorld()
    rng = np.random.default_rng(seed)
    chosen = []
    for N in N_VALUES:
        pool = sample_candidate_pool(N, seed=int(rng.integers(0, 2**31 - 1)), world=world)
        selected = select_best(pool)
        chosen.append((N, selected))

    fig, axes = plt.subplots(2, len(chosen), figsize=(14.0, 4.2), dpi=150)
    for col, (N, candidate) in enumerate(chosen):
        pred = candidate.predicted_frames[-1]
        true = candidate.true_frames[-1]
        axes[0, col].imshow(pred)
        axes[1, col].imshow(true)
        overlay = candidate_validity_overlay(candidate, world)
        invalid = overlay.count("x") + overlay.count("!")
        axes[0, col].set_title(f"N={N}\nscore {candidate.scores['combined_internal_score']:.2f}")
        axes[1, col].set_title(f"exec {candidate.real_utility:.2f}, bad {invalid}")
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#c43b3b" if invalid else "#2f8f46")
                spine.set_linewidth(2.0)
    axes[0, 0].set_ylabel("selected video")
    axes[1, 0].set_ylabel("executed video")
    fig.suptitle("Counterfactual video lineup: visual tails versus action execution")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_counterfactual_video_lineup.png")
    plt.close(fig)


def run(*, smoke: bool = False, seed: int = 0) -> dict[str, object]:
    trials = 8 if smoke else 45
    rows, summary = evaluate_best_of_n(trials=trials, seed=seed, world=GridVideoWorld())
    x = list(N_VALUES)
    fig, ax1 = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    ax1.plot(x, [summary[str(N)]["visual_plausibility"] for N in x], marker="o", label="visual plausibility")
    ax1.plot(x, [summary[str(N)]["action_consistency_violation_rate"] for N in x], marker="o", label="action violation")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(N) for N in x])
    ax1.set_xlabel("N")
    ax1.set_ylabel("score or rate")
    ax2 = ax1.twinx()
    ax2.plot(x, [summary[str(N)]["real_utility"] for N in x], marker="s", color="#7a3db8", label="executed utility")
    ax2.set_ylabel("executed utility")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    ax1.grid(True, alpha=0.25)
    fig.suptitle("Raw visual Best-of-N selects action-invalid tails")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_failure_metrics.png")
    plt.close(fig)
    make_counterfactual_lineup(seed + 1000)

    table_rows = summary_rows(summary)
    write_csv(TABLES / "experiment_video_best_of_n_failure.csv", table_rows)
    write_csv(TABLES / "experiment_video_best_of_n_failure_trials.csv", rows)
    key_result = {
        "raw_plausibility_delta_high_n": high_minus_low(summary, "visual_plausibility"),
        "raw_action_violation_delta_high_n": high_minus_low(summary, "action_consistency_violation_rate"),
        "raw_real_utility_delta_high_n": high_minus_low(summary, "real_utility"),
        "n64_trap_rate_ghost_wall": summary["64"].get("trap_rate_ghost_wall", 0.0),
        "n64_trap_rate_occlusion_skip": summary["64"].get("trap_rate_occlusion_skip", 0.0),
    }
    payload = {
        "schema_version": 1,
        "smoke": bool(smoke),
        "seed": int(seed),
        "trials_per_n": trials,
        "summary_by_n": summary,
        "key_result": key_result,
    }
    write_json(RESULTS / "experiment_video_best_of_n_failure.json", payload)
    return payload


if __name__ == "__main__":
    run()
