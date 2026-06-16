"""Repair ladder for action-invalid video tails."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import FIGURES, RESULTS, TABLES, summary_rows, write_csv, write_json
from counterfactual_video_audit.calibration import fit_video_real_calibrator
from counterfactual_video_audit.candidates import sample_candidate_pool
from counterfactual_video_audit.config import N_VALUES
from counterfactual_video_audit.envs import GridVideoWorld
from counterfactual_video_audit.evaluation import evaluate_score_selected
from counterfactual_video_audit.scorers import select_best, select_oracle, select_random, select_with_filter


def action_check_selector(pool: list[Any], rng: np.random.Generator) -> Any:
    return select_with_filter(pool, lambda c: c.diagnostics["action_consistency_violation_rate"] <= 0.0)


def temporal_selector(pool: list[Any], rng: np.random.Generator) -> Any:
    return select_with_filter(
        pool,
        lambda c: c.diagnostics["action_consistency_violation_rate"] <= 0.0
        and c.diagnostics["temporal_causal_violation_rate"] <= 0.0,
    )


def occlusion_selector(pool: list[Any], rng: np.random.Generator) -> Any:
    return select_with_filter(
        pool,
        lambda c: c.diagnostics["action_consistency_violation_rate"] <= 0.0
        and c.diagnostics["temporal_causal_violation_rate"] <= 0.0
        and c.diagnostics["occlusion_uncertainty"] <= 0.45,
    )


def frame_state_selector(pool: list[Any], rng: np.random.Generator) -> Any:
    return select_with_filter(
        pool,
        lambda c: c.diagnostics["action_consistency_violation_rate"] <= 0.0
        and c.diagnostics["temporal_causal_violation_rate"] <= 0.0
        and c.diagnostics["frame_to_state_consistency_error"] <= 0.15,
    )


def make_calibrated_selector(seed: int, pilot_count: int) -> Any:
    world = GridVideoWorld()
    pilot = sample_candidate_pool(pilot_count, seed=seed + 300, world=world)
    calibrator = fit_video_real_calibrator(pilot)

    def selector(pool: list[Any], rng: np.random.Generator) -> Any:
        safe_pool = [
            candidate
            for candidate in pool
            if candidate.diagnostics["action_consistency_violation_rate"] <= 0.0
            and candidate.diagnostics["temporal_causal_violation_rate"] <= 0.05
            and candidate.diagnostics["frame_to_state_consistency_error"] <= 0.25
        ]
        candidates = safe_pool if safe_pool else pool
        return max(candidates, key=lambda candidate: calibrator.predict(candidate))

    selector.calibrator = calibrator  # type: ignore[attr-defined]
    return selector


def run(*, smoke: bool = False, seed: int = 0, write_artifacts: bool = True) -> dict[str, object]:
    trials = 6 if smoke else 32
    pilot_count = 12 if smoke else 36
    strategies = {
        "raw_visual": lambda pool, rng: select_best(pool),
        "action_check": action_check_selector,
        "temporal_filter": temporal_selector,
        "occlusion_filter": occlusion_selector,
        "frame_state_filter": frame_state_selector,
        "calibrated_repair": make_calibrated_selector(seed, pilot_count),
        "oracle": lambda pool, rng: select_oracle(pool),
        "random": lambda pool, rng: select_random(pool, rng),
    }
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    all_rows: list[dict[str, object]] = []
    for name, selector in strategies.items():
        rows, summary = evaluate_score_selected(selector, trials=trials, seed=seed + 17 * (len(summaries) + 1), world=GridVideoWorld())
        summaries[name] = summary
        for row in rows:
            row = dict(row)
            row["strategy"] = name
            all_rows.append(row)

    if write_artifacts:
        x = list(N_VALUES)
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
        for name in ["raw_visual", "action_check", "frame_state_filter", "calibrated_repair", "oracle", "random"]:
            ax.plot(x, [summaries[name][str(N)]["real_utility"] for N in x], marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(N) for N in x])
        ax.set_xlabel("N")
        ax.set_ylabel("executed utility")
        ax.set_title("Repair ladder versus raw visual selection")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "figure2_repair_ladder.png")
        plt.close(fig)

    table = []
    for strategy, summary in summaries.items():
        for row in summary_rows(summary):
            row = dict(row)
            row["strategy"] = strategy
            table.append(row)
    if write_artifacts:
        write_csv(TABLES / "experiment_video_repairs.csv", table)
        write_csv(TABLES / "experiment_video_repairs_trials.csv", all_rows)
    raw_n64 = summaries["raw_visual"]["64"]["real_utility"]
    calibrated_n64 = summaries["calibrated_repair"]["64"]["real_utility"]
    oracle_n64 = summaries["oracle"]["64"]["real_utility"]
    raw_to_oracle = max(1e-9, oracle_n64 - raw_n64)
    key_result = {
        "calibrated_repair_n64_improvement_over_raw": float(calibrated_n64 - raw_n64),
        "calibrated_repair_fraction_of_oracle_gap_closed": float((calibrated_n64 - raw_n64) / raw_to_oracle),
        "action_check_n64_improvement_over_raw": float(summaries["action_check"]["64"]["real_utility"] - raw_n64),
    }
    payload = {
        "schema_version": 1,
        "smoke": bool(smoke),
        "seed": int(seed),
        "trials_per_n": trials,
        "pilot_count": pilot_count,
        "summaries": summaries,
        "key_result": key_result,
    }
    if write_artifacts:
        write_json(RESULTS / "experiment_video_repairs.json", payload)
    return payload


if __name__ == "__main__":
    run()
