"""Expanded video-tail stress suite for the final paper."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import FIGURES, RESULTS, TABLES, write_csv, write_json
from experiments.experiment_video_repairs import (
    action_check_selector,
    frame_state_selector,
    make_calibrated_selector,
    occlusion_selector,
    temporal_selector,
)
from counterfactual_video_audit.envs import GridVideoWorld
from counterfactual_video_audit.evaluation import Selector, evaluate_score_selected
from counterfactual_video_audit.scorers import select_best, select_oracle, select_random


EXPANSION_DIR = RESULTS / "expansion"


def _tag_rows(
    rows: list[dict[str, Any]],
    *,
    family: str,
    setting: str,
    value: str | int | float,
    selector: str,
) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        item = dict(row)
        item["family"] = family
        item["setting"] = setting
        item["setting_value"] = value
        item["selector"] = selector
        tagged.append(item)
    return tagged


def _summary_rows(
    summary: dict[str, dict[str, float]],
    *,
    family: str,
    setting: str,
    value: str | int | float,
    selector: str,
) -> list[dict[str, Any]]:
    rows = []
    for n, metrics in sorted(summary.items(), key=lambda item: int(item[0])):
        row: dict[str, Any] = {
            "N": int(n),
            "family": family,
            "setting": setting,
            "setting_value": value,
            "selector": selector,
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _run_selector(
    *,
    family: str,
    setting: str,
    value: str | int | float,
    selector_name: str,
    selector: Selector,
    world: GridVideoWorld,
    n_values: tuple[int, ...],
    trials: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, summary = evaluate_score_selected(
        selector,
        n_values=n_values,
        trials=trials,
        seed=seed,
        world=world,
    )
    return (
        _summary_rows(summary, family=family, setting=setting, value=value, selector=selector_name),
        _tag_rows(rows, family=family, setting=setting, value=value, selector=selector_name),
    )


def _score_key_selector(key: str) -> Selector:
    def selector(pool: list[Any], rng: np.random.Generator) -> Any:
        return select_best(pool, key=key)

    return selector


def _claim(status: bool, value: float, threshold: float, description: str) -> dict[str, Any]:
    return {
        "status": "pass" if status else "fail",
        "value": float(value),
        "threshold": float(threshold),
        "description": description,
    }


def _row(
    rows: list[dict[str, Any]],
    *,
    family: str,
    selector: str,
    n: int,
    setting_value: str | int | float | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["family"] == family
        and row["selector"] == selector
        and int(row["N"]) == int(n)
        and (setting_value is None or str(row["setting_value"]) == str(setting_value))
    ]
    if not matches:
        raise ValueError(f"missing row for {family=} {selector=} {n=} {setting_value=}")
    return matches[0]


def _span(rows: list[dict[str, Any]], *, family: str, selector: str, n: int, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["family"] == family and row["selector"] == selector and int(row["N"]) == n]
    return float(max(values) - min(values)) if values else 0.0


def audit_expansion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_1 = _row(rows, family="candidate_count_tail", selector="raw_visual", n=1)
    raw_256 = _row(rows, family="candidate_count_tail", selector="raw_visual", n=256)
    repair_raw_256 = _row(rows, family="repair_ladder", selector="raw_visual", n=256)
    repair_cal_256 = _row(rows, family="repair_ladder", selector="calibrated_repair", n=256)
    repair_action_256 = _row(rows, family="repair_ladder", selector="action_check", n=256)

    visual_gain = float(raw_256["visual_plausibility"] - raw_1["visual_plausibility"])
    utility_change = float(raw_256["real_utility"] - raw_1["real_utility"])
    action_gain = float(raw_256["action_consistency_violation_rate"] - raw_1["action_consistency_violation_rate"])
    calibrated_repair = float(repair_cal_256["real_utility"] - repair_raw_256["real_utility"])
    action_repair = float(repair_action_256["real_utility"] - repair_raw_256["real_utility"])
    horizon_span = _span(rows, family="horizon_sweep", selector="raw_visual", n=256, metric="occlusion_uncertainty")
    occlusion_span = _span(rows, family="occlusion_sweep", selector="raw_visual", n=256, metric="occlusion_uncertainty")
    selector_span = _span(rows, family="score_key_ablation", selector="score_key", n=256, metric="real_utility")
    selector_action_span = _span(
        rows, family="score_key_ablation", selector="score_key", n=256, metric="action_consistency_violation_rate"
    )

    claims = {
        "candidate_tail_visual_extremizes_to_256": _claim(
            visual_gain > 0.05,
            visual_gain,
            0.05,
            "Increasing candidate count to 256 raises selected visual plausibility under raw video selection.",
        ),
        "candidate_tail_real_utility_drops_to_256": _claim(
            utility_change < -1.0,
            utility_change,
            -1.0,
            "The same raw high-candidate selection lowers real executed utility by more than one unit.",
        ),
        "candidate_tail_action_violation_rises": _claim(
            action_gain > 0.25,
            action_gain,
            0.25,
            "The high-candidate raw selection increases action-consistency violation rate.",
        ),
        "calibrated_repair_recovers_high_candidate_tail": _claim(
            calibrated_repair > 3.0,
            calibrated_repair,
            3.0,
            "The calibrated repair improves executed utility at N=256 relative to raw visual selection.",
        ),
        "action_check_recovers_high_candidate_tail": _claim(
            action_repair > 3.0,
            action_repair,
            3.0,
            "The action-consistency filter improves executed utility at N=256 relative to raw visual selection.",
        ),
        "horizon_changes_high_candidate_occlusion_exposure": _claim(
            horizon_span > 0.2,
            horizon_span,
            0.2,
            "Horizon changes high-candidate occlusion exposure under raw video selection.",
        ),
        "occlusion_width_changes_uncertainty": _claim(
            occlusion_span > 0.4,
            occlusion_span,
            0.4,
            "Occlusion width changes high-candidate occlusion uncertainty.",
        ),
        "score_key_changes_high_candidate_utility": _claim(
            selector_span > 1.0,
            selector_span,
            1.0,
            "The internal video score component used for selection changes high-candidate executed utility.",
        ),
        "score_key_changes_action_consistency": _claim(
            selector_action_span > 0.4,
            selector_action_span,
            0.4,
            "The internal score component used for selection changes action-consistency violation rate.",
        ),
    }
    return {
        "all_passed": all(item["status"] == "pass" for item in claims.values()),
        "claims": claims,
        "summary": (
            f"N=256 visual gain {visual_gain:.3f}, real utility change {utility_change:.3f}, "
            f"calibrated repair {calibrated_repair:.3f}, score-key utility span {selector_span:.3f}."
        ),
    }


def plot_expansion(rows: list[dict[str, Any]]) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    tail = sorted(
        [row for row in rows if row["family"] == "candidate_count_tail" and row["selector"] == "raw_visual"],
        key=lambda row: int(row["N"]),
    )
    fig, ax1 = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    x = [int(row["N"]) for row in tail]
    ax1.plot(x, [float(row["visual_plausibility"]) for row in tail], marker="o", label="visual plausibility")
    ax1.plot(x, [float(row["action_consistency_violation_rate"]) for row in tail], marker="o", label="action violation")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("N")
    ax1.set_ylabel("score or rate")
    ax2 = ax1.twinx()
    ax2.plot(x, [float(row["real_utility"]) for row in tail], marker="s", color="#7a3db8", label="real utility")
    ax2.set_ylabel("real utility")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8)
    ax1.grid(True, alpha=0.25)
    fig.suptitle("Candidate-count tail extended to 256")
    fig.tight_layout()
    path = FIGURES / "figure6_candidate_count_256.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=150)
    for family, ax, ylabel, title in [
        ("horizon_sweep", axes[0], "real utility", "horizon stress"),
        ("occlusion_sweep", axes[1], "occlusion uncertainty", "occlusion stress"),
    ]:
        sub = [row for row in rows if row["family"] == family and row["selector"] == "raw_visual" and int(row["N"]) == 256]
        sub = sorted(sub, key=lambda row: str(row["setting_value"]))
        ax.bar([str(row["setting_value"]) for row in sub], [float(row[ylabel.replace(" ", "_")]) for row in sub])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = FIGURES / "figure7_video_stress_sweeps.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    sub = [row for row in rows if row["family"] == "score_key_ablation" and int(row["N"]) == 256]
    sub = sorted(sub, key=lambda row: str(row["setting_value"]))
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    ax.bar([str(row["setting_value"]) for row in sub], [float(row["real_utility"]) for row in sub])
    ax.set_ylabel("real utility")
    ax.set_title("Score-key ablation at N=256")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = FIGURES / "figure8_score_key_ablation.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    sub = [row for row in rows if row["family"] == "repair_ladder" and int(row["N"]) == 256]
    order = ["random", "raw_visual", "action_check", "temporal_filter", "occlusion_filter", "frame_state_filter", "calibrated_repair", "oracle"]
    sub = sorted(sub, key=lambda row: order.index(str(row["selector"])) if str(row["selector"]) in order else 999)
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=150)
    ax.bar([str(row["selector"]) for row in sub], [float(row["real_utility"]) for row in sub])
    ax.set_ylabel("real utility")
    ax.set_title("Repair ladder at N=256")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = FIGURES / "figure9_repair_ladder_256.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))
    return paths


def run(*, quick: bool = False, output_dir: str | Path | None = None) -> dict[str, Any]:
    outdir = Path(output_dir) if output_dir is not None else EXPANSION_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    all_summary_rows: list[dict[str, Any]] = []
    all_trial_rows: list[dict[str, Any]] = []

    tail_trials = 10 if quick else 52
    sweep_trials = 6 if quick else 28
    repair_trials = 6 if quick else 30
    n_tail = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    n_stress = (1, 64, 256)

    summary, trials = _run_selector(
        family="candidate_count_tail",
        setting="default_world",
        value="Nmax256",
        selector_name="raw_visual",
        selector=lambda pool, rng: select_best(pool),
        world=GridVideoWorld(),
        n_values=n_tail,
        trials=tail_trials,
        seed=410,
    )
    all_summary_rows.extend(summary)
    all_trial_rows.extend(trials)

    for horizon in [7, 11, 15, 19]:
        summary, trials = _run_selector(
            family="horizon_sweep",
            setting="horizon",
            value=horizon,
            selector_name="raw_visual",
            selector=lambda pool, rng: select_best(pool),
            world=GridVideoWorld(horizon=horizon),
            n_values=n_stress,
            trials=sweep_trials,
            seed=500 + horizon,
        )
        all_summary_rows.extend(summary)
        all_trial_rows.extend(trials)

    for label, cols in [
        ("thin", (4, 4)),
        ("standard", (3, 4)),
        ("wide", (2, 5)),
        ("full", (2, 6)),
    ]:
        summary, trials = _run_selector(
            family="occlusion_sweep",
            setting="occlusion",
            value=label,
            selector_name="raw_visual",
            selector=lambda pool, rng: select_best(pool),
            world=GridVideoWorld(occlusion_cols=cols),
            n_values=n_stress,
            trials=sweep_trials,
            seed=610 + len(label),
        )
        all_summary_rows.extend(summary)
        all_trial_rows.extend(trials)

    for key in [
        "combined_internal_score",
        "visual_plausibility",
        "temporal_smoothness",
        "goal_frame_similarity",
        "learned_video_score",
    ]:
        summary, trials = _run_selector(
            family="score_key_ablation",
            setting="score_key",
            value=key,
            selector_name="score_key",
            selector=_score_key_selector(key),
            world=GridVideoWorld(),
            n_values=n_stress,
            trials=sweep_trials,
            seed=720 + len(key),
        )
        all_summary_rows.extend(summary)
        all_trial_rows.extend(trials)

    repair_selectors: dict[str, Selector] = {
        "raw_visual": lambda pool, rng: select_best(pool),
        "action_check": action_check_selector,
        "temporal_filter": temporal_selector,
        "occlusion_filter": occlusion_selector,
        "frame_state_filter": frame_state_selector,
        "calibrated_repair": make_calibrated_selector(800, 16 if quick else 48),
        "oracle": lambda pool, rng: select_oracle(pool),
        "random": lambda pool, rng: select_random(pool, rng),
    }
    for idx, (name, selector) in enumerate(repair_selectors.items()):
        summary, trials = _run_selector(
            family="repair_ladder",
            setting="repair",
            value="N256",
            selector_name=name,
            selector=selector,
            world=GridVideoWorld(),
            n_values=(64, 128, 256),
            trials=repair_trials,
            seed=830 + 17 * idx,
        )
        all_summary_rows.extend(summary)
        all_trial_rows.extend(trials)

    claims = audit_expansion(all_summary_rows)
    figures = [] if quick else plot_expansion(all_summary_rows)
    write_csv(outdir / "expanded_summary.csv", all_summary_rows)
    write_csv(outdir / "expanded_trials.csv", all_trial_rows)
    write_json(outdir / "claims.json", claims)
    manifest = {
        "summary": str(outdir / "expanded_summary.csv"),
        "trials": str(outdir / "expanded_trials.csv"),
        "claims": str(outdir / "claims.json"),
        "figures": figures,
        "claim_summary": claims["summary"],
    }
    write_json(outdir / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    manifest = run()
    print(manifest["claim_summary"])
    print(f"Manifest: {manifest['claims']}")
