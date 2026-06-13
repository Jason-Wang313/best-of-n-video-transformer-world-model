"""Diagnostics and deployment-gate behavior across N."""

from __future__ import annotations

import matplotlib.pyplot as plt

from experiments.common import FIGURES, RESULTS, TABLES, summary_rows, write_csv, write_json
from counterfactual_video_audit.config import N_VALUES
from counterfactual_video_audit.envs import GridVideoWorld
from counterfactual_video_audit.evaluation import evaluate_score_selected
from counterfactual_video_audit.gate import deployment_gate


def run(*, smoke: bool = False, seed: int = 0) -> dict[str, object]:
    trials = 8 if smoke else 40
    rows, summary = evaluate_score_selected(trials=trials, seed=seed, world=GridVideoWorld())
    labels = []
    for N in N_VALUES:
        metrics = summary[str(N)]
        labels.append(
            deployment_gate(
                n=N,
                action_violation_rate=metrics["action_consistency_violation_rate"],
                temporal_causal_violation_rate=metrics["temporal_causal_violation_rate"],
                occlusion_uncertainty=metrics["occlusion_uncertainty"],
                frame_to_state_consistency_error=metrics["frame_to_state_consistency_error"],
                pilot_label_count=0 if N >= 16 else 16,
                calibration_mae=None,
            )
        )

    x = list(N_VALUES)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    for metric in [
        "action_consistency_violation_rate",
        "temporal_causal_violation_rate",
        "occlusion_uncertainty",
        "frame_to_state_consistency_error",
    ]:
        ax.plot(x, [summary[str(N)][metric] for N in x], marker="o", label=metric.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in x])
    ax.set_xlabel("N")
    ax.set_ylabel("rate or normalized error")
    ax.set_title("Video-specific diagnostics under raw selection")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_video_diagnostics.png")
    plt.close(fig)

    table_rows = summary_rows(summary)
    for row, label in zip(table_rows, labels):
        row["gate_label"] = label
    write_csv(TABLES / "experiment_video_diagnostics.csv", table_rows)
    payload = {
        "schema_version": 1,
        "smoke": bool(smoke),
        "seed": int(seed),
        "summary_by_n": summary,
        "gate_labels": labels,
    }
    write_json(RESULTS / "experiment_video_diagnostics.json", payload)
    return payload


if __name__ == "__main__":
    run()
