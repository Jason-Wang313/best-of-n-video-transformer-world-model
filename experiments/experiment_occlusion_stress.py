"""Stress the video selection failure under wider occlusion."""

from __future__ import annotations

import matplotlib.pyplot as plt

from experiments.common import FIGURES, RESULTS, TABLES, write_csv, write_json
from video_transformer_best_of_n.envs import GridVideoWorld
from video_transformer_best_of_n.evaluation import evaluate_best_of_n


def run(*, smoke: bool = False, seed: int = 0) -> dict[str, object]:
    trials = 6 if smoke else 30
    regimes = {
        "thin": (3, 3),
        "standard": (3, 4),
        "wide": (2, 5),
    }
    rows = []
    for idx, (name, cols) in enumerate(regimes.items()):
        world = GridVideoWorld(occlusion_cols=cols)
        _, summary = evaluate_best_of_n(trials=trials, seed=seed + 100 * idx, world=world)
        rows.append(
            {
                "regime": name,
                "occlusion_cols": f"{cols[0]}-{cols[1]}",
                "n64_real_utility": summary["64"]["real_utility"],
                "n64_visual_plausibility": summary["64"]["visual_plausibility"],
                "n64_occlusion_uncertainty": summary["64"]["occlusion_uncertainty"],
                "n64_action_violation": summary["64"]["action_consistency_violation_rate"],
            }
        )

    fig, ax1 = plt.subplots(figsize=(6.6, 4.0), dpi=150)
    names = [row["regime"] for row in rows]
    x = range(len(names))
    ax1.bar([v - 0.18 for v in x], [row["n64_occlusion_uncertainty"] for row in rows], width=0.36, label="occlusion uncertainty")
    ax1.bar([v + 0.18 for v in x], [row["n64_action_violation"] for row in rows], width=0.36, label="action violation")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names)
    ax1.set_ylabel("rate")
    ax2 = ax1.twinx()
    ax2.plot(list(x), [row["n64_real_utility"] for row in rows], marker="o", color="#7a3db8", label="executed utility")
    ax2.set_ylabel("executed utility")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    ax1.set_title("Occlusion stress at N=64")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_occlusion_stress.png")
    plt.close(fig)

    write_csv(TABLES / "experiment_occlusion_stress.csv", rows)
    slope = float(rows[-1]["n64_occlusion_uncertainty"] - rows[0]["n64_occlusion_uncertainty"])
    payload = {
        "schema_version": 1,
        "smoke": bool(smoke),
        "seed": int(seed),
        "rows": rows,
        "key_result": {
            "occlusion_uncertainty_slope": slope,
            "wide_minus_thin_real_utility": float(rows[-1]["n64_real_utility"] - rows[0]["n64_real_utility"]),
        },
    }
    write_json(RESULTS / "experiment_occlusion_stress.json", payload)
    return payload


if __name__ == "__main__":
    run()
