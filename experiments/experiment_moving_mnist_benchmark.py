"""Run a compact Moving-MNIST video-prediction benchmark stress tier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from experiments.common import FIGURES, RESULTS, write_csv, write_json
from counterfactual_video_audit.moving_mnist_benchmark import (
    benchmark_claims,
    run_moving_mnist_benchmark,
    summarize_benchmark_rows,
)


BENCHMARK_DIR = RESULTS / "moving_mnist_benchmark"


def run(
    *,
    quick: bool = False,
    output_dir: str | Path | None = None,
    allow_download: bool = True,
) -> dict[str, Any]:
    outdir = Path(output_dir) if output_dir is not None else BENCHMARK_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    rows, meta = run_moving_mnist_benchmark(
        cache_dir=Path(__file__).resolve().parents[1] / "data" / "mnist",
        sequence_count=12 if quick else 48,
        seed=2026,
        allow_download=allow_download,
    )
    summary = summarize_benchmark_rows(rows)
    claims = benchmark_claims(summary)
    write_csv(outdir / "metrics.csv", rows)
    write_csv(outdir / "aggregate_metrics.csv", summary)
    write_json(outdir / "claims.json", claims)
    write_json(outdir / "summary.json", {"meta": meta, "claim_summary": claims["summary"]})
    figures = [] if quick else [_plot(summary)]
    manifest = {
        "metrics": str(outdir / "metrics.csv"),
        "aggregate_metrics": str(outdir / "aggregate_metrics.csv"),
        "claims": str(outdir / "claims.json"),
        "summary": str(outdir / "summary.json"),
        "figures": figures,
        "claim_summary": claims["summary"],
        "digit_source": meta["digit_source"],
    }
    write_json(outdir / "manifest.json", manifest)
    return manifest


def _series(summary: list[dict[str, Any]], selector: str, key: str) -> tuple[list[int], list[float]]:
    rows = sorted([row for row in summary if str(row["selector"]) == selector], key=lambda row: int(row["N"]))
    return [int(row["N"]) for row in rows], [float(row[key]) for row in rows]


def _plot(summary: list[dict[str, Any]]) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    selectors = ["raw_visual", "motion_gate", "constant_velocity_bounce", "persistence", "oracle"]
    labels = {
        "raw_visual": "raw visual",
        "motion_gate": "motion gate",
        "constant_velocity_bounce": "bounce baseline",
        "persistence": "persistence",
        "oracle": "oracle",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), dpi=160)
    for selector in selectors:
        x, visual = _series(summary, selector, "visual_score")
        _, mse = _series(summary, selector, "future_mse")
        _, ade = _series(summary, selector, "centroid_ade")
        axes[0].plot(x, visual, marker="o", label=labels[selector])
        axes[1].plot(x, mse, marker="o", label=labels[selector])
        axes[2].plot(x, ade, marker="o", label=labels[selector])
    for ax, ylabel, title in [
        (axes[0], "internal visual score", "selected score"),
        (axes[1], "future MSE", "frame prediction error"),
        (axes[2], "centroid ADE", "motion error"),
    ]:
        ax.set_xscale("log", base=4)
        ax.set_xticks([1, 4, 16, 64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ax.set_xlabel("candidate count")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Moving-MNIST 10-to-10 benchmark stress")
    fig.tight_layout()
    path = FIGURES / "figure10_moving_mnist_benchmark.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    manifest = run(quick=args.quick, output_dir=args.output_dir, allow_download=not args.no_download)
    print(manifest["claim_summary"])
    print(f"Manifest: {manifest['claims']}")


if __name__ == "__main__":
    main()
