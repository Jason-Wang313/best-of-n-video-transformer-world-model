"""Shared experiment IO and plotting helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "figures"


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dirs()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dirs()
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_line_plot(
    path: Path,
    x: list[int],
    series: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    xlabel: str = "N",
) -> None:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for label, values in series.items():
        ax.plot(x, values, marker="o", linewidth=2.0, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary_rows(summary: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for key, metrics in sorted(summary.items(), key=lambda item: int(item[0])):
        row: dict[str, Any] = {"N": int(key)}
        row.update(metrics)
        rows.append(row)
    return rows


def ci95(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "lo": mean, "hi": mean}
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return {"mean": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se}
