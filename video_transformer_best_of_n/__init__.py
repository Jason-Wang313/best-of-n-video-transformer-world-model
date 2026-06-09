"""Source-tree import bridge for shells that drop PYTHONPATH."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "video_transformer_best_of_n"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))  # type: ignore[name-defined]

from video_transformer_best_of_n.config import GATE_LABELS, N_VALUES

__all__ = ["GATE_LABELS", "N_VALUES"]
