"""Source-tree import bridge for shells that drop PYTHONPATH."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "counterfactual_video_audit"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))  # type: ignore[name-defined]

from counterfactual_video_audit.config import GATE_LABELS, N_VALUES

__all__ = ["GATE_LABELS", "N_VALUES"]
