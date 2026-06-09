"""Controlled video Best-of-N experiments.

The package is intentionally small and deterministic. It supports a low-res
video world where visually plausible generated futures can be action-invalid,
plus diagnostics, repair gates, and an exact finite Best-of-N law.
"""

from video_transformer_best_of_n.config import GATE_LABELS, N_VALUES

__all__ = ["GATE_LABELS", "N_VALUES"]
