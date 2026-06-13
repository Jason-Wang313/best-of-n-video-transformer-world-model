"""Controlled counterfactual video audit experiments.

The package is intentionally small and deterministic. It supports a low-res
video world where visually plausible generated futures can be action-invalid,
plus diagnostics, repair gates, and an exact finite-pool selection law.
"""

from counterfactual_video_audit.config import GATE_LABELS, N_VALUES

__all__ = ["GATE_LABELS", "N_VALUES"]
