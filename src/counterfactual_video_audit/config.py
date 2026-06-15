"""Shared constants for the counterfactual video audit."""

from __future__ import annotations

N_VALUES = (1, 2, 4, 8, 16, 32, 64)

GATE_LABELS = (
    "allow_high_n",
    "stop_early",
    "collect_pilot_labels",
    "require_action_consistency_check",
    "block_high_n",
)

FINAL_AUDIT_STATUSES = (
    "submission-ready v4",
    "needs stronger learned model",
    "needs benchmark validation",
    "redesign required",
)
