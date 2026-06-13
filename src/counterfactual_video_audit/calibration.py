"""Small held-out video-real calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VideoRealCalibrator:
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def predict(self, candidate: Any) -> float:
        x = candidate_features(candidate)
        z = (x - self.feature_mean) / self.feature_scale
        return float(np.r_[1.0, z] @ self.weights)


def candidate_features(candidate: Any) -> np.ndarray:
    return np.asarray(
        [
            candidate.scores["combined_internal_score"],
            candidate.scores["goal_frame_similarity"],
            candidate.diagnostics["action_consistency_violation_rate"],
            candidate.diagnostics["temporal_causal_violation_rate"],
            candidate.diagnostics["occlusion_uncertainty"],
            candidate.diagnostics["frame_to_state_consistency_error"],
        ],
        dtype=float,
    )


def fit_video_real_calibrator(candidates: list[Any], ridge: float = 1e-3) -> VideoRealCalibrator:
    if len(candidates) < 2:
        raise ValueError("at least two pilot candidates are required")
    x = np.stack([candidate_features(candidate) for candidate in candidates], axis=0)
    y = np.asarray([candidate.real_utility for candidate in candidates], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6
    z = (x - mean) / scale
    design = np.c_[np.ones(len(z)), z]
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return VideoRealCalibrator(weights=weights, feature_mean=mean, feature_scale=scale)


def calibration_mae(calibrator: VideoRealCalibrator, candidates: list[Any]) -> float:
    if not candidates:
        return 0.0
    errors = [abs(calibrator.predict(candidate) - candidate.real_utility) for candidate in candidates]
    return float(np.mean(errors))
