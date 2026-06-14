# Claims

## Supported After `scripts/run_all.sh`

- Large-candidate-count raw visual selection can increase selected visual plausibility while action-conditioned executed utility stalls or drops in this controlled world.
- The selected-tail failure is associated with action-consistency violations, temporal-causality issues, occlusion uncertainty, and frame-to-state inconsistencies.
- A repair ladder that filters action-invalid futures, screens temporal anomalies, penalizes ambiguous occlusion, checks frame-state agreement, and uses a small held-out calibration set improves selected executed utility in this scaffold.
- The finite tie-aware law exactly predicts empirical selected utility for any finite score and utility pool.

## Supported After `experiments.experiment_expansion_suite`

- The expanded candidate-count suite extends the raw-tail stress test to `N=256`, where visual plausibility keeps rising while executed utility drops in this controlled world.
- The `N=256` repair ladder recovers high-candidate-count utility by selecting action-consistent video futures rather than the visually most flattering future.
- Horizon length, occlusion width, and score-key choices materially change the measured failure mode, which prevents the paper from relying on one cherry-picked rendering condition.

## Boundary

The repo studies one controlled video environment and smoke-scale learned video prediction. It does not claim broad external validity, hardware deployment evidence, or a guarantee that one diagnostic removes all risk.
