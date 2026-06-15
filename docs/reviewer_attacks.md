# Reviewer Attacks

## Attack: The environment is too simple.

Agreed. The paper should present this as a controlled counterexample and diagnostic testbed, not a broad benchmark. The value is that every frame, state, action, and selected-tail score is auditable.

## Attack: The learned model is tiny.

Also agreed. The transformer is intentionally CPU-feasible and smoke-scale. The claim is that the repo includes a learned video component and artifact discipline, not that scale has been resolved.

## Attack: The repair ladder may overfit the simulator.

The docs should say exactly that. The repair gates are evidence for this controlled setting. External use would require new calibration and held-out execution labels.

## Attack: Raw large-candidate-count can help in other settings.

Yes. The claim is existence and measurement of a video-tail failure under controlled conditions, not a universal monotonic law.

## Attack: The scorer is hand-built.

The scorer is intentionally transparent so the failure can be audited. The tiny transformer score is included as a learned component, while the visual tail mechanism remains interpretable.

## Attack: The paper looks like a reused top-score template.

The manuscript should be judged on video-specific evidence rather than generic top-score language: rendered counterfactual futures, action-consistency violations, occlusion uncertainty, frame-state checks, visual score-key ablations, frame-level repair ladders, and a Moving-MNIST future-prediction stress tier. The finite-pool law is only bookkeeping; the scientific object is video fidelity under the selected support assumptions.

## Attack: The expanded experiments still do not prove broad deployment risk.

Correct. The expanded `N=256`, horizon, occlusion, score-key, repair, and Moving-MNIST tests strengthen internal validity and attack obvious cherry-picking concerns, but they do not turn this into a hardware, leaderboard, or foundation-video-model benchmark.
