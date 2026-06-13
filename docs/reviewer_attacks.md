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
