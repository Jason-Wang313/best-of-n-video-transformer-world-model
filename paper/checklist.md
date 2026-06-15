# Checklist

- Controlled dynamics are documented.
- Candidate generation includes action-valid and action-inconsistent futures.
- Real utility is measured only by ground-truth action execution.
- Candidate-count sweeps include `N = {1,2,4,8,16,32,64}`.
- Diagnostics include action consistency, temporal causality, occlusion uncertainty, and frame-to-state consistency.
- Repair gates are evaluated against raw, oracle, and random baselines.
- The finite tie-aware law is validated by tests and an experiment.
- Moving-MNIST benchmark rows compare raw visual selection, motion support, persistence, constant-velocity bounce, and oracle baselines.
- Claims are audited for scope.
