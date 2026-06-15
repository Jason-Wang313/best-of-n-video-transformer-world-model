# Claim Status

- Project: When Plausible Videos Lie
- Unsupported count: 0
- Weak strong-evidence checks: 0
- Full multi-seed evidence: True

| Category | Status | Strength | Claim | Evidence |
| --- | --- | --- | --- | --- |
| finite law | SUPPORTED | STRONG | The tie-aware finite-pool top-score expectation matches brute-force enumeration and Monte Carlo validation within sampling error. | results/exact_law_validation.json; figures/figure5_exact_law_validation.png |
| video tail failure | SUPPORTED | STRONG | In the controlled video world, visually top-ranked futures become more plausible as N grows while executed action-conditioned utility stagnates or worsens. | results/experiment_video_tail_failure.json; figures/figure1_counterfactual_video_lineup.png |
| learned video scorer | SUPPORTED | STRONG | A compact CPU-trained transformer over video patches is trained and saved as a tiny learned scorer artifact for smoke-scale checks. | results/experiment_learned_video_transformer.json; results/learned_tiny_video_transformer.pt |
| repair ladder | SUPPORTED | STRONG | Video-specific checks and a small eval-disjoint pilot calibration recover utility over raw visual selection in this controlled setting. | results/experiment_video_repairs.json; figures/figure2_repair_ladder.png |
| gate contract | SUPPORTED | STRONG | The deployment gate emits exactly one allowed label for each evaluated N. | results/experiment_video_diagnostics.json; figures/figure3_video_diagnostics.png |
| occlusion stress | SUPPORTED | STRONG | Occlusion stress increases uncertainty and exposes when visual selection is most vulnerable to hidden crossing artifacts. | results/experiment_occlusion_stress.json; figures/figure4_occlusion_stress.png |
| final audit | SUPPORTED | STRONG | The repository final audit chooses one allowed readiness status. | {"allowed": ["submission-ready v4", "needs stronger learned model", "needs benchmark validation", "redesign required"], "status": "submission-ready v4"} |
| expanded stress suite | SUPPORTED | STRONG | The expansion suite passes its generated candidate-count, horizon, occlusion, score-key, and repair claims. | results/expansion/claims.json |
| moving mnist benchmark | SUPPORTED | STRONG | The Moving-MNIST benchmark tier passes raw-selection failure, motion-gate repair, baseline, and oracle-gap claim gates. | results/moving_mnist_benchmark/claims.json |
| final pdf | SUPPORTED | STRONG | The repository final PDF exists and is at least 25 pages. | {"pages": 28, "path": "paper/final/best of n video transformer world model-v4.pdf"} |
| forbidden overclaims | SUPPORTED | STRONG | Universal video, robotics, and blanket top-score video overclaims are absent from README, docs, and paper text. | {"blocked_phrases": ["solves video world modeling", "solves robot planning", "top-score selection always hurts", "video transformers are bad", "action consistency always fixes it", "real-robot validation", "submission-ready v2", "submission-ready v3", "v2 copy", "v2.pdf", "v3.pdf", "best of n video transformer world model-v3.pdf", "iclr_submission"], "hits": []} |

## Audit Block List

- `solves video world modeling`
- `solves robot planning`
- `top-score selection always hurts`
- `video transformers are bad`
- `action consistency always fixes it`
- `real-robot validation`
- `submission-ready v2`
- `submission-ready v3`
- `v2 copy`
- `v2.pdf`
- `v3.pdf`
- `best of n video transformer world model-v3.pdf`
- `iclr_submission`
