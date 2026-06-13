# When Plausible Videos Lie

This repository is a controlled research scaffold for a video-specific visual-tail failure mode. In a low-resolution action-conditioned world, generated futures can become smoother, more goal-like, and more visually plausible as the sample budget grows, while the selected action sequence can execute poorly in the ground-truth dynamics.

The central artifact is a counterfactual video lineup: for each `N`, the repo renders the selected generated future, the future produced by executing the same actions in the real simulator, and compact diagnostics for action validity, temporal causality, occlusion uncertainty, and frame-to-state consistency.

## Scope

The claims are deliberately narrow:

- Controlled grid-video dynamics with rendered RGB frames.
- Finite-pool top-score video selection over `N = {1,2,4,8,16,32,64}`.
- Video-specific diagnostics and repair gates, not a universal safety recipe.
- A tiny CPU-trained transformer over frame patches, used as a smoke-scale learned video component.
- No hardware experiments, no broad robotics claim, and no blanket statement about all video predictors.

## Run

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

Key artifacts are written under `results/`, `results/tables/`, and `figures/`. The learned model checkpoint is `results/learned_tiny_video_transformer.pt`.

## Paper

The anonymous ICLR-style submission source is in `paper/main.tex` and uses the official ICLR 2026 style files from the ICLR Master-Template.

Build the PDF with:

```powershell
powershell -ExecutionPolicy Bypass -File paper/build_submission.ps1
```

or:

```bash
bash paper/build_submission.sh
```

The PowerShell build writes `paper/when_plausible_videos_lie_iclr_submission.pdf` and copies the same PDF to `C:\Users\wangz\OneDrive\Desktop\best of n video transformer world model-v2.pdf`.

## Main Files

- `src/counterfactual_video_audit/envs.py`: action-conditioned video world.
- `src/counterfactual_video_audit/candidates.py`: valid and action-inconsistent generated futures.
- `src/counterfactual_video_audit/scorers.py`: visual plausibility, smoothness, goal-frame similarity, and learned-video proxy scores.
- `src/counterfactual_video_audit/diagnostics.py`: video-specific failure measurements.
- `src/counterfactual_video_audit/gate.py`: deployment gate with exactly one allowed label.
- `src/counterfactual_video_audit/theory.py`: tie-aware finite-pool top-score law.
- `experiments/`: reproducible scripts for figures, tables, and claim status.

## Expected Figures

- `figures/figure1_counterfactual_video_lineup.png`
- `figures/figure2_repair_ladder.png`
- `figures/figure3_video_diagnostics.png`
- `figures/figure4_occlusion_stress.png`
- `figures/figure5_exact_law_validation.png`

## Final Audit

The intended readiness label for this scaffold is `submission-ready v2`: it is a scoped controlled-counterexample submission with explicit limits on external validity and benchmark breadth.
