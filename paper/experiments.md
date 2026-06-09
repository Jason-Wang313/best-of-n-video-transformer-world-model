# Experiments

The primary experiment sweeps `N = {1,2,4,8,16,32,64}`. For each `N`, the selected generated video, executed rollout, real utility, and diagnostics are recorded. The failure figure shows the visual future above the real execution for the same actions.

The repair experiment compares raw visual selection against random selection, oracle selection, and the diagnostic ladder. The diagnostics experiment records action violation, temporal-causal violation, occlusion uncertainty, and frame-state error as functions of `N`.

The occlusion stress experiment changes the hidden region width and measures how uncertainty and selected utility shift. The theory experiment validates the finite law against enumeration and Monte Carlo sampling.
