# Difference From Prior Projects

The repo is distinct from tokenized world-model aliasing work because the central error is not a codebook collision. The generated futures are rendered RGB frame sequences, and the failure is visible in the counterfactual video lineup.

It is also distinct from latent-dynamics mismatch work because the selected object is not an internal rollout value alone. The selected item includes frames, action labels, predicted states, and a real execution trace for the same actions.

The point is to make the tail selector inspectable. As `N` grows, the highest-scoring selected video can become the most attractive visual story while also becoming less faithful to action-conditioned dynamics.

The v4 benchmark tier also checks a standard Moving-MNIST-style future-prediction setting, where the failure is not wall crossing but visually smooth motion collapse under a raw video score. This makes the paper less like a renamed copy of a scalar selected-tail template.
