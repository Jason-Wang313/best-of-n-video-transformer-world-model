# Difference From World-Action Model Framing

This scaffold is not about imagined-versus-real latent dynamics in a world-action model setting. The candidate object here is a rendered video future conditioned on an action sequence. The failure appears because visually plausible videos can cross occluded or blocked regions in ways that the action-conditioned transition model would not permit.

The measured quantities are video-native: temporal smoothness, goal-frame similarity, occlusion uncertainty, frame-to-state consistency, and action-consistency violation rate. The executed utility is measured by replaying the selected action sequence in the ground-truth simulator.

The finite-pool selection law is shared mathematical bookkeeping, but the scientific object is different: video futures and their causal relation to actions.

The Moving-MNIST tier further separates this repo from world-action-model framing: it tests future-frame prediction under standard bouncing-digit structure, using baselines and stress gates rather than a latent action-value claim.
