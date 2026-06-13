# Introduction

Video futures are persuasive artifacts. They show an outcome, not merely a number. This makes top-score video selection attractive: sample several futures, score them, and pick the one that looks best. The risk is that the selected future may exploit a scorer's visual preferences while becoming less faithful to the actions that supposedly caused it.

We construct a controlled setting where the direct-looking path crosses a blocked region hidden by an occluder. Generated videos can show a smooth crossing and a goal-reaching final frame, yet executing the selected actions in the ground-truth dynamics leaves the agent stuck. Increasing `N` makes this selected tail more likely.

The contribution is an auditable scaffold: rendered counterfactual lineups, exact finite-pool selection accounting, diagnostics for video causality, and a repair ladder. The scope is intentionally narrow and controlled.
