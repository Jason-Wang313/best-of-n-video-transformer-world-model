# Tie-Aware Finite Best-of-N Law

Consider a finite candidate pool with scores `S_i` and measured utilities `U_i`. A Best-of-N selector samples `N` candidates with replacement, takes the highest observed score, and breaks score ties uniformly among the tied sampled candidates.

Sort the pool by increasing score. Let a score tie group `g` occupy one-indexed ranks `r_min(g)` through `r_max(g)` in the sorted pool, and let `mean_U(g)` be the average utility inside that tie group. With pool size `M`, the probability that group `g` is the highest score group observed in `N` draws is

```text
(r_max(g) / M)^N - ((r_min(g) - 1) / M)^N.
```

Therefore the exact expected selected utility is

```text
E[U_selected] =
  sum_g mean_U(g) * [(r_max(g) / M)^N - ((r_min(g) - 1) / M)^N].
```

The implementation in `src/video_transformer_best_of_n/theory.py` validates this formula against brute-force enumeration for small tied pools and Monte Carlo sampling for larger checks. The same law covers oracle-aligned scores, anti-aligned scores, constant utilities, and binary success utilities.

In this repo the law is not the scientific novelty by itself. It is the audit layer that turns a selected-tail video pool into an exact prediction of selected executed utility, given the empirical score and utility pairs.
