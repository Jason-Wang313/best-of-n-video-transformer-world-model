# Theory

The exact finite law treats a candidate pool as paired scores and utilities. For each score tie group, the selected utility contribution is the group's mean utility times the probability that the group is the highest observed score among `N` samples.

This tie-aware formula is useful because it separates the selection mechanism from the source of the candidates. Once a finite video pool has internal scores and executed utilities, the law predicts the expected selected utility of top-score video selection.

The law also makes anti-alignment visible. If visual scores rank action-invalid videos above action-valid videos, increasing `N` can move selected utility downward even as selected score rises.
