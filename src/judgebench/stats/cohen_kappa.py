"""Cohen's kappa for inter-rater agreement."""

from __future__ import annotations

import numpy as np


def cohens_kappa(labels_a: list[str], labels_b: list[str], categories: list[str] | None = None) -> float:
    """Compute Cohen's kappa between two sets of labels.

    Args:
        labels_a: First rater's labels.
        labels_b: Second rater's labels.
        categories: All possible label values. Inferred from data if None.

    Returns:
        Kappa coefficient in [-1, 1]. 1 = perfect agreement, 0 = chance agreement.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have equal length")
    if len(labels_a) == 0:
        raise ValueError("Label lists must not be empty")

    if categories is None:
        categories = sorted(set(labels_a) | set(labels_b))

    n = len(labels_a)
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Build confusion matrix
    matrix = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(labels_a, labels_b):
        matrix[cat_idx[a], cat_idx[b]] += 1

    # Observed agreement
    p_o = np.trace(matrix) / n

    # Expected agreement by chance
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    p_e = float(np.sum(row_sums * col_sums))

    if p_e == 1.0:
        return 1.0  # Both raters use only one category

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa)
