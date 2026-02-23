"""Raw agreement metrics and confusion matrix."""

from __future__ import annotations

from collections import Counter

import numpy as np


def raw_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    """Fraction of labels that match exactly.

    Returns:
        Agreement rate in [0.0, 1.0].
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have equal length")
    if len(labels_a) == 0:
        return 0.0

    matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return matches / len(labels_a)


def confusion_matrix(
    labels_a: list[str],
    labels_b: list[str],
    categories: list[str] | None = None,
) -> dict:
    """Build a confusion matrix between two sets of labels.

    Args:
        labels_a: Row labels (e.g., human).
        labels_b: Column labels (e.g., judge).
        categories: Label categories. Inferred if None.

    Returns:
        dict with keys:
            matrix: 2D list (row=labels_a, col=labels_b)
            categories: list of category names
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have equal length")

    if categories is None:
        categories = sorted(set(labels_a) | set(labels_b))

    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    mat = np.zeros((k, k), dtype=int)

    for a, b in zip(labels_a, labels_b):
        mat[cat_idx[a], cat_idx[b]] += 1

    return {
        "matrix": mat.tolist(),
        "categories": categories,
    }


def per_category_agreement(
    labels_a: list[str],
    labels_b: list[str],
    categories: list[str] | None = None,
) -> dict[str, float]:
    """Agreement rate broken down by category of labels_a.

    Returns:
        dict mapping category -> agreement rate for items in that category.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have equal length")

    if categories is None:
        categories = sorted(set(labels_a) | set(labels_b))

    counts: dict[str, int] = Counter()
    matches: dict[str, int] = Counter()

    for a, b in zip(labels_a, labels_b):
        counts[a] += 1
        if a == b:
            matches[a] += 1

    result = {}
    for cat in categories:
        if counts[cat] > 0:
            result[cat] = matches[cat] / counts[cat]
        else:
            result[cat] = 0.0
    return result
