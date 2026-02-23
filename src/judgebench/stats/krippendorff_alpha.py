"""Krippendorff's alpha for inter-rater reliability.

Implemented from scratch using numpy only, following Krippendorff (2011).
Supports nominal and ordinal distance functions.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def _nominal_metric(v1: int, v2: int) -> float:
    """Nominal distance: 0 if same, 1 if different."""
    return 0.0 if v1 == v2 else 1.0


def _ordinal_metric(v1: int, v2: int, n_values: np.ndarray) -> float:
    """Ordinal distance function per Krippendorff.

    n_values: array of value counts across all codings.
    """
    lo, hi = min(v1, v2), max(v1, v2)
    # Sum of counts from lo to hi, minus half the endpoints
    s = float(np.sum(n_values[lo : hi + 1]))
    s = s - (n_values[lo] + n_values[hi]) / 2.0
    return s * s


def krippendorff_alpha(
    reliability_data: list[list[int | None]],
    level: Literal["nominal", "ordinal"] = "nominal",
) -> float:
    """Compute Krippendorff's alpha.

    Args:
        reliability_data: Matrix of shape (observers, units).
            Each cell is a coded value (int) or None for missing.
            Multiple observers can rate overlapping units.
        level: "nominal" or "ordinal" distance metric.

    Returns:
        Alpha coefficient. 1 = perfect, 0 = chance, <0 = worse than chance.
    """
    n_observers = len(reliability_data)
    if n_observers < 2:
        raise ValueError("Need at least 2 observers")

    n_units = len(reliability_data[0])
    for row in reliability_data:
        if len(row) != n_units:
            raise ValueError("All observer rows must have equal length")

    # Collect all non-None values to determine categories
    all_values = set()
    for row in reliability_data:
        for v in row:
            if v is not None:
                all_values.add(v)

    if len(all_values) < 2:
        # If only one category observed, alpha is undefined; return 1.0 by convention
        return 1.0

    values = sorted(all_values)
    n_values_count = max(values) + 1

    # Build coincidence matrix
    coincidence = np.zeros((n_values_count, n_values_count), dtype=np.float64)

    # For each unit, count how many observers assigned each value
    for u in range(n_units):
        codings = [reliability_data[o][u] for o in range(n_observers) if reliability_data[o][u] is not None]
        m_u = len(codings)
        if m_u < 2:
            continue
        for i, c in enumerate(codings):
            for j, k in enumerate(codings):
                if i != j:
                    coincidence[c, k] += 1.0 / (m_u - 1)

    # Marginal counts
    n_c = np.sum(coincidence, axis=1)  # marginals per value
    n_total = np.sum(n_c)

    if n_total == 0:
        return 0.0

    # Observed disagreement
    d_o = 0.0
    # Expected disagreement
    d_e = 0.0

    if level == "nominal":
        for c in values:
            for k in values:
                if c == k:
                    continue
                d_o += coincidence[c, k] * _nominal_metric(c, k)
                d_e += n_c[c] * n_c[k] * _nominal_metric(c, k)
    elif level == "ordinal":
        for c in values:
            for k in values:
                if c == k:
                    continue
                dist = _ordinal_metric(c, k, n_c)
                d_o += coincidence[c, k] * dist
                d_e += n_c[c] * n_c[k] * dist
    else:
        raise ValueError(f"Unknown level: {level}")

    if d_e == 0:
        return 1.0

    d_e_norm = d_e / (n_total - 1)

    if d_e_norm == 0:
        return 1.0

    alpha = 1.0 - (d_o / d_e_norm)
    return float(alpha)
