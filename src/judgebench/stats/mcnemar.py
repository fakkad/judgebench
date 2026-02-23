"""McNemar's test for paired nominal data."""

from __future__ import annotations

import math


def mcnemar_test(labels_a: list[str], labels_b: list[str], reference: list[str]) -> dict:
    """McNemar's test comparing two raters against a reference.

    Checks whether rater A and rater B make errors asymmetrically
    relative to the reference labels.

    When used for judge evaluation, labels_a = judge labels,
    labels_b = human labels, and reference = human labels.
    We test whether the judge's errors are symmetric (equally likely
    to err in each direction).

    For the simpler single-judge case: compare judge_correct vs judge_incorrect
    on matched pairs.

    Args:
        labels_a: First set of labels (e.g., judge).
        labels_b: Second set of labels (e.g., another judge or human).
        reference: Ground truth labels.

    Returns:
        dict with keys:
            b: count where A correct, B incorrect
            c: count where A incorrect, B correct
            chi_squared: McNemar chi-squared statistic
            p_value: p-value (chi-squared distribution, df=1)
    """
    if not (len(labels_a) == len(labels_b) == len(reference)):
        raise ValueError("All label lists must have equal length")

    n = len(labels_a)
    if n == 0:
        raise ValueError("Label lists must not be empty")

    # b = A correct, B incorrect
    # c = A incorrect, B correct
    b = 0
    c = 0
    for a, bv, ref in zip(labels_a, labels_b, reference):
        a_correct = a == ref
        b_correct = bv == ref
        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    # McNemar's test statistic (with continuity correction)
    if b + c == 0:
        return {"b": b, "c": c, "chi_squared": 0.0, "p_value": 1.0}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared distribution with 1 df
    # Using survival function approximation
    p_value = _chi2_sf(chi2, df=1)

    return {"b": b, "c": c, "chi_squared": float(chi2), "p_value": float(p_value)}


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function (1 - CDF) for chi-squared distribution.

    Uses the regularized incomplete gamma function for df=1.
    """
    if x <= 0:
        return 1.0
    if df != 1:
        raise NotImplementedError("Only df=1 supported")

    # For df=1, chi2 SF = 2 * (1 - Phi(sqrt(x)))
    # where Phi is the standard normal CDF
    z = math.sqrt(x)
    return 2.0 * (1.0 - _normal_cdf(z))


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
