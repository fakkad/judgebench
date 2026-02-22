"""Position bias detector.

Measures the fraction of pairs where the judge's verdict changes
when the order of responses is swapped. Ties are excluded.
"""

from judgebench.models import JudgeVerdict, LabeledPair


def detect(
    verdicts: list[JudgeVerdict],
    pairs: list[LabeledPair] | None = None,
) -> float:
    """Compute position bias rate.

    Position bias = fraction of non-tie pairs where forward_choice != reversed_choice
    (i.e., swapping positions changed the verdict).

    Args:
        verdicts: List of judge verdicts with forward and reversed choices
        pairs: Optional list of pairs (used to exclude ties by human_label)

    Returns:
        Float between 0.0 and 1.0. Higher = more position bias.
        Returns 0.0 if no eligible pairs.
    """
    # Build lookup for human labels to exclude ties
    tie_ids: set[str] = set()
    if pairs is not None:
        tie_ids = {p.id for p in pairs if p.human_label == "tie"}

    eligible = [v for v in verdicts if v.pair_id not in tie_ids]

    if not eligible:
        return 0.0

    inconsistent = sum(1 for v in eligible if not v.consistent)
    return inconsistent / len(eligible)
