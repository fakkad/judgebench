"""Position bias detector.

Checks whether a judge systematically favors whichever response appears first
(or second) regardless of content. Uses the position-swap pairs to measure
consistency.
"""

from __future__ import annotations

from judgebench.models import BiasReport, JudgeVerdict


def detect_position_bias(verdicts: list[JudgeVerdict]) -> BiasReport:
    """Detect position bias from paired original/swapped verdicts.

    For each pair_id, there should be one 'original' and one 'swapped' verdict.
    Position bias score = 1 - consistency rate.
    Consistency means the judge picks the same underlying response regardless of order.

    A score of 0.0 = no position bias (perfectly consistent).
    A score of 1.0 = maximum position bias (always follows position).
    """
    # Group verdicts by pair_id
    by_pair: dict[str, dict[str, JudgeVerdict]] = {}
    for v in verdicts:
        by_pair.setdefault(v.pair_id, {})[v.position] = v

    consistent = 0
    total = 0
    first_position_wins = 0
    total_verdicts = 0

    for pair_id, positions in by_pair.items():
        if "original" not in positions or "swapped" not in positions:
            continue

        orig = positions["original"]
        swap = positions["swapped"]
        total += 1

        # In original ordering: A is first, B is second
        # In swapped ordering: B is first, A is second
        # If judge picks "A" in original and "B" in swapped, it's following position (first)
        # If judge picks "A" in original and "A" in swapped, it's consistent (truly prefers A)

        # Check if the judge picked the same underlying response
        # In original: label refers to response A/B directly
        # In swapped: label refers to the swapped positions, so:
        #   "A" in swapped means the judge picked what was shown first (which is original B)
        #   "B" in swapped means the judge picked what was shown second (which is original A)

        # Map swapped verdict back to original response identity
        swap_mapped = _map_swapped_label(swap.judge_label)

        if orig.judge_label == swap_mapped:
            consistent += 1
        elif orig.judge_label == "tie" or swap_mapped == "tie":
            # Tie in one but not the other - partial inconsistency
            pass

        # Track first-position preference
        if orig.judge_label == "A":  # picked first in original
            first_position_wins += 1
        if swap.judge_label == "A":  # picked first in swapped
            first_position_wins += 1
        total_verdicts += 2

    if total == 0:
        return BiasReport(
            bias_type="position",
            score=0.0,
            details={"note": "no paired verdicts found"},
            flagged=False,
        )

    consistency_rate = consistent / total
    score = 1.0 - consistency_rate

    first_pref_rate = first_position_wins / total_verdicts if total_verdicts > 0 else 0.5

    return BiasReport(
        bias_type="position",
        score=score,
        details={
            "consistency_rate": consistency_rate,
            "total_pairs": total,
            "consistent_pairs": consistent,
            "first_position_preference_rate": first_pref_rate,
        },
        flagged=score > 0.3,
    )


def _map_swapped_label(label: str) -> str:
    """Map a label from swapped ordering back to original response identity."""
    if label == "A":
        return "B"
    elif label == "B":
        return "A"
    return "tie"
