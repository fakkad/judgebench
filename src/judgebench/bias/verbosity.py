"""Verbosity bias detector.

Measures whether the judge systematically prefers longer responses
by computing Spearman correlation between chosen response length and a
binary score (1 = chosen, 0 = not chosen).
"""

from scipy.stats import spearmanr

from judgebench.models import JudgeVerdict, LabeledPair


def detect(pairs: list[LabeledPair], verdicts: list[JudgeVerdict]) -> float:
    """Compute verbosity bias as Spearman correlation.

    For each pair, we record the length of both responses and whether each
    was chosen (1) or not (0). Then compute Spearman rho between length
    and chosen status across all responses.

    Args:
        pairs: List of labeled pairs with response text
        verdicts: List of judge verdicts

    Returns:
        Spearman rho. Positive = prefers longer responses. Returns 0.0 if
        insufficient data or constant values.
    """
    pair_map = {p.id: p for p in pairs}

    lengths: list[int] = []
    chosen: list[int] = []

    for v in verdicts:
        pair = pair_map.get(v.pair_id)
        if pair is None:
            continue

        # Use forward_choice as the primary verdict
        choice = v.forward_choice

        len_a = len(pair.response_a)
        len_b = len(pair.response_b)

        # Response A
        lengths.append(len_a)
        chosen.append(1 if choice == "a" else 0)

        # Response B
        lengths.append(len_b)
        chosen.append(1 if choice == "b" else 0)

    if len(lengths) < 3:
        return 0.0

    # Check for constant arrays (Spearman undefined)
    if len(set(lengths)) <= 1 or len(set(chosen)) <= 1:
        return 0.0

    rho, _ = spearmanr(lengths, chosen)
    return float(rho) if rho == rho else 0.0  # NaN check
