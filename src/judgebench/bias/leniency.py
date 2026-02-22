"""Leniency bias detector.

Measures whether the judge is systematically more lenient (higher TPR)
or stricter (higher TNR) compared to human labels.
"""

from judgebench.models import JudgeVerdict, LabeledPair


def detect(pairs: list[LabeledPair], verdicts: list[JudgeVerdict]) -> float:
    """Compute leniency score as TPR / (TPR + FNR) asymmetry.

    We treat human_label="a" as the positive class. Then:
    - TP = judge picks "a" when human says "a"
    - FN = judge picks "b" when human says "a"
    - TN = judge picks "b" when human says "b"
    - FP = judge picks "a" when human says "b"

    Leniency = TPR - TNR, where:
    - TPR = TP / (TP + FN)
    - TNR = TN / (TN + FP)

    A positive value means the judge is more lenient (agrees more when
    human says "a"). A negative value means the judge is stricter.

    Args:
        pairs: List of labeled pairs with human labels
        verdicts: List of judge verdicts

    Returns:
        Leniency score between -1.0 and 1.0.
        Returns 0.0 if insufficient data.
    """
    pair_map = {p.id: p for p in pairs}

    tp = 0
    fn = 0
    tn = 0
    fp = 0

    for v in verdicts:
        pair = pair_map.get(v.pair_id)
        if pair is None:
            continue

        human = pair.human_label
        judge = v.forward_choice

        # Skip ties in human labels
        if human == "tie":
            continue

        if human == "a" and judge == "a":
            tp += 1
        elif human == "a" and judge == "b":
            fn += 1
        elif human == "b" and judge == "b":
            tn += 1
        elif human == "b" and judge == "a":
            fp += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return tpr - tnr
