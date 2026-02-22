"""Statistical agreement metrics between judge and human labels."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import krippendorff

from judgebench.models import AgreementMetrics, JudgeVerdict, LabeledPair


def _mcnemars_test(
    human_labels: list[str], judge_labels: list[str]
) -> tuple[float, float]:
    """Compute McNemar's test statistic and p-value.

    Compares paired correctness: cases where one rater is "correct"
    (agrees with some reference) and the other is not.

    For pairwise comparison, we build a 2x2 contingency table of
    agreement/disagreement patterns.

    Args:
        human_labels: List of human labels
        judge_labels: List of judge labels

    Returns:
        Tuple of (chi2, p_value)
    """
    from scipy.stats import chi2

    # Contingency: both agree vs disagree patterns
    # b = human says "a" but judge says "b"
    # c = human says "b" but judge says "a"
    b = 0  # human=a, judge=b
    c = 0  # human=b, judge=a

    for h, j in zip(human_labels, judge_labels):
        if h == "a" and j == "b":
            b += 1
        elif h == "b" and j == "a":
            c += 1

    # McNemar's chi-squared with continuity correction
    if b + c == 0:
        return 0.0, 1.0

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return float(chi2_stat), float(p_value)


def compute_agreement(
    pairs: list[LabeledPair],
    verdicts: list[JudgeVerdict],
) -> AgreementMetrics:
    """Compute all agreement metrics between judge and human labels.

    Pairs with human_label="tie" are excluded.

    Args:
        pairs: List of labeled pairs with human labels
        verdicts: List of judge verdicts

    Returns:
        AgreementMetrics with all computed statistics
    """
    pair_map = {p.id: p for p in pairs}

    human_labels: list[str] = []
    judge_labels: list[str] = []

    for v in verdicts:
        pair = pair_map.get(v.pair_id)
        if pair is None:
            continue
        if pair.human_label == "tie":
            continue

        human_labels.append(pair.human_label)
        judge_labels.append(v.forward_choice)

    if len(human_labels) < 2:
        return AgreementMetrics(
            cohens_kappa=0.0,
            krippendorffs_alpha=0.0,
            spearman_rho=0.0,
            spearman_p=1.0,
            mcnemars_chi2=0.0,
            mcnemars_p=1.0,
        )

    # Encode labels as numeric for correlation
    label_to_num = {"a": 1, "b": 0}
    human_numeric = [label_to_num.get(l, 0) for l in human_labels]
    judge_numeric = [label_to_num.get(l, 0) for l in judge_labels]

    # Cohen's kappa
    kappa = cohen_kappa_score(human_labels, judge_labels)

    # Krippendorff's alpha
    # Requires at least 2 distinct values in the combined domain
    all_values = set(human_numeric) | set(judge_numeric)
    if len(all_values) > 1:
        reliability_data = np.array([human_numeric, judge_numeric])
        alpha = krippendorff.alpha(
            reliability_data=reliability_data, level_of_measurement="nominal"
        )
    else:
        alpha = 1.0 if human_numeric == judge_numeric else 0.0

    # Spearman correlation
    if len(set(human_numeric)) > 1 and len(set(judge_numeric)) > 1:
        rho, p_value = spearmanr(human_numeric, judge_numeric)
    else:
        rho, p_value = 0.0, 1.0

    # McNemar's test
    chi2_stat, mcnemar_p = _mcnemars_test(human_labels, judge_labels)

    return AgreementMetrics(
        cohens_kappa=float(kappa),
        krippendorffs_alpha=float(alpha),
        spearman_rho=float(rho) if rho == rho else 0.0,
        spearman_p=float(p_value) if p_value == p_value else 1.0,
        mcnemars_chi2=float(chi2_stat),
        mcnemars_p=float(mcnemar_p),
    )
