"""Verbosity bias detector.

Checks whether a judge systematically favors longer responses regardless of
quality, by computing the correlation between response length difference and
verdict direction.
"""

from __future__ import annotations

import numpy as np

from judgebench.models import BiasReport, JudgeVerdict, LabeledPair


def detect_verbosity_bias(
    verdicts: list[JudgeVerdict],
    pairs: list[LabeledPair],
) -> BiasReport:
    """Detect verbosity bias.

    Computes Pearson correlation between (len_a - len_b) and verdict direction.
    Verdict direction: +1 if judge picks A, -1 if picks B, 0 if tie.

    A positive correlation means the judge favors whichever response is longer.

    Score = |correlation| normalized to [0, 1].
    """
    pair_map = {p.id: p for p in pairs}

    length_diffs: list[float] = []
    verdict_directions: list[float] = []

    for v in verdicts:
        if v.pair_id not in pair_map:
            continue
        pair = pair_map[v.pair_id]

        if v.position == "original":
            len_a = len(pair.response_a)
            len_b = len(pair.response_b)
        else:
            # Swapped: A shown as B and vice versa
            len_a = len(pair.response_b)
            len_b = len(pair.response_a)

        length_diffs.append(float(len_a - len_b))

        if v.judge_label == "A":
            verdict_directions.append(1.0)
        elif v.judge_label == "B":
            verdict_directions.append(-1.0)
        else:
            verdict_directions.append(0.0)

    if len(length_diffs) < 3:
        return BiasReport(
            bias_type="verbosity",
            score=0.0,
            details={"note": "insufficient data for correlation"},
            flagged=False,
        )

    x = np.array(length_diffs)
    y = np.array(verdict_directions)

    # Pearson correlation
    if np.std(x) == 0 or np.std(y) == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(x, y)[0, 1])

    score = min(abs(r), 1.0)

    return BiasReport(
        bias_type="verbosity",
        score=score,
        details={
            "pearson_r": r,
            "n_verdicts": len(length_diffs),
            "favors_longer": r > 0,
        },
        flagged=score > 0.3,
    )
