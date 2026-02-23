"""Leniency bias detector.

Checks whether a judge gives more 'tie' verdicts than humans or systematically
avoids declaring a winner.
"""

from __future__ import annotations

from judgebench.models import BiasReport, JudgeVerdict, LabeledPair


def detect_leniency_bias(
    verdicts: list[JudgeVerdict],
    pairs: list[LabeledPair],
) -> BiasReport:
    """Detect leniency bias.

    Compares the judge's tie rate against the human tie rate.
    Also checks if the judge systematically avoids picking the losing response.

    Leniency index = |judge_tie_rate - human_tie_rate| + excess_tie_rate
    Normalized to [0, 1].
    """
    if not verdicts or not pairs:
        return BiasReport(
            bias_type="leniency",
            score=0.0,
            details={"note": "no data"},
            flagged=False,
        )

    # Human tie rate
    human_ties = sum(1 for p in pairs if p.human_label == "tie")
    human_tie_rate = human_ties / len(pairs) if pairs else 0.0

    # Judge tie rate (use only original position to avoid double-counting)
    original_verdicts = [v for v in verdicts if v.position == "original"]
    if not original_verdicts:
        original_verdicts = verdicts  # fallback if no position info

    judge_ties = sum(1 for v in original_verdicts if v.judge_label == "tie")
    judge_tie_rate = judge_ties / len(original_verdicts)

    # Tie rate difference
    tie_diff = judge_tie_rate - human_tie_rate

    # Label distribution divergence
    pair_map = {p.id: p for p in pairs}
    judge_a = sum(1 for v in original_verdicts if v.judge_label == "A")
    judge_b = sum(1 for v in original_verdicts if v.judge_label == "B")
    human_a = sum(1 for p in pairs if p.human_label == "A")
    human_b = sum(1 for p in pairs if p.human_label == "B")

    n_judge = len(original_verdicts)
    n_human = len(pairs)

    judge_dist = {
        "A": judge_a / n_judge if n_judge else 0,
        "B": judge_b / n_judge if n_judge else 0,
        "tie": judge_tie_rate,
    }
    human_dist = {
        "A": human_a / n_human if n_human else 0,
        "B": human_b / n_human if n_human else 0,
        "tie": human_tie_rate,
    }

    # Leniency score: primarily driven by excess ties
    # Also penalize general distribution divergence
    excess_ties = max(0.0, tie_diff)
    dist_divergence = sum(abs(judge_dist[k] - human_dist[k]) for k in ["A", "B", "tie"]) / 2.0

    score = min(1.0, excess_ties + dist_divergence * 0.5)

    return BiasReport(
        bias_type="leniency",
        score=score,
        details={
            "judge_tie_rate": judge_tie_rate,
            "human_tie_rate": human_tie_rate,
            "tie_rate_difference": tie_diff,
            "judge_distribution": judge_dist,
            "human_distribution": human_dist,
        },
        flagged=score > 0.3,
    )
