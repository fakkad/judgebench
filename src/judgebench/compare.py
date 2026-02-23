"""Compare two judge benchmark results."""

from __future__ import annotations

from judgebench.models import BenchResult


def compare_results(result_a: BenchResult, result_b: BenchResult) -> dict:
    """Compare two BenchResult objects and return a comparison summary.

    Args:
        result_a: First judge's results.
        result_b: Second judge's results.

    Returns:
        dict with comparison data for each metric.
    """
    comparison = {
        "judge_a": {
            "provider": result_a.judge_config.provider,
            "model": result_a.judge_config.model,
        },
        "judge_b": {
            "provider": result_b.judge_config.provider,
            "model": result_b.judge_config.model,
        },
        "reliability": {
            "judge_a": result_a.overall_reliability,
            "judge_b": result_b.overall_reliability,
            "delta": result_a.overall_reliability - result_b.overall_reliability,
            "better": "A" if result_a.overall_reliability >= result_b.overall_reliability else "B",
        },
        "agreement_metrics": {},
        "bias_comparison": {},
    }

    # Compare agreement metrics
    for key in ["raw_agreement", "cohens_kappa", "krippendorff_alpha_nominal"]:
        val_a = result_a.agreement_metrics.get(key)
        val_b = result_b.agreement_metrics.get(key)
        if val_a is not None and val_b is not None:
            comparison["agreement_metrics"][key] = {
                "judge_a": val_a,
                "judge_b": val_b,
                "delta": val_a - val_b,
                "better": "A" if val_a >= val_b else "B",
            }

    # Compare bias scores
    biases_a = {b.bias_type: b for b in result_a.bias_reports}
    biases_b = {b.bias_type: b for b in result_b.bias_reports}

    all_types = set(biases_a.keys()) | set(biases_b.keys())
    for bt in sorted(all_types):
        score_a = biases_a[bt].score if bt in biases_a else None
        score_b = biases_b[bt].score if bt in biases_b else None
        if score_a is not None and score_b is not None:
            comparison["bias_comparison"][bt] = {
                "judge_a": score_a,
                "judge_b": score_b,
                "delta": score_a - score_b,
                "better": "A" if score_a <= score_b else "B",  # Lower bias = better
            }

    return comparison
