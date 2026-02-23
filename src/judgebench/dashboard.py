"""Dashboard generator — renders HTML calibration report."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from judgebench.models import BenchResult, Dataset, LabeledPair


TEMPLATE_DIR = Path(__file__).parent / "templates"


def _reliability_color(score: float) -> str:
    """Color coding: green (reliable), yellow (marginal), red (unreliable)."""
    if score >= 0.67:
        return "green"
    elif score >= 0.4:
        return "yellow"
    return "red"


def _bias_color(score: float) -> str:
    if score <= 0.15:
        return "green"
    elif score <= 0.3:
        return "yellow"
    return "red"


def generate_dashboard(
    result: BenchResult,
    dataset: Dataset | None = None,
    output_path: str = "dashboard.html",
) -> str:
    """Generate an HTML dashboard from benchmark results.

    Args:
        result: The benchmark result to render.
        dataset: Optional dataset for per-pair details.
        output_path: Where to write the HTML file.

    Returns:
        Path to the generated HTML file.
    """
    pair_map = {p.id: p for p in dataset.pairs} if dataset else {}

    # Build per-pair detail rows
    original_verdicts = [v for v in result.verdicts if v.position == "original"]
    pair_details = []
    for v in original_verdicts:
        pair = pair_map.get(v.pair_id)
        match = v.judge_label == pair.human_label if pair else None
        pair_details.append({
            "id": v.pair_id,
            "human_label": pair.human_label if pair else "?",
            "judge_label": v.judge_label,
            "match": match,
            "confidence": f"{v.confidence:.2f}",
            "reasoning": v.reasoning[:200],
        })

    # Bias data for radar chart
    bias_data = {b.bias_type: b.score for b in result.bias_reports}

    # McNemar summary
    mcnemar = result.agreement_metrics.get("mcnemar", {})

    context = {
        "judge_model": result.judge_config.model,
        "judge_provider": result.judge_config.provider,
        "overall_reliability": result.overall_reliability,
        "reliability_color": _reliability_color(result.overall_reliability),
        "raw_agreement": result.agreement_metrics.get("raw_agreement", 0),
        "cohens_kappa": result.agreement_metrics.get("cohens_kappa", 0),
        "krippendorff_alpha": result.agreement_metrics.get("krippendorff_alpha_nominal", 0),
        "mcnemar_chi2": mcnemar.get("chi_squared", 0),
        "mcnemar_p": mcnemar.get("p_value", 1),
        "confusion": result.agreement_metrics.get("confusion_matrix", {}),
        "per_category": result.agreement_metrics.get("per_category_agreement", {}),
        "bias_data": bias_data,
        "bias_reports": result.bias_reports,
        "bias_color": _bias_color,
        "pair_details": pair_details,
        "total_pairs": len(pair_details),
        "total_verdicts": len(result.verdicts),
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("dashboard.html")
    html = template.render(**context)

    Path(output_path).write_text(html)
    return output_path
