"""HTML bias dashboard reporter."""

from pathlib import Path

from jinja2 import Template

from judgebench.models import AgreementMetrics, BiasReport, JudgeVerdict, LabeledPair

DASHBOARD_TEMPLATE = Template(
    """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JudgeBench Report</title>
<style>
  :root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --card: #0f3460;
    --accent: #e94560;
    --text: #eaeaea;
    --text-dim: #a0a0b0;
    --good: #00d2ff;
    --warn: #ffc107;
    --bad: #e94560;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }
  h1 { color: var(--good); margin-bottom: 0.5rem; font-size: 2rem; }
  h2 { color: var(--text); margin: 2rem 0 1rem; font-size: 1.4rem; }
  h3 { color: var(--text-dim); margin-bottom: 0.5rem; }
  .subtitle { color: var(--text-dim); margin-bottom: 2rem; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }
  .card {
    background: var(--surface);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.05);
  }
  .gauge {
    text-align: center;
  }
  .gauge-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0.5rem 0;
  }
  .gauge-label {
    color: var(--text-dim);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .gauge-bar {
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    margin-top: 1rem;
    overflow: hidden;
  }
  .gauge-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
  }
  .good { color: var(--good); }
  .good .gauge-fill { background: var(--good); }
  .warn { color: var(--warn); }
  .warn .gauge-fill { background: var(--warn); }
  .bad { color: var(--bad); }
  .bad .gauge-fill { background: var(--bad); }
  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border-radius: 12px;
    overflow: hidden;
  }
  th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  th {
    background: var(--card);
    color: var(--good);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
  }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .tag {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .tag-agree { background: rgba(0,210,255,0.15); color: var(--good); }
  .tag-disagree { background: rgba(233,69,96,0.15); color: var(--bad); }
  .tag-category {
    background: rgba(255,255,255,0.08);
    color: var(--text-dim);
  }
  .footer {
    margin-top: 3rem;
    color: var(--text-dim);
    font-size: 0.85rem;
    text-align: center;
  }
</style>
</head>
<body>

<h1>JudgeBench Report</h1>
<p class="subtitle">LLM Judge Quality Evaluation Dashboard</p>

<h2>Bias Metrics</h2>
<div class="grid">
  <div class="card gauge {{ position_class }}">
    <div class="gauge-label">Position Bias</div>
    <div class="gauge-value">{{ "%.1f"|format(bias.position_bias_rate * 100) }}%</div>
    <div class="gauge-bar"><div class="gauge-fill" style="width: {{ (bias.position_bias_rate * 100)|int }}%"></div></div>
  </div>
  <div class="card gauge {{ verbosity_class }}">
    <div class="gauge-label">Verbosity Bias (rho)</div>
    <div class="gauge-value">{{ "%.3f"|format(bias.verbosity_bias_rho) }}</div>
    <div class="gauge-bar"><div class="gauge-fill" style="width: {{ (bias.verbosity_bias_rho|abs * 100)|int }}%"></div></div>
  </div>
  <div class="card gauge {{ self_enhance_class }}">
    <div class="gauge-label">Self-Enhancement</div>
    <div class="gauge-value">{{ "%.3f"|format(bias.self_enhance_delta) }}</div>
    <div class="gauge-bar"><div class="gauge-fill" style="width: {{ (bias.self_enhance_delta|abs * 100)|int }}%"></div></div>
  </div>
  <div class="card gauge {{ leniency_class }}">
    <div class="gauge-label">Leniency</div>
    <div class="gauge-value">{{ "%.3f"|format(bias.leniency_score) }}</div>
    <div class="gauge-bar"><div class="gauge-fill" style="width: {{ (bias.leniency_score|abs * 100)|int }}%"></div></div>
  </div>
</div>

<h2>Agreement Metrics</h2>
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Cohen's Kappa</td>
      <td>{{ "%.4f"|format(agreement.cohens_kappa) }}</td>
      <td>{{ kappa_interp }}</td>
    </tr>
    <tr>
      <td>Krippendorff's Alpha</td>
      <td>{{ "%.4f"|format(agreement.krippendorffs_alpha) }}</td>
      <td>{{ alpha_interp }}</td>
    </tr>
    <tr>
      <td>Spearman rho</td>
      <td>{{ "%.4f"|format(agreement.spearman_rho) }}</td>
      <td>p = {{ "%.4f"|format(agreement.spearman_p) }}</td>
    </tr>
    <tr>
      <td>McNemar's chi2</td>
      <td>{{ "%.4f"|format(agreement.mcnemars_chi2) }}</td>
      <td>p = {{ "%.4f"|format(agreement.mcnemars_p) }}</td>
    </tr>
  </tbody>
</table>

<h2>Per-Pair Disagreement Drill-Down</h2>
<table>
  <thead>
    <tr>
      <th>Pair ID</th>
      <th>Category</th>
      <th>Human</th>
      <th>Judge (fwd)</th>
      <th>Judge (rev)</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    {% for row in drilldown %}
    <tr>
      <td>{{ row.pair_id }}</td>
      <td><span class="tag tag-category">{{ row.category }}</span></td>
      <td>{{ row.human_label }}</td>
      <td>{{ row.forward_choice }}</td>
      <td>{{ row.reversed_choice }}</td>
      <td>
        {% if row.agrees %}
          <span class="tag tag-agree">AGREE</span>
        {% else %}
          <span class="tag tag-disagree">DISAGREE</span>
        {% endif %}
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<h2>Category Breakdown</h2>
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Total</th>
      <th>Agreements</th>
      <th>Agreement Rate</th>
    </tr>
  </thead>
  <tbody>
    {% for cat in categories %}
    <tr>
      <td><span class="tag tag-category">{{ cat.name }}</span></td>
      <td>{{ cat.total }}</td>
      <td>{{ cat.agreements }}</td>
      <td>{{ "%.1f"|format(cat.rate * 100) }}%</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<div class="footer">
  Generated by JudgeBench v0.1.0
</div>

</body>
</html>
"""
)


def _severity_class(value: float, thresholds: tuple[float, float] = (0.1, 0.3)) -> str:
    """Return CSS class based on severity thresholds."""
    abs_val = abs(value)
    if abs_val < thresholds[0]:
        return "good"
    elif abs_val < thresholds[1]:
        return "warn"
    return "bad"


def _interpret_kappa(k: float) -> str:
    """Interpret Cohen's kappa value."""
    if k < 0:
        return "Poor (worse than chance)"
    elif k < 0.2:
        return "Slight agreement"
    elif k < 0.4:
        return "Fair agreement"
    elif k < 0.6:
        return "Moderate agreement"
    elif k < 0.8:
        return "Substantial agreement"
    return "Almost perfect agreement"


def _interpret_alpha(a: float) -> str:
    """Interpret Krippendorff's alpha."""
    if a < 0.667:
        return "Unreliable"
    elif a < 0.8:
        return "Tentatively acceptable"
    return "Reliable"


def generate_report(
    pairs: list[LabeledPair],
    verdicts: list[JudgeVerdict],
    bias: BiasReport,
    agreement: AgreementMetrics,
    output_path: str | Path,
) -> Path:
    """Generate an HTML bias dashboard report.

    Args:
        pairs: List of labeled pairs
        verdicts: List of judge verdicts
        bias: Aggregated bias metrics
        agreement: Agreement metrics
        output_path: Where to write the HTML file

    Returns:
        Path to the generated HTML file
    """
    pair_map = {p.id: p for p in pairs}
    verdict_map = {v.pair_id: v for v in verdicts}

    # Build drill-down rows
    drilldown = []
    for v in verdicts:
        pair = pair_map.get(v.pair_id)
        if pair is None:
            continue
        agrees = pair.human_label == v.forward_choice or pair.human_label == "tie"
        drilldown.append(
            {
                "pair_id": v.pair_id,
                "category": pair.category,
                "human_label": pair.human_label,
                "forward_choice": v.forward_choice,
                "reversed_choice": v.reversed_choice,
                "agrees": agrees,
            }
        )

    # Build category breakdown
    cat_stats: dict[str, dict[str, int]] = {}
    for row in drilldown:
        cat = row["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "agreements": 0}
        cat_stats[cat]["total"] += 1
        if row["agrees"]:
            cat_stats[cat]["agreements"] += 1

    categories = [
        {
            "name": name,
            "total": stats["total"],
            "agreements": stats["agreements"],
            "rate": stats["agreements"] / stats["total"] if stats["total"] > 0 else 0,
        }
        for name, stats in sorted(cat_stats.items())
    ]

    html = DASHBOARD_TEMPLATE.render(
        bias=bias,
        agreement=agreement,
        drilldown=drilldown,
        categories=categories,
        position_class=_severity_class(bias.position_bias_rate),
        verbosity_class=_severity_class(bias.verbosity_bias_rho),
        self_enhance_class=_severity_class(bias.self_enhance_delta),
        leniency_class=_severity_class(bias.leniency_score),
        kappa_interp=_interpret_kappa(agreement.cohens_kappa),
        alpha_interp=_interpret_alpha(agreement.krippendorffs_alpha),
    )

    output_path = Path(output_path)
    output_path.write_text(html)
    return output_path
