# JudgeBench

CLI tool that evaluates LLM judge quality. Runs judges on labeled datasets in both response orders, detects bias types, and produces calibration reports with statistical agreement metrics.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Run judge evaluation on a dataset
judgebench run --data data/sample_pairs.yaml --output results.jsonl

# Generate HTML bias dashboard from results
judgebench report results.jsonl --output report.html

# Show version
judgebench version
```

## Bias Detectors

- **Position bias**: Fraction of pairs where swapping A/B order changes the verdict
- **Verbosity bias**: Spearman correlation between chosen response length and score
- **Self-enhancement bias**: Score delta when the judge evaluates its own outputs
- **Leniency bias**: TPR asymmetry vs human labels

## Agreement Metrics

- Cohen's kappa (judge vs human)
- Krippendorff's alpha (inter-rater reliability)
- Spearman rank correlation
- McNemar's test (paired correctness comparison)

## Development

```bash
pytest -v
```
