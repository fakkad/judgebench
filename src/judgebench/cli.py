"""CLI interface for judgebench."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
import yaml

from judgebench.models import BenchResult, Dataset, JudgeConfig, JudgeVerdict

app = typer.Typer(
    name="judgebench",
    help="LLM judge quality evaluator with bias detection and calibration dashboard.",
    no_args_is_help=True,
)


def _load_dataset(path: str) -> Dataset:
    """Load and validate a YAML dataset."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Dataset(**raw)


def _save_results(result: BenchResult, output_dir: Path) -> Path:
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.json"
    out_path.write_text(result.model_dump_json(indent=2))
    return out_path


@app.command()
def run(
    dataset_path: str = typer.Argument(..., help="Path to labeled dataset YAML"),
    judge_model: str = typer.Option("claude-haiku-4-5-20251001", "--judge-model", "-m", help="Judge model name"),
    judge_provider: str = typer.Option("anthropic", "--judge-provider", "-p", help="Judge provider (anthropic/openai)"),
    output_dir: str = typer.Option("./results", "--output-dir", "-o", help="Output directory"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Max concurrent LLM calls"),
    system_prompt: str = typer.Option(None, "--system-prompt", "-s", help="Custom system prompt for judge"),
):
    """Run judge evaluation against a labeled dataset."""
    typer.echo(f"Loading dataset: {dataset_path}")
    dataset = _load_dataset(dataset_path)
    typer.echo(f"Loaded {len(dataset.pairs)} pairs from '{dataset.name}'")

    judge_config = JudgeConfig(
        provider=judge_provider,
        model=judge_model,
        system_prompt=system_prompt,
    )

    def progress(done: int, total: int):
        typer.echo(f"  [{done}/{total}] verdicts collected", nl=False)
        typer.echo("\r", nl=False)

    typer.echo(f"Running judge: {judge_provider}/{judge_model}")
    typer.echo(f"Concurrency: {concurrency}")

    from judgebench.judge_runner import run_judge

    result = asyncio.run(run_judge(dataset, judge_config, concurrency, progress))
    typer.echo("")

    # Save results
    out = Path(output_dir)
    results_path = _save_results(result, out)
    typer.echo(f"Results saved: {results_path}")

    # Generate dashboard
    from judgebench.dashboard import generate_dashboard

    dash_path = generate_dashboard(result, dataset, str(out / "dashboard.html"))
    typer.echo(f"Dashboard saved: {dash_path}")

    # Summary
    typer.echo("")
    typer.echo("--- Summary ---")
    typer.echo(f"Overall reliability: {result.overall_reliability:.3f}")
    typer.echo(f"Raw agreement: {result.agreement_metrics.get('raw_agreement', 0):.3f}")
    typer.echo(f"Cohen's kappa: {result.agreement_metrics.get('cohens_kappa', 0):.3f}")
    typer.echo(f"Krippendorff's alpha: {result.agreement_metrics.get('krippendorff_alpha_nominal', 0):.3f}")

    for b in result.bias_reports:
        flag = " [FLAGGED]" if b.flagged else ""
        typer.echo(f"Bias ({b.bias_type}): {b.score:.3f}{flag}")

    # Exit code based on reliability
    alpha = result.agreement_metrics.get("krippendorff_alpha_nominal", 0)
    if alpha < 0.67:
        typer.echo("\nJudge is UNRELIABLE (alpha < 0.67)")
        raise typer.Exit(code=1)
    else:
        typer.echo("\nJudge is RELIABLE (alpha >= 0.67)")


@app.command()
def analyze(
    results_path: str = typer.Argument(..., help="Path to results.json"),
    dataset_path: str = typer.Option(None, "--dataset", "-d", help="Original dataset YAML (for per-pair details)"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory for dashboard"),
):
    """Re-generate dashboard from saved results."""
    raw = json.loads(Path(results_path).read_text())
    result = BenchResult(**raw)

    dataset = None
    if dataset_path:
        dataset = _load_dataset(dataset_path)

    out_dir = output_dir or str(Path(results_path).parent)
    from judgebench.dashboard import generate_dashboard

    dash_path = generate_dashboard(result, dataset, str(Path(out_dir) / "dashboard.html"))
    typer.echo(f"Dashboard saved: {dash_path}")


@app.command()
def compare(
    results1: str = typer.Argument(..., help="First results.json"),
    results2: str = typer.Argument(..., help="Second results.json"),
    output_dir: str = typer.Option("./comparison", "--output-dir", "-o"),
):
    """Compare two judge results."""
    from judgebench.compare import compare_results

    r1 = BenchResult(**json.loads(Path(results1).read_text()))
    r2 = BenchResult(**json.loads(Path(results2).read_text()))

    comparison = compare_results(r1, r2)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    typer.echo(f"Comparison saved: {out_path}")

    # Print summary
    typer.echo("")
    typer.echo(f"Judge A: {comparison['judge_a']['provider']}/{comparison['judge_a']['model']}")
    typer.echo(f"Judge B: {comparison['judge_b']['provider']}/{comparison['judge_b']['model']}")
    typer.echo("")

    rel = comparison["reliability"]
    typer.echo(f"Reliability: A={rel['judge_a']:.3f}  B={rel['judge_b']:.3f}  (better: {rel['better']})")

    for key, data in comparison["agreement_metrics"].items():
        typer.echo(f"{key}: A={data['judge_a']:.3f}  B={data['judge_b']:.3f}  (better: {data['better']})")

    for bt, data in comparison["bias_comparison"].items():
        typer.echo(f"Bias ({bt}): A={data['judge_a']:.3f}  B={data['judge_b']:.3f}  (better: {data['better']})")


@app.command()
def validate(
    dataset_path: str = typer.Argument(..., help="Path to dataset YAML"),
):
    """Validate a dataset YAML file."""
    try:
        dataset = _load_dataset(dataset_path)
        typer.echo(f"Valid dataset: '{dataset.name}'")
        typer.echo(f"  Pairs: {len(dataset.pairs)}")
        typer.echo(f"  Description: {dataset.description}")

        # Check for duplicate IDs
        ids = [p.id for p in dataset.pairs]
        dupes = [x for x in ids if ids.count(x) > 1]
        if dupes:
            typer.echo(f"  WARNING: Duplicate IDs: {set(dupes)}")
        else:
            typer.echo("  No duplicate IDs")

        # Label distribution
        from collections import Counter
        dist = Counter(p.human_label for p in dataset.pairs)
        typer.echo(f"  Labels: {dict(dist)}")

    except Exception as e:
        typer.echo(f"Invalid dataset: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def init(
    output: str = typer.Option("example_pairs.yaml", "--output", "-o", help="Output file path"),
):
    """Generate an example dataset YAML file."""
    example = {
        "name": "judge-eval-v1",
        "description": "Example labeled pairs for judge calibration",
        "pairs": [
            {
                "id": "pair-001",
                "prompt": "Explain quantum computing to a 5-year-old",
                "response_a": "Imagine you have a magic coin that can be both heads and tails at the same time! That's kind of like how a quantum computer works. Regular computers use coins that are either heads or tails, but quantum computers use these magic coins called qubits.",
                "response_b": "Quantum computing uses qubits instead of classical bits to perform computations leveraging superposition and entanglement principles.",
                "human_label": "A",
                "metadata": {"category": "explanation", "difficulty": "easy"},
            },
            {
                "id": "pair-002",
                "prompt": "Write a haiku about programming",
                "response_a": "Code flows like water\nBugs swim upstream in the night\nTests catch them at dawn",
                "response_b": "Programming is fun\nI like to write code all day\nComputers are cool",
                "human_label": "A",
                "metadata": {"category": "creative", "difficulty": "easy"},
            },
            {
                "id": "pair-003",
                "prompt": "What is the capital of France?",
                "response_a": "Paris",
                "response_b": "The capital of France is Paris. It has been the capital since the 10th century and is the country's largest city, known for landmarks like the Eiffel Tower and the Louvre.",
                "human_label": "B",
                "metadata": {"category": "factual", "difficulty": "easy"},
            },
        ],
    }

    with open(output, "w") as f:
        yaml.dump(example, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    typer.echo(f"Example dataset written to: {output}")


@app.command(name="generate-synthetic")
def generate_synthetic_cmd(
    base_dataset: str = typer.Option(..., "--base-dataset", "-b", help="Seed dataset YAML"),
    count: int = typer.Option(170, "--count", "-n", help="Number of synthetic pairs to generate"),
    model: str = typer.Option("claude-haiku-4-5-20251001", "--model", "-m"),
    provider: str = typer.Option("anthropic", "--provider", "-p"),
    output: str = typer.Option("synthetic_pairs.yaml", "--output", "-o"),
    concurrency: int = typer.Option(5, "--concurrency", "-c"),
):
    """Generate synthetic pairs from a seed dataset."""
    from judgebench.synthetic import generate_synthetic

    dataset = _load_dataset(base_dataset)
    typer.echo(f"Seed dataset: {len(dataset.pairs)} pairs")
    typer.echo(f"Generating {count} synthetic pairs...")

    def progress(done: int, total: int):
        typer.echo(f"  [{done}/{total}]", nl=False)
        typer.echo("\r", nl=False)

    result = asyncio.run(
        generate_synthetic(dataset, count, provider, model, concurrency, progress)
    )
    typer.echo("")

    # Save as YAML
    with open(output, "w") as f:
        yaml.dump(result.model_dump(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    typer.echo(f"Generated {len(result.pairs)} synthetic pairs -> {output}")


if __name__ == "__main__":
    app()
