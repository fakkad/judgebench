"""CLI entry point for JudgeBench."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from judgebench import __version__
from judgebench.agreement import compute_agreement
from judgebench.bias import (
    detect_leniency,
    detect_position,
    detect_self_enhance,
    detect_verbosity,
)
from judgebench.judge import DEFAULT_MODEL, run_judge
from judgebench.loader import load_pairs
from judgebench.models import BiasReport, JudgeVerdict
from judgebench.reporter import generate_report

app = typer.Typer(
    name="judgebench",
    help="Evaluate LLM judge quality with bias detection and agreement metrics.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to labeled pairs (YAML or JSONL)"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for results JSONL"),
    ] = Path("results.jsonl"),
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Anthropic model ID for the judge"),
    ] = DEFAULT_MODEL,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key"),
    ] = None,
) -> None:
    """Run the judge on a labeled dataset and save results."""
    console.print(f"[bold cyan]JudgeBench[/] v{__version__}")
    console.print(f"Loading pairs from [bold]{data}[/]...")

    pairs = load_pairs(data)
    console.print(f"Loaded [bold]{len(pairs)}[/] pairs across {len({p.category for p in pairs})} categories")

    console.print(f"Running judge: [bold]{model}[/]")
    verdicts = run_judge(pairs, model=model, api_key=api_key)

    # Write results
    with open(output, "w") as f:
        for v in verdicts:
            f.write(v.model_dump_json() + "\n")

    console.print(f"[green]Results written to {output}[/]")

    # Quick summary
    consistent = sum(1 for v in verdicts if v.consistent)
    console.print(
        f"Consistency: {consistent}/{len(verdicts)} "
        f"({consistent / len(verdicts) * 100:.1f}%)"
    )


@app.command()
def report(
    results: Annotated[
        Path,
        typer.Argument(help="Path to results JSONL from 'run' command"),
    ],
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to original labeled pairs"),
    ] = Path("data/sample_pairs.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for HTML report"),
    ] = Path("report.html"),
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Judge model ID (for self-enhancement)"),
    ] = DEFAULT_MODEL,
) -> None:
    """Generate an HTML bias dashboard from results."""
    console.print(f"[bold cyan]JudgeBench[/] Report Generator")

    # Load data
    pairs = load_pairs(data)

    verdicts: list[JudgeVerdict] = []
    with open(results) as f:
        for line in f:
            line = line.strip()
            if line:
                verdicts.append(JudgeVerdict(**json.loads(line)))

    console.print(f"Loaded {len(pairs)} pairs and {len(verdicts)} verdicts")

    # Compute bias metrics
    position = detect_position(verdicts, pairs)
    verbosity = detect_verbosity(pairs, verdicts)
    self_enhance = detect_self_enhance(pairs, verdicts, model)
    leniency = detect_leniency(pairs, verdicts)

    bias = BiasReport(
        position_bias_rate=position,
        verbosity_bias_rho=verbosity,
        self_enhance_delta=self_enhance,
        leniency_score=leniency,
    )

    # Compute agreement
    agreement = compute_agreement(pairs, verdicts)

    # Display summary
    table = Table(title="Bias Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Position Bias Rate", f"{position:.3f}")
    table.add_row("Verbosity Bias (rho)", f"{verbosity:.3f}")
    table.add_row("Self-Enhancement Delta", f"{self_enhance:.3f}")
    table.add_row("Leniency Score", f"{leniency:.3f}")
    console.print(table)

    table2 = Table(title="Agreement Metrics")
    table2.add_column("Metric", style="cyan")
    table2.add_column("Value", style="bold")
    table2.add_row("Cohen's Kappa", f"{agreement.cohens_kappa:.4f}")
    table2.add_row("Krippendorff's Alpha", f"{agreement.krippendorffs_alpha:.4f}")
    table2.add_row("Spearman rho", f"{agreement.spearman_rho:.4f} (p={agreement.spearman_p:.4f})")
    table2.add_row("McNemar's chi2", f"{agreement.mcnemars_chi2:.4f} (p={agreement.mcnemars_p:.4f})")
    console.print(table2)

    # Generate HTML
    report_path = generate_report(pairs, verdicts, bias, agreement, output)
    console.print(f"[green]Report written to {report_path}[/]")


@app.command()
def version() -> None:
    """Show JudgeBench version."""
    console.print(f"[bold cyan]JudgeBench[/] v{__version__}")


if __name__ == "__main__":
    app()
