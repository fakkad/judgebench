"""Judge runner — evaluates an LLM judge against labeled pairs."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

from judgebench.models import (
    BenchResult,
    BiasReport,
    Dataset,
    JudgeConfig,
    JudgeVerdict,
    LabeledPair,
)
from judgebench.providers import get_provider
from judgebench.providers.base import BaseProvider
from judgebench.bias import (
    detect_position_bias,
    detect_verbosity_bias,
    detect_leniency_bias,
    detect_self_enhancement_bias,
)
from judgebench.stats import (
    cohens_kappa,
    krippendorff_alpha,
    raw_agreement,
    confusion_matrix,
    per_category_agreement,
)


JUDGE_PROMPT_TEMPLATE = """\
You are evaluating two responses to a prompt. Choose which response is better.

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Evaluate based on: helpfulness, accuracy, clarity, and completeness.
Output your verdict as JSON: {{"winner": "A" or "B" or "tie", "confidence": 0.0-1.0, "reasoning": "..."}}
"""


def _build_prompt(prompt: str, response_a: str, response_b: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )


async def _judge_single(
    provider: BaseProvider,
    pair: LabeledPair,
    position: str,
    semaphore: asyncio.Semaphore,
) -> JudgeVerdict:
    """Run the judge on a single pair in one ordering."""
    if position == "original":
        prompt_text = _build_prompt(pair.prompt, pair.response_a, pair.response_b)
    else:
        # Swapped: show B first as "Response A" and A second as "Response B"
        prompt_text = _build_prompt(pair.prompt, pair.response_b, pair.response_a)

    async with semaphore:
        result = await provider.judge(prompt_text)

    winner = result.get("winner", "tie")
    # Normalize winner value
    if winner.upper() in ("A", "B"):
        winner = winner.upper()
    else:
        winner = "tie"

    confidence = float(result.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    reasoning = result.get("reasoning", "")

    return JudgeVerdict(
        pair_id=pair.id,
        judge_label=winner,
        confidence=confidence,
        reasoning=reasoning,
        position=position,
    )


async def run_judge(
    dataset: Dataset,
    judge_config: JudgeConfig,
    concurrency: int = 5,
    progress_callback: callable | None = None,
) -> BenchResult:
    """Run a judge against a labeled dataset.

    For each pair, runs the judge in both orderings (original + swapped)
    to enable position bias detection.

    Args:
        dataset: The labeled evaluation dataset.
        judge_config: Configuration for the judge LLM.
        concurrency: Max concurrent LLM calls.
        progress_callback: Optional callback(completed, total) for progress.

    Returns:
        BenchResult with verdicts, agreement metrics, and bias reports.
    """
    provider_cls = get_provider(judge_config.provider)
    provider = provider_cls(
        model=judge_config.model,
        params=judge_config.params,
        system_prompt=judge_config.system_prompt,
    )

    semaphore = asyncio.Semaphore(concurrency)
    total_tasks = len(dataset.pairs) * 2
    completed = 0

    async def _run_with_progress(pair: LabeledPair, position: str) -> JudgeVerdict:
        nonlocal completed
        verdict = await _judge_single(provider, pair, position, semaphore)
        completed += 1
        if progress_callback:
            progress_callback(completed, total_tasks)
        return verdict

    # Create tasks for both orderings
    tasks = []
    for pair in dataset.pairs:
        tasks.append(_run_with_progress(pair, "original"))
        tasks.append(_run_with_progress(pair, "swapped"))

    verdicts = await asyncio.gather(*tasks)
    verdicts = list(verdicts)

    return _compute_results(verdicts, dataset, judge_config)


def _compute_results(
    verdicts: list[JudgeVerdict],
    dataset: Dataset,
    judge_config: JudgeConfig,
) -> BenchResult:
    """Compute agreement metrics and bias reports from verdicts."""
    # Get original-ordering verdicts for agreement computation
    original_verdicts = [v for v in verdicts if v.position == "original"]
    pair_map = {p.id: p for p in dataset.pairs}

    # Build label lists aligned by pair_id
    human_labels = []
    judge_labels = []
    for v in original_verdicts:
        if v.pair_id in pair_map:
            human_labels.append(pair_map[v.pair_id].human_label)
            judge_labels.append(v.judge_label)

    categories = ["A", "B", "tie"]

    # Agreement metrics
    metrics = {}
    if human_labels and judge_labels:
        metrics["raw_agreement"] = raw_agreement(human_labels, judge_labels)
        metrics["cohens_kappa"] = cohens_kappa(human_labels, judge_labels, categories)

        # Krippendorff's alpha: encode as integers
        label_to_int = {"A": 0, "B": 1, "tie": 2}
        human_ints = [label_to_int[l] for l in human_labels]
        judge_ints = [label_to_int[l] for l in judge_labels]
        metrics["krippendorff_alpha_nominal"] = krippendorff_alpha(
            [human_ints, judge_ints], level="nominal"
        )

        # McNemar (comparing judge vs a "perfect" rater that matches human)
        from judgebench.stats import mcnemar_test

        mcnemar_result = mcnemar_test(judge_labels, human_labels, human_labels)
        metrics["mcnemar"] = mcnemar_result

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(human_labels, judge_labels, categories)

        # Per-category
        metrics["per_category_agreement"] = per_category_agreement(
            human_labels, judge_labels, categories
        )

    # Bias reports
    bias_reports = [
        detect_position_bias(verdicts),
        detect_verbosity_bias(verdicts, dataset.pairs),
        detect_leniency_bias(verdicts, dataset.pairs),
        detect_self_enhancement_bias(verdicts, dataset.pairs, judge_config),
    ]

    # Overall reliability: weighted combination
    kappa = metrics.get("cohens_kappa", 0.0)
    alpha = metrics.get("krippendorff_alpha_nominal", 0.0)
    overall = 0.5 * max(0.0, kappa) + 0.5 * max(0.0, alpha)

    # Penalize for flagged biases
    flagged_count = sum(1 for b in bias_reports if b.flagged)
    overall *= max(0.0, 1.0 - 0.1 * flagged_count)

    return BenchResult(
        judge_config=judge_config,
        verdicts=verdicts,
        agreement_metrics=metrics,
        bias_reports=bias_reports,
        overall_reliability=min(1.0, max(0.0, overall)),
    )


def compute_results_from_verdicts(
    verdicts: list[JudgeVerdict],
    dataset: Dataset,
    judge_config: JudgeConfig,
) -> BenchResult:
    """Public wrapper for computing results from pre-existing verdicts."""
    return _compute_results(verdicts, dataset, judge_config)
