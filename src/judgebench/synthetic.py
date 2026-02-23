"""Synthetic data generator.

Given a seed dataset, generates additional pairs by calling an LLM
to paraphrase, scale difficulty, and diversify categories.
"""

from __future__ import annotations

import asyncio
import json
import random

from judgebench.models import Dataset, LabeledPair
from judgebench.providers import get_provider
from judgebench.providers.base import BaseProvider


PARAPHRASE_PROMPT = """\
You are generating synthetic evaluation data. Given a prompt and two responses
with a known quality label, create a NEW version by paraphrasing both responses
while preserving the quality difference.

Original prompt: {prompt}

Original Response A:
{response_a}

Original Response B:
{response_b}

The better response is: {human_label}

Generate a new version. Change the wording substantially but keep the same
quality relationship ({human_label} should still be better, or keep it a tie).

Output JSON:
{{
  "prompt": "...",
  "response_a": "...",
  "response_b": "...",
  "human_label": "{human_label}",
  "category": "{category}"
}}
"""


async def _generate_one(
    provider: BaseProvider,
    pair: LabeledPair,
    new_id: str,
    semaphore: asyncio.Semaphore,
) -> LabeledPair | None:
    """Generate one synthetic pair from a seed pair."""
    category = pair.metadata.get("category", "general")
    prompt_text = PARAPHRASE_PROMPT.format(
        prompt=pair.prompt,
        response_a=pair.response_a,
        response_b=pair.response_b,
        human_label=pair.human_label,
        category=category,
    )

    try:
        async with semaphore:
            result = await provider.judge(prompt_text)

        return LabeledPair(
            id=new_id,
            prompt=result.get("prompt", pair.prompt),
            response_a=result.get("response_a", pair.response_a),
            response_b=result.get("response_b", pair.response_b),
            human_label=result.get("human_label", pair.human_label),
            metadata={
                "category": result.get("category", category),
                "source": "synthetic",
                "seed_id": pair.id,
            },
        )
    except Exception:
        return None


async def generate_synthetic(
    seed_dataset: Dataset,
    count: int = 170,
    provider_name: str = "anthropic",
    model: str = "claude-haiku-4-5-20251001",
    concurrency: int = 5,
    progress_callback: callable | None = None,
) -> Dataset:
    """Generate synthetic pairs from a seed dataset.

    Distributes generation across seed pairs, creating ~count/len(seed) copies
    per seed pair, with paraphrasing and category diversification.

    Args:
        seed_dataset: The seed dataset with manually labeled pairs.
        count: Target number of synthetic pairs to generate.
        provider_name: LLM provider to use for generation.
        model: Model name for generation.
        concurrency: Max concurrent API calls.
        progress_callback: Optional callback(completed, total).

    Returns:
        New Dataset containing only the synthetic pairs.
    """
    provider_cls = get_provider(provider_name)
    provider = provider_cls(model=model)

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    # Distribute count across seed pairs
    seeds = seed_dataset.pairs
    if not seeds:
        raise ValueError("Seed dataset has no pairs")

    tasks = []
    for i in range(count):
        seed_pair = seeds[i % len(seeds)]
        new_id = f"synthetic-{i + 1:04d}"

        async def _run(pair=seed_pair, nid=new_id):
            nonlocal completed
            result = await _generate_one(provider, pair, nid, semaphore)
            completed += 1
            if progress_callback:
                progress_callback(completed, count)
            return result

        tasks.append(_run())

    results = await asyncio.gather(*tasks)
    pairs = [r for r in results if r is not None]

    return Dataset(
        name=f"{seed_dataset.name}-synthetic",
        description=f"Synthetic expansion of {seed_dataset.name} ({len(pairs)} pairs)",
        pairs=pairs,
    )
