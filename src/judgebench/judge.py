"""Judge runner — runs Claude as a pairwise judge with position swapping."""

import json

import anthropic
from rich.progress import Progress, SpinnerColumn, TextColumn

from judgebench.models import JudgeVerdict, LabeledPair

DEFAULT_MODEL = "claude-haiku-4-5-20241022"

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating the quality of two AI responses to a prompt.
Compare Response A and Response B carefully. Choose the better response.
You MUST respond with valid JSON only, no other text.
Format: {"choice": "a" or "b", "reasoning": "brief explanation"}"""

JUDGE_USER_TEMPLATE = """Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Respond with JSON only: {{"choice": "a" or "b", "reasoning": "..."}}"""


def _call_judge(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    response_a: str,
    response_b: str,
) -> tuple[str, str]:
    """Make a single judge call and parse the result.

    Returns:
        Tuple of (choice, reasoning)
    """
    message = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                ),
            }
        ],
    )

    raw = message.content[0].text.strip()

    # Parse JSON from response, handling potential markdown wrapping
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    result = json.loads(raw)
    choice = result["choice"].lower().strip()
    reasoning = result.get("reasoning", "")

    if choice not in ("a", "b"):
        raise ValueError(f"Invalid choice from judge: {choice}")

    return choice, reasoning


def run_judge(
    pairs: list[LabeledPair],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> list[JudgeVerdict]:
    """Run the judge on all pairs in both orderings.

    For each pair:
    1. Forward: present (response_a, response_b) and get a choice
    2. Reversed: present (response_b, response_a) and get a choice

    Consistency check: if forward picks "a", reversed should pick "b"
    (since the positions are swapped).

    Args:
        pairs: List of labeled pairs to judge
        model: Anthropic model ID to use as judge
        api_key: Optional API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        List of JudgeVerdict objects
    """
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    verdicts = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Judging pairs...", total=len(pairs))

        for pair in pairs:
            progress.update(task, description=f"Judging {pair.id}...")

            # Forward: A in position 1, B in position 2
            fwd_choice, fwd_reasoning = _call_judge(
                client, model, pair.prompt, pair.response_a, pair.response_b
            )

            # Reversed: B in position 1, A in position 2
            rev_choice_raw, rev_reasoning = _call_judge(
                client, model, pair.prompt, pair.response_b, pair.response_a
            )

            # Map reversed choice back to original labels:
            # If reversed picks "a" (position 1 = original B), that means "b"
            # If reversed picks "b" (position 2 = original A), that means "a"
            rev_choice = "b" if rev_choice_raw == "a" else "a"

            # Consistent if both orderings agree on the same original response
            consistent = fwd_choice == rev_choice

            verdicts.append(
                JudgeVerdict(
                    pair_id=pair.id,
                    forward_choice=fwd_choice,
                    reversed_choice=rev_choice,
                    forward_reasoning=fwd_reasoning,
                    reversed_reasoning=rev_reasoning,
                    consistent=consistent,
                )
            )

            progress.advance(task)

    return verdicts
