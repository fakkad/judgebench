"""Generate synthetic labeled pairs by expanding the sample dataset via Claude.

Usage:
    python data/synthetic_gen.py --count 200 --output data/synthetic_pairs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic
import yaml

CATEGORIES = ["factual", "creative", "reasoning", "safety", "coding"]

GENERATION_PROMPT = """\
Generate a labeled pair for evaluating LLM judge quality.

Category: {category}

Create a realistic prompt and two responses where one is clearly better.
The better response should be labeled as the winner.

Respond with JSON only:
{{
  "prompt": "the user prompt",
  "response_a": "first response",
  "response_b": "second response",
  "human_label": "a" or "b",
  "category": "{category}"
}}

Make the responses substantively different in quality — one should have
clear errors, be shallow, or miss the point, while the other should be
accurate, thorough, and well-structured. Vary which position (a or b) wins.
"""


def generate_pairs(count: int, api_key: str | None = None) -> list[dict]:
    """Generate synthetic labeled pairs using Claude."""
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    pairs = []
    per_category = count // len(CATEGORIES)
    remainder = count % len(CATEGORIES)

    for i, category in enumerate(CATEGORIES):
        n = per_category + (1 if i < remainder else 0)

        for j in range(n):
            pair_id = f"syn-{category[:3]}-{j + 1:03d}"

            try:
                message = client.messages.create(
                    model="claude-haiku-4-5-20241022",
                    max_tokens=1024,
                    temperature=0.8,
                    messages=[
                        {
                            "role": "user",
                            "content": GENERATION_PROMPT.format(category=category),
                        }
                    ],
                )

                raw = message.content[0].text.strip()
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1])

                data = json.loads(raw)
                data["id"] = pair_id
                data["category"] = category
                data["metadata"] = {"synthetic": True}
                pairs.append(data)
                print(f"  Generated {pair_id}", file=sys.stderr)

            except Exception as e:
                print(f"  Error generating {pair_id}: {e}", file=sys.stderr)
                continue

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic labeled pairs")
    parser.add_argument("--count", type=int, default=200, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default="data/synthetic_pairs.jsonl")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    print(f"Generating {args.count} synthetic pairs...", file=sys.stderr)
    pairs = generate_pairs(args.count, args.api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Wrote {len(pairs)} pairs to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
