#!/usr/bin/env bash
# Basic judgebench usage example
set -euo pipefail

# Validate the example dataset
judgebench validate datasets/example_pairs.yaml

# Run judge evaluation (requires ANTHROPIC_API_KEY)
# judgebench run datasets/example_pairs.yaml --judge-model claude-haiku-4-5-20251001 --output-dir ./results

# Re-generate dashboard from saved results
# judgebench analyze results/results.json --dataset datasets/example_pairs.yaml

# Compare two judge results
# judgebench compare results_judge1/results.json results_judge2/results.json

# Generate synthetic pairs from seed data
# judgebench generate-synthetic --base-dataset datasets/example_pairs.yaml --count 170 --output synthetic.yaml

# Initialize a new example dataset
# judgebench init --output my_pairs.yaml
