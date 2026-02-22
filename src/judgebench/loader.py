"""Load labeled pairs from YAML or JSONL files."""

import json
from pathlib import Path

import yaml

from judgebench.models import LabeledPair


def load_pairs(path: str | Path) -> list[LabeledPair]:
    """Load labeled pairs from a YAML or JSONL file.

    Args:
        path: Path to the data file (.yaml/.yml or .jsonl)

    Returns:
        List of LabeledPair objects

    Raises:
        ValueError: If the file format is not supported
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    elif suffix == ".jsonl":
        return _load_jsonl(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .yaml, .yml, or .jsonl")


def _load_yaml(path: Path) -> list[LabeledPair]:
    """Load pairs from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("YAML file must contain a list of pairs at the top level")

    return [LabeledPair(**item) for item in data]


def _load_jsonl(path: Path) -> list[LabeledPair]:
    """Load pairs from a JSONL file."""
    pairs = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pairs.append(LabeledPair(**data))
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Error on line {line_num}: {e}") from e
    return pairs
