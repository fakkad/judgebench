"""Data models for JudgeBench."""

from typing import Any

from pydantic import BaseModel


class LabeledPair(BaseModel):
    """A labeled pair of responses for pairwise comparison."""

    id: str
    prompt: str
    response_a: str
    response_b: str
    human_label: str  # "a", "b", or "tie"
    category: str  # "factual", "creative", "reasoning", "safety", "coding"
    metadata: dict[str, Any] = {}


class JudgeVerdict(BaseModel):
    """Result of judging a single pair in both orderings."""

    pair_id: str
    forward_choice: str  # "a" or "b"
    reversed_choice: str  # "a" or "b" (after swapping positions)
    forward_reasoning: str
    reversed_reasoning: str
    consistent: bool  # forward matches reversed after accounting for swap


class BiasReport(BaseModel):
    """Aggregated bias metrics across all pairs."""

    position_bias_rate: float  # fraction of inconsistent swaps
    verbosity_bias_rho: float  # Spearman correlation(score, length)
    self_enhance_delta: float  # score difference when model evaluates own output
    leniency_score: float  # TPR / (TPR + TNR) asymmetry


class AgreementMetrics(BaseModel):
    """Statistical agreement between judge and human labels."""

    cohens_kappa: float
    krippendorffs_alpha: float
    spearman_rho: float
    spearman_p: float
    mcnemars_chi2: float
    mcnemars_p: float
