"""Core pydantic models for judgebench."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Label = Literal["A", "B", "tie"]


class LabeledPair(BaseModel):
    """A single evaluation pair with a human-assigned ground-truth label."""

    id: str
    prompt: str
    response_a: str
    response_b: str
    human_label: Label
    metadata: dict = Field(default_factory=dict)


class JudgeConfig(BaseModel):
    """Configuration for the judge LLM."""

    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    params: dict = Field(default_factory=dict)
    system_prompt: str | None = None


class JudgeVerdict(BaseModel):
    """A single verdict from the judge on one pair in one ordering."""

    pair_id: str
    judge_label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    position: str = Field(
        description="Which ordering was used: 'original' (A first) or 'swapped' (B first)"
    )


class BiasReport(BaseModel):
    """Result of a single bias detection analysis."""

    bias_type: str
    score: float = Field(ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)
    flagged: bool = False


class BenchResult(BaseModel):
    """Full benchmark result for a single judge."""

    judge_config: JudgeConfig
    verdicts: list[JudgeVerdict] = Field(default_factory=list)
    agreement_metrics: dict = Field(default_factory=dict)
    bias_reports: list[BiasReport] = Field(default_factory=list)
    overall_reliability: float = Field(ge=0.0, le=1.0, default=0.0)


class Dataset(BaseModel):
    """A dataset of labeled pairs."""

    name: str
    description: str = ""
    pairs: list[LabeledPair] = Field(default_factory=list)
