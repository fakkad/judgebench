"""Tests for pydantic models."""

import pytest
from pydantic import ValidationError

from judgebench.models import (
    BenchResult,
    BiasReport,
    Dataset,
    JudgeConfig,
    JudgeVerdict,
    LabeledPair,
)


class TestLabeledPair:
    def test_valid_pair(self):
        p = LabeledPair(
            id="p1",
            prompt="Test",
            response_a="A",
            response_b="B",
            human_label="A",
        )
        assert p.id == "p1"
        assert p.human_label == "A"
        assert p.metadata == {}

    def test_with_metadata(self):
        p = LabeledPair(
            id="p2",
            prompt="Test",
            response_a="A",
            response_b="B",
            human_label="tie",
            metadata={"category": "test"},
        )
        assert p.metadata["category"] == "test"

    def test_invalid_label(self):
        with pytest.raises(ValidationError):
            LabeledPair(
                id="p3",
                prompt="Test",
                response_a="A",
                response_b="B",
                human_label="C",
            )


class TestJudgeConfig:
    def test_defaults(self):
        c = JudgeConfig()
        assert c.provider == "anthropic"
        assert c.model == "claude-haiku-4-5-20251001"
        assert c.params == {}
        assert c.system_prompt is None

    def test_custom(self):
        c = JudgeConfig(
            provider="openai",
            model="gpt-4o",
            params={"temperature": 0.0},
            system_prompt="Be strict.",
        )
        assert c.provider == "openai"
        assert c.system_prompt == "Be strict."


class TestJudgeVerdict:
    def test_valid(self):
        v = JudgeVerdict(
            pair_id="p1",
            judge_label="A",
            confidence=0.95,
            reasoning="A is better",
            position="original",
        )
        assert v.confidence == 0.95

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(
                pair_id="p1",
                judge_label="A",
                confidence=1.5,
                position="original",
            )
        with pytest.raises(ValidationError):
            JudgeVerdict(
                pair_id="p1",
                judge_label="B",
                confidence=-0.1,
                position="original",
            )


class TestBiasReport:
    def test_valid(self):
        b = BiasReport(bias_type="position", score=0.5, flagged=True)
        assert b.flagged

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            BiasReport(bias_type="test", score=1.5)


class TestBenchResult:
    def test_default(self):
        r = BenchResult(judge_config=JudgeConfig())
        assert r.overall_reliability == 0.0
        assert r.verdicts == []


class TestDataset:
    def test_empty(self):
        d = Dataset(name="test")
        assert d.pairs == []

    def test_with_pairs(self):
        d = Dataset(
            name="test",
            description="desc",
            pairs=[
                LabeledPair(
                    id="p1", prompt="Q", response_a="A", response_b="B", human_label="A"
                )
            ],
        )
        assert len(d.pairs) == 1
