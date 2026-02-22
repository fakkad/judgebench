"""Tests for data models."""

import json

import pytest

from judgebench.models import (
    AgreementMetrics,
    BiasReport,
    JudgeVerdict,
    LabeledPair,
)


class TestLabeledPair:
    def test_create_minimal(self):
        pair = LabeledPair(
            id="test-001",
            prompt="What is 2+2?",
            response_a="4",
            response_b="5",
            human_label="a",
            category="factual",
        )
        assert pair.id == "test-001"
        assert pair.human_label == "a"
        assert pair.metadata == {}

    def test_create_with_metadata(self):
        pair = LabeledPair(
            id="test-002",
            prompt="Write a poem",
            response_a="Roses are red",
            response_b="The sun sets low",
            human_label="b",
            category="creative",
            metadata={"source_model_a": "gpt-4", "source_model_b": "claude-3"},
        )
        assert pair.metadata["source_model_a"] == "gpt-4"

    def test_serialization_roundtrip(self):
        pair = LabeledPair(
            id="test-003",
            prompt="Test prompt",
            response_a="A",
            response_b="B",
            human_label="tie",
            category="reasoning",
        )
        data = json.loads(pair.model_dump_json())
        restored = LabeledPair(**data)
        assert restored == pair

    def test_all_categories(self):
        for cat in ["factual", "creative", "reasoning", "safety", "coding"]:
            pair = LabeledPair(
                id=f"cat-{cat}",
                prompt="p",
                response_a="a",
                response_b="b",
                human_label="a",
                category=cat,
            )
            assert pair.category == cat

    def test_all_labels(self):
        for label in ["a", "b", "tie"]:
            pair = LabeledPair(
                id=f"lbl-{label}",
                prompt="p",
                response_a="a",
                response_b="b",
                human_label=label,
                category="factual",
            )
            assert pair.human_label == label


class TestJudgeVerdict:
    def test_consistent_verdict(self):
        v = JudgeVerdict(
            pair_id="v-001",
            forward_choice="a",
            reversed_choice="a",
            forward_reasoning="A is better",
            reversed_reasoning="A is still better",
            consistent=True,
        )
        assert v.consistent is True

    def test_inconsistent_verdict(self):
        v = JudgeVerdict(
            pair_id="v-002",
            forward_choice="a",
            reversed_choice="b",
            forward_reasoning="A is better",
            reversed_reasoning="B is better now",
            consistent=False,
        )
        assert v.consistent is False

    def test_serialization_roundtrip(self):
        v = JudgeVerdict(
            pair_id="v-003",
            forward_choice="b",
            reversed_choice="b",
            forward_reasoning="B wins",
            reversed_reasoning="B wins again",
            consistent=True,
        )
        data = json.loads(v.model_dump_json())
        restored = JudgeVerdict(**data)
        assert restored == v


class TestBiasReport:
    def test_create(self):
        report = BiasReport(
            position_bias_rate=0.15,
            verbosity_bias_rho=0.3,
            self_enhance_delta=0.05,
            leniency_score=-0.1,
        )
        assert report.position_bias_rate == 0.15
        assert report.verbosity_bias_rho == 0.3

    def test_zero_bias(self):
        report = BiasReport(
            position_bias_rate=0.0,
            verbosity_bias_rho=0.0,
            self_enhance_delta=0.0,
            leniency_score=0.0,
        )
        assert report.position_bias_rate == 0.0

    def test_serialization_roundtrip(self):
        report = BiasReport(
            position_bias_rate=0.2,
            verbosity_bias_rho=-0.5,
            self_enhance_delta=0.1,
            leniency_score=0.3,
        )
        data = json.loads(report.model_dump_json())
        restored = BiasReport(**data)
        assert restored == report


class TestAgreementMetrics:
    def test_create(self):
        metrics = AgreementMetrics(
            cohens_kappa=0.8,
            krippendorffs_alpha=0.75,
            spearman_rho=0.85,
            spearman_p=0.001,
            mcnemars_chi2=2.5,
            mcnemars_p=0.11,
        )
        assert metrics.cohens_kappa == 0.8
        assert metrics.mcnemars_p == 0.11

    def test_serialization_roundtrip(self):
        metrics = AgreementMetrics(
            cohens_kappa=0.6,
            krippendorffs_alpha=0.55,
            spearman_rho=0.7,
            spearman_p=0.01,
            mcnemars_chi2=3.0,
            mcnemars_p=0.08,
        )
        data = json.loads(metrics.model_dump_json())
        restored = AgreementMetrics(**data)
        assert restored == metrics
