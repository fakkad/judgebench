"""Tests for agreement metrics with known-answer test data."""

import pytest

from judgebench.agreement import compute_agreement
from judgebench.models import JudgeVerdict, LabeledPair


def _make_pair(pair_id: str, human_label: str) -> LabeledPair:
    return LabeledPair(
        id=pair_id,
        prompt="Test prompt",
        response_a="Response A",
        response_b="Response B",
        human_label=human_label,
        category="factual",
    )


def _make_verdict(pair_id: str, forward: str) -> JudgeVerdict:
    return JudgeVerdict(
        pair_id=pair_id,
        forward_choice=forward,
        reversed_choice=forward,
        forward_reasoning="Reasoning",
        reversed_reasoning="Reasoning",
        consistent=True,
    )


class TestCohenKappa:
    def test_perfect_agreement(self):
        """Judge matches human perfectly -> kappa = 1.0."""
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "b"),
            _make_pair("p3", "a"),
            _make_pair("p4", "b"),
        ]
        verdicts = [
            _make_verdict("p1", "a"),
            _make_verdict("p2", "b"),
            _make_verdict("p3", "a"),
            _make_verdict("p4", "b"),
        ]
        metrics = compute_agreement(pairs, verdicts)
        assert metrics.cohens_kappa == pytest.approx(1.0)

    def test_no_agreement(self):
        """Judge always disagrees -> kappa < 0."""
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "b"),
            _make_pair("p3", "a"),
            _make_pair("p4", "b"),
        ]
        verdicts = [
            _make_verdict("p1", "b"),
            _make_verdict("p2", "a"),
            _make_verdict("p3", "b"),
            _make_verdict("p4", "a"),
        ]
        metrics = compute_agreement(pairs, verdicts)
        assert metrics.cohens_kappa < 0.0

    def test_partial_agreement(self):
        """Mix of agreement and disagreement."""
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "b"),
            _make_pair("p3", "a"),
            _make_pair("p4", "b"),
        ]
        verdicts = [
            _make_verdict("p1", "a"),  # agree
            _make_verdict("p2", "b"),  # agree
            _make_verdict("p3", "b"),  # disagree
            _make_verdict("p4", "a"),  # disagree
        ]
        metrics = compute_agreement(pairs, verdicts)
        assert -1.0 <= metrics.cohens_kappa <= 1.0
        assert metrics.cohens_kappa == pytest.approx(0.0, abs=0.01)


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "b"),
            _make_pair("p3", "a"),
            _make_pair("p4", "b"),
        ]
        verdicts = [
            _make_verdict("p1", "a"),
            _make_verdict("p2", "b"),
            _make_verdict("p3", "a"),
            _make_verdict("p4", "b"),
        ]
        metrics = compute_agreement(pairs, verdicts)
        assert metrics.krippendorffs_alpha == pytest.approx(1.0)

    def test_no_agreement(self):
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "b"),
            _make_pair("p3", "a"),
            _make_pair("p4", "b"),
        ]
        verdicts = [
            _make_verdict("p1", "b"),
            _make_verdict("p2", "a"),
            _make_verdict("p3", "b"),
            _make_verdict("p4", "a"),
        ]
        metrics = compute_agreement(pairs, verdicts)
        assert metrics.krippendorffs_alpha < 0.0


class TestSpearman:
    def test_perfect_correlation(self):
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "b"),
            _make_pair("p3", "a"),
            _make_pair("p4", "b"),
        ]
        verdicts = [
            _make_verdict("p1", "a"),
            _make_verdict("p2", "b"),
            _make_verdict("p3", "a"),
            _make_verdict("p4", "b"),
        ]
        metrics = compute_agreement(pairs, verdicts)
        assert metrics.spearman_rho == pytest.approx(1.0)
        assert metrics.spearman_p < 0.05

    def test_inverse_correlation(self):
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "b"),
            _make_pair("p3", "a"),
            _make_pair("p4", "b"),
        ]
        verdicts = [
            _make_verdict("p1", "b"),
            _make_verdict("p2", "a"),
            _make_verdict("p3", "b"),
            _make_verdict("p4", "a"),
        ]
        metrics = compute_agreement(pairs, verdicts)
        assert metrics.spearman_rho == pytest.approx(-1.0)


class TestMcNemar:
    def test_symmetric_disagreement(self):
        """Equal disagreement in both directions -> non-significant."""
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "b"),
            _make_pair("p3", "a"),
            _make_pair("p4", "b"),
        ]
        verdicts = [
            _make_verdict("p1", "a"),  # agree
            _make_verdict("p2", "a"),  # disagree: human=b, judge=a
            _make_verdict("p3", "b"),  # disagree: human=a, judge=b
            _make_verdict("p4", "b"),  # agree
        ]
        metrics = compute_agreement(pairs, verdicts)
        # b=1, c=1 -> chi2 = (|1-1|-1)^2 / 2 = 0 (with continuity correction)
        assert metrics.mcnemars_chi2 == pytest.approx(0.0, abs=0.5)
        assert metrics.mcnemars_p > 0.05

    def test_asymmetric_disagreement(self):
        """All disagreement in one direction."""
        pairs = [
            _make_pair(f"p{i}", "a") for i in range(10)
        ] + [
            _make_pair(f"p{i+10}", "b") for i in range(10)
        ]
        verdicts = [
            _make_verdict(f"p{i}", "a") for i in range(10)  # all agree on "a"
        ] + [
            _make_verdict(f"p{i+10}", "a") for i in range(10)  # all disagree on "b"
        ]
        metrics = compute_agreement(pairs, verdicts)
        # b=0, c=10 -> large chi2
        assert metrics.mcnemars_chi2 > 0
        assert metrics.mcnemars_p < 0.05


class TestEdgeCases:
    def test_ties_excluded(self):
        """Pairs with human_label='tie' should be excluded."""
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "tie"),
            _make_pair("p3", "b"),
        ]
        verdicts = [
            _make_verdict("p1", "a"),
            _make_verdict("p2", "a"),  # tie, should be excluded
            _make_verdict("p3", "b"),
        ]
        metrics = compute_agreement(pairs, verdicts)
        assert metrics.cohens_kappa == pytest.approx(1.0)

    def test_insufficient_data(self):
        """Less than 2 pairs -> default metrics."""
        pairs = [_make_pair("p1", "a")]
        verdicts = [_make_verdict("p1", "a")]
        metrics = compute_agreement(pairs, verdicts)
        assert metrics.cohens_kappa == 0.0
        assert metrics.spearman_p == 1.0

    def test_missing_verdict(self):
        """Pair without corresponding verdict is skipped."""
        pairs = [
            _make_pair("p1", "a"),
            _make_pair("p2", "b"),
            _make_pair("p3", "a"),
        ]
        verdicts = [
            _make_verdict("p1", "a"),
            # p2 missing
            _make_verdict("p3", "a"),
        ]
        metrics = compute_agreement(pairs, verdicts)
        # Only p1 and p3 counted, both agree
        assert metrics.cohens_kappa is not None
