"""Tests for statistical agreement metrics."""

import math

import pytest

from judgebench.stats.cohen_kappa import cohens_kappa
from judgebench.stats.krippendorff_alpha import krippendorff_alpha
from judgebench.stats.mcnemar import mcnemar_test
from judgebench.stats.agreement import raw_agreement, confusion_matrix, per_category_agreement


class TestCohensKappa:
    def test_perfect_agreement(self):
        labels = ["A", "B", "A", "B", "tie"]
        k = cohens_kappa(labels, labels)
        assert k == pytest.approx(1.0)

    def test_no_agreement_beyond_chance(self):
        """When two raters assign labels randomly but with similar marginals,
        kappa should be near zero."""
        a = ["A", "A", "B", "B", "A", "A", "B", "B"]
        b = ["A", "B", "A", "B", "A", "B", "A", "B"]
        k = cohens_kappa(a, b)
        assert -0.3 < k < 0.3

    def test_known_value(self):
        """Hand-calculated example.
        a = [A, A, B, B, A]
        b = [A, B, B, B, A]
        Observed agreement: 4/5 = 0.8
        P(A_a) = 3/5, P(B_a) = 2/5
        P(A_b) = 2/5, P(B_b) = 3/5
        P_e = (3/5)(2/5) + (2/5)(3/5) = 12/25 = 0.48
        kappa = (0.8 - 0.48) / (1 - 0.48) = 0.32/0.52 ~ 0.6154
        """
        a = ["A", "A", "B", "B", "A"]
        b = ["A", "B", "B", "B", "A"]
        k = cohens_kappa(a, b)
        assert k == pytest.approx(0.6154, abs=0.01)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa(["A"], ["A", "B"])

    def test_all_same_category(self):
        """When both raters always pick the same single category."""
        a = ["A", "A", "A"]
        b = ["A", "A", "A"]
        k = cohens_kappa(a, b)
        assert k == pytest.approx(1.0)


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        data = [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 0, 1],
        ]
        alpha = krippendorff_alpha(data, level="nominal")
        assert alpha == pytest.approx(1.0)

    def test_no_agreement(self):
        """Systematically opposite ratings should yield alpha < 0."""
        data = [
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
        ]
        alpha = krippendorff_alpha(data, level="nominal")
        assert alpha < 0

    def test_reference_value_nominal(self):
        """Test against a known Krippendorff example.
        From Krippendorff (2011), simplified.
        Observers rate 6 units with values {0, 1, 2}.
        """
        data = [
            [0, 1, 0, 0, 1, 2],
            [0, 1, 1, 0, 1, 2],
        ]
        alpha = krippendorff_alpha(data, level="nominal")
        # Should be moderate agreement
        assert 0.5 < alpha < 1.0

    def test_ordinal(self):
        data = [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 0, 1],
        ]
        alpha = krippendorff_alpha(data, level="ordinal")
        assert alpha == pytest.approx(1.0)

    def test_with_missing(self):
        data = [
            [0, 1, None, 0, 1],
            [0, 1, 2, None, 1],
        ]
        alpha = krippendorff_alpha(data, level="nominal")
        # Should still compute with available pairs
        assert -1.0 <= alpha <= 1.0

    def test_single_value_returns_one(self):
        data = [
            [0, 0, 0],
            [0, 0, 0],
        ]
        alpha = krippendorff_alpha(data, level="nominal")
        assert alpha == pytest.approx(1.0)

    def test_too_few_observers(self):
        with pytest.raises(ValueError):
            krippendorff_alpha([[0, 1, 2]])


class TestMcNemar:
    def test_symmetric_errors(self):
        """When errors are equally distributed, chi-squared should be small."""
        a =   ["A", "B", "A", "B", "B", "A"]
        b =   ["A", "B", "B", "A", "B", "A"]
        ref = ["A", "B", "A", "A", "B", "A"]
        result = mcnemar_test(a, b, ref)
        # b=1 (A correct, B wrong on idx 2), c=1 (A wrong, B correct on idx 3)
        assert result["b"] == 1
        assert result["c"] == 1
        # Symmetric with continuity correction: (|1-1|-1)^2/(1+1) = 1/2 = 0.5
        assert result["chi_squared"] == pytest.approx(0.5)
        # p-value should be large (not significant)
        assert result["p_value"] > 0.3

    def test_asymmetric_errors(self):
        """When errors are one-sided, chi-squared should be significant."""
        # A is wrong on indices 0-3 and 8-9 (6 items), B is always correct
        a =   ["A", "A", "A", "A", "B", "B", "B", "B", "A", "A"]
        b =   ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
        ref = ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
        result = mcnemar_test(a, b, ref)
        # b = 0 (A correct & B wrong: never, since B==ref always)
        # c = 6 (A wrong & B correct: indices 0,1,2,3,8,9)
        assert result["b"] == 0
        assert result["c"] == 6
        assert result["chi_squared"] > 0
        assert result["p_value"] < 0.05

    def test_perfect_agreement(self):
        labels = ["A", "B", "A"]
        result = mcnemar_test(labels, labels, labels)
        assert result["b"] == 0
        assert result["c"] == 0
        assert result["chi_squared"] == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([], [], [])


class TestRawAgreement:
    def test_perfect(self):
        assert raw_agreement(["A", "B"], ["A", "B"]) == 1.0

    def test_none(self):
        assert raw_agreement(["A", "A"], ["B", "B"]) == 0.0

    def test_partial(self):
        assert raw_agreement(["A", "B", "A"], ["A", "A", "A"]) == pytest.approx(2 / 3)

    def test_empty(self):
        assert raw_agreement([], []) == 0.0


class TestConfusionMatrix:
    def test_basic(self):
        result = confusion_matrix(["A", "B", "A"], ["A", "A", "B"], categories=["A", "B"])
        mat = result["matrix"]
        assert mat[0][0] == 1  # A->A
        assert mat[0][1] == 1  # A->B
        assert mat[1][0] == 1  # B->A
        assert mat[1][1] == 0  # B->B

    def test_with_tie(self):
        result = confusion_matrix(
            ["A", "tie"], ["tie", "tie"], categories=["A", "B", "tie"]
        )
        mat = result["matrix"]
        assert mat[2][2] == 1  # tie->tie
        assert mat[0][2] == 1  # A->tie


class TestPerCategoryAgreement:
    def test_basic(self):
        result = per_category_agreement(
            ["A", "A", "B", "B"],
            ["A", "B", "B", "A"],
        )
        assert result["A"] == pytest.approx(0.5)
        assert result["B"] == pytest.approx(0.5)
