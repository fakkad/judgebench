"""Tests for bias detectors using synthetic data (no LLM calls)."""

import pytest

from judgebench.bias.leniency import detect as detect_leniency
from judgebench.bias.position import detect as detect_position
from judgebench.bias.self_enhance import detect as detect_self_enhance
from judgebench.bias.verbosity import detect as detect_verbosity
from judgebench.models import JudgeVerdict, LabeledPair


def _make_pair(
    pair_id: str,
    human_label: str = "a",
    response_a: str = "Short A",
    response_b: str = "Short B",
    category: str = "factual",
    metadata: dict | None = None,
) -> LabeledPair:
    return LabeledPair(
        id=pair_id,
        prompt="Test prompt",
        response_a=response_a,
        response_b=response_b,
        human_label=human_label,
        category=category,
        metadata=metadata or {},
    )


def _make_verdict(
    pair_id: str,
    forward: str = "a",
    reversed_: str = "a",
    consistent: bool | None = None,
) -> JudgeVerdict:
    if consistent is None:
        consistent = forward == reversed_
    return JudgeVerdict(
        pair_id=pair_id,
        forward_choice=forward,
        reversed_choice=reversed_,
        forward_reasoning="Forward reasoning",
        reversed_reasoning="Reversed reasoning",
        consistent=consistent,
    )


class TestPositionBias:
    def test_no_bias(self):
        """All verdicts consistent -> 0 position bias."""
        verdicts = [
            _make_verdict("p1", "a", "a", True),
            _make_verdict("p2", "b", "b", True),
            _make_verdict("p3", "a", "a", True),
        ]
        assert detect_position(verdicts) == 0.0

    def test_full_bias(self):
        """All verdicts inconsistent -> 1.0 position bias."""
        verdicts = [
            _make_verdict("p1", "a", "b", False),
            _make_verdict("p2", "b", "a", False),
        ]
        assert detect_position(verdicts) == 1.0

    def test_partial_bias(self):
        """Mix of consistent and inconsistent."""
        verdicts = [
            _make_verdict("p1", "a", "a", True),
            _make_verdict("p2", "a", "b", False),
            _make_verdict("p3", "b", "b", True),
            _make_verdict("p4", "b", "a", False),
        ]
        assert detect_position(verdicts) == 0.5

    def test_excludes_ties(self):
        """Ties in human labels should be excluded."""
        pairs = [
            _make_pair("p1", human_label="a"),
            _make_pair("p2", human_label="tie"),
            _make_pair("p3", human_label="b"),
        ]
        verdicts = [
            _make_verdict("p1", "a", "a", True),
            _make_verdict("p2", "a", "b", False),  # tie, should be excluded
            _make_verdict("p3", "b", "a", False),
        ]
        # Only p1 (consistent) and p3 (inconsistent) count -> 0.5
        result = detect_position(verdicts, pairs)
        assert result == 0.5

    def test_empty_verdicts(self):
        assert detect_position([]) == 0.0


class TestVerbosityBias:
    def test_prefers_longer(self):
        """Judge always picks the longer response -> positive rho."""
        pairs = [
            _make_pair("p1", response_a="short", response_b="a much longer response here"),
            _make_pair("p2", response_a="tiny", response_b="this is a significantly longer response"),
            _make_pair("p3", response_a="sm", response_b="a very lengthy and detailed response text"),
        ]
        verdicts = [
            _make_verdict("p1", "b", "b"),  # picks longer
            _make_verdict("p2", "b", "b"),  # picks longer
            _make_verdict("p3", "b", "b"),  # picks longer
        ]
        rho = detect_verbosity(pairs, verdicts)
        assert rho > 0.0, f"Expected positive rho, got {rho}"

    def test_prefers_shorter(self):
        """Judge always picks the shorter response -> negative rho."""
        pairs = [
            _make_pair("p1", response_a="short", response_b="a much longer response here"),
            _make_pair("p2", response_a="tiny", response_b="this is a significantly longer response"),
            _make_pair("p3", response_a="sm", response_b="a very lengthy and detailed response text"),
        ]
        verdicts = [
            _make_verdict("p1", "a", "a"),  # picks shorter
            _make_verdict("p2", "a", "a"),  # picks shorter
            _make_verdict("p3", "a", "a"),  # picks shorter
        ]
        rho = detect_verbosity(pairs, verdicts)
        assert rho < 0.0, f"Expected negative rho, got {rho}"

    def test_no_length_correlation(self):
        """Equal length responses -> rho near 0."""
        pairs = [
            _make_pair("p1", response_a="abcde", response_b="fghij"),
            _make_pair("p2", response_a="klmno", response_b="pqrst"),
            _make_pair("p3", response_a="uvwxy", response_b="zabcd"),
        ]
        verdicts = [
            _make_verdict("p1", "a", "a"),
            _make_verdict("p2", "b", "b"),
            _make_verdict("p3", "a", "a"),
        ]
        rho = detect_verbosity(pairs, verdicts)
        assert abs(rho) < 0.5, f"Expected near-zero rho, got {rho}"

    def test_empty_data(self):
        assert detect_verbosity([], []) == 0.0


class TestSelfEnhanceBias:
    def test_self_enhancement_detected(self):
        """Judge always picks its own model's output."""
        judge_model = "claude-test"
        pairs = [
            _make_pair(
                "p1",
                metadata={"source_model_a": "claude-test", "source_model_b": "gpt-4"},
            ),
            _make_pair(
                "p2",
                metadata={"source_model_a": "gpt-4", "source_model_b": "claude-test"},
            ),
        ]
        verdicts = [
            _make_verdict("p1", "a", "a"),  # picks claude-test (a)
            _make_verdict("p2", "b", "b"),  # picks claude-test (b)
        ]
        delta = detect_self_enhance(pairs, verdicts, judge_model)
        assert delta > 0.0, f"Expected positive delta, got {delta}"

    def test_no_self_enhancement(self):
        """Judge never picks its own output."""
        judge_model = "claude-test"
        pairs = [
            _make_pair(
                "p1",
                metadata={"source_model_a": "claude-test", "source_model_b": "gpt-4"},
            ),
            _make_pair(
                "p2",
                metadata={"source_model_a": "gpt-4", "source_model_b": "claude-test"},
            ),
        ]
        verdicts = [
            _make_verdict("p1", "b", "b"),  # picks gpt-4 (b)
            _make_verdict("p2", "a", "a"),  # picks gpt-4 (a)
        ]
        delta = detect_self_enhance(pairs, verdicts, judge_model)
        assert delta < 0.0, f"Expected negative delta, got {delta}"

    def test_no_self_generated_responses(self):
        """No source_model metadata -> returns 0.0."""
        pairs = [
            _make_pair("p1"),
            _make_pair("p2"),
        ]
        verdicts = [
            _make_verdict("p1", "a", "a"),
            _make_verdict("p2", "b", "b"),
        ]
        delta = detect_self_enhance(pairs, verdicts, "claude-test")
        assert delta == 0.0


class TestLeniencyBias:
    def test_lenient_judge(self):
        """Judge agrees with human on 'a' more than 'b' -> positive leniency."""
        pairs = [
            _make_pair("p1", human_label="a"),
            _make_pair("p2", human_label="a"),
            _make_pair("p3", human_label="b"),
            _make_pair("p4", human_label="b"),
        ]
        verdicts = [
            _make_verdict("p1", "a", "a"),  # TP
            _make_verdict("p2", "a", "a"),  # TP
            _make_verdict("p3", "a", "a"),  # FP
            _make_verdict("p4", "b", "b"),  # TN
        ]
        # TPR = 2/2 = 1.0, TNR = 1/2 = 0.5, leniency = 0.5
        leniency = detect_leniency(pairs, verdicts)
        assert leniency == pytest.approx(0.5)

    def test_strict_judge(self):
        """Judge agrees with human on 'b' more than 'a' -> negative leniency."""
        pairs = [
            _make_pair("p1", human_label="a"),
            _make_pair("p2", human_label="a"),
            _make_pair("p3", human_label="b"),
            _make_pair("p4", human_label="b"),
        ]
        verdicts = [
            _make_verdict("p1", "a", "a"),  # TP
            _make_verdict("p2", "b", "b"),  # FN
            _make_verdict("p3", "b", "b"),  # TN
            _make_verdict("p4", "b", "b"),  # TN
        ]
        # TPR = 1/2 = 0.5, TNR = 2/2 = 1.0, leniency = -0.5
        leniency = detect_leniency(pairs, verdicts)
        assert leniency == pytest.approx(-0.5)

    def test_balanced_judge(self):
        """Judge agrees equally -> leniency near 0."""
        pairs = [
            _make_pair("p1", human_label="a"),
            _make_pair("p2", human_label="b"),
        ]
        verdicts = [
            _make_verdict("p1", "a", "a"),  # TP
            _make_verdict("p2", "b", "b"),  # TN
        ]
        # TPR = 1.0, TNR = 1.0, leniency = 0.0
        leniency = detect_leniency(pairs, verdicts)
        assert leniency == pytest.approx(0.0)

    def test_excludes_ties(self):
        """Pairs with human_label='tie' should be skipped."""
        pairs = [
            _make_pair("p1", human_label="a"),
            _make_pair("p2", human_label="tie"),
            _make_pair("p3", human_label="b"),
        ]
        verdicts = [
            _make_verdict("p1", "a", "a"),  # TP
            _make_verdict("p2", "a", "a"),  # skipped (tie)
            _make_verdict("p3", "b", "b"),  # TN
        ]
        leniency = detect_leniency(pairs, verdicts)
        assert leniency == pytest.approx(0.0)

    def test_empty_data(self):
        assert detect_leniency([], []) == 0.0
