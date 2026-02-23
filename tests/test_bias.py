"""Tests for bias detectors with deterministic inputs."""

import pytest

from judgebench.models import BiasReport, JudgeConfig, JudgeVerdict, LabeledPair
from judgebench.bias.position import detect_position_bias
from judgebench.bias.verbosity import detect_verbosity_bias
from judgebench.bias.leniency import detect_leniency_bias
from judgebench.bias.self_enhancement import detect_self_enhancement_bias


def _verdict(pair_id: str, label: str, position: str, confidence: float = 0.8) -> JudgeVerdict:
    return JudgeVerdict(
        pair_id=pair_id,
        judge_label=label,
        confidence=confidence,
        reasoning="test",
        position=position,
    )


def _pair(pid: str, label: str, len_a: int = 100, len_b: int = 100, **meta) -> LabeledPair:
    return LabeledPair(
        id=pid,
        prompt="Test prompt",
        response_a="x" * len_a,
        response_b="y" * len_b,
        human_label=label,
        metadata=meta,
    )


class TestPositionBias:
    def test_no_bias_consistent(self):
        """Judge always picks the same underlying response regardless of position."""
        verdicts = [
            # pair-1: picks A in original, picks B in swapped (= original A) -> consistent
            _verdict("p1", "A", "original"),
            _verdict("p1", "B", "swapped"),
            # pair-2: picks B in original, picks A in swapped (= original B) -> consistent
            _verdict("p2", "B", "original"),
            _verdict("p2", "A", "swapped"),
        ]
        report = detect_position_bias(verdicts)
        assert report.score == pytest.approx(0.0)
        assert not report.flagged

    def test_full_position_bias(self):
        """Judge always picks whichever is shown first (position 1)."""
        verdicts = [
            # pair-1: picks A (first) in original, picks A (first) in swapped -> inconsistent
            _verdict("p1", "A", "original"),
            _verdict("p1", "A", "swapped"),
            # pair-2: same pattern
            _verdict("p2", "A", "original"),
            _verdict("p2", "A", "swapped"),
        ]
        report = detect_position_bias(verdicts)
        assert report.score == pytest.approx(1.0)
        assert report.flagged

    def test_partial_bias(self):
        verdicts = [
            _verdict("p1", "A", "original"),
            _verdict("p1", "B", "swapped"),  # consistent
            _verdict("p2", "A", "original"),
            _verdict("p2", "A", "swapped"),  # inconsistent
        ]
        report = detect_position_bias(verdicts)
        assert report.score == pytest.approx(0.5)

    def test_no_pairs(self):
        report = detect_position_bias([])
        assert report.score == 0.0


class TestVerbosityBias:
    def test_no_bias(self):
        """Judge picks based on quality, not length."""
        pairs = [
            _pair("p1", "A", len_a=50, len_b=200),   # short A is better
            _pair("p2", "B", len_a=200, len_b=50),    # short B is better
            _pair("p3", "A", len_a=200, len_b=50),    # long A is better
            _pair("p4", "B", len_a=50, len_b=200),    # long B is better
        ]
        verdicts = [
            _verdict("p1", "A", "original"),  # picks short
            _verdict("p2", "B", "original"),  # picks short
            _verdict("p3", "A", "original"),  # picks long
            _verdict("p4", "B", "original"),  # picks long
        ]
        report = detect_verbosity_bias(verdicts, pairs)
        assert report.score < 0.3
        assert not report.flagged

    def test_always_picks_longer(self):
        """Judge always picks the longer response."""
        pairs = [
            _pair("p1", "A", len_a=200, len_b=50),
            _pair("p2", "A", len_a=200, len_b=50),
            _pair("p3", "B", len_a=50, len_b=200),
            _pair("p4", "B", len_a=50, len_b=200),
        ]
        verdicts = [
            _verdict("p1", "A", "original"),  # picks longer A
            _verdict("p2", "A", "original"),  # picks longer A
            _verdict("p3", "B", "original"),  # picks longer B
            _verdict("p4", "B", "original"),  # picks longer B
        ]
        report = detect_verbosity_bias(verdicts, pairs)
        assert report.score > 0.8
        assert report.flagged

    def test_insufficient_data(self):
        report = detect_verbosity_bias([], [])
        assert report.score == 0.0


class TestLeniencyBias:
    def test_matching_distributions(self):
        """Judge distribution matches human distribution."""
        pairs = [
            _pair("p1", "A"),
            _pair("p2", "B"),
            _pair("p3", "A"),
            _pair("p4", "tie"),
        ]
        verdicts = [
            _verdict("p1", "A", "original"),
            _verdict("p2", "B", "original"),
            _verdict("p3", "A", "original"),
            _verdict("p4", "tie", "original"),
        ]
        report = detect_leniency_bias(verdicts, pairs)
        assert report.score == pytest.approx(0.0)
        assert not report.flagged

    def test_excess_ties(self):
        """Judge gives many more ties than humans."""
        pairs = [
            _pair("p1", "A"),
            _pair("p2", "B"),
            _pair("p3", "A"),
            _pair("p4", "B"),
        ]
        verdicts = [
            _verdict("p1", "tie", "original"),
            _verdict("p2", "tie", "original"),
            _verdict("p3", "tie", "original"),
            _verdict("p4", "tie", "original"),
        ]
        report = detect_leniency_bias(verdicts, pairs)
        assert report.score > 0.5
        assert report.flagged

    def test_empty(self):
        report = detect_leniency_bias([], [])
        assert report.score == 0.0


class TestSelfEnhancementBias:
    def test_no_model_metadata(self):
        """Without model metadata, can't detect bias."""
        pairs = [_pair("p1", "A")]
        verdicts = [_verdict("p1", "A", "original")]
        config = JudgeConfig(model="claude-haiku-4-5-20251001")
        report = detect_self_enhancement_bias(verdicts, pairs, config)
        assert report.score == 0.0

    def test_self_preference(self):
        """Judge from Anthropic family prefers Anthropic responses over human baseline."""
        pairs = [
            _pair("p1", "B", model_a="gpt-4o", model_b="claude-3-opus"),
            _pair("p2", "B", model_a="gpt-4o", model_b="claude-3-sonnet"),
            _pair("p3", "A", model_a="gpt-4o", model_b="claude-3-haiku"),
            _pair("p4", "A", model_a="gpt-4o", model_b="claude-3-5-sonnet"),
        ]
        # Judge always picks the claude response (self-preference)
        verdicts = [
            _verdict("p1", "B", "original"),
            _verdict("p2", "B", "original"),
            _verdict("p3", "B", "original"),  # disagrees with human
            _verdict("p4", "B", "original"),  # disagrees with human
        ]
        config = JudgeConfig(model="claude-haiku-4-5-20251001")
        report = detect_self_enhancement_bias(verdicts, pairs, config)
        # Judge self-preference = 4/4 = 1.0, human = 2/4 = 0.5
        # excess = 0.5, score = min(1.0, 0.5 * 2) = 1.0
        assert report.score > 0.5
        assert report.flagged

    def test_no_self_preference(self):
        """Judge from Anthropic family doesn't favor its own outputs."""
        pairs = [
            _pair("p1", "A", model_a="gpt-4o", model_b="claude-3-opus"),
            _pair("p2", "A", model_a="gpt-4o", model_b="claude-3-sonnet"),
        ]
        # Judge picks GPT (not self) both times, matching human
        verdicts = [
            _verdict("p1", "A", "original"),
            _verdict("p2", "A", "original"),
        ]
        config = JudgeConfig(model="claude-haiku-4-5-20251001")
        report = detect_self_enhancement_bias(verdicts, pairs, config)
        assert report.score == pytest.approx(0.0)

    def test_unknown_judge_family(self):
        config = JudgeConfig(model="some-unknown-model-v1")
        report = detect_self_enhancement_bias([], [], config)
        assert report.score == 0.0
