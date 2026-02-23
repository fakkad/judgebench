"""Tests for judge runner with mocked LLM responses."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from judgebench.models import Dataset, JudgeConfig, LabeledPair
from judgebench.judge_runner import run_judge, compute_results_from_verdicts, _build_prompt


class TestBuildPrompt:
    def test_contains_all_parts(self):
        prompt = _build_prompt("My question", "Response A text", "Response B text")
        assert "My question" in prompt
        assert "Response A text" in prompt
        assert "Response B text" in prompt
        assert "Response A:" in prompt
        assert "Response B:" in prompt
        assert "winner" in prompt


class TestRunJudge:
    @pytest.fixture
    def dataset(self):
        return Dataset(
            name="test",
            pairs=[
                LabeledPair(
                    id="p1",
                    prompt="Test question",
                    response_a="Answer A",
                    response_b="Answer B",
                    human_label="A",
                ),
                LabeledPair(
                    id="p2",
                    prompt="Another question",
                    response_a="Resp A",
                    response_b="Resp B",
                    human_label="B",
                ),
            ],
        )

    @pytest.fixture
    def judge_config(self):
        return JudgeConfig(provider="anthropic", model="test-model")

    @pytest.mark.asyncio
    async def test_run_with_mocked_provider(self, dataset, judge_config):
        mock_provider = AsyncMock()
        # Always returns A as winner
        mock_provider.judge.return_value = {
            "winner": "A",
            "confidence": 0.9,
            "reasoning": "A is better",
        }

        with patch("judgebench.judge_runner.get_provider") as mock_get:
            mock_get.return_value = lambda **kwargs: mock_provider

            result = await run_judge(dataset, judge_config, concurrency=2)

        # 2 pairs * 2 orderings = 4 verdicts
        assert len(result.verdicts) == 4
        assert result.overall_reliability >= 0.0
        assert len(result.bias_reports) == 4

    @pytest.mark.asyncio
    async def test_agreement_metrics_populated(self, dataset, judge_config):
        mock_provider = AsyncMock()
        # Match human labels: p1=A, p2=B
        call_count = 0

        async def mock_judge(prompt):
            nonlocal call_count
            call_count += 1
            # Original verdicts: odd calls are original, even are swapped
            if "Test question" in prompt:
                return {"winner": "A", "confidence": 0.9, "reasoning": "good"}
            else:
                return {"winner": "B", "confidence": 0.85, "reasoning": "good"}

        mock_provider.judge = mock_judge

        with patch("judgebench.judge_runner.get_provider") as mock_get:
            mock_get.return_value = lambda **kwargs: mock_provider

            result = await run_judge(dataset, judge_config)

        assert "raw_agreement" in result.agreement_metrics
        assert "cohens_kappa" in result.agreement_metrics
        assert "krippendorff_alpha_nominal" in result.agreement_metrics


class TestComputeResultsFromVerdicts:
    def test_basic_computation(self):
        from judgebench.models import JudgeVerdict

        dataset = Dataset(
            name="test",
            pairs=[
                LabeledPair(id="p1", prompt="Q", response_a="A", response_b="B", human_label="A"),
            ],
        )
        verdicts = [
            JudgeVerdict(pair_id="p1", judge_label="A", confidence=0.9, position="original"),
            JudgeVerdict(pair_id="p1", judge_label="B", confidence=0.9, position="swapped"),
        ]
        config = JudgeConfig()

        result = compute_results_from_verdicts(verdicts, dataset, config)
        assert result.agreement_metrics["raw_agreement"] == 1.0
        assert len(result.bias_reports) == 4
