"""Tests for synthetic data generator with mocked LLM."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from judgebench.models import Dataset, LabeledPair
from judgebench.synthetic import generate_synthetic


class TestGenerateSynthetic:
    @pytest.fixture
    def seed_dataset(self):
        return Dataset(
            name="seed",
            pairs=[
                LabeledPair(
                    id="s1",
                    prompt="Original question",
                    response_a="Original A",
                    response_b="Original B",
                    human_label="A",
                    metadata={"category": "test"},
                ),
                LabeledPair(
                    id="s2",
                    prompt="Another question",
                    response_a="Another A",
                    response_b="Another B",
                    human_label="B",
                    metadata={"category": "other"},
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_generates_requested_count(self, seed_dataset):
        mock_provider = AsyncMock()
        mock_provider.judge.return_value = {
            "prompt": "Paraphrased question",
            "response_a": "Paraphrased A",
            "response_b": "Paraphrased B",
            "human_label": "A",
            "category": "test",
        }

        with patch("judgebench.synthetic.get_provider") as mock_get:
            mock_get.return_value = lambda **kwargs: mock_provider

            result = await generate_synthetic(seed_dataset, count=5, concurrency=2)

        assert len(result.pairs) == 5
        assert result.name == "seed-synthetic"

    @pytest.mark.asyncio
    async def test_synthetic_pairs_have_metadata(self, seed_dataset):
        mock_provider = AsyncMock()
        mock_provider.judge.return_value = {
            "prompt": "New Q",
            "response_a": "New A",
            "response_b": "New B",
            "human_label": "A",
            "category": "test",
        }

        with patch("judgebench.synthetic.get_provider") as mock_get:
            mock_get.return_value = lambda **kwargs: mock_provider

            result = await generate_synthetic(seed_dataset, count=2, concurrency=1)

        for pair in result.pairs:
            assert pair.metadata.get("source") == "synthetic"
            assert "seed_id" in pair.metadata

    @pytest.mark.asyncio
    async def test_handles_failures_gracefully(self, seed_dataset):
        mock_provider = AsyncMock()
        call_count = 0

        async def flaky_judge(prompt):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("API error")
            return {
                "prompt": "Q",
                "response_a": "A",
                "response_b": "B",
                "human_label": "A",
                "category": "test",
            }

        mock_provider.judge = flaky_judge

        with patch("judgebench.synthetic.get_provider") as mock_get:
            mock_get.return_value = lambda **kwargs: mock_provider

            result = await generate_synthetic(seed_dataset, count=4, concurrency=1)

        # Some should succeed, some fail
        assert len(result.pairs) > 0
        assert len(result.pairs) < 4

    @pytest.mark.asyncio
    async def test_empty_seed_raises(self):
        empty = Dataset(name="empty", pairs=[])
        with pytest.raises(ValueError, match="no pairs"):
            await generate_synthetic(empty, count=5)
