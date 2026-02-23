"""Base provider interface."""

from __future__ import annotations

import abc


class BaseProvider(abc.ABC):
    """Abstract base for LLM API providers."""

    def __init__(self, model: str, params: dict | None = None, system_prompt: str | None = None):
        self.model = model
        self.params = params or {}
        self.system_prompt = system_prompt

    @abc.abstractmethod
    async def judge(self, prompt: str) -> dict:
        """Send a judge prompt and return parsed JSON with winner/confidence/reasoning.

        Returns:
            dict with keys: winner (str), confidence (float), reasoning (str)
        """
        ...
