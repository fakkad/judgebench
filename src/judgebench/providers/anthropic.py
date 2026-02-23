"""Anthropic provider adapter."""

from __future__ import annotations

import json
import os

import httpx

from judgebench.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Call Anthropic Messages API."""

    API_URL = "https://api.anthropic.com/v1/messages"

    async def judge(self, prompt: str) -> dict:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

        messages = [{"role": "user", "content": prompt}]
        body: dict = {
            "model": self.model,
            "max_tokens": self.params.get("max_tokens", 1024),
            "messages": messages,
        }
        if self.system_prompt:
            body["system"] = self.system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.API_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data["content"][0]["text"]
        # Extract JSON from response (may be wrapped in markdown fences)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        return json.loads(text)
