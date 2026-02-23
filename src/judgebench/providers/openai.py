"""OpenAI provider adapter."""

from __future__ import annotations

import json
import os

import httpx

from judgebench.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Call OpenAI Chat Completions API."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    async def judge(self, prompt: str) -> dict:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.params.get("max_tokens", 1024),
            "temperature": self.params.get("temperature", 0.0),
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.API_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        return json.loads(text)
