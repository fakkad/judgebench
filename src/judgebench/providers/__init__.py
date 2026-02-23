"""LLM provider adapters."""

from judgebench.providers.base import BaseProvider
from judgebench.providers.anthropic import AnthropicProvider
from judgebench.providers.openai import OpenAIProvider

PROVIDERS: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider(name: str) -> type[BaseProvider]:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name]
