"""Pluggable LLM backends.

Every backend implements one method: ``complete_json(system, user, schema)``,
returning a dict that conforms to the given JSON schema. This keeps the
extraction and judge stages backend-agnostic:

* ``anthropic`` - Claude via the official SDK (structured outputs)
* ``ollama``    - a fully local model served by Ollama on localhost
* ``mock``      - deterministic, scriptable backend for tests and smoke runs
"""

from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    """Raised when a backend cannot produce a usable response."""


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


def get_client(backend: str, model: str | None = None, **kwargs) -> "LLMClient":
    if backend == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model=model, **kwargs) if model else AnthropicClient(**kwargs)
    if backend == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(model=model, **kwargs) if model else OllamaClient(**kwargs)
    if backend == "mock":
        from .mock import MockLLM

        return MockLLM(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend!r} (expected anthropic, ollama, or mock)")
