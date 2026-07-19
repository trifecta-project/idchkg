"""Offline guard for the Ollama client's error handling — no server needed."""

import pytest

from idchkg.llm import LLMError
from idchkg.llm.ollama_client import OllamaClient


def test_unreachable_or_timed_out_ollama_raises_llmerror():
    # Port 1 is closed -> connection refused / timeout. The client must surface
    # this as LLMError (which the pipeline catches per-chunk), not a bare
    # URLError/TimeoutError that would crash the whole run.
    client = OllamaClient(model="none", base_url="http://127.0.0.1:1", timeout=1)
    with pytest.raises(LLMError):
        client.complete_json("system", "user", {"type": "object"})
