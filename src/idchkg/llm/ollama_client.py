"""Fully local backend: a model served by Ollama on localhost.

Uses Ollama's native /api/chat endpoint with the `format` parameter set to a
JSON schema, which constrains decoding to schema-conformant JSON. No extra
dependencies - stdlib urllib only.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from . import LLMError

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        model: str = "granite4.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # A read timeout surfaces as a bare TimeoutError (not URLError); wrap
            # everything network-ish as LLMError so the pipeline records it as a
            # per-chunk error instead of crashing the whole run.
            raise LLMError(
                f"Ollama request to {self.base_url} failed ({exc}); is `ollama serve` "
                "running and not overloaded? A high --concurrency can cause timeouts."
            ) from exc
        if logger.isEnabledFor(logging.DEBUG):
            wall = time.perf_counter() - started
            eval_count = body.get("eval_count") or 0
            eval_s = (body.get("eval_duration") or 0) / 1e9  # ns -> s
            tok_s = eval_count / eval_s if eval_s else 0.0
            logger.debug(
                "ollama %s: %.1fs wall, %d tok @ %.0f tok/s, load %.0fms, prompt %d tok",
                self.model, wall, eval_count, tok_s,
                (body.get("load_duration") or 0) / 1e6, body.get("prompt_eval_count") or 0,
            )
        content = body.get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama returned invalid JSON: {content[:200]!r}") from exc
