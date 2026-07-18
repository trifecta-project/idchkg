"""Fully local backend: a model served by Ollama on localhost.

Uses Ollama's native /api/chat endpoint with the `format` parameter set to a
JSON schema, which constrains decoding to schema-conformant JSON. No extra
dependencies - stdlib urllib only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import LLMError


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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.base_url} - is `ollama serve` running? ({exc})"
            ) from exc
        content = body.get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama returned invalid JSON: {content[:200]!r}") from exc
