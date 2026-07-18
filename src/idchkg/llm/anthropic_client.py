"""Claude backend via the official Anthropic SDK, using structured outputs
so responses are guaranteed to match the requested JSON schema."""

from __future__ import annotations

import json
import logging
import time

from . import LLMError

logger = logging.getLogger(__name__)


class AnthropicClient:
    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 16000):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "The anthropic SDK is not installed. Run: pip install 'idchkg[anthropic]'"
            ) from exc
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        started = time.perf_counter()
        # No extended thinking: extraction and 0-1 judging are mechanical, the
        # output is already constrained by output_config.format, and adaptive
        # thinking would consume the max_tokens budget (truncating the JSON) and
        # add latency for no benefit here.
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if logger.isEnabledFor(logging.DEBUG):
            usage = getattr(response, "usage", None)
            logger.debug(
                "anthropic %s: %.1fs, in %s tok, out %s tok",
                self.model, time.perf_counter() - started,
                getattr(usage, "input_tokens", "?"), getattr(usage, "output_tokens", "?"),
            )
        if response.stop_reason == "refusal":
            raise LLMError("The model refused this request (stop_reason=refusal).")
        if response.stop_reason == "max_tokens":
            raise LLMError(
                "Response truncated at max_tokens; the structured output may be incomplete."
            )
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("No text block in model response.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned invalid JSON: {exc}") from exc
