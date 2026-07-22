"""Claude backend via the official Anthropic SDK, using structured outputs
so responses are guaranteed to match the requested JSON schema."""

from __future__ import annotations

import json
import logging
import random
import time

from . import LLMError

logger = logging.getLogger(__name__)


class AnthropicClient:
    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 16000, max_retries: int = 6):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "The anthropic SDK is not installed. Run: pip install 'idchkg[anthropic]'"
            ) from exc
        # A high --concurrency fans out many requests at once; the SDK retries
        # 429/5xx with exponential backoff, but its default ceiling (2) is too low
        # for bursty extraction/judging, so a transient rate-limit spike would
        # surface as per-chunk errors. Raise it so bursts are absorbed instead.
        self._client = anthropic.Anthropic(max_retries=max_retries)
        self._anthropic = anthropic  # for exception classes in complete_json
        self.model = model
        self.max_tokens = max_tokens
        # "Grammar compilation timed out" is a transient 400 the SDK won't retry
        # (structured-output schema compilation, worst on the cold-cache burst at
        # high concurrency). Retry it a few times before giving up on the chunk.
        self._grammar_retries = 4
        self.name = f"anthropic:{model}"

    def _create(self, system: str, user: str, schema: dict):
        """Call messages.create, converting API failures to LLMError so one bad
        chunk is recorded and skipped instead of crashing the whole run. The
        transient "grammar compilation timed out" 400 is retried (the SDK won't
        retry 400s, but that error clears once the schema caches server-side)."""
        for attempt in range(self._grammar_retries + 1):
            try:
                return self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": user}],
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                )
            except self._anthropic.APIStatusError as exc:
                message = str(getattr(exc, "message", "") or exc)
                transient = exc.status_code == 400 and "grammar compilation" in message.lower()
                if transient and attempt < self._grammar_retries:
                    # Jittered backoff so 24 workers don't retry in lockstep —
                    # desync lets the first compile cache the schema server-side.
                    time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1.5))
                    continue
                raise LLMError(f"Anthropic request failed ({exc.status_code}): {message}") from exc
            except self._anthropic.APIError as exc:  # connection/timeout after SDK retries
                raise LLMError(f"Anthropic request failed: {exc}") from exc

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        started = time.perf_counter()
        # No extended thinking: extraction and 0-1 judging are mechanical, the
        # output is already constrained by output_config.format, and adaptive
        # thinking would consume the max_tokens budget (truncating the JSON) and
        # add latency for no benefit here.
        # The extraction/judge system prompts are static and re-sent on every
        # call (one per chunk, two per claim), so cache the system prefix:
        # cache reads cost ~0.1x input and cut latency. No-op below the model's
        # min cacheable prefix (e.g. 4096 tok on Haiku) — check
        # usage.cache_read_input_tokens under -v to confirm it's taking effect.
        response = self._create(system, user, schema)
        if logger.isEnabledFor(logging.DEBUG):
            usage = getattr(response, "usage", None)
            logger.debug(
                "anthropic %s: %.1fs, in %s tok, out %s tok, cache read %s tok",
                self.model, time.perf_counter() - started,
                getattr(usage, "input_tokens", "?"), getattr(usage, "output_tokens", "?"),
                getattr(usage, "cache_read_input_tokens", "?"),
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
