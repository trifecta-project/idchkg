"""Deterministic backend for tests and offline smoke runs.

Responses can be scripted with ``queue()``; when the queue is empty the mock
synthesises a minimal schema-conformant default (empty extraction, passing
verdict), so the whole pipeline can run end-to-end with no model at all.
"""

from __future__ import annotations

import threading


class MockLLM:
    name = "mock"

    def __init__(self):
        self._queue: list[dict] = []
        self.calls: list[dict] = []  # recorded {system, user, schema}
        self._lock = threading.Lock()  # pipeline may call from worker threads

    def queue(self, *responses: dict) -> "MockLLM":
        self._queue.extend(responses)
        return self

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        with self._lock:
            self.calls.append({"system": system, "user": user, "schema": schema})
            response = self._queue.pop(0) if self._queue else None
        return response if response is not None else self._default_for(schema)

    @staticmethod
    def _default_for(schema: dict) -> dict:
        """Build a minimal instance of the schema (objects/arrays/scalars)."""

        def build(node: dict):
            node_type = node.get("type")
            if "enum" in node:
                return node["enum"][0]
            if node_type == "object":
                return {
                    key: build(sub)
                    for key, sub in node.get("properties", {}).items()
                    if key in node.get("required", [])
                }
            if node_type == "array":
                return []
            if node_type == "string":
                return ""
            if node_type == "number":
                return 1.0
            if node_type == "integer":
                return 1
            if node_type == "boolean":
                return True
            if isinstance(node_type, list):  # e.g. ["string", "null"]
                return None if "null" in node_type else ""
            return None

        return build(schema)
