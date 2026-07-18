"""Live integration tests against a running Ollama server.

These are opt-in: the whole module **auto-skips** when Ollama is unreachable (or
`IDCHKG_SKIP_OLLAMA=1` is set), so the offline suite stays green. When Ollama is
up they exercise the real extraction/judge path and the SQLite store together.

Model choice: prefer extraction/instruct models and **skip reasoning models**
(`deepseek-r1`) — the Ollama backend forces JSON via `format: <schema>`, which
collides with a model that wants to emit a chain-of-thought first. Preference
order: nuextract -> phi4-ie -> phi3, then any non-reasoning model present.

Assertions are structural (valid schema, pipeline completes, round-trips), never
exact claim content — LLM output is nondeterministic.
"""

import json
import os
import urllib.error
import urllib.request

import pytest

from idchkg.llm.ollama_client import OllamaClient
from idchkg.models import Document
from idchkg.pipeline import Pipeline, PipelineConfig
from idchkg.prompts import (
    EXTRACTION_SCHEMA,
    EXTRACTION_SYSTEM,
    GROUNDEDNESS_SYSTEM,
    VERDICT_SCHEMA,
    extraction_user,
    judge_user,
)
from idchkg.store import load_sqlite, retrieve

_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_PREFERENCES = ["nuextract", "phi4-ie", "phi3"]


def _available_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{_BASE_URL}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [m["name"] for m in data.get("models", [])]


def _pick_model() -> str | None:
    if os.environ.get("IDCHKG_SKIP_OLLAMA"):
        return None
    models = [m for m in _available_models() if "deepseek-r1" not in m]  # skip reasoning models
    if not models:
        return None
    for pref in _PREFERENCES:
        for name in models:
            if pref in name:
                return name
    return models[0]


_MODEL = _pick_model()
pytestmark = pytest.mark.skipif(
    _MODEL is None,
    reason="Ollama not reachable, no usable model, or IDCHKG_SKIP_OLLAMA set",
)


def test_extraction_returns_schema_valid_json():
    client = OllamaClient(model=_MODEL)
    raw = client.complete_json(
        EXTRACTION_SYSTEM,
        extraction_user("Coffee is a beverage brewed from the roasted seeds of the Coffea plant."),
        EXTRACTION_SCHEMA,
    )
    assert isinstance(raw, dict)
    assert isinstance(raw.get("entities"), list)
    assert isinstance(raw.get("claims"), list)


def test_verdict_returns_schema_valid_json():
    client = OllamaClient(model=_MODEL)
    claim_json = json.dumps({"subject": "coffee", "predicate": "is a", "object": "beverage", "quote": "Coffee is a beverage"})
    raw = client.complete_json(
        GROUNDEDNESS_SYSTEM,
        judge_user("Coffee is a beverage.", claim_json),
        VERDICT_SCHEMA,
    )
    assert isinstance(raw.get("score"), (int, float))
    assert isinstance(raw.get("passed"), bool)
    assert isinstance(raw.get("feedback"), str)


def test_pipeline_end_to_end_completes():
    pipeline = Pipeline(PipelineConfig(backend="ollama", model=_MODEL))
    doc = Document(id="coffee", title="coffee",
                   text="Coffee is a beverage made from roasted seeds. Historians link it to colonial trade.")
    result = pipeline.run([doc])

    stats = result.stats()
    assert {"entities", "claims_accepted", "claims_rejected", "documents"} <= set(stats)
    # Structural, not content: every claim bucket is a list; nothing crashed.
    assert isinstance(result.accepted, list)
    assert isinstance(result.rejected, list)
    assert result.graph is not None


def test_ollama_graph_saves_and_retrieves(tmp_path):
    pipeline = Pipeline(PipelineConfig(backend="ollama", model=_MODEL))
    doc = Document(id="sugar", title="sugar",
                   text="Sugar was a major commodity of the Atlantic trade and plantation economy.")
    result = pipeline.run([doc])

    path = tmp_path / "ollama.db"
    result.graph.save_sqlite(path)
    # The SQLite store + FTS index work on a real Ollama-built graph.
    assert load_sqlite(path) is not None
    hits = retrieve(path, "Atlantic trade", k=3)
    assert hits and hits[0]["document_id"] == "sugar"
