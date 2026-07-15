# idchkg

A **local, LLM-based knowledge-graph pipeline with built-in judges**, inspired by
Marieke van Erp's vision paper *"Unflattening Knowledge Graphs"* (K-CAP 2023,
[DOI 10.1145/3587259.3630082](https://doi.org/10.1145/3587259.3630082)).

Most knowledge graphs *flatten* the world into bare `(subject, predicate, object)`
triples. The paper argues that entities and concepts carry **multiple
perspectives**, **change over time**, and connect to **different narratives** —
and that this richness is exactly what current KGs throw away. Its running
example is *coffee*: a plant, a drink, an activity, and a concept whose history
is entangled with colonialism and status, described very differently depending
on who is speaking and when.

This project builds an **"unflattened" graph**: every statement is a reified,
first-class **claim** that keeps three extra dimensions the paper asks for:

| Paper's challenge | What we keep on every claim |
| --- | --- |
| **Polyvocality** — many perspectives | a `Perspective` (viewpoint, stance, sentiment) |
| **Time** — concepts change | a `TemporalScope` (source expression + normalised dates) |
| **Provenance** — where it came from | a `Provenance` (document, chunk, verbatim quote, judge verdicts) |

Extraction is done by an LLM, and every extracted claim is reviewed by
**built-in LLM judges** before it can enter the graph.

## Pipeline

```
ingest → chunk → extract (LLM) → judge (LLM panel) → repair (LLM, judge-guided)
       → resolve entities → in-memory graph (networkx)
```

- **Extract** — an LLM turns each chunk into entities and qualified claims,
  with a verbatim supporting quote (`src/idchkg/extract.py`).
- **Judge** — two independent LLM judges score each claim
  (`src/idchkg/judge.py`):
  - **groundedness** — is the claim actually stated by the passage? A failure
    means hallucination → the claim is **rejected**.
  - **qualifier fidelity** — are the perspective/time/quote faithful? A failure
    → the claim is sent back for **one judge-guided repair round**, then
    re-judged.
- **Resolve** — mentions of the same entity are merged into one canonical node
  (`src/idchkg/resolve.py`).
- **Graph** — an in-memory `networkx.MultiDiGraph` where claims are nodes, so
  the same two entities can be linked by many claims from different viewpoints
  and times without collapsing them (`src/idchkg/graph.py`).

Every claim carries its judges' verdicts in its provenance, so the graph is
auditable: you can always ask *why* a claim is in (or out of) the graph.

## Runs locally, in memory

- The graph lives in memory (networkx); persistence is plain JSON, plus
  GraphML and reified-statement Turtle/RDF export.
- **Backends** (`--backend`):
  - `ollama` — a fully local model via [Ollama](https://ollama.com) on
    `localhost:11434` (no cloud, no key).
  - `anthropic` — Claude via the official SDK (structured outputs). Set
    `ANTHROPIC_API_KEY` or run `ant auth login`.
  - `mock` — deterministic, scriptable backend for tests and offline demos.

The same schema-constrained interface (`complete_json`) is used for both
extraction and judging, so any backend works for both roles — and you can even
use *different* models for extractor and judge.

## Install

```bash
pip install -e .                 # core (networkx only)
pip install -e '.[anthropic]'    # + Claude backend
pip install -e '.[dev]'          # + pytest
```

## Use it

Offline demo (no API key — scripted mock plays extractor and judges):

```bash
python examples/demo_coffee.py
```

Build a graph from your own documents and query it:

```bash
# Fully local with Ollama:
idchkg build examples/corpus --backend ollama --model llama3.1 -o out

# Or with Claude:
idchkg build examples/corpus --backend anthropic -o out

idchkg query out/graph.json coffee --view perspectives
idchkg query out/graph.json coffee --view timeline
idchkg query out/graph.json coffee --view provenance
idchkg query out/graph.json        --view conflicts
```

`build` writes `graph.json`, `graph.graphml`, `graph.ttl`, and a `report.json`
containing the run stats plus every rejected claim with the judges' reasoning.

### As a library

```python
from idchkg import Pipeline, PipelineConfig

pipeline = Pipeline(PipelineConfig(backend="ollama", model="llama3.1"))
docs = pipeline.load_documents(["examples/corpus"])
result = pipeline.run(docs)

kg = result.graph
coffee = kg.find_entity("coffee")
kg.perspectives_on(coffee)   # {viewpoint: [claims]}
kg.timeline(coffee)          # claims ordered by time
kg.conflicts(coffee)         # same subject+predicate, differing object/stance
kg.provenance_of(claim_id)   # source + quote + judge verdicts
```

## Query views (the "unflattening")

- **perspectives** — claims about an entity grouped by the viewpoint that
  makes them (Challenge 1).
- **timeline** — claims about an entity ordered by temporal scope (Challenge 2).
- **provenance** — full trace for each claim: source document, verbatim quote,
  extractor, and every judge verdict (Challenge 3).
- **conflicts** — groups of claims that share subject and predicate but differ
  in object or stance: the disagreements a flat KG would silently reconcile.

## Layout

```
src/idchkg/
  models.py     data model: Entity, Claim, Perspective, TemporalScope, Provenance
  llm/          backends: anthropic, ollama, mock (schema-constrained JSON)
  prompts.py    extraction + judge prompts and JSON schemas
  extract.py    chunking + LLM extraction (+ deterministic validation gate)
  judge.py      LLM judge panel (groundedness, qualifier fidelity)
  resolve.py    entity resolution
  graph.py      in-memory knowledge graph + JSON/GraphML/Turtle export
  pipeline.py   orchestration (extract → judge → repair → load)
  cli.py        `idchkg build` / `idchkg query`
examples/       coffee corpus + offline demo
tests/          pytest suite (runs offline via the mock backend)
```

## Tests

```bash
pytest
```

The suite runs entirely offline using the scriptable mock backend — no model
calls, no network.

## Notes and next steps

This is a working foundation, not a finished product. Natural extensions:

- an LLM **merge judge** for entity resolution (the default resolver
  deliberately under-merges, since resolution mistakes are silent);
- richer temporal normalisation (the extractor is asked for ISO dates, but a
  dedicated date-parsing pass would be more robust);
- concurrency across chunks;
- a persistent backend (e.g. SQLite or an RDF store) behind the same graph API.
