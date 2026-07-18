# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, LLM-based knowledge-graph pipeline with built-in LLM judges, implementing the "unflattening" ideas from van Erp's *Unflattening Knowledge Graphs* (K-CAP 2023). The design is organised around the paper's **three challenges**, and that framing is load-bearing throughout the code — keep it in mind when changing anything:

1. **Identity** — entities are keyed by `(label, sense)`, so different senses of a word (coffee-plant vs coffee-beverage vs coffee-commodity) are *distinct nodes* joined by context-scoped near-identity links, never merged.
2. **Change** — claims carry a `TemporalScope`; the graph exposes drift over time and records `EvolutionEvent`s (split/merge/cease/gains/loses).
3. **Long tail** — every claim keeps full `Provenance` (source, quote, judge verdicts) and entities track a `mention_count` (head vs long tail).

A fourth, complementary dimension — `Perspective` (viewpoint/stance/sentiment) — lets contradictory claims coexist instead of being reconciled.

## Commands

```bash
pip install -e '.[dev]'          # dev install (adds pytest); extras: '.[anthropic]' Claude backend, '.[viz]' yFiles
pytest                           # full suite — offline via the mock backend; live Ollama tests auto-skip if unreachable
pytest tests/test_pipeline.py::test_accepted_claim_enters_graph   # single test
python examples/demo_coffee.py   # offline end-to-end demo (scripted mock plays extractor + judges)
```

There is **no linter, formatter, or build step** configured — `pytest` is the only check. This repo is used with a `.venv/` (Python 3.13); install extras into it (`uv pip install --python .venv/bin/python -e '.[dev]'`) — the notebook and Ollama both run against that venv.

Running the real pipeline (makes LLM calls):

```bash
idchkg build examples/corpus --backend ollama --model phi4-ie -o out   # or --backend anthropic
idchkg query out/graph.json coffee --view identity     # also: change, longtail, provenance, perspectives, conflicts
idchkg search out/graph.db "coffee plant" -k 5         # BM25 retrieval over the chunk index
python examples/corpora/fetch.py                       # populate real corpora (Gutenberg + Wikipedia) into gitignored raw/ & wiki/
```

`build` writes `graph.json`, `graph.graphml`, `graph.ttl`, `graph.db`, and `report.json` (run stats + every rejected claim with judge reasoning) into the output dir (`out/`, gitignored).

## Architecture

Pipeline flow (`src/idchkg/pipeline.py` orchestrates it):

```
load_documents → chunk → extract (LLM) → judge (LLM panel) → repair (LLM, ≤1 round) → resolve → in-memory graph
```

Key structural decisions to preserve when editing:

- **Backend abstraction (`llm/`).** Every backend (`anthropic`, `ollama`, `mock`) implements a single method: `complete_json(system, user, schema) -> dict`. Both extraction *and* judging go through this one interface, so any backend works for either role, and extractor/judge can use different models (`judge_backend`/`judge_model` in `PipelineConfig`). Add new backends by wiring them into `llm/__init__.py::get_client`.

- **Judges gate everything (`judge.py`).** No claim enters the graph without passing the `JudgePanel`. Two independent judges run in sequence: **groundedness** first (failure → immediate REJECT, treated as hallucination, unrepairable), then **qualifier fidelity** (failure → REVISE → one judge-guided repair round in the pipeline, then re-judged). The numeric `score` vs `threshold` is authoritative — the judge's boolean `passed` flag is only trusted if it agrees with the score.

- **Claims are reified as graph nodes (`graph.py`), not edges.** The graph is a `networkx.MultiDiGraph` where each `Claim` is its own node, so the same two entities can be linked by many claims from different viewpoints/times without collapsing. Every claim carries its judges' verdicts in `Provenance`, making the graph auditable ("why is this claim in/out?").

- **Resolution deliberately under-merges (`resolve.py`).** `EntityResolver` keys entities by `(canonical_label, sense)`. A wrong merge is silent and destructive, so when a claim references a label+sense not yet seen, it only reuses an existing node if the label has *exactly one* sense on record; otherwise it creates a new node. New senses of a known label are auto-linked with `near_identity` `IdentityLink`s.

- **Data model (`models.py`).** `Claim` is a reified statement (`subject/predicate/object` + `Perspective` + `TemporalScope` + `Provenance` + `context`/`subject_sense`/`object_sense`). Claim `id` is a content hash — changing which fields feed the hash (see `__post_init__`) changes claim identity and dedup. Positional field order in `Claim` is kept backward-compatible on purpose (note the comment before `context`).

- **Persistence & retrieval (`store.py`).** `save_sqlite`/`load_sqlite` (also on `KnowledgeGraph`) mirror `to_dict()`/`from_dict` into a single-file SQLite DB, plus an **FTS5** virtual table over `chunk_document` chunks; `retrieve(db, query, k)` ranks with `bm25()`. networkx stays the in-memory projection — SQLite is durable storage + the search index, not a replacement graph engine. `save_json`/`from_dict` drop document text; the SQLite store keeps it (needed to rebuild chunks). See the design/decision doc at `~/.claude/plans/right-now-this-code-serialized-tiger.md` (why SQLite over DuckDB/Kùzu/lance-graph; Kùzu is archived).

- **Visualization (`viz.py`, optional `[viz]` extra).** `to_yfiles_widget(kg)` renders the graph in a Jupyter notebook via yFiles; `entity_subgraph(kg, label)` scopes a focused view. Both consume `kg.g`, so they're persistence-backend-agnostic.

## Testing conventions

Tests drive the whole pipeline through a scripted `MockLLM` (`llm/mock.py`): `.queue(...)` returns responses in the **exact order the pipeline consumes them** — per chunk one extraction, then per claim a groundedness verdict, a qualifier verdict, and (on revision) a repair extraction. When the queue empties, the mock synthesises a minimal schema-conformant default. If you change the number or order of LLM calls in the pipeline/judge, the queued sequences in `tests/test_pipeline.py` must be updated to match.

`tests/test_store.py` is offline (deterministic SQLite round-trip + BM25). `tests/test_ollama_live.py` hits a **running Ollama** and **auto-skips** when it's unreachable (or `IDCHKG_SKIP_OLLAMA=1`); it prefers extraction/instruct models (`nuextract`/`phi4-ie`/`phi3`) and skips reasoning models (`deepseek-r1`), whose chain-of-thought collides with the backend's forced `format: <schema>` decoding. Live-model assertions are structural (valid schema, pipeline completes), never exact claim content.
