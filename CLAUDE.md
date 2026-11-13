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
idchkg build examples/corpus --backend ollama --model granite4.1:8b -o out   # or --backend anthropic
idchkg build big_corpus --store sqlite --concurrency 8 -o out                 # stream to graph.db, bounded memory
idchkg query out/graph.json coffee --view identity     # also: change, longtail, provenance, perspectives, conflicts
idchkg query out/graph.db coffee --view identity       # a .db graph is queried against SQLite (no full load)
idchkg search out/graph.db "coffee plant" -k 5         # BM25 retrieval over the chunk index
idchkg site out/graph.db -o site/                      # generate a static browse site (open site/index.html)
idchkg build examples/corpus --link-dbpedia -o out     # also link entities to DBpedia during the build
idchkg link out/graph.db                               # enrich an existing graph.db with DBpedia links (no rebuild)
idchkg link out/graph.db --target wikidata --types place   # link to Wikidata instead (coords P625, start/end dates); cached on disk
python examples/osint_map.py out/graph.db -o out/osint_map.html   # interactive globe/map "atlas" of place claims
open examples/atlas_demo.html                          # committed example atlas (commodities corpus, Wikidata-linked); regenerate rather than hand-edit
python examples/corpora/fetch.py                       # populate real corpora (Gutenberg + Wikipedia) into gitignored raw/ & wiki/
```

`build` runs LLM calls concurrently across chunks (`--concurrency`, default 4; the mutation of the graph stays single-threaded and deterministic). Default `--store memory` builds in RAM and writes `graph.json`, `graph.graphml`, `graph.ttl`, `graph.db`, `report.json`; `--store sqlite` streams claims straight to `graph.db` (bounded memory) and writes only `graph.db` + `report.json`. For Ollama, real speedup needs `OLLAMA_NUM_PARALLEL>1` on the server.

`build` logs (stdlib `logging`, configured in `cli.py:main`): INFO shows per-chunk progress + an **end-of-run timing breakdown** (extraction vs judging wall time/calls — the "why is it slow" summary); `-v` adds per-call LLM metrics (tok/s, model-load ms — from Ollama's own response fields); `-q` quiets to warnings. Library code uses module loggers and emits no output unless the host configures logging (or `Pipeline.run(verbose=True)`, which turns on a console handler — used by the notebook). Judging is usually the call-count driver: **2 judge calls per claim vs 1 extraction per chunk**.

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

- **Persistence & retrieval (`store.py`).** Two SQLite paths over one schema + an **FTS5** chunk index (`retrieve(db, query, k)` ranks with `bm25()`): (1) `save_sqlite`/`load_sqlite` serialize a whole in-memory `KnowledgeGraph`; (2) **`GraphStore`** is a real DB-backed graph — `mode="build"` **duck-types the `KnowledgeGraph` build API** (`add_document`/`add_claim`/`add_identity_link`/`finalize_entities`) so `Pipeline.run(..., graph=store)` **streams** claims to disk without holding them in RAM, and `mode="query"` answers the view methods via SQL. GraphStore keeps only entities (few) in memory and fetches claims per query; corpus-wide `conflicts` is a SQL `GROUP BY`; per-entity views build a tiny subgraph and reuse `graph.py` logic verbatim. networkx stays the in-memory projection for the `save/load` path and viz. See the design doc at `~/.claude/plans/right-now-this-code-serialized-tiger.md` (why SQLite over DuckDB/Kùzu/lance-graph; Kùzu is archived).

- **Concurrency (`pipeline.py`).** `run()` maps `_process_chunk` over chunks through a `ThreadPoolExecutor` (`PipelineConfig.concurrency`), but applies every result to the shared resolver + graph on the main thread in chunk order, so the graph is deterministic regardless of concurrency. `_map` runs sequentially for ≤1 item or `concurrency<=1` (the scripted `MockLLM` relies on deterministic call order; `MockLLM` is also lock-guarded).

- **DBpedia linking (`link.py`, opt-in).** `DBpediaLinker` links resolved entities to DBpedia resources via a **Spotlight** server (public API by default; `--spotlight-url` for a local one), feeding each entity's label *plus* its descriptions/contexts so disambiguation is context-aware (Challenge 1). It stores a JSON blob in a new `entities.dbpedia` column: `{uri, score, surfaceForm, types, dbo_type}` — `types` are the resource's ontology classes read free from Spotlight's `@types`; `dbo_type` is the `dbo:type` property fetched per entity from the resource's data JSON. Linking runs in `Pipeline.run` **after resolve, before `finalize_entities`** (so both the in-memory graph and the streaming store persist it in one write) when `PipelineConfig.link_dbpedia` is set, or standalone via `store.enrich_sqlite` (CLI `idchkg link`), which `ALTER TABLE ... ADD COLUMN` migrates pre-existing DBs and reads tolerate the column's absence. It's **opt-in and network-bound** (entity labels + context leave the machine on the public endpoint); everything is best-effort — a failed lookup leaves an entity unlinked, never aborting the run. The linker is injectable so tests stay offline.

- **Visualization (`viz.py`, optional `[viz]` extra).** `to_yfiles_widget(kg)` renders the graph in a Jupyter notebook via yFiles; `entity_subgraph(kg, label)` scopes a focused view. Both consume `kg.g`, so they're persistence-backend-agnostic.

- **Wikidata linking & dates (`wikidata.py`, `dates.py`, opt-in).** `WikidataLinker` mirrors `DBpediaLinker` (injectable `fetch`, best-effort) but scores candidates with deterministic, context-aware heuristics (sense/descriptions/contexts, P31 types, coordinates, start/end vs years in context) and requires evidence beyond the name match; each `(label, sense)` is scored independently, so senses may link to different QIDs. Stored as JSON in `entities.wikidata` (retrofit column added by `store.enrich_wikidata_sqlite`, like `dbpedia`). `dates.normalize(text) -> (earliest, latest, edtf)` is a rule-based parser for the free-text `when_`/`valid_*` fields; "present" is deliberately None (depends on the unknown document date).

- **Atlas prototype (`examples/osint_map.py` + `osint_map_template.html`).** Exports place claims (Wikidata coords, gazetteer fallback) as JSON embedded in a single page: MapLibre globe/flat map + deck.gl raised arcs + linked time/viewpoint/stance/commodity filters. deck's `ArcLayer`/`LineLayer` don't render on a globe (deck 9.4), so arcs are explicit 3D great-circle `PathLayer` paths. A `/*__LAND__*/` placeholder lets a build embed an offline basemap for CSP-restricted hosts (e.g. claude.ai artifacts).

- **Static web site (`site.py`).** `build_site(graph, out_dir)` (CLI: `idchkg site`) renders a zero-dependency, entity-per-page browse site — pre-rendered HTML + one CSS + a few lines of vanilla JS, works over `file://`. It calls the same view API on a `GraphStore` or `KnowledgeGraph`, so it's source-agnostic. The entity page centres on **perspectives** (`perspectives_on` claims grouped by viewpoint, with stance/sentiment chips, verbatim quote, source, and `<details>` judge verdicts). All LLM-extracted text is `html.escape`d. Styling target is trifecta.dhlab.nl (clean academic/DH). Cross-links use a `(canonical label, sense)` lookup mirroring the resolver.

## Testing conventions

Tests drive the whole pipeline through a scripted `MockLLM` (`llm/mock.py`): `.queue(...)` returns responses in the **exact order the pipeline consumes them** — per chunk one extraction, then per claim a groundedness verdict, a qualifier verdict, and (on revision) a repair extraction. When the queue empties, the mock synthesises a minimal schema-conformant default. If you change the number or order of LLM calls in the pipeline/judge, the queued sequences in `tests/test_pipeline.py` must be updated to match.

`tests/test_store.py` is offline (deterministic SQLite round-trip + BM25). `tests/test_ollama_live.py` hits a **running Ollama** and **auto-skips** when it's unreachable (or `IDCHKG_SKIP_OLLAMA=1`); it prefers extraction/instruct models (`nuextract`/`phi4-ie`/`phi3`) and skips reasoning models (`deepseek-r1`), whose chain-of-thought collides with the backend's forced `format: <schema>` decoding. Live-model assertions are structural (valid schema, pipeline completes), never exact claim content.
