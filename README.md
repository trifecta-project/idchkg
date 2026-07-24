# idchkg

A **LLM-based knowledge-graph pipeline for building thick knowledge graphs**, based on the vision paper *"Unflattening Knowledge Graphs"* (K-CAP 2023,
[DOI 10.1145/3587259.3630082](https://doi.org/10.1145/3587259.3630082)).

Most knowledge graphs *flatten* the world into `(subject, predicate, object)`
triples. For example, Wikidata represents  *coffee*:  as a
"brewed beverage made from seeds of Coffea genus", so we cannot tell apart a
reference to the drink, the plant, the activity of drinking it, or the colonial
commodity. Its history is  absent from the structured data.

The vision paper identifies **three crucial challenges** for "unflattening" KGs:

| Paper's challenge | How this project addresses it |
| --- | --- |
| **1. Identity** — concepts have many dimensions; `owl:sameAs` over-merges | entities are keyed by **(label, sense)**, so coffee-the-plant, coffee-the-beverage and coffee-the-commodity are **distinct nodes** joined by context-scoped **near-identity** links, not one flat node |
| **2. Change** — identities drift, split, merge, cease | claims carry a `TemporalScope`; |
| **3. The long tail** — infrequent entities vanish without provenance | every claim stores full **`Provenance`** (source, quote, judge verdicts) and entities keep a **mention count** to separate head from long-tail |

Extraction is done by an LLM, and every extracted claim is reviewed by
**built-in LLM judges** before it can enter the graph.

> **Complementary dimension.** Each claim also carries a `Perspective`
> (viewpoint / stance / sentiment). It lets the graph keep
> disagreements (e.g. colonial writers vs. historians on coffee) side by side.

## Pipeline

```
ingest → chunk → extract (LLM) → judge (LLM panel) → repair (LLM, ≤1 round)
       → resolve entities → [link to DBpedia] → knowledge graph
```

Chunks are processed **concurrently** (`--concurrency`), but every accepted
claim is applied to the shared graph on a single thread in document/chunk order,
so the resulting graph is **deterministic** regardless of concurrency
(`src/idchkg/pipeline.py`).

- **Extract** — an LLM turns each chunk into entities and qualified claims,
  with a verbatim supporting quote (`src/idchkg/extract.py`).
- **Judge** — two independent LLM judges score each claim
  (`src/idchkg/judge.py`):
  - **groundedness** — is the claim actually stated by the passage? A failure
    means hallucination → the claim is **rejected** (unrepairable).
  - **qualifier fidelity** — are the perspective/time/quote faithful? A failure
    → the claim is sent back for **one judge-guided repair round**, then
    re-judged.
- **Resolve** — mentions are merged by **(label, sense)**, so different senses
  of a word stay distinct and are linked by near-identity; the resolver
  deliberately under-merges, since a wrong merge is silent and destructive
  (`src/idchkg/resolve.py`).
- **Link** *(optional, `--link-dbpedia`)* — each resolved entity is linked to
  its DBpedia resource via Spotlight (context-aware), after resolve and before
  the entities are finalized (`src/idchkg/link.py`; see
  [Link entities to DBpedia](#link-entities-to-dbpedia)).
- **Graph** — claims are reified as **nodes**, so the same two entities can be
  linked by many claims from different viewpoints and times without collapsing
  them. The target is either an in-memory `networkx.MultiDiGraph`
  (`src/idchkg/graph.py`) or, for large corpora, a **streaming SQLite store**
  that writes claims straight to `graph.db` under bounded memory
  (`--store sqlite`, `src/idchkg/store.py`).

Every claim carries its judges' verdicts in its provenance, so the graph is
auditable: you can always ask *why* a claim is in (or out of) the graph.

## Runs locally

- **Storage** — two paths over one SQLite schema (`src/idchkg/store.py`):
  - `--store memory` (default) builds the graph in RAM (networkx) and writes
    `graph.json`, `graph.graphml`, reified-statement `graph.ttl`, and `graph.db`.
  - `--store sqlite` streams claims straight to `graph.db` under bounded memory
    and writes only `graph.db` — so large corpora don't have to fit in RAM.

  Either way `graph.db` carries an **FTS5** chunk index for BM25 retrieval
  (`idchkg search`), and a `.db` graph is **queried via SQL** without loading it
  in full. Both runs also emit a `report.json` with run stats and every rejected
  claim.
- **Backends** (`--backend`):
  - `ollama` — a fully local model via [Ollama](https://ollama.com) on
    `localhost:11434` (no cloud, no key).
  - `anthropic` — Claude via the official SDK (structured outputs). Set
    `ANTHROPIC_API_KEY` or run `ant auth login`.
  - `mock` — deterministic, scriptable backend for tests and offline demos.

The same schema-constrained interface (`complete_json`) is used for both
extraction and judging, so any backend works for both roles — and you can even
use *different* models for extractor and judge.

## Setup

### 1. Prerequisites

- **Python 3.10+** (developed on 3.13). Check with `python3 --version`.
- **git**, to clone the repo.
- **An LLM backend** — pick at least one:
  - **Ollama** (recommended, fully local, no key) — install from
    [ollama.com](https://ollama.com), then pull a model (see step 4).
  - **Anthropic / Claude** — an `ANTHROPIC_API_KEY` (structured-output SDK).
  - Neither is needed to run the offline demo or the test suite, which use the
    built-in scriptable `mock` backend.

### 2. Get the code

```bash
git clone <this-repo-url> idchkg
cd idchkg
```

### 3. Create a virtual environment and install

Using the standard library `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # core (networkx only)
pip install -e '.[dev]'              # + pytest (to run the tests)
pip install -e '.[anthropic]'        # + Claude backend (optional)
pip install -e '.[viz]'              # + yFiles notebook viz (optional)
# combine extras: pip install -e '.[dev,anthropic,viz]'
```

Prefer [`uv`](https://docs.astral.sh/uv/)? It's faster and what this repo is
developed with:

```bash
uv venv                              # creates .venv/
uv pip install --python .venv/bin/python -e '.[dev]'
```

### 4. (If using Ollama) start the server and pull a model

```bash
ollama serve                         # or just run the Ollama app; listens on localhost:11434
ollama pull granite4.1:8b            # the model used in the examples below
```

Any instruction/extraction model works. For real concurrency across chunks,
start the server with `OLLAMA_NUM_PARALLEL=4` (or higher).

### 5. Verify the install

```bash
pytest                               # full suite, runs offline via the mock backend
python examples/demo_coffee.py       # offline end-to-end demo (no API key)
```

Both should pass without any network access or API key.

## Use it

Offline demo (no API key — scripted mock plays extractor and judges):

```bash
python examples/demo_coffee.py
```

Build a graph from your own documents and query it:

```bash
# Fully local with Ollama:
idchkg build examples/corpus --backend ollama --model granite4.1:8b -o out

# Or with Claude:
idchkg build examples/corpus --backend anthropic -o out

# the paper's three challenges:
idchkg query out/graph.json coffee --view identity     # Ch.1 senses + near-identity
idchkg query out/graph.json coffee --view change       # Ch.2 drift over time
idchkg query out/graph.json        --view longtail     # Ch.3 head vs long tail
idchkg query out/graph.json coffee --view provenance   # Ch.3 source + judge verdicts

# complementary:
idchkg query out/graph.json coffee --view perspectives
idchkg query out/graph.json        --view conflicts
```

`build` writes `graph.json`, `graph.graphml`, `graph.ttl`, `graph.db`, and a
`report.json` containing the run stats plus every rejected claim with the judges'
reasoning. Build logs per-chunk progress and an end-of-run timing breakdown
(`-v` for per-call LLM metrics, `-q` to quiet).

For larger corpora, stream straight to SQLite instead of holding the graph in
RAM, and run LLM calls concurrently across chunks:

```bash
idchkg build big_corpus --store sqlite --concurrency 8 -o out   # writes only graph.db + report.json
idchkg query out/graph.db coffee --view identity                # a .db graph is queried via SQL, not fully loaded
idchkg search out/graph.db "coffee plant" -k 5                  # BM25 retrieval over the chunk index
idchkg site out/graph.db -o site/                              # static browse site (open site/index.html)
```

Want to try it on real text? `python examples/corpora/fetch.py` populates
representative corpora (Project Gutenberg + Wikipedia) into a gitignored
`raw/`/`wiki/` you can point `build` at.

### Link entities to DBpedia

Optionally connect each entity to its **DBpedia** resource via [DBpedia
Spotlight](https://www.dbpedia.org/resources/spotlight/), which disambiguates
using context (so *coffee* the plant and *coffee* the beverage resolve to
different resources). The link — resource URI, confidence, and its DBpedia
ontology types (`rdf:type` and `dbo:type`) — is stored in a new `dbpedia` column
on the `entities` table and shown on each entity's page in the browse site.

```bash
idchkg build examples/corpus --link-dbpedia -o out   # link during the build
idchkg link out/graph.db                             # or enrich an existing graph.db in place (no rebuild)
idchkg link out/graph.db --spotlight-url http://localhost:2222/rest   # use a local Spotlight
```

> **Note:** linking is **opt-in** and, on the default public endpoint, sends
> entity labels + a little context to the DBpedia Spotlight service — the only
> step that leaves your machine. Point `--spotlight-url` at a local Spotlight to
> keep it fully offline.

### Browse it as a web site

Turn a built graph into a static, zero-dependency browse site — one page per
entity, centred on perspectives (stance/sentiment, verbatim quote, source, and
the judge verdicts behind each claim):

```bash
idchkg site out/graph.db -o site/     # works on graph.db or graph.json
```

`-o/--out` sets the output directory (default: `site`). The result is plain
pre-rendered HTML + CSS + a little vanilla JS, so you can just open it:

```bash
open site/index.html                  # macOS; or double-click the file
python -m http.server -d site         # or serve it at http://localhost:8000
```

### As a library

```python
from idchkg import Pipeline, PipelineConfig

pipeline = Pipeline(PipelineConfig(backend="ollama", model="llama3.1"))
docs = pipeline.load_documents(["examples/corpus"])
result = pipeline.run(docs)

kg = result.graph

# Challenge 1 — identity
kg.senses_of("coffee")               # distinct sense-entities for one label
plant = kg.find_entity("coffee", sense="plant")
kg.identity_profile(plant)           # defining properties in context
kg.near_identities(plant)            # context-scoped near-identity links

# Challenge 2 — change
kg.drift(plant)                      # properties bucketed by time
kg.evolution_of(plant)              # split / merge / cease / gains / loses

# Challenge 3 — the long tail
kg.long_tail()                       # {head: [...], tail: [...]} by mention count
kg.provenance_of(claim_id)           # source + quote + judge verdicts

# complementary
kg.perspectives_on(plant)            # {viewpoint: [claims]}
kg.conflicts(plant)                  # same subject+predicate, differing object/stance
```

## Query views (the "unflattening")

The three views named for the paper's challenges:

- **identity** (Ch.1) — the distinct **senses** of a concept (coffee: plant /
  beverage / commodity …), each with its defining-property profile, plus the
  near-identity links between them.
- **change** (Ch.2) — how an entity's properties **drift** across time buckets,
  plus any recorded evolution events.
- **longtail** (Ch.3) — entities partitioned into **head** vs **long tail** by
  mention count.
- **provenance** (Ch.3) — full trace for each claim: source document, verbatim
  quote, extractor, and every judge verdict.

Complementary:

- **perspectives** — claims about an entity grouped by the viewpoint that makes
  them.
- **conflicts** — groups of claims that share subject and predicate but differ
  in object or stance: disagreements a flat KG would silently reconcile.

## Layout

```
src/idchkg/
  models.py     data model: Entity(+sense), Claim(+context), Perspective,
                TemporalScope, Provenance, IdentityLink, EvolutionEvent
  llm/          backends: anthropic, ollama, mock (schema-constrained JSON)
  prompts.py    extraction + judge prompts and JSON schemas
  extract.py    chunking + LLM extraction (+ deterministic validation gate)
  judge.py      LLM judge panel (groundedness, qualifier fidelity)
  resolve.py    (label, sense) entity resolution + near-identity links
  link.py       optional DBpedia entity linking via Spotlight (context-aware)
  graph.py      in-memory knowledge graph + JSON/GraphML/Turtle export
  pipeline.py   orchestration (extract → judge → repair → [link] → load)
  cli.py        `idchkg build` / `idchkg query` / `idchkg link` / `idchkg site`
examples/       coffee demo + offline demo script
  corpora/      representative food- & maritime-history corpora + a catalogue
                (examples/corpora/README.md) of real datasets to scale to
tests/          pytest suite (runs offline via the mock backend)
```

## Tests

```bash
pytest
```

The suite runs entirely offline using the scriptable mock backend — no model
calls, no network.

