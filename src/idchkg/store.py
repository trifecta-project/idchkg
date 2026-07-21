"""SQLite persistence + a BM25 retrieval index for the knowledge graph.

A single ``.db`` file holds the whole graph (documents, entities, reified
claims, identity links, evolution events) in relational tables, plus an FTS5
virtual table over document chunks that gives keyword retrieval via ``bm25()``
with no extra dependency.

    from idchkg.store import save_sqlite, load_sqlite, retrieve
    save_sqlite(kg, "graph.db")
    kg2 = load_sqlite("graph.db")            # round-trips the in-memory graph
    hits = retrieve("graph.db", "coffee")    # BM25-ranked chunk hits

The graph tables mirror ``KnowledgeGraph.to_dict()``; loading reconstructs that
same shape and hands it to ``KnowledgeGraph.from_dict``. Chunks are produced by
the same ``chunk_document`` the pipeline uses, so retrieval sees exactly the
passages extraction saw. networkx stays the in-memory projection — this module
only adds durable storage + search.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .extract import chunk_document
from .graph import KnowledgeGraph
from .models import (
    Claim,
    Document,
    Entity,
    EvolutionEvent,
    IdentityLink,
    JudgeVerdict,
    Perspective,
    Provenance,
    TemporalScope,
)

_SCHEMA = """
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    text TEXT,
    metadata TEXT
);
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    label TEXT,
    type TEXT,
    sense TEXT,
    mention_count INTEGER,
    aliases TEXT,
    descriptions TEXT,
    contexts TEXT,
    dbpedia TEXT
);
CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    subject TEXT, predicate TEXT, object TEXT,
    subject_id TEXT, object_id TEXT,
    subject_sense TEXT, object_sense TEXT,
    context TEXT, status TEXT,
    viewpoint TEXT, stance TEXT, sentiment TEXT,
    when_ TEXT, valid_from TEXT, valid_to TEXT,
    document_id TEXT, chunk_id TEXT, quote TEXT,
    extractor TEXT, extracted_at TEXT, verdicts TEXT
);
CREATE TABLE identity_links (
    left_id TEXT, right_id TEXT, relation TEXT, context TEXT, note TEXT
);
CREATE TABLE evolution (
    entity_id TEXT, kind TEXT, when_ TEXT, note TEXT, related_id TEXT
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED, document_id UNINDEXED, text
);
"""


def save_sqlite(kg: KnowledgeGraph, path: str | Path) -> None:
    """Write the whole graph + an FTS5 chunk index to a fresh SQLite file."""
    path = Path(path)
    if path.exists():
        path.unlink()
    data = kg.to_dict()
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)

        # Documents keep full text (to_dict drops it); source it from the graph.
        for doc_id, doc in kg._documents.items():
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?)",
                (doc.id, doc.title, doc.text, json.dumps(doc.metadata, ensure_ascii=False)),
            )

        for e in data["entities"]:
            conn.execute(
                "INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    e["id"], e["label"], e["type"], e["sense"], e["mention_count"],
                    json.dumps(e["aliases"], ensure_ascii=False),
                    json.dumps(e["descriptions"], ensure_ascii=False),
                    json.dumps(e["contexts"], ensure_ascii=False),
                    json.dumps(e["dbpedia"], ensure_ascii=False) if e.get("dbpedia") else None,
                ),
            )

        for c in data["claims"]:
            prov = c.get("provenance") or {}
            persp, time = c["perspective"], c["time"]
            conn.execute(
                "INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    c["id"], c["subject"], c["predicate"], c["object"],
                    c["subject_id"], c["object_id"],
                    c["subject_sense"], c["object_sense"], c["context"], c["status"],
                    persp["viewpoint"], persp["stance"], persp["sentiment"],
                    time["when"], time["valid_from"], time["valid_to"],
                    prov.get("document_id"), prov.get("chunk_id"), prov.get("quote"),
                    prov.get("extractor"), prov.get("extracted_at"),
                    json.dumps(prov.get("verdicts", []), ensure_ascii=False) if prov else None,
                ),
            )

        for link in data["identity_links"]:
            conn.execute(
                "INSERT INTO identity_links VALUES (?,?,?,?,?)",
                (link["left_id"], link["right_id"], link["relation"], link["context"], link["note"]),
            )
        for ev in data["evolution"]:
            conn.execute(
                "INSERT INTO evolution VALUES (?,?,?,?,?)",
                (ev["entity_id"], ev["kind"], ev["when"], ev["note"], ev["related_id"]),
            )

        # Build the retrieval index from the same chunks the pipeline extracts.
        for doc in kg._documents.values():
            for chunk in chunk_document(doc):
                conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, document_id, text) VALUES (?,?,?)",
                    (chunk.id, chunk.document_id, chunk.text),
                )
        conn.commit()
    finally:
        conn.close()


def load_sqlite(path: str | Path) -> KnowledgeGraph:
    """Reconstruct a ``KnowledgeGraph`` from a file written by ``save_sqlite``."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        entities = [
            {
                "id": r["id"], "label": r["label"], "type": r["type"], "sense": r["sense"],
                "mention_count": r["mention_count"],
                "aliases": json.loads(r["aliases"]),
                "descriptions": json.loads(r["descriptions"]),
                "contexts": json.loads(r["contexts"]),
                "dbpedia": _load_dbpedia(r),
            }
            for r in conn.execute("SELECT * FROM entities")
        ]
        claims = []
        for r in conn.execute("SELECT * FROM claims"):
            prov = None
            if r["document_id"] is not None:
                prov = {
                    "document_id": r["document_id"], "chunk_id": r["chunk_id"],
                    "quote": r["quote"] or "", "extractor": r["extractor"] or "",
                    "extracted_at": r["extracted_at"] or "",
                    "verdicts": json.loads(r["verdicts"]) if r["verdicts"] else [],
                }
            claims.append(
                {
                    "id": r["id"], "subject": r["subject"], "predicate": r["predicate"],
                    "object": r["object"], "subject_id": r["subject_id"], "object_id": r["object_id"],
                    "subject_sense": r["subject_sense"], "object_sense": r["object_sense"],
                    "context": r["context"], "status": r["status"],
                    "perspective": {"viewpoint": r["viewpoint"], "stance": r["stance"], "sentiment": r["sentiment"]},
                    "time": {"when": r["when_"], "valid_from": r["valid_from"], "valid_to": r["valid_to"]},
                    "provenance": prov,
                }
            )
        documents = [
            {"id": r["id"], "title": r["title"], "text": r["text"], "metadata": json.loads(r["metadata"] or "{}")}
            for r in conn.execute("SELECT * FROM documents")
        ]
        identity_links = [dict(r) for r in conn.execute("SELECT * FROM identity_links")]
        evolution = [
            {"entity_id": r["entity_id"], "kind": r["kind"], "when": r["when_"],
             "note": r["note"], "related_id": r["related_id"]}
            for r in conn.execute("SELECT * FROM evolution")
        ]
    finally:
        conn.close()

    kg = KnowledgeGraph.from_dict(
        {
            "entities": entities, "claims": claims,
            "documents": [{"id": d["id"], "title": d["title"], "metadata": d["metadata"]} for d in documents],
            "identity_links": identity_links, "evolution": evolution,
        }
    )
    # Reattach full document text (from_dict drops it) so a re-save keeps chunks.
    for d in documents:
        kg._documents[d["id"]] = Document(id=d["id"], text=d["text"] or "", title=d["title"] or "", metadata=d["metadata"])
    return kg


_WORD = re.compile(r"\w+", re.UNICODE)


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression (quoted terms OR-ed)."""
    terms = _WORD.findall(query)
    return " OR ".join(f'"{t}"' for t in terms)


def retrieve(path: str | Path, query: str, k: int = 5) -> list[dict]:
    """BM25-ranked chunk hits for ``query`` (best first). The search index the
    on-the-fly / RAG idea needs; independent of the in-memory graph."""
    match = _fts_query(query)
    if not match:
        return []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT chunk_id, document_id, text, bm25(chunks_fts) AS score "
            "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (match, k),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"chunk_id": r["chunk_id"], "document_id": r["document_id"], "text": r["text"], "score": r["score"]}
        for r in rows
    ]


def _entities_has_dbpedia(conn) -> bool:
    return any(row[1] == "dbpedia" for row in conn.execute("PRAGMA table_info(entities)"))


def enrich_sqlite(path: str | Path, linker, relink: bool = False, concurrency: int = 4) -> int:
    """Link an existing graph.db's entities to DBpedia in place, adding the
    ``dbpedia`` column if the file predates it. Returns the number newly linked.

    ``linker`` is a ``DBpediaLinker`` (or any object with ``link_entities``);
    injecting one keeps this testable offline. Already-linked entities are left
    alone unless ``relink=True``.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        if not _entities_has_dbpedia(conn):
            conn.execute("ALTER TABLE entities ADD COLUMN dbpedia TEXT")
            conn.commit()
        rows = conn.execute("SELECT * FROM entities").fetchall()
        todo = [_entity_from_row(r) for r in rows if relink or not _load_dbpedia(r)]
        linked = linker.link_entities(todo, concurrency=concurrency)
        for entity in todo:
            if entity.dbpedia:
                conn.execute(
                    "UPDATE entities SET dbpedia=? WHERE id=?",
                    (json.dumps(entity.dbpedia, ensure_ascii=False), entity.id),
                )
        conn.commit()
        return linked
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Row reconstruction (shared by GraphStore)
# --------------------------------------------------------------------------

def _load_dbpedia(r) -> dict | None:
    """Parse the entity's ``dbpedia`` JSON, tolerating DBs written before the
    column existed (an old file's row simply has no such key)."""
    if "dbpedia" not in r.keys() or not r["dbpedia"]:
        return None
    try:
        return json.loads(r["dbpedia"])
    except (TypeError, json.JSONDecodeError):
        return None


def _entity_from_row(r) -> Entity:
    return Entity(
        id=r["id"], label=r["label"], type=r["type"], sense=r["sense"],
        mention_count=r["mention_count"],
        aliases=json.loads(r["aliases"]) if r["aliases"] else [],
        descriptions=json.loads(r["descriptions"]) if r["descriptions"] else [],
        contexts=json.loads(r["contexts"]) if r["contexts"] else [],
        dbpedia=_load_dbpedia(r),
    )


def _claim_from_row(r) -> tuple[Claim, str, str]:
    prov = None
    if r["document_id"] is not None:
        prov = Provenance(
            document_id=r["document_id"], chunk_id=r["chunk_id"] or "",
            quote=r["quote"] or "", extractor=r["extractor"] or "",
            extracted_at=r["extracted_at"] or "",
            verdicts=[JudgeVerdict(**v) for v in (json.loads(r["verdicts"]) if r["verdicts"] else [])],
        )
    claim = Claim(
        subject=r["subject"], predicate=r["predicate"], object=r["object"],
        perspective=Perspective(viewpoint=r["viewpoint"], stance=r["stance"], sentiment=r["sentiment"]),
        time=TemporalScope(when=r["when_"], valid_from=r["valid_from"], valid_to=r["valid_to"]),
        provenance=prov, status=r["status"], id=r["id"],
        context=r["context"] or "", subject_sense=r["subject_sense"] or "", object_sense=r["object_sense"] or "",
    )
    return claim, r["subject_id"], r["object_id"]


# --------------------------------------------------------------------------
# GraphStore: a SQLite-backed graph that streams during build and answers
# query views via SQL — the corpus need not be held in memory.
# --------------------------------------------------------------------------

class GraphStore:
    """SQLite-backed knowledge graph.

    * ``mode="build"`` — a write sink that **duck-types** ``KnowledgeGraph``'s
      build API (`add_document`, `add_claim`, `add_identity_link`,
      `finalize_entities`), so ``Pipeline.run(..., graph=store)`` streams claims
      straight to disk instead of accumulating a networkx graph in RAM.
    * ``mode="query"`` — mirrors ``KnowledgeGraph``'s query API for the CLI.

    Entities (few — one per label+sense) are cached in memory for entity-level
    views; claims (the bulk) stay on disk and are fetched per query. Corpus-wide
    ``conflicts`` runs as a SQL GROUP BY so it never materialises all claims.
    """

    def __init__(self, path, mode: str = "query"):
        self.path = str(path)
        if mode == "build":
            p = Path(self.path)
            if p.exists():
                p.unlink()
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        if mode == "build":
            self.conn.executescript(_SCHEMA)
        self._meta: KnowledgeGraph | None = None  # lazily-built entities-only view

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self.conn.close()

    # ----------------------------------------------- build sink (streaming)

    def add_document(self, document: Document) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?)",
            (document.id, document.title, document.text, json.dumps(document.metadata, ensure_ascii=False)),
        )
        for chunk in chunk_document(document):
            self.conn.execute(
                "INSERT INTO chunks_fts (chunk_id, document_id, text) VALUES (?,?,?)",
                (chunk.id, chunk.document_id, chunk.text),
            )
        self.conn.commit()  # per-document checkpoint

    def add_entity(self, entity: Entity) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO entities VALUES (?,?,?,?,?,?,?,?,?)",
            (
                entity.id, entity.label, entity.type, entity.sense, entity.mention_count,
                json.dumps(entity.aliases, ensure_ascii=False),
                json.dumps(entity.descriptions, ensure_ascii=False),
                json.dumps(entity.contexts, ensure_ascii=False),
                json.dumps(entity.dbpedia, ensure_ascii=False) if entity.dbpedia else None,
            ),
        )

    def add_claim(self, claim: Claim, subject: Entity, obj: Entity) -> None:
        self.add_entity(subject)
        self.add_entity(obj)
        p = claim.provenance
        self.conn.execute(
            "INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                claim.id, claim.subject, claim.predicate, claim.object,
                subject.id, obj.id, claim.subject_sense, claim.object_sense, claim.context, claim.status,
                claim.perspective.viewpoint, claim.perspective.stance, claim.perspective.sentiment,
                claim.time.when, claim.time.valid_from, claim.time.valid_to,
                p.document_id if p else None, p.chunk_id if p else None, p.quote if p else None,
                p.extractor if p else None, p.extracted_at if p else None,
                json.dumps([vars(v) for v in p.verdicts], ensure_ascii=False) if p else None,
            ),
        )

    def add_identity_link(self, link: IdentityLink) -> None:
        self.conn.execute(
            "INSERT INTO identity_links VALUES (?,?,?,?,?)",
            (link.left_id, link.right_id, link.relation, link.context, link.note),
        )

    def add_evolution(self, event: EvolutionEvent) -> None:
        self.conn.execute(
            "INSERT INTO evolution VALUES (?,?,?,?,?)",
            (event.entity_id, event.kind, event.when, event.note, event.related_id),
        )

    def finalize_entities(self, entities: list[Entity]) -> None:
        for entity in entities:
            self.add_entity(entity)  # write final mention counts
        self.conn.commit()

    def entity_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    def document_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def claim_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

    def documents(self) -> list[Document]:
        return [
            Document(id=r["id"], text=r["text"] or "", title=r["title"] or "",
                     metadata=json.loads(r["metadata"] or "{}"))
            for r in self.conn.execute("SELECT * FROM documents")
        ]

    def claims_in_document(self, document_id: str) -> list[Claim]:
        rows = self.conn.execute("SELECT * FROM claims WHERE document_id=?", (document_id,)).fetchall()
        return [_claim_from_row(r)[0] for r in rows]

    def dated_claims(self) -> list[Claim]:
        rows = self.conn.execute(
            "SELECT * FROM claims WHERE when_ IS NOT NULL OR valid_from IS NOT NULL"
        ).fetchall()
        return sorted((_claim_from_row(r)[0] for r in rows), key=lambda c: c.time.sort_key())

    # ----------------------------------------------------- query (SQL-backed)

    @property
    def _meta_kg(self) -> KnowledgeGraph:
        """Entities-only in-memory view (entities + identity links + evolution),
        for entity-level queries. Claims are never loaded here."""
        if self._meta is None:
            kg = KnowledgeGraph()
            for r in self.conn.execute("SELECT * FROM entities"):
                kg.add_entity(_entity_from_row(r))
            for r in self.conn.execute("SELECT * FROM identity_links"):
                kg.add_identity_link(IdentityLink(
                    left_id=r["left_id"], right_id=r["right_id"], relation=r["relation"],
                    context=r["context"], note=r["note"]))
            for r in self.conn.execute("SELECT * FROM evolution"):
                kg.add_evolution(EvolutionEvent(
                    entity_id=r["entity_id"], kind=r["kind"], when=r["when_"],
                    note=r["note"], related_id=r["related_id"]))
            self._meta = kg
        return self._meta

    @property
    def entities(self) -> list[Entity]:
        return self._meta_kg.entities

    def find_entity(self, name, sense=None):
        return self._meta_kg.find_entity(name, sense)

    def senses_of(self, label):
        return self._meta_kg.senses_of(label)

    def long_tail(self, head_fraction: float = 0.2):
        return self._meta_kg.long_tail(head_fraction)

    def near_identities(self, entity):
        return self._meta_kg.near_identities(entity)

    def evolution_of(self, entity):
        return self._meta_kg.evolution_of(entity)

    def _kg_from_claim_rows(self, rows) -> KnowledgeGraph:
        """Build a tiny KnowledgeGraph from some claim rows + their endpoint
        entities, so the existing graph.py view logic can be reused verbatim."""
        kg = KnowledgeGraph()
        parsed, needed = [], set()
        for r in rows:
            claim, sid, oid = _claim_from_row(r)
            parsed.append((claim, sid, oid))
            needed.update((sid, oid))
        ents: dict[str, Entity] = {}
        if needed:
            placeholders = ",".join("?" * len(needed))
            for er in self.conn.execute(f"SELECT * FROM entities WHERE id IN ({placeholders})", tuple(needed)):
                ents[er["id"]] = _entity_from_row(er)
        for claim, sid, oid in parsed:
            kg.add_claim(claim, ents.get(sid) or Entity(id=sid, label=sid), ents.get(oid) or Entity(id=oid, label=oid))
        return kg

    def _subgraph(self, entity_id: str) -> KnowledgeGraph:
        rows = self.conn.execute(
            "SELECT * FROM claims WHERE subject_id=? OR object_id=?", (entity_id, entity_id)
        ).fetchall()
        return self._kg_from_claim_rows(rows)

    def claims_about(self, entity):
        return self._subgraph(entity.id).claims_about(entity)

    def identity_profile(self, entity):
        return self._subgraph(entity.id).identity_profile(entity)

    def drift(self, entity):
        return self._subgraph(entity.id).drift(entity)

    def timeline(self, entity):
        return self._subgraph(entity.id).timeline(entity)

    def perspectives_on(self, entity):
        return self._subgraph(entity.id).perspectives_on(entity)

    def conflicts(self, entity=None):
        if entity is not None:
            return self._subgraph(entity.id).conflicts(entity)
        # Corpus-wide: find conflicting (subject, predicate) groups in SQL, then
        # fetch only those claims — never materialise the whole claim set.
        groups = []
        pairs = self.conn.execute(
            "SELECT subject_id, predicate FROM claims GROUP BY subject_id, predicate "
            "HAVING COUNT(DISTINCT object_id) > 1 OR COUNT(DISTINCT stance) > 1"
        ).fetchall()
        for sid, pred in pairs:
            rows = self.conn.execute(
                "SELECT * FROM claims WHERE subject_id=? AND predicate=?", (sid, pred)
            ).fetchall()
            groups.append([_claim_from_row(r)[0] for r in rows])
        return groups

    def provenance_of(self, claim_id: str) -> dict:
        r = self.conn.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
        if r is None:
            return {}
        claim, _, _ = _claim_from_row(r)
        return asdict(claim.provenance) if claim.provenance else {}

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        return retrieve(self.path, query, k)
