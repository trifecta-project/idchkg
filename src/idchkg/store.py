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
from pathlib import Path

from .extract import chunk_document
from .graph import KnowledgeGraph
from .models import Document

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
    contexts TEXT
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
                "INSERT INTO entities VALUES (?,?,?,?,?,?,?,?)",
                (
                    e["id"], e["label"], e["type"], e["sense"], e["mention_count"],
                    json.dumps(e["aliases"], ensure_ascii=False),
                    json.dumps(e["descriptions"], ensure_ascii=False),
                    json.dumps(e["contexts"], ensure_ascii=False),
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
