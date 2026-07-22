"""Offline tests for the SQLite store: save/load parity + BM25 retrieval.

Deterministic — no LLM, no network. Builds a small graph by hand (via the
resolver, so near-identity links are exercised too), round-trips it through
SQLite, and checks the FTS5 retrieval index.
"""

import shutil

from idchkg.graph import KnowledgeGraph
from idchkg.models import Claim, Document, Entity, Perspective, Provenance, TemporalScope, JudgeVerdict
from idchkg.resolve import EntityResolver
from idchkg.store import GraphStore, load_sqlite, retrieve, save_sqlite


def _build_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_document(Document(
        id="coffee",
        title="coffee",
        text="Coffee is a brewed beverage prepared from roasted seeds.\n\n"
             "Coffee also names the plant, genus Coffea, cultivated for those seeds.",
    ))
    kg.add_document(Document(
        id="sugar",
        title="sugar",
        text="Sugar was a major commodity of the Atlantic trade and plantation economy.",
    ))

    resolver = EntityResolver()
    from idchkg.models import EntityMention
    plant = resolver.add_mention(EntityMention(name="coffee", sense="plant"))
    plant = resolver.add_mention(EntityMention(name="coffee", sense="plant"))  # count=2
    resolver.add_mention(EntityMention(name="coffee", sense="beverage"))       # -> near-identity link
    resolver.add_mention(EntityMention(name="seeds"))

    claim = Claim(
        "coffee", "cultivated for", "seeds",
        perspective=Perspective("author", "asserts", "neutral"),
        time=TemporalScope(when="19th century"),
        context="botanical", subject_sense="plant",
        provenance=Provenance("coffee", "coffee#c0", "cultivated for those seeds", "mock",
                              verdicts=[JudgeVerdict("groundedness", 0.9, True, "ok", "mock")]),
    )
    subj = resolver.resolve("coffee", "plant")
    obj = resolver.resolve("seeds")
    kg.add_claim(claim, subj, obj)

    for link in resolver.identity_links:
        kg.add_identity_link(link)
    kg.finalize_entities(resolver.entities)
    return kg


def test_save_load_roundtrip_parity(tmp_path):
    kg = _build_graph()
    path = tmp_path / "g.db"
    save_sqlite(kg, path)
    kg2 = load_sqlite(path)

    assert len(kg2.entities) == len(kg.entities)
    assert len(kg2.claims) == len(kg.claims)
    # Challenge 1: both senses of "coffee" survive, with mention counts.
    senses = {e.sense: e.mention_count for e in kg2.senses_of("coffee")}
    assert senses == {"plant": 2, "beverage": 1}
    # near-identity link between the two senses round-trips
    assert len(kg2.identity_links) == len(kg.identity_links) == 1


def test_claim_qualifiers_and_provenance_roundtrip(tmp_path):
    kg = _build_graph()
    path = tmp_path / "g.db"
    save_sqlite(kg, path)
    claim = load_sqlite(path).claims[0]

    assert (claim.subject, claim.predicate, claim.object) == ("coffee", "cultivated for", "seeds")
    assert claim.time.when == "19th century"
    assert claim.context == "botanical"
    assert claim.subject_sense == "plant"
    assert claim.perspective.stance == "asserts"
    assert claim.provenance.quote == "cultivated for those seeds"
    assert len(claim.provenance.verdicts) == 1
    assert claim.provenance.verdicts[0].criterion == "groundedness"


def test_retrieve_ranks_relevant_chunk_first(tmp_path):
    kg = _build_graph()
    path = tmp_path / "g.db"
    save_sqlite(kg, path)

    top = retrieve(path, "Atlantic trade commodity", k=3)
    assert top, "expected at least one hit"
    assert top[0]["document_id"] == "sugar"

    top = retrieve(path, "plant genus Coffea", k=3)
    assert top[0]["document_id"] == "coffee"

    assert retrieve(path, "", k=3) == []  # empty query -> no hits, no crash


def test_portability_reopen_copied_file(tmp_path):
    kg = _build_graph()
    path = tmp_path / "g.db"
    save_sqlite(kg, path)

    copied = tmp_path / "copy.db"
    shutil.copyfile(path, copied)  # single-file portability
    kg2 = load_sqlite(copied)
    assert len(kg2.claims) == 1
    assert retrieve(copied, "sugar", k=1)[0]["document_id"] == "sugar"


# --- DB-backed GraphStore (streaming build + SQL-backed queries) -----------

def _replay_into_store(kg: KnowledgeGraph, path) -> GraphStore:
    """Stream an in-memory graph into a GraphStore via the build sink API."""
    store = GraphStore(path, mode="build")
    for doc in kg._documents.values():
        store.add_document(doc)
    for claim in kg.claims:
        sid, oid = kg._entity_of_claim[claim.id]
        store.add_claim(claim, kg._entities[sid], kg._entities[oid])
    for link in kg.identity_links:
        store.add_identity_link(link)
    store.finalize_entities(kg.entities)
    store.close()
    return GraphStore(path, mode="query")


def _props(view, entity):
    return sorted((p["role"], p["predicate"], p["other"]) for p in view.identity_profile(entity)["properties"])


def test_graphstore_parity_with_in_memory(tmp_path):
    kg = _build_graph()
    store = _replay_into_store(kg, tmp_path / "gs.db")

    assert store.entity_count() == kg.entity_count()
    assert store.document_count() == kg.document_count()
    # Challenge 1: senses + near-identity, computed from SQL, match the in-memory graph.
    assert {(e.sense, e.mention_count) for e in store.senses_of("coffee")} == \
           {(e.sense, e.mention_count) for e in kg.senses_of("coffee")}
    assert store.long_tail() == kg.long_tail()

    e_mem = kg.find_entity("coffee", "plant")
    e_db = store.find_entity("coffee", "plant")
    assert e_db is not None and e_db.id == e_mem.id
    assert _props(store, e_db) == _props(kg, e_mem)           # per-entity subgraph view
    assert len(store.near_identities(e_db)) == len(kg.near_identities(e_mem))

    cid = kg.claims[0].id
    assert store.provenance_of(cid)["quote"] == kg.provenance_of(cid)["quote"]
    assert store.retrieve("Atlantic trade", 1)[0]["document_id"] == "sugar"
    store.close()


def test_graphstore_conflicts_via_sql(tmp_path):
    store = GraphStore(tmp_path / "c.db", mode="build")
    coffee = Entity(id="ent-coffee", label="coffee", mention_count=2)
    drink = Entity(id="ent-drink", label="drink")
    commodity = Entity(id="ent-commodity", label="commodity")
    persp = Perspective("author", "asserts", "neutral")
    store.add_document(Document(id="d", text="Coffee is a drink and a commodity."))
    store.add_claim(Claim("coffee", "is a", "drink", perspective=persp,
                          provenance=Provenance("d", "d#c0", "a drink")), coffee, drink)
    store.add_claim(Claim("coffee", "is a", "commodity", perspective=persp,
                          provenance=Provenance("d", "d#c0", "a commodity")), coffee, commodity)
    store.finalize_entities([coffee, drink, commodity])

    groups = store.conflicts()  # corpus-wide, via SQL GROUP BY
    assert len(groups) == 1
    assert {c.object for c in groups[0]} == {"drink", "commodity"}
    store.close()


# --- merge_sqlite: union two graph dbs (keep both, dedupe by deterministic id) --

def _build_other_graph() -> KnowledgeGraph:
    """A second graph sharing the coffee/plant entity but a new doc + claim, so a
    merge must union documents/claims and *sum* the shared entity's count."""
    from idchkg.models import EntityMention

    kg = KnowledgeGraph()
    kg.add_document(Document(
        id="tea", title="tea",
        text="Tea and coffee are plants cultivated in large plantations.",
    ))
    resolver = EntityResolver()
    resolver.add_mention(EntityMention(name="coffee", sense="plant"))  # same id as dst's coffee/plant
    resolver.add_mention(EntityMention(name="plantation"))
    claim = Claim(
        "coffee", "grown in", "plantation",
        perspective=Perspective("author", "asserts", "neutral"),
        context="agriculture", subject_sense="plant",
        provenance=Provenance("tea", "tea#c0", "cultivated in large plantations", "mock"),
    )
    kg.add_claim(claim, resolver.resolve("coffee", "plant"), resolver.resolve("plantation"))
    kg.finalize_entities(resolver.entities)
    return kg


def test_merge_sqlite_unions_and_sums_shared_entities(tmp_path):
    from idchkg.store import merge_sqlite

    dst = tmp_path / "dst.db"
    src = tmp_path / "src.db"
    save_sqlite(_build_graph(), dst)        # coffee+sugar docs; coffee/plant count=2
    save_sqlite(_build_other_graph(), src)  # tea doc; coffee/plant count=1

    merge_sqlite(src, dst)                   # union src INTO dst, in place
    merged = load_sqlite(dst)

    # Documents from both graphs survive (nothing on the dst side is lost).
    assert {d.id for d in merged._documents.values()} == {"coffee", "sugar", "tea"}
    # Distinct claims (different content hashes) are both kept.
    preds = {(c.subject, c.predicate, c.object) for c in merged.claims}
    assert ("coffee", "cultivated for", "seeds") in preds
    assert ("coffee", "grown in", "plantation") in preds
    # Shared coffee/plant entity: mention_count summed 2+1; beverage sense intact.
    senses = {e.sense: e.mention_count for e in merged.senses_of("coffee")}
    assert senses == {"plant": 3, "beverage": 1}
    # New entity from src is present; near-identity link not duplicated.
    assert merged.find_entity("plantation") is not None
    assert len(merged.identity_links) == 1
    # FTS index carries the new doc's chunks (deduped by chunk_id, no crash).
    # "tea" is unique to the merged-in document.
    assert retrieve(dst, "tea", k=1)[0]["document_id"] == "tea"


_DBP = {"uri": "http://dbpedia.org/resource/Coffee", "score": 0.98,
        "surfaceForm": "coffee", "types": ["http://dbpedia.org/ontology/Beverage"],
        "dbo_type": ["http://dbpedia.org/resource/Espresso"]}


def test_dbpedia_field_roundtrips_json_and_sqlite(tmp_path):
    kg = KnowledgeGraph()
    kg.add_document(Document(id="d", text="Coffee is a beverage."))
    kg.add_entity(Entity(id="ent-coffee", label="coffee", dbpedia=_DBP))
    kg.add_entity(Entity(id="ent-plain", label="plain"))  # unlinked -> stays None

    path = tmp_path / "g.db"
    save_sqlite(kg, path)
    kg2 = load_sqlite(path)
    by_id = {e.id: e for e in kg2.entities}
    assert by_id["ent-coffee"].dbpedia == _DBP
    assert by_id["ent-plain"].dbpedia is None

    # JSON export path carries it too.
    jpath = tmp_path / "g.json"
    kg.save_json(jpath)
    kg3 = KnowledgeGraph.load_json(jpath)
    assert {e.id: e.dbpedia for e in kg3.entities}["ent-coffee"] == _DBP


class _FakeLinker:
    """Links any entity whose label is 'coffee'; ignores the rest."""

    def link_entities(self, entities, concurrency=4):
        n = 0
        for e in entities:
            if e.label.casefold() == "coffee":
                e.dbpedia = _DBP
                n += 1
        return n


def test_enrich_sqlite_adds_column_to_old_db_and_populates(tmp_path):
    from idchkg.store import enrich_sqlite

    # Simulate a pre-feature DB: build normally, then drop the dbpedia column
    # so the file looks like one written before the column existed.
    kg = KnowledgeGraph()
    kg.add_document(Document(id="d", text="Coffee is a beverage."))
    kg.add_entity(Entity(id="ent-coffee", label="coffee"))
    kg.add_entity(Entity(id="ent-sugar", label="sugar"))
    path = tmp_path / "old.db"
    save_sqlite(kg, path)

    import sqlite3
    conn = sqlite3.connect(str(path))
    conn.execute("ALTER TABLE entities DROP COLUMN dbpedia")
    conn.commit()
    conn.close()

    # Querying the column-less DB must not error (tolerant reader).
    kg_old = load_sqlite(path)
    assert all(e.dbpedia is None for e in kg_old.entities)

    # Enrich in place: adds the column back and populates the matching entity.
    n = enrich_sqlite(path, _FakeLinker())
    assert n == 1
    by_id = {e.id: e for e in load_sqlite(path).entities}
    assert by_id["ent-coffee"].dbpedia == _DBP
    assert by_id["ent-sugar"].dbpedia is None
