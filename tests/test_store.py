"""Offline tests for the SQLite store: save/load parity + BM25 retrieval.

Deterministic — no LLM, no network. Builds a small graph by hand (via the
resolver, so near-identity links are exercised too), round-trips it through
SQLite, and checks the FTS5 retrieval index.
"""

import shutil

from idchkg.graph import KnowledgeGraph
from idchkg.models import Claim, Document, Perspective, Provenance, TemporalScope, JudgeVerdict
from idchkg.resolve import EntityResolver
from idchkg.store import load_sqlite, retrieve, save_sqlite


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
