"""Offline tests for Wikidata linking: HTTP is replaced by a fake ``fetch`` that
answers wbsearchentities / wbgetentities from canned JSON."""

import json
import sqlite3
import urllib.parse

from idchkg.graph import KnowledgeGraph
from idchkg.models import Entity
from idchkg.store import enrich_wikidata_sqlite, save_sqlite
from idchkg.wikidata import WikidataLinker, summarize_item


def _time(year):
    sign = "-" if year < 0 else "+"
    return {"time": f"{sign}{abs(year):04d}-00-00T00:00:00Z"}


def _item(qid, label, desc, p31=(), coord=None, start=None, end=None, aliases=(), sitelinks=0):
    claims = {"P31": [{"mainsnak": {"datavalue": {"value": {"id": c}}}} for c in p31]}
    if coord:
        claims["P625"] = [{"mainsnak": {"datavalue": {"value": {
            "latitude": coord[0], "longitude": coord[1], "globe": "http://www.wikidata.org/entity/Q2"}}}}]
    if start is not None:
        claims["P571"] = [{"mainsnak": {"datavalue": {"value": _time(start)}}}]
    if end is not None:
        claims["P576"] = [{"mainsnak": {"datavalue": {"value": _time(end)}}}]
    return {
        "labels": {"en": {"value": label}}, "descriptions": {"en": {"value": desc}},
        "aliases": {"en": [{"value": a} for a in aliases]}, "claims": claims,
        "sitelinks": {f"w{i}": {} for i in range(sitelinks)},
    }


CLASSES = {"Q5": "human", "Q23442": "island", "Q3624078": "sovereign state",
           "Q3024240": "historical country", "Q1549591": "big city", "Q515": "city",
           "Q11424": "film", "Q5084": "hamlet"}

ITEMS = {
    # Java
    "Q3757": _item("Q3757", "Java", "island of Indonesia", ["Q23442"], (-7.5, 110.0), sitelinks=150),
    "Q251": _item("Q251", "Java", "object-oriented programming language", [], sitelinks=120),
    "Q99": _item("Q99", "Java", "hamlet in Belgium", ["Q5084"], (50.5, 5.3), sitelinks=1),
    "Q188161": _item("Q188161", "Dutch East Indies", "Dutch colony in Southeast Asia (1800–1949)",
                     ["Q3024240"], (-6.2, 106.8), start=1800, end=1949, aliases=["Java colony"],
                     sitelinks=80),
    # Batavia
    "Q1199713": _item("Q1199713", "Batavia", "capital of the Dutch East Indies (1619–1949), now Jakarta",
                      ["Q515"], (-6.13, 106.81), start=1619, end=1949, sitelinks=40),
    "Q1380988": _item("Q1380988", "Batavia", "city in New York, USA", ["Q515"], (43.0, -78.2), sitelinks=30),
    # Person who must never win for a place
    "Q7": _item("Q7", "Smith", "English person", ["Q5"], sitelinks=10),
}

SEARCH = {
    "java": ["Q251", "Q3757", "Q99"],
    "java colony": ["Q188161"],
    "batavia": ["Q1380988", "Q1199713"],
    "smith": ["Q7"],
}


class FakeWikidata:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, url):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.calls.append(q)
        if self.fail:
            raise OSError("network down")
        if q["action"] == "wbsearchentities":
            ids = SEARCH.get(q["search"].casefold(), [])
            return {"search": [{"id": i, "label": ITEMS[i]["labels"]["en"]["value"]} for i in ids]}
        if q["action"] == "wbgetentities":
            ids = q["ids"].split("|")
            assert len(ids) <= 50
            if q["props"] == "labels":
                return {"entities": {i: {"labels": {"en": {"value": CLASSES.get(i, "")}}} for i in ids}}
            return {"entities": {i: ITEMS.get(i, {"missing": ""}) for i in ids}}
        raise AssertionError(q)


def _place(eid, label, sense="", desc=(), ctx=(), aliases=()):
    return Entity(id=eid, label=label, type="place", sense=sense, descriptions=list(desc),
                  contexts=list(ctx), aliases=list(aliases), mention_count=3)


def test_summarize_item_extracts_coord_and_years():
    s = summarize_item("Q1199713", ITEMS["Q1199713"])
    assert s["coord"] == [-6.13, 106.81]
    assert (s["start"], s["end"]) == (1619, 1949)
    assert s["p31"] == ["Q515"]


def test_place_prefers_island_over_programming_language_and_hamlet():
    linker = WikidataLinker(fetch=FakeWikidata())
    e = _place("e1", "Java", "island", desc=["An island where coffee was planted in 1696"])
    assert linker.link_entities([e], concurrency=1) == 1
    assert e.wikidata["qid"] == "Q3757"
    assert e.wikidata["coord"] == [-7.5, 110.0]
    assert "island" in e.wikidata["types"]


def test_senses_link_independently():
    """Identity: two senses of 'Java' may land on different items."""
    linker = WikidataLinker(fetch=FakeWikidata())
    island = _place("e1", "Java", "island", desc=["Indonesian island"])
    colony = _place("e2", "Java", "colony", desc=["Dutch colony, historically governed from Batavia"],
                    ctx=["1830"], aliases=["Java colony"])
    linker.link_entities([island, colony], concurrency=1)
    assert island.wikidata["qid"] == "Q3757"
    assert colony.wikidata["qid"] == "Q188161"


def test_historical_context_prefers_historical_item():
    linker = WikidataLinker(fetch=FakeWikidata())
    e = _place("e1", "Batavia", desc=["capital of the Dutch East Indies"], ctx=["1696", "colonial"])
    linker.link_entities([e], concurrency=1)
    assert e.wikidata["qid"] == "Q1199713"
    assert (e.wikidata["start"], e.wikidata["end"]) == (1619, 1949)


def test_place_never_links_to_person():
    linker = WikidataLinker(fetch=FakeWikidata())
    e = _place("e1", "Smith", desc=["a place"])
    assert linker.link_entities([e], concurrency=1) == 0
    assert e.wikidata is None


def test_obscure_homonym_without_evidence_is_rejected():
    linker = WikidataLinker(fetch=FakeWikidata())
    # Searching "java" with no island/context cues: the hamlet must not win by name alone.
    e = _place("e1", "Java")
    linker.link_entities([e], concurrency=1)
    assert e.wikidata is None or e.wikidata["qid"] != "Q99"


def test_network_failure_leaves_unlinked_and_does_not_raise():
    fake = FakeWikidata(fail=True)
    linker = WikidataLinker(fetch=fake)
    e = _place("e1", "Java", "island")
    assert linker.link_entities([e], concurrency=1) == 0
    assert e.wikidata is None
    assert linker.failures >= 1


def test_cache_makes_rerun_free(tmp_path):
    cache = tmp_path / "wd.json"
    fake = FakeWikidata()
    WikidataLinker(fetch=fake, cache_path=cache).link_entities([_place("e1", "Java", "island")], 1)
    assert fake.calls and cache.exists()
    fake2 = FakeWikidata()
    e = _place("e1", "Java", "island")
    WikidataLinker(fetch=fake2, cache_path=cache).link_entities([e], 1)
    assert fake2.calls == []
    assert e.wikidata["qid"] == "Q3757"


def _make_db(path, with_column_absent=True):
    kg = KnowledgeGraph()
    kg.add_entity(_place("p1", "Java", "island", desc=["Indonesian island"]))
    kg.add_entity(_place("p2", "Batavia", desc=["capital of the Dutch East Indies"], ctx=["1700"]))
    kg.add_entity(Entity(id="c1", label="Java", type="concept", sense="coffee"))
    save_sqlite(kg, path)


def test_enrich_wikidata_migrates_column_and_filters_types(tmp_path):
    db = tmp_path / "graph.db"
    _make_db(db)
    cols = [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(entities)")]
    assert "wikidata" not in cols
    # Reads tolerate the column's absence.
    from idchkg.store import load_sqlite
    assert load_sqlite(db)._entities["p1"].wikidata is None

    n = enrich_wikidata_sqlite(db, WikidataLinker(fetch=FakeWikidata()), types=["place"], concurrency=1)
    assert n == 2
    rows = dict(sqlite3.connect(db).execute("SELECT id, wikidata FROM entities"))
    assert json.loads(rows["p1"])["qid"] == "Q3757"
    assert json.loads(rows["p2"])["qid"] == "Q1199713"
    assert rows["c1"] is None  # type filter respected

    # Round-trips through the normal reader.
    kg = load_sqlite(db)
    assert kg._entities["p1"].wikidata["qid"] == "Q3757"


def test_enrich_wikidata_limit_links_head_first(tmp_path):
    db = tmp_path / "graph.db"
    _make_db(db)
    con = sqlite3.connect(db)
    con.execute("UPDATE entities SET mention_count=50 WHERE id='p2'")
    con.commit()
    enrich_wikidata_sqlite(db, WikidataLinker(fetch=FakeWikidata()), types=["place"], limit=1, concurrency=1)
    rows = dict(sqlite3.connect(db).execute("SELECT id, wikidata FROM entities"))
    assert rows["p2"] and not rows["p1"]


def test_user_agent_is_generic():
    ua = WikidataLinker().user_agent
    assert ua.startswith("idchkg/") and "@" not in ua
