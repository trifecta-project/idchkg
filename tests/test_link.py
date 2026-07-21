"""Offline tests for DBpedia entity linking.

The linker's HTTP methods (`annotate`, `fetch_dbo_type`) are stubbed so nothing
touches the network — we exercise the selection/mapping logic and the wiring
into the pipeline, store enrichment, and site.
"""

import logging

from idchkg.link import DBpediaLinker, _ConnError
from idchkg.models import Entity


def _resource(uri, surface, score, types=""):
    return {"@URI": uri, "@surfaceForm": surface, "@similarityScore": str(score), "@types": types}


class StubLinker(DBpediaLinker):
    """A DBpediaLinker with the two network calls replaced by canned data."""

    def __init__(self, resources_by_text=None, dbo_types_by_uri=None, confidence=0.5):
        super().__init__(confidence=confidence)
        self._resources = resources_by_text or {}
        self._dbo = dbo_types_by_uri or {}

    def annotate(self, text):
        # Return the first canned resource set whose key appears in the text.
        for key, resources in self._resources.items():
            if key.casefold() in text.casefold():
                return resources
        return []

    def fetch_dbo_type(self, uri):
        return self._dbo.get(uri, [])


COFFEE_URI = "http://dbpedia.org/resource/Coffee"


def test_link_entity_picks_surface_match_and_maps_types():
    linker = StubLinker(
        resources_by_text={"coffee": [
            _resource(COFFEE_URI, "coffee", 0.99, types="DBpedia:Beverage,Schema:Food,Wikidata:Q8386"),
            _resource("http://dbpedia.org/resource/Bean", "bean", 0.4),
        ]},
        dbo_types_by_uri={COFFEE_URI: ["http://dbpedia.org/resource/Espresso"]},
    )
    link = linker.link_entity(Entity(id="e", label="coffee", descriptions=["a beverage"]))
    assert link["uri"] == COFFEE_URI
    assert link["surfaceForm"] == "coffee"
    assert link["score"] == 0.99
    # @types filtered to DBpedia ontology classes, expanded to dbo: URIs.
    assert link["types"] == ["http://dbpedia.org/ontology/Beverage"]
    # dbo:type property fetched separately.
    assert link["dbo_type"] == ["http://dbpedia.org/resource/Espresso"]


def test_link_entity_below_threshold_returns_none():
    linker = StubLinker(
        resources_by_text={"coffee": [_resource(COFFEE_URI, "coffee", 0.2)]},
        confidence=0.5,
    )
    assert linker.link_entity(Entity(id="e", label="coffee")) is None


def test_link_entity_no_resources_returns_none():
    assert StubLinker().link_entity(Entity(id="e", label="nothingmatches")) is None


def test_link_entities_sets_field_and_counts():
    linker = StubLinker(resources_by_text={"coffee": [_resource(COFFEE_URI, "coffee", 0.9)]})
    a = Entity(id="a", label="coffee")
    b = Entity(id="b", label="unmatched")
    n = linker.link_entities([a, b], concurrency=1)
    assert n == 1
    assert a.dbpedia["uri"] == COFFEE_URI
    assert b.dbpedia is None


class _DownLinker(DBpediaLinker):
    """Every request fails to connect (simulates a down Spotlight endpoint)."""

    def _get_json(self, url, headers):
        raise _ConnError("connection refused")


def test_warns_when_requests_fail_to_connect(caplog):
    linker = _DownLinker()
    ents = [Entity(id="a", label="coffee"), Entity(id="b", label="tea")]
    with caplog.at_level(logging.WARNING, logger="idchkg.link"):
        n = linker.link_entities(ents, concurrency=1)
    assert n == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("could not connect to the Spotlight server" in m for m in warnings)


def test_no_warning_when_endpoint_up_but_nothing_matches(caplog):
    # StubLinker reaches the "server" fine; it just links some and not others.
    linker = StubLinker(resources_by_text={"coffee": [_resource(COFFEE_URI, "coffee", 0.9)]})
    ents = [Entity(id="a", label="coffee"), Entity(id="b", label="unmatched")]
    with caplog.at_level(logging.WARNING, logger="idchkg.link"):
        n = linker.link_entities(ents, concurrency=1)
    assert n == 1
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_context_text_includes_descriptions_and_contexts():
    e = Entity(id="e", label="coffee", descriptions=["the plant"], contexts=["grown in Brazil"])
    text = DBpediaLinker._context_text(e)
    assert "coffee" in text and "the plant" in text and "grown in Brazil" in text
