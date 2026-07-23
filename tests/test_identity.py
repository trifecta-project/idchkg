"""Tests for the three-challenge additions: identity (senses + near-identity),
change (drift + evolution), and the long tail (frequency)."""

from idchkg.graph import KnowledgeGraph
from idchkg.models import (
    CEASE,
    Claim,
    Document,
    EntityMention,
    EvolutionEvent,
    Perspective,
    Provenance,
    TemporalScope,
)
from idchkg.resolve import EntityResolver


# ---------------------------------------------------------------- resolver

def test_different_senses_become_distinct_entities_with_near_identity():
    r = EntityResolver()
    plant = r.add_mention(EntityMention("coffee", sense="plant"))
    bev = r.add_mention(EntityMention("coffee", sense="beverage"))
    assert plant.id != bev.id
    assert plant.sense == "plant" and bev.sense == "beverage"
    # one near-identity link connects the two senses of the same word
    links = r.identity_links
    assert len(links) == 1
    assert {links[0].left_id, links[0].right_id} == {plant.id, bev.id}
    assert links[0].relation == "near_identity"


def test_same_label_and_sense_merges_and_counts_mentions():
    r = EntityResolver()
    a = r.add_mention(EntityMention("coffee", sense="plant"))
    b = r.add_mention(EntityMention("Coffee", sense="plant"))  # case-insensitive
    assert a.id == b.id
    assert a.mention_count == 2
    assert "Coffee" in a.aliases


def test_resolve_uses_sole_sense_when_unspecified():
    r = EntityResolver()
    plant = r.add_mention(EntityMention("coffee", sense="plant"))
    assert r.resolve("coffee").id == plant.id  # only one sense on record


def test_resolve_creates_generic_node_when_sense_ambiguous():
    r = EntityResolver()
    r.add_mention(EntityMention("coffee", sense="plant"))
    r.add_mention(EntityMention("coffee", sense="beverage"))
    generic = r.resolve("coffee")  # ambiguous -> its own senseless node
    assert generic.sense == ""


def test_aliases_do_not_cross_senses():
    r = EntityResolver(aliases={"java": "coffee"})
    plant = r.add_mention(EntityMention("coffee", sense="plant"))
    java = r.add_mention(EntityMention("java", sense="plant"))
    assert plant.id == java.id  # alias + same sense merge


# ---------------------------------------------------------------- graph views

def _graph_with_senses():
    kg = KnowledgeGraph()
    kg.add_document(Document(id="d", text="", title="d"))
    r = EntityResolver()
    for m in [
        EntityMention("coffee", sense="plant"),
        EntityMention("coffee", sense="beverage"),
        EntityMention("Ethiopia", type="place"),
    ]:
        r.add_mention(m)

    def prov():
        return Provenance(document_id="d", chunk_id="d#c0", quote="q")

    plant = r.resolve("coffee", "plant")
    bev = r.resolve("coffee", "beverage")
    eth = r.resolve("Ethiopia")
    kg.add_claim(Claim("coffee", "native to", "Ethiopia",
                       Perspective("author"), TemporalScope(when="antiquity"),
                       provenance=prov(), subject_sense="plant"), plant, eth)
    kg.add_claim(Claim("coffee", "brewed from", "seeds",
                       Perspective("author"), TemporalScope(when="today"),
                       provenance=prov(), subject_sense="beverage",
                       context="the drink"), bev, plant)
    for link in r.identity_links:
        kg.add_identity_link(link)
    kg.finalize_entities(r.entities)
    return kg


def test_senses_of_returns_both_dimensions():
    kg = _graph_with_senses()
    senses = {e.sense for e in kg.senses_of("coffee")}
    assert senses == {"plant", "beverage"}


def test_identity_profile_lists_defining_properties():
    kg = _graph_with_senses()
    plant = kg.find_entity("coffee", sense="plant")
    profile = kg.identity_profile(plant)
    predicates = {p["predicate"] for p in profile["properties"]}
    assert "native to" in predicates


def test_near_identity_links_survive_json_round_trip():
    kg = _graph_with_senses()
    restored = KnowledgeGraph.from_dict(kg.to_dict())
    assert len(restored.identity_links) == 1
    plant = restored.find_entity("coffee", sense="plant")
    assert len(restored.near_identities(plant)) == 1


def test_drift_buckets_properties_by_time():
    kg = _graph_with_senses()
    bev = kg.find_entity("coffee", sense="beverage")
    buckets = kg.drift(bev)
    assert any(b["when"] == "today" for b in buckets)


def test_evolution_event_records_and_queries():
    kg = _graph_with_senses()
    plant = kg.find_entity("coffee", sense="plant")
    bev = kg.find_entity("coffee", sense="beverage")
    kg.add_evolution(EvolutionEvent(entity_id=plant.id, kind=CEASE,
                                    when="modern", note="sense split", related_id=bev.id))
    events = kg.evolution_of(plant)
    assert len(events) == 1 and events[0].kind == CEASE


def test_long_tail_partitions_by_mention_count():
    kg = _graph_with_senses()
    # bump one sense's mentions so it lands in the head
    plant = kg.find_entity("coffee", sense="plant")
    plant.mention_count = 10
    kg.add_entity(plant)
    split = kg.long_tail(head_fraction=0.34)
    head_labels = {(l, s) for l, s, _ in split["head"]}
    assert ("coffee", "plant") in head_labels


def test_claim_context_round_trips():
    kg = _graph_with_senses()
    restored = KnowledgeGraph.from_dict(kg.to_dict())
    contexts = {c.context for c in restored.claims}
    assert "the drink" in contexts


def test_entity_type_is_majority_vote_not_first_specific():
    # 3 'concept' mentions + 1 stray 'organization' -> majority keeps 'concept',
    # where the old first-specific-type rule flipped the whole entity.
    r = EntityResolver()
    for t in ("concept", "organization", "concept", "concept"):
        e = r.add_mention(EntityMention("coffee", sense="beverage", type=t))
    assert e.type == "concept"
    assert e.mention_count == 4

    stats = r.type_vote_stats()
    assert stats["reclassified"] == 1   # differs from the old-rule outcome
    assert stats["mixed"] == 1          # saw >1 mention type


def test_entity_type_promotes_when_specific_type_is_the_majority():
    r = EntityResolver()
    for t in ("concept", "person", "person"):
        e = r.add_mention(EntityMention("Okakura", type=t))
    assert e.type == "person"           # majority is 'person'
    # old rule also gave 'person' here, so no reclassification
    assert r.type_vote_stats()["reclassified"] == 0
