from idchkg.graph import KnowledgeGraph
from idchkg.models import Claim, Document, Entity, Perspective, Provenance, TemporalScope


def _kg():
    kg = KnowledgeGraph()
    kg.add_document(Document(id="coffee", text="", title="Coffee"))
    coffee = Entity(id="ent-coffee", label="coffee", type="concept")
    beverage = Entity(id="ent-beverage", label="beverage")
    plant = Entity(id="ent-plant", label="coffee plant")
    colonialism = Entity(id="ent-colonialism", label="colonialism")
    for e in (coffee, beverage, plant, colonialism):
        kg.add_entity(e)

    prov = lambda: Provenance(document_id="coffee", chunk_id="coffee#c0", quote="q")
    kg.add_claim(
        Claim("coffee", "is a", "beverage", Perspective("author"), TemporalScope(), prov()),
        coffee, beverage,
    )
    kg.add_claim(
        Claim("coffee", "native to", "coffee plant",
              Perspective("author"), TemporalScope(when="antiquity", valid_from="0001"), prov()),
        coffee, plant,
    )
    # two viewpoints disagree about coffee's relation to colonialism
    kg.add_claim(
        Claim("coffee", "relates to", "colonialism",
              Perspective("colonial writers", "disputes", "positive"), TemporalScope(), prov()),
        coffee, colonialism,
    )
    kg.add_claim(
        Claim("coffee", "relates to", "colonialism",
              Perspective("historians", "asserts", "negative"), TemporalScope(), prov()),
        coffee, colonialism,
    )
    return kg, coffee


def test_perspectives_group_by_viewpoint():
    kg, coffee = _kg()
    grouped = kg.perspectives_on(coffee)
    assert set(grouped) == {"author", "colonial writers", "historians"}


def test_timeline_orders_dated_first():
    kg, coffee = _kg()
    timeline = kg.timeline(coffee)
    assert timeline[0].time.valid_from == "0001"  # the only dated claim leads


def test_conflicts_detects_stance_disagreement():
    kg, coffee = _kg()
    groups = kg.conflicts(coffee)
    assert len(groups) == 1
    stances = {c.perspective.stance for c in groups[0]}
    assert stances == {"disputes", "asserts"}


def test_json_round_trip_preserves_claims_and_views():
    kg, coffee = _kg()
    restored = KnowledgeGraph.from_dict(kg.to_dict())
    r_coffee = restored.find_entity("coffee")
    assert r_coffee is not None
    assert len(restored.conflicts(r_coffee)) == 1
    assert set(restored.perspectives_on(r_coffee)) == {"author", "colonial writers", "historians"}


def test_turtle_export_contains_reified_statements(tmp_path):
    kg, _ = _kg()
    path = tmp_path / "g.ttl"
    kg.save_turtle(path)
    text = path.read_text()
    assert "rdf:Statement" in text
    assert "idc:viewpoint" in text and "idc:stance" in text


def test_graphml_export_is_writable(tmp_path):
    kg, _ = _kg()
    path = tmp_path / "g.graphml"
    kg.save_graphml(path)
    assert path.exists() and path.stat().st_size > 0
