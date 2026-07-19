from idchkg.models import Claim, Perspective, Provenance, TemporalScope, slugify


def test_slugify():
    assert slugify("Coffee Plant") == "coffee-plant"
    assert slugify("  Éthíopia!! ") == "th-opia"  # non-ascii chars become separators
    assert slugify("") == "unnamed"


def test_claim_id_is_deterministic_and_distinguishes_perspective():
    prov = Provenance(document_id="d", chunk_id="d#c0")
    base = dict(subject="coffee", predicate="is", object="beverage", provenance=prov)
    a = Claim(**base, perspective=Perspective(viewpoint="author"))
    b = Claim(**base, perspective=Perspective(viewpoint="author"))
    c = Claim(**base, perspective=Perspective(viewpoint="colonial writers"))
    assert a.id == b.id          # same inputs -> same id
    assert a.id != c.id          # different viewpoint -> different id
    assert a.id.startswith("clm-")


def test_claim_round_trips_through_dict():
    claim = Claim(
        subject="coffee",
        predicate="native to",
        object="ethiopia",
        perspective=Perspective(viewpoint="author", stance="asserts", sentiment="neutral"),
        time=TemporalScope(when="15th century", valid_from="1400"),
        provenance=Provenance(document_id="coffee", chunk_id="coffee#c0", quote="native to..."),
    )
    restored = Claim.from_dict(claim.to_dict())
    assert restored.id == claim.id
    assert restored.time.valid_from == "1400"
    assert restored.provenance.quote == "native to..."


def test_temporal_sort_key_orders_dated_before_undated():
    dated = TemporalScope(valid_from="1400")
    expr = TemporalScope(when="later")
    empty = TemporalScope()
    keys = sorted([empty, expr, dated], key=lambda t: t.sort_key())
    assert keys[0] is dated and keys[-1] is empty
