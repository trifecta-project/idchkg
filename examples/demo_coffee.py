"""Offline demo of the full pipeline on the paper's coffee example.

Runs with no API key: a scripted MockLLM plays the extractor and the judge
panel, so you can see an "unflattened" graph organised around van Erp's three
challenges (identity, change, long tail) without any model calls. Swap MockLLM
for a real backend (see README) to run it for real.

    python examples/demo_coffee.py
"""

from idchkg.llm.mock import MockLLM
from idchkg.models import CEASE, GAINS, EvolutionEvent
from idchkg.models import Document
from idchkg.pipeline import Pipeline, PipelineConfig


def entity(name, etype, sense="", desc=""):
    return {"name": name, "type": etype, "sense": sense, "description": desc, "context": ""}


def claim(subject, predicate, obj, subject_sense="", object_sense="", context="",
          viewpoint="author", stance="asserts", sentiment="neutral", when=None, quote="q"):
    return {
        "subject": subject, "subject_sense": subject_sense,
        "predicate": predicate, "object": obj, "object_sense": object_sense,
        "context": context,
        "perspective": {"viewpoint": viewpoint, "stance": stance, "sentiment": sentiment},
        "time": {"when": when, "valid_from": None, "valid_to": None},
        "quote": quote,
    }


def passing():
    return {"score": 1.0, "passed": True, "feedback": ""}


# One chunk: the concept "coffee" appears in several distinct senses (identity),
# claims are scoped in time (change), and each is judged before entering.
extraction = {
    "entities": [
        entity("coffee", "concept", "plant", "the shrub of genus Coffea"),
        entity("coffee", "concept", "beverage", "the brewed drink"),
        entity("coffee", "concept", "commodity", "colonial trade good"),
        entity("Ethiopia", "place"),
        entity("Yemen", "place"),
        entity("colonialism", "concept"),
    ],
    "claims": [
        claim("coffee", "native to", "Ethiopia",
              subject_sense="plant", when="before 15th century"),
        claim("coffee", "cultivated in", "Yemen",
              subject_sense="plant", when="15th century"),
        claim("coffee", "brewed from", "coffee",
              subject_sense="beverage", object_sense="plant",
              context="the drink is made from the plant's seeds"),
        # two viewpoints, same subject+predicate, opposing stance (a conflict)
        claim("coffee", "was", "civilising luxury",
              subject_sense="commodity", context="17th-century coffeehouses",
              viewpoint="colonial writers", stance="asserts", sentiment="positive",
              when="17th century"),
        claim("coffee", "was", "product of forced labour",
              subject_sense="commodity", context="colonial plantations",
              viewpoint="historians", stance="disputes", sentiment="negative",
              when="17th century"),
        claim("coffee", "inseparable from", "colonialism",
              subject_sense="commodity",
              viewpoint="historians", stance="asserts", sentiment="negative"),
    ],
}

mock = MockLLM().queue(extraction, *([passing()] * (2 * len(extraction["claims"]))))

doc = Document(id="coffee", text=open("examples/corpus/coffee.txt").read(), title="Coffee")
pipeline = Pipeline(PipelineConfig(backend="mock"), extractor_client=mock, judge_client=mock)
result = pipeline.run([doc])
kg = result.graph

# Record an identity-evolution event by hand (Challenge 2): the plant/beverage
# senses split off a distinct colonial-commodity identity in the colonial era.
commodity = kg.find_entity("coffee", sense="commodity")
plant = kg.find_entity("coffee", sense="plant")
kg.add_evolution(EvolutionEvent(
    entity_id=commodity.id, kind=GAINS, when="17th century",
    note="coffee acquires a colonial-commodity identity", related_id=plant.id,
))

print("stats:", result.stats())

print("\n== Challenge 1 — IDENTITY: the senses of 'coffee' ==")
for e in kg.senses_of("coffee"):
    prof = kg.identity_profile(e)
    print(f"  coffee/{e.sense or 'unspecified'}  (mentions: {prof['mention_count']})")
    for p in prof["properties"]:
        ctx = f"  ⟨{p['context']}⟩" if p["context"] else ""
        print(f"      {p['role']}: {p['predicate']} {p['other']}{ctx}")
print("  near-identity links between senses:")
seen = set()
for link in kg.identity_links:
    if link.left_id not in seen or link.right_id not in seen:
        print(f"      {link.left_id} ~ {link.right_id}  ({link.context})")

print("\n== Challenge 2 — CHANGE: drift of coffee/commodity over time ==")
for bucket in kg.drift(commodity):
    print(f"  [{bucket['when']}] " + "; ".join(bucket["properties"]))
for ev in kg.evolution_of(commodity):
    print(f"  evolution: {ev.kind} @ {ev.when} — {ev.note}")

print("\n== Challenge 3 — THE LONG TAIL: head vs tail + provenance ==")
split = kg.long_tail()
print("  head:", [f"{l}/{s or '-'}×{c}" for l, s, c in split["head"]])
print("  tail:", [f"{l}/{s or '-'}×{c}" for l, s, c in split["tail"]])
first = kg.claims_about(commodity)[0]
prov = kg.provenance_of(first.id)
print(f"  provenance of one claim: source={prov['document_id']}/{prov['chunk_id']}, "
      f"judged={[v['criterion'] for v in prov['verdicts']]}")

print("\n== Complementary — conflicting viewpoints a flat KG would erase ==")
for group in kg.conflicts():
    print("  " + " | ".join(f"{c.perspective.viewpoint}: {c.object}" for c in group))
