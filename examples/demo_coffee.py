"""Offline demo of the full pipeline on the paper's coffee example.

Runs with no API key: a scripted MockLLM plays the role of the extractor and
the judge panel, so you can see the "unflattened" graph and its query views
without any model calls. Swap MockLLM for a real backend (see README) to run
it for real.

    python examples/demo_coffee.py
"""

from idchkg.llm.mock import MockLLM
from idchkg.models import Document
from idchkg.pipeline import Pipeline, PipelineConfig


def claim(subject, predicate, obj, viewpoint="author", stance="asserts",
          sentiment="neutral", when=None, quote="q"):
    return {
        "subject": subject, "predicate": predicate, "object": obj,
        "perspective": {"viewpoint": viewpoint, "stance": stance, "sentiment": sentiment},
        "time": {"when": when, "valid_from": None, "valid_to": None},
        "quote": quote,
    }


def pass_verdict():
    return {"score": 1.0, "passed": True, "feedback": ""}


# One chunk of extracted claims capturing multiple perspectives + time.
extraction = {
    "entities": [
        {"name": "coffee", "type": "concept", "description": "the beverage and plant"},
        {"name": "Ethiopia", "type": "place", "description": "origin of the coffee plant"},
        {"name": "colonialism", "type": "concept", "description": ""},
    ],
    "claims": [
        claim("coffee", "native to", "Ethiopia", when="before 15th century"),
        claim("coffee", "cultivated in", "Yemen", when="15th century"),
        # Two viewpoints, same subject+predicate, opposing object and stance:
        # a genuine disagreement the graph must preserve, not reconcile.
        claim("coffee", "was", "civilising luxury",
              viewpoint="colonial writers", stance="asserts", sentiment="positive",
              when="17th century"),
        claim("coffee", "was", "product of forced labour",
              viewpoint="historians", stance="disputes", sentiment="negative",
              when="17th century"),
        claim("coffee", "source of", "national pride",
              viewpoint="author", sentiment="positive", when="today"),
    ],
}

# The mock plays extractor once, then two passing verdicts per claim.
mock = MockLLM().queue(extraction, *([pass_verdict()] * (2 * len(extraction["claims"]))))

doc = Document(id="coffee", text=open("examples/corpus/coffee.txt").read(), title="Coffee")
pipeline = Pipeline(PipelineConfig(backend="mock"), extractor_client=mock, judge_client=mock)
result = pipeline.run([doc])

print("stats:", result.stats())
kg = result.graph
coffee = kg.find_entity("coffee")

print("\n== Perspectives on 'coffee' (Challenge 1: polyvocality) ==")
for viewpoint, claims in kg.perspectives_on(coffee).items():
    print(f"  {viewpoint}:")
    for c in claims:
        print(f"    {c.subject} --{c.predicate}--> {c.object}  ({c.perspective.sentiment})")

print("\n== Timeline of 'coffee' (Challenge 2: time) ==")
for c in kg.timeline(coffee):
    print(f"  [{c.time.when or 'undated'}] {c.subject} --{c.predicate}--> {c.object}")

print("\n== Conflicts (viewpoints a flat KG would erase) ==")
for group in kg.conflicts():
    print("  " + " | ".join(f"{c.perspective.viewpoint}: {c.object}" for c in group))

print("\n== Provenance of one claim (Challenge 3) ==")
first = kg.claims_about(coffee)[0]
prov = kg.provenance_of(first.id)
print(f"  claim: {first.subject} --{first.predicate}--> {first.object}")
print(f"  source: {prov['document_id']} / {prov['chunk_id']}")
print(f"  judged by: {[v['criterion'] + '=' + str(v['score']) for v in prov['verdicts']]}")
