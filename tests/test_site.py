"""Offline test for the static site generator — deterministic, no LLM."""

from idchkg.models import Claim, Document, Entity, JudgeVerdict, Perspective, Provenance, TemporalScope
from idchkg.site import build_site
from idchkg.store import GraphStore


def _store(tmp_path) -> GraphStore:
    store = GraphStore(tmp_path / "g.db", mode="build")
    coffee = Entity(id="ent-coffee--beverage", label="coffee", sense="beverage", mention_count=3)
    society = Entity(id="ent-society", label="society", mention_count=2)
    store.add_document(Document(id="doc", text="Coffee and society.", title="Coffee",
                               metadata={"path": "examples/doc.txt"}))
    # Two viewpoints on the same subject+predicate, differing in stance -> a
    # perspective grouping AND a conflict. The quote carries HTML-special chars.
    # The first claim is dated, so it shows on the timeline.
    store.add_claim(
        Claim("coffee", "refines", "society",
              perspective=Perspective("colonial writers", "asserts", "positive"),
              time=TemporalScope(when="17th century", valid_from="1601"),
              provenance=Provenance("doc", "doc#c0", 'coffee "refined" <society>', "mock",
                                    verdicts=[JudgeVerdict("groundedness", 0.9, True, "ok", "mock")])),
        coffee, society)
    store.add_claim(
        Claim("coffee", "refines", "society",
              perspective=Perspective("historians", "disputes", "negative"),
              provenance=Provenance("doc", "doc#c1", "historians dispute this framing", "mock")),
        coffee, society)
    store.finalize_entities([coffee, society])
    store.close()
    return GraphStore(tmp_path / "g.db", mode="query")


def test_build_site_writes_pages(tmp_path):
    store = _store(tmp_path)
    out = build_site(store, tmp_path / "site")
    store.close()

    assert (out / "index.html").exists()
    assert (out / "style.css").exists()
    assert (out / "entity" / "ent-coffee--beverage.html").exists()

    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Browse the knowledge graph" in index
    assert "entity/ent-coffee--beverage.html" in index


def test_entity_page_groups_perspectives_and_escapes(tmp_path):
    store = _store(tmp_path)
    build_site(store, tmp_path / "site")
    store.close()

    page = (tmp_path / "site" / "entity" / "ent-coffee--beverage.html").read_text(encoding="utf-8")
    # Perspectives centrepiece: one viewpoint heading per voice.
    assert "<h2>Perspectives</h2>" in page
    assert "<h3>colonial writers</h3>" in page
    assert "<h3>historians</h3>" in page
    # Stance chips for both stances; judge verdicts surfaced.
    assert 'class="chip stance-asserts"' in page
    assert 'class="chip stance-disputes"' in page
    assert "judge verdicts (1)" in page
    # HTML in the quote is escaped, not injected.
    assert "&lt;society&gt;" in page
    assert "<society>" not in page


def test_conflicts_section_present(tmp_path):
    store = _store(tmp_path)
    build_site(store, tmp_path / "site")
    store.close()
    # asserts vs disputes on the same subject+predicate -> a disagreement.
    assert "<h2>Disagreements</h2>" in (tmp_path / "site" / "index.html").read_text(encoding="utf-8")


def test_document_page_and_provenance_link(tmp_path):
    store = _store(tmp_path)
    build_site(store, tmp_path / "site")
    store.close()
    site = tmp_path / "site"

    # A source-document page with the text and the claims extracted from it.
    doc = (site / "doc" / "doc.html").read_text(encoding="utf-8")
    assert "<h2>Text</h2>" in doc and "Coffee and society." in doc
    assert "Claims extracted here" in doc

    # The claim's source on the entity page links back to that document.
    entity = (site / "entity" / "ent-coffee--beverage.html").read_text(encoding="utf-8")
    assert 'class="src-doc" href="../doc/doc.html"' in entity


def test_timeline_page(tmp_path):
    store = _store(tmp_path)
    build_site(store, tmp_path / "site")
    store.close()
    timeline = (tmp_path / "site" / "timeline.html").read_text(encoding="utf-8")
    assert "<h1>Timeline</h1>" in timeline
    assert 'class="timeline-viz"' in timeline
    assert "17th century" in timeline  # the dated claim appears
    # nav link to the timeline is present on pages
    assert "timeline.html" in (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
