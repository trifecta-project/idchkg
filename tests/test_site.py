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
    # A single-sense corpus has no polysemous concepts to highlight.
    assert "<h2>Most senses</h2>" not in index
    assert "<h2>Most mentioned</h2>" not in index


def test_most_senses_section_leads_with_polysemous_entities(tmp_path):
    # A label with two senses (coffee: beverage + plant) is the polysemy the
    # "Most senses" section is meant to surface; single-sense labels are absent.
    store = GraphStore(tmp_path / "g.db", mode="build")
    bev = Entity(id="ent-coffee--beverage", label="coffee", sense="beverage", mention_count=6)
    plant = Entity(id="ent-coffee--plant", label="coffee", sense="plant", mention_count=9)
    society = Entity(id="ent-society", label="society", mention_count=2)
    store.add_document(Document(id="doc", text="Coffee.", title="Coffee"))
    store.add_claim(
        Claim("coffee", "grows-in", "society",
              perspective=Perspective("writers", "asserts", "neutral"),
              provenance=Provenance("doc", "doc#c0", "coffee", "mock")),
        plant, society)
    store.finalize_entities([bev, plant, society])
    store.close()

    store = GraphStore(tmp_path / "g.db", mode="query")
    out = build_site(store, tmp_path / "site")
    store.close()

    index = (out / "index.html").read_text(encoding="utf-8")
    assert "<h2>Most senses</h2>" in index
    assert "<h2>Most mentioned</h2>" not in index

    # The "Most senses" section links both coffee senses; society (single sense)
    # is not in it (it still appears in the "All entities" list below).
    head = index.split("<h2>Most senses</h2>")[1].split("<h2>All entities</h2>")[0]
    assert "entity/ent-coffee--beverage.html" in head
    assert "entity/ent-coffee--plant.html" in head
    assert "ent-society.html" not in head


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


def test_branding_assets_and_funding_statement(tmp_path):
    store = _store(tmp_path)
    out = build_site(store, tmp_path / "site")
    store.close()

    # Brand logos are written as assets and referenced by the page chrome.
    assert (out / "assets" / "trifecta-logo.png").exists()
    assert (out / "assets" / "erc-logo.png").exists()
    for f in ("assets/trifecta-logo.png", "assets/erc-logo.png"):
        assert (out / f).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    index = (out / "index.html").read_text(encoding="utf-8")
    # Header: TRIFECTA logo + tagline.
    assert 'src="assets/trifecta-logo.png"' in index
    assert "Better Knowledge Graphs for Humanities Research" in index
    # Footer: ERC logo + the required EU funding statement.
    assert 'src="assets/erc-logo.png"' in index
    assert "Funded by the European Union under grant agreement 101088548" in index
    assert "can be held responsible for them." in index

    # Deeper pages reference the same assets via the ../ depth prefix.
    entity = (out / "entity" / "ent-coffee--beverage.html").read_text(encoding="utf-8")
    assert 'src="../assets/trifecta-logo.png"' in entity
    assert 'src="../assets/erc-logo.png"' in entity
    assert "grant agreement 101088548" in entity


def test_over_long_entity_is_skipped_and_logged(tmp_path):
    # A hallucinated, sentence-length label (extraction error) would produce a
    # filename past the filesystem limit. The build must not crash: skip its
    # page and record it in skipped_entities.log instead.
    long_slug = "x" * 1000
    store = GraphStore(tmp_path / "g.db", mode="build")
    good = Entity(id="ent-coffee--beverage", label="coffee", sense="beverage", mention_count=3)
    bad = Entity(id=f"ent-{long_slug}", label="a whole hallucinated sentence " * 30,
                 sense="", mention_count=1)
    store.add_document(Document(id="doc", text="Coffee.", title="Coffee"))
    store.add_claim(
        Claim("coffee", "is", "coffee",
              perspective=Perspective("writers", "asserts", "neutral"),
              provenance=Provenance("doc", "doc#c0", "coffee", "mock")),
        good, good)
    store.finalize_entities([good, bad])
    store.close()

    store = GraphStore(tmp_path / "g.db", mode="query")
    out = build_site(store, tmp_path / "site")  # must not raise
    store.close()

    # The good entity's page exists; the bad one's was never written.
    # (Don't stat the over-long path directly — that itself raises OSError.)
    assert (out / "entity" / "ent-coffee--beverage.html").exists()
    assert not any(long_slug in p.name for p in (out / "entity").iterdir())

    # The skip is recorded in the error log with the offending id.
    log = (out / "skipped_entities.log").read_text(encoding="utf-8")
    assert f"ent-{long_slug}" in log

    # The good entity still lists on the index; the bad one is absent.
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "entity/ent-coffee--beverage.html" in index
    assert f"ent-{long_slug}.html" not in index


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
