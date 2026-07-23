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


def test_source_link_points_to_the_quote_paragraph(tmp_path):
    # A multi-paragraph document; the claim's quote lives in the 3rd paragraph.
    store = GraphStore(tmp_path / "g.db", mode="build")
    coffee = Entity(id="ent-coffee--beverage", label="coffee", sense="beverage", mention_count=1)
    society = Entity(id="ent-society", label="society", mention_count=1)
    text = "First paragraph about nothing.\n\nSecond paragraph, still nothing.\n\nCoffee refined society in the coffee-house era."
    store.add_document(Document(id="doc", text=text, title="Coffee"))
    store.add_claim(
        Claim("coffee", "refines", "society",
              perspective=Perspective("writers", "asserts", "neutral"),
              provenance=Provenance("doc", "doc#c0", "Coffee refined society", "mock")),
        coffee, society)
    store.finalize_entities([coffee, society])
    store.close()

    store = GraphStore(tmp_path / "g.db", mode="query")
    out = build_site(store, tmp_path / "site")
    store.close()

    # The document page numbers its paragraphs, and the quote is in p2.
    doc = (out / "doc" / "doc.html").read_text(encoding="utf-8")
    assert '<p id="p0">' in doc and '<p id="p2">' in doc
    assert 'id="p2">Coffee refined society' in doc

    # The 'source' link on the entity page targets that exact paragraph.
    entity = (out / "entity" / "ent-coffee--beverage.html").read_text(encoding="utf-8")
    assert 'class="src-doc" href="../doc/doc.html#p2"' in entity


def test_overview_stat_tiles_and_charts(tmp_path):
    store = GraphStore(tmp_path / "g.db", mode="build")
    coffee = Entity(id="ent-coffee--beverage", label="coffee", sense="beverage", mention_count=5,
                    dbpedia={"uri": "http://dbpedia.org/resource/Coffee", "score": 0.99,
                             "surfaceForm": "coffee",
                             "types": ["http://dbpedia.org/ontology/Beverage",
                                       "http://dbpedia.org/ontology/Food"], "dbo_type": []})
    society = Entity(id="ent-society", label="society", mention_count=1)  # hapax
    store.add_document(Document(id="doc", text="Coffee refined society."))
    store.add_claim(
        Claim("coffee", "refines", "society",
              perspective=Perspective("writers", "asserts", "positive"),
              provenance=Provenance("doc", "doc#c0", "coffee")),
        coffee, society)
    store.finalize_entities([coffee, society])
    store.close()

    store = GraphStore(tmp_path / "g.db", mode="query")
    out = build_site(store, tmp_path / "site")
    store.close()
    index = (out / "index.html").read_text(encoding="utf-8")

    # Stat-tile band with headline numbers, incl. DBpedia coverage.
    assert "<h2>At a glance</h2>" in index
    assert 'class="stat-num">50%</span><span class="stat-label">DBpedia-linked' in index
    # 'What the corpus is about' chart is driven by entity senses.
    assert "What the corpus is about" in index
    assert ">beverage</span>" in index                   # coffee's sense
    # Long-tail chart + hapax note (1 of 2 entities appears once).
    assert "Mention frequency" in index and "appear exactly once" in index
    # Sentiment diverging bar with a legend entry.
    assert "Positive 1" in index


def test_overview_omits_senses_chart_when_no_senses(tmp_path):
    # Entities without a sense -> the 'What the corpus is about' chart is skipped,
    # but the band and the always-available charts still render.
    store = GraphStore(tmp_path / "g.db", mode="build")
    a = Entity(id="ent-a", label="alpha", mention_count=2)   # no sense
    b = Entity(id="ent-b", label="beta", mention_count=1)
    store.add_document(Document(id="d", text="Alpha and beta."))
    store.finalize_entities([a, b])
    store.close()
    store = GraphStore(tmp_path / "g.db", mode="query")
    out = build_site(store, tmp_path / "site")
    store.close()
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "<h2>At a glance</h2>" in index
    assert "What the corpus is about" not in index      # no senses to chart
    assert "Mention frequency" in index                  # long tail always shown


def test_streams_page_collapses_labels_and_omits_thin_concepts(tmp_path):
    store = GraphStore(tmp_path / "g.db", mode="build")
    # 'coffee' concept collapses coffee + roasted coffee: two labels -> pooled
    # senses (beverage in the 1600s, product in the 1800s).
    bev = Entity(id="ent-coffee--beverage", label="coffee", sense="beverage", mention_count=5)
    prod = Entity(id="ent-roasted--product", label="roasted coffee", sense="product", mention_count=5)
    tea = Entity(id="ent-tea", label="tea", sense="beverage", mention_count=1)  # single sense -> omitted
    obj = Entity(id="ent-x", label="x")
    store.add_document(Document(id="d", text="Coffee."))
    for i in range(5):
        store.add_claim(
            Claim("coffee", f"p{i}", "x", subject_sense="beverage",
                  perspective=Perspective("w", "asserts", "neutral"),
                  time=TemporalScope(valid_from="1650-01-01"),
                  provenance=Provenance("d", "d#0", "coffee")), bev, obj)
    for i in range(5):
        store.add_claim(
            Claim("roasted coffee", f"g{i}", "x", subject_sense="product",
                  perspective=Perspective("w", "asserts", "neutral"),
                  time=TemporalScope(valid_from="1850-01-01"),
                  provenance=Provenance("d", "d#0", "roasted")), prod, obj)
    store.add_claim(
        Claim("tea", "is", "x", subject_sense="beverage",
              perspective=Perspective("w", "asserts", "neutral"),
              time=TemporalScope(valid_from="1800-01-01"),
              provenance=Provenance("d", "d#0", "tea")), tea, obj)
    store.finalize_entities([bev, prod, tea, obj])
    store.close()

    store = GraphStore(tmp_path / "g.db", mode="query")
    out = build_site(store, tmp_path / "site")
    store.close()

    streams = (out / "streams.html").read_text(encoding="utf-8")
    # Coffee renders (collapse pooled two labels' senses across two centuries).
    assert "<h3>Coffee</h3>" in streams
    assert 'class="stream-svg"' in streams
    assert "beverage 5" in streams and "product 5" in streams
    assert ">1600s</text>" in streams and ">1800s</text>" in streams
    # Thin concept (tea: one sense) is omitted; nutmeg (absent) too.
    assert "<h3>Tea</h3>" not in streams
    assert "<h3>Nutmeg</h3>" not in streams

    # Nav link is present; stream no longer appears on the entity page.
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="streams.html">Streams</a>' in index
    entity = (out / "entity" / "ent-coffee--beverage.html").read_text(encoding="utf-8")
    assert "Senses over time" not in entity


def test_geomap_is_zoomable(tmp_path):
    store = GraphStore(tmp_path / "g.db", mode="build")
    brazil = Entity(id="ent-brazil", label="Brazil", type="place", mention_count=5)
    store.add_document(Document(id="d", text="Coffee grew in Brazil."))
    store.finalize_entities([brazil])
    store.close()

    store = GraphStore(tmp_path / "g.db", mode="query")
    out = build_site(store, tmp_path / "site")
    store.close()

    geo = (out / "geomap.html").read_text(encoding="utf-8")
    assert 'id="geomap-svg"' in geo          # the SVG is targetable by the zoom script
    assert 'id="geo-in"' in geo and 'id="geo-reset"' in geo  # zoom controls
    assert "addEventListener('wheel'" in geo  # wheel-zoom wired up


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
    # Header: TRIFECTA logo links out to the project site; 'idchkg' links home.
    assert 'src="assets/trifecta-logo.png"' in index
    assert 'class="brand-logo-link" href="https://trifecta.dhlab.nl"' in index
    assert 'class="brand-name" href="index.html">idchkg</a>' in index
    assert "An Unflattened Identity and Change Knowledge Graph" in index
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

    # The good entity is in the search index; the bad one is absent from both.
    ents_js = (out / "entities.js").read_text(encoding="utf-8")
    assert "ent-coffee--beverage" in ents_js
    assert f"ent-{long_slug}" not in ents_js


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


def test_index_uses_search_index_not_flat_list(tmp_path):
    store = _store(tmp_path)
    out = build_site(store, tmp_path / "site")
    store.close()
    # Every entity lives in a compact entities.js (loaded via <script src>), not
    # inlined into the index — so the index stays small at any scale.
    ents = (out / "entities.js").read_text(encoding="utf-8")
    assert ents.startswith("window.ENTITIES=")
    assert "ent-coffee--beverage" in ents
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'id="ent-search"' in index and 'class="facet-bar"' in index
    assert 'src="entities.js"' in index
    assert 'id="entity-index"' not in index  # the old flat all-entities list is gone


def test_disagreements_page(tmp_path):
    store = _store(tmp_path)  # coffee 'refines' society: asserts vs disputes
    out = build_site(store, tmp_path / "site")
    store.close()
    page = (out / "disagreements.html").read_text(encoding="utf-8")
    assert "<h1>Disagreements</h1>" in page
    assert "conflict-claims" in page
    assert 'href="disagreements.html">Disagreements</a>' in page  # nav link present


def test_sources_page(tmp_path):
    store = _store(tmp_path)  # one claim carries a groundedness verdict (0.90)
    out = build_site(store, tmp_path / "site")
    store.close()
    page = (out / "sources.html").read_text(encoding="utf-8")
    assert "Sources &amp; reliability" in page
    assert "Claims per source" in page
    assert "Judge scores — groundedness" in page
    assert 'href="sources.html">Sources</a>' in page  # nav link present


def test_relationship_graph_shown_only_when_connected(tmp_path):
    store = GraphStore(tmp_path / "g.db", mode="build")
    hub = Entity(id="ent-coffee--beverage", label="coffee", sense="beverage", mention_count=5)
    places = [Entity(id=f"ent-{p}", label=p.title(), type="place", mention_count=2)
              for p in ("brazil", "yemen", "london")]
    lonely = Entity(id="ent-lonely", label="lonely", mention_count=1)
    store.add_document(Document(id="d", text="Coffee."))
    for i, nb in enumerate(places):  # 3 distinct neighbours -> graph shown
        store.add_claim(
            Claim("coffee", f"grown-in-{i}", nb.label, subject_sense="beverage",
                  perspective=Perspective("w", "asserts", "neutral"),
                  provenance=Provenance("d", "d#0", "coffee")), hub, nb)
    store.finalize_entities([hub, *places, lonely])
    store.close()

    store = GraphStore(tmp_path / "g.db", mode="query")
    out = build_site(store, tmp_path / "site")
    store.close()
    hub_page = (out / "entity" / "ent-coffee--beverage.html").read_text(encoding="utf-8")
    assert "<h2>Connections</h2>" in hub_page and 'class="ego-svg"' in hub_page
    lonely_page = (out / "entity" / "ent-lonely.html").read_text(encoding="utf-8")
    assert "<h2>Connections</h2>" not in lonely_page  # isolated entity -> omitted
