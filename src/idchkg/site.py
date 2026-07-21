"""Static site generator for browsing a knowledge graph.

`build_site(graph, out_dir)` renders a self-contained, entity-centric web site
from a built graph (a `GraphStore` opened in query mode, or an in-memory
`KnowledgeGraph`). Output is pre-rendered HTML + one CSS file + a few lines of
vanilla JS, so it works over `file://`, on GitHub Pages, or any static host —
no server, no build step, no dependencies.

The entity page centres on **perspectives**: the claims about a concept grouped
by the viewpoint that makes them, with stance/sentiment, the verbatim quote, its
source, and the judge verdicts — so the different narratives on one concept sit
side by side (the paper's "unflattening").
"""

from __future__ import annotations

import html
import json
import logging
from collections import Counter
from pathlib import Path

from idchkg._assets import ERC_LOGO_PNG, TRIFECTA_LOGO_PNG
from idchkg._world_svg import COUNTRIES_SVG

logger = logging.getLogger(__name__)

# Single-component filename limit on common filesystems (bytes). Entity page
# files are named ``{entity.id}.html``; an over-long id (a hallucinated,
# sentence-length label slugified in resolve.py) blows past this and makes
# Path.write_text raise. Such entities are extraction errors — we skip them.
_MAX_FILENAME_BYTES = 255

# ------------------------------------------------------------------ helpers


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _filename_too_long(entity) -> bool:
    """True when the entity's page filename would exceed the filesystem limit."""
    return len(f"{entity.id}.html".encode("utf-8")) > _MAX_FILENAME_BYTES


def _key(label: str, sense: str) -> tuple[str, str]:
    return (label.strip().casefold(), (sense or "").strip().casefold())


def _entity_link(label: str, sense: str, by_key: dict, prefix: str) -> str:
    """Cross-link an endpoint label to its entity page when resolvable, keyed by
    (canonical label, sense) as the resolver does; else plain text."""
    entity = by_key.get(_key(label, sense))
    if entity is None and not (sense or "").strip():
        candidates = [e for (lbl, _), e in by_key.items() if lbl == _key(label, sense)[0]]
        if len(candidates) == 1:
            entity = candidates[0]
    if entity is not None:
        return f'<a href="{prefix}{_esc(entity.id)}.html">{_esc(label)}</a>'
    return f"<span class=\"ent\">{_esc(label)}</span>"


def _paragraphs(text: str) -> list[str]:
    """Document text split into the same non-empty paragraphs the document page
    renders — so paragraph indices line up with the ``#p<i>`` anchors."""
    return [p for p in (text or "").split("\n\n") if p.strip()]


def _norm(s: str) -> str:
    """Whitespace-collapsed, case-folded text for quote↔paragraph matching."""
    return " ".join((s or "").split()).casefold()


def _claim_anchors(graph, documents) -> dict[str, str]:
    """Map each claim id → the ``#p<i>`` anchor of the paragraph its provenance
    quote appears in, so a 'source' link jumps to the exact paragraph (not just
    the top of a long document). Claims whose quote can't be located are absent
    (the link falls back to the document top)."""
    anchors: dict[str, str] = {}
    for doc in documents:
        norm_paras = [_norm(p) for p in _paragraphs(doc.text or "")]
        if not norm_paras:
            continue
        for claim in graph.claims_in_document(doc.id):
            prov = claim.provenance
            quote = _norm(prov.quote) if prov and prov.quote else ""
            if not quote:
                continue
            for i, para in enumerate(norm_paras):
                if quote in para:
                    anchors[claim.id] = f"#p{i}"
                    break
    return anchors


def _prop_html(prop: dict, by_key: dict, prefix: str) -> str:
    other = _entity_link(prop["other"], "", by_key, prefix)
    ctx = f' <span class="ctx">⟨{_esc(prop["context"])}⟩</span>' if prop.get("context") else ""
    return f'<li><span class="pred">{_esc(prop["predicate"])}</span> {other}{ctx}</li>'


def _claim_html(claim, by_key: dict, prefix: str, doc_prefix: str = "doc/", anchors: dict | None = None) -> str:
    subj = _entity_link(claim.subject, claim.subject_sense, by_key, prefix)
    obj = _entity_link(claim.object, claim.object_sense, by_key, prefix)
    stance = claim.perspective.stance
    sentiment = claim.perspective.sentiment
    chips = (
        f'<span class="chip stance-{_esc(stance)}">{_esc(stance)}</span>'
        f'<span class="chip sent-{_esc(sentiment)}">{_esc(sentiment)}</span>'
    )
    when = f'<span class="when">{_esc(claim.time.when)}</span>' if claim.time.when else ""
    ctx = f'<span class="ctx">⟨{_esc(claim.context)}⟩</span>' if claim.context else ""

    quote = source = verdicts = ""
    prov = claim.provenance
    if prov:
        if prov.quote:
            quote = f'<blockquote>{_esc(prov.quote)}</blockquote>'
        if prov.document_id:
            # Link the source back to the underlying document (provenance), and
            # to the exact paragraph when we could locate the quote there.
            anchor = (anchors or {}).get(claim.id, "")
            source = f'<a class="src-doc" href="{doc_prefix}{_esc(prov.document_id)}.html{anchor}">source: {_esc(prov.document_id)}</a>'
        if prov.verdicts:
            rows = "".join(
                f'<li><span class="crit">{_esc(v.criterion)}</span> '
                f'<span class="score">{v.score:.2f}</span> '
                f'{"pass" if v.passed else "fail"} — {_esc(v.feedback)}</li>'
                for v in prov.verdicts
            )
            verdicts = (
                f'<details><summary>judge verdicts ({len(prov.verdicts)})</summary>'
                f'<ul class="verdicts">{rows}</ul></details>'
            )
    return (
        '<li class="claim">'
        f'<div class="stmt">{subj} <span class="pred">{_esc(claim.predicate)}</span> {obj} {when} {ctx}</div>'
        f'<div class="chips">{chips}{source}</div>'
        f"{quote}{verdicts}"
        "</li>"
    )


def _page(title: str, body: str, depth: int) -> str:
    """Wrap body in the site chrome. depth 0 = home/timeline, 1 = entity/ or doc/ page."""
    up = "" if depth == 0 else "../"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<link rel="stylesheet" href="{up}style.css">
</head>
<body>
<header class="site"><div class="wrap"><div class="brand">
<a class="brand-logo-link" href="https://trifecta.dhlab.nl" title="TRIFECTA — trifecta.dhlab.nl"><img class="brand-logo" src="{up}assets/trifecta-logo.png" alt="TRIFECTA" width="54" height="60"></a>
<a class="brand-name" href="{up}index.html">idchkg</a></div>
<span class="tagline">An Unflattened Identity and Change Knowledge Graph</span>
<nav><a href="{up}index.html">Browse</a> <a href="{up}timeline.html">Timeline</a> <a href="{up}heatmap.html">Heatmap</a> <a href="{up}geomap.html">Map</a></nav></div></header>
<main class="wrap">
{body}
</main>
<footer class="site"><div class="wrap">
<div class="funding">
<img class="erc-logo" src="{up}assets/erc-logo.png" alt="Funded by the European Union — European Research Council" width="240" height="101">
<p>Funded by the European Union under grant agreement 101088548 — TRIFECTA. Views and opinions expressed are however those of the author only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.</p>
</div>
<p class="colophon">Generated by idchkg — entities keyed by (label, sense); every claim keeps its perspective, time, and provenance.</p>
</div></footer>
</body>
</html>
"""


# -------------------------------------------------------------------- pages


def _render_index(graph, entities, by_key, documents) -> str:
    stats = (
        f'<ul class="stats">'
        f'<li><b>{graph.document_count()}</b> documents</li>'
        f'<li><b>{graph.entity_count()}</b> entities</li>'
        f'<li><b>{graph.claim_count()}</b> claims</li>'
        f"</ul>"
    )

    # Lead with the polysemous concepts (Challenge 1: identity) — labels held
    # apart into multiple senses, most senses first. Single-sense entities stay
    # in the full "All entities" list below.
    sense_counts = Counter(e.label.casefold() for e in entities)
    multi = [e for e in entities if sense_counts[e.label.casefold()] >= 2]
    multi.sort(key=lambda e: (-sense_counts[e.label.casefold()], e.label.casefold(), -e.mention_count))
    head_items = "".join(
        f'<li><a href="entity/{_esc(e.id)}.html">{_esc(e.label)}'
        f'{f" <em>({_esc(e.sense)})</em>" if e.sense else ""}</a>'
        f' <span class="count">{e.mention_count}</span></li>'
        for e in multi
    )
    head = (
        '<section><h2>Most senses</h2>'
        '<p class="muted">Concepts one word holds apart into distinct senses — '
        'the same label, kept un-flattened. Those with the most senses come first.</p>'
        f'<ul class="entity-list">{head_items}</ul></section>'
    ) if head_items else ""

    # Full, searchable index (data-label drives the JS filter).
    index_items = "".join(
        f'<li data-label="{_esc((e.label + " " + (e.sense or "")).casefold())}">'
        f'<a href="entity/{_esc(e.id)}.html">{_esc(e.label)}'
        f'{f" <em>({_esc(e.sense)})</em>" if e.sense else ""}</a>'
        f' <span class="count">{e.mention_count}</span></li>'
        for e in entities
    )
    index = (
        '<section><h2>All entities</h2>'
        '<input id="filter" type="search" placeholder="Filter entities…" autocomplete="off">'
        f'<ul id="entity-index" class="entity-list">{index_items}</ul></section>'
    )

    conflicts = _render_conflicts_section(graph, by_key, prefix="entity/")

    doc_items = "".join(
        f'<li><a href="doc/{_esc(d.id)}.html">{_esc(d.title or d.id)}</a></li>'
        for d in documents
    )
    documents_section = (
        f'<section><h2>Documents</h2><p class="muted">The sources — every claim links back to one.</p>'
        f'<ul class="entity-list">{doc_items}</ul></section>' if doc_items else ""
    )

    body = (
        '<section class="intro"><h1>Browse the knowledge graph</h1>'
        "<p>The same concept can be a plant, a drink, and a colonial commodity at once. "
        "Here concepts are kept as distinct <em>senses</em>, and every claim carries the "
        "<em>perspective</em> that makes it — so different narratives sit side by side. "
        'See the <a href="timeline.html">timeline</a> for the historical sweep.</p>'
        f"{stats}</section>"
        f"{head}{index}{conflicts}{documents_section}"
    )
    # tiny inline filter (works over file:// — no fetch)
    body += (
        "<script>const f=document.getElementById('filter'),"
        "items=[...document.querySelectorAll('#entity-index li')];"
        "f.addEventListener('input',()=>{const q=f.value.trim().toLowerCase();"
        "for(const li of items) li.style.display=li.dataset.label.includes(q)?'':'none';});</script>"
    )
    return _page("idchkg — knowledge graph", body, depth=0)


def _render_conflicts_section(graph, by_key, prefix: str, limit: int = 25) -> str:
    groups = graph.conflicts()
    if not groups:
        return ""
    blocks = []
    for group in groups[:limit]:
        rows = "".join(
            f'<li>{_entity_link(c.object, c.object_sense, by_key, prefix)} '
            f'<span class="chip stance-{_esc(c.perspective.stance)}">{_esc(c.perspective.stance)}</span> '
            f'<span class="src">{_esc(c.perspective.viewpoint)}</span></li>'
            for c in group
        )
        head = group[0]
        subj = _entity_link(head.subject, head.subject_sense, by_key, prefix)
        blocks.append(
            f'<div class="conflict"><div class="stmt">{subj} '
            f'<span class="pred">{_esc(head.predicate)}</span> …</div>'
            f'<ul class="conflict-claims">{rows}</ul></div>'
        )
    more = f'<p class="muted">…and {len(groups) - limit} more.</p>' if len(groups) > limit else ""
    return (
        '<section><h2>Disagreements</h2>'
        '<p class="muted">Claims that share a subject and predicate but differ in object or stance — '
        "disagreements a flat graph would silently reconcile.</p>"
        f'{"".join(blocks)}{more}</section>'
    )


def _short_uri(uri: str) -> str:
    """Last path/fragment segment of a URI, for a readable chip label."""
    tail = uri.rstrip("/").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return tail.replace("_", " ") or uri


def _dbpedia_html(entity) -> str:
    """The DBpedia link + ontology-type chips for an entity, when linked."""
    link = getattr(entity, "dbpedia", None)
    if not link or not link.get("uri"):
        return ""
    uri = link["uri"]
    score = link.get("score")
    title = f'Spotlight confidence {score:.2f}' if isinstance(score, (int, float)) else "DBpedia resource"
    # De-duplicate ontology classes (@types) and the dbo:type property values.
    type_uris = list(dict.fromkeys((link.get("types") or []) + (link.get("dbo_type") or [])))
    chips = "".join(
        f'<a class="dbo-chip" href="{_esc(t)}">{_esc(_short_uri(t))}</a>' for t in type_uris
    )
    chips_html = f'<span class="dbo-types">{chips}</span>' if chips else ""
    return (
        '<p class="dbpedia">'
        f'<a class="dbpedia-link" href="{_esc(uri)}" title="{_esc(title)}">DBpedia ↗</a>'
        f'{chips_html}</p>'
    )


def _render_entity(graph, entity, by_key, anchors=None) -> str:
    prefix = ""  # sibling entity pages live in the same entity/ directory
    label = _esc(entity.label)
    sub = f'<span class="sense">{_esc(entity.sense)}</span>' if entity.sense else '<span class="sense muted">unspecified sense</span>'
    header = (
        f'<section class="entity-head"><h1>{label} {sub}</h1>'
        f'<p class="meta"><span class="type">{_esc(entity.type)}</span> · '
        f'{entity.mention_count} mention{"s" if entity.mention_count != 1 else ""}</p>'
    )
    if entity.descriptions:
        header += f'<p class="desc">{_esc(entity.descriptions[0])}</p>'
    header += _dbpedia_html(entity)
    header += "</section>"

    # Sense switcher (Challenge 1: identity) — sibling senses of the same label.
    senses = [e for e in graph.senses_of(entity.label) if e.id != entity.id]
    senses_html = ""
    if senses:
        links = " ".join(
            f'<a href="{_esc(e.id)}.html" class="chip sense-chip">{_esc(e.sense or "unspecified")}</a>'
            for e in senses
        )
        senses_html = f'<section><h2>Other senses of “{label}”</h2><p>{links}</p></section>'

    # Defining properties at a glance.
    profile = graph.identity_profile(entity)
    props = profile["properties"]
    props_html = ""
    if props:
        rows = "".join(_prop_html(p, by_key, prefix) for p in props)
        props_html = f'<section><h2>Defining properties</h2><ul class="props">{rows}</ul></section>'

    # Perspectives — the centrepiece.
    grouped = graph.perspectives_on(entity)
    persp_cards = ""
    for viewpoint, claims in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        claim_lis = "".join(_claim_html(c, by_key, prefix, doc_prefix="../doc/", anchors=anchors) for c in claims)
        persp_cards += (
            f'<article class="viewpoint"><h3>{_esc(viewpoint)}</h3>'
            f'<ul class="claims">{claim_lis}</ul></article>'
        )
    perspectives = (
        '<section class="perspectives"><h2>Perspectives</h2>'
        '<p class="muted">Claims grouped by the voice that makes them.</p>'
        f'<div class="viewpoints">{persp_cards}</div></section>'
        if persp_cards
        else ""
    )

    # Change over time (Challenge 2) as a small visual timeline, when dated.
    drift = [b for b in graph.drift(entity) if b["when"] and b["when"] != "undated"]
    drift_html = ""
    if drift:
        items = "".join(
            f'<div class="tl-item"><div class="tl-when">{_esc(b["when"])}</div>'
            f'<div class="tl-body"><ul>{"".join(f"<li>{_esc(p)}</li>" for p in b["properties"])}</ul></div></div>'
            for b in drift
        )
        drift_html = f'<section><h2>Change over time</h2><div class="timeline-viz">{items}</div></section>'

    # Conflicts involving this entity.
    groups = graph.conflicts(entity)
    conflicts_html = ""
    if groups:
        blocks = []
        for group in groups:
            rows = "".join(
                f'<li>{_entity_link(c.object, c.object_sense, by_key, prefix)} '
                f'<span class="chip stance-{_esc(c.perspective.stance)}">{_esc(c.perspective.stance)}</span> '
                f'<span class="src">{_esc(c.perspective.viewpoint)}</span></li>'
                for c in group
            )
            blocks.append(
                f'<div class="conflict"><div class="stmt"><span class="pred">{_esc(group[0].predicate)}</span> …</div>'
                f'<ul class="conflict-claims">{rows}</ul></div>'
            )
        conflicts_html = f'<section><h2>Disagreements</h2>{"".join(blocks)}</section>'

    body = header + senses_html + perspectives + props_html + drift_html + conflicts_html
    return _page(f"{entity.label} — idchkg", body, depth=1)


def _render_heatmap(graph, entities, by_key) -> str:
    """A heatmap page: entities as tiles coloured by mention frequency."""
    if not entities:
        return _page("Heatmap — idchkg", '<p class="muted">No entities.</p>', depth=0)

    max_count = max(e.mention_count for e in entities)
    min_count = min(e.mention_count for e in entities)
    span = max(max_count - min_count, 1)

    tiles = []
    for e in sorted(entities, key=lambda e: -e.mention_count):
        t = (e.mention_count - min_count) / span  # 0..1, 1 = hottest
        sense = f' <em>({_esc(e.sense)})</em>' if e.sense else ""
        tiles.append(
            f'<a class="hm-tile" href="entity/{_esc(e.id)}.html" '
            f'style="--heat:{t:.3f}" title="{_esc(e.label)}: {e.mention_count} mentions">'
            f'<span class="hm-label">{_esc(e.label)}{sense}</span>'
            f'<span class="hm-count">{e.mention_count}</span></a>'
        )

    legend = (
        '<div class="hm-legend">'
        f'<span class="hm-lo">{min_count}</span>'
        '<span class="hm-bar"></span>'
        f'<span class="hm-hi">{max_count}</span>'
        '<span class="muted" style="margin-left:8px">mentions</span>'
        '</div>'
    )

    body = (
        '<section class="intro"><h1>Entity heatmap</h1>'
        '<p>Entities sized and coloured by how often they appear in the corpus — '
        'brighter tiles are mentioned more frequently.</p></section>'
        f'{legend}<div class="hm-grid">{"".join(tiles)}</div>'
    )
    return _page("Heatmap — idchkg", body, depth=0)


# Common place names → (lat, lon).  Equirectangular projection: x = lon, y = -lat.
_GEOCODE: dict[str, tuple[float, float]] = {
    "amsterdam": (52.37, 4.90), "athens": (37.98, 23.73), "baghdad": (33.31, 44.37),
    "bangkok": (13.76, 100.50), "barcelona": (41.39, 2.17), "beijing": (39.90, 116.40),
    "berlin": (52.52, 13.41), "bogota": (4.71, -74.07), "boston": (42.36, -71.06),
    "brussels": (50.85, 4.35), "budapest": (47.50, 19.04), "buenos aires": (-34.60, -58.38),
    "cairo": (30.04, 31.24), "cape town": (-33.93, 18.42), "chicago": (41.88, -87.63),
    "copenhagen": (55.68, 12.57), "dakar": (14.72, -17.47), "damascus": (33.51, 36.29),
    "delhi": (28.61, 77.21), "new delhi": (28.61, 77.21), "dubai": (25.20, 55.27),
    "dublin": (53.35, -6.26), "edinburgh": (55.95, -3.19), "florence": (43.77, 11.25),
    "geneva": (46.20, 6.14), "havana": (23.11, -82.37), "helsinki": (60.17, 24.94),
    "hong kong": (22.32, 114.17), "istanbul": (41.01, 28.98), "jakarta": (-6.21, 106.85),
    "jerusalem": (31.77, 35.23), "johannesburg": (-26.20, 28.05),
    "kabul": (34.53, 69.17), "karachi": (24.86, 67.01), "kathmandu": (27.72, 85.32),
    "kyiv": (50.45, 30.52), "kiev": (50.45, 30.52), "kuala lumpur": (3.14, 101.69),
    "lagos": (6.52, 3.38), "lima": (-12.05, -77.04), "lisbon": (38.72, -9.14),
    "london": (51.51, -0.13), "los angeles": (34.05, -118.24), "madrid": (40.42, -3.70),
    "manila": (14.60, 120.98), "mecca": (21.39, 39.86), "medina": (24.47, 39.61),
    "melbourne": (-37.81, 144.96), "mexico city": (19.43, -99.13), "milan": (45.46, 9.19),
    "moscow": (55.76, 37.62), "mumbai": (19.08, 72.88), "munich": (48.14, 11.58),
    "nairobi": (-1.29, 36.82), "new york": (40.71, -74.01), "new york city": (40.71, -74.01),
    "osaka": (34.69, 135.50), "oslo": (59.91, 10.75), "ottawa": (45.42, -75.70),
    "paris": (48.86, 2.35), "prague": (50.08, 14.44), "quito": (-0.18, -78.47),
    "rabat": (34.02, -6.84), "reykjavik": (64.15, -21.94), "rio de janeiro": (-22.91, -43.17),
    "riyadh": (24.69, 46.72), "rome": (41.90, 12.50), "san francisco": (37.77, -122.42),
    "santiago": (-33.45, -70.67), "são paulo": (-23.55, -46.63), "sao paulo": (-23.55, -46.63),
    "seoul": (37.57, 126.98), "shanghai": (31.23, 121.47), "singapore": (1.35, 103.82),
    "stockholm": (59.33, 18.07), "sydney": (-33.87, 151.21), "taipei": (25.03, 121.57),
    "tehran": (35.69, 51.39), "tokyo": (35.68, 139.69), "toronto": (43.65, -79.38),
    "tunis": (36.81, 10.18), "venice": (45.44, 12.32), "vienna": (48.21, 16.37),
    "warsaw": (52.23, 21.01), "washington": (38.91, -77.04), "washington dc": (38.91, -77.04),
    "zurich": (47.38, 8.54),
    # countries / regions → approximate centroids
    "afghanistan": (33.0, 65.0), "algeria": (28.0, 3.0), "argentina": (-34.0, -64.0),
    "australia": (-25.0, 134.0), "austria": (47.3, 13.3), "bangladesh": (24.0, 90.0),
    "belgium": (50.5, 4.0), "bolivia": (-17.0, -65.0), "brazil": (-10.0, -55.0),
    "cambodia": (13.0, 105.0), "cameroon": (6.0, 12.0), "canada": (56.0, -106.0),
    "chile": (-30.0, -71.0), "china": (35.0, 105.0), "colombia": (4.0, -72.0),
    "congo": (-1.0, 15.0), "costa rica": (10.0, -84.0), "croatia": (45.2, 15.5),
    "cuba": (22.0, -80.0), "czech republic": (49.8, 15.5), "czechia": (49.8, 15.5),
    "denmark": (56.0, 10.0), "ecuador": (-2.0, -77.5), "egypt": (27.0, 30.0),
    "england": (52.0, -1.0), "ethiopia": (8.0, 38.0), "finland": (64.0, 26.0),
    "france": (46.0, 2.0), "germany": (51.0, 10.0), "ghana": (8.0, -2.0),
    "greece": (39.0, 22.0), "guatemala": (15.5, -90.3), "hungary": (47.0, 20.0),
    "iceland": (65.0, -18.0), "india": (21.0, 78.0), "indonesia": (-5.0, 120.0),
    "iran": (32.0, 53.0), "iraq": (33.0, 44.0), "ireland": (53.0, -8.0),
    "israel": (31.0, 35.0), "italy": (42.5, 12.5), "jamaica": (18.1, -77.3),
    "japan": (36.0, 138.0), "jordan": (31.0, 36.0), "kazakhstan": (48.0, 68.0),
    "kenya": (1.0, 38.0), "korea": (36.0, 128.0), "south korea": (36.0, 128.0),
    "north korea": (40.0, 127.0), "kuwait": (29.5, 47.8), "laos": (18.0, 105.0),
    "lebanon": (33.9, 35.9), "libya": (27.0, 17.0), "malaysia": (4.0, 109.0),
    "mali": (17.0, -4.0), "mexico": (23.0, -102.0), "mongolia": (46.0, 105.0),
    "morocco": (32.0, -5.0), "mozambique": (-18.3, 35.0), "myanmar": (22.0, 98.0),
    "nepal": (28.0, 84.0), "netherlands": (52.1, 5.3), "holland": (52.1, 5.3),
    "new zealand": (-42.0, 174.0), "nigeria": (10.0, 8.0), "norway": (62.0, 10.0),
    "pakistan": (30.0, 70.0), "palestine": (31.9, 35.2), "panama": (9.0, -80.0),
    "paraguay": (-23.0, -58.0), "peru": (-10.0, -76.0), "philippines": (13.0, 122.0),
    "poland": (52.0, 20.0), "portugal": (39.5, -8.0), "qatar": (25.3, 51.2),
    "romania": (46.0, 25.0), "russia": (60.0, 100.0), "saudi arabia": (24.0, 45.0),
    "scotland": (57.0, -4.0), "senegal": (14.5, -14.5), "serbia": (44.0, 21.0),
    "south africa": (-30.0, 25.0), "spain": (40.0, -4.0), "sri lanka": (7.0, 81.0),
    "sudan": (15.0, 30.0), "sweden": (62.0, 15.0), "switzerland": (47.0, 8.0),
    "syria": (35.0, 38.0), "taiwan": (24.0, 121.0), "tanzania": (-6.0, 35.0),
    "thailand": (15.0, 100.0), "tunisia": (34.0, 9.0), "turkey": (39.0, 35.0),
    "turkiye": (39.0, 35.0), "uganda": (1.0, 32.0), "ukraine": (49.0, 32.0),
    "united arab emirates": (24.0, 54.0), "uae": (24.0, 54.0),
    "united kingdom": (54.0, -2.0), "uk": (54.0, -2.0), "britain": (54.0, -2.0),
    "great britain": (54.0, -2.0),
    "united states": (39.0, -98.0), "usa": (39.0, -98.0),
    "united states of america": (39.0, -98.0), "america": (39.0, -98.0),
    "uruguay": (-33.0, -56.0), "uzbekistan": (41.0, 64.0),
    "venezuela": (8.0, -66.0), "vietnam": (16.0, 108.0), "wales": (52.1, -3.6),
    "yemen": (15.0, 48.0), "zambia": (-15.0, 28.0), "zimbabwe": (-20.0, 30.0),
    # major historical / cultural places
    "constantinople": (41.01, 28.98), "byzantium": (41.01, 28.98),
    "persia": (32.0, 53.0), "ottoman empire": (39.0, 35.0), "mesopotamia": (33.0, 44.0),
    "anatolia": (39.0, 32.0), "levant": (34.0, 36.0), "maghreb": (33.0, 2.0),
    "sahara": (23.0, 12.0), "siberia": (60.0, 105.0), "scandinavia": (62.0, 15.0),
    "balkans": (42.0, 21.0), "caucasus": (42.0, 44.0), "tibet": (32.0, 88.0),
    "java": (-7.6, 110.4), "sumatra": (0.0, 102.0), "borneo": (1.0, 114.0),
    "sicily": (37.5, 14.0), "crete": (35.2, 24.9), "sardinia": (40.0, 9.0),
    "corsica": (42.2, 9.1), "ceylon": (7.0, 81.0), "formosa": (24.0, 121.0),
    "east indies": (-5.0, 120.0), "west indies": (17.0, -65.0),
    "caribbean": (15.0, -75.0), "mediterranean": (35.0, 18.0),
    "middle east": (29.0, 41.0), "southeast asia": (10.0, 107.0),
    "east africa": (-3.0, 37.0), "west africa": (10.0, -5.0), "north africa": (30.0, 10.0),
    "central america": (13.0, -85.0), "south america": (-15.0, -60.0),
    "north america": (45.0, -100.0), "central asia": (42.0, 65.0),
    "east asia": (35.0, 115.0), "south asia": (22.0, 78.0),
    "europe": (50.0, 10.0), "africa": (2.0, 22.0), "asia": (34.0, 100.0),
    "oceania": (-22.0, 140.0), "pacific": (0.0, -160.0), "atlantic": (25.0, -40.0),
    "arctic": (80.0, 0.0), "antarctica": (-80.0, 0.0),
    # rivers / landmarks
    "nile": (25.0, 32.0), "amazon": (-3.0, -60.0), "ganges": (25.0, 83.0),
    "danube": (47.0, 19.0), "rhine": (50.0, 7.0), "thames": (51.5, -0.1),
    "mississippi": (32.0, -91.0), "yangtze": (30.0, 112.0), "mekong": (16.0, 105.0),
    "alps": (46.5, 10.0), "himalayas": (28.0, 84.0), "andes": (-20.0, -66.0),
    "sahel": (14.0, 0.0),
}


def _geocode_entity(entity) -> tuple[float, float] | None:
    """Try to geocode an entity by its label (and aliases) against the built-in table."""
    for name in [entity.label] + getattr(entity, "aliases", []):
        coords = _GEOCODE.get(name.strip().casefold())
        if coords is not None:
            return coords
    return None


# Self-contained pan/zoom for the map SVG (viewBox manipulation; works over
# file://). Scroll/pinch to zoom toward the cursor, drag to pan, buttons to step.
_GEO_ZOOM_JS = """<script>
(function(){
 var svg=document.getElementById('geomap-svg');if(!svg)return;
 var W=800,H=400,MINW=100,vb={x:0,y:0,w:W,h:H};
 function apply(){svg.setAttribute('viewBox',vb.x.toFixed(2)+' '+vb.y.toFixed(2)+' '+vb.w.toFixed(2)+' '+vb.h.toFixed(2));}
 function clamp(){var mx=W*0.15,my=H*0.15;vb.x=Math.max(-mx,Math.min(W-vb.w+mx,vb.x));vb.y=Math.max(-my,Math.min(H-vb.h+my,vb.y));}
 function zoom(f,cx,cy){var nw=Math.max(MINW,Math.min(W,vb.w*f)),nh=nw*(H/W);vb.x=cx-(cx-vb.x)*(nw/vb.w);vb.y=cy-(cy-vb.y)*(nh/vb.h);vb.w=nw;vb.h=nh;clamp();apply();}
 function pt(e){var r=svg.getBoundingClientRect();return{x:vb.x+(e.clientX-r.left)/r.width*vb.w,y:vb.y+(e.clientY-r.top)/r.height*vb.h};}
 svg.addEventListener('wheel',function(e){e.preventDefault();var p=pt(e);zoom(e.deltaY<0?0.85:1/0.85,p.x,p.y);},{passive:false});
 var drag=null,moved=false;
 svg.addEventListener('pointerdown',function(e){drag={x:e.clientX,y:e.clientY,vx:vb.x,vy:vb.y};moved=false;try{svg.setPointerCapture(e.pointerId);}catch(_){}});
 svg.addEventListener('pointermove',function(e){if(!drag)return;var r=svg.getBoundingClientRect();if(Math.abs(e.clientX-drag.x)+Math.abs(e.clientY-drag.y)>3)moved=true;vb.x=drag.vx-(e.clientX-drag.x)/r.width*vb.w;vb.y=drag.vy-(e.clientY-drag.y)/r.height*vb.h;clamp();apply();});
 function end(){drag=null;}svg.addEventListener('pointerup',end);svg.addEventListener('pointerleave',end);
 svg.addEventListener('click',function(e){if(moved){e.preventDefault();e.stopPropagation();}},true);
 function step(f){return function(){zoom(f,vb.x+vb.w/2,vb.y+vb.h/2);};}
 document.getElementById('geo-in').addEventListener('click',step(0.7));
 document.getElementById('geo-out').addEventListener('click',step(1/0.7));
 document.getElementById('geo-reset').addEventListener('click',function(){vb={x:0,y:0,w:W,h:H};apply();});
})();
</script>"""


def _render_geomap(graph, entities, by_key) -> str:
    """A geographical heatmap: place entities plotted on a world map, coloured by frequency."""
    places = [e for e in entities if e.type == "place"]
    if not places:
        return _page(
            "Map — idchkg",
            '<section class="intro"><h1>Geographical heatmap</h1>'
            '<p class="muted">No place entities found in the graph.</p></section>',
            depth=0,
        )

    # Geocode what we can.
    resolved: list[tuple] = []  # (entity, lat, lon)
    unresolved: list = []
    for e in places:
        coords = _geocode_entity(e)
        if coords:
            resolved.append((e, *coords))
        else:
            unresolved.append(e)

    max_count = max(e.mention_count for e in places)
    min_count = min(e.mention_count for e in places)
    span = max(max_count - min_count, 1)

    # Equirectangular projection: x = lon+180, y = 90-lat, mapped to 0..800 x 0..400
    W, H = 800, 400

    def project(lat: float, lon: float) -> tuple[float, float]:
        return ((lon + 180) / 360 * W, (90 - lat) / 180 * H)

    # Build SVG markers.
    markers = []
    for e, lat, lon in sorted(resolved, key=lambda t: t[0].mention_count):
        x, y = project(lat, lon)
        t = (e.mention_count - min_count) / span
        # Radius proportional to sqrt of count for area-proportional sizing.
        r = 4 + 14 * (t ** 0.5)
        markers.append(
            f'<a href="entity/{_esc(e.id)}.html">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            f'fill-opacity=".75" class="geo-dot" style="--heat:{t:.3f}">'
            f'<title>{_esc(e.label)}: {e.mention_count} mentions</title>'
            f'</circle></a>'
        )

    # Minimal world coastline outline (simplified Natural Earth).
    # We use a very simple, recognisable outline as a background.
    map_bg = (
        '<rect width="800" height="400" fill="var(--geo-sea)" rx="6"/>'
        # Simple continent outlines as rounded rectangles for recognisability
        # (a real SVG path would be huge; this gives the spatial context needed)
        # Instead we draw a graticule (grid lines) that orients the viewer.
    )
    graticule = []
    for lon in range(-180, 181, 30):
        x, _ = project(0, lon)
        graticule.append(f'<line x1="{x:.0f}" y1="0" x2="{x:.0f}" y2="{H}" />')
    for lat in range(-90, 91, 30):
        _, y = project(lat, 0)
        graticule.append(f'<line x1="0" y1="{y:.0f}" x2="{W}" y2="{y:.0f}" />')
    # Equator and prime meridian slightly emphasised via class
    eq_y = H / 2
    pm_x = W / 2
    graticule_svg = (
        f'<g class="geo-grid" stroke="var(--geo-grid)" stroke-width=".5" stroke-dasharray="4 4">'
        f'{"".join(graticule)}</g>'
        f'<line x1="0" y1="{eq_y}" x2="{W}" y2="{eq_y}" stroke="var(--geo-grid)" stroke-width="1" opacity=".6"/>'
        f'<line x1="{pm_x}" y1="0" x2="{pm_x}" y2="{H}" stroke="var(--geo-grid)" stroke-width="1" opacity=".6"/>'
    )

    # Axis labels (a few lat/lon labels for orientation).
    axis_labels = []
    for lon in range(-180, 181, 60):
        x, _ = project(0, lon)
        axis_labels.append(f'<text x="{x:.0f}" y="{H - 4}" class="geo-axis">{lon}°</text>')
    for lat in range(-60, 91, 30):
        _, y = project(lat, 0)
        axis_labels.append(f'<text x="4" y="{y:.0f}" class="geo-axis">{lat}°</text>')

    svg = (
        f'<svg id="geomap-svg" class="geo-svg" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
        f'{map_bg}'
        f'<g class="geo-land">{COUNTRIES_SVG}</g>'
        f'{graticule_svg}'
        f'{"".join(axis_labels)}'
        f'{"".join(markers)}'
        '</svg>'
    )

    legend = (
        '<div class="hm-legend">'
        f'<span class="hm-lo">{min_count}</span>'
        '<span class="hm-bar"></span>'
        f'<span class="hm-hi">{max_count}</span>'
        '<span class="muted" style="margin-left:8px">mentions</span>'
        '</div>'
    )

    unresolved_html = ""
    if unresolved:
        items = "".join(
            f'<li><a href="entity/{_esc(e.id)}.html">{_esc(e.label)}'
            f'{f" <em>({_esc(e.sense)})</em>" if e.sense else ""}</a>'
            f' <span class="count">{e.mention_count}</span></li>'
            for e in sorted(unresolved, key=lambda e: -e.mention_count)
        )
        unresolved_html = (
            '<section><h2>Unmapped places</h2>'
            '<p class="muted">Place entities not matched to known coordinates.</p>'
            f'<ul class="entity-list">{items}</ul></section>'
        )

    controls = (
        '<div class="geo-controls">'
        '<button type="button" id="geo-in" title="Zoom in" aria-label="Zoom in">+</button>'
        '<button type="button" id="geo-out" title="Zoom out" aria-label="Zoom out">−</button>'
        '<button type="button" id="geo-reset" title="Reset view" aria-label="Reset view">⟲</button>'
        '</div>'
    )
    geo = f'<div class="geo-wrap">{svg}{controls}</div>'

    body = (
        '<section class="intro"><h1>Geographical heatmap</h1>'
        f'<p>Location entities plotted on a world map — larger, warmer circles mean more mentions. '
        f'{len(resolved)} of {len(places)} place entities mapped. '
        '<span class="muted">Scroll to zoom, drag to pan.</span></p></section>'
        f'{legend}{geo}{unresolved_html}{_GEO_ZOOM_JS}'
    )
    return _page("Map — idchkg", body, depth=0)


# -------------------------------------------------------------------- build


def build_site(graph, out_dir: str | Path) -> Path:
    """Render the whole site for `graph` into `out_dir`. Returns the output path."""
    out = Path(out_dir)
    (out / "entity").mkdir(parents=True, exist_ok=True)
    (out / "doc").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)

    # Brand assets used by the shared page chrome (header logo + footer funding).
    (out / "assets" / "trifecta-logo.png").write_bytes(TRIFECTA_LOGO_PNG)
    (out / "assets" / "erc-logo.png").write_bytes(ERC_LOGO_PNG)

    all_entities = sorted(graph.entities, key=lambda e: (e.label.casefold(), e.sense))
    # Over-long ids (hallucinated labels from the extraction step) produce
    # filenames past the filesystem limit; skip their pages and log them so the
    # rest of the site still builds. Cross-links to them fall back to plain text
    # because they are absent from by_key.
    entities = [e for e in all_entities if not _filename_too_long(e)]
    skipped = [e for e in all_entities if _filename_too_long(e)]
    by_key = {_key(e.label, e.sense): e for e in entities}
    documents = sorted(graph.documents(), key=lambda d: (d.title or d.id).casefold())

    if skipped:
        log_path = out / "skipped_entities.log"
        lines = [
            "# Entities skipped during site generation (over-long names — likely "
            "extraction errors).",
            "# Fields: mention_count\tsense\tid\tlabel",
        ]
        lines += [
            f"{e.mention_count}\t{e.sense}\t{e.id}\t{e.label}" for e in skipped
        ]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.warning(
            "Skipped %d entity page(s) with over-long names; see %s",
            len(skipped),
            log_path,
        )

    # Locate each claim's quote in its source document, so 'source' links can
    # jump to the exact paragraph (matters for long documents).
    anchors = _claim_anchors(graph, documents)

    (out / "style.css").write_text(_CSS, encoding="utf-8")
    (out / "index.html").write_text(_render_index(graph, entities, by_key, documents), encoding="utf-8")
    (out / "timeline.html").write_text(_render_timeline(graph, by_key, anchors), encoding="utf-8")
    (out / "heatmap.html").write_text(_render_heatmap(graph, entities, by_key), encoding="utf-8")
    (out / "geomap.html").write_text(_render_geomap(graph, entities, by_key), encoding="utf-8")
    for entity in entities:
        (out / "entity" / f"{entity.id}.html").write_text(
            _render_entity(graph, entity, by_key, anchors), encoding="utf-8"
        )
    for doc in documents:
        (out / "doc" / f"{doc.id}.html").write_text(
            _render_document(graph, doc, by_key, anchors), encoding="utf-8"
        )
    return out


def _render_document(graph, doc, by_key, anchors=None) -> str:
    """A source document page: full text + the claims extracted from it (provenance)."""
    path = doc.metadata.get("path") if isinstance(doc.metadata, dict) else None
    header = (
        f'<section class="entity-head"><h1>{_esc(doc.title or doc.id)}</h1>'
        f'<p class="meta">source document{f" · {_esc(path)}" if path else ""}</p></section>'
    )
    # Each paragraph gets an id so a claim's 'source' link can jump straight to it.
    paragraphs = "".join(f'<p id="p{i}">{_esc(p)}</p>' for i, p in enumerate(_paragraphs(doc.text or "")))
    text_html = f'<section><h2>Text</h2><article class="doctext">{paragraphs}</article></section>' if paragraphs else ""

    claims = graph.claims_in_document(doc.id)
    claims_html = ""
    if claims:
        lis = "".join(_claim_html(c, by_key, prefix="../entity/", doc_prefix="../doc/", anchors=anchors) for c in claims)
        claims_html = (
            f'<section><h2>Claims extracted here ({len(claims)})</h2>'
            f'<ul class="claims">{lis}</ul></section>'
        )
    return _page(f"{doc.title or doc.id} — idchkg", header + text_html + claims_html, depth=1)


def _render_timeline(graph, by_key, anchors=None) -> str:
    """A corpus-wide vertical timeline of every dated claim (Challenge 2: change)."""
    claims = graph.dated_claims()
    intro = (
        '<section class="intro"><h1>Timeline</h1>'
        '<p>Every claim that carries a time, in chronological order — the corpus\'s '
        'historical sweep, from the source\'s own dating.</p></section>'
    )
    if not claims:
        return _page("Timeline — idchkg", intro + "<p class=\"muted\">No dated claims.</p>", depth=0)
    items = []
    for c in claims:
        when = c.time.when or c.time.valid_from or "undated"
        subj = _entity_link(c.subject, c.subject_sense, by_key, "entity/")
        obj = _entity_link(c.object, c.object_sense, by_key, "entity/")
        src = ""
        if c.provenance and c.provenance.document_id:
            anchor = (anchors or {}).get(c.id, "")
            src = f' <a class="src-doc" href="doc/{_esc(c.provenance.document_id)}.html{anchor}">source: {_esc(c.provenance.document_id)}</a>'
        items.append(
            f'<div class="tl-item"><div class="tl-when">{_esc(when)}</div>'
            f'<div class="tl-body"><div class="stmt">{subj} <span class="pred">{_esc(c.predicate)}</span> {obj}</div>'
            f'<div class="chips"><span class="chip stance-{_esc(c.perspective.stance)}">{_esc(c.perspective.stance)}</span>'
            f'<span class="src">{_esc(c.perspective.viewpoint)}</span>{src}</div></div></div>'
        )
    body = intro + f'<div class="timeline-viz">{"".join(items)}</div>'
    return _page("Timeline — idchkg", body, depth=0)


_CSS = """\
:root{
  --bg:#faf9f6; --panel:#ffffff; --ink:#1c1b19; --muted:#6b6a66; --line:#e7e4dd;
  --accent:#12857a; --brand:#1fcbb0; --asserts:#3a7d5d; --disputes:#b4472f; --questions:#b07d1e;
}
*{box-sizing:border-box}
html{font-size:17px}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto;padding:0 24px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
header.site{border-top:3px solid var(--brand);border-bottom:1px solid var(--line);background:var(--panel)}
header.site .wrap{display:flex;align-items:center;gap:14px;padding:14px 24px}
.brand{display:flex;align-items:center;gap:10px}
.brand-logo-link{display:flex}
.brand-logo-link:hover{text-decoration:none}
.brand-logo{height:40px;width:auto;display:block}
.brand-name{font-weight:700;font-size:1.15rem;color:var(--ink)}
.brand-name:hover{text-decoration:none}
.tagline{color:var(--muted);font-size:.85rem}
footer.site{border-top:1px solid var(--line);margin-top:64px;color:var(--muted);font-size:.82rem}
footer.site .wrap{padding:26px 24px}
.funding{display:flex;align-items:center;gap:18px;padding-bottom:16px;margin-bottom:14px;border-bottom:1px solid var(--line)}
.erc-logo{flex:0 0 auto;height:56px;width:auto;display:block}
.funding p{margin:0;font-size:.78rem;line-height:1.5;color:var(--muted)}
.colophon{margin:0}
@media(max-width:560px){.funding{flex-direction:column;align-items:flex-start;gap:10px}}
main.wrap{padding-top:36px;padding-bottom:36px}
h1{font-size:2rem;line-height:1.2;margin:.2em 0 .3em}
h2{font-size:1.15rem;margin:2.2em 0 .6em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:0 0 .5em}
.intro p{font-size:1.05rem;color:#333;max-width:60ch}
.muted{color:var(--muted)}
ul.stats{list-style:none;display:flex;gap:28px;padding:0;margin:1.4em 0 0}
ul.stats b{font-size:1.5rem;display:block;color:var(--accent)}
ul.stats li{font-size:.85rem;color:var(--muted)}
input#filter{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:8px;
  font-size:1rem;margin:.4em 0 1em;background:var(--panel)}
ul.entity-list{list-style:none;padding:0;margin:0;columns:2;column-gap:32px}
ul.entity-list li{break-inside:avoid;margin:.15em 0}
ul.entity-list em{color:var(--muted);font-style:normal}
.count{color:var(--muted);font-size:.78rem}
/* entity page */
.entity-head h1{display:flex;align-items:baseline;gap:.4em;flex-wrap:wrap}
.sense{font-size:1.1rem;font-weight:600;color:var(--accent)}
.meta{color:var(--muted);margin:.1em 0 .4em}
.type{text-transform:capitalize}
.desc{max-width:60ch}
.dbpedia{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:.5em 0}
.dbpedia-link{font-weight:600;font-size:.85rem}
.dbo-types{display:flex;gap:6px;flex-wrap:wrap}
.dbo-chip{font-size:.72rem;font-weight:500;padding:1px 8px;border-radius:999px;background:#e4efec;
  color:#0e6f66;border:1px solid #bcd8d2;text-decoration:none}
.dbo-chip:hover{background:#d5e8e3;text-decoration:none}
.props{list-style:none;padding:0;margin:0}
.props li{padding:.25em 0;border-bottom:1px solid var(--line)}
.pred{color:var(--muted)}
.ent{color:var(--ink)}
/* perspectives */
.viewpoints{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.viewpoint{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.viewpoint h3{color:var(--accent);border-bottom:1px solid var(--line);padding-bottom:.4em}
ul.claims{list-style:none;padding:0;margin:0}
li.claim{padding:.7em 0;border-bottom:1px solid var(--line)}
li.claim:last-child{border-bottom:0}
.stmt{font-weight:500}
.chips{margin:.35em 0;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.chip{display:inline-block;font-size:.72rem;padding:1px 8px;border-radius:999px;
  border:1px solid var(--line);color:#fff}
.stance-asserts{background:var(--asserts);border-color:var(--asserts)}
.stance-disputes{background:var(--disputes);border-color:var(--disputes)}
.stance-questions{background:var(--questions);border-color:var(--questions)}
.sent-positive{background:#eaf5ee;color:var(--asserts);border-color:#cfe6d8}
.sent-negative{background:#fbece7;color:var(--disputes);border-color:#f2d6cc}
.sent-neutral{background:#f0efec;color:var(--muted);border-color:var(--line)}
.sent-ambivalent{background:#efeaf5;color:#6a4a9c;border-color:#ddd2ee}
.sense-chip{background:var(--accent);border-color:var(--accent);color:#fff;text-decoration:none}
.when{color:var(--muted);font-size:.82rem}
.ctx{color:var(--muted);font-size:.82rem;font-style:italic}
.src{color:var(--muted);font-size:.78rem}
.src-doc{color:var(--accent);font-size:.78rem;white-space:nowrap}
.src-doc::before{content:"↳ ";color:var(--muted)}
.src-doc:hover{text-decoration:underline}
header.site nav{margin-left:auto;display:flex;gap:18px;font-size:.92rem}
blockquote{margin:.4em 0;padding:.3em 0 .3em .8em;border-left:3px solid var(--line);
  color:#444;font-size:.92rem}
blockquote::before{content:"“"}blockquote::after{content:"”"}
details{margin:.3em 0;font-size:.85rem}
summary{cursor:pointer;color:var(--muted)}
ul.verdicts{margin:.4em 0;padding-left:1.1em}
ul.verdicts .crit{font-weight:600}
ul.verdicts .score{color:var(--accent)}
/* timeline visualization */
.timeline-viz{position:relative;margin:1em 0 0;padding-left:26px;border-left:2px solid var(--line)}
.tl-item{position:relative;margin:0 0 1.3em}
.tl-item::before{content:"";position:absolute;left:-33px;top:6px;width:11px;height:11px;
  border-radius:50%;background:var(--accent);border:2px solid var(--bg)}
.tl-when{font-weight:600;color:var(--accent);font-size:.85rem;margin-bottom:.15em}
.tl-body .stmt{font-weight:500}
.tl-body ul{margin:.2em 0;padding-left:1.1em}
/* documents */
.doctext{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 20px;
  max-width:66ch;color:#333}
.doctext p{margin:.7em 0;scroll-margin-top:80px;border-radius:4px;transition:background-color .8s ease}
/* the paragraph a 'source' link points at is highlighted on arrival */
.doctext p:target{background:#fff5d4;box-shadow:-10px 0 0 #fff5d4,-13px 0 0 var(--questions)}
/* conflicts */
.conflict{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:.6em 0}
ul.conflict-claims{list-style:none;padding:0;margin:.3em 0 0}
ul.conflict-claims li{padding:.2em 0}
/* geographical heatmap */
:root{--geo-sea:#e8eff4;--geo-grid:#c0cdd8}
.geo-land path{fill:#d6dcc4;stroke:#fff;stroke-width:.5;stroke-linejoin:round}
.geo-wrap{position:relative;max-width:820px;margin:.6em 0}
.geo-svg{display:block;width:100%;border-radius:8px;border:1px solid var(--line);
  touch-action:none;cursor:grab}
.geo-svg:active{cursor:grabbing}
.geo-controls{position:absolute;top:10px;right:10px;display:flex;flex-direction:column;gap:5px}
.geo-controls button{width:32px;height:32px;padding:0;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);border-radius:7px;font-size:1.2rem;line-height:1;
  cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.geo-controls button:hover{background:#f0efec;border-color:var(--accent)}
.geo-dot{fill:color-mix(in srgb,#b4472f calc(var(--heat)*100%),#e8c87a);
  stroke:#fff;stroke-width:1;cursor:pointer;transition:r .15s}
.geo-dot:hover{r:20;fill-opacity:1}
.geo-axis{font-size:9px;fill:var(--muted);dominant-baseline:middle}
/* heatmap */
.hm-legend{display:flex;align-items:center;gap:6px;margin:1.2em 0 .8em;font-size:.82rem;color:var(--muted)}
.hm-bar{display:inline-block;width:120px;height:12px;border-radius:6px;
  background:linear-gradient(90deg,#e8e4df,#f0c26b,#e07340,#b4472f)}
.hm-lo,.hm-hi{font-weight:600;color:var(--ink)}
.hm-grid{display:flex;flex-wrap:wrap;gap:6px}
.hm-tile{display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:10px 14px;border-radius:8px;text-decoration:none;
  min-width:80px;min-height:56px;
  background:color-mix(in srgb,#b4472f calc(var(--heat)*100%),#e8e4df);
  color:color-mix(in srgb,#fff calc(var(--heat)*100%),var(--ink));
  border:1px solid color-mix(in srgb,#b4472f calc(var(--heat)*60%),var(--line));
  transition:transform .12s}
.hm-tile:hover{transform:scale(1.06);text-decoration:none}
.hm-label{font-size:.82rem;font-weight:600;text-align:center;line-height:1.2}
.hm-label em{font-style:normal;font-weight:400;font-size:.72rem;opacity:.85}
.hm-count{font-size:.72rem;margin-top:2px;opacity:.8}
@media(max-width:600px){ul.entity-list{columns:1}html{font-size:16px}}
"""
