"""Command-line interface.

    idchkg build examples/corpus -o out              # build a graph (LLM calls)

    # The paper's three challenges:
    idchkg query out/graph.json coffee --view identity   # Ch.1 senses/near-identity
    idchkg query out/graph.json coffee --view change     # Ch.2 drift over time
    idchkg query out/graph.json        --view longtail   # Ch.3 head vs long tail
    idchkg query out/graph.json coffee --view provenance # Ch.3 source + verdicts

    # Complementary views:
    idchkg query out/graph.json coffee --view perspectives
    idchkg query out/graph.json        --view conflicts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .graph import KnowledgeGraph
from .models import Claim
from .pipeline import Pipeline, PipelineConfig


def _format_claim(claim: Claim) -> str:
    time = f" [{claim.time.when}]" if claim.time.when else ""
    context = f" ⟨{claim.context}⟩" if claim.context else ""
    return (
        f"({claim.perspective.viewpoint}, {claim.perspective.stance}, "
        f"{claim.perspective.sentiment}){time}{context}  "
        f"{claim.subject} --{claim.predicate}--> {claim.object}"
    )


def cmd_build(args: argparse.Namespace) -> int:
    config = PipelineConfig(
        backend=args.backend,
        model=args.model,
        judge_model=args.judge_model,
        judge_threshold=args.threshold,
        concurrency=args.concurrency,
        link_dbpedia=args.link_dbpedia,
        spotlight_url=args.spotlight_url,
        spotlight_confidence=args.spotlight_confidence,
    )
    pipeline = Pipeline(config)
    documents = pipeline.load_documents(args.paths)
    if not documents:
        print("No .txt or .md documents found.", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Building graph from {len(documents)} document(s) with backend '{args.backend}' (store={args.store})...")

    if args.store == "sqlite":
        # Stream claims straight to graph.db — the graph is never held in RAM.
        # The in-memory-only exports (json/graphml/ttl) are skipped.
        from .store import GraphStore

        store = GraphStore(out / "graph.db", mode="build")
        try:
            result = pipeline.run(documents, graph=store)
            stats = result.stats()
            _write_report(out, result, stats)
        finally:
            store.close()
        outputs = "graph.db, report.json"
    else:
        result = pipeline.run(documents)
        result.graph.save_json(out / "graph.json")
        result.graph.save_graphml(out / "graph.graphml")
        result.graph.save_turtle(out / "graph.ttl")
        result.graph.save_sqlite(out / "graph.db")
        stats = result.stats()
        _write_report(out, result, stats)
        outputs = "graph.json, graph.graphml, graph.ttl, graph.db, report.json"

    print(json.dumps(stats, indent=2))
    print(f"Wrote {outputs} to {out}/")
    return 0


def _write_report(out: Path, result, stats: dict) -> None:
    report = {
        "stats": stats,
        "rejected": [asdict(c.provenance) | {"claim": _format_claim(c)} for c in result.rejected if c.provenance],
        "errors": result.errors,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_search(args: argparse.Namespace) -> int:
    from .store import retrieve

    hits = retrieve(args.db, args.query, k=args.k)
    if not hits:
        print("No matches.")
        return 0
    for hit in hits:
        snippet = " ".join(hit["text"].split())[:120]
        print(f"[{hit['score']:.3f}] {hit['document_id']} ({hit['chunk_id']})  {snippet}")
    return 0


def cmd_site(args: argparse.Namespace) -> int:
    from .site import build_site

    if str(args.graph).endswith(".db"):
        from .store import GraphStore

        graph = GraphStore(args.graph, mode="query")
        try:
            out = build_site(graph, args.out, claimless=args.claimless)
        finally:
            graph.close()
    else:
        out = build_site(KnowledgeGraph.load_json(args.graph), args.out, claimless=args.claimless)
    print(f"Wrote static site to {out}/ — open {out}/index.html (or: python -m http.server -d {out})")
    return 0


def cmd_link(args: argparse.Namespace) -> int:
    """Enrich an existing graph.db by linking its entities to DBpedia in place."""
    from .link import DBpediaLinker
    from .store import enrich_sqlite

    if not str(args.db).endswith(".db"):
        print("link operates on a graph.db (SQLite) file.", file=sys.stderr)
        return 1
    kwargs = {"confidence": args.spotlight_confidence}
    if args.spotlight_url:
        kwargs["spotlight_url"] = args.spotlight_url
    linker = DBpediaLinker(**kwargs)
    print(f"Linking entities in {args.db} to DBpedia (this sends entity labels + context to the Spotlight service)...")
    n = enrich_sqlite(args.db, linker, relink=args.relink, concurrency=args.concurrency)
    print(f"Linked {n} entit{'y' if n == 1 else 'ies'} to DBpedia.")
    return 0


def cmd_reclassify_types(args: argparse.Namespace) -> int:
    """Fill a heuristic `type_heuristic` column (label-majority type) in place."""
    from .store import reclassify_types

    if not str(args.db).endswith(".db"):
        print("reclassify-types operates on a graph.db (SQLite) file.", file=sys.stderr)
        return 1
    s = reclassify_types(args.db)
    print(
        f"Wrote type_heuristic for {s['total']} entities — {s['changed']} differ from the stored "
        f"type ({s['low_confidence_labels']} multi-sense labels had no clear majority). "
        "Stored in a separate column; the site still shows the original type."
    )
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Union one graph.db into another in place, keeping both graphs' data."""
    from .store import merge_sqlite

    for label, p in (("source", args.src), ("destination", args.dst)):
        if not str(p).endswith(".db"):
            print(f"merge operates on graph.db (SQLite) files; {label} {p!r} is not a .db", file=sys.stderr)
            return 1
        if not Path(p).exists():
            print(f"{label} database not found: {p}", file=sys.stderr)
            return 1
    merge_sqlite(args.src, args.dst)
    print(f"Merged {args.src} into {args.dst}.")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    # A .db graph is queried against SQLite (no full load); .json loads into RAM.
    if str(args.graph).endswith(".db"):
        from .store import GraphStore

        kg = GraphStore(args.graph, mode="query")
    else:
        kg = KnowledgeGraph.load_json(args.graph)

    # Corpus-wide views (no entity required).
    if args.view == "conflicts":
        entity = kg.find_entity(args.entity) if args.entity else None
        groups = kg.conflicts(entity)
        if not groups:
            print("No conflicting claim groups found.")
            return 0
        for i, group in enumerate(groups, 1):
            print(f"Conflict group {i}:")
            for claim in group:
                print(f"  {_format_claim(claim)}")
        return 0

    if args.view == "longtail":
        split = kg.long_tail()
        print("Head entities (frequently mentioned):")
        for label, sense, count in split["head"]:
            print(f"  {count:>3}  {label}{f' ({sense})' if sense else ''}")
        print("Long tail (infrequent):")
        for label, sense, count in split["tail"]:
            print(f"  {count:>3}  {label}{f' ({sense})' if sense else ''}")
        return 0

    if not args.entity:
        print("This view requires an entity name.", file=sys.stderr)
        return 1
    entity = kg.find_entity(args.entity)
    if entity is None:
        known = ", ".join(sorted({e.label for e in kg.entities}))
        print(f"Entity {args.entity!r} not found. Known entities: {known}", file=sys.stderr)
        return 1

    if args.view == "identity":
        senses = kg.senses_of(entity.label)
        print(f"Identities of '{entity.label}' ({len(senses)} sense(s)):")
        for sense_entity in senses:
            profile = kg.identity_profile(sense_entity)
            label = f"{sense_entity.label} ({sense_entity.sense or 'unspecified sense'})"
            print(f"  {label}  [mentions: {profile['mention_count']}]")
            for prop in profile["properties"]:
                ctx = f"  ⟨{prop['context']}⟩" if prop["context"] else ""
                print(f"      {prop['role']}: {prop['predicate']} {prop['other']}{ctx}")
        links = kg.near_identities(entity)
        if links:
            print("Near-identity links:")
            for link in links:
                print(f"  {link.left_id} ~ {link.right_id}  ({link.context})")
    elif args.view in ("change", "timeline"):
        print(f"Change over time for '{entity.label}':")
        for bucket in kg.drift(entity):
            print(f"  [{bucket['when']}]")
            for prop in bucket["properties"]:
                print(f"      {prop}")
        events = kg.evolution_of(entity)
        if events:
            print("Evolution events:")
            for event in events:
                print(f"  {event.kind} @ {event.when or '?'}: {event.note}")
    elif args.view == "perspectives":
        for viewpoint, claims in kg.perspectives_on(entity).items():
            print(f"Perspective: {viewpoint}")
            for claim in claims:
                print(f"  {_format_claim(claim)}")
    elif args.view == "provenance":
        for claim in kg.claims_about(entity):
            print(f"{_format_claim(claim)}")
            print(json.dumps(kg.provenance_of(claim.id), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="idchkg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Run the extraction pipeline over documents")
    build.add_argument("paths", nargs="+", help="Text files or directories of .txt/.md files")
    build.add_argument("-o", "--out", default="out", help="Output directory (default: out)")
    build.add_argument("--backend", default="anthropic", choices=["anthropic", "ollama", "mock"])
    build.add_argument("--model", default=None, help="Model name for the chosen backend")
    build.add_argument("--judge-model", default=None, help="Separate model for the judges")
    build.add_argument("--threshold", type=float, default=0.7, help="Judge pass threshold")
    build.add_argument("--concurrency", type=int, default=4, help="Worker threads for LLM calls (default: 4)")
    build.add_argument(
        "--store", default="memory", choices=["memory", "sqlite"],
        help="memory: build in RAM + write all exports (default). sqlite: stream to graph.db, bounded memory (graph.db only)",
    )
    build.add_argument(
        "--link-dbpedia", action="store_true",
        help="Link entities to DBpedia via Spotlight (opt-in; sends entity labels + context to the Spotlight service)",
    )
    build.add_argument("--spotlight-url", default=None, help="Spotlight REST base URL (default: public api.dbpedia-spotlight.org/en)")
    build.add_argument("--spotlight-confidence", type=float, default=0.5, help="Spotlight/link confidence cutoff (default: 0.5)")
    build.add_argument("-q", "--quiet", action="store_true", help="Only warnings/errors")
    build.add_argument("-v", "--verbose", action="store_true", help="Per-call LLM timing/token metrics (DEBUG)")
    build.set_defaults(func=cmd_build)

    query = sub.add_parser("query", help="Query a previously built graph.json")
    query.add_argument("graph", help="Path to graph.json")
    query.add_argument("entity", nargs="?", default=None, help="Entity label to inspect")
    query.add_argument(
        "--view",
        default="identity",
        choices=[
            # the paper's three challenges
            "identity", "change", "longtail", "provenance",
            # complementary
            "perspectives", "conflicts", "timeline",
        ],
    )
    query.set_defaults(func=cmd_query)

    search = sub.add_parser("search", help="BM25 retrieval over a graph.db chunk index")
    search.add_argument("db", help="Path to graph.db")
    search.add_argument("query", help="Search query")
    search.add_argument("-k", type=int, default=5, help="Number of hits (default: 5)")
    search.set_defaults(func=cmd_search)

    site = sub.add_parser("site", help="Generate a static web site to browse a built graph")
    site.add_argument("graph", help="Path to graph.db or graph.json")
    site.add_argument("-o", "--out", default="site", help="Output directory (default: site)")
    site.add_argument(
        "--claimless", choices=["skip", "keep"], default="skip",
        help="mention-only entities (no claims): skip their pages (default) or keep them with a "
             "note. Either way they're listed in claimless_entities.log",
    )
    site.set_defaults(func=cmd_site)

    merge = sub.add_parser("merge", help="Union one graph.db into another (in place)")
    merge.add_argument("src", help="Source graph.db to merge FROM")
    merge.add_argument("dst", help="Destination graph.db to merge INTO (modified in place)")
    merge.set_defaults(func=cmd_merge)

    link = sub.add_parser("link", help="Link an existing graph.db's entities to DBpedia (in place)")
    link.add_argument("db", help="Path to graph.db")
    link.add_argument("--spotlight-url", default=None, help="Spotlight REST base URL (default: public api.dbpedia-spotlight.org/en)")
    link.add_argument("--spotlight-confidence", type=float, default=0.5, help="Spotlight/link confidence cutoff (default: 0.5)")
    link.add_argument("--relink", action="store_true", help="Re-link entities that already have a DBpedia link")
    link.add_argument("--concurrency", type=int, default=4, help="Parallel linking requests (default: 4)")
    link.set_defaults(func=cmd_link)

    reclassify = sub.add_parser(
        "reclassify-types",
        help="Fill a heuristic type_heuristic column (label-majority type) on an existing graph.db",
    )
    reclassify.add_argument("db", help="Path to graph.db")
    reclassify.set_defaults(func=cmd_reclassify_types)

    args = parser.parse_args(argv)

    # INFO shows progress + the end-of-run timing breakdown; -v adds per-call LLM
    # metrics (tok/s, load time); -q quiets to warnings.
    level = logging.DEBUG if getattr(args, "verbose", False) else (
        logging.WARNING if getattr(args, "quiet", False) else logging.INFO
    )
    logging.basicConfig(
        level=level, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
    )
    # Under -v (root=DEBUG) the HTTP stack floods the log with per-request line
    # noise; keep our own DEBUG (timing/token/cache metrics) but quiet theirs.
    for noisy in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
