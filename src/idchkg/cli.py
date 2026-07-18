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
    )
    pipeline = Pipeline(config)
    documents = pipeline.load_documents(args.paths)
    if not documents:
        print("No .txt or .md documents found.", file=sys.stderr)
        return 1
    print(f"Building graph from {len(documents)} document(s) with backend '{args.backend}'...")
    result = pipeline.run(documents, verbose=not args.quiet)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.graph.save_json(out / "graph.json")
    result.graph.save_graphml(out / "graph.graphml")
    result.graph.save_turtle(out / "graph.ttl")
    result.graph.save_sqlite(out / "graph.db")
    report = {
        "stats": result.stats(),
        "rejected": [asdict(c.provenance) | {"claim": _format_claim(c)} for c in result.rejected if c.provenance],
        "errors": result.errors,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(json.dumps(result.stats(), indent=2))
    print(f"Wrote graph.json, graph.graphml, graph.ttl, graph.db, report.json to {out}/")
    return 0


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


def cmd_query(args: argparse.Namespace) -> int:
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
    build.add_argument("-q", "--quiet", action="store_true")
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
