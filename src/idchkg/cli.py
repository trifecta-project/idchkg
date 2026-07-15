"""Command-line interface.

    idchkg build examples/corpus -o out            # build a graph (LLM calls)
    idchkg query out/graph.json coffee --view perspectives
    idchkg query out/graph.json coffee --view timeline
    idchkg query out/graph.json --view conflicts
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
    return (
        f"({claim.perspective.viewpoint}, {claim.perspective.stance}, "
        f"{claim.perspective.sentiment}){time}  "
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
    report = {
        "stats": result.stats(),
        "rejected": [asdict(c.provenance) | {"claim": _format_claim(c)} for c in result.rejected if c.provenance],
        "errors": result.errors,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(json.dumps(result.stats(), indent=2))
    print(f"Wrote graph.json, graph.graphml, graph.ttl, report.json to {out}/")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    kg = KnowledgeGraph.load_json(args.graph)

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

    if not args.entity:
        print("This view requires an entity name.", file=sys.stderr)
        return 1
    entity = kg.find_entity(args.entity)
    if entity is None:
        known = ", ".join(sorted(e.label for e in kg.entities))
        print(f"Entity {args.entity!r} not found. Known entities: {known}", file=sys.stderr)
        return 1

    if args.view == "perspectives":
        for viewpoint, claims in kg.perspectives_on(entity).items():
            print(f"Perspective: {viewpoint}")
            for claim in claims:
                print(f"  {_format_claim(claim)}")
    elif args.view == "timeline":
        for claim in kg.timeline(entity):
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
        default="perspectives",
        choices=["perspectives", "timeline", "conflicts", "provenance"],
    )
    query.set_defaults(func=cmd_query)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
