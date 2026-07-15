"""In-memory knowledge graph store.

Backed by a networkx MultiDiGraph. The structure is deliberately
"unflattened": claims are first-class nodes (reified statements) linked to
their subject/object entities, their perspective, and their source document -
so the same pair of entities can be connected by many claims from different
viewpoints and times without collapsing them into one edge.

Node kinds: entity | claim | perspective | document
Edge kinds: subject | object | perspective | source
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from .models import Claim, Document, Entity, slugify


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._claims: dict[str, Claim] = {}
        self._entities: dict[str, Entity] = {}
        self._documents: dict[str, Document] = {}
        self._entity_of_claim: dict[str, tuple[str, str]] = {}  # claim id -> (subj id, obj id)

    # ------------------------------------------------------------------ build

    def add_document(self, document: Document) -> None:
        self._documents[document.id] = document
        self.g.add_node(
            f"doc:{document.id}", kind="document", title=document.title or document.id
        )

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity
        self.g.add_node(
            entity.id,
            kind="entity",
            label=entity.label,
            type=entity.type,
            aliases=list(entity.aliases),
        )

    def add_claim(self, claim: Claim, subject: Entity, obj: Entity) -> None:
        if subject.id not in self._entities:
            self.add_entity(subject)
        if obj.id not in self._entities:
            self.add_entity(obj)
        self._claims[claim.id] = claim
        self._entity_of_claim[claim.id] = (subject.id, obj.id)

        self.g.add_node(
            claim.id,
            kind="claim",
            predicate=claim.predicate,
            stance=claim.perspective.stance,
            sentiment=claim.perspective.sentiment,
            viewpoint=claim.perspective.viewpoint,
            when=claim.time.when or "",
            status=claim.status,
        )
        self.g.add_edge(claim.id, subject.id, kind="subject")
        self.g.add_edge(claim.id, obj.id, kind="object")

        persp_id = f"persp:{claim.perspective.key()}"
        if persp_id not in self.g:
            self.g.add_node(persp_id, kind="perspective", viewpoint=claim.perspective.viewpoint)
        self.g.add_edge(claim.id, persp_id, kind="perspective")

        if claim.provenance:
            doc_node = f"doc:{claim.provenance.document_id}"
            if doc_node not in self.g:
                self.g.add_node(doc_node, kind="document", title=claim.provenance.document_id)
            self.g.add_edge(claim.id, doc_node, kind="source")

    # ------------------------------------------------------------------ access

    @property
    def entities(self) -> list[Entity]:
        return list(self._entities.values())

    @property
    def claims(self) -> list[Claim]:
        return list(self._claims.values())

    def find_entity(self, name: str) -> Entity | None:
        needle = name.strip().casefold()
        for entity in self._entities.values():
            if entity.label.casefold() == needle or any(
                a.casefold() == needle for a in entity.aliases
            ):
                return entity
        # fall back to slug match (e.g. CLI-friendly ids)
        for entity in self._entities.values():
            if entity.id == name or entity.id == f"ent-{slugify(name)}":
                return entity
        return None

    def claims_about(self, entity: Entity) -> list[Claim]:
        """All claims in which the entity is subject or object."""
        return [
            self._claims[cid]
            for cid, (s, o) in self._entity_of_claim.items()
            if entity.id in (s, o)
        ]

    # ------------------------------------------------------------- unflattened views

    def perspectives_on(self, entity: Entity) -> dict[str, list[Claim]]:
        """Challenge 1 (polyvocality): group claims about an entity by the
        viewpoint that makes them."""
        grouped: dict[str, list[Claim]] = defaultdict(list)
        for claim in self.claims_about(entity):
            grouped[claim.perspective.viewpoint].append(claim)
        return dict(grouped)

    def timeline(self, entity: Entity) -> list[Claim]:
        """Challenge 2 (time): claims about an entity ordered by temporal
        scope; untimed claims sort last."""
        return sorted(self.claims_about(entity), key=lambda c: c.time.sort_key())

    def conflicts(self, entity: Entity | None = None) -> list[list[Claim]]:
        """Groups of claims that share subject and predicate but differ in
        object or stance - the disagreements a flat KG would erase."""
        buckets: dict[tuple[str, str], list[Claim]] = defaultdict(list)
        scope = self.claims_about(entity) if entity else self.claims
        for claim in scope:
            subj_id, _ = self._entity_of_claim[claim.id]
            buckets[(subj_id, claim.predicate)].append(claim)
        groups = []
        for claims in buckets.values():
            if len(claims) < 2:
                continue
            objects = {self._entity_of_claim[c.id][1] for c in claims}
            stances = {c.perspective.stance for c in claims}
            if len(objects) > 1 or len(stances) > 1:
                groups.append(claims)
        return groups

    def provenance_of(self, claim_id: str) -> dict:
        """Challenge 3 (provenance): full trace for one claim - source, quote,
        extractor, and every judge verdict."""
        claim = self._claims[claim_id]
        return asdict(claim.provenance) if claim.provenance else {}

    # ------------------------------------------------------------------ export

    def to_dict(self) -> dict:
        return {
            "entities": [asdict(e) for e in self._entities.values()],
            "claims": [
                {**c.to_dict(), "subject_id": s, "object_id": o}
                for c in self._claims.values()
                for s, o in [self._entity_of_claim[c.id]]
            ],
            "documents": [
                {"id": d.id, "title": d.title, "metadata": d.metadata}
                for d in self._documents.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        kg = cls()
        for doc in data.get("documents", []):
            kg.add_document(Document(id=doc["id"], text="", title=doc.get("title", "")))
        entities: dict[str, Entity] = {}
        for item in data.get("entities", []):
            entity = Entity(
                id=item["id"],
                label=item["label"],
                type=item.get("type", "concept"),
                aliases=item.get("aliases", []),
                descriptions=item.get("descriptions", []),
            )
            entities[entity.id] = entity
            kg.add_entity(entity)
        for item in data.get("claims", []):
            claim = Claim.from_dict(item)
            kg.add_claim(claim, entities[item["subject_id"]], entities[item["object_id"]])
        return kg

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load_json(cls, path: str | Path) -> "KnowledgeGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save_graphml(self, path: str | Path) -> None:
        # GraphML supports only scalar attributes.
        export = nx.MultiDiGraph()
        for node, attrs in self.g.nodes(data=True):
            export.add_node(
                node,
                **{
                    k: (", ".join(v) if isinstance(v, list) else v)
                    for k, v in attrs.items()
                },
            )
        for u, v, attrs in self.g.edges(data=True):
            export.add_edge(u, v, **attrs)
        nx.write_graphml(export, str(path))

    def save_turtle(self, path: str | Path) -> None:
        """Serialise as Turtle with reified statements, so the perspective,
        time, and provenance qualifiers survive in RDF."""

        def lit(value: str) -> str:
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

        lines = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix idc: <http://example.org/idchkg#> .",
            "",
        ]
        for entity in self._entities.values():
            lines.append(
                f"idc:{entity.id} a idc:Entity ; rdfs:label {lit(entity.label)} ; "
                f"idc:entityType {lit(entity.type)} ."
            )
        lines.append("")
        for claim in self._claims.values():
            subj_id, obj_id = self._entity_of_claim[claim.id]
            parts = [
                f"idc:{claim.id} a rdf:Statement",
                f"rdf:subject idc:{subj_id}",
                f"rdf:predicate idc:p-{slugify(claim.predicate)}",
                f"rdf:object idc:{obj_id}",
                f"idc:viewpoint {lit(claim.perspective.viewpoint)}",
                f"idc:stance {lit(claim.perspective.stance)}",
                f"idc:sentiment {lit(claim.perspective.sentiment)}",
            ]
            if claim.time.when:
                parts.append(f"idc:when {lit(claim.time.when)}")
            if claim.time.valid_from:
                parts.append(f"idc:validFrom {lit(claim.time.valid_from)}")
            if claim.time.valid_to:
                parts.append(f"idc:validTo {lit(claim.time.valid_to)}")
            if claim.provenance:
                parts.append(f"idc:sourceDocument idc:doc-{slugify(claim.provenance.document_id)}")
                if claim.provenance.quote:
                    parts.append(f"idc:quote {lit(claim.provenance.quote)}")
            lines.append(" ;\n    ".join(parts) + " .")
        Path(path).write_text("\n".join(lines) + "\n")
