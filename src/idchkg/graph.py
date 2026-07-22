"""In-memory knowledge graph store.

Backed by a networkx MultiDiGraph. The structure is deliberately "unflattened"
(after van Erp, K-CAP 2023): claims are first-class reified nodes, and the
graph is organised around the paper's three challenges.

* Challenge 1 - IDENTITY: entities are keyed by (label, sense), so the plant,
  the beverage, and the activity called "coffee" are distinct nodes, joined by
  context-scoped near-identity links rather than one flat owl:sameAs.
* Challenge 2 - CHANGE: claims carry temporal scope and entities carry
  evolution events (split / merge / cease / gains / loses), so identity can be
  followed as it drifts over time instead of a frozen snapshot.
* Challenge 3 - THE LONG TAIL: every claim stores its provenance and judge
  verdicts, and entities keep a mention count so head vs long-tail entities can
  be told apart.

Node kinds: entity | claim | perspective | document
Edge kinds: subject | object | perspective | source | near_identity | evolves
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from .models import Claim, Document, Entity, EvolutionEvent, IdentityLink, slugify

# Plausible year window for temporal binning; drops mis-normalised outliers
# (some corpora carry stray years like 0575 or a century-1 artefact).
_MIN_YEAR, _MAX_YEAR = 1300, 2025


def _valid_from_year(valid_from) -> int | None:
    """The 4-digit year from a normalised ``valid_from`` (e.g. ``1830-01-01``),
    or ``None`` when absent/implausible. Shared by the senses-over-time views."""
    if not valid_from or len(valid_from) < 4 or not valid_from[:4].isdigit():
        return None
    year = int(valid_from[:4])
    return year if _MIN_YEAR <= year <= _MAX_YEAR else None


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._claims: dict[str, Claim] = {}
        self._entities: dict[str, Entity] = {}
        self._documents: dict[str, Document] = {}
        self._entity_of_claim: dict[str, tuple[str, str]] = {}  # claim id -> (subj id, obj id)
        self._identity_links: list[IdentityLink] = []
        self._evolution: list[EvolutionEvent] = []

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
            sense=entity.sense,
            aliases=list(entity.aliases),
            mention_count=entity.mention_count,
        )

    def finalize_entities(self, entities: list[Entity]) -> None:
        """Register every resolver entity (including those never used in an
        accepted claim) and sync mention counts."""
        for entity in entities:
            self.add_entity(entity)

    def add_identity_link(self, link: IdentityLink) -> None:
        self._identity_links.append(link)
        if link.left_id in self.g and link.right_id in self.g:
            self.g.add_edge(
                link.left_id, link.right_id,
                kind="near_identity", relation=link.relation, context=link.context,
            )

    def add_evolution(self, event: EvolutionEvent) -> None:
        self._evolution.append(event)
        if event.related_id and event.entity_id in self.g and event.related_id in self.g:
            self.g.add_edge(
                event.entity_id, event.related_id,
                kind="evolves", evolution=event.kind, when=event.when or "",
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
            context=claim.context,
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

    def entity_count(self) -> int:
        return len(self._entities)

    def document_count(self) -> int:
        return len(self._documents)

    def claim_count(self) -> int:
        return len(self._claims)

    def perspective_stats(self) -> dict:
        """Corpus-wide stance/sentiment tallies + distinct-viewpoint count — the
        Perspective dimension, for the overview charts."""
        from collections import Counter

        stance: Counter = Counter()
        sentiment: Counter = Counter()
        viewpoints = set()
        for c in self._claims.values():
            stance[c.perspective.stance] += 1
            sentiment[c.perspective.sentiment] += 1
            viewpoints.add(c.perspective.viewpoint)
        return {"stance": dict(stance), "sentiment": dict(sentiment), "viewpoints": len(viewpoints)}

    def senses_over_time(self, labels) -> dict:
        """Dated-claim counts per sense per century for a concept, `{sense:
        {century_start_year: count}}` (e.g. 1600 = the 1600s). `labels` is a
        single label or a list of labels to **collapse** into one concept (their
        senses are pooled). Feeds the 'senses over time' stream graph (Challenge
        1 identity × Challenge 2 change). Uses the normalised `valid_from` year
        only, clamped to a plausible window, mirroring the GraphStore SQL path."""
        from collections import Counter, defaultdict

        if isinstance(labels, str):
            labels = [labels]
        per_sense: dict = defaultdict(Counter)
        seen: set = set()  # a shared entity (e.g. via aliases) is counted once
        for label in labels:
            for entity in self.senses_of(label):
                if not entity.sense or entity.id in seen:
                    continue
                seen.add(entity.id)
                for claim in self.claims_about(entity):
                    year = _valid_from_year(claim.time.valid_from)
                    if year is not None:
                        per_sense[entity.sense][(year // 100) * 100] += 1
        return {sense: dict(counts) for sense, counts in per_sense.items() if counts}

    def documents(self) -> list[Document]:
        return list(self._documents.values())

    def claims_in_document(self, document_id: str) -> list[Claim]:
        return [
            c for c in self._claims.values()
            if c.provenance and c.provenance.document_id == document_id
        ]

    def dated_claims(self) -> list[Claim]:
        """Claims with any temporal scope, in chronological order (for a timeline)."""
        return sorted(
            (c for c in self._claims.values() if not c.time.is_empty()),
            key=lambda c: c.time.sort_key(),
        )

    @property
    def identity_links(self) -> list[IdentityLink]:
        return list(self._identity_links)

    def find_entity(self, name: str, sense: str | None = None) -> Entity | None:
        needle = name.strip().casefold()
        matches = [
            e for e in self._entities.values()
            if e.label.casefold() == needle or any(a.casefold() == needle for a in e.aliases)
        ]
        if sense is not None:
            matches = [e for e in matches if e.sense == sense.strip().casefold()]
        if matches:
            # Prefer the most-mentioned sense when the caller didn't specify one.
            return max(matches, key=lambda e: e.mention_count)
        for entity in self._entities.values():
            if entity.id in (name, f"ent-{slugify(name)}"):
                return entity
        return None

    def claims_about(self, entity: Entity) -> list[Claim]:
        return [
            self._claims[cid]
            for cid, (s, o) in self._entity_of_claim.items()
            if entity.id in (s, o)
        ]

    def subgraph_of(self, entity: Entity) -> "KnowledgeGraph":
        """The graph on which this entity's views run. For the in-memory graph
        that's the graph itself (its views already filter to the entity) — the
        method exists so callers can build a per-entity view source once and
        share it across views, matching GraphStore's DB-backed subgraph."""
        return self

    # ------------------------------------------- Challenge 1: identity

    def senses_of(self, label: str) -> list[Entity]:
        """The distinct sense-entities sharing a label - the un-flattened
        identity of a multi-dimensional concept (e.g. coffee: plant / beverage
        / activity / commodity)."""
        needle = label.strip().casefold()
        senses = [
            e for e in self._entities.values()
            if e.label.casefold() == needle or any(a.casefold() == needle for a in e.aliases)
        ]
        return sorted(senses, key=lambda e: (e.sense or "", -e.mention_count))

    def identity_profile(self, entity: Entity) -> dict:
        """The set of properties that define an entity in its context - what
        the paper calls a richer identity profile."""
        properties = []
        for claim in self.claims_about(entity):
            subj_id, obj_id = self._entity_of_claim[claim.id]
            role = "subject" if subj_id == entity.id else "object"
            properties.append(
                {
                    "role": role,
                    "predicate": claim.predicate,
                    "other": claim.object if role == "subject" else claim.subject,
                    "context": claim.context,
                    "when": claim.time.when,
                }
            )
        return {
            "id": entity.id,
            "label": entity.label,
            "sense": entity.sense,
            "type": entity.type,
            "contexts": entity.contexts,
            "mention_count": entity.mention_count,
            "properties": properties,
        }

    def near_identities(self, entity: Entity) -> list[IdentityLink]:
        return [
            link for link in self._identity_links
            if entity.id in (link.left_id, link.right_id)
        ]

    # ------------------------------------------- Challenge 2: change

    def timeline(self, entity: Entity) -> list[Claim]:
        """Claims about an entity ordered by temporal scope; untimed last."""
        return sorted(self.claims_about(entity), key=lambda c: c.time.sort_key())

    def drift(self, entity: Entity) -> list[dict]:
        """Concept drift: for each distinct temporal scope, the properties the
        entity holds - so you can see what changes as it evolves over time."""
        buckets: dict[str, list[str]] = defaultdict(list)
        for claim in self.timeline(entity):
            when = claim.time.when or "undated"
            subj_id, _ = self._entity_of_claim[claim.id]
            other = claim.object if subj_id == entity.id else claim.subject
            buckets[when].append(f"{claim.predicate} {other}")
        order = [c.time.when or "undated" for c in self.timeline(entity)]
        seen: list[str] = []
        for when in order:
            if when not in seen:
                seen.append(when)
        return [{"when": when, "properties": buckets[when]} for when in seen]

    def evolution_of(self, entity: Entity) -> list[EvolutionEvent]:
        return [e for e in self._evolution if entity.id in (e.entity_id, e.related_id)]

    # ------------------------------------------- Challenge 3: the long tail

    def long_tail(self, head_fraction: float = 0.2) -> dict:
        """Split entities into head (frequently mentioned) and long tail. The
        head is the top `head_fraction` of entities by mention count."""
        ranked = sorted(self._entities.values(), key=lambda e: e.mention_count, reverse=True)
        if not ranked:
            return {"head": [], "tail": []}
        cut = max(1, int(len(ranked) * head_fraction))
        return {
            "head": [(e.label, e.sense, e.mention_count) for e in ranked[:cut]],
            "tail": [(e.label, e.sense, e.mention_count) for e in ranked[cut:]],
        }

    def provenance_of(self, claim_id: str) -> dict:
        """Full source trace for one claim - the provenance the paper insists
        must be stored so long-tail facts don't disappear."""
        claim = self._claims[claim_id]
        return asdict(claim.provenance) if claim.provenance else {}

    # ------------------------------------------- complementary views

    def perspectives_on(self, entity: Entity) -> dict[str, list[Claim]]:
        """Claims about an entity grouped by the viewpoint that makes them.
        (Not one of the paper's three challenges, but drawn from its remark
        that concepts "connect to many different narratives with different
        opinions".)"""
        grouped: dict[str, list[Claim]] = defaultdict(list)
        for claim in self.claims_about(entity):
            grouped[claim.perspective.viewpoint].append(claim)
        return dict(grouped)

    def conflicts(self, entity: Entity | None = None) -> list[list[Claim]]:
        """Claim groups sharing subject+predicate but differing in object or
        stance - disagreements a flat KG would silently reconcile."""
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
            "identity_links": [asdict(link) for link in self._identity_links],
            "evolution": [asdict(e) for e in self._evolution],
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
                sense=item.get("sense", ""),
                aliases=item.get("aliases", []),
                descriptions=item.get("descriptions", []),
                contexts=item.get("contexts", []),
                mention_count=item.get("mention_count", 0),
                dbpedia=item.get("dbpedia"),
            )
            entities[entity.id] = entity
            kg.add_entity(entity)
        for item in data.get("claims", []):
            claim = Claim.from_dict(item)
            kg.add_claim(claim, entities[item["subject_id"]], entities[item["object_id"]])
        for link in data.get("identity_links", []):
            kg.add_identity_link(IdentityLink(**link))
        for event in data.get("evolution", []):
            kg.add_evolution(EvolutionEvent(**event))
        return kg

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load_json(cls, path: str | Path) -> "KnowledgeGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save_sqlite(self, path: str | Path) -> None:
        """Persist the graph + a BM25 chunk index to a single SQLite file."""
        from .store import save_sqlite

        save_sqlite(self, path)

    @classmethod
    def load_sqlite(cls, path: str | Path) -> "KnowledgeGraph":
        from .store import load_sqlite

        return load_sqlite(path)

    def save_graphml(self, path: str | Path) -> None:
        # GraphML supports only scalar attributes.
        export = nx.MultiDiGraph()
        for node, attrs in self.g.nodes(data=True):
            export.add_node(
                node,
                **{k: (", ".join(v) if isinstance(v, list) else v) for k, v in attrs.items()},
            )
        for u, v, attrs in self.g.edges(data=True):
            export.add_edge(u, v, **attrs)
        nx.write_graphml(export, str(path))

    def save_turtle(self, path: str | Path) -> None:
        """Serialise as Turtle with reified statements, so perspective, time,
        context, and provenance qualifiers survive in RDF."""

        def lit(value: str) -> str:
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

        lines = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix idc: <http://example.org/idchkg#> .",
            "",
        ]
        for entity in self._entities.values():
            parts = [
                f"idc:{entity.id} a idc:Entity",
                f"rdfs:label {lit(entity.label)}",
                f"idc:entityType {lit(entity.type)}",
            ]
            if entity.sense:
                parts.append(f"idc:sense {lit(entity.sense)}")
            lines.append(" ; ".join(parts) + " .")
        lines.append("")
        for link in self._identity_links:
            lines.append(
                f"idc:{link.left_id} idc:nearIdentity idc:{link.right_id} ."
                f"  # {link.context}"
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
            if claim.context:
                parts.append(f"idc:context {lit(claim.context)}")
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
