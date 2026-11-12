"""Core data model.

The model follows the "unflattening" idea from van Erp (K-CAP 2023):
a knowledge graph should not be a flat set of triples. Every claim is a
first-class, reified statement carrying three extra dimensions:

* a Perspective  - whose voice the claim is in, its stance and connotation
* a TemporalScope - when the claim holds / when it was expressed
* a Provenance   - where it came from and how it was validated (judges)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.strip().casefold())
    return text.strip("-") or "unnamed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Document:
    id: str
    text: str
    title: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str


@dataclass
class Perspective:
    """Whose view a claim represents, and how it is expressed."""

    viewpoint: str = "author"  # the voice: "author" or an attributed source
    stance: str = "asserts"    # asserts | disputes | questions
    sentiment: str = "neutral"  # positive | negative | neutral | ambivalent

    def key(self) -> str:
        return slugify(self.viewpoint)


@dataclass
class TemporalScope:
    """When a claim holds. `when` keeps the source's own time expression;
    valid_from / valid_to are best-effort normalisations (ISO dates or years)."""

    when: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None

    def sort_key(self) -> tuple[int, str]:
        # Claims with a normalised start sort first, then by expression text.
        if self.valid_from:
            return (0, self.valid_from)
        if self.when:
            return (1, self.when)
        return (2, "")

    def is_empty(self) -> bool:
        return not (self.when or self.valid_from or self.valid_to)


@dataclass
class JudgeVerdict:
    criterion: str
    score: float
    passed: bool
    feedback: str = ""
    judge: str = ""


@dataclass
class Provenance:
    document_id: str
    chunk_id: str
    quote: str = ""
    extractor: str = ""
    extracted_at: str = field(default_factory=_now)
    verdicts: list[JudgeVerdict] = field(default_factory=list)


@dataclass
class EntityMention:
    """A mention of an entity in text.

    ``sense`` is the dimension the mention is used in - the paper's Challenge 1
    (identity): the concept *coffee* has a plant sense, a beverage sense, an
    activity sense, a commodity sense. Distinguishing them is the whole point
    of "unflattening" identity, so mentions of different senses become
    different nodes rather than one underspecified node.

    ``context`` is the qualifying scope in which the mention holds (e.g. "in
    the botanical sense", "the ruler of Spain 1939-1975") - the contextual /
    near-identity the paper argues owl:sameAs fails to capture.
    """

    name: str
    type: str = "concept"
    sense: str = ""
    description: str = ""
    context: str = ""

    def canonical_key(self) -> str:
        return re.sub(r"\s+", " ", self.name.strip().casefold())


@dataclass
class Entity:
    """A resolved entity node. Aggregates mentions across chunks/documents.

    An entity is identified by (label, sense): "coffee" the plant and "coffee"
    the beverage are distinct entities, linked by near-identity rather than
    merged (Challenge 1: identity)."""

    id: str
    label: str
    type: str = "concept"
    sense: str = ""
    aliases: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    mention_count: int = 0  # head vs long-tail signal (Challenge 3)
    # DBpedia link (optional, set by the linking step / `idchkg link`):
    # {"uri", "score", "surfaceForm", "types", "dbo_type"} or None when unlinked.
    dbpedia: dict | None = None
    # Heuristic label-majority type (optional, set by `idchkg reclassify-types`).
    # Kept separate from `type` so the site can opt in only if it proves reliable.
    type_heuristic: str | None = None
    # Wikidata link (optional, set by `idchkg link --target wikidata`; retrofit
    # column on graph.db): {"qid", "label", "description", "score", "margin",
    # "types", "coord": [lat, lon] | None, "start", "end"} or None when unlinked.
    wikidata: dict | None = None


@dataclass
class IdentityLink:
    """A context-scoped relationship between two entity nodes - the paper's
    contextual / near-identity (Challenge 1). ``relation`` is typically
    "near_identity" (e.g. two senses of the same word) or "same_as" (only
    valid within ``context``)."""

    left_id: str
    right_id: str
    relation: str = "near_identity"
    context: str = ""
    note: str = ""


# Kinds of identity change over time (Challenge 2)
SPLIT = "split"
MERGE = "merge"
CEASE = "cease"
GAINS = "gains"   # entity gains a role/property
LOSES = "loses"


@dataclass
class EvolutionEvent:
    """How an entity's identity changes over time (Challenge 2: change).
    Records where an identity splits, merges, ceases, or gains/loses a role,
    so the graph is not a snapshot frozen in time."""

    entity_id: str
    kind: str = GAINS
    when: str | None = None
    note: str = ""
    related_id: str | None = None  # the other side of a split/merge


# Claim lifecycle statuses
PENDING = "pending"
ACCEPTED = "accepted"
REPAIRED = "repaired"   # accepted after one judge-guided revision
REJECTED = "rejected"


@dataclass
class Claim:
    """A reified statement: subject --predicate--> object, plus the
    unflattening qualifiers (perspective, time, provenance)."""

    subject: str
    predicate: str
    object: str
    perspective: Perspective = field(default_factory=Perspective)
    time: TemporalScope = field(default_factory=TemporalScope)
    provenance: Provenance | None = None
    status: str = PENDING
    id: str = ""
    # Qualifying scope in which the claim holds, and the sense of its subject /
    # object entities (Challenge 1: identity). Kept after id so positional
    # construction stays compatible.
    context: str = ""
    subject_sense: str = ""
    object_sense: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            basis = "|".join(
                [
                    self.subject.casefold(),
                    self.predicate.casefold(),
                    self.object.casefold(),
                    self.perspective.viewpoint.casefold(),
                    self.perspective.stance,
                    self.time.when or "",
                    self.context.casefold(),
                    self.provenance.chunk_id if self.provenance else "",
                ]
            )
            digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]
            self.id = f"clm-{digest}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Claim":
        prov = data.get("provenance")
        return cls(
            id=data.get("id", ""),
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            perspective=Perspective(**data.get("perspective", {})),
            time=TemporalScope(**data.get("time", {})),
            context=data.get("context", ""),
            subject_sense=data.get("subject_sense", ""),
            object_sense=data.get("object_sense", ""),
            provenance=Provenance(
                document_id=prov["document_id"],
                chunk_id=prov["chunk_id"],
                quote=prov.get("quote", ""),
                extractor=prov.get("extractor", ""),
                extracted_at=prov.get("extracted_at", _now()),
                verdicts=[JudgeVerdict(**v) for v in prov.get("verdicts", [])],
            )
            if prov
            else None,
            status=data.get("status", PENDING),
        )
