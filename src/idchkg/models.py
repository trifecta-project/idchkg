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
    name: str
    type: str = "concept"
    description: str = ""

    def canonical_key(self) -> str:
        return re.sub(r"\s+", " ", self.name.strip().casefold())


@dataclass
class Entity:
    """A resolved entity node. Aggregates mentions across chunks/documents."""

    id: str
    label: str
    type: str = "concept"
    aliases: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)


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
