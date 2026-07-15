"""Ingestion (chunking) and the LLM extraction stage."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import LLMClient
from .models import Chunk, Claim, Document, EntityMention, Perspective, Provenance, TemporalScope
from .prompts import (
    EXTRACTION_SCHEMA,
    EXTRACTION_SYSTEM,
    SENTIMENTS,
    STANCES,
    extraction_user,
    repair_user,
)


def chunk_document(document: Document, max_chars: int = 1600) -> list[Chunk]:
    """Split a document into paragraph-aligned chunks of roughly max_chars."""
    paragraphs = [p.strip() for p in document.text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            chunks.append(
                Chunk(
                    id=f"{document.id}#c{len(chunks)}",
                    document_id=document.id,
                    text="\n\n".join(buffer),
                )
            )
            buffer, size = [], 0

    for paragraph in paragraphs:
        if size and size + len(paragraph) > max_chars:
            flush()
        buffer.append(paragraph)
        size += len(paragraph)
    flush()
    return chunks


@dataclass
class Extraction:
    entities: list[EntityMention]
    claims: list[Claim]
    errors: list[str]


class Extractor:
    """Turns a chunk of text into entity mentions and qualified claims."""

    def __init__(self, client: LLMClient):
        self.client = client

    def extract(self, chunk: Chunk) -> Extraction:
        raw = self.client.complete_json(
            EXTRACTION_SYSTEM, extraction_user(chunk.text), EXTRACTION_SCHEMA
        )
        return self._parse(raw, chunk)

    def repair(self, chunk: Chunk, claim: Claim, feedback: str) -> Extraction:
        """Re-extract a single claim using judge feedback."""
        claim_json = json.dumps(
            {
                "subject": claim.subject,
                "predicate": claim.predicate,
                "object": claim.object,
                "perspective": vars(claim.perspective),
                "time": vars(claim.time),
                "quote": claim.provenance.quote if claim.provenance else "",
            },
            indent=2,
        )
        raw = self.client.complete_json(
            EXTRACTION_SYSTEM, repair_user(chunk.text, claim_json, feedback), EXTRACTION_SCHEMA
        )
        return self._parse(raw, chunk)

    def _parse(self, raw: dict, chunk: Chunk) -> Extraction:
        entities: list[EntityMention] = []
        claims: list[Claim] = []
        errors: list[str] = []

        for item in raw.get("entities", []):
            name = str(item.get("name", "")).strip()
            if not name:
                errors.append("entity with empty name skipped")
                continue
            entities.append(
                EntityMention(
                    name=name,
                    type=str(item.get("type", "concept")) or "concept",
                    description=str(item.get("description", "")),
                )
            )

        for item in raw.get("claims", []):
            problem = _validate_claim_dict(item)
            if problem:
                errors.append(f"claim skipped: {problem}: {json.dumps(item)[:120]}")
                continue
            perspective_data = item["perspective"]
            time_data = item["time"]
            claims.append(
                Claim(
                    subject=item["subject"].strip(),
                    predicate=item["predicate"].strip().casefold(),
                    object=item["object"].strip(),
                    perspective=Perspective(
                        viewpoint=perspective_data["viewpoint"].strip() or "author",
                        stance=perspective_data["stance"],
                        sentiment=perspective_data["sentiment"],
                    ),
                    time=TemporalScope(
                        when=time_data.get("when") or None,
                        valid_from=time_data.get("valid_from") or None,
                        valid_to=time_data.get("valid_to") or None,
                    ),
                    provenance=Provenance(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        quote=str(item.get("quote", "")),
                        extractor=self.client.name,
                    ),
                )
            )
        return Extraction(entities=entities, claims=claims, errors=errors)


def _validate_claim_dict(item: dict) -> str | None:
    """Deterministic schema gate, independent of what the backend guarantees."""
    for key in ("subject", "predicate", "object"):
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"missing or empty {key}"
    perspective = item.get("perspective")
    if not isinstance(perspective, dict):
        return "missing perspective"
    if perspective.get("stance") not in STANCES:
        return f"invalid stance {perspective.get('stance')!r}"
    if perspective.get("sentiment") not in SENTIMENTS:
        return f"invalid sentiment {perspective.get('sentiment')!r}"
    if not isinstance(perspective.get("viewpoint"), str):
        return "missing viewpoint"
    if not isinstance(item.get("time"), dict):
        return "missing time"
    return None
