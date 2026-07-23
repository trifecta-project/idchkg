"""Entity resolution (Challenge 1: identity).

The paper argues that a global owl:sameAs over-merges: two names are "the same"
only within a context, and a concept like *coffee* has several senses (plant,
beverage, activity, commodity) that current KGs flatten into one node.

So resolution here is keyed by (canonical label, sense): different senses of
the same word become *distinct* entity nodes, related by context-scoped
near-identity rather than merged. The default is deterministic and
conservative - it under-merges on purpose, because a wrong merge is silent and
destructive. mention_count is tracked to expose the head vs long-tail
distribution (Challenge 3).
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from .models import Entity, EntityMention, IdentityLink, slugify

logger = logging.getLogger(__name__)


def _canonical(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().casefold())


class EntityResolver:
    def __init__(self, aliases: dict[str, str] | None = None):
        # alias -> canonical name, both matched case-insensitively
        self._alias_map = {_canonical(a): _canonical(c) for a, c in (aliases or {}).items()}
        self._by_key: dict[tuple[str, str], Entity] = {}
        self._links: dict[tuple[str, str], IdentityLink] = {}
        # Per-entity type resolution: tally every mention's type and use the
        # majority, instead of locking in the first non-"concept" one (which let
        # a single stray label mistype a whole entity, e.g. coffee -> organization).
        self._type_votes: dict[tuple[str, str], Counter] = {}
        # The type the old "first specific type wins" rule would have produced —
        # kept only to report how many entities the majority vote reclassifies.
        self._first_type: dict[tuple[str, str], str] = {}

    def _canonical_label(self, name: str) -> str:
        key = _canonical(name)
        return self._alias_map.get(key, key)

    def _key_for(self, name: str, sense: str = "") -> tuple[str, str]:
        return (self._canonical_label(name), sense.strip().casefold())

    def add_mention(self, mention: EntityMention) -> Entity:
        label_key, sense = self._key_for(mention.name, mention.sense)
        key = (label_key, sense)
        entity = self._by_key.get(key)
        if entity is None:
            entity = Entity(
                id=self._make_id(label_key, sense),
                label=mention.name.strip(),
                type=mention.type,
                sense=sense,
            )
            self._by_key[key] = entity
            self._link_senses(label_key, entity)
            self._first_type[key] = mention.type
        entity.mention_count += 1
        name = mention.name.strip()
        if name and name != entity.label and name not in entity.aliases:
            entity.aliases.append(name)
        if mention.description and mention.description not in entity.descriptions:
            entity.descriptions.append(mention.description)
        if mention.context and mention.context not in entity.contexts:
            entity.contexts.append(mention.context)
        # Majority vote on the type across all mentions.
        votes = self._type_votes.setdefault(key, Counter())
        votes[mention.type] += 1
        entity.type = self._majority_type(votes)
        # Track what the old first-specific-type rule would have chosen.
        if self._first_type[key] == "concept" and mention.type != "concept":
            self._first_type[key] = mention.type
        return entity

    @staticmethod
    def _majority_type(votes: Counter) -> str:
        """The most-voted mention type; ties broken deterministically by type
        name (so e.g. a concept↔organization tie stays 'concept')."""
        return min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def type_vote_stats(self) -> dict:
        """How the majority vote compares to the old rule — for the build log.
        `reclassified` = entities whose type differs from the old rule; `mixed` =
        entities that saw more than one mention type."""
        reclassified = sum(
            1 for key, entity in self._by_key.items()
            if self._first_type.get(key, entity.type) != entity.type
        )
        mixed = sum(1 for votes in self._type_votes.values() if len(votes) > 1)
        return {"total": len(self._by_key), "mixed": mixed, "reclassified": reclassified}

    def resolve(self, name: str, sense: str = "") -> Entity:
        """Resolve a claim's subject/object. If the exact (label, sense) is
        unknown but the label has exactly one sense on record, use it;
        otherwise create a node for this (label, sense)."""
        key = self._key_for(name, sense)
        entity = self._by_key.get(key)
        if entity is not None:
            return entity
        if not sense:
            candidates = [e for (lbl, _), e in self._by_key.items() if lbl == key[0]]
            if len(candidates) == 1:
                return candidates[0]
        return self.add_mention(EntityMention(name=name, sense=sense))

    def _make_id(self, label_key: str, sense: str) -> str:
        base = f"ent-{slugify(label_key)}"
        return f"{base}--{slugify(sense)}" if sense else base

    def _link_senses(self, label_key: str, new_entity: Entity) -> None:
        """A newly seen sense of an existing label is near-identical to the
        other senses of that label, within the 'same word' context."""
        for (lbl, _), other in self._by_key.items():
            if lbl == label_key and other.id != new_entity.id:
                pair = tuple(sorted((other.id, new_entity.id)))
                if pair not in self._links:
                    self._links[pair] = IdentityLink(
                        left_id=pair[0],
                        right_id=pair[1],
                        relation="near_identity",
                        context=f"senses of '{new_entity.label}'",
                        note=f"'{other.sense or 'unspecified'}' vs '{new_entity.sense or 'unspecified'}'",
                    )

    @property
    def entities(self) -> list[Entity]:
        return list(self._by_key.values())

    @property
    def identity_links(self) -> list[IdentityLink]:
        return list(self._links.values())
