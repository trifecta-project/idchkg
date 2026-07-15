"""Entity resolution: merge mentions of the same entity across chunks and
documents into a single canonical node.

The default strategy is deterministic (case/whitespace-insensitive label
match plus an explicit alias table). An LLM-based merge judge is a natural
extension point but deliberately not the default - resolution mistakes are
silent and destructive, so the conservative behaviour is to under-merge.
"""

from __future__ import annotations

import re

from .models import Entity, EntityMention, slugify


def _canonical(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().casefold())


class EntityResolver:
    def __init__(self, aliases: dict[str, str] | None = None):
        # alias -> canonical name, both matched case-insensitively
        self._alias_map = {_canonical(a): _canonical(c) for a, c in (aliases or {}).items()}
        self._by_key: dict[str, Entity] = {}

    def _key_for(self, name: str) -> str:
        key = _canonical(name)
        return self._alias_map.get(key, key)

    def add_mention(self, mention: EntityMention) -> Entity:
        key = self._key_for(mention.name)
        entity = self._by_key.get(key)
        if entity is None:
            entity = Entity(
                id=f"ent-{slugify(key)}",
                label=mention.name.strip(),
                type=mention.type,
            )
            self._by_key[key] = entity
        if mention.name.strip() not in entity.aliases and mention.name.strip() != entity.label:
            entity.aliases.append(mention.name.strip())
        if mention.description and mention.description not in entity.descriptions:
            entity.descriptions.append(mention.description)
        # Prefer a specific type over the generic default.
        if entity.type == "concept" and mention.type != "concept":
            entity.type = mention.type
        return entity

    def resolve(self, name: str) -> Entity:
        """Resolve a claim's subject/object string, creating the entity if the
        extractor mentioned it in a claim but not in the entity list."""
        key = self._key_for(name)
        entity = self._by_key.get(key)
        if entity is None:
            return self.add_mention(EntityMention(name=name))
        return entity

    @property
    def entities(self) -> list[Entity]:
        return list(self._by_key.values())
