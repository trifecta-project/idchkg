"""Entity linking to DBpedia via DBpedia Spotlight.

Given the graph's resolved entities, this attaches each to a DBpedia resource so
the local knowledge graph joins the wider Linked Data cloud. Linking is
**context-aware**: Spotlight disambiguates a surface form using surrounding text,
so we feed each entity's label *plus* its descriptions/contexts (already gathered
during extraction) rather than the bare label — "coffee" the plant vs the
beverage resolve to different resources.

Two kinds of type information are captured for the chosen resource:

* ``types``    - the resource's DBpedia ontology classes (rdf:type, e.g.
                 ``dbo:Beverage``), read for free from Spotlight's ``@types``.
* ``dbo_type`` - the ``dbo:type`` property value(s), which need one extra fetch
                 of the resource's data JSON per entity.

Everything is best-effort: any network or parse failure leaves an entity
unlinked (or a field empty) and never aborts the run. Stdlib ``urllib`` only —
no new dependencies, mirroring ``llm/ollama_client.py``.

The linker is injectable (`Pipeline(..., linker=...)`, `enrich_sqlite(..., linker=...)`)
so tests exercise the wiring with a stub and never touch the network.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

DEFAULT_SPOTLIGHT_URL = "https://api.dbpedia-spotlight.org/en"

# Spotlight reports types with namespace prefixes ("DBpedia:Beverage",
# "Schema:Food", "Wikidata:Q...", "DUL:..."); we keep only the DBpedia ontology
# classes and expand them to full dbo: URIs.
_SPOTLIGHT_DBPEDIA_PREFIX = "DBpedia:"
_DBO_NS = "http://dbpedia.org/ontology/"
_DBO_TYPE = _DBO_NS + "type"


class DBpediaLinker:
    """Links entities to DBpedia resources using a Spotlight server.

    ``spotlight_url`` points at a Spotlight REST base (default: the public
    ``api.dbpedia-spotlight.org/en``; override with a local server for privacy or
    throughput). ``confidence`` is Spotlight's disambiguation cutoff and also the
    minimum similarity a candidate must clear to be accepted here.
    """

    def __init__(
        self,
        spotlight_url: str = DEFAULT_SPOTLIGHT_URL,
        confidence: float = 0.5,
        timeout: float = 30.0,
        fetch_dbo_type: bool = True,
        data_host: str = "https://dbpedia.org",
    ):
        self.spotlight_url = spotlight_url.rstrip("/")
        self.confidence = confidence
        self.timeout = timeout
        self._fetch_dbo_type = fetch_dbo_type
        self.data_host = data_host.rstrip("/")

    # ------------------------------------------------------------- HTTP helpers

    def _get_json(self, url: str, headers: dict) -> dict | list | None:
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            logger.debug("DBpedia request to %s failed: %s", url, exc)
            return None

    def annotate(self, text: str) -> list[dict]:
        """Spotlight-annotate ``text``; return the ``Resources`` list (possibly
        empty). Network/parse failures return ``[]`` rather than raising."""
        if not text.strip():
            return []
        query = urllib.parse.urlencode({"text": text, "confidence": self.confidence})
        body = self._get_json(
            f"{self.spotlight_url}/annotate?{query}", {"Accept": "application/json"}
        )
        if not isinstance(body, dict):
            return []
        resources = body.get("Resources") or []
        return resources if isinstance(resources, list) else []

    def fetch_dbo_type(self, uri: str) -> list[str]:
        """The ``dbo:type`` property value(s) of a DBpedia resource, via its data
        JSON (`<host>/data/<Name>.json`). ``[]`` when absent or on any failure."""
        name = uri.rsplit("/resource/", 1)[-1]
        if not name or name == uri:
            return []
        data = self._get_json(
            f"{self.data_host}/data/{urllib.parse.quote(name)}.json",
            {"Accept": "application/json"},
        )
        if not isinstance(data, dict):
            return []
        preds = data.get(uri)
        if not isinstance(preds, dict):
            return []
        objects = preds.get(_DBO_TYPE) or []
        return [o["value"] for o in objects if isinstance(o, dict) and o.get("value")]

    # ---------------------------------------------------------------- linking

    @staticmethod
    def _context_text(entity) -> str:
        """Label + a little extraction context, to disambiguate the surface form."""
        parts = [entity.label]
        parts += list(entity.descriptions or [])[:1]
        parts += list(entity.contexts or [])[:3]
        return ". ".join(p for p in parts if p)

    @staticmethod
    def _dbpedia_types(resource: dict) -> list[str]:
        raw = resource.get("@types") or ""
        out = []
        for t in raw.split(","):
            t = t.strip()
            if t.startswith(_SPOTLIGHT_DBPEDIA_PREFIX):
                out.append(_DBO_NS + t[len(_SPOTLIGHT_DBPEDIA_PREFIX):])
        return out

    def link_entity(self, entity) -> dict | None:
        """Best DBpedia link for ``entity`` (or ``None``). Prefers the resource
        whose surface form matches the label; only accepts scores ≥ confidence."""
        resources = self.annotate(self._context_text(entity))
        if not resources:
            return None
        label = entity.label.strip().casefold()

        def score(r) -> float:
            try:
                return float(r.get("@similarityScore", 0.0))
            except (TypeError, ValueError):
                return 0.0

        exact = [r for r in resources if (r.get("@surfaceForm", "").strip().casefold() == label)]
        contains = [r for r in resources if label and label in r.get("@surfaceForm", "").strip().casefold()]
        candidates = exact or contains or resources
        best = max(candidates, key=score)
        if score(best) < self.confidence or not best.get("@URI"):
            return None

        uri = best["@URI"]
        return {
            "uri": uri,
            "score": score(best),
            "surfaceForm": best.get("@surfaceForm", ""),
            "types": self._dbpedia_types(best),
            "dbo_type": self.fetch_dbo_type(uri) if self._fetch_dbo_type else [],
        }

    def link_entities(self, entities, concurrency: int = 4) -> int:
        """Set ``entity.dbpedia`` for each entity; return how many were linked.
        Per-entity failures are swallowed (that entity stays unlinked)."""
        entities = list(entities)
        if not entities:
            return 0

        def _one(entity):
            try:
                return self.link_entity(entity)
            except Exception as exc:  # never let one entity abort the batch
                logger.debug("linking %r failed: %s", getattr(entity, "label", "?"), exc)
                return None

        if concurrency > 1 and len(entities) > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(_one, entities))
        else:
            results = [_one(e) for e in entities]

        linked = 0
        for entity, link in zip(entities, results):
            if link:
                entity.dbpedia = link
                linked += 1
        logger.info("DBpedia linking: %d/%d entities linked", linked, len(entities))
        return linked
