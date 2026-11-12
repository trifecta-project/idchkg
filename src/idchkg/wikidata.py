"""Entity linking to Wikidata (opt-in, network-bound, best-effort).

Mirrors :mod:`idchkg.link` (the DBpedia linker) structurally, but targets
Wikidata, which carries what the map/timeline needs directly on the item:
coordinates (P625) and lifespans (P571 inception, P576 dissolved, P580/P582
start/end). That matters for the paper's challenges:

* **Identity (Challenge 1).** Linking is done *per entity node*, i.e. per
  ``(label, sense)``. Two senses of "Java" (the island vs. a colonial
  coffee-producing region) are scored independently against the candidate set
  and may land on different QIDs; nothing here forces senses together.
* **Change (Challenge 2).** Historical names ("Batavia", "Ceylon") should land
  on the *historical* item (Batavia Q1199713, Dutch/British Ceylon) rather than
  the modern successor when the entity's own context says so. We use years in
  the entity's contexts/descriptions and the item's start/end dates as one
  scoring signal, plus "historical/former/colony" cues on both sides.

Pipeline, done in phases so requests batch well and reruns are cheap:

1. ``wbsearchentities`` for each distinct query string (label + up to two
   aliases) -> candidate QIDs with search rank.
2. ``wbgetentities`` (<=50 ids/call) for all candidates -> labels, aliases,
   descriptions, P31 (instance of), P625, P571/P576/P580/P582.
3. ``wbgetentities`` for the P31 *class* items -> class labels (used for type
   compatibility by keyword, e.g. "island", "historical country", "human").
4. Deterministic heuristic scoring per entity (see :meth:`WikidataLinker.score`).
   An LLM judge (like :mod:`idchkg.judge`) could later re-rank the top-k
   candidates for low-margin cases; it would slot in at ``_choose``.

All HTTP goes through one injectable ``fetch(url) -> dict`` callable, so tests
pass canned JSON and never touch the network. Every failure is swallowed per
request: an entity whose lookups fail simply stays unlinked. Responses are
cached on disk (``cache_path``) as *extracted summaries* (not raw payloads), so
a rerun issues no requests for anything seen before.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .dates import normalize as normalize_date

logger = logging.getLogger(__name__)

API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "idchkg/0.1 (knowledge-graph research tool; https://www.wikidata.org/wiki/Wikidata:Data_access)"
_BATCH = 50

# Class-label keywords. Matched as whole words against the English labels of an
# item's P31 classes.
_PLACE_KW = (
    "city", "town", "village", "hamlet", "settlement", "municipality", "capital", "port",
    "country", "state", "colony", "island", "islands", "archipelago", "region", "province",
    "county", "district", "territory", "kingdom", "empire", "republic", "continent",
    "peninsula", "mountain", "mountain range", "river", "sea", "ocean", "bay", "gulf", "lake",
    "valley", "coast", "location", "geographic", "geographical", "administrative",
    "protectorate", "sultanate", "emirate", "caliphate", "dominion", "commune", "metropolis",
    "borough", "oasis", "desert", "plateau", "cape", "strait", "highland", "polity",
    "department", "prefecture", "governorate", "residency", "presidency", "neighborhood",
    "neighbourhood", "quarter", "street", "square", "harbour", "harbor", "building",
    "coffeehouse", "estate", "plantation", "area", "landscape", "subcontinent", "nation",
    "federated state", "canton", "parish", "duchy", "principality", "realm", "land",
    "megacity", "big city", "human settlement", "trading post", "fort", "fortress",
)
_NEG_KW = (
    "human", "family name", "given name", "surname", "disambiguation", "ship", "film",
    "album", "song", "single", "book", "literary work", "painting", "taxon", "species",
    "genus", "band", "musical group", "television", "video game", "newspaper", "periodical",
    "magazine", "scholarly article", "ethnic group", "language", "football club",
    "sports team", "battle", "treaty", "asteroid", "minor planet", "wikimedia", "fictional",
    "character", "episode", "written work", "version", "edition", "brand", "military unit",
    "rail", "railway station", "airport", "school", "university", "company", "business",
    "organization", "event", "election", "constituency", "electoral",
)
_HIST_KW = ("historical", "former", "colony", "ancient", "dissolved", "protectorate",
            "empire", "kingdom", "residency", "presidency", "historic", "old")
_ENTITY_HIST_RE = re.compile(
    r"\b(historical|historically|former|formerly|old name|colonial|colony|ancient|then|"
    r"once|dutch east indies|east india company)\b", re.I)

_STOP = set("""a an the of and or in on at to for from by with as is was were be been are its it
this that these those into which who where when than then also such not but has have had
coffee tea sugar place location""".split())

# Head nouns saying what *kind* of place an entity is ("port city", "island").
_KIND_WORDS = ("city", "town", "village", "port", "capital", "island", "islands", "country",
               "nation", "state", "colony", "region", "province", "district", "continent",
               "kingdom", "empire", "republic", "river", "peninsula", "mountain", "mountains",
               "coast", "sea", "bay", "territory", "archipelago", "county", "coffeehouse",
               "street", "harbour", "harbor")

_TIME_PROPS = {"P571": "inception", "P576": "dissolved", "P580": "start", "P582": "end"}


def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().casefold())
    return re.sub(r"^the ", "", s)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-]+", (text or "").lower())
            if len(w) > 2 and w not in _STOP}


def _has_kw(labels: list[str], kws) -> bool:
    joined = " | ".join(l.lower() for l in labels)
    return any(re.search(rf"\b{re.escape(k)}\b", joined) for k in kws)


def _wd_year(value: dict) -> int | None:
    """Year from a Wikidata time datavalue ``{"time": "+1619-00-00T00:00:00Z", ...}``."""
    t = (value or {}).get("time") if isinstance(value, dict) else None
    m = re.match(r"([+-])(\d+)-", t or "")
    if not m:
        return None
    y = int(m.group(2))
    return -y if m.group(1) == "-" else y


def _claim_values(claims: dict, prop: str) -> list:
    out = []
    for st in claims.get(prop, []) or []:
        if st.get("rank") == "deprecated":
            continue
        dv = (st.get("mainsnak") or {}).get("datavalue") or {}
        if "value" in dv:
            out.append(dv["value"])
    return out


def summarize_item(qid: str, raw: dict) -> dict:
    """Reduce a ``wbgetentities`` entity to the fields the linker uses."""
    claims = raw.get("claims") or {}
    en = lambda d: ((raw.get(d) or {}).get("en") or {}).get("value", "")  # noqa: E731
    aliases = [a.get("value", "") for a in (raw.get("aliases") or {}).get("en", [])]
    p31 = [v.get("id") for v in _claim_values(claims, "P31") if isinstance(v, dict) and v.get("id")]
    coord = None
    for v in _claim_values(claims, "P625"):
        if isinstance(v, dict) and v.get("latitude") is not None and \
                (v.get("globe") or "").endswith("/Q2"):
            coord = [round(float(v["latitude"]), 5), round(float(v["longitude"]), 5)]
            break
    years = {}
    for prop, name in _TIME_PROPS.items():
        ys = [y for y in (_wd_year(v) for v in _claim_values(claims, prop)) if y is not None]
        if ys:
            years[name] = min(ys) if name in ("inception", "start") else max(ys)
    start = years.get("inception", years.get("start"))
    end = years.get("dissolved", years.get("end"))
    return {"qid": qid, "label": en("labels"), "description": en("descriptions"),
            "aliases": aliases, "p31": p31, "coord": coord, "start": start, "end": end,
            "sitelinks": len(raw.get("sitelinks") or {})}


def ent_text_list(entity) -> list[str]:
    return list(entity.descriptions or [])[:20] + list(entity.contexts or [])[:20]


class WikidataLinker:
    """Links entities to Wikidata items with context-aware heuristic scoring.

    ``fetch`` (url -> parsed JSON) is injectable for tests; by default a polite
    urllib GET with a generic User-Agent, a global min-interval throttle and
    retry on 429/maxlag. ``cache_path`` (JSON) persists search/item/class
    summaries across runs. ``min_score`` is the acceptance cutoff.
    """

    def __init__(
        self,
        fetch: Callable[[str], dict | None] | None = None,
        cache_path: str | Path | None = None,
        min_score: float = 2.0,
        min_evidence: float = 0.25,
        search_limit: int = 10,
        timeout: float = 30.0,
        min_interval: float = 0.15,
        api_url: str = API_URL,
        user_agent: str = USER_AGENT,
    ):
        self.api_url = api_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.min_interval = min_interval
        self.min_score = min_score
        self.min_evidence = min_evidence
        self.search_limit = search_limit
        self._fetch = fetch or self._http_get
        self._lock = threading.Lock()
        self._throttle_lock = threading.Lock()
        self._last_request = 0.0
        self.requests = 0
        self.failures = 0
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict = {"search": {}, "items": {}, "classes": {}}
        if self.cache_path and self.cache_path.exists():
            try:
                loaded = json.loads(self.cache_path.read_text())
                for k in self.cache:
                    self.cache[k].update(loaded.get(k) or {})
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("ignoring unreadable Wikidata cache %s: %s", self.cache_path, exc)

    # ------------------------------------------------------------------ HTTP

    def _http_get(self, url: str) -> dict | None:
        for attempt in range(4):
            with self._throttle_lock:  # global politeness: <= 1 request / min_interval
                wait = self._last_request + self.min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last_request = time.monotonic()
            req = urllib.request.Request(url, headers={
                "User-Agent": self.user_agent, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 503) and attempt < 3:
                    delay = float(exc.headers.get("Retry-After") or 5 * (attempt + 1))
                    logger.debug("Wikidata %s; backing off %.0fs", exc.code, delay)
                    time.sleep(min(delay, 60))
                    continue
                raise
            if isinstance(body, dict) and (body.get("error") or {}).get("code") == "maxlag" and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            return body
        return None

    def _get(self, params: dict) -> dict | None:
        url = self.api_url + "?" + urllib.parse.urlencode({**params, "format": "json", "maxlag": 5})
        with self._lock:
            self.requests += 1
        try:
            body = self._fetch(url)
            return body if isinstance(body, dict) else None
        except Exception as exc:  # best-effort: one failed request never aborts
            with self._lock:
                self.failures += 1
            logger.debug("Wikidata request failed (%s): %s", url, exc)
            return None

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        with self._lock:
            data = json.dumps(self.cache, ensure_ascii=False)
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(data)
            tmp.replace(self.cache_path)
        except OSError as exc:
            logger.warning("could not write Wikidata cache %s: %s", self.cache_path, exc)

    # ------------------------------------------------------------- API calls

    def search(self, query: str) -> list[dict] | None:
        """Cached ``wbsearchentities``: ``[{id, label, description, rank}]``.
        Returns ``None`` (uncached) when the request failed."""
        key = _norm(query)
        with self._lock:
            if key in self.cache["search"]:
                return self.cache["search"][key]
        body = self._get({"action": "wbsearchentities", "search": query, "language": "en",
                          "uselang": "en", "type": "item", "limit": self.search_limit})
        if body is None or "search" not in body:
            return None
        hits = [{"id": h.get("id"), "label": h.get("label", ""),
                 "description": h.get("description", ""), "rank": i}
                for i, h in enumerate(body.get("search") or []) if h.get("id")]
        with self._lock:
            self.cache["search"][key] = hits
        return hits

    def _get_entities(self, ids: list[str], props: str) -> dict:
        body = self._get({"action": "wbgetentities", "ids": "|".join(ids), "props": props,
                          "languages": "en"})
        return (body or {}).get("entities") or {}

    def fetch_items(self, qids, concurrency: int = 4) -> None:
        """Ensure item summaries for ``qids`` are cached (batched, <=50/call)."""
        with self._lock:
            todo = sorted({q for q in qids if q and q not in self.cache["items"]})
        batches = [todo[i:i + _BATCH] for i in range(0, len(todo), _BATCH)]

        done = [0]

        def _one(batch):
            ents = self._get_entities(batch, "labels|descriptions|aliases|claims|sitelinks")
            out = {q: summarize_item(q, raw) for q, raw in ents.items() if "missing" not in raw}
            with self._lock:
                self.cache["items"].update(out)
                done[0] += 1
                n = done[0]
            if n % 50 == 0:
                logger.info("Wikidata: fetched %d/%d item batches", n, len(batches))
                self.save_cache()

        self._map(_one, batches, concurrency)

    def fetch_classes(self, qids, concurrency: int = 4) -> None:
        """Ensure English labels of P31 class items are cached."""
        with self._lock:
            todo = sorted({q for q in qids if q and q not in self.cache["classes"]})
        batches = [todo[i:i + _BATCH] for i in range(0, len(todo), _BATCH)]

        def _one(batch):
            ents = self._get_entities(batch, "labels")
            out = {q: ((raw.get("labels") or {}).get("en") or {}).get("value", "")
                   for q, raw in ents.items()}
            with self._lock:
                self.cache["classes"].update(out)

        self._map(_one, batches, concurrency)

    @staticmethod
    def _map(fn, items, concurrency):
        items = list(items)
        if concurrency > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=min(concurrency, 4)) as pool:
                list(pool.map(fn, items))
        else:
            for it in items:
                fn(it)

    # ------------------------------------------------------------- scoring

    @staticmethod
    def _queries(entity) -> list[str]:
        out, seen = [], set()
        for name in [entity.label] + list(entity.aliases or [])[:2]:
            q = re.sub(r"^the\s+", "", (name or "").strip(), flags=re.I)
            if q and _norm(q) not in seen:
                seen.add(_norm(q))
                out.append(q)
        return out

    @staticmethod
    def _entity_years(entity) -> list[int]:
        ys = []
        for text in list(entity.contexts or []) + list(entity.descriptions or []):
            lo, hi, _ = normalize_date(text)
            for y in (lo, hi):
                if y is not None and -3000 < y < 2100:
                    ys.append(y)
        return ys

    def class_labels(self, item: dict) -> list[str]:
        return [self.cache["classes"].get(c, "") for c in item.get("p31", [])]

    def score(self, entity, item: dict, rank: int, place: bool) -> tuple[float, dict]:
        """Heuristic compatibility of ``entity`` with a candidate ``item``.

        Components (all deterministic): name match, search rank, type
        compatibility via P31 class labels, coordinates, lexical overlap of the
        entity's sense/descriptions/contexts with the item description and
        classes, sense keyword hit, historical cues, and time overlap between
        years in the entity's contexts and the item's start/end.
        """
        parts: dict[str, float] = {}
        label = _norm(entity.label)
        names = {_norm(n) for n in [entity.label] + list(entity.aliases or [])}
        if _norm(item.get("label", "")) == label:
            parts["name"] = 1.0
        elif _norm(item.get("label", "")) in names:
            parts["name"] = 0.8
        elif names & {_norm(a) for a in item.get("aliases", [])}:
            parts["name"] = 0.6
        elif label and re.match(rf"{re.escape(label)}\b", _norm(item.get("label", ""))):
            parts["name"] = 0.3  # "New York" ~ "New York City"
        else:
            # Search hit on some other string ("United States" -> "United Mexican
            # States" = Mexico): a different name is evidence *against*.
            parts["name"] = -1.0
        parts["rank"] = 0.6 * max(0.0, 1 - rank / max(self.search_limit, 1))
        parts["popularity"] = min(0.4, 0.004 * item.get("sitelinks", 0))

        classes = [c for c in self.class_labels(item) if c]
        if place:
            if _has_kw(classes, _NEG_KW) and not _has_kw(classes, _PLACE_KW):
                parts["type"] = -3.0
            elif _has_kw(classes, _PLACE_KW):
                parts["type"] = 1.0
            else:
                parts["type"] = 0.0
            parts["coord"] = 0.4 if item.get("coord") else -0.5

        item_text = " ".join([item.get("description", "")] + classes)
        item_tok = _tokens(item_text)
        ent_text = " ".join([entity.sense or ""] + list(entity.descriptions or [])[:8]
                            + list(entity.contexts or [])[:8])
        overlap = len(_tokens(ent_text) & item_tok)
        parts["context"] = min(1.0, 0.25 * overlap)
        sense_tok = _tokens(entity.sense or "")
        parts["sense"] = 0.5 if sense_tok and sense_tok & item_tok else 0.0

        # Where agreement: proper nouns in the entity's text ("in China", "Dutch East
        # Indies") that also appear in the item's description ("...Guangdong, China").
        label_words = set(_norm(entity.label).split())
        proper = {w.lower() for text in ent_text_list(entity)
                  for w in re.findall(r"(?<=[a-z,] )([A-Z][a-z]{2,})", text)
                  if w.lower() not in label_words and w.lower() not in _STOP}
        desc_words = set(re.findall(r"[a-z]+", (item.get("description") or "").lower()))
        if proper & desc_words:
            parts["where"] = 0.6

        # Kind agreement: which place-kind words does the entity's text use (counted
        # across all its descriptions, so the dominant reading wins), and how many of
        # those mentions does the item's description/classes share?
        kind_counts: dict[str, int] = {}
        for text in [entity.sense or ""] + list(entity.descriptions or [])[:20]:
            for w in set(re.findall(r"[a-z]+", text.lower())) & set(_KIND_WORDS):
                kind_counts[w] = kind_counts.get(w, 0) + 1
        if kind_counts:
            hit = sum(c for w, c in kind_counts.items() if w in item_tok)
            parts["kind"] = round(0.8 * hit / sum(kind_counts.values()), 3)

        ent_hist = bool(_ENTITY_HIST_RE.search(ent_text))
        item_hist = _has_kw(classes + [item.get("description", "")], _HIST_KW) or item.get("end") is not None
        parts["historical"] = 0.4 if (ent_hist and item_hist) else 0.0

        years = self._entity_years(entity)
        start, end = item.get("start"), item.get("end")
        if years and (start is not None or end is not None):
            lo = (start if start is not None else -10_000) - 25
            hi = (end if end is not None else 10_000) + 25
            inside = sum(lo <= y <= hi for y in years) / len(years)
            parts["time"] = 0.6 * inside - 0.4 * (1 - inside)
        return round(sum(parts.values()), 3), parts

    def _choose(self, entity, place: bool) -> dict | None:
        """Pick the best candidate (or None). An LLM judge could re-rank the
        top-k here when the margin between the top two is small."""
        ranks: dict[str, int] = {}
        for q in self._queries(entity):
            for h in self.cache["search"].get(_norm(q)) or []:
                ranks[h["id"]] = min(ranks.get(h["id"], 99), h["rank"])
        scored = []
        for qid, rank in ranks.items():
            item = self.cache["items"].get(qid)
            if not item:
                continue
            s, parts = self.score(entity, item, rank, place)
            scored.append((s, -rank, qid, parts))
        if not scored:
            return None
        # A 'place' whose own sense says it is a language/person/ship etc. was
        # mistyped upstream; don't force a geographic link onto it.
        if place and entity.sense and _has_kw([entity.sense], ("language", "person", "ship", "people",
                                                               "dialect", "human", "company")):
            return None
        scored.sort(reverse=True)
        s, _, qid, parts = scored[0]
        if s < self.min_score or (place and parts.get("type", 0) < 0):
            return None
        # Require *some* evidence beyond "a place with this name exists": context /
        # kind / sense / era agreement, or a well-known item. Stops long-tail labels
        # landing on an obscure homonymous hamlet.
        evidence = sum(parts.get(k, 0.0) for k in ("context", "kind", "sense", "historical", "popularity", "where"))
        evidence += max(parts.get("time", 0.0), 0.0)
        if evidence < self.min_evidence:
            return None
        item = self.cache["items"][qid]
        margin = s - scored[1][0] if len(scored) > 1 else s
        return {
            "qid": qid, "label": item.get("label", ""), "description": item.get("description", ""),
            "score": s, "margin": round(margin, 3),
            "types": [c for c in self.class_labels(item) if c],
            "coord": item.get("coord"), "start": item.get("start"), "end": item.get("end"),
        }

    # --------------------------------------------------------------- linking

    def link_entity(self, entity) -> dict | None:
        """Link one entity (runs the request phases for just this entity)."""
        self.link_entities([entity], concurrency=1)
        return getattr(entity, "wikidata", None)

    def link_entities(self, entities, concurrency: int = 4, place_types=("place",),
                      progress_every: int = 500) -> int:
        """Set ``entity.wikidata`` for each entity; return how many were linked.

        ``place_types``: entity types that get the geographic type preference
        (and must not resolve to e.g. a person or a ship)."""
        entities = list(entities)
        if not entities:
            return 0
        concurrency = max(1, min(int(concurrency), 4))

        queries = sorted({q for e in entities for q in self._queries(e)})
        todo = [q for q in queries if _norm(q) not in self.cache["search"]]
        logger.info("Wikidata: %d entities, %d search queries (%d uncached)",
                    len(entities), len(queries), len(todo))
        done = [0]

        def _search(q):
            self.search(q)
            with self._lock:
                done[0] += 1
                n = done[0]
            if n % progress_every == 0:
                logger.info("Wikidata: searched %d/%d", n, len(todo))
                self.save_cache()

        self._map(_search, todo, concurrency)
        self.save_cache()

        cand = {h["id"] for q in queries for h in (self.cache["search"].get(_norm(q)) or [])}
        logger.info("Wikidata: fetching %d candidate items", len(cand))
        self.fetch_items(cand, concurrency)
        self.save_cache()
        classes = {c for q in cand for c in (self.cache["items"].get(q) or {}).get("p31", [])}
        self.fetch_classes(classes, concurrency)
        self.save_cache()

        linked = 0
        for e in entities:
            try:
                link = self._choose(e, place=(e.type in place_types))
            except Exception as exc:  # never let one entity abort the batch
                logger.debug("Wikidata linking %r failed: %s", e.label, exc)
                link = None
            if link:
                e.wikidata = link
                linked += 1
        if self.failures and self.failures >= 0.5 * max(self.requests, 1):
            logger.warning("Wikidata linking: %d of %d requests failed — is %s reachable?",
                           self.failures, self.requests, self.api_url)
        logger.info("Wikidata linking: %d/%d entities linked (%d requests, %d failed)",
                    linked, len(entities), self.requests, self.failures)
        return linked
