"""Prototype: a single-file, browser-based OSINT-style map over an idchkg graph.db.

    python examples/osint_map.py commodities_out/graph.db -o commodities_out/osint_map.html

Place entities are geocoded from their Wikidata link (``entities.wikidata``, filled by
``idchkg link graph.db --target wikidata --types place``) when present, falling back
to the site's small built-in gazetteer (``site._GEOCODE``); every claim touching a geocoded place is embedded as JSON alongside a MapLibre
map with linked filters (time, viewpoint, stance, commodity, long tail). Claims
whose subject *and* object are places become arcs. Senses stay distinct: a map
location lists every (label, sense) node that sits there rather than merging them.

Times are normalised with ``idchkg.dates.normalize`` into a year interval: claims
carry ``y0``/``y1`` (either may be null for open-ended "after 1912"/"before 1811")
and ``y`` = the representative year = ``y0`` if known, else ``y1``. Senses carry the
Wikidata ``qid``/``wdesc``/``start``/``end`` when linked.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from idchkg.dates import normalize as normalize_date
from idchkg.site import _DOC_AUTHORS, _GEOCODE

COMMODITIES = {
    "coffee": ["coffee", "mocha", "arabica", "robusta"],
    "tea": ["tea", "bohea", "hyson", "souchong"],
    "sugar": ["sugar", "cane", "molasses", "rum", "beet"],
    "spice": ["spice", "pepper", "clove", "nutmeg", "cinnamon", "mace", "ginger"],
    "cocoa": ["cocoa", "cacao", "chocolate"],
    "tobacco": ["tobacco"],
}
def parse_interval(when_: str | None, vfrom: str | None = None,
                   vto: str | None = None) -> tuple[int | None, int | None]:
    """Year interval for a claim: ``when_`` first, else ``valid_from``/``valid_to``."""
    lo, hi, _ = normalize_date(when_)
    if lo is None and hi is None:
        lo = normalize_date(vfrom)[0]
        hi = normalize_date(vto)[1] if vto else (lo if lo is not None else None)
        if lo is None and hi is None:
            return None, None
    return lo, hi


def parse_year(*texts: str | None) -> int | None:
    """Representative year (earliest bound, else latest) of the first parseable text."""
    for t in texts:
        lo, hi, _ = normalize_date(t)
        if lo is not None or hi is not None:
            return lo if lo is not None else hi
    return None


def commodities_of(text: str) -> list[str]:
    low = text.lower()
    return [c for c, words in COMMODITIES.items()
            if any(re.search(rf"\b{w}", low) for w in words)]


def export(db_path: Path) -> dict:
    db = sqlite3.connect(db_path)
    has_wd = any(r[1] == "wikidata" for r in db.execute("PRAGMA table_info(entities)"))
    wd_col = "wikidata" if has_wd else "NULL"
    places: dict[str, dict] = {}
    for eid, label, sense, mentions, aliases, wd in db.execute(
        f"SELECT id, label, sense, mention_count, aliases, {wd_col} FROM entities WHERE type='place'"
    ):
        p = {"id": eid, "label": label, "sense": sense or "", "mentions": mentions or 0}
        link = json.loads(wd) if wd else None
        if link:
            p.update({"qid": link.get("qid"), "wdesc": link.get("description", ""),
                      "start": link.get("start"), "end": link.get("end")})
        coords = tuple(link["coord"]) if link and link.get("coord") else None
        if coords:
            p["geo"] = "wikidata"
        else:
            for name in [label] + (json.loads(aliases) if aliases else []):
                coords = _GEOCODE.get(name.strip().casefold())
                if coords:
                    p["geo"] = "gazetteer"
                    break
        if coords:
            p["lat"], p["lon"] = round(coords[0], 4), round(coords[1], 4)
            places[eid] = p

    # Group senses sharing a location into one map site (keeps senses visible, distinct).
    sites: dict[tuple, dict] = {}
    for p in places.values():
        key = (p["lat"], p["lon"])
        s = sites.setdefault(key, {"lat": p["lat"], "lon": p["lon"], "senses": []})
        s["senses"].append(p)
    site_list = list(sites.values())
    for i, s in enumerate(site_list):
        s["i"] = i
        s["senses"].sort(key=lambda p: -p["mentions"])
        s["name"] = s["senses"][0]["label"]
        s["mentions"] = sum(p["mentions"] for p in s["senses"])
    site_of = {p["id"]: sites[(p["lat"], p["lon"])]["i"] for p in places.values()}
    sense_ix = {p["id"]: j for s in site_list for j, p in enumerate(s["senses"])}

    claims = []
    for (cid, subj, pred, obj, sid, oid, status, viewpoint, stance, sentiment,
         when_, vfrom, vto, doc, quote) in db.execute(
        "SELECT id, subject, predicate, object, subject_id, object_id, status, viewpoint, "
        "stance, sentiment, when_, valid_from, valid_to, document_id, quote FROM claims"
    ):
        if sid not in places and oid not in places:
            continue
        vp = viewpoint or "unknown"
        if vp.strip().lower() == "author" and doc in _DOC_AUTHORS:
            vp = f"{_DOC_AUTHORS[doc]} (author)"
        text = f"{subj} {pred} {obj}"
        y0, y1 = parse_interval(when_, vfrom, vto)
        claims.append({
            "s": subj, "p": pred, "o": obj,
            "a": site_of.get(sid), "b": site_of.get(oid),
            "as": sense_ix.get(sid), "bs": sense_ix.get(oid),
            "y": y0 if y0 is not None else y1, "y0": y0, "y1": y1, "w": when_ or vfrom or "",
            "v": vp, "st": stance or "asserts", "se": sentiment or "",
            "d": doc, "q": (quote or "")[:400], "r": status == "repaired",
            "c": commodities_of(text + " " + (quote or "")),
        })
    docs = {d: t for d, t in db.execute("SELECT id, title FROM documents")}
    return {"sites": site_list, "claims": claims, "docs": docs,
            "commodities": list(COMMODITIES)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("db", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("osint_map.html"))
    args = ap.parse_args()
    data = export(args.db)
    template = (Path(__file__).parent / "osint_map_template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    args.out.write_text(template.replace("/*__DATA__*/null", payload))
    print(f"{len(data['sites'])} sites, {len(data['claims'])} claims -> {args.out}")


if __name__ == "__main__":
    main()
