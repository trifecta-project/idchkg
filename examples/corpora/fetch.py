"""Download real food- and maritime-history corpora into the corpora folders.

Two sources, both stdlib-only, run in an environment with open network access
(some sandboxes block gutenberg.org / wikipedia.org):

* Project Gutenberg public-domain books  -> food/raw/,  maritime/raw/
* Wikipedia article plain text (CC BY-SA) -> food/wiki/, maritime/wiki/

    python examples/corpora/fetch.py               # both sources
    python examples/corpora/fetch.py --gutenberg   # books only
    python examples/corpora/fetch.py --wikipedia   # wiki only

    idchkg build examples/corpora --backend ollama --model phi4-ie -o out

The fetched folders are gitignored; the small hand-written samples in food/ and
maritime/ stay committed. Wikipedia text is CC BY-SA — attribute if you
redistribute derived graphs (see README).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# --- Project Gutenberg (public domain) ------------------------------------
# (gutenberg ebook id, output filename). Plain-text URLs are derived below.
FOOD = [
    (28500, "ukers_all_about_coffee.txt"),      # W. H. Ukers, All About Coffee (1922)
    (769, "okakura_book_of_tea.txt"),           # Okakura, The Book of Tea (1906)
    (57340, "simmonds_coffee_and_chicory.txt"), # P. L. Simmonds, Coffee and Chicory
    (46775, "roth_tea_18c_america.txt"),        # Rodris Roth, Tea Drinking in 18th-c America
]
MARITIME = [
    (13529, "mahan_sea_power.txt"),             # Mahan, Influence of Sea Power upon History
    (2055, "dana_two_years_before_the_mast.txt"),  # Dana, Two Years Before the Mast
]

# --- Wikipedia (CC BY-SA) -------------------------------------------------
# Cross-linked history articles from the catalogue; the interlinking exercises
# cross-document entity resolution, multiple senses, and opposing viewpoints.
WIKI = {
    "food": [
        "History of coffee", "History of sugar", "History of tea",
        "Spice trade", "Columbian exchange",
    ],
    "maritime": [
        "Dutch East India Company", "Age of Sail", "Atlantic slave trade",
        "Hanseatic League", "Triangular trade",
    ],
}

HERE = Path(__file__).resolve().parent
USER_AGENT = "idchkg-corpus-fetch (https://github.com/; educational use)"


def _slug(title: str) -> str:
    return title.lower().replace(" ", "_").replace("/", "-") + ".txt"


def gutenberg_urls(ebook_id: int) -> list[str]:
    # Try the common plain-text layouts, newest first.
    return [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
    ]


def _get(url: str, timeout: float = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_gutenberg(ebook_id: int, dest: Path) -> bool:
    for url in gutenberg_urls(ebook_id):
        try:
            data = _get(url)
        except urllib.error.URLError:
            continue
        dest.write_text(data, encoding="utf-8")
        print(f"  ok   {dest.name}  ({len(data):,} chars)")
        return True
    print(f"  FAIL {dest.name}  (ebook {ebook_id} unreachable)", file=sys.stderr)
    return False


def fetch_wikipedia(title: str, dest: Path) -> bool:
    # action=query&prop=extracts&explaintext gives the full article as plain text.
    params = urllib.parse.urlencode(
        {
            "action": "query", "prop": "extracts", "explaintext": "1",
            "redirects": "1", "format": "json", "titles": title,
        }
    )
    try:
        payload = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{params}", timeout=60))
        pages = payload["query"]["pages"]
        page = next(iter(pages.values()))
        text = page.get("extract", "")
    except (urllib.error.URLError, KeyError, StopIteration, json.JSONDecodeError):
        text = ""
    if not text.strip():
        print(f"  FAIL {dest.name}  ('{title}' unreachable or empty)", file=sys.stderr)
        return False
    header = f"# {title}\n# Source: Wikipedia (CC BY-SA 4.0)\n\n"
    dest.write_text(header + text, encoding="utf-8")
    print(f"  ok   {dest.name}  ({len(text):,} chars)")
    return True


def run_gutenberg() -> bool:
    ok = True
    for group, items in (("food", FOOD), ("maritime", MARITIME)):
        out = HERE / group / "raw"
        out.mkdir(parents=True, exist_ok=True)
        print(f"{group}/raw (Project Gutenberg):")
        for ebook_id, name in items:
            ok = fetch_gutenberg(ebook_id, out / name) and ok
    return ok


def run_wikipedia() -> bool:
    ok = True
    for group, titles in WIKI.items():
        out = HERE / group / "wiki"
        out.mkdir(parents=True, exist_ok=True)
        print(f"{group}/wiki (Wikipedia, CC BY-SA):")
        for title in titles:
            ok = fetch_wikipedia(title, out / _slug(title)) and ok
    return ok


def main(argv: list[str]) -> int:
    do_gutenberg = "--wikipedia" not in argv
    do_wikipedia = "--gutenberg" not in argv
    ok = True
    if do_gutenberg:
        ok = run_gutenberg() and ok
    if do_wikipedia:
        ok = run_wikipedia() and ok
    if not ok:
        print("\nSome downloads failed (network likely restricted).", file=sys.stderr)
        return 1
    print("\nDone. Build with e.g.:  idchkg build examples/corpora --backend ollama --model phi4-ie -o out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
