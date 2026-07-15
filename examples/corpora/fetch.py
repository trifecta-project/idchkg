"""Download a curated set of public-domain food- and maritime-history texts
from Project Gutenberg into food/raw/ and maritime/raw/.

Run this in an environment with open network access (some sandboxes block
gutenberg.org). Stdlib only.

    python examples/corpora/fetch.py
    idchkg build examples/corpora/food/raw --backend ollama -o out-food
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

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

HERE = Path(__file__).resolve().parent


def gutenberg_urls(ebook_id: int) -> list[str]:
    # Try the common plain-text layouts, newest first.
    return [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
    ]


def fetch(ebook_id: int, dest: Path) -> bool:
    for url in gutenberg_urls(ebook_id):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "idchkg-corpus-fetch"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            dest.write_text(data, encoding="utf-8")
            print(f"  ok   {dest.name}  ({len(data):,} chars)")
            return True
        except urllib.error.URLError:
            continue
    print(f"  FAIL {dest.name}  (ebook {ebook_id} unreachable)", file=sys.stderr)
    return False


def main() -> int:
    ok = True
    for group, items in (("food", FOOD), ("maritime", MARITIME)):
        out = HERE / group / "raw"
        out.mkdir(parents=True, exist_ok=True)
        print(f"{group}/raw:")
        for ebook_id, name in items:
            ok = fetch(ebook_id, out / name) and ok
    if not ok:
        print("\nSome downloads failed (network likely restricted).", file=sys.stderr)
        return 1
    print("\nDone. Build with e.g.:  idchkg build examples/corpora/food/raw -o out-food")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
