from pathlib import Path

import pytest

from idchkg.gutenberg import looks_like_gutenberg, strip_gutenberg_boilerplate

# A miniature etext mirroring the real Gutenberg structure: metadata header,
# START marker, a duplicated "Produced by" credit, the work, END marker, license.
SAMPLE = """The Project Gutenberg eBook of The Book of Tea

Title: The Book of Tea
Author: Kakuzo Okakura
Credits: Produced by Matthew and David Widger

*** START OF THE PROJECT GUTENBERG EBOOK THE BOOK OF TEA ***


Produced by Matthew and David Widger


THE BOOK OF TEA

Tea began as a medicine and grew into a beverage.

*** END OF THE PROJECT GUTENBERG EBOOK THE BOOK OF TEA ***

*** START: FULL LICENSE ***
This eBook is for the use of anyone anywhere... you agree to indemnify...
"""


def test_strips_header_footer_and_credit():
    out = strip_gutenberg_boilerplate(SAMPLE)
    assert out.startswith("THE BOOK OF TEA")
    assert out.endswith("grew into a beverage.")
    # Boilerplate gone from both ends.
    assert "Project Gutenberg" not in out
    assert "indemnify" not in out
    assert "Produced by" not in out  # duplicated credit line dropped


def test_noop_without_markers():
    plain = "Tea began as a medicine and grew into a beverage."
    assert strip_gutenberg_boilerplate(plain) is plain  # unchanged object
    assert not looks_like_gutenberg(plain)


def test_looks_like_gutenberg_detects_markers():
    assert looks_like_gutenberg(SAMPLE)


def test_only_start_marker_trims_header():
    text = "License preamble...\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nThe body."
    assert strip_gutenberg_boilerplate(text) == "The body."


def test_only_end_marker_trims_footer():
    text = "The body.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nFull license text..."
    assert strip_gutenberg_boilerplate(text) == "The body."


def test_legacy_small_print_and_etext_wording():
    text = (
        "*** START OF THIS PROJECT GUTENBERG ETEXT OLD BOOK ***\n"
        "Old body text.\n"
        "*END* THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS\n"
        "legalese..."
    )
    assert strip_gutenberg_boilerplate(text) == "Old body text."


def test_real_book_if_present():
    """The gitignored raw/ books, when fetched, must strip cleanly."""
    path = Path("examples/corpora/food/raw/okakura_book_of_tea.txt")
    if not path.exists():
        pytest.skip("raw corpus not fetched")
    raw = path.read_text(encoding="utf-8")
    out = strip_gutenberg_boilerplate(raw)
    assert 0 < len(out) < len(raw)
    assert "START OF THE PROJECT GUTENBERG" not in out
    assert "END OF THE PROJECT GUTENBERG" not in out
    assert "indemnify" not in out.lower()  # license footer removed
