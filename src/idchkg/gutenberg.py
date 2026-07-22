"""Detect and strip Project Gutenberg boilerplate from etext.

Raw Gutenberg books wrap the actual work in a metadata header + license footer.
Left in, that boilerplate becomes chunks the pipeline extracts and *judges* —
wasted LLM spend, plus junk claims/entities (e.g. "user agrees to indemnify the
Foundation"). We remove it before extraction.

The reliable structure across modern PG files is a pair of marker lines:

    *** START OF THE PROJECT GUTENBERG EBOOK <TITLE> ***
    ...the actual book...
    *** END OF THE PROJECT GUTENBERG EBOOK <TITLE> ***

`strip_gutenberg_boilerplate` keeps only what lies between them (tolerating the
older ``E-TEXT``/``ETEXT`` wording, ``THIS``/``THE``, and the legacy
``*END*THE SMALL PRINT!`` footer), and drops the duplicated "Produced by" credit
line PG often repeats just after the START marker. When no markers are found it
returns the text unchanged, so it is a safe no-op on non-Gutenberg input.
"""

from __future__ import annotations

import re

# Marker lines. Tolerant of EBOOK / ETEXT / E-TEXT and THE / THIS; the trailing
# ``***`` is optional (some early files omit it). Anchored per line.
_START = re.compile(r"^\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG.*$", re.IGNORECASE | re.MULTILINE)
_END = re.compile(r"^\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG.*$", re.IGNORECASE | re.MULTILINE)
# Legacy footer used by pre-2000s files instead of an END marker.
_SMALL_PRINT_END = re.compile(r"^\*END[* ].*SMALL PRINT.*$", re.IGNORECASE | re.MULTILINE)

# A transcriber/credits line PG frequently repeats right after START; drop it
# (and its wrapped continuation) so it doesn't lead the extracted text.
_CREDIT_LINE = re.compile(
    r"^\s*(?:Produced by|Credits?:|Transcriber|E-?text (?:prepared|produced) by|This etext was produced by)\b",
    re.IGNORECASE,
)


def looks_like_gutenberg(text: str) -> bool:
    """True if the text carries a Project Gutenberg start/end marker."""
    return bool(_START.search(text) or _END.search(text) or _SMALL_PRINT_END.search(text))


def _drop_leading_credit(body: str) -> str:
    """Strip a leading 'Produced by ...' credit block (up to its blank line)."""
    lines = body.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():  # skip blank lines after START
        i += 1
    if i < len(lines) and _CREDIT_LINE.match(lines[i]):
        i += 1
        while i < len(lines) and lines[i].strip():  # drop wrapped continuation
            i += 1
    return "\n".join(lines[i:])


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return `text` with the Gutenberg header/footer removed.

    Keeps the span after the START marker and before the END (or legacy
    small-print) marker. If only one marker is present, trims that side. If
    neither is present, returns `text` unchanged.
    """
    start = _START.search(text)
    body = text[start.end():] if start else text

    end = _END.search(body) or _SMALL_PRINT_END.search(body)
    if end:
        body = body[: end.start()]

    if start:
        body = _drop_leading_credit(body)

    # Only report a change when a marker actually matched; otherwise hand back
    # the original object so callers can treat it as a no-op.
    return body.strip() if (start or end) else text
