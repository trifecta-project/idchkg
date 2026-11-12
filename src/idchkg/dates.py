"""Rule-based normalisation of free-text claim times into year intervals.

Claims carry time as free text (``claims.when_`` / ``valid_from`` / ``valid_to``)
exactly as the extractor phrased it: "in 1663", "1811-16", "17th century",
"about 1650", "from 1692 to middle of the following century", "in dry years".
This is Challenge 2 (change): to place claims on a timeline we need a *year
interval*, not a single point, and we must be honest about what is unknown.

``normalize(text) -> (earliest, latest, edtf)``

* ``earliest`` / ``latest`` are inclusive integer years (``None`` = open/unknown).
  BC years are negative in the historian's sense (``500 BC`` -> ``-500``).
* ``edtf`` is an `Extended Date/Time Format`_ (ISO 8601-2) rendering:
  ``1663``, ``1692/1720``, ``1650~`` (circa), ``16XX`` (a whole century),
  ``176X`` (a decade), ``../1811`` / ``1912/..`` (open-ended). EDTF uses
  astronomical numbering for BC (``-0499`` = 500 BC).
* Non-temporal text ("in dry years", "every morning") -> ``(None, None, None)``.
* Deictic text ("present", "now", "present (at time of writing)") also returns
  ``None``: its meaning depends on the *document's* date, which this function
  does not know. A caller with document metadata could resolve it later.

Deliberately conservative: bare decade words ("the sixties") and relative
phrases ("before the war", "a century ago") are left unresolved because the
anchor century/war is a property of the source document, not the string.

.. _Extended Date/Time Format: https://www.loc.gov/standards/datetime/
"""

from __future__ import annotations

import re

Result = tuple[int | None, int | None, str | None]
_NONE: Result = (None, None, None)

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "twenty-first": 21,
}
_ORD = r"(?:(\d{1,2})(?:st|nd|rd|th)|(" + "|".join(
    sorted(_ORDINAL_WORDS, key=len, reverse=True)) + r"))"

# Named periods that recur in the corpus and have an uncontroversial span.
_PERIODS: dict[str, tuple[int, int]] = {
    "middle ages": (500, 1500),
    "tang dynasty": (618, 907),
    "safavid period": (1501, 1736),
    "the world war": (1914, 1918),
    "world war i": (1914, 1918),
    "first world war": (1914, 1918),
    "great war": (1914, 1918),
    "french and indian war": (1754, 1763),
}

# Qualifiers on a century: fraction of the century they select.
_PART = [
    (r"first half of", (0, 49)),
    (r"second half of|latter half of", (50, 99)),
    (r"early part of|early in|beginning of|the beginning of|start of|early|opening years of", (0, 32)),
    (r"(?:about |toward |towards )?(?:the )?(?:middle|mid)(?: part)? of|mid-|mid |middle", (33, 66)),
    (r"latter part of|late part of|close of|closing years of|end of|toward the close of|"
     r"towards the close of|toward the end of|towards the end of|late", (67, 99)),
]

_CIRCA = r"(?:circa|ca\.?|c\.|about|around|approximately|roughly|abt\.?)"
_YEAR4 = r"(1[0-9]{3}|20[0-2][0-9])"
_ERA = r"(?:\s*(b\.?\s?c\.?(?:e\.?)?|a\.?\s?d\.?|c\.?e\.?)\b)?"

_DEICTIC = re.compile(r"^(?:the )?(?:present|now|today|to-day|nowadays|currently|at present|"
                      r"at the present (?:time|day)|present day|at (?:the )?time of writing)\b")


def _edtf_year(y: int) -> str:
    """EDTF year string; negative input is historian BC (astronomical = y+1)."""
    if y < 0:
        return f"-{(-y - 1):04d}"
    return f"{y:04d}"


def _edtf(lo: int | None, hi: int | None, approx: bool = False) -> str | None:
    if lo is None and hi is None:
        return None
    if lo is not None and lo == hi:
        return _edtf_year(lo) + ("~" if approx else "")
    # Whole centuries/decades get the compact X-notation.
    if lo is not None and hi is not None and lo >= 0 and not approx:
        if lo % 100 == 0 and hi == lo + 99:
            return f"{lo // 100:02d}XX"
        if lo % 10 == 0 and hi == lo + 9:
            return f"{lo // 10:03d}X"
    a = _edtf_year(lo) if lo is not None else ".."
    b = _edtf_year(hi) if hi is not None else ".."
    if approx:
        a = a + "~" if lo is not None else a
        b = b + "~" if hi is not None else b
    return f"{a}/{b}"


def _ok(lo: int | None, hi: int | None, approx: bool = False) -> Result:
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return lo, hi, _edtf(lo, hi, approx)


def _century_num(m: re.Match, g1: int, g2: int) -> int:
    return int(m.group(g1)) if m.group(g1) else _ORDINAL_WORDS[m.group(g2)]


def _century_span(n: int, part: tuple[int, int] | None, bc: bool) -> tuple[int, int]:
    """Year span for the n-th century (AD: 1601-1700 is conventionally the 17th
    century; we use 1600-1699, matching common usage and EDTF ``16XX``)."""
    p0, p1 = part or (0, 99)
    if bc:  # 5th century BC = 500..401 BC -> (-500, -401)
        start = -(n * 100)
        return start + p0, start + p1
    base = (n - 1) * 100
    return base + p0, base + p1


def _part_of(prefix: str) -> tuple[int, int] | None:
    prefix = prefix.strip()
    if not prefix:
        return None
    for pat, span in _PART:
        if re.search(rf"(?:^|\b)(?:{pat})\s*$", prefix):
            return span
    return None


_CENTURY_RE = re.compile(
    rf"(?P<prefix>(?:[a-z]+[\s-]+){{0,4}}?)(?:the\s+)?{_ORD}(?:[\s-]+(?:and|to|or|&)\s+(?:the\s+)?{_ORD})?"
    rf"[\s-]+cent(?:ury|uries|\.)(?P<era>\s*(?:b\.?\s?c\.?|a\.?\s?d\.?))?"
)


def _centuries(t: str) -> Result | None:
    m = _CENTURY_RE.search(t)
    if not m:
        return None
    n1 = _century_num(m, 2, 3)
    n2 = int(m.group(4)) if m.group(4) else (_ORDINAL_WORDS[m.group(5)] if m.group(5) else None)
    bc = bool(m.group("era") and "b" in m.group("era"))
    bc = bc or bool(re.search(r"\bb\.?\s?c\.?\b", t[m.end():]))
    part = _part_of(m.group("prefix") or "")
    lo, hi = _century_span(n1, part if n2 is None else None, bc)
    if n2 is not None:
        lo2, hi2 = _century_span(n2, None, bc)
        lo, hi = min(lo, lo2), max(hi, hi2)
    # "from 1692 to middle of the following century" handled elsewhere.
    prefix = (m.group("prefix") or "").strip()
    if re.search(r"\b(?:since|after|from)$", prefix):
        return lo, None, _edtf(lo, None)
    if re.search(r"\b(?:before|prior to)$", prefix):
        return None, lo - 1, _edtf(None, lo - 1)
    approx = bool(re.search(_CIRCA, prefix))
    return _ok(lo, hi, approx)


def _year_with_era(num: str, era: str | None) -> int:
    y = int(num)
    if era and era.replace(" ", "").replace(".", "").lower().startswith("b"):
        return -y
    return y


def normalize(text: str | None) -> Result:
    """Normalise a free-text time expression to ``(earliest, latest, edtf)``.

    See the module docstring for conventions. Never raises.
    """
    if not text:
        return _NONE
    try:
        return _normalize(text)
    except Exception:  # defensive: a weird string must never break an export
        return _NONE


def _normalize(text: str) -> Result:
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = t.replace("–", "-").replace("—", "-")
    if not t:
        return _NONE
    if _DEICTIC.match(t) and not re.search(r"\d{3,4}", t):
        return _NONE

    # ISO dates as stored in valid_from/valid_to ("1801-01-01").
    m = re.fullmatch(r"(-?\d{4})-\d{2}-\d{2}(?:t[\d:.z+-]*)?", t)
    if m:
        y = int(m.group(1))
        return _ok(y, y)

    for name, (lo, hi) in _PERIODS.items():
        m = re.search(rf"\b{re.escape(name)}\b", t)
        if m:
            if re.search(r"\b(?:before|prior to|until|up to)\s+(?:the\s+)?$", t[: m.start()]):
                return None, lo - 1, _edtf(None, lo - 1)
            if re.search(r"\b(?:after|since|following)\s+(?:the\s+)?$", t[: m.start()]):
                return hi + 1, None, _edtf(hi + 1, None)
            return _ok(lo, hi)

    # BC/AD ranges and years with explicit era (allow 1-4 digits).
    m = re.search(r"\b(\d{1,4})\s*(b\.?\s?c\.?|a\.?\s?d\.?)?\s*(?:-|to|until|and)\s*(\d{1,4})\s*"
                  r"(b\.?\s?c\.?(?:e\.?)?|a\.?\s?d\.?|c\.?e\.?)(?=\W|$)", t)
    if m:
        era2 = m.group(4)
        era1 = m.group(2) or era2
        return _ok(_year_with_era(m.group(1), era1), _year_with_era(m.group(3), era2),
                   bool(re.search(_CIRCA, t[: m.start()])))
    m = re.search(r"(?:\ba\.?\s?d\.?\s*(\d{1,4})\b)|(?:\b(\d{1,4})\s*(b\.?\s?c\.?(?:e\.?)?|a\.?\s?d\.?|c\.?e\.?)(?=\W|$))", t)
    if m and not re.search(r"centur", t):
        y = int(m.group(1)) if m.group(1) else _year_with_era(m.group(2), m.group(3))
        approx = bool(re.search(_CIRCA, t[: m.start()]))
        return _ok(y, y, approx)

    # Bare 3-digit years/ranges (early tea history: "703", "850-922", "in 801").
    m = re.fullmatch(rf"(?:in |about |circa |c\. |ca\. |around )?(\d{{3}})(?:\s*-\s*(\d{{3}}))?", t)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2)) if m.group(2) else y1
        return _ok(y1, y2, bool(re.match(_CIRCA, t)))

    # Millennia BC/AD: "2nd millennium BC".
    m = re.search(r"\b(\d)(?:st|nd|rd|th) millenni(?:um|a)\s*(b\.?\s?c\.?(?:e\.?)?)?", t)
    if m:
        n = int(m.group(1))
        if m.group(2):
            return _ok(-n * 1000, -n * 1000 + 999)
        return _ok((n - 1) * 1000, n * 1000 - 1)

    # Year ranges: "1811-16", "1880-81", "from 1692 to 1720", "between 1860 and 1870".
    m = re.search(rf"{_YEAR4}\s*(?:-|/|to|until|till|and|or|through)\s*(\d{{4}}|\d{{1,2}})\b(?!-\d)(?!\s*(?:per|%|centur))", t)
    if m:
        y1, tail = int(m.group(1)), m.group(2)
        if len(tail) < 4:  # "1811-16", "1832-3": inherit the leading digits
            mod = 10 ** len(tail)
            y2 = (y1 // mod) * mod + int(tail)
            if y2 < y1:
                y2 += mod
        else:
            y2 = int(tail)
        if y1 <= y2 <= y1 + 400:
            return _ok(y1, y2, bool(re.search(_CIRCA, t[: m.start()])))

    # "from 1692 to (the) middle of the following century"
    m = re.search(rf"{_YEAR4}\b.*\b(?:following|next|succeeding) century", t)
    if m:
        y1 = int(m.group(1))
        part = _part_of(t[: t.rfind("following") if "following" in t else len(t)]
                        .rsplit(" to ", 1)[-1].replace("the ", "").strip() + " ")
        nxt = (y1 // 100 + 1) * 100
        p0, p1 = part or (0, 99)
        return _ok(y1, nxt + p1)

    # "1700s" (a century) and decades "1760s", "1780's".
    m = re.search(r"\b(1[0-9]{2}0)'?s\b", t)
    if m:
        lo = int(m.group(1))
        span = 100 if lo % 100 == 0 else 10
        part = _part_of(t[: m.start()].replace("the ", "") + " ")
        if part:
            return _ok(lo + part[0] * span // 100, lo + part[1] * span // 100)
        return _ok(lo, lo + span - 1, bool(re.search(_CIRCA, t[: m.start()])))

    # Centuries (ordinal digits or words; qualifiers; "X and Y centuries").
    r = _centuries(t)
    if r:
        return r

    # Single (or several) four-digit years with optional open-ended qualifiers.
    years = [int(y) for y in re.findall(rf"(?<![\d,.$£])\b{_YEAR4}\b(?!\s*(?:%|per\b|lbs?\b|pounds|bags|tons))", t)]
    if not years:
        return _NONE
    lo, hi = min(years), max(years)
    first = re.search(rf"\b{lo}\b", t)
    before = t[: first.start()] if first else ""
    approx = bool(re.search(rf"{_CIRCA}\s+(?:the year\s+)?$", before)) or \
        bool(re.search(r"^\s*(?:ca\.?|circa|c\.)(?:\W|$)", t[first.end():] if first else ""))
    # Optional "Month [day]," between a qualifier and the year ("before January 1, 1907").
    md = r"(?:the year\s+|[a-z]+\.?\s+(?:\d{1,2}(?:st|nd|rd|th|d)?,?\s+)?|[a-z]+,\s+)?$"
    if len(years) == 1:
        if re.search(r"\b(?:before|prior to|up to|until|till|by|not later than)\s+" + md, before):
            return None, hi, _edtf(None, hi)
        if re.search(r"\b(?:after|since|from|subsequent to|later than|effective)\s+" + md, before) \
                and not re.search(r"\bto\b(?!.*\b(?:present|date|now|day)\b)", t[first.end():] if first else ""):
            return lo, None, _edtf(lo, None)
    return _ok(lo, hi, approx)


def year(text: str | None) -> int | None:
    """Convenience: a single representative year (earliest, else latest)."""
    lo, hi, _ = normalize(text)
    return lo if lo is not None else hi
