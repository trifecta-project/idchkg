"""Table-driven tests for idchkg.dates.normalize, using strings that really occur
in claims.when_ of the commodities corpus."""

import pytest

from idchkg.dates import normalize, year

CASES = [
    # single years / full dates
    ("1663", (1663, 1663, "1663")),
    ("in 1663", (1663, 1663, "1663")),
    ("In 1801", (1801, 1801, "1801")),
    ("in the year 1699", (1699, 1699, "1699")),
    ("September 16, 1736", (1736, 1736, "1736")),
    ("May 3d, A. D. 1493", (1493, 1493, "1493")),
    ("1914 (fis. yr.)", (1914, 1914, "1914")),
    ("calendar year 1921", (1921, 1921, "1921")),
    ("November 14-17, 1916", (1916, 1916, "1916")),
    ("March 29 to April 4, 1920", (1920, 1920, "1920")),
    # ranges
    ("1811-16", (1811, 1816, "1811/1816")),
    ("five years, 1811-16", (1811, 1816, "1811/1816")),
    ("1880-81", (1880, 1881, "1880/1881")),
    ("1906-07", (1906, 1907, "1906/1907")),
    ("1832-3", (1832, 1833, "1832/1833")),
    ("1899-00", (1899, 1900, "1899/1900")),
    ("from 1692 to 1720", (1692, 1720, "1692/1720")),
    ("from 1801 to 1804", (1801, 1804, "1801/1804")),
    ("between 1860 and 1870", (1860, 1870, "1860/1870")),
    ("1789 to 1921", (1789, 1921, "1789/1921")),
    ("tax year 1915-16", (1915, 1916, "1915/1916")),
    ("1669 or 1672", (1669, 1672, "1669/1672")),
    ("from 1692 to middle of the following century", (1692, 1766, "1692/1766")),
    ("1805-01-01", (1805, 1805, "1805")),
    ("1804-12-31", (1804, 1804, "1804")),
    # open-ended
    ("after 1912", (1912, None, "1912/..")),
    ("since 1914", (1914, None, "1914/..")),
    ("after July 1, 1864", (1864, None, "1864/..")),
    ("before 1811", (None, 1811, "../1811")),
    ("before January 1, 1907", (None, 1907, "../1907")),
    ("up to 1850", (None, 1850, "../1850")),
    ("by 1814", (None, 1814, "../1814")),
    ("from 1835 down to the present", (1835, None, "1835/..")),
    # circa
    ("about 1650", (1650, 1650, "1650~")),
    ("About 1570", (1570, 1570, "1570~")),
    ("ca. 1730", (1730, 1730, "1730~")),
    ("1735 ca.", (1735, 1735, "1735~")),
    ("about the year 1808", (1808, 1808, "1808~")),
    ("roughly 1889", (1889, 1889, "1889~")),
    # decades / "1700s"
    ("1760s", (1760, 1769, "176X")),
    ("1780's", (1780, 1789, "178X")),
    ("the 1760s", (1760, 1769, "176X")),
    ("1700s", (1700, 1799, "17XX")),
    ("early 1700s", (1700, 1732, "1700/1732")),
    # centuries
    ("17th century", (1600, 1699, "16XX")),
    ("18th century", (1700, 1799, "17XX")),
    ("in the 18th century", (1700, 1799, "17XX")),
    ("18th-century America", (1700, 1799, "17XX")),
    ("the sixteenth century", (1500, 1599, "15XX")),
    ("seventeenth century", (1600, 1699, "16XX")),
    ("17th cent.", (1600, 1699, "16XX")),
    ("early 18th century", (1700, 1732, "1700/1732")),
    ("early part of the nineteenth century", (1800, 1832, "1800/1832")),
    ("mid-18th century", (1733, 1766, "1733/1766")),
    ("middle of the seventeenth century", (1633, 1666, "1633/1666")),
    ("about the middle of the seventeenth century", (1633, 1666, "1633~/1666~")),
    ("late 18th century", (1767, 1799, "1767/1799")),
    ("latter part of the eighteenth century", (1767, 1799, "1767/1799")),
    ("toward the close of the sixteenth century", (1567, 1599, "1567/1599")),
    ("first half of the 18th century", (1700, 1749, "1700/1749")),
    ("second half of the eighteenth century", (1750, 1799, "1750/1799")),
    ("seventeenth and eighteenth centuries", (1600, 1799, "1600/1799")),
    ("13th and 14th centuries", (1200, 1399, "1200/1399")),
    ("since the sixteenth century", (1500, None, "1500/..")),
    # BC / AD / early years
    ("238 B.C.", (-238, -238, "-0237")),
    ("627 A.D.", (627, 627, "0627")),
    ("640-646 A.D.", (640, 646, "0640/0646")),
    ("around 350 AD", (350, 350, "0350~")),
    ("from around 1500 BC to 600 BC", (-1500, -600, "-1499~/-0599~")),
    ("8th century BCE", (-800, -701, "-0799/-0700")),
    ("850-922", (850, 922, "0850/0922")),
    ("In 801", (801, 801, "0801")),
    # named periods
    ("Middle Ages", (500, 1500, "0500/1500")),
    ("Tang dynasty", (618, 907, "0618/0907")),
    ("During the World War", (1914, 1918, "1914/1918")),
    ("before the World War", (None, 1913, "../1913")),
    # non-temporal / deictic / unanchored -> None
    ("in dry years", (None, None, None)),
    ("every morning", (None, None, None)),
    ("present (at time of writing)", (None, None, None)),
    ("now", (None, None, None)),
    ("today", (None, None, None)),
    ("formerly", (None, None, None)),
    ("the sixties", (None, None, None)),
    ("before the war", (None, None, None)),
    ("April 20th", (None, None, None)),
    ("30 minutes", (None, None, None)),
    ("at 15 deg. C.", (None, None, None)),
    ("", (None, None, None)),
    (None, (None, None, None)),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_year_helper_prefers_earliest_then_latest():
    assert year("1811-16") == 1811
    assert year("before 1811") == 1811
    assert year("in dry years") is None


def test_never_raises_on_odd_input():
    for s in ["%%%", "1" * 50, "centuries", "the 0th century", "B.C.", "-"]:
        normalize(s)  # must not raise
