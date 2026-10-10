"""Formatting helpers for underwriting report output.

Converts raw Vietnamese-đồng monetary figures into millions (triệu VNĐ) with 2
decimals and Vietnamese number formatting (``.`` thousands, ``,`` decimal), e.g.
``1.850.000.000`` -> ``1.850,00 triệu VNĐ`` and ``25.500.000`` -> ``25,50 triệu
VNĐ``.

Millions rather than billions because a retail file lives at that scale: a
salary of 25 triệu shown in tỷ reads "0,03", and a 1,85 tỷ mortgage is still
perfectly readable as 1.850 triệu. One unit for the whole memo keeps income,
instalments and loan amounts directly comparable on the page.
"""

from __future__ import annotations

import re

VND_PER_BILLION = 1_000_000_000
VND_PER_MILLION = 1_000_000
DISPLAY_UNIT = "triệu VNĐ"

# Grouped integer amounts only: at least one thousands group of exactly 3 digits.
# This deliberately does NOT match contiguous digit runs (tax/registration codes
# like "0104498100"), 4-digit years, or decimals such as "8.8".
_AMOUNT_TOKEN = re.compile(r"[+-]?\d{1,3}(?:[.,]\d{3})+(?![\d.,])")
# Optional trailing currency word to absorb into the replacement.
_TRAILING_CURRENCY = re.compile(r"\s*(?:VN[ĐD]|đồng|VND)\b", re.IGNORECASE)
# Units that mean the number is already scaled — never re-convert these.
_ALREADY_SCALED = re.compile(r"^\s*(?:tỷ|triệu|nghìn\s+tỷ|ngàn\s+tỷ)\b", re.IGNORECASE)


def format_vn_number(value: float, decimals: int = 2) -> str:
    """Format a number Vietnamese-style: '.' thousands, ',' decimal."""
    formatted = f"{value:,.{decimals}f}"  # US style: 3,991.12
    return formatted.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def to_billion_vnd(value: float, decimals: int = 2) -> str:
    """Render a raw đồng amount as a 'tỷ VNĐ' string."""
    return f"{format_vn_number(value / VND_PER_BILLION, decimals)} tỷ VNĐ"


def to_million_vnd(value: float, decimals: int = 2) -> str:
    """Render a raw đồng amount as a 'triệu VNĐ' string."""
    return f"{format_vn_number(value / VND_PER_MILLION, decimals)} {DISPLAY_UNIT}"


def format_million(value: float | None, decimals: int = 2) -> str:
    """A raw đồng amount as a bare triệu-VNĐ number, for tables that state the
    unit once in their header. Empty for a missing value."""
    if value is None:
        return ""
    return format_vn_number(value / VND_PER_MILLION, decimals)


def _parse_grouped_amount(token: str) -> float | None:
    """Parse a grouped integer amount in either VN or US locale."""
    text = token.strip()
    sign = -1 if text.startswith("-") else 1
    text = text.lstrip("+-")

    if "," in text and "." in text:
        # Decimal separator is whichever appears last.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    else:
        # Only one separator kind present, and (by the token regex) it groups
        # thousands — strip it regardless of which symbol was used.
        text = text.replace(".", "").replace(",", "")

    try:
        return sign * float(text)
    except ValueError:
        return None


def convert_amounts_in_text(text: str, decimals: int = 2) -> str:
    """Convert every raw đồng amount in a markdown/text block to triệu VNĐ.

    Skips percentages, values already expressed in tỷ/triệu/nghìn tỷ, and
    contiguous identifier digits (tax codes, registration numbers). Absorbs a
    trailing "VNĐ"/"đồng" so the result is not double-labelled.
    """
    if not text:
        return text

    # A trailing "VNĐ"/"đồng" must be swallowed together with the number, which
    # re.sub cannot do cleanly, so scan and rebuild manually.
    result = []
    cursor = 0
    for match in _AMOUNT_TOKEN.finditer(text):
        if match.start() < cursor:
            continue
        result.append(text[cursor:match.start()])
        token = match.group()
        tail = text[match.end():]

        if tail[:1] == "%" or _ALREADY_SCALED.match(tail):
            result.append(token)
            cursor = match.end()
            continue

        value = _parse_grouped_amount(token)
        if value is None:
            result.append(token)
            cursor = match.end()
            continue

        currency = _TRAILING_CURRENCY.match(tail)
        # Small grouped figures (below 100.000 đ) are not money at memo scale —
        # a count, an area in m² or a code fragment — and dividing them by a
        # million would print "0,05 triệu VNĐ" for "50.000". Leave them alone.
        if abs(value) < 100_000 and not currency:
            result.append(token)
            cursor = match.end()
            continue
        replacement = to_million_vnd(value, decimals)
        # Preserve an explicit leading "+" (deltas / chênh lệch); the "-" sign is
        # already produced by number formatting for negative values.
        if token.lstrip()[:1] == "+" and value >= 0:
            replacement = "+" + replacement
        result.append(replacement)
        cursor = match.end() + (currency.end() if currency else 0)

    result.append(text[cursor:])
    return "".join(result)
