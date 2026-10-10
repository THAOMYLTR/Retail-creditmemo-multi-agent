"""Structural markdown fixes that ``markdown``'s parser is strict about.

Pure text transforms, no LLM call — same spirit as ``citations.py`` but for a
different failure mode. Kept in its own module because it has nothing to do
with citations: it is about paragraph/list boundaries, which any block of
report prose can get wrong regardless of whether it carries a footnote.
"""

from __future__ import annotations

import re

_LIST_MARKER = re.compile(r"^\s*([-*+]|\d+[.)])\s+\S")
_FENCE = re.compile(r"^\s*(```|~~~)")

# "8,00%" and "8.0%" -> "8%". Only when every decimal digit is zero: "35,20%"
# keeps its digits, because dropping a trailing zero after a significant one is
# a different rule and nobody asked for it.
_ZERO_DECIMALS = re.compile(r"(?<=\d)[.,]0+(?=\s*%)")
# A table cell holding nothing but a zero, in any of the shapes a model writes
# one: 0, 0%, 0,00, 0.00%, and with the emphasis a model sometimes adds.
_ZERO_CELL = re.compile(r"^\*{0,2}0(?:[.,]0+)?\s*%?\*{0,2}$")
# The |---|---:| row under a table header, which must keep its dashes.
_TABLE_RULE = re.compile(r"^[\s|:-]+$")


def ensure_blank_line_before_lists(text: str) -> str:
    """Insert a blank line before a list that starts right after prose.

    Measured against ``markdown==3.10.3``: a "-"/"*"/"1." line with no blank
    line above it is lazy continuation of the previous paragraph whenever that
    line is non-blank, non-list text — the marker renders as a literal
    character inside the ``<p>``, not as a ``<ul>``/``<ol>``. Confirmed with
    "**Nhận định**: ...\\n- điểm 1" rendering with a bare hyphen, and the same
    block with a blank line inserted rendering as a real list.

    A no-op on a list that already follows a blank line or another list item,
    and on anything inside a fenced code block.
    """

    if not text:
        return text

    out: list[str] = []
    in_code = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
        elif not in_code and _LIST_MARKER.match(line):
            prev = out[-1] if out else ""
            if prev.strip() and not _LIST_MARKER.match(prev):
                out.append("")
        out.append(line)
    return "\n".join(out)


def tidy_numbers(text: str) -> str:
    """Drop all-zero decimals from percentages, and write a zero cell as "-".

    Two rules the report asks of every agent, applied here as well because a
    rule in a prompt is a request. Both are presentation-only: "8,0%" and "8%"
    are the same number, and a table cell reading "-" is the same zero it read
    before.

    The dash is deliberately NOT the same thing as an empty cell. Empty means
    the dossier does not give the figure; "-" means the figure is zero and was
    read from the dossier. EVIDENCE RULE forbids filling a gap with a dash for
    exactly that reason, so this only ever rewrites a cell that already holds a
    number.

    Percentages are tidied everywhere, fenced blocks included: the only fenced
    content carrying one is a mermaid diagram, whose edge labels are part of the
    report the reader sees. The dash rule is confined to table rows, where a
    column of figures makes a zero worth distinguishing at a glance.
    """

    if not text:
        return text

    out: list[str] = []
    in_code = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
            out.append(line)
            continue
        line = _ZERO_DECIMALS.sub("", line)
        if not in_code and line.lstrip().startswith("|") and not _TABLE_RULE.match(line):
            cells = line.split("|")
            for index, cell in enumerate(cells):
                if _ZERO_CELL.match(cell.strip()):
                    body = " - " if cell.startswith(" ") or cell.endswith(" ") else "-"
                    cells[index] = body
            line = "|".join(cells)
        out.append(line)
    return "\n".join(out)
