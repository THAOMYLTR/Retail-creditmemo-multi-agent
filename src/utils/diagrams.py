"""Parse Mermaid flowcharts and draw them so they survive PDF export.

The agents emit ```mermaid blocks because that renders natively on GitHub and in
VS Code. WeasyPrint has no JavaScript, so a mermaid block would reach the PDF as
a raw ``<pre><code>`` dump.

Parsing lives here; drawing is ``graph_svg.render_svg``. Flex-box was tried first
and lost on measurement: it cannot draw a hub once and connect it to several
spokes, so every partner diagram came out with arrows pointing at blank page. The
CSS that remains covers only the two cases with no layout to compute — a chart
with no edges at all, and the last-resort row-per-edge fallback.

Anything this parser cannot read is left as the original mermaid block rather
than dropped, so content is never lost.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, replace

from src.utils.graph_svg import render_svg

MERMAID_BLOCK = re.compile(
    r"^```mermaid[ \t]*\n(.*?)^```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
# "flowchart LR" / "graph TD" ...
_HEADER = re.compile(r"^\s*(?:flowchart|graph)\s+(LR|RL|TD|TB|BT)\b", re.IGNORECASE)
# A --> B, A -->|nhãn| B, A --- B, A -.-> B, A ==> B, and mermaid's other way of
# writing an edge label: A -- nhãn --> B, A == nhãn ==> B.
#
# The inline-label forms must come FIRST. Tried in the other order, "-{2,3}>"
# matches the leading "--" of "-- 65% -->" and the label text is then read as
# part of the next node id, which collapses the whole line into one box.
_CONNECTOR = re.compile(
    r"\s*(?:"
    r"-{2}\s*(?P<dashlabel>[^|>=-][^>]*?)\s*-{2,3}>"
    r"|={2}\s*(?P<eqlabel>[^|>=-][^>]*?)\s*={2,}>"
    r"|-{2,3}>|-\.->|={2,}>|-{3}"
    r")\s*(?:\|(?P<pipelabel>[^|]*)\|\s*)?"
)
_NODE_ID = re.compile(r"\s*(?P<id>[A-Za-z0-9_]+)\s*")
_OPEN_TO_CLOSE = {"[": "]", "(": ")", "{": "}"}
# classDef hilite fill:#C6E0B4,stroke:#333
_CLASSDEF = re.compile(r"^classDef\s+(?P<name>[A-Za-z0-9_]+)\s+(?P<body>.+)$")
# class A,B hilite
_CLASS_APPLY = re.compile(r"^class\s+(?P<ids>[A-Za-z0-9_,\s]+?)\s+(?P<name>[A-Za-z0-9_]+)\s*$")
# style A fill:#eee
_STYLE_DECL = re.compile(r"^style\s+(?P<id>[A-Za-z0-9_]+)\s+(?P<body>.+)$")
# The ":::hilite" suffix, which follows the node's label when it has one
# ("VNM[Vinamilk]:::hilite"), so it has to be consumed while scanning rather
# than matched against the bare id.
_INLINE_CLASS = re.compile(r"\s*:::(?P<name>[A-Za-z0-9_]+)")
# subgraph id[Tiêu đề]  |  subgraph Tiêu đề
_SUBGRAPH = re.compile(
    r"^subgraph\s+(?:(?P<id>[A-Za-z0-9_]+)\s*\[(?P<title>.+?)\]|(?P<plain>.+?))\s*$"
)


def _scan_label(text: str, start: int) -> tuple[str | None, int]:
    """Read a bracketed node label at ``start``, honouring nesting like [[x]]."""

    if start >= len(text) or text[start] not in _OPEN_TO_CLOSE:
        return None, start
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char in _OPEN_TO_CLOSE:
            depth += 1
        elif char in _OPEN_TO_CLOSE.values():
            depth -= 1
            if depth == 0:
                return text[start : index + 1], index + 1
        index += 1
    return None, start


def _scan_class(text: str, start: int) -> tuple[str | None, int]:
    """Read a ":::className" suffix at ``start``, if present."""

    match = _INLINE_CLASS.match(text, start)
    if not match:
        return None, start
    return match.group("name"), match.end()


def _scan_line(
    line: str,
) -> tuple[list[tuple[str, str | None, str | None]], list[tuple[int, str]]]:
    """Split ``A[x] -->|l| B[y]:::cls --> C`` into its nodes and the links.

    Chains must be walked sequentially — a plain regex scan would treat each
    destination as consumed and silently drop the rest of the chain.
    """

    nodes: list[tuple[str, str | None, str | None]] = []
    links: list[tuple[int, str]] = []
    pos = 0
    match = _NODE_ID.match(line, pos)
    if not match:
        return [], []
    label, pos = _scan_label(line, match.end())
    node_class, pos = _scan_class(line, pos)
    nodes.append((match.group("id"), label, node_class))

    while pos < len(line):
        connector = _CONNECTOR.match(line, pos)
        if not connector:
            break
        pos = connector.end()
        match = _NODE_ID.match(line, pos)
        if not match:
            break
        label, pos = _scan_label(line, match.end())
        node_class, pos = _scan_class(line, pos)
        edge_label = (
            connector.group("pipelabel")
            or connector.group("dashlabel")
            or connector.group("eqlabel")
            or ""
        )
        links.append((len(nodes) - 1, edge_label.strip()))
        nodes.append((match.group("id"), label, node_class))
    return nodes, links


# Colours match mermaid's default theme, because the same report is read as
# markdown (where mermaid renders for real) and as PDF (where it cannot, so this
# stands in). Arrows are drawn from pseudo-elements rather than typed, since a
# "→" glyph next to a real diagram looks like a fallback.
#
# Only the fallback paths in _render still use these classes; everything with
# edges is drawn by graph_svg.
DIAGRAM_CSS = """
/* Diagrams and charts are emitted as bare <svg> with explicit width and height.
   Only the centring is left to CSS: a top-down flowchart is much narrower than
   the page and otherwise sits against the left margin. Sizing stays out of CSS
   on purpose — width:100% with height:auto makes WeasyPrint compute zero height
   and draw nothing. */
svg{display:block;margin:14px auto}
.mmd{margin:14px 0}
/* nowrap + shrinkable nodes: a wrapped row would leave the last box stretched
   across its own line, which reads worse than slightly narrower boxes. */
.mmd-row{display:flex;align-items:stretch;flex-wrap:nowrap;margin:6px 0;
  justify-content:center}
.mmd-col{display:flex;flex-direction:column;align-items:center;margin:6px 0}
.mmd-node{border:1px solid #9370DB;border-radius:5px;background:#ECECFF;
  color:#111;padding:8px 14px;text-align:center;font-size:9.5pt;line-height:1.35;
  flex:0 1 auto;min-width:0;overflow-wrap:break-word}
.mmd-col .mmd-node{flex:0 0 auto;min-width:150px;font-size:10pt}
/* Shaft + head built from pseudo-elements: no image, no external SVG, and
   WeasyPrint renders border-triangles exactly. */
.mmd-arrow{flex:0 0 auto;align-self:center;position:relative;
  width:36px;min-height:14px;text-align:center}
.mmd-arrow::before{content:"";position:absolute;left:2px;top:6px;
  width:24px;height:2px;background:#333}
.mmd-arrow::after{content:"";position:absolute;left:26px;top:1px;
  border-top:6px solid transparent;border-bottom:6px solid transparent;
  border-left:9px solid #333}
.mmd-col .mmd-arrow{width:14px;min-height:36px}
.mmd-col .mmd-arrow::before{left:6px;top:2px;width:2px;height:24px}
.mmd-col .mmd-arrow::after{left:1px;top:26px;border-left:6px solid transparent;
  border-right:6px solid transparent;border-top:9px solid #333;border-bottom:0}
/* Edge labels (A -->|Thanh toán| B) are not in any template yet but the parser
   emits them, so they keep a visible slot instead of vanishing: the label sits
   above the arrow, which drops to the bottom of the box. */
.mmd-arrow-labelled{width:auto;min-width:64px;max-width:110px;padding:0 4px 14px}
.mmd-arrow-labelled::before{left:50%;margin-left:-13px;top:auto;bottom:6px}
.mmd-arrow-labelled::after{left:50%;margin-left:11px;top:auto;bottom:1px}
.mmd-col .mmd-arrow-labelled{width:auto;min-width:0;padding:0 0 0 14px}
.mmd-col .mmd-arrow-labelled::before{left:6px;margin-left:0;top:2px;bottom:auto}
.mmd-col .mmd-arrow-labelled::after{left:1px;margin-left:0;top:26px;bottom:auto}
.mmd-elabel{display:block;font-size:8pt;color:#456;line-height:1.2;
  text-align:center;overflow-wrap:break-word}
.mmd-note{font-size:8pt;color:#8A6D3B;text-align:center;margin:2px 0 0}
"""


def _label_html(text: str) -> str:
    """Escape label text, keeping mermaid's <br/> line breaks.

    Shared by node and edge labels so the two behave identically — edge labels
    used to be escaped wholesale, which printed a literal "<br/>" where a node
    in the same diagram would have broken the line.
    """

    parts = [part.strip() for part in re.split(r"<br\s*/?>", text) if part.strip()]
    return "<br>".join(html.escape(part) for part in parts)


def _clean_label(raw: str | None, fallback: str) -> str:
    """Strip mermaid node brackets and quotes, keeping the visible text."""

    if not raw:
        return fallback
    text = raw.strip()
    text = re.sub(r"^[\[\({]+", "", text)
    text = re.sub(r"[\]\)}]+$", "", text)
    text = text.strip().strip('"').strip("'").strip()
    return _label_html(text) or fallback


@dataclass(frozen=True)
class Flowchart:
    """A parsed mermaid flowchart.

    A dataclass rather than a widening tuple: styling and grouping took the
    parse result past the point where positional unpacking stays readable.
    """

    direction: str
    order: list[str]
    labels: dict[str, str]
    edges: list[tuple[str, str, str]]
    # node id -> style, resolved from classDef/class/:::/style declarations.
    node_style: dict[str, dict[str, str]]
    # (title, member node ids) for each subgraph, in declaration order.
    groups: list[tuple[str, list[str]]]

    @property
    def vertical(self) -> bool:
        return self.direction in {"TD", "TB", "BT"}


def _parse_style_pairs(text: str) -> dict[str, str]:
    """Read mermaid's "fill:#eee,stroke:#333" style body into a dict."""

    style: dict[str, str] = {}
    for chunk in text.split(","):
        key, _, value = chunk.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key in {"fill", "stroke", "color"} and value:
            style[key] = value
    return style


def _parse(source: str) -> Flowchart:
    """Parse one mermaid flowchart body."""

    direction = "LR"
    labels: dict[str, str] = {}
    order: list[str] = []
    edges: list[tuple[str, str, str]] = []
    class_styles: dict[str, dict[str, str]] = {}
    node_classes: dict[str, str] = {}
    node_style: dict[str, dict[str, str]] = {}
    groups: list[tuple[str, list[str]]] = []
    group_stack: list[tuple[str, list[str]]] = []

    def remember(node_id: str, raw_label: str | None) -> None:
        if node_id not in labels:
            labels[node_id] = _clean_label(raw_label, html.escape(node_id))
            order.append(node_id)
        elif raw_label:
            labels[node_id] = _clean_label(raw_label, labels[node_id])
        for _, members in group_stack:
            if node_id not in members:
                members.append(node_id)

    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        header = _HEADER.match(stripped)
        if header:
            direction = header.group(1).upper()
            continue

        subgraph = _SUBGRAPH.match(stripped)
        if subgraph:
            raw_title = subgraph.group("title") or subgraph.group("plain") or ""
            group_stack.append((_label_html(raw_title.strip().strip('"')), []))
            continue
        if stripped == "end":
            if group_stack:
                groups.append(group_stack.pop())
            continue

        class_def = _CLASSDEF.match(stripped)
        if class_def:
            class_styles[class_def.group("name")] = _parse_style_pairs(
                class_def.group("body")
            )
            continue
        class_apply = _CLASS_APPLY.match(stripped)
        if class_apply:
            for node_id in class_apply.group("ids").split(","):
                node_classes[node_id.strip()] = class_apply.group("name")
            continue
        node_style_decl = _STYLE_DECL.match(stripped)
        if node_style_decl:
            node_style[node_style_decl.group("id")] = _parse_style_pairs(
                node_style_decl.group("body")
            )
            continue
        if stripped.startswith("linkStyle"):
            continue

        line_nodes, line_links = _scan_line(stripped)
        if line_links:
            for node_id, raw_label, node_class in line_nodes:
                remember(node_id, raw_label)
                if node_class:
                    node_classes[node_id] = node_class
            for index, edge_label in line_links:
                edges.append(
                    (line_nodes[index][0], edge_label, line_nodes[index + 1][0])
                )
            continue

        line_nodes, _ = _scan_line(stripped)
        if len(line_nodes) == 1 and (line_nodes[0][1] or line_nodes[0][2]):
            node_id, raw_label, node_class = line_nodes[0]
            remember(node_id, raw_label)
            if node_class:
                node_classes[node_id] = node_class

    # Unclosed subgraphs still count: a missing "end" should not discard the
    # grouping the author clearly intended.
    while group_stack:
        groups.append(group_stack.pop())

    for node_id, class_name in node_classes.items():
        style = class_styles.get(class_name)
        if style and node_id not in node_style:
            node_style[node_id] = style

    return Flowchart(
        direction=direction,
        order=order,
        labels=labels,
        edges=edges,
        node_style=node_style,
        groups=[(title, members) for title, members in groups if members],
    )


# Share at which one box in a diagram is highlighted as a concentration (e.g.
# one income source carrying most of the household's money). Inherited from the
# SME supply-chain diagrams, where it flagged a dominant partner.
_CONCENTRATION_THRESHOLD = 40.0
_PERCENT_LABEL = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
# Rose, and rose is used nowhere else: graph_svg's level palette runs green,
# blue, slate, amber, violet precisely so this hue stays free. A flag sharing a
# colour with an ordinary level stops being a flag.
#
# The text colour carries as much of the signal as the fill does. Every other box
# prints near-black on a pale tint, so a box whose words are themselves coloured
# reads as marked before its background is even noticed — which matters because
# these fills all sit at the same lightness and cannot separate by weight.
_AUTO_WARN_STYLE = {"fill": "#f7e2e5", "stroke": "#c1616f", "color": "#c1616f"}


def _auto_color_concentration(chart: Flowchart) -> Flowchart:
    """Highlight a partner node the model forgot to color despite a high-% edge.

    Measured against a real specialist run: guidance.md and the report
    skeleton both ask for a `:::warn`/`:::hi` highlight on any partner at
    ~40%+ concentration, but the model coloured one of two equally-qualifying
    nodes in the same report and missed the other (non-determinism, not a
    one-off). Same gap the citation and list fixups close elsewhere — a prompt
    rule the model sometimes skips gets a code-level fallback so the reader
    sees it every time, not on a coin flip.

    Overrides whatever the node already carried. It used to fill a gap only,
    leaving a model-styled node alone, and the report above is why that was
    wrong: the model wrote `classDef warn fill:#FFC000` and applied it to the
    55% customer, then forgot the 68% supplier in the next diagram. One
    concentration finding came out amber and the other rose, in the same
    report, for the same reason — and a reader has no way to know the two
    marks mean the same thing. The house style wins so that they match.

    A node the model marked but that does not clear the threshold is left
    alone; this function only speaks for the rule it enforces.

    The "hub" side of a fan (out-degree/in-degree > 1) is never coloured by
    this — only the single-degree partner at the other end of the qualifying
    edge, which is the shape mục 4/5's fan diagrams always have.
    """

    degree: dict[str, int] = {}
    for src, _label, dst in chart.edges:
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1

    node_style = dict(chart.node_style)
    changed = False
    for src, label, dst in chart.edges:
        match = _PERCENT_LABEL.search(label)
        if not match:
            continue
        value = float(match.group(1).replace(",", "."))
        if value < _CONCENTRATION_THRESHOLD:
            continue
        # The hub is whichever end has more connections; the other end is the
        # partner this edge is about. Compared, not tested against 1: the old
        # rule demanded the partner be a leaf, and in the shape mục 1 actually
        # draws — product -> partner -> company — a supplier carries its own
        # product edge, so a 45% supplier scored degree 2 and was passed over
        # while a 40% customer with no product box was flagged. Same finding,
        # different topology, opposite outcome.
        src_degree, dst_degree = degree.get(src, 0), degree.get(dst, 0)
        if src_degree > dst_degree:
            partner = dst
        elif dst_degree > src_degree:
            partner = src
        else:
            continue
        if node_style.get(partner) != _AUTO_WARN_STYLE:
            node_style[partner] = dict(_AUTO_WARN_STYLE)
            changed = True

    return replace(chart, node_style=node_style) if changed else chart


# The same <br/> spelling _label_html and graph_svg._text_lines both accept:
# mermaid writes <br>, <br/> and <br /> interchangeably.
_BR = re.compile(r"<br\s*/?>")
# "3,65 tỷ", "1.850 triệu đồng", "2,48 tỉ VNĐ" — a bare money amount carrying no
# percent sign of its own. A magnitude word may be followed by a currency word,
# which is what "3.650 triệu đồng" is and what an earlier version of this missed:
# it required the line to end at "triệu".
#
# One of the two words has to be there. Digits alone are left alone — a bare "45"
# beside a percentage is not necessarily money, and this only removes what it can
# name.
_MONEY_UNIT = r"(?:tỷ|tỉ|triệu|nghìn|ngàn|tr)"
_MONEY_CUR = r"(?:đồng|đ|vnđ|vnd)"
_MONEY_BODY = (
    rf"\d[\d.,\s]*(?:{_MONEY_UNIT}\s*{_MONEY_CUR}?|{_MONEY_CUR})"
)
_MONEY_ONLY = re.compile(rf"^[\s(]*{_MONEY_BODY}[\s.,)]*$", re.IGNORECASE)
_MONEY_PAREN = re.compile(rf"\s*\(\s*{_MONEY_BODY}\s*\)", re.IGNORECASE)


def _percent_only_edge_labels(chart: Flowchart) -> Flowchart:
    """Drop the absolute figure from an edge label that already carries a %.

    A share of revenue is what a connector in mục 1 is annotating, and the
    percentage says it. Adding "3,65 tỷ" underneath says the same thing twice in
    a place with no room for it: the label is the widest thing in the gap between
    two boxes, and the gap is sized from it, so the second line pushes the whole
    diagram wider and then gets scaled back down — every box on the page loses
    text size to a number the table beside it already gives.

    Narrow on purpose. Only a line that is nothing but an amount goes, and only
    when another line in the same label carries a percent sign. "45 ngày" on the
    inventory connector has no percentage next to it and survives; so does
    "trả chậm 30 ngày", and so does a lone "3,65 tỷ" on a diagram that quotes no
    shares at all.
    """

    edges = []
    changed = False
    for src, label, dst in chart.edges:
        lines = _BR.split(label) if label else []
        if len(lines) > 1 and any("%" in line for line in lines):
            kept = [line for line in lines
                    if "%" in line or not _MONEY_ONLY.match(line.strip())]
            if kept and kept != lines:
                label = "<br/>".join(kept)
                changed = True
        if "%" in label:
            stripped = _MONEY_PAREN.sub("", label)
            if stripped.strip() and stripped != label:
                label = stripped
                changed = True
        edges.append((src, label, dst))

    return replace(chart, edges=edges) if changed else chart


def _node_html(label: str) -> str:
    return f'<div class="mmd-node">{label}</div>'


def _arrow_html(label: str, vertical: bool) -> str:
    """One arrow between two nodes.

    The arrow itself is drawn in CSS (see DIAGRAM_CSS), so no glyph goes in the
    markup — emitting one too would stack a character on top of the drawn
    shape. Labelled arrows carry their own class rather than relying on a
    :has() selector, which keeps the styling independent of how much of
    Selectors Level 4 the PDF renderer implements.
    """

    if not label:
        return '<div class="mmd-arrow"></div>'
    return (
        '<div class="mmd-arrow mmd-arrow-labelled">'
        f'<span class="mmd-elabel">{_label_html(label)}</span>'
        "</div>"
    )


def _render(source: str) -> str | None:
    """Render one mermaid flowchart as HTML, or None if it is not parseable."""

    chart = _parse(source)
    if not chart.labels:
        return None
    chart = _percent_only_edge_labels(chart)
    chart = _auto_color_concentration(chart)
    order, labels, edges = chart.order, chart.labels, chart.edges
    if not edges:
        # Nodes only: no layout to compute, so a single row of boxes.
        boxes = "".join(_node_html(labels[node]) for node in order)
        return f'<div class="mmd"><div class="mmd-row">{boxes}</div></div>'

    # Everything with edges goes to the layered SVG drawer.
    #
    # There used to be flex-box special cases ahead of this — a linear chain and
    # a hub-and-spokes fan — and the fan one was wrong: flex draws the hub once,
    # in the row it happens to sit in, so every other spoke's arrow pointed at
    # empty page. Measured side by side on the section 4 and 5 diagrams, the SVG
    # drawer also beat the chain case: uniform box heights, no ragged wrapping,
    # and a third of the vertical space.
    svg = render_svg(chart)
    if svg is not None:
        return svg

    # Last resort, if the layout could not run at all: one row per edge. Ugly,
    # and it repeats a hub's name once per edge, but it still shows every node
    # and every connection rather than dropping the diagram.
    rows = [
        '<div class="mmd-row">'
        + _node_html(labels[src])
        + _arrow_html(label, False)
        + _node_html(labels[dst])
        + "</div>"
        for src, label, dst in edges
    ]
    return f'<div class="mmd">{"".join(rows)}</div>'


def mermaid_to_html(text: str) -> str:
    """Replace ```mermaid blocks with HTML/CSS boxes for non-JS renderers.

    Unparseable blocks are returned untouched so no content is ever lost.
    """

    def replace(match: re.Match[str]) -> str:
        rendered = _render(match.group(1))
        return rendered if rendered else match.group(0)

    return MERMAID_BLOCK.sub(replace, text or "")
