"""Lay out a multi-level flowchart and draw it as inline SVG.

The CSS renderer in ``diagrams.py`` covers a single chain and a one-level fan.
Anything wider — a supply chain running suppliers → company → buyers over
several levels — has no sensible flex-box form: it degrades to one row per edge,
repeating the hub's name on every line.

SVG is the way out. WeasyPrint draws inline SVG properly (rounded boxes, fills,
elbowed connectors, arrowheads), so a real diagram reaches the PDF without
pulling Node and a headless browser into the project just to run mermaid.

What this does is the classic layered-graph recipe: assign each node a level,
order the nodes inside each level to keep connectors from crossing, place them,
then route orthogonal edges. It is not a general graph drawer — it assumes the
graph flows in one direction, which is what the report templates produce.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from functools import lru_cache

# Geometry, in SVG user units (the whole drawing is scaled to the page later).
# The floor is low on purpose. At 96 it bound on every short label — "Đầu ra"
# was stretched from the 70 its text needs to 96 — and since the box grew while
# the text did not, the visible margin round the words went from 14 to 27. Boxes
# then looked like they had been padded by different amounts, because they had.
# 56 only catches labels of three characters or fewer, where a box would
# otherwise be a sliver.
NODE_MIN_WIDTH = 56
NODE_MAX_WIDTH = 190
# Roomier than before, following the pen's 20px. Boxes that hug their text read
# as cramped next to 9.5pt body copy; the extra air is most of why the original
# looks calmer than what this drew before.
NODE_PADDING_X = 14
NODE_PADDING_Y = 11
LINE_HEIGHT = 17
# SVG user units are CSS px, and WeasyPrint prints them at 72/96 — so a size
# here is 0.75 of what lands on the page. Measured, not assumed: a 20px label
# came out of the PDF at exactly 15.00pt.
PX_TO_PT = 0.75
# 13.5px = 10.1pt against 9.5pt body text: the boxes read a touch larger than
# the prose, which is what a diagram wants. Reliable only because a wrapped
# chain now lays out into the page width instead of being scaled down to reach
# it — before that, this constant was multiplied by an 0.45 fit factor and the
# nine-box supply chain printed at 3.5pt.
FONT_SIZE = 13.5
EDGE_FONT_SIZE = 11.34
RANK_GAP = 92
# Top-down charts need less room between levels: a horizontal gap has to fit an
# edge label *beside* the connector, a vertical one only above and below it.
VERTICAL_RANK_GAP = 46
# What a level needs when nothing is written between it and the next one: room
# for the connector and its arrowhead, no more. RANK_GAP is sized for an edge
# label sitting beside the connector, and on a chain with no labels at all those
# 74s were 36% of the drawing's width — spent on space for text that does not
# exist, and paid for by shrinking the text that does.
BARE_RANK_GAP = 44
BARE_VERTICAL_RANK_GAP = 26
# Clear space either side of an edge label — per side, not shared between them.
# The earlier constant was the total, which left 7px a side once the label was
# centred, and the arrowhead is 8px long: the label finished exactly where the
# arrow began. A gap is sized from what has to fit inside it, and that is the
# label plus room to breathe plus the arrowhead.
EDGE_LABEL_CLEARANCE = 13
# Clear space between the bottom of an edge label and the wire it annotates.
# The old code lifted the baseline by a flat 4px, chosen when the edge font was
# 8.5. At 10.5 the descenders reached to within 1.7px of the wire, and
# Vietnamese puts marks below the baseline — ạ, ộ, ệ — so the label sat on the
# line. Derived from the font rather than picked, so it stays right if the size
# changes again.
EDGE_LABEL_MARGIN = 4.0
# How far glyphs reach below the baseline, as a fraction of the font size.
DESCENDER_RATIO = 0.22
ARROW_LENGTH = 8
EDGE_STROKE_WIDTH = 1.4
# Vertical room between wrapped rows: enough for the connector to drop out of
# one row, run back to the left and arrive on top of the next.
WRAP_ROW_GAP = 42
NODE_GAP = 14
MARGIN = 8

# Fallback only. Counting characters cannot work: measured against a rendered
# PDF, the real advance per character ranged from 0.514 to 0.665 of the font
# size depending on the word, so any single ratio is wrong for most labels. At
# 0.52 the long ones came out 24% too narrow, which showed up as boxes whose
# text nearly touched the edge while short boxes had room to spare — the same
# padding in the code, four times the padding on the page.
CHAR_WIDTH = FONT_SIZE * 0.57
EDGE_CHAR_WIDTH = EDGE_FONT_SIZE * 0.57

# WeasyPrint lays the text out with the report's own font stack, so that is what
# gets measured. The correction is empirical: PIL's advance widths came out a
# consistent 11% under what the PDF actually drew — consistent enough to correct
# for, where the character count was not. With it, padding lands between 9.8 and
# 11.3pt against a 10.5pt target; the character count gave 4.8 to 10.6.
FONT_STACK = ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
FONT_MEASURE_CORRECTION = 1.115
_MEASURE_SIZE = 64
# Kept for the fallback path only. Wrapping by character count stopped being
# right when box width started being measured: twenty-one wide characters
# measured 291px against a 162px limit and ran out of the box, while twenty-one
# narrow ones used 69px and wasted most of it. Lines are broken on measured
# width now, so a Vietnamese label — mostly narrow glyphs — fits more per line
# and needs fewer of them.
MAX_CHARS_PER_LINE = int((NODE_MAX_WIDTH - 2 * NODE_PADDING_X) / CHAR_WIDTH)

# Usable width of the report page: A4 (210mm) less the 11/9mm margins in
# REPORT_CSS, at 96dpi. The drawing is scaled to this at generation time rather
# than left to CSS max-width — WeasyPrint scales such an SVG's box but not its
# contents, which paints the diagram twice on one page.
PAGE_CONTENT_WIDTH = 726.0
# Below this the labels stop being readable, so the caller is told the diagram
# needs splitting rather than being handed an unreadable picture. In px, like
# every size here: 8px is 6pt on the page.
MIN_READABLE_FONT = 8.0

# The report's own palette (report_style.py) rather than mermaid's purple
# defaults, which made every diagram look pasted in from another document. The
# blue is the same accent the blockquote rule and footnote links already use.
DEFAULT_FILL = "#f2f7fb"
DEFAULT_STROKE = "#2f6f9f"
DEFAULT_TEXT = "#1f2a33"
# Darker and thinner than before, following the CodePen org chart this styling
# came from: it draws 2px near-black connectors, which read far better on paper
# than the soft grey that was here.
EDGE_COLOUR = "#3d4a55"
EDGE_LABEL_COLOUR = "#445566"
# "bold", not a numeric weight: 600 asks for a semibold that the report's
# font stack does not ship, and the fallback is the regular face — the
# emphasis would silently not happen.
HIGHLIGHT_WEIGHT = "bold"
GROUP_STROKE = "#b9c6d1"
# Hand-drawn depth, because neither CSS box-shadow nor an SVG filter survives
# WeasyPrint. Small numbers on purpose: a 1.5px offset reads as a lifted card,
# more reads as a printing misregistration.
SHADOW_COLOUR = "#1f2a33"
SHADOW_OPACITY = 0.13
SHADOW_OFFSET = 1.5

# One colour per level, the idea worth taking from that pen. It is not only
# decoration here: Business Activity's section 1 is a chain whose levels are
# real stages — đầu vào, sản xuất, tồn kho, đầu ra, thu tiền — so colouring by
# level captions the diagram as well as brightening it.
#
# The pen's own colours (#8dccad, #f5cc7f, #7b9fe0, #f27c8d) are pitched for a
# screen and would shout next to the report's tables. These keep the flat pastel
# feel at lower saturation, with the fill tinted and the stroke carrying the
# colour, and stay inside the blue-grey family the rest of the document uses.
LEVEL_COLOURS = (
    ("#dfeee7", "#4e9b7c"),   # inputs — green, the start of the chain
    ("#dce7f3", "#2f6f9f"),   # the report's own accent
    ("#e4e9ee", "#5c7285"),   # neutral slate for the middle of a long chain
    ("#f7eeda", "#b5852f"),   # amber, warming toward the output end
    ("#ece4f3", "#7d5ba6"),   # violet — cash in
)
# Rose is deliberately absent from the levels above: it belongs to the
# concentration flag in diagrams.py and nothing else. A flag that shares a hue
# with an ordinary level is not a flag, and this one has to stand out wherever
# in the chain it lands.


@dataclass
class _Node:
    node_id: str
    lines: list[str]
    fill: str
    stroke: str
    colour: str
    rank: int = 0
    order: float = 0.0
    # Which wrapped row this node sits in; 0 for every diagram that fits on one.
    row: int = 0
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2


@dataclass
class _Edge:
    src: str
    dst: str
    label: str
    lines: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _measure_font():
    """The report's font at a large size, or None when none can be loaded.

    Measured big and scaled down: advance widths are more accurate away from
    hinting at small sizes.
    """

    try:
        from matplotlib import font_manager
        from PIL import ImageFont
    except Exception:
        return None
    for name in FONT_STACK:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=name),
                fallback_to_default=False,
            )
            return ImageFont.truetype(path, size=_MEASURE_SIZE)
        except Exception:
            continue
    return None


def text_width(text: str, font_size: float) -> float:
    """How wide this text will draw, in SVG user units.

    Falls back to the character count when no font can be loaded, so a machine
    without the measuring libraries still gets a diagram — a slightly ragged one
    rather than none.
    """

    font = _measure_font()
    if font is None:
        return len(text) * font_size * 0.57
    return font.getlength(text) / _MEASURE_SIZE * font_size * FONT_MEASURE_CORRECTION


def _text_lines(label_html: str, max_width: float = NODE_MAX_WIDTH) -> list[str]:
    """Turn a label back into plain text lines, wrapping long ones.

    Labels arrive HTML-escaped with <br> separators because the CSS renderer
    consumes them that way; SVG needs the raw characters instead.

    Nothing is ever cut. An edge label briefly ended in an ellipsis when it ran
    long, and an ellipsis in a credit memo is worse than a wide diagram: the
    reader cannot tell whether the missing words were "kể từ ngày nghiệm thu" or
    a condition that changes the meaning. The length is controlled where it
    should be — the guidance asks the agents for ten words — and whatever
    arrives here is drawn in full.
    """

    raw = [html.unescape(part) for part in re.split(r"<br\s*/?>", label_html)]
    limit = max_width - 2 * NODE_PADDING_X
    lines: list[str] = []
    for part in raw:
        part = part.strip()
        if not part:
            continue
        current = ""
        for word in part.split():
            # A single token with nothing to break on — a code, a run of capitals
            # — would otherwise sit wider than its own box. Split it mid-word;
            # ugly, but visible, where an overflowing line is neither.
            while text_width(word, FONT_SIZE) > limit:
                cut = len(word)
                while cut > 1 and text_width(word[:cut], FONT_SIZE) > limit:
                    cut -= 1
                if current:
                    lines.append(current)
                    current = ""
                lines.append(word[:cut])
                word = word[cut:]
            candidate = f"{current} {word}".strip()
            if current and text_width(candidate, FONT_SIZE) > limit:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)

    return lines or [""]


def _assign_ranks(
    node_ids: list[str],
    edges: list[_Edge],
) -> dict[str, int]:
    """Longest-path layering, ignoring edges that would close a cycle.

    A cycle in the input must not hang the PDF export, and an LLM can certainly
    write one, so back edges are dropped rather than trusted.
    """

    outgoing: dict[str, list[str]] = {node: [] for node in node_ids}
    for edge in edges:
        if edge.src in outgoing and edge.dst in outgoing:
            outgoing[edge.src].append(edge.dst)

    rank = {node: 0 for node in node_ids}
    # Depth-first longest path with an explicit on-stack set: revisiting a node
    # already on the current path means a cycle, and that edge is skipped.
    state: dict[str, int] = {}  # 0 = unvisited, 1 = on stack, 2 = done

    def visit(node: str) -> int:
        if state.get(node) == 1:
            return 0
        if state.get(node) == 2:
            return rank[node]
        state[node] = 1
        depth = 0
        for target in outgoing[node]:
            depth = max(depth, visit(target) + 1)
        state[node] = 2
        rank[node] = depth
        return depth

    for node in node_ids:
        visit(node)

    # visit() measured distance to a sink; flip it so rank 0 is a source, which
    # is what "left to right" means to a reader.
    deepest = max(rank.values(), default=0)
    return {node: deepest - value for node, value in rank.items()}


def _order_within_ranks(
    ranks: dict[str, list[str]],
    edges: list[_Edge],
    nodes: dict[str, _Node],
) -> None:
    """Median heuristic, a few sweeps, to keep connectors from crossing."""

    predecessors: dict[str, list[str]] = {node: [] for node in nodes}
    successors: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        if edge.src in nodes and edge.dst in nodes:
            predecessors[edge.dst].append(edge.src)
            successors[edge.src].append(edge.dst)

    for rank_nodes in ranks.values():
        for position, node_id in enumerate(rank_nodes):
            nodes[node_id].order = float(position)

    def median(neighbours: list[str]) -> float | None:
        positions = sorted(nodes[n].order for n in neighbours if n in nodes)
        if not positions:
            return None
        middle = len(positions) // 2
        if len(positions) % 2:
            return positions[middle]
        return (positions[middle - 1] + positions[middle]) / 2

    rank_keys = sorted(ranks)
    for sweep in range(4):
        keys = rank_keys if sweep % 2 == 0 else list(reversed(rank_keys))
        for key in keys:
            neighbours = predecessors if sweep % 2 == 0 else successors
            for node_id in ranks[key]:
                value = median(neighbours[node_id])
                if value is not None:
                    nodes[node_id].order = value
            ranks[key].sort(key=lambda n: nodes[n].order)
            for position, node_id in enumerate(ranks[key]):
                nodes[node_id].order = float(position)


def _label_gaps(
    edges: list["_Edge"],
    nodes: dict[str, _Node],
) -> dict[int, float]:
    """How much room each level needs after it for its own edge labels.

    Sized from the widest label rather than a fixed constant. RANK_GAP was that
    constant, and a label longer than it simply ran over the box on the far side
    — "thanh toán 30 ngày" needs 98px and got 74. A gap exists to hold something;
    how much it needs depends on what it is holding.
    """

    out_degree: dict[str, int] = {}
    in_degree: dict[str, int] = {}
    for edge in edges:
        out_degree[edge.src] = out_degree.get(edge.src, 0) + 1
        in_degree[edge.dst] = in_degree.get(edge.dst, 0) + 1

    needed: dict[int, float] = {}
    for edge in edges:
        if not edge.lines or edge.src not in nodes:
            continue
        widest = max(text_width(line, EDGE_FONT_SIZE) for line in edge.lines)
        room = widest + 2 * EDGE_LABEL_CLEARANCE + ARROW_LENGTH
        # A fan's label does not get the whole gap. Its connector doglegs across
        # a shared trunk halfway along, and the label sits on the horizontal run
        # to one side of that trunk — so it has half the gap to fit in, and the
        # gap has to be twice as wide for the same label. Sizing it as though the
        # label were centred in the whole gap is what put "6,03% / 3,65 tỷ"
        # underneath a supplier box in a real report: from about 50px of label
        # the text runs back over the box, and boxes are painted after edges, so
        # it does not overlap — it disappears.
        if out_degree.get(edge.src, 0) > 1 or in_degree.get(edge.dst, 0) > 1:
            room = 2 * (widest + EDGE_LABEL_CLEARANCE) + ARROW_LENGTH
        rank = nodes[edge.src].rank
        needed[rank] = max(needed.get(rank, 0.0), room)
    return needed


def _colour_by_level(chart, nodes: dict[str, _Node]) -> None:
    """Tint each node by the level it sits on, in place.

    Only where the source said nothing. A mermaid ``style`` or ``classDef`` is
    the author being deliberate — and one of those carries meaning rather than
    taste: diagrams.py paints a concentration warning amber, and a level palette
    overwriting that would turn a risk flag into decoration.
    """

    last = max((node.rank for node in nodes.values()), default=0)
    for node_id, node in nodes.items():
        if chart.node_style.get(node_id):
            continue
        # Spread across the palette rather than cycling through it. A nine-box
        # chain against five colours would otherwise reuse the first colour
        # halfway along, which reads as "back to the start" on a diagram whose
        # whole point is direction.
        index = round(node.rank * (len(LEVEL_COLOURS) - 1) / last) if last else 0
        node.fill, node.stroke = LEVEL_COLOURS[index]


def _size_nodes(nodes: dict[str, _Node], max_width: float = NODE_MAX_WIDTH) -> None:
    """Size every box to its own text, with the same padding all round.

    Height follows the number of lines, so a two-line label gets a taller box
    rather than being squeezed into a one-line one. Boxes of different heights
    are centred against each other by _place, which is what keeps a level
    looking level without forcing every box to the same size.
    """

    for node in nodes.values():
        width = max(text_width(line, FONT_SIZE) for line in node.lines) + 2 * NODE_PADDING_X
        node.width = min(max_width, max(NODE_MIN_WIDTH, width))
        node.height = len(node.lines) * LINE_HEIGHT + 2 * NODE_PADDING_Y


def _place(  # noqa: PLR0913
    ranks: dict[int, list[str]],
    nodes: dict[str, _Node],
    vertical: bool = False,
    labelled: dict[int, float] | None = None,
    max_width: float = NODE_MAX_WIDTH,
) -> tuple[float, float]:
    """Assign coordinates and return the drawing size.

    The two directions are the same layout with the axes swapped: levels advance
    along one axis, the nodes of a level spread along the other, and each level
    is centred against the largest one so the diagram reads balanced instead of
    hanging off one edge.

    The gap after a level is wide only when an edge leaving it carries a label.
    """

    _size_nodes(nodes, max_width)

    # Extent of each level along the cross axis.
    spans: dict[int, float] = {}
    for key, rank_nodes in ranks.items():
        sizes = [nodes[n].width if vertical else nodes[n].height for n in rank_nodes]
        spans[key] = sum(sizes) + NODE_GAP * max(0, len(rank_nodes) - 1)
    widest = max(spans.values(), default=0.0)

    labelled = labelled or {}
    wide_gap = VERTICAL_RANK_GAP if vertical else RANK_GAP
    bare_gap = BARE_VERTICAL_RANK_GAP if vertical else BARE_RANK_GAP

    def gap_after(rank_key: int) -> float:
        if rank_key not in labelled:
            return bare_gap
        return max(wide_gap, labelled[rank_key])

    along = MARGIN
    last_gap = bare_gap
    for key in sorted(ranks):
        rank_nodes = ranks[key]
        thickness = max(
            nodes[n].height if vertical else nodes[n].width for n in rank_nodes
        )
        # Every box on a level takes that level's width. Connectors leave from a
        # right edge and arrive at a left one, so both have to be straight, and
        # they cannot both be unless the boxes match: two suppliers whose names
        # differ in length left their level 62px apart and drew two wires of
        # different lengths into the same factory.
        for node_id in rank_nodes:
            if vertical:
                nodes[node_id].height = thickness
            else:
                nodes[node_id].width = thickness
        across = MARGIN + (widest - spans[key]) / 2
        for node_id in rank_nodes:
            node = nodes[node_id]
            if vertical:
                node.x = across
                node.y = along + (thickness - node.height) / 2
                across += node.width + NODE_GAP
            else:
                node.x = along
                node.y = across
                across += node.height + NODE_GAP
        last_gap = gap_after(key)
        along += thickness + last_gap

    extent = along - last_gap + MARGIN
    if vertical:
        return widest + 2 * MARGIN, extent
    return extent, widest + 2 * MARGIN


def _is_linear_chain(nodes: dict[str, _Node], edges: list["_Edge"]) -> bool:
    """True when the diagram is one unbranched run of boxes.

    Only this shape can be wrapped onto a second row without the picture losing
    its meaning: there is exactly one path through it, so a reader who reaches
    the end of a row has only one place to continue. A branching diagram wrapped
    the same way would put siblings on different rows and imply an order between
    them that the data does not have.
    """

    if not edges:
        return False
    out_count: dict[str, int] = {}
    in_count: dict[str, int] = {}
    for edge in edges:
        out_count[edge.src] = out_count.get(edge.src, 0) + 1
        in_count[edge.dst] = in_count.get(edge.dst, 0) + 1
    return (
        len(edges) == len(nodes) - 1
        and all(count <= 1 for count in out_count.values())
        and all(count <= 1 for count in in_count.values())
    )


def _place_wrapped(
    order: list[str],
    nodes: dict[str, _Node],
    budget: float,
) -> tuple[float, float]:
    """Lay a chain out over as many rows as the page width needs.

    This is the chart module's contract applied to flowcharts: the page width is
    fixed and the content arranges itself into it, rather than the drawing being
    scaled down until it fits. charts.py packs more months into the same width by
    moving points closer together; a chain does it by starting a new row.

    Before this, the nine-box supply chain in Business Activity's section 1 came
    out 1,649px wide, was scaled to 45% to reach the page, and printed at 4.7pt
    against 9.5pt body text.
    """

    rows: list[list[str]] = []
    current: list[str] = []
    used = 0.0
    for node_id in order:
        width = nodes[node_id].width
        addition = width if not current else width + BARE_RANK_GAP
        if current and used + addition > budget:
            rows.append(current)
            current, used = [node_id], width
        else:
            current.append(node_id)
            used += addition
    if current:
        rows.append(current)

    y = MARGIN
    widest = 0.0
    for row_index, row in enumerate(rows):
        height = max(nodes[n].height for n in row)
        x = MARGIN
        for node_id in row:
            node = nodes[node_id]
            node.row = row_index
            node.x = x
            node.y = y + (height - node.height) / 2
            x += node.width + BARE_RANK_GAP
        widest = max(widest, x - BARE_RANK_GAP)
        y += height + WRAP_ROW_GAP

    return widest + MARGIN, y - WRAP_ROW_GAP + MARGIN


def _edge_ends(
    src: _Node,
    dst: _Node,
    vertical: bool,
) -> tuple[float, float, float, float]:
    """Where a connector leaves the source and meets the target."""

    if vertical:
        return src.cx, src.y + src.height, dst.cx, dst.y
    return src.x + src.width, src.cy, dst.x, dst.cy


def _wrap_edge_path(src: _Node, dst: _Node) -> str:
    """Connector from the end of one wrapped row to the start of the next.

    Leaves the source downwards and arrives on top of the target, so the arrow
    reads as "continue below" rather than as a link back up the chain. Direction
    matters more here than anywhere else in these diagrams: section 1 is a supply
    chain, and its guidance is explicit that inputs must not appear to flow from
    the output end.
    """

    mid_y = src.y + src.height + (dst.y - (src.y + src.height)) / 2
    return (
        f"M{src.cx:.1f} {src.y + src.height:.1f} "
        f"L{src.cx:.1f} {mid_y:.1f} "
        f"L{dst.cx:.1f} {mid_y:.1f} "
        f"L{dst.cx:.1f} {dst.y:.1f}"
    )


def _edge_path(src: _Node, dst: _Node, vertical: bool = False) -> str:
    """Orthogonal connector: out of the source, across, into the target."""

    if src.row != dst.row:
        return _wrap_edge_path(src, dst)

    x1, y1, x2, y2 = _edge_ends(src, dst, vertical)
    if vertical:
        if abs(x1 - x2) < 0.5:
            return f"M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}"
        mid = y1 + (y2 - y1) / 2
        return (
            f"M{x1:.1f} {y1:.1f} L{x1:.1f} {mid:.1f} "
            f"L{x2:.1f} {mid:.1f} L{x2:.1f} {y2:.1f}"
        )
    if abs(y1 - y2) < 0.5:
        return f"M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}"
    mid = x1 + (x2 - x1) / 2
    return (
        f"M{x1:.1f} {y1:.1f} L{mid:.1f} {y1:.1f} "
        f"L{mid:.1f} {y2:.1f} L{x2:.1f} {y2:.1f}"
    )


def render_svg(chart) -> str | None:
    """Draw a parsed flowchart as inline SVG, or None if it has no edges."""

    if not chart.edges:
        return None

    def _build(max_width: float):
        built: dict[str, _Node] = {}
        for node_id in chart.order:
            style = chart.node_style.get(node_id, {})
            built[node_id] = _Node(
                node_id=node_id,
                lines=_text_lines(chart.labels[node_id], max_width),
                fill=style.get("fill", DEFAULT_FILL),
                stroke=style.get("stroke", DEFAULT_STROKE),
                colour=style.get("color", DEFAULT_TEXT),
            )
        built_edges = [
            _Edge(src, dst, label, _text_lines(label) if label else [])
            for src, label, dst in chart.edges
            if src in built and dst in built
        ]
        return built, built_edges

    def _layout(max_width: float):
        built, built_edges = _build(max_width)
        if not built_edges:
            return None
        for node_id, rank in _assign_ranks(list(built), built_edges).items():
            built[node_id].rank = rank
        by_rank: dict[int, list[str]] = {}
        for node_id in chart.order:
            by_rank.setdefault(built[node_id].rank, []).append(node_id)
        _order_within_ranks(by_rank, built_edges, built)
        _colour_by_level(chart, built)
        # "flowchart TD" must come out top-down here too. Drawn left-to-right it
        # would contradict the markdown view of the same report, which is the
        # exact mismatch this renderer exists to prevent.
        is_vertical = bool(getattr(chart, "vertical", False))
        gaps = _label_gaps(built_edges, built)
        w, h = _place(by_rank, built, is_vertical, gaps, max_width)
        # Too wide for the page, and shaped so that wrapping keeps its meaning:
        # lay it out into the page width instead of shrinking it to reach it.
        if (
            not is_vertical
            and w > PAGE_CONTENT_WIDTH
            and not gaps
            and _is_linear_chain(built, built_edges)
        ):
            w, h = _place_wrapped(
                chart.order, built, PAGE_CONTENT_WIDTH - 2 * MARGIN
            )
        return built, built_edges, by_rank, is_vertical, w, h

    first = _layout(NODE_MAX_WIDTH)
    if first is None:
        return None
    nodes, edges, ranks, vertical, width, height = first
    # Narrow the boxes before shrinking the text. A diagram wider than the page
    # is scaled to fit, and the scale applies to the type as well — a five-rank
    # chart of Vietnamese company names came out at 6.5pt on paper, below the
    # 8pt this file treats as the floor for readable print. Re-wrapping the
    # labels into narrower boxes trades width for height, and height is free:
    # the page scrolls, the width does not. Same trade _place_wrapped makes for
    # a linear chain, applied to shapes it cannot help.
    if not vertical and width > PAGE_CONTENT_WIDTH:
        for candidate in (160, 140, 120, 100):
            if FONT_SIZE * (PAGE_CONTENT_WIDTH / width) * PX_TO_PT >= MIN_READABLE_FONT:
                break
            retry = _layout(candidate)
            if retry is None:
                break
            nodes, edges, ranks, vertical, width, height = retry

    tallest_label = max((len(edge.lines) for edge in edges), default=0)
    # How far a label block reaches either side of its connector. The drawing is
    # sized from the boxes, and a tall label sits outside them — a three-line one
    # ran off the top of the viewBox and was clipped in the PDF.
    label_overhang = (
        (tallest_label - 1) * (EDGE_FONT_SIZE + 1) / 2 + EDGE_FONT_SIZE
        if tallest_label > 1
        else 0.0
    )

    out_degree: dict[str, int] = {node_id: 0 for node_id in nodes}
    in_degree: dict[str, int] = {node_id: 0 for node_id in nodes}
    for edge in edges:
        out_degree[edge.src] += 1
        in_degree[edge.dst] += 1

    # Scale here, not in CSS. Explicit width AND height attributes too, since
    # width="100%" with height:auto makes WeasyPrint compute zero height and
    # draw nothing at all.
    if label_overhang:
        for node in nodes.values():
            node.y += label_overhang
        height += 2 * label_overhang

    scale = min(1.0, PAGE_CONTENT_WIDTH / width) if width else 1.0
    draw_width, draw_height = width * scale, height * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{draw_width:.0f}" height="{draw_height:.0f}" '
        f'font-family="inherit" role="img">',
        '<defs><marker id="mmdarrow" markerWidth="8" markerHeight="6" refX="8" '
        'refY="3" orient="auto" markerUnits="userSpaceOnUse">'
        f'<polygon points="0 0, 8 3, 0 6" fill="{EDGE_COLOUR}"/></marker></defs>',
    ]

    for edge in edges:
        src, dst = nodes[edge.src], nodes[edge.dst]
        parts.append(
            f'<path d="{_edge_path(src, dst, vertical)}" fill="none" '
            f'stroke="{EDGE_COLOUR}" stroke-width="{EDGE_STROKE_WIDTH}" '
            f'marker-end="url(#mmdarrow)"/>'
        )
        if edge.lines:
            x1, y1, x2, y2 = _edge_ends(src, dst, vertical)
            if vertical:
                # Beside the connector, not on it: a top-down connector is
                # vertical, so anything centred on it lands on the line.
                label_x = (x1 + x2) / 2 + 4
                label_y = (y1 + y2) / 2
                anchor = "start"
            elif in_degree[edge.dst] > 1 and out_degree[edge.src] <= 1:
                # A fan. Its connectors dogleg across a shared vertical trunk, so
                # the true middle of one is *on* that trunk, and its middle height
                # is between two rows — a label centred there covers the trunk and
                # floats between the two connectors it might belong to. Centre it
                # on the horizontal run at the end where the edges fan apart,
                # which is the stretch this edge has a y to itself.
                #
                # Keyed on the fan and not on the dogleg, because the middle spoke
                # of an odd fan runs straight into the hub: measured from its own
                # ends it landed 23px left of its neighbours, the ragged look
                # again. Every label in one fan shares an x.
                trunk = x1 + (x2 - x1) / 2
                half = max(text_width(line, EDGE_FONT_SIZE) for line in edge.lines) / 2
                label_x = min((x1 + trunk) / 2, trunk - EDGE_LABEL_MARGIN - half)
                label_y = y1
                anchor = "middle"
            elif out_degree[edge.src] > 1 and in_degree[edge.dst] <= 1:
                trunk = x1 + (x2 - x1) / 2
                half = max(text_width(line, EDGE_FONT_SIZE) for line in edge.lines) / 2
                label_x = max((trunk + x2 - ARROW_LENGTH) / 2,
                              trunk + EDGE_LABEL_MARGIN + half)
                label_y = y2
                anchor = "middle"
            else:
                # Centred between the source box and where the arrowhead starts,
                # not in the whole gap: the arrow eats the last 8px of it.
                label_x = x1 + (x2 - ARROW_LENGTH - x1) / 2
                label_y = y1
                anchor = "middle"
            # Centre the block on the connector. Shifting it up by a full
            # line per extra line put a three-line label entirely above the
            # wire, and on a one-rank diagram that is above the drawing.
            # Every label sits the same distance above its own wire, sloping or
            # flat. They used to be handled differently — centred on a sloping
            # connector, lifted above a flat one — and in a fan, where the middle
            # branch is flat and the rest slope, that put one label in a
            # different relationship to its wire than its neighbours. Measured on
            # a five-way fan the gaps between labels came out 53, 45.5, 60.5, 53:
            # even spacing was never possible while two rules were in play.
            block = (len(edge.lines) - 1) * (EDGE_FONT_SIZE + 1)
            label_y -= (
                block
                + EDGE_LABEL_MARGIN
                + EDGE_FONT_SIZE * DESCENDER_RATIO
                + EDGE_STROKE_WIDTH / 2
            )
            # A concentration warning is a property of the *pair*, not of the
            # box: "52%" is the finding and the box is who it is about. Drawn in
            # the default grey it read as ordinary annotation on a red box. Take
            # the colour from whichever endpoint diagrams.py styled — the hub is
            # never the styled one, so checking the target then the source picks
            # out the flagged partner whether it supplies or buys.
            warned = chart.node_style.get(edge.dst) or chart.node_style.get(edge.src)
            label_fill = (warned or {}).get("color") or EDGE_LABEL_COLOUR
            weight = f' font-weight="{HIGHLIGHT_WEIGHT}"' if warned else ""
            for index, line in enumerate(edge.lines):
                parts.append(
                    f'<text x="{label_x:.1f}" '
                    f'y="{label_y + index * (EDGE_FONT_SIZE + 1):.1f}" '
                    f'font-size="{EDGE_FONT_SIZE}" fill="{label_fill}"{weight} '
                    f'text-anchor="{anchor}">{html.escape(line)}</text>'
                )

    for node in nodes.values():
        # The depth in the pen this styling came from is a box-shadow, and two
        # measurements decided how to get it here. WeasyPrint drops box-shadow
        # with a warning; it drops feDropShadow with no warning at all — a
        # filtered rect and an unfiltered one produced byte-identical output.
        # An offset rectangle behind the box is plain geometry, so it cannot be
        # dropped, and opacity does render: sampled from the PDF, an 0.13 fill
        # comes out at tone 228 against 41 for the same colour at full strength.
        parts.append(
            f'<rect x="{node.x + SHADOW_OFFSET:.1f}" y="{node.y + SHADOW_OFFSET:.1f}" '
            f'width="{node.width:.1f}" height="{node.height:.1f}" rx="3" '
            f'fill="{SHADOW_COLOUR}" opacity="{SHADOW_OPACITY}"/>'
        )
        parts.append(
            f'<rect x="{node.x:.1f}" y="{node.y:.1f}" width="{node.width:.1f}" '
            f'height="{node.height:.1f}" rx="3" fill="{node.fill}" '
            f'stroke="{node.stroke}" stroke-width="1.2"/>'
        )
        # Centred in the box rather than pinned to its top, now that every box
        # is as tall as the tallest label needs.
        block = len(node.lines) * LINE_HEIGHT
        first = node.y + (node.height - block) / 2 + LINE_HEIGHT * 0.78
        weight = (f' font-weight="{HIGHLIGHT_WEIGHT}"'
                  if chart.node_style.get(node.node_id) else "")
        for index, line in enumerate(node.lines):
            parts.append(
                f'<text x="{node.cx:.1f}" y="{first + index * LINE_HEIGHT:.1f}" '
                f'font-size="{FONT_SIZE}" fill="{node.colour}"{weight} '
                f'text-anchor="middle">{html.escape(line)}</text>'
            )

    parts.append("</svg>")
    return f'<div class="mmd mmd-svg">{"".join(parts)}</div>'
