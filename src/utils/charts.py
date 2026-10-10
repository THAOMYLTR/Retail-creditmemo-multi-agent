"""Render ```linechart blocks as inline SVG so they survive PDF export.

The sibling of ``diagrams.mermaid_to_html``, and it exists for the same reason:
WeasyPrint has no JavaScript, but it renders inline SVG exactly.

The split of labour is deliberate. The pipeline writes the fenced block from
extracted JSON — the model never retypes the figures — and the block itself is
plain text, so ``final_response.md`` stays readable even though only the PDF
gets the drawing. Anything this parser cannot read is left as the original block
rather than dropped, so content is never lost.

Two things this renderer does that the Excel chart it was modelled on does not:
it picks a display unit from the data, and it moves data labels apart where the
two lines cross. Overlapping labels at a crossing are exactly the point where a
reader most needs to tell the series apart.
"""

from __future__ import annotations

import html
import math
import re

from src.utils.formatting import format_vn_number
from src.utils.graph_svg import MIN_READABLE_FONT, PAGE_CONTENT_WIDTH

CHART_BLOCK = re.compile(
    r"^```linechart[ \t]*\n(.*?)^```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

# Matches the sample workbook: Office blue and orange, then extras for a third
# and fourth series should one ever be added.
SERIES_COLOURS = ("#4472C4", "#ED7D31", "#548235", "#7030A0")

PLOT_HEIGHT = 196.0
TITLE_HEIGHT = 20.0
XLABEL_HEIGHT = 36.0
LEGEND_HEIGHT = 18.0
NOTE_HEIGHT = 15.0
MARGIN_RIGHT = 10.0
MARGIN_TOP = 8.0

TITLE_FONT = 10.5
AXIS_FONT = 6.8
VALUE_FONT = 6.0
LEGEND_FONT = 7.5
NOTE_FONT = 7.0
# Rough advance width per character; the same 0.52 factor graph_svg uses.
CHAR_WIDTH_RATIO = 0.52
# Beyond this many labelled points the chart stops being readable even though the
# labels still clear each other horizontally: a steeply zig-zagging series runs
# its own line straight through the label of the point below it. Measured at 28
# points, where roughly a third of the labels had a stroke through them.
MAX_VALUE_LABELS = 16

GRID_COLOUR = "#D9D9D9"
AXIS_TEXT = "#666666"
VALUE_TEXT = "#444444"
NOTE_TEXT = "#8A6D3B"

# Steps that produce round gridline labels, cycled across powers of ten.
_NICE_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)
# Aiming at 5 rounds a 58-wide span up to a step of 20, which leaves four
# gridlines and a third of the plot empty below the data. Six lands on 10.
_TARGET_GRIDLINES = 6

_UNIT_SCALES = (
    (10**9, "tỷ VNĐ"),
    (10**6, "triệu VNĐ"),
    (1, "VNĐ"),
)


def pick_unit(values: list[float]) -> tuple[float, str]:
    """Choose a display scale from the data's magnitude.

    The report body is converted to triệu VNĐ wholesale, but a mortgage balance
    in triệu reads 1.850 while a salary reads 25 — so the chart picks its own
    unit from the data and prints it on the axis rather than inheriting one.
    """

    largest = max((abs(value) for value in values if value is not None), default=0.0)
    for divisor, label in _UNIT_SCALES:
        if largest >= divisor:
            return float(divisor), label
    return 1.0, "VNĐ"


def _nice_step(span: float) -> float:
    """Gridline interval giving round labels and roughly _TARGET_GRIDLINES lines."""

    if span <= 0:
        return 1.0
    rough = span / _TARGET_GRIDLINES
    power = 10.0 ** math.floor(math.log10(rough))
    for step in _NICE_STEPS:
        if step * power >= rough:
            return step * power
    return 10.0 * power


class ChartSpec:
    """A parsed ```linechart block."""

    def __init__(
        self,
        title: str,
        unit: str,
        note: str,
        labels: list[str],
        columns: list[str],
        rows: list[list[float | None]],
    ) -> None:
        self.title = title
        self.unit = unit
        self.note = note
        self.labels = labels
        self.columns = columns
        self.rows = rows

    def series(self, index: int) -> list[float | None]:
        return [row[index] if index < len(row) else None for row in self.rows]


def _parse_number(text: str) -> float | None:
    """Read one cell. '-' and '' both mean "no figure for this month"."""

    cleaned = text.strip()
    if cleaned in ("", "-", "–", "—", "null", "None"):
        return None
    # Written Vietnamese-style by build_linechart_block: '.' groups, ',' decimal.
    cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_linechart(body: str) -> ChartSpec | None:
    """Parse a ```linechart body, or None when it is not readable.

    Returning None rather than raising is the contract that lets
    ``charts_to_html`` leave a malformed block untouched instead of eating it.
    """

    title = unit = note = ""
    columns: list[str] = []
    labels: list[str] = []
    rows: list[list[float | None]] = []

    for line in (body or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        head, _, tail = stripped.partition(":")
        key = head.strip().lower()
        if key == "title" and tail:
            title = tail.strip()
            continue
        if key == "unit" and tail:
            unit = tail.strip()
            continue
        if key == "note" and tail:
            note = tail.strip()
            continue
        if key == "columns" and tail:
            columns = [part.strip() for part in tail.split("|") if part.strip()]
            continue
        cells = [cell.strip() for cell in stripped.split("|")]
        if len(cells) < 2:
            continue
        labels.append(cells[0])
        rows.append([_parse_number(cell) for cell in cells[1:]])

    if not labels or not columns or not rows:
        return None
    if not any(value is not None for row in rows for value in row):
        return None
    return ChartSpec(title, unit, note, labels, columns, rows)


def build_linechart_block(
    title: str,
    unit: str,
    columns: list[str],
    labels: list[str],
    series: list[list[float | None]],
    note: str = "",
) -> str:
    """Write a ```linechart block from already-computed figures.

    Called by the pipeline, never by an agent: the whole point is that these
    numbers are transcribed from extracted JSON rather than retyped by a model.
    """

    lines = [f"title: {title}", f"unit: {unit}"]
    if note:
        lines.append(f"note: {note}")
    lines.append("columns: " + " | ".join(columns))
    for index, label in enumerate(labels):
        cells = []
        for column in series:
            value = column[index] if index < len(column) else None
            cells.append("-" if value is None else format_vn_number(value, 2))
        lines.append(" | ".join([label, *cells]))
    return "```linechart\n" + "\n".join(lines) + "\n```"


def _text_width(text: str, font_size: float) -> float:
    return len(text) * font_size * CHAR_WIDTH_RATIO


def _stride_for(count: int, available: float, widest: float) -> int:
    """How many points to skip so labels of ``widest`` px stop colliding."""

    if count <= 1 or available <= 0:
        return 1
    per_point = available / (count - 1)
    if per_point <= 0:
        return count
    return max(1, int(-(-widest // per_point)))  # ceil


def line_chart_svg(spec: ChartSpec) -> str | None:
    """Draw a parsed spec as inline SVG sized to the printable page width."""

    if not spec.labels or not spec.columns:
        return None
    numbers = [
        value for index in range(len(spec.columns)) for value in spec.series(index)
        if value is not None
    ]
    if not numbers:
        return None

    highest = max(numbers)
    lowest = min(0.0, min(numbers))
    step = _nice_step(highest - lowest)
    top = step * (int(highest / step) + 1) if highest > 0 else step
    bottom = -step * (int(-lowest / step) + 1) if lowest < 0 else 0.0

    tick_values = []
    tick = bottom
    while tick <= top + step / 2:
        tick_values.append(tick)
        tick += step
    tick_labels = [format_vn_number(value, 1) for value in tick_values]

    left = max(_text_width(label, AXIS_FONT) for label in tick_labels) + 10.0
    width = PAGE_CONTENT_WIDTH
    plot_width = width - left - MARGIN_RIGHT

    top_y = MARGIN_TOP + (TITLE_HEIGHT if spec.title else 0.0)
    plot_bottom = top_y + PLOT_HEIGHT
    legend_y = plot_bottom + XLABEL_HEIGHT + LEGEND_HEIGHT * 0.6
    height = plot_bottom + XLABEL_HEIGHT + LEGEND_HEIGHT + (
        NOTE_HEIGHT if spec.note else 0.0
    )

    count = len(spec.labels)

    def x_of(index: int) -> float:
        if count == 1:
            return left + plot_width / 2
        return left + plot_width * index / (count - 1)

    def y_of(value: float) -> float:
        span = top - bottom or 1.0
        return plot_bottom - PLOT_HEIGHT * (value - bottom) / span

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" font-family="inherit" role="img">'
    ]
    if spec.title:
        parts.append(
            f'<text x="{width / 2:.1f}" y="{MARGIN_TOP + TITLE_FONT:.1f}" '
            f'font-size="{TITLE_FONT}" font-weight="bold" fill="#111111" '
            f'text-anchor="middle">{html.escape(spec.title)}</text>'
        )

    for value, label in zip(tick_values, tick_labels):
        y = y_of(value)
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{width - MARGIN_RIGHT:.1f}" '
            f'y2="{y:.1f}" stroke="{GRID_COLOUR}" stroke-width="0.6"/>'
        )
        parts.append(
            f'<text x="{left - 5:.1f}" y="{y + AXIS_FONT * 0.36:.1f}" '
            f'font-size="{AXIS_FONT}" fill="{AXIS_TEXT}" text-anchor="end">'
            f"{html.escape(label)}</text>"
        )
    if spec.unit:
        parts.append(
            f'<text x="{left:.1f}" y="{top_y - 2:.1f}" font-size="{AXIS_FONT}" '
            f'fill="{AXIS_TEXT}" text-anchor="start">'
            f"{html.escape('Đơn vị: ' + spec.unit)}</text>"
        )

    # Rotated month labels, thinned only as far as collisions require.
    widest_x = max(_text_width(label, AXIS_FONT) for label in spec.labels) * 0.72
    x_stride = _stride_for(count, plot_width, widest_x)
    for index, label in enumerate(spec.labels):
        if index % x_stride:
            continue
        x = x_of(index)
        y = plot_bottom + 9
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{AXIS_FONT}" fill="{AXIS_TEXT}" '
            f'text-anchor="end" transform="rotate(-45 {x:.1f} {y:.1f})">'
            f"{html.escape(label)}</text>"
        )

    all_series = [spec.series(index) for index in range(len(spec.columns))]

    # A point's label goes above or below depending on where the other series
    # sits at that x, so the two never land on each other at a crossing.
    def label_offset(series_index: int, point_index: int) -> float:
        own = all_series[series_index][point_index]
        others = [
            other[point_index]
            for position, other in enumerate(all_series)
            if position != series_index and other[point_index] is not None
        ]
        if own is None or not others:
            return -4.0
        return -4.0 if own >= max(others) else VALUE_FONT + 3.0

    widest_value = max(
        (_text_width(format_vn_number(value, 1), VALUE_FONT) for value in numbers),
        default=0.0,
    )
    value_stride = max(
        _stride_for(count, plot_width, widest_value),
        -(-count // MAX_VALUE_LABELS),  # ceil
    )

    for series_index, values in enumerate(all_series):
        colour = SERIES_COLOURS[series_index % len(SERIES_COLOURS)]
        # Break the line wherever a month has no figure. A missing reporting
        # period is not a value of zero and must not be drawn as a slope.
        run: list[str] = []
        for index, value in enumerate(values):
            if value is None:
                if len(run) > 1:
                    parts.append(
                        f'<polyline points="{" ".join(run)}" fill="none" '
                        f'stroke="{colour}" stroke-width="1.4"/>'
                    )
                run = []
                continue
            run.append(f"{x_of(index):.1f},{y_of(value):.1f}")
        if len(run) > 1:
            parts.append(
                f'<polyline points="{" ".join(run)}" fill="none" stroke="{colour}" '
                f'stroke-width="1.4"/>'
            )

        for index, value in enumerate(values):
            if value is None:
                continue
            x, y = x_of(index), y_of(value)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8" fill="{colour}"/>')
            if index % value_stride:
                continue
            parts.append(
                f'<text x="{x:.1f}" y="{y + label_offset(series_index, index):.1f}" '
                f'font-size="{VALUE_FONT}" fill="{VALUE_TEXT}" text-anchor="middle">'
                f"{html.escape(format_vn_number(value, 1))}</text>"
            )

    entry_widths = [
        24.0 + _text_width(name, LEGEND_FONT) + 16.0 for name in spec.columns
    ]
    cursor = (width - sum(entry_widths)) / 2
    for index, name in enumerate(spec.columns):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        parts.append(
            f'<line x1="{cursor:.1f}" y1="{legend_y:.1f}" x2="{cursor + 16:.1f}" '
            f'y2="{legend_y:.1f}" stroke="{colour}" stroke-width="1.6"/>'
        )
        parts.append(
            f'<circle cx="{cursor + 8:.1f}" cy="{legend_y:.1f}" r="2.2" fill="{colour}"/>'
        )
        parts.append(
            f'<text x="{cursor + 21:.1f}" y="{legend_y + LEGEND_FONT * 0.35:.1f}" '
            f'font-size="{LEGEND_FONT}" fill="#333333">{html.escape(name)}</text>'
        )
        cursor += entry_widths[index]

    if spec.note:
        parts.append(
            f'<text x="{width / 2:.1f}" y="{height - 4:.1f}" font-size="{NOTE_FONT}" '
            f'fill="{NOTE_TEXT}" text-anchor="middle" font-style="italic">'
            f"{html.escape(spec.note)}</text>"
        )

    parts.append("</svg>")
    # Every font here is a fixed constant rather than a scale factor, so this is
    # a design guard rather than a runtime one: it fires only if someone tunes a
    # size below what prints legibly.
    if min(AXIS_FONT, VALUE_FONT, NOTE_FONT) < MIN_READABLE_FONT:
        parts.append(
            f"<!-- chart font below {MIN_READABLE_FONT}pt; illegible in print -->"
        )
    return "".join(parts)


def charts_to_html(text: str) -> str:
    """Replace ```linechart blocks with inline SVG for non-JS renderers.

    Unparseable blocks are returned untouched so no content is ever lost.
    """

    def replace(match: re.Match[str]) -> str:
        spec = parse_linechart(match.group(1))
        if spec is None:
            return match.group(0)
        rendered = line_chart_svg(spec)
        return rendered if rendered else match.group(0)

    return CHART_BLOCK.sub(replace, text or "")
