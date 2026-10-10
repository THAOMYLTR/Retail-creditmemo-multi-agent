"""The "Nguồn dữ liệu" list at the top of every report.

One line per thing the reader would think of as one source, each naming what the
document is and then the files it was read from:

    - Hợp đồng lao động không xác định thời hạn với Công ty ABC, ký 01/03/2021 —  <em>HDLD.pdf</em>
    - Sao kê tài khoản —  <em>SAO_KE_01.2026.pdf, SAO_KE_02.2026.pdf, SAO_KE_03.2026.pdf</em>

The list is built here rather than asked of the model, and the reason is on
record. The rule that used to ask for it shipped with worked examples; given
sixteen monthly returns the model answered with a four-month range, character
for character the example it had been shown. A credit memo that names four returns
when sixteen were read understates its own evidence, and nothing downstream can
tell. So the assembly is code, and the model is handed finished lines.

Two rules follow from what that failure was:

- **No document may vanish.** Every filename that comes in appears in the
  output, in full, extension included. A long line is a cosmetic problem; a
  short one is a false statement about what was read.
- **Nothing is invented.** The caption is either a description the classifier
  wrote from the document's own text, or the document type's short label from
  the matrix. Never a guess assembled from a filename.
"""

from __future__ import annotations

from src.matrix.document_matrix import get_type

# The classifier is asked for at most twenty words. Anything longer is not
# shortened here — it is dropped in favour of the type's name. Cutting a caption
# mid-sentence would leave the reader unable to tell whether what went missing
# was "(ký 03/01/2023)" or a condition that changes the meaning, which is the
# same reasoning that keeps ellipses out of the diagram labels.
MAX_DESCRIPTION_WORDS = 20


def _usable_description(description: str) -> str:
    """The caption a description can serve as, or "" when it cannot."""

    text = " ".join((description or "").split())
    if not text or len(text.split()) > MAX_DESCRIPTION_WORDS:
        return ""
    return text


def _short_label(document_type: str) -> str:
    """What to call a group the classifier gave no usable description for."""

    matched = get_type(document_type) if document_type else None
    return matched.short_label if matched else "Tài liệu khác"


def _cell_safe(text: str) -> str:
    """Escape what would break the markdown table cell this line lands in.

    An unescaped pipe ends the cell: measured through markdown 3.10.3, a caption
    containing "A|B" splits the row, the header's column count wins, and
    everything past the pipe is dropped — a document silently missing from the
    source list, which is the one thing this module exists to prevent.
    """

    return text.replace("|", r"\|")


def _line(caption: str, filenames: list[str]) -> str:
    """One source line: what the document is, in bold, then the files it came from."""

    names = ", ".join(_cell_safe(name) for name in filenames)
    return f"**{_cell_safe(caption)}** —  <em>{names}</em>"


def build_source_lines(
    documents: list[tuple[str, str, str]],
) -> list[str]:
    """Turn (filename, document_type, description) triples into report lines.

    Grouped by document type, because the type is already decided by the routing
    matrix and three bank statements carry the same id whatever they are called.

    Within a group the descriptions decide the shape, and this is the part worth
    reading twice. Files whose captions agree are one line: three monthly
    payslips say the same thing about themselves, so listing them separately
    would be three copies of one sentence. Files whose captions differ get a line
    each: two rental contracts are two different agreements, and collapsing them
    under "Hợp đồng cho thuê tài sản" would throw away the tenant, the property
    and the date that make each one worth citing.

    Input order is preserved, so the list follows the order documents were
    routed in.
    """

    groups: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    types: dict[str, str] = {}
    for index, (filename, document_type, description) in enumerate(documents):
        # An untyped document has nothing in common with any other, not even its
        # own unknown-ness, so each one keys to itself and stands alone.
        key = document_type or f"__unknown_{index}"
        if key not in groups:
            groups[key] = []
            order.append(key)
            types[key] = document_type
        groups[key].append((filename, _usable_description(description)))

    lines: list[str] = []
    for key in order:
        members = groups[key]
        fallback = _short_label(types[key])
        captions = {description or fallback for _, description in members}
        if len(captions) == 1:
            lines.append(_line(captions.pop(), [name for name, _ in members]))
            continue
        for filename, description in members:
            lines.append(_line(description or fallback, [filename]))
    return lines
