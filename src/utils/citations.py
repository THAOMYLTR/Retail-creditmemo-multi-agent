"""Source citations as markdown footnotes, and the checks that keep them honest.

Specialists cite a source by writing a marker — ``[^1]`` — beside the figure it
came from, and one definition line per marker at the end of their section. The
memo composer then merges five such reports into one, which is where the naive
version breaks: five agents each numbering from 1 produce five different
``[^1]``, and ``markdown`` resolves a repeated label by silently keeping the
LAST definition and discarding the rest. A citation would then point at another
agent's document with nothing to show it had happened.

So the labels are namespaced per agent before the merge (``[^pf1]``, ``[^cr1]``),
and gathered afterwards by ``consolidate_footnotes`` rather than by the composer
LLM — the same division of labour as heading renumbering, where the prompt asks
and ``_post_process_memo`` enforces.

Three failure modes were measured against ``markdown==3.10.3`` rather than
assumed, and each is reported instead of being silently repaired:

- a repeated label loses every definition but the last, with no warning;
- a marker with no definition renders as literal ``[^7]`` text in the PDF;
- a definition nobody references still appears in the source list.

Two more were measured the same way but ARE silently repaired, because both are
pure formatting slips rather than questions about the evidence:

- an edit that runs two definitions together on one source line
  (``[^pf1]: x [^pf2]: y``) produces exactly the second failure mode above for
  ``[^pf2]`` — no line anchors it as its own definition, so it is swallowed into
  ``[^pf1]``'s source text. ``_split_squeezed_definitions`` un-squeezes these
  before either function below ever sees them, so the audit never has to tell a
  genuinely missing source from one that just needed a line break;
- two labels defined with the same source, which prints one document twice in
  the list. The prompt already tells agents to reuse a marker for a repeated
  source; ``_canonical_labels`` is the enforcement half, collapsing them onto
  the label the reader meets first and rewriting the markers to match. Nothing
  is in doubt here — the two definitions say the same thing — so there is
  nothing to ask a reviewer.

Numbering is the library's job, ours is the order: it numbers definitions in the
order they are *defined*, so a list assembled in any other order makes the reader
meet footnote 3 before footnote 1. ``consolidate_footnotes`` therefore sorts by
first reference in the body.

Pure text analysis — no LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.common import normalize_text

# Which prefix each agent's labels get before the composer merges them. Two
# letters, because the label is visible in the .md source a reviewer reads.
AGENT_LABEL_PREFIXES: dict[str, str] = {
    "CUSTOMER_PROFILE_AGENT": "pf",
    "CREDIT_RELATIONSHIP_AGENT": "cr",
    "INCOME_ANALYSIS_AGENT": "ia",
    "CREDIT_PROPOSAL_AGENT": "cp",
    "RISK_ASSESSMENT_AGENT": "ra",
}

# A definition line: "[^label]: source text". Must be anchored at line start —
# an inline "[^1]:" mid-sentence is a reference followed by a colon, not a
# definition, and markdown treats it that way too.
_DEFINITION = re.compile(r"^\[\^([^\]\s]+)\]:[ \t]?(.*)$")
# A reference: "[^label]" NOT followed by a colon. The negative lookahead is how
# markdown itself tells the two apart, so the same text splits the same way here.
_REFERENCE = re.compile(r"\[\^([^\]\s]+)\](?!:)")
_FENCE = re.compile(r"^\s*(```|~~~)")
# Where a definition starts, found anywhere in a line rather than anchored —
# used only to detect two definitions squeezed onto one source line (see
# _split_squeezed_definitions). _DEFINITION above stays anchored because that
# is the correct, narrower rule for what a lone definition line looks like.
_DEFINITION_START = re.compile(r"\[\^[^\]\s]+\]:")


@dataclass(frozen=True)
class FootnoteAudit:
    """What did not line up between markers and definitions.

    Empty lists mean a clean report. Nothing here is fixed automatically: a
    citation that does not resolve is a question about the evidence, and the
    reviewer is the one who can answer it.
    """

    orphan_references: list[str]
    unused_definitions: list[tuple[str, str]]
    duplicate_definitions: list[tuple[str, list[str]]]

    @property
    def is_clean(self) -> bool:
        return not (
            self.orphan_references
            or self.unused_definitions
            or self.duplicate_definitions
        )


def _mask_code_fences(text: str) -> list[bool]:
    """Mark which lines sit inside a fenced code block.

    A report carries ```linechart and ```mermaid blocks, and documentation of
    this very syntax would carry a literal "[^1]:" example. Neither is a
    citation, so both are skipped — the same guard ``_post_process_memo`` uses
    before it renumbers headings.
    """

    inside = False
    flags: list[bool] = []
    for line in text.splitlines():
        if _FENCE.match(line):
            # The fence line itself belongs to the block on both sides.
            flags.append(True)
            inside = not inside
            continue
        flags.append(inside)
    return flags


def _split_squeezed_definitions(text: str) -> str:
    """Break a line carrying 2+ footnote definitions into one per line.

    Measured failure mode: an LLM edit occasionally runs definitions together
    on one source line — "[^pf1]: nguồn 1 [^pf2]: nguồn 2" — while writing a
    single definition per line the rest of the time. ``_DEFINITION`` is
    anchored at line start, so on a squeezed line it captures the whole
    remainder as the first definition's source text; the second definition is
    never recognised. Its reference then has nothing to resolve to and prints
    as literal "[^pf2]" instead of a superscript, and the source list ends up
    with the two definitions run together on one line instead of two.

    A no-op on every line with 0 or 1 definition marker, which covers all
    text this has ever been measured against.
    """

    if not text:
        return text

    flags = _mask_code_fences(text)
    out: list[str] = []
    for line, in_code in zip(text.splitlines(), flags):
        if in_code:
            out.append(line)
            continue
        starts = [m.start() for m in _DEFINITION_START.finditer(line)]
        if len(starts) < 2:
            out.append(line)
            continue
        bounds = starts + [len(line)]
        if starts[0] > 0:
            out.append(line[: starts[0]].strip())
        for start, end in zip(bounds, bounds[1:]):
            out.append(line[start:end].strip())
    return "\n".join(out)


def _canonical_labels(
    seen: dict[str, list[str]],
    referenced: list[str],
) -> dict[str, str]:
    """Map every label to the one label that will represent its source.

    Labels whose authoritative source text is the same thing said the same way
    collapse onto one representative; everything else maps to itself. Grouped on
    ``normalize_text`` rather than the raw string so "…, trang 1", "…, trang 1."
    and "…,  trang 1" count as one source, while "trang 1" and "trang 2" stay
    apart.

    The representative is whichever member the reader meets first, so numbering
    still follows reading order. A group nobody references keeps its
    first-defined label — there is no reading order to follow.
    """

    order = {label: index for index, label in enumerate(referenced)}
    unreferenced_rank = len(order)
    groups: dict[str, list[str]] = {}
    for label, sources in seen.items():
        # Last definition wins, matching how markdown resolves a repeated label,
        # so grouping sees the same source text the reader would have got.
        groups.setdefault(normalize_text(sources[-1]), []).append(label)

    canonical: dict[str, str] = {}
    for members in groups.values():
        if len(members) == 1:
            canonical[members[0]] = members[0]
            continue
        winner = min(members, key=lambda label: order.get(label, unreferenced_rank))
        for label in members:
            canonical[label] = winner
    return canonical


def _rewrite_references(lines: list[str], canonical: dict[str, str]) -> list[str]:
    """Point every marker at its representative label, leaving code blocks alone."""

    flags = _mask_code_fences("\n".join(lines))
    out: list[str] = []
    for line, in_code in zip(lines, flags):
        if in_code:
            out.append(line)
            continue
        out.append(
            _REFERENCE.sub(
                lambda m: f"[^{canonical.get(m.group(1), m.group(1))}]",
                line,
            )
        )
    return out


def namespace_footnotes(text: str, prefix: str) -> str:
    """Prefix every footnote label so one agent's markers cannot collide.

    ``[^1]`` becomes ``[^pf1]`` in both references and definitions. Whole
    labels are matched, never substrings: renaming "1" by substring would also
    rewrite the "1" inside "10" and quietly point two footnotes at one source.

    Returns the text unchanged when there is no prefix or no footnote in it.
    """

    if not prefix or not text:
        return text

    text = _split_squeezed_definitions(text)
    flags = _mask_code_fences(text)
    out: list[str] = []
    for line, in_code in zip(text.splitlines(), flags):
        if in_code:
            out.append(line)
            continue
        line = _DEFINITION.sub(
            lambda m: f"[^{prefix}{m.group(1)}]:"
            + (f" {m.group(2)}" if m.group(2) else ""),
            line,
        )
        line = _REFERENCE.sub(lambda m: f"[^{prefix}{m.group(1)}]", line)
        out.append(line)
    return "\n".join(out)


def consolidate_footnotes(text: str) -> tuple[str, FootnoteAudit]:
    """Move every definition into one list at the end, ordered by first use.

    Returns ``(text, audit)`` from a single pass, and the pairing is deliberate.
    Gathering the definitions necessarily collapses repeated labels — keeping
    the last, as markdown does — which destroys the evidence that a collision
    ever happened. A separate checker running afterwards could never see it, so
    the gathering reports what it collapsed.

    Idempotent: consolidating an already-consolidated report is a no-op.
    """

    if not text:
        return text, FootnoteAudit([], [], [])

    text = _split_squeezed_definitions(text)
    flags = _mask_code_fences(text)
    lines = text.splitlines()

    # Pass 1: lift out the definitions, keeping every occurrence so repeats stay
    # visible. Body lines are kept in order and stripped of definitions.
    seen: dict[str, list[str]] = {}
    body: list[str] = []
    for line, in_code in zip(lines, flags):
        match = None if in_code else _DEFINITION.match(line)
        if match is None:
            body.append(line)
            continue
        label, source = match.group(1), match.group(2).strip()
        seen.setdefault(label, []).append(source)
        # Each specialist writes its own "---" above its definition block. Once
        # the definitions are lifted out, that rule introduces nothing — and
        # since only the agents that cited anything wrote one, the leftovers
        # appear between some sections and not others, reading as a layout bug.
        # Drops only a rule that directly preceded definitions; a rule the model
        # put between two paragraphs is followed by prose, not by a definition,
        # so it is never reached here.
        end = len(body)
        while end and not body[end - 1].strip():
            end -= 1
        if end and body[end - 1].strip() == "---":
            del body[end - 1:]

    if not seen:
        # Nothing to gather. Return untouched rather than appending an empty
        # rule — this runs on every response, including plain chat answers.
        return text, FootnoteAudit([], [], [])

    # Lifting the definitions out of an already-consolidated report leaves our
    # own separator behind with nothing under it. Drop trailing rules and blank
    # lines so running this twice does not stack a second "---" onto the first.
    while body and body[-1].strip() in {"", "---"}:
        body.pop()

    # Pass 2: order of first reference in the body, which is the order the
    # reader meets them and therefore the order they must be numbered in.
    def _reference_order(source_lines: list[str]) -> list[str]:
        flags = _mask_code_fences("\n".join(source_lines))
        found: list[str] = []
        for line, in_code in zip(source_lines, flags):
            if in_code:
                continue
            for label in _REFERENCE.findall(line):
                if label not in found:
                    found.append(label)
        return found

    # Same source under two labels is the reader seeing one document listed
    # twice. The prompt already asks agents to reuse a marker for a repeated
    # source, so this is the enforcement half of that rule — and it has to
    # rewrite the markers, not just the list, or the collapsed labels would
    # leave references pointing at definitions that no longer exist.
    # Captured before the merge drops labels: this reports one label defined
    # twice with *different* sources, which is a disagreement about the evidence
    # and must survive whatever the merge below collapses.
    duplicates = [
        (label, sources)
        for label, sources in seen.items()
        # Only a genuine disagreement is worth reporting: the same source
        # written twice is redundant, not dangerous.
        if len(sources) > 1 and len(set(sources)) > 1
    ]

    canonical = _canonical_labels(seen, _reference_order(body))
    if any(label != winner for label, winner in canonical.items()):
        body = _rewrite_references(body, canonical)
    for label, winner in canonical.items():
        if label != winner:
            del seen[label]

    body_text = "\n".join(body)
    referenced = _reference_order(body)

    orphans = [label for label in referenced if label not in seen]
    # Computed after the merge, and only over surviving labels: a definition
    # that went unreferenced but repeated a source already in the list is
    # redundant, not lost evidence, and reporting it would be noise. One with a
    # source of its own is still reported, exactly as before.
    unused = [
        (label, sources[-1]) for label, sources in seen.items() if label not in referenced
    ]

    # Referenced definitions first, in reading order; then the unreferenced
    # ones, kept rather than dropped so a numbering slip cannot delete evidence.
    ordered = [label for label in referenced if label in seen]
    ordered += [label for label, _ in unused]
    # Last definition wins, matching how markdown resolves a repeated label —
    # so what the audit calls authoritative is what the PDF will show.
    definitions = [f"[^{label}]: {seen[label][-1]}" for label in ordered]

    # The blank line before "---" is load-bearing: a rule directly under a text
    # line is setext syntax and swallows that line into an <h2>.
    consolidated = body_text.rstrip() + "\n\n---\n\n" + "\n".join(definitions) + "\n"
    return consolidated, FootnoteAudit(orphans, unused, duplicates)


def format_footnote_findings(audit: FootnoteAudit) -> list[str]:
    """Render an audit as reader-facing Vietnamese findings."""

    findings: list[str] = []
    for label in audit.orphan_references:
        findings.append(
            f"Chú thích [^{label}] được dùng trong bài nhưng không có dòng định "
            "nghĩa nguồn tương ứng — có thể mô hình đã bịa số hoặc quên ghi nguồn."
        )
    for label, source in audit.unused_definitions:
        findings.append(
            f'Nguồn [^{label}] ("{_excerpt(source)}") không được câu nào trong bài '
            "trích dẫn — có thể là nguồn thừa hoặc lỗi đánh số."
        )
    for label, sources in audit.duplicate_definitions:
        listed = "; ".join(f'"{_excerpt(s)}"' for s in sources)
        findings.append(
            f"Chú thích [^{label}] được định nghĩa {len(sources)} lần với nội dung "
            f"khác nhau ({listed}) — chỉ định nghĩa cuối cùng được giữ, cần rà soát "
            "để không mất nguồn."
        )
    return findings


def _excerpt(text: str, limit: int = 60) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"
