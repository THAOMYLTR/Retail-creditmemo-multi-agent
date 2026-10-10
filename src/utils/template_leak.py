"""Detect template scaffolding that leaked into a finished report.

The specialist prompt ships a layout skeleton plus internal analysis guidance.
A model that treats the skeleton as text to reproduce — rather than a form to
fill — copies both into the deliverable. This was observed verbatim in a real
run, where the report handed to the reader opened with the internal block
"**CÁCH DÙNG MẪU NÀY**: Đây là **mẫu bố cục tham khảo**…" followed by unfilled
placeholders ``[CompanyName]``, ``[TaxCode]``, ``[Year]``.

Pure text analysis — no LLM call.
"""

from __future__ import annotations

import re
import unicodedata

# Placeholders the model must substitute. The structure files write {{Name}};
# ChatPromptTemplate unescapes that to {Name} before the model sees it, so a
# leaked placeholder appears single-braced. Square-bracket placeholders are the
# older convention and still worth catching.
_CURLY_PLACEHOLDER = re.compile(r"\{\{?\s*[A-Za-z][A-Za-z0-9_\-]{2,40}\s*\}?\}")
_SQUARE_PLACEHOLDER = re.compile(
    r"\[(?:CompanyName|TaxCode|RegistrationNumber|Year(?:-\d)?|ReportType|"
    r"AuditingUnit|AuditOpinion|List_Input_Documents|High/Medium/Low|"
    r"GSO Sector[^\]]*)\]"
)

# Sentinel phrases that only exist in the guidance files or the prompt scaffold.
# Accent-insensitive comparison, so a paraphrase with different diacritics still
# registers.
SCAFFOLD_MARKERS = (
    "cach dung bo cuc",
    "cach dung mau nay",
    "huong dan phan tich",
    "chi dan noi bo",
    "khong duoc chep",
    "trong tam phan tich",
    "bo cuc bao cao",
    "mau bo cuc tham khao",
    "evidence rule",
    "citation rule",
    "highlight rule",
    "nhan dinh rule",
    # Kept after the rule it replaced was renamed: the old heading can still
    # reach a report through a cached response or an older prompt revision, and
    # a marker that matches nothing costs a substring scan.
    "assertion separation rule",
    "monetary unit rule",
    "language rule",
    "xoa han dong khong co du lieu",
    # Headings of the structured blocks in the prompt. An agent required to cite
    # a figure but given no filename will reach for the only label in front of
    # it, and "Căn cứ: [BẢNG TÍNH KHẢ NĂNG TRẢ NỢ]" means nothing to whoever
    # reads the memo. _strip_accents lowercases and drops accents but keeps
    # punctuation, so these are written in that form.
    "[bang tinh kha nang tra no]",
    "[bang tinh so tien vay toi da]",
    "[du lieu thu nhap da trich xuat]",
    "[du lieu giay de nghi vay von]",
    "[extracted cic credit report]",
    "[extracted cic r21 report]",
    "[extracted appraisal report]",
)


def _strip_accents(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _excerpt(text: str, limit: int = 110) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def check_template_leakage(report: str) -> list[str]:
    """Return findings for unfilled placeholders and copied scaffolding."""

    text = report or ""
    findings: list[str] = []

    placeholders = sorted(
        {match.group(0) for match in _CURLY_PLACEHOLDER.finditer(text)}
        | {match.group(0) for match in _SQUARE_PLACEHOLDER.finditer(text)}
    )
    if placeholders:
        shown = ", ".join(placeholders[:6])
        more = f" (và {len(placeholders) - 6} chỗ khác)" if len(placeholders) > 6 else ""
        findings.append(
            f"Báo cáo còn chỗ trống chưa thay bằng dữ liệu thật: {shown}{more}."
        )

    normalized = _strip_accents(text)
    leaked = [
        marker for marker in SCAFFOLD_MARKERS if marker in normalized
    ]
    if leaked:
        for line in text.splitlines():
            line_norm = _strip_accents(line)
            if any(marker in line_norm for marker in leaked):
                findings.append(
                    "Báo cáo chép cả hướng dẫn nội bộ / khung mẫu, cần loại bỏ: "
                    f"{_excerpt(line)}"
                )
                break
    return findings
