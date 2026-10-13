"""Render the corp_income result as Markdown — from your template if there is one.

Template file: ``src/templates/corp-income-structure.md`` (same folder as the
retail agents' templates). When it exists, the report is that file with every
``{{placeholder}}`` filled in; when it does not, the built-in layout below
(``DEFAULT_TEMPLATE``) is used. Either way the content comes from the
extraction and the checks — the template only decides layout and wording, so a
figure in the report can only come from a document.

Placeholders available in the template (see ``placeholder_names()``):

    General      {{as_of}} {{borrowers}} {{summary_table}} {{documents_table}} {{scope_note}}
    Item 1       {{item1_title}} {{item1_table}} {{item1_findings}} {{item1_conclusion}}
                 one per field, value and source:
                 {{company_name}} {{company_name_source}} {{tax_code}} {{tax_code_source}}
                 {{company_name_foreign}} {{abbreviation}} {{legal_form}} {{establishment_date}}
                 {{registration_change_number}} {{latest_change_date}} {{charter_capital}}
                 {{head_office_address}} {{legal_representatives}} {{operating_status}}
    Item 2       {{item2_title}} {{members_table}} {{borrower_ownership}} {{item2_findings}} {{item2_conclusion}}
    Item 3       {{item3_title}} {{legal_rep_table}} {{actual_operator}} {{item3_findings}} {{item3_conclusion}}
    Item 4       {{item4_title}} {{tax_debts_table}} {{social_insurance_debts_table}}
                 {{administrative_penalties_table}} {{enforcement_actions_table}} {{disputes_table}}
                 {{item4_findings}} {{item4_conclusion}}

A placeholder the agent does not know is rendered as "—" and listed in
``unknown_placeholders(template)``, so a typo shows up instead of passing silently.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.agents.corp_income.checks import SOURCE_LABELS, fmt_money

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "src" / "templates" / "corp-income-structure.md"
PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

ICONS = {"alert": "🔴", "warning": "🟠", "info": "🔵", "ok": "🟢"}
ACQUISITION_LABELS = {
    "gop_von_thanh_lap": "Góp vốn thành lập",
    "nhan_chuyen_nhuong": "Nhận chuyển nhượng",
    "tang_von": "Tăng vốn",
    "thua_ke_tang_cho": "Thừa kế/tặng cho",
    "khac": "Khác",
}
COLUMN_LABELS = {
    "authority": "Cơ quan thuế", "tax_type": "Loại thuế", "amount": "Số tiền", "overdue_days": "Số ngày quá hạn",
    "as_of": "Tại ngày", "status": "Tình trạng", "agency": "Cơ quan BHXH", "months_overdue": "Số tháng nợ",
    "decision_no": "Số quyết định", "date": "Ngày", "domain": "Lĩnh vực", "violation": "Hành vi vi phạm",
    "remediation_status": "Khắc phục", "measure": "Biện pháp", "case_no": "Số vụ việc", "forum": "Tòa án/trọng tài",
    "role": "Vai trò DN", "counterparty": "Bên liên quan", "subject": "Nội dung", "claim_amount": "Giá trị tranh chấp",
}
SEVERITY_LABELS = {"alert": "Cảnh báo", "warning": "Cần làm rõ", "info": "Ghi nhận", "ok": "Khớp"}
ITEM1_FIELDS = (
    "company_name", "company_name_foreign", "abbreviation", "tax_code", "legal_form", "establishment_date",
    "registration_change_number", "latest_change_date", "charter_capital", "head_office_address",
    "legal_representatives", "operating_status",
)
ITEM4_TABLES = {
    "tax_debts": ("authority", "tax_type", "amount", "overdue_days", "as_of", "status"),
    "social_insurance_debts": ("agency", "amount", "months_overdue", "as_of", "status"),
    "administrative_penalties": ("decision_no", "date", "domain", "violation", "amount", "remediation_status"),
    "enforcement_actions": ("decision_no", "date", "measure", "amount"),
    "disputes": ("case_no", "forum", "role", "counterparty", "subject", "claim_amount", "status"),
}

# The built-in layout, written with the same placeholders a custom template uses.
DEFAULT_TEMPLATE = """# Corp Income Assessment — Pháp lý doanh nghiệp (mục 1–4)

- Ngày tham chiếu (thời điểm vay): **{{as_of}}** — cửa sổ kiểm tra 12 tháng tính đến ngày này
- KH cá nhân vay vốn: {{borrowers}}

## Tóm tắt

{{summary_table}}

## 1. {{item1_title}}

{{item1_table}}

{{item1_findings}}

## 2. {{item2_title}}

Vốn điều lệ: **{{charter_capital}}**

{{members_table}}

{{item2_findings}}

## 3. {{item3_title}}

{{legal_rep_table}}

Người điều hành thực tế: {{actual_operator}}

{{item3_findings}}

## 4. {{item4_title}}

**Nợ thuế:** {{tax_debts_table}}

**Nợ BHXH:** {{social_insurance_debts_table}}

**Xử phạt vi phạm hành chính:** {{administrative_penalties_table}}

**Cưỡng chế:** {{enforcement_actions_table}}

**Tranh chấp / kiện tụng:** {{disputes_table}}

{{item4_findings}}

## Hồ sơ đã xử lý

{{documents_table}}

> {{scope_note}}
"""


# --------------------------------------------------------------------------- #
def _cell(value: Any) -> str:
    text = "—" if value in (None, "", []) else str(value)
    return text.replace("|", "/").replace("\n", " ")


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _findings(findings: list[dict[str, Any]]) -> str:
    lines = ["**Kết quả kiểm tra**", ""]
    for finding in findings:
        lines.append(f"- {ICONS[finding['severity']]} **{SEVERITY_LABELS[finding['severity']]}** — {finding['message']}")
        lines += [f"  - {evidence}" for evidence in finding.get("evidence") or []]
    return "\n".join(lines)


def build_context(result: dict[str, Any]) -> dict[str, str]:
    """Every placeholder → its rendered Markdown."""

    items = result["items"]
    meta = result["meta"]
    ctx: dict[str, str] = {
        "as_of": meta["as_of"],
        "borrowers": ", ".join(meta["borrower_names"]) or "—",
        "summary_table": _table(
            ["Mục", "Nội dung", "Kết luận"],
            [[str(n), item["title"], _cell(item["conclusion"])] for n, item in items.items()],
        ),
        "scope_note": ("Kiểm tra AML/BL/WL chỉ đối chiếu với danh sách cảnh báo được truyền vào (`watchlist`). "
                       "Phát hiện hồ sơ bị chỉnh sửa (Fraud Detection) nằm ngoài phạm vi agent này."),
    }
    for number, item in items.items():
        ctx[f"item{number}_title"] = item["title"]
        ctx[f"item{number}_findings"] = _findings(item["findings"])
        ctx[f"item{number}_conclusion"] = item["conclusion"] or "—"

    # Item 1
    rows = []
    by_field = {entry.get("field"): entry for entry in items[1]["summary"].values()}
    for label, entry in items[1]["summary"].items():
        rows.append([label, _cell(entry["value"]), _cell(entry["source"])])
    ctx["item1_table"] = _table(["Thông tin", "Giá trị", "Nguồn"], rows)
    for field in ITEM1_FIELDS:
        entry = by_field.get(field) or {}
        ctx[field] = _cell(entry.get("value"))
        ctx[f"{field}_source"] = _cell(entry.get("source"))

    # Item 2
    summary2 = items[2]["summary"]
    if summary2.get("charter_capital"):
        ctx["charter_capital"] = fmt_money(summary2["charter_capital"])
    member_rows = []
    for member in summary2.get("members", []):
        pct = member["ownership_pct"]
        pct_text = "—" if pct is None else f"{pct}%" + (" (tính toán)" if member["pct_source"] == "tính toán" else "")
        member_rows.append([
            _cell(member["name"]),
            "Pháp nhân" if member["member_type"] == "phap_nhan" else "Cá nhân",
            fmt_money(member["capital_amount"]),
            pct_text,
            _cell(member["acquisition_date"]),
            _cell(ACQUISITION_LABELS.get(member["acquisition_method"], member["acquisition_method"])),
            _cell(member["source"]),
        ])
    ctx["members_table"] = _table(
        ["Thành viên/cổ đông", "Loại", "Vốn góp", "Tỷ lệ", "Ngày nhận vốn góp", "Hình thức", "Nguồn"], member_rows)
    borrowers = summary2.get("borrowers") or []
    ctx["borrower_ownership"] = "; ".join(
        f"{b['name']}: {b['ownership_pct']}%" if b["ownership_pct"] is not None else f"{b['name']}: chưa rõ tỷ lệ"
        for b in borrowers
    ) or "—"

    # Item 3
    summary3 = items[3]["summary"]
    ctx["legal_rep_table"] = _table(
        ["Người đại diện", "Chức danh", "Ngày bổ nhiệm", "Nguồn"],
        [[_cell(r.get("name")), _cell(r.get("title")), _cell(r.get("appointed_date")), _cell(r.get("source"))]
         for r in summary3.get("legal_representatives", [])],
    )
    operator = summary3.get("actual_operator")
    ctx["actual_operator"] = (f"**{operator['name']}** ({operator.get('role') or '—'}) — {operator['source']}"
                              if operator else "chưa có thông tin")

    # Item 4
    summary4 = items[4]["summary"]
    for key, columns in ITEM4_TABLES.items():
        rows4 = summary4.get(key) or []
        if not rows4:
            ctx[f"{key}_table"] = "không ghi nhận"
            continue
        body = [[fmt_money(row.get(c)) if c in ("amount", "claim_amount") else _cell(row.get(c)) for c in columns]
                + [_cell(row.get("source"))] for row in rows4]
        ctx[f"{key}_table"] = "\n\n" + _table([COLUMN_LABELS.get(c, c) for c in columns] + ["Nguồn"], body)

    ctx["documents_table"] = _table(
        ["File", "Loại", "Nguồn", "Pass đã chạy", "Lỗi"],
        [[_cell(d["filename"]), _cell(d["label"]), SOURCE_LABELS.get(d["source"], d["source"]),
          ", ".join(d["passes_done"]) or "—", _cell("; ".join(f"{k}: {v}" for k, v in d["errors"].items()))]
         for d in result["documents"]],
    )
    ctx.setdefault("charter_capital", "—")
    return ctx


def placeholder_names() -> list[str]:
    """Every placeholder a template may use (built from an empty result)."""

    empty_item = {"title": "", "conclusion": "", "findings": [], "summary": {}}
    sample = {"meta": {"as_of": "", "borrower_names": []}, "documents": [],
              "items": {n: dict(empty_item) for n in (1, 2, 3, 4)}}
    return sorted(build_context(sample))


def unknown_placeholders(template: str) -> list[str]:
    known = set(placeholder_names())
    body = HTML_COMMENT.sub("", template)
    return sorted({name for name in PLACEHOLDER.findall(body) if name not in known})


def load_template(path: str | Path | None = None) -> tuple[str, str]:
    """Return ``(template_text, where_it_came_from)``."""

    candidate = Path(path) if path else TEMPLATE_PATH
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8"), str(candidate)
    if path:
        raise FileNotFoundError(f"corp_income template not found: {candidate}")
    return DEFAULT_TEMPLATE, "built-in"


def render_markdown(result: dict[str, Any], template: str | None = None) -> str:
    """Fill ``template`` (default: src/templates/corp-income-structure.md, else built-in)."""

    text = template if template is not None else load_template()[0]
    # Notes to the template author (<!-- ... -->) stay out of the report.
    text = HTML_COMMENT.sub("", text).lstrip("\n")
    ctx = build_context(result)
    return PLACEHOLDER.sub(lambda m: ctx.get(m.group(1), "—"), text)
