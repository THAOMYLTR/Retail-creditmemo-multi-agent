"""Render the corp_income result as Markdown (BRD "template" view) and plain JSON."""

from __future__ import annotations

from typing import Any

from src.agents.corp_income.checks import SOURCE_LABELS, fmt_money

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


def _cell(value: Any) -> str:
    text = "—" if value in (None, "", []) else str(value)
    return text.replace("|", "/").replace("\n", " ")


def _findings(lines: list[str], findings: list[dict[str, Any]]) -> None:
    lines.append("**Kết quả kiểm tra**\n")
    for finding in findings:
        lines.append(f"- {ICONS[finding['severity']]} **{SEVERITY_LABELS[finding['severity']]}** — {finding['message']}")
        for evidence in finding.get("evidence") or []:
            lines.append(f"  - {evidence}")
    lines.append("")


def render_markdown(result: dict[str, Any]) -> str:
    items = result["items"]
    meta = result["meta"]
    lines = [
        "# Corp Income Assessment — Pháp lý doanh nghiệp (mục 1–4)",
        "",
        f"- Ngày tham chiếu (thời điểm vay): **{meta['as_of']}** — cửa sổ kiểm tra 12 tháng tính đến ngày này",
        f"- KH cá nhân vay vốn: {', '.join(meta['borrower_names']) or '—'}",
        "",
        "## Tóm tắt",
        "",
        "| Mục | Nội dung | Kết luận |",
        "|---|---|---|",
    ]
    for number, item in items.items():
        lines.append(f"| {number} | {item['title']} | {_cell(item['conclusion'])} |")
    lines.append("")

    # 1
    item = items[1]
    lines += [f"## 1. {item['title']}", "", "| Thông tin | Giá trị | Nguồn |", "|---|---|---|"]
    for label, entry in item["summary"].items():
        lines.append(f"| {label} | {_cell(entry['value'])} | {_cell(entry['source'])} |")
    lines.append("")
    _findings(lines, item["findings"])

    # 2
    item = items[2]
    lines += [f"## 2. {item['title']}", ""]
    charter = item["summary"].get("charter_capital")
    if charter:
        lines.append(f"Vốn điều lệ: **{fmt_money(charter)}**\n")
    lines += [
        "| Thành viên/cổ đông | Loại | Vốn góp | Tỷ lệ | Ngày nhận vốn góp | Hình thức | Nguồn |",
        "|---|---|---|---|---|---|---|",
    ]
    for member in item["summary"].get("members", []):
        pct = member["ownership_pct"]
        pct_text = "—" if pct is None else f"{pct}%" + (" (tính toán)" if member["pct_source"] == "tính toán" else "")
        lines.append(
            f"| {_cell(member['name'])} | {'Pháp nhân' if member['member_type'] == 'phap_nhan' else 'Cá nhân'} | "
            f"{fmt_money(member['capital_amount'])} | {pct_text} | {_cell(member['acquisition_date'])} | "
            f"{_cell(ACQUISITION_LABELS.get(member['acquisition_method'], member['acquisition_method']))} | {_cell(member['source'])} |"
        )
    lines.append("")
    _findings(lines, item["findings"])

    # 3
    item = items[3]
    lines += [f"## 3. {item['title']}", "", "| Người đại diện | Chức danh | Ngày bổ nhiệm | Nguồn |", "|---|---|---|---|"]
    for rep in item["summary"].get("legal_representatives", []):
        lines.append(f"| {_cell(rep.get('name'))} | {_cell(rep.get('title'))} | "
                     f"{_cell(rep.get('appointed_date'))} | {_cell(rep.get('source'))} |")
    operator = item["summary"].get("actual_operator")
    lines.append("")
    lines.append(f"Người điều hành thực tế: **{operator['name']}** ({operator.get('role') or '—'}) — {operator['source']}\n"
                 if operator else "Người điều hành thực tế: chưa có thông tin\n")
    _findings(lines, item["findings"])

    # 4
    item = items[4]
    summary = item["summary"]
    lines += [f"## 4. {item['title']}", ""]
    tables = (
        ("Nợ thuế", "tax_debts", ("authority", "tax_type", "amount", "overdue_days", "as_of", "status")),
        ("Nợ BHXH", "social_insurance_debts", ("agency", "amount", "months_overdue", "as_of", "status")),
        ("Xử phạt vi phạm hành chính", "administrative_penalties",
         ("decision_no", "date", "domain", "violation", "amount", "remediation_status")),
        ("Cưỡng chế", "enforcement_actions", ("decision_no", "date", "measure", "amount")),
        ("Tranh chấp / kiện tụng", "disputes", ("case_no", "forum", "role", "counterparty", "subject", "claim_amount", "status")),
    )
    for title, key, columns in tables:
        rows = summary.get(key) or []
        lines.append(f"**{title}:** " + ("không ghi nhận" if not rows else ""))
        if rows:
            header = [COLUMN_LABELS.get(c, c) for c in columns] + ["Nguồn"]
            lines += ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            for row in rows:
                cells = [fmt_money(row.get(c)) if c in ("amount", "claim_amount") else _cell(row.get(c)) for c in columns]
                lines.append("| " + " | ".join(cells + [_cell(row.get("source"))]) + " |")
        lines.append("")
    _findings(lines, item["findings"])

    # Documents
    lines += ["## Hồ sơ đã xử lý", "", "| File | Loại | Nguồn | Pass đã chạy | Lỗi |", "|---|---|---|---|---|"]
    for doc in result["documents"]:
        errors = "; ".join(f"{k}: {v}" for k, v in doc["errors"].items())
        lines.append(f"| {_cell(doc['filename'])} | {_cell(doc['label'])} | {SOURCE_LABELS.get(doc['source'], doc['source'])} | "
                     f"{', '.join(doc['passes_done']) or '—'} | {_cell(errors)} |")
    lines += [
        "",
        "> Kiểm tra AML/BL/WL chỉ đối chiếu với danh sách cảnh báo được truyền vào (`watchlist`). "
        "Phát hiện hồ sơ bị chỉnh sửa (Fraud Detection) nằm ngoài phạm vi agent này.",
        "",
    ]
    return "\n".join(lines)
