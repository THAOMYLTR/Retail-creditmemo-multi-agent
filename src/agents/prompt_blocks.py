"""The data blocks that go into a specialist's prompt, lifted out of Supervisor.

The functions here take documents and return a piece of text. None reads
Supervisor state and none calls back into it, which is why this group could live
apart while the rest of supervisor.py could not: the graph nodes, the routing and
the extraction dispatch all share ``self.config``, the extraction chains and
``self.guardrails``.

The routing tables (``INCOME_JSON_AGENTS``, ``CIC_S10A_JSON_AGENTS``, …) stay on
Supervisor, because they decide which agent reads what; ``_build_user_input``
passes the relevant slice in as an argument. Only the constants these blocks
alone use live here.

Supervisor rebinds each function under its own name, so every call site —
``self._build_...``, ``Supervisor._build_...`` — works the same way.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.affordability_calculator import (
    AffordabilityResult,
    Row,
    build_affordability,
)
from src.agents.cic_s10a_extraction import merge_debt_series
from src.agents.loan_sizing_calculator import build_loan_sizing_table
from src.matrix.document_matrix import get_type
from src.types import ClassifiedDocument
from src.utils.charts import build_linechart_block, pick_unit
from src.utils.formatting import DISPLAY_UNIT, format_million, format_vn_number
from src.utils.source_list import build_source_lines

# The deterministic repayment-capacity table goes to the agents that reason
# about the figures: income analysis is built on it, risk judges DSR and the
# surplus against policy, and credit proposal sizes the loan from it.
AFFORDABILITY_BLOCK_AGENTS = (
    "INCOME_ANALYSIS_AGENT",
    "RISK_ASSESSMENT_AGENT",
    "CREDIT_PROPOSAL_AGENT",
)
AFFORDABILITY_BLOCK_HEADING = "[BẢNG TÍNH KHẢ NĂNG TRẢ NỢ]"

# Who gets the maximum-loan table. Credit proposal because the table *is* its
# report; risk because LTV and the binding limit are what it judges collateral
# coverage and policy breaches against — on a single risk route there is no
# proposal output for it to read them from.
LOAN_SIZING_BLOCK_AGENTS = ("CREDIT_PROPOSAL_AGENT", "RISK_ASSESSMENT_AGENT")
LOAN_SIZING_BLOCK_HEADING = "[BẢNG TÍNH SỐ TIỀN VAY TỐI ĐA]"

# Headings of the structured blocks. Named constants because two things have to
# agree on the exact string: the block itself, and the pointer that replaces the
# document's raw OCR.
INCOME_BLOCK_HEADING = "[DỮ LIỆU THU NHẬP ĐÃ TRÍCH XUẤT]"
PROPOSAL_BLOCK_HEADING = "[DỮ LIỆU GIẤY ĐỀ NGHỊ VAY VỐN]"
CIC_S10A_BLOCK_HEADING = "[EXTRACTED CIC CREDIT REPORT]"
CIC_R21_BLOCK_HEADING = "[EXTRACTED CIC R21 REPORT]"
SITEVISIT_BLOCK_HEADING = "[EXTRACTED APPRAISAL REPORT]"
SOURCE_LIST_BLOCK_HEADING = "[DANH SÁCH NGUỒN — CHÉP NGUYÊN VĂN]"

# The debt/income chart inserted into the credit-relationship section.
DEBT_CHART_TITLE = "Diễn biến dư nợ và thu nhập hằng tháng"
# Used when no proof-of-income document gave a monthly series, so the chart
# carries the debt column alone. A title naming a series the chart does not
# draw reads as missing data rather than data that was never claimed.
DEBT_CHART_TITLE_DEBT_ONLY = "Diễn biến dư nợ 12 tháng gần nhất"
DEBT_CHART_COLUMNS = ("Tổng dư nợ (CIC)", "Thu nhập có chứng từ")


def _loan_program(documents: list[ClassifiedDocument]) -> str:
    """The product the documents were classified under ("" when none named)."""

    return next((doc.loan_program for doc in documents if doc.loan_program), "")


def _document_block_header(
    doc: ClassifiedDocument,
    target_agent: str,
) -> str:
    """The metadata lines that precede a document's content in the prompt.

    Split out from block assembly so _build_user_input can measure the
    overhead before dividing the remaining characters between documents.
    """

    level = doc.agent_relevance.get(target_agent)
    if level == "R":
        relevance = "Relevance to this agent: required evidence"
    elif level == "O":
        relevance = "Relevance to this agent: optional supporting evidence"
    else:
        relevance = (
            "Relevance to this agent: general context — this document "
            "matched no known document type, so it is shared with every "
            "agent. Use it only if it is genuinely relevant."
        )
    matched_type = get_type(doc.document_type)
    return "\n".join(
        [
            f"Document filename: {doc.filename}",
            # Naming the document *type* tells the agent what it is holding
            # (a payslip reads very differently from a bank statement); the
            # agent list alone would not.
            "Document type: "
            + (
                f"{doc.document_type} — {matched_type.label}"
                if matched_type
                else "không xác định"
            ),
            relevance,
            f"Classification reason: {doc.reasoning}",
            f"Extraction status: {doc.extraction_status}",
            f"Extraction error: {doc.extraction_error}",
            "",
        ]
    )


def _build_source_list_block(
    documents: list[ClassifiedDocument],
) -> str:
    """The finished "Nguồn dữ liệu" list, for the agent to copy verbatim.

    Computed rather than described: collapsing a file list is arithmetic, and
    arithmetic asked of a model comes back wrong quietly. Built from the
    documents that survived extraction, which is the set the report is
    actually written from.
    """

    lines = build_source_lines(
        [(doc.filename, doc.document_type, doc.description) for doc in documents]
    )
    if not lines:
        return ""
    return "\n".join(
        [
            SOURCE_LIST_BLOCK_HEADING,
            "Danh sách dưới đã được hệ thống lập sẵn từ đúng những tài liệu "
            "bạn đang đọc. CHÉP NGUYÊN VĂN vào ô \"Nguồn dữ liệu\" của bảng "
            "Thông tin chung, giữ nguyên cả thẻ <em>. TUYỆT ĐỐI không gom "
            "thêm, không rút gọn thêm, không bỏ dòng nào, không đổi thứ tự, "
            "không bỏ đuôi tệp.",
            # A markdown table cell cannot hold a real list; joined with <br>
            # the same lines come out one per row with <em> intact.
            "Cả danh sách nằm trong MỘT ô, các dòng nối với nhau bằng <br> "
            "(không xuống dòng thật, vì xuống dòng sẽ phá vỡ bảng). Dạng đúng: "
            "- dòng 1<br>- dòng 2<br>- dòng 3",
            "",
            *(f"- {line}" for line in lines),
        ]
    )


def _format_row_value(value: float | None, unit: str) -> str:
    """One value cell of a calculator table, on the memo's display scale."""

    if value is None:
        return ""
    if unit == "VNĐ":
        return format_million(value)
    if unit == "%":
        return format_vn_number(value * 100, 1) + "%"
    if unit == "%/năm":
        return format_vn_number(value, 2) + "%"
    if unit == "lần":
        return format_vn_number(value, 2)
    return format_vn_number(value, 0 if float(value).is_integer() else 1)


def _format_rows(rows: list[Row]) -> list[str]:
    lines = [
        "| Chỉ tiêu | Giá trị | Đơn vị | Ngưỡng chính sách | Đánh giá | Nguồn | Ghi chú |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        unit = DISPLAY_UNIT if row.unit == "VNĐ" else row.unit
        limit = _format_row_value(row.limit, row.unit) if row.limit is not None else ""
        note = row.note.replace("|", "/")
        lines.append(
            f"| {row.label} | {_format_row_value(row.value, row.unit)} | {unit} "
            f"| {limit} | {row.status} | {row.source} | {note} |"
        )
    return lines


def _affordability(documents: list[ClassifiedDocument]) -> AffordabilityResult | None:
    usable = [doc for doc in documents if doc.extraction_status == "success"]
    if not usable:
        return None
    return build_affordability(usable, _loan_program(usable) or None)


def _build_affordability_block(
    documents: list[ClassifiedDocument],
    target_agent: str,
) -> str:
    """Deterministic repayment-capacity table for the agents that reason on it.

    Built from every successfully-extracted document rather than the target
    agent's routed subset: income, obligations and DSR are derived facts about
    the borrower, and the CIC report the obligations come from is not routed to
    every one of these agents.
    """

    if target_agent not in AFFORDABILITY_BLOCK_AGENTS:
        return ""
    try:
        result = _affordability(documents)
    except Exception as exc:
        return f"{AFFORDABILITY_BLOCK_HEADING} unavailable: {type(exc).__name__}: {exc}"
    if result is None or not result.rows:
        return ""
    lines = [
        AFFORDABILITY_BLOCK_HEADING,
        "Bảng dưới đã được hệ thống TÍNH SẴN bằng công thức cố định từ dữ liệu "
        "đã trích xuất. Dùng thẳng các con số này, TUYỆT ĐỐI không tự tính lại "
        "từ số liệu thô.",
        f"Sản phẩm áp dụng ngưỡng chính sách: {result.product_label}.",
        f"ĐƠN VỊ: các dòng tiền ghi bằng **{DISPLAY_UNIT}**; dòng %, tháng, "
        "tuổi, lần giữ nguyên đơn vị của nó.",
        "CỘT \"Nguồn\" BẮT BUỘC phải nêu lại khi diễn giải: \"chính sách\" nghĩa "
        "là hồ sơ KHÔNG nêu và hệ thống dùng tham số chính sách — không được "
        "trình bày như số liệu của khách hàng. \"tính toán\" là suy ra từ các "
        "dòng khác trong chính bảng này. \"đề nghị vay\" là số khách hàng tự kê "
        "khai, chưa phải số đã chứng minh.",
        "",
        *_format_rows(result.rows),
    ]
    if result.warnings:
        lines.append("")
        lines.extend(f"CẢNH BÁO: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _build_loan_sizing_block(
    documents: list[ClassifiedDocument],
    target_agent: str,
) -> str:
    """Render the maximum-loan table for the credit proposal prompt."""

    if target_agent not in LOAN_SIZING_BLOCK_AGENTS:
        return ""
    try:
        affordability = _affordability(documents)
        if affordability is None:
            return ""
        usable = [doc for doc in documents if doc.extraction_status == "success"]
        table = build_loan_sizing_table(
            usable, affordability, _loan_program(usable) or None
        )
    except Exception as exc:
        return f"{LOAN_SIZING_BLOCK_HEADING} unavailable: {type(exc).__name__}: {exc}"
    if not table.rows:
        return ""
    lines = [
        LOAN_SIZING_BLOCK_HEADING,
        "Bảng dưới đã được hệ thống TÍNH SẴN: số tiền vay tối đa theo từng giới "
        "hạn (khả năng trả nợ/DSR, LTV, vốn tự có, trần sản phẩm, số tiền đề "
        "nghị) và giới hạn nào đang CHẶN. Dùng thẳng các con số này, TUYỆT ĐỐI "
        "không tự tính lại.",
        f"Sản phẩm áp dụng ngưỡng chính sách: {table.product_label}.",
        f"ĐƠN VỊ: các dòng tiền ghi bằng **{DISPLAY_UNIT}**.",
        "CỘT \"Nguồn\" bắt buộc nêu lại khi diễn giải, như ở "
        f"{AFFORDABILITY_BLOCK_HEADING}.",
        "",
        *_format_rows(table.rows),
    ]
    if table.warnings:
        lines.append("")
        lines.extend(f"CẢNH BÁO: {warning}" for warning in table.warnings)
    return "\n".join(lines)


def _build_income_structured_block(
    selected: list[ClassifiedDocument],
) -> str:
    """Render the extracted proof-of-income records."""

    income_docs = [
        doc for doc in selected if doc.is_income and doc.income_extraction
    ]
    if not income_docs:
        return ""
    parts = [
        INCOME_BLOCK_HEADING,
        "Trích xuất từ các tài liệu chứng minh thu nhập (hợp đồng lao động, "
        "xác nhận lương, sao kê tài khoản, thuế TNCN, hợp đồng cho thuê, sổ "
        "sách hộ kinh doanh...).",
        "ĐƠN VỊ: mọi số tiền đã quy về ĐỒNG (VNĐ). Trường \"source_unit\" là "
        "đơn vị ghi trên bản gốc, chỉ để đối chiếu — không nhân/chia lại.",
        "\"contract_terms\" là mức THỎA THUẬN trên hợp đồng, chưa phải tiền đã "
        "nhận. \"recurring_debits\" là các khoản trả nợ định kỳ thấy trên sao "
        "kê — đối chiếu với nghĩa vụ trả nợ khách hàng kê khai.",
    ]
    for doc in income_docs:
        parts.append(
            f"--- {doc.filename} ---\n"
            + json.dumps(doc.income_extraction, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(parts)


def _build_proposal_structured_block(
    selected: list[ClassifiedDocument],
) -> str:
    """Render the extracted loan application records."""

    proposal_docs = [
        doc for doc in selected if doc.is_proposal and doc.proposal_extraction
    ]
    if not proposal_docs:
        return ""
    parts = [
        PROPOSAL_BLOCK_HEADING,
        "Trích xuất từ Giấy đề nghị vay vốn kiêm phương án trả nợ: thông tin "
        "người vay, mục đích và cơ cấu nguồn vốn, số tiền/thời hạn/lãi suất/"
        "phương thức trả nợ đề nghị, thu nhập và chi phí tự kê khai, các khoản "
        "nợ hiện hữu, tài sản bảo đảm.",
        "Mọi số tiền đã quy về ĐỒNG (VNĐ). Trường \"source_unit\" là đơn vị "
        "ghi trên bản gốc, chỉ để đối chiếu — không nhân/chia lại lần nữa.",
        "Thu nhập, chi phí và nợ trong khối này là KHÁCH HÀNG TỰ KÊ KHAI — "
        "phải đối chiếu với chứng từ, không được coi là số đã chứng minh.",
    ]
    for doc in proposal_docs:
        parts.append(
            f"--- {doc.filename} ---\n"
            + json.dumps(doc.proposal_extraction, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(parts)


def _build_debt_chart_block(
    documents: list[ClassifiedDocument],
    sub_agent_outputs: dict[str, str],
) -> tuple[str, str]:
    """Build the ```linechart block from extracted CIC and income data.

    Returns ``(block, title)``, or ``("", "")`` when there is no chart to draw.
    The title travels with the block because it varies with the data (see
    DEBT_CHART_TITLE_DEBT_ONLY) and _insert_debt_chart needs the same string
    for its fallback heading.

    Both series are written here from extracted JSON rather than by an agent:
    a model asked to retype two dozen figures into a chart is a model given two
    dozen chances to invent one. The income line is the evidenced monthly income
    the affordability calculator assembled — every source summed, before the
    recognition discount — so it is exactly what the payslips and statements
    show.
    """

    # This chart is the credit-relationship section's content, and every caller
    # hands over the *unfiltered* document list — so without this gate any run
    # that merely had a CIC file in the folder grew a debt chart, landing at the
    # end of reports that have no credit-relationship heading.
    if "CREDIT_RELATIONSHIP_AGENT" not in sub_agent_outputs:
        return "", ""

    series = merge_debt_series(
        (doc.filename, doc.cic_s10a_extraction)
        for doc in documents
        if doc.is_cic_s10a and doc.cic_s10a_extraction
    )
    # One point is a dot, not a trend.
    if len(series) < 2:
        return "", ""

    months = [row["thang"] for row in series]
    debt = [row["du_no"] for row in series]

    try:
        affordability = _affordability(documents)
    except Exception:
        affordability = None
    income_series = affordability.monthly_income_series if affordability else {}
    income = [income_series.get(month) for month in months]
    has_income = any(value is not None for value in income)

    values_for_unit = list(debt)
    if has_income:
        values_for_unit += [value for value in income if value is not None]
    divisor, unit = pick_unit(values_for_unit)

    columns = [DEBT_CHART_COLUMNS[0]]
    series_values = [[value / divisor for value in debt]]
    if has_income:
        columns.append(DEBT_CHART_COLUMNS[1])
        series_values.append(
            [None if value is None else value / divisor for value in income]
        )

    title = DEBT_CHART_TITLE if has_income else DEBT_CHART_TITLE_DEBT_ONLY
    return (
        build_linechart_block(
            title=title,
            unit=unit,
            columns=columns,
            labels=months,
            series=series_values,
        ),
        title,
    )


def _build_cic_s10a_structured_block(
    selected: list[ClassifiedDocument],
) -> str:
    """Render the extracted CIC credit report records for the prompt.

    Units are spelled out because the source report uses two at once and the
    extraction only rescales one of them: VND figures were printed in triệu
    đồng and are now in đồng, while foreign-currency figures are unchanged.
    """

    cic_docs = [
        doc for doc in selected if doc.is_cic_s10a and doc.cic_s10a_extraction
    ]
    if not cic_docs:
        return ""
    parts = [
        CIC_S10A_BLOCK_HEADING,
        "Trích xuất từ Báo cáo thông tin tín dụng CIC của khách hàng vay: dư "
        "nợ hiện tại theo từng TCTD, thẻ tín dụng và hạn mức thẻ, diễn biến dư "
        "nợ 12 tháng gần nhất, cam kết ngoại bảng, điểm/xếp hạng tín dụng và "
        "lịch sử cảnh báo.",
        "ĐƠN VỊ: các trường VNĐ (\"vnd\", \"han_muc\", \"du_no\", "
        "\"du_no_vay\", \"du_no_the\", \"tong_du_no\") đã quy về ĐỒNG. Các "
        "trường ngoại tệ (\"ngoai_te\") giữ NGUYÊN TỆ theo bản gốc — không quy "
        "đổi, không cộng với cột VNĐ.",
        "Dư nợ trong \"du_no_12_thang\" ĐÃ bao gồm dư nợ ngoại tệ quy đổi; "
        "không cộng thêm số ngoại tệ ở khối khác vào, sẽ thành tính hai lần.",
        "Giá trị null nghĩa là kỳ đó thiếu số liệu báo cáo — KHÔNG phải bằng 0.",
    ]
    for doc in cic_docs:
        parts.append(
            f"--- {doc.filename} ---\n"
            + json.dumps(doc.cic_s10a_extraction, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(parts)


def _build_cic_r21_structured_block(
    selected: list[ClassifiedDocument],
) -> str:
    """Render the extracted CIC R20/R21 collateral records for the prompt."""

    cic_docs = [
        doc for doc in selected if doc.is_cic_r21 and doc.cic_r21_extraction
    ]
    if not cic_docs:
        return ""
    parts = [
        CIC_R21_BLOCK_HEADING,
        "Trích xuất từ Báo cáo thông tin bảo đảm tiền vay CIC (mã phiếu "
        "R20/R21): danh sách tổ chức tín dụng đang nhận bảo đảm và chi "
        "tiết từng tài sản bảo đảm theo tổ chức tín dụng đó.",
        "ĐƠN VỊ: trường \"gia_tri_trieu_vnd\" đã quy về ĐỒNG dù tên trường "
        "vẫn giữ nguyên (đơn vị gốc trên giấy là triệu đồng).",
        "\"loai_tai_san\" là MÃ SỐ hai chữ số của CIC (vd \"08\"), không "
        "phải nhãn mô tả — báo cáo không kèm bảng chú giải mã, không tự "
        "suy diễn ý nghĩa mã này.",
        "\"ngay_giai_chap\" là null nghĩa là tài sản CHƯA giải chấp (vẫn "
        "đang thế chấp), không phải thiếu dữ liệu.",
        "Một khối có \"mo_ta_tai_san\": \"Không có bảo đảm tiền vay bằng "
        "tài sản\" nghĩa là tổ chức tín dụng đó xác nhận KHÔNG nhận tài "
        "sản bảo đảm nào — đây là thông tin có thật, không phải lỗi.",
    ]
    for doc in cic_docs:
        parts.append(
            f"--- {doc.filename} ---\n"
            + json.dumps(doc.cic_r21_extraction, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(parts)


def _build_sitevisit_structured_block(
    selected: list[ClassifiedDocument],
) -> str:
    """Render the extracted appraisal / verification report for the prompt.

    The warning about ``conclusion`` is the point of writing a header at all.
    Every other block is what the officer checked; this one ends with one
    person's judgement, and an agent that cites it the same way would be telling
    the reader the file records a fact when it records an opinion.
    """

    sitevisit_docs = [
        doc for doc in selected if doc.is_sitevisit and doc.sitevisit_extraction
    ]
    if not sitevisit_docs:
        return ""
    parts = [
        SITEVISIT_BLOCK_HEADING,
        "Trích xuất từ Báo cáo thẩm định/biên bản xác minh thực tế: thông tin "
        "cuộc xác minh, nơi cư trú, công việc và thu nhập, hoạt động kinh "
        "doanh (nếu có), tài sản bảo đảm, và kết luận của cán bộ thẩm định.",
        "ĐƠN VỊ: mọi số tiền đã quy về ĐỒNG. Trường \"source_unit\" ghi đơn vị "
        "gốc in trên báo cáo.",
        "QUAN TRỌNG - khối \"conclusion\" (overall_assessment, risks_noted, "
        "recommendation, conditions) là Ý KIẾN CHỦ QUAN của cán bộ thẩm định, "
        "KHÔNG phải dữ kiện. Được dùng làm tham khảo và phải nói rõ là nhận "
        "định của cán bộ thẩm định khi nhắc tới; TUYỆT ĐỐI không trích dẫn như "
        "dữ kiện đọc từ hồ sơ.",
        "\"verification_method\" cho biết thông tin được xác minh thế nào "
        "(đến tận nơi, gọi điện, chưa xác minh). Thông tin \"chưa xác minh\" "
        "không được trình bày như đã kiểm chứng.",
        "Các khối còn lại dùng để đối chiếu với giấy tờ: chênh lệch giữa "
        "điều cán bộ xác minh được và điều khách hàng kê khai là thông tin "
        "đáng nêu.",
    ]
    for doc in sitevisit_docs:
        parts.append(
            f"--- {doc.filename} ---\n"
            + json.dumps(doc.sitevisit_extraction, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(parts)


def affordability_data(documents: list[ClassifiedDocument]) -> dict[str, Any]:
    """The affordability table as data, for the run log rather than a prompt."""

    try:
        result = _affordability(documents)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return result.as_dict() if result else {}


def loan_sizing_data(documents: list[ClassifiedDocument]) -> dict[str, Any]:
    """The maximum-loan table as data, for the run log rather than a prompt."""

    try:
        affordability = _affordability(documents)
        if affordability is None:
            return {}
        usable = [doc for doc in documents if doc.extraction_status == "success"]
        return build_loan_sizing_table(
            usable, affordability, _loan_program(usable) or None
        ).as_dict()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
