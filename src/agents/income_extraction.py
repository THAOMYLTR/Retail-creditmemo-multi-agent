"""LLM extraction of proof-of-income documents into a structured monthly record.

Runs once per document whose matrix type is flagged ``income_extraction`` (see
src/matrix/document_matrix.yaml and ``is_income_type``): the labour contract,
the employer's salary confirmation or payslips, the bank statement of the salary
account, the PIT return, a rental contract, household-business books, and
pension or savings papers.

These documents look nothing alike, but the credit question they answer is the
same — *how much money reliably reaches this person each month, from where* —
so they all leave in one shape: a list of month-by-month income lines, plus the
contract terms and account summary when the document has them. The
affordability calculator then reads income without caring which kind of paper
it came from, and the evidence hierarchy (a credit actually received on the
statement beats a figure on a contract) is applied in code, not by the model.

Two more things the statement carries are pulled out because they matter to the
DSR and nothing else in the dossier states them:

- ``recurring_debits``: instalments to other lenders visible on the account.
  A borrower who declares no existing debt but pays 8 triệu a month to a
  finance company is the single most common gap in a retail file.
- ``account_summary``: the running balance, which shows whether income is
  spent the day it arrives.

Money is read exactly as printed and scaled here from ``source_unit`` — the
same arrangement every other pass uses, for the same reason: a model asked to
convert units drops or adds three zeros often enough to matter, and nothing
downstream could tell.
"""

from __future__ import annotations

from typing import Any

from src.agents.cic_s10a_extraction import month_sort_key, normalize_month_label
from src.agents.structured_extraction import (
    build_extraction_chain,
    resolve_money_multiplier,
    run_extraction,
)
from src.utils.common import normalize_text

REQUIRED_TOP_LEVEL_KEYS = {
    "document_kind",
    "income_owner",
    "monthly_income",
    "source_unit",
}

DOCUMENT_KINDS = (
    "hop_dong_lao_dong",
    "xac_nhan_luong",
    "sao_ke_tai_khoan",
    "thue_tncn",
    "hop_dong_cho_thue",
    "ho_kinh_doanh",
    "huu_tri",
    "tien_gui",
    "khac",
)

INCOME_SOURCE_TYPES = ("luong", "huu_tri", "cho_thue", "kinh_doanh", "khac")

INCOME_EXTRACTION_SYSTEM_PROMPT = """
Bạn trích xuất dữ liệu THU NHẬP từ văn bản OCR của một tài liệu chứng minh thu
nhập của khách hàng cá nhân vay vốn tại ngân hàng Việt Nam (khách hàng bán lẻ).
Tài liệu có thể là: hợp đồng lao động, giấy xác nhận lương/bảng lương/phiếu
lương, sao kê tài khoản ngân hàng, chứng từ/tờ khai quyết toán thuế TNCN, hợp
đồng cho thuê tài sản, sổ sách/thông báo thuế của hộ kinh doanh, quyết định
hưởng lương hưu, sổ tiết kiệm/xác nhận số dư.

Câu hỏi tín dụng duy nhất cần trả lời: MỖI THÁNG khách hàng NHẬN ĐƯỢC BAO NHIÊU
TIỀN, TỪ NGUỒN NÀO. Chỉ ghi thứ đọc được trên giấy. Không suy diễn, không bịa
số, không tự tính trung bình.

QUY TẮC ĐƠN VỊ TIỀN:
- GHI ĐÚNG CON SỐ IN TRÊN GIẤY, chỉ bỏ dấu phân cách hàng nghìn. TUYỆT ĐỐI
  KHÔNG tự quy đổi đơn vị — chương trình tự nhân theo "source_unit". Giấy ghi
  "25,5 triệu đồng" thì trả 25.5 và source_unit "trieu dong"; giấy ghi
  "25.500.000" thì trả 25500000 và source_unit "dong".
- "source_unit": "dong" | "nghin dong" | "trieu dong". Không chú thích đơn vị
  nào thì để "dong". Một tài liệu dùng MỘT đơn vị cho toàn bộ số liệu; nếu
  thấy trộn đơn vị, quy về đơn vị chiếm đa số và ghi chú vào extraction_notes.

QUY TẮC "monthly_income" — mỗi dòng là MỘT khoản thu nhập của MỘT tháng:
- Sao kê tài khoản: chỉ lấy các giao dịch GHI CÓ là THU NHẬP (lương, thưởng,
  tiền thuê nhà, lương hưu, doanh thu bán hàng chuyển khoản đều đặn). KHÔNG lấy
  tiền khách tự chuyển từ tài khoản khác của mình, tiền vay giải ngân, tiền
  hoàn trả, tiền chuyển qua lại với người thân. Nhiều giao dịch lương cùng một
  tháng thì cộng lại thành một dòng và ghi rõ trong "note".
- Bảng lương/xác nhận lương: mỗi tháng có số liệu là một dòng. Ghi rõ
  "is_net": true nếu là lương THỰC LĨNH (sau thuế, bảo hiểm), false nếu là
  lương GỘP (trước khấu trừ), null nếu không rõ.
- Hợp đồng lao động/hợp đồng cho thuê: KHÔNG tạo dòng theo tháng — điều khoản
  ghi vào "contract_terms". Đây là mức THỎA THUẬN, chưa phải tiền đã nhận.
- Tờ khai/chứng từ thuế TNCN: ghi thu nhập cả năm vào "annual_income", KHÔNG
  tự chia 12.
- "source_type" của từng dòng: "luong" (tiền lương, thưởng, phụ cấp),
  "huu_tri" (lương hưu, trợ cấp), "cho_thue" (tiền cho thuê tài sản),
  "kinh_doanh" (thu từ hoạt động kinh doanh/hộ kinh doanh), "khac" (cổ tức,
  lãi tiền gửi, nguồn khác).
- "payer": tên đơn vị trả lương/người thuê/nguồn trả nếu đọc được.

QUY TẮC "recurring_debits" (chỉ với sao kê tài khoản):
- Liệt kê các khoản GHI NỢ lặp lại hằng tháng có dấu hiệu là TRẢ NỢ: trả góp,
  thanh toán khoản vay, trích nợ tự động, thanh toán thẻ tín dụng, trả nợ cho
  công ty tài chính. Ghi số tiền mỗi tháng và số tháng thấy xuất hiện.
- Không đưa chi tiêu sinh hoạt thông thường vào đây.

QUY TẮC GHI THÁNG: mọi nhãn tháng ghi dạng "MM/YYYY", ví dụ "03/2026".

"page" là số trang chứa dữ liệu, đọc từ mốc "--- Page N ---"; không xác định
được thì null. Không bịa số trang.

Trả về CHÍNH XÁC JSON theo schema sau, không thêm text nào khác:
{{
  "document_kind": "hop_dong_lao_dong | xac_nhan_luong | sao_ke_tai_khoan | thue_tncn | hop_dong_cho_thue | ho_kinh_doanh | huu_tri | tien_gui | khac",
  "income_owner": {{
    "name": "họ tên người có thu nhập, hoặc ''",
    "id_number": "số CCCD/CMND nếu in trên giấy, hoặc ''"
  }},
  "issuer": "đơn vị phát hành tài liệu (công ty, ngân hàng, cơ quan thuế...), hoặc ''",
  "period": {{"from": "MM/YYYY hoặc ''", "to": "MM/YYYY hoặc ''"}},
  "source_unit": "dong | nghin dong | trieu dong",
  "contract_terms": {{
    "position": "chức danh/công việc, hoặc ''",
    "contract_type": "không xác định thời hạn | xác định thời hạn | thời vụ | cho thuê | '' ",
    "start_date": "dd/mm/yyyy hoặc ''",
    "end_date": "dd/mm/yyyy hoặc '' (không thời hạn thì để '')",
    "monthly_amount": <lương/tiền thuê THỎA THUẬN mỗi tháng, số như in trên giấy, hoặc null>,
    "monthly_allowances": <phụ cấp cố định mỗi tháng, số như in trên giấy, hoặc null>,
    "page": <số nguyên hoặc null>
  }},
  "monthly_income": [
    {{"month": "MM/YYYY", "source_type": "luong | huu_tri | cho_thue | kinh_doanh | khac",
      "payer": "...", "amount": <số như in trên giấy>, "is_net": true | false | null,
      "note": "ghi chú ngắn hoặc ''", "page": <số nguyên hoặc null>}}
  ],
  "annual_income": [
    {{"year": "YYYY", "source_type": "luong | ...", "amount": <tổng thu nhập chịu thuế cả năm, số như in trên giấy>,
      "tax_paid": <thuế TNCN đã khấu trừ/nộp, hoặc null>, "page": <số nguyên hoặc null>}}
  ],
  "account_summary": {{
    "account_number": "số tài khoản (có thể che bớt), hoặc ''",
    "opening_balance": <số như in trên giấy, hoặc null>,
    "closing_balance": <số như in trên giấy, hoặc null>,
    "total_credits": <tổng ghi có cả kỳ, hoặc null>,
    "total_debits": <tổng ghi nợ cả kỳ, hoặc null>,
    "lowest_balance": <số dư thấp nhất thấy được trong kỳ, hoặc null>,
    "page": <số nguyên hoặc null>
  }},
  "recurring_debits": [
    {{"description": "vd: Trả góp khoản vay tại FE Credit", "counterparty": "...",
      "monthly_amount": <số như in trên giấy>, "months_seen": <số tháng thấy xuất hiện>,
      "page": <số nguyên hoặc null>}}
  ],
  "extraction_notes": ["ghi chú về dữ liệu thiếu, không chắc chắn, OCR kém, hoặc giao dịch bất thường"]
}}

Khối không áp dụng cho loại tài liệu này thì để giá trị rỗng (null, [] hoặc
''), KHÔNG bỏ khối đó đi.
"""


def build_income_extraction_chain(llm: Any):
    """Build the JSON-output extraction chain for proof-of-income documents."""

    return build_extraction_chain(INCOME_EXTRACTION_SYSTEM_PROMPT, llm)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _scale(value: Any, multiplier: int) -> Any:
    """Scale one figure to đồng and round; leave anything else alone."""

    number = _num(value)
    if number is None:
        return value
    return round(number * multiplier, 2)


def _normalize_source_type(raw: Any) -> str:
    """Map whatever the model wrote onto one of INCOME_SOURCE_TYPES.

    The calculator recognises each source type at a different share (salary in
    full, business income discounted), so an unrecognised label must not slip
    through as its own category — it lands on "khac", the most conservative.
    """

    text = normalize_text(str(raw or ""))
    if not text:
        return "khac"
    compact = text.replace(" ", "_")
    if compact in INCOME_SOURCE_TYPES:
        return compact
    if any(word in text for word in ("luong", "thuong", "phu cap", "salary", "wage")):
        return "huu_tri" if "huu" in text else "luong"
    if "huu" in text or "pension" in text:
        return "huu_tri"
    if "thue" in text or "rent" in text:
        return "cho_thue"
    if "kinh doanh" in text or "ban hang" in text or "business" in text:
        return "kinh_doanh"
    return "khac"


def normalize_income_record(result: dict[str, Any]) -> dict[str, Any]:
    """Scale every amount to đồng, tidy labels and sort months, in place.

    One ``source_unit`` per document: unlike the loan application, these
    documents do not mix units between tables, and the prompt says to pick the
    majority unit and note any exception.
    """

    multiplier = resolve_money_multiplier(result.get("source_unit"))

    kind = normalize_text(str(result.get("document_kind") or "")).replace(" ", "_")
    result["document_kind"] = kind if kind in DOCUMENT_KINDS else "khac"

    terms = result.get("contract_terms")
    if isinstance(terms, dict):
        for key in ("monthly_amount", "monthly_allowances"):
            terms[key] = _scale(terms.get(key), multiplier)

    rows = []
    for row in result.get("monthly_income") or []:
        if not isinstance(row, dict):
            continue
        row["month"] = normalize_month_label(row.get("month"))
        row["source_type"] = _normalize_source_type(row.get("source_type"))
        row["amount"] = _scale(row.get("amount"), multiplier)
        if _num(row.get("amount")) is None:
            continue
        rows.append(row)
    rows.sort(key=lambda r: month_sort_key(r["month"]))
    result["monthly_income"] = rows

    for row in result.get("annual_income") or []:
        if isinstance(row, dict):
            row["source_type"] = _normalize_source_type(row.get("source_type"))
            row["amount"] = _scale(row.get("amount"), multiplier)
            row["tax_paid"] = _scale(row.get("tax_paid"), multiplier)

    summary = result.get("account_summary")
    if isinstance(summary, dict):
        for key in (
            "opening_balance",
            "closing_balance",
            "total_credits",
            "total_debits",
            "lowest_balance",
        ):
            summary[key] = _scale(summary.get(key), multiplier)

    for row in result.get("recurring_debits") or []:
        if isinstance(row, dict):
            row["monthly_amount"] = _scale(row.get("monthly_amount"), multiplier)

    period = result.get("period")
    if isinstance(period, dict):
        for key in ("from", "to"):
            if period.get(key):
                period[key] = normalize_month_label(period.get(key))
    return result


def extract_income_structured_data(
    chain: Any,
    filename: str,
    content: str,
    # Accepted so all five passes keep one signature with the shared runner.
    path: str = "",
) -> tuple[dict[str, Any] | None, str]:
    """Run the extraction chain and validate its shape. Never raises."""

    result, error = run_extraction(
        chain,
        filename,
        content,
        REQUIRED_TOP_LEVEL_KEYS,
        "No income extraction LLM configured.",
    )
    if result is None:
        return None, error
    return normalize_income_record(result), ""
