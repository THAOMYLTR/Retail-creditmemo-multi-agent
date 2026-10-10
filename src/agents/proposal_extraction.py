"""LLM extraction of the retail loan application into structured JSON.

Runs once per document whose matrix type is flagged ``proposal_extraction``
(see src/matrix/document_matrix.yaml and ``is_proposal_type``) — the individual
borrower's "Giấy đề nghị vay vốn kiêm phương án sử dụng vốn và phương án trả
nợ". It carries every number the affordability and loan-sizing calculators
start from: the amount, tenor, rate and repayment method asked for; what the
purchase costs and how much the borrower puts in; the incomes and living costs
the household declares; the debts it already services; and the collateral
offered. Left as raw OCR those are exactly the figures an LLM is most tempted
to invent.

The form mixes money units between blocks — a house price in tỷ đồng next to a
salary in triệu đồng is the normal case — so every amount is normalised to đồng
on the way out, block by block from its own ``source_unit``, with the unit it
was read in kept alongside so a number can be traced back to the page.
"""

from __future__ import annotations

from typing import Any

from src.agents.structured_extraction import (
    build_extraction_chain,
    resolve_money_multiplier,
    run_extraction,
)

REQUIRED_TOP_LEVEL_KEYS = {
    "borrower",
    "loan_request",
    "purpose_financing",
    "declared_income",
    "declared_expenses",
    "existing_debts",
    "collateral",
}

# Scalar money fields per block. List-valued blocks are handled separately in
# normalize_amounts because their amounts live one level down.
_AMOUNT_FIELDS: dict[str, tuple[str, ...]] = {
    "loan_request": ("amount",),
    "purpose_financing": ("total_cost", "own_contribution", "loan_amount"),
    "declared_expenses": ("monthly_living_expenses", "other_monthly_expenses"),
    "repayment_plan": ("monthly_payment_declared",),
    "collateral": ("total_value",),
}
# (block, list field, amount keys) for blocks that hold rows.
_LIST_AMOUNT_FIELDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("declared_income", "items", ("monthly_amount",)),
    ("existing_debts", "items", ("outstanding", "monthly_payment", "limit")),
    ("collateral", "items", ("value",)),
    ("co_borrowers", "items", ("monthly_income",)),
)

PROPOSAL_EXTRACTION_SYSTEM_PROMPT = """
Bạn trích xuất GIẤY ĐỀ NGHỊ VAY VỐN KIÊM PHƯƠNG ÁN SỬ DỤNG VỐN VÀ PHƯƠNG ÁN
TRẢ NỢ của khách hàng CÁ NHÂN thành JSON.

QUY TẮC ĐƠN VỊ TIỀN — quan trọng nhất:
Biểu mẫu thường dùng nhiều đơn vị khác nhau giữa các mục: giá nhà/xe và số
tiền vay hay ghi bằng TRIỆU hoặc TỶ ĐỒNG, thu nhập và chi phí hằng tháng ghi
bằng TRIỆU ĐỒNG hoặc ĐỒNG.
- KHÔNG tự quy đổi. Ghi con số ĐÚNG NHƯ IN trên giấy: "1,5 tỷ đồng" thì trả 1.5
  với source_unit "ty dong"; "25 triệu đồng" thì trả 25 với source_unit
  "trieu dong". Hệ thống tự nhân theo "source_unit" — bạn quy đổi thêm lần nữa
  là số sai gấp triệu lần.
- "source_unit" của TỪNG khối ghi đơn vị in ngay tại mục đó:
  "dong" | "nghin dong" | "trieu dong" | "ty dong". Mục không ghi đơn vị thì
  để "dong". Trong một khối, mọi số phải cùng một đơn vị; nếu trên giấy một
  dòng khác đơn vị, quy dòng đó về đơn vị của khối theo đúng tỷ lệ in trên giấy
  và ghi chú vào extraction_notes.

QUY TẮC CHUNG:
- Chỉ trích thứ có trên giấy. Ô trống thì để null hoặc '', KHÔNG suy diễn,
  KHÔNG lấy số từ mục khác sang.
- Lãi suất ghi theo %/NĂM. Giấy ghi lãi theo tháng (vd 0,8%/tháng) thì ghi
  nguyên số đó vào "interest_rate_as_printed" và để "interest_rate_pct_year"
  là null — KHÔNG tự nhân 12.
- Thời hạn vay quy về SỐ THÁNG khi giấy ghi rõ ("5 năm" -> 60). Giấy không ghi
  thì null.
- "repayment_method": chọn "goc_deu_lai_giam_dan" (trả gốc đều, lãi tính trên
  dư nợ giảm dần), "tra_gop_deu" (gốc + lãi trả đều hằng tháng/annuity),
  "goc_cuoi_ky" (trả gốc một lần cuối kỳ), hoặc "khac". Không rõ thì "".
- Người phụ thuộc là số người khách hàng nuôi dưỡng (con nhỏ, cha mẹ không có
  thu nhập...) — chỉ ghi khi giấy nêu.
- Liệt kê HẾT các khoản nợ hiện hữu và tài sản bảo đảm, không rút gọn.
- "page" là số trang in ở chân trang hoặc mốc "--- Page N ---", null nếu
  không rõ.

Trả về CHÍNH XÁC JSON theo schema sau, không thêm text nào khác:
{{
  "borrower": {{
    "name": "...", "date_of_birth": "dd/mm/yyyy hoặc ''", "id_number": "...",
    "marital_status": "độc thân | đã kết hôn | ly hôn | góa | ''",
    "dependents": <số người phụ thuộc, hoặc null>,
    "household_size": <tổng số người trong hộ, hoặc null>,
    "occupation": "nghề nghiệp/chức vụ, hoặc ''",
    "employer": "nơi công tác/tên hộ kinh doanh, hoặc ''",
    "years_in_job": <số năm công tác/kinh doanh, hoặc null>,
    "residence_address": "...", "residence_status": "nhà sở hữu | ở cùng gia đình | nhà thuê | khác | ''",
    "page": <số nguyên hoặc null>
  }},
  "co_borrowers": {{
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "items": [
      {{"name": "...", "relationship": "vd: vợ, chồng, người bảo lãnh",
        "date_of_birth": "dd/mm/yyyy hoặc ''", "occupation": "...",
        "monthly_income": <số như in trên giấy, hoặc null>}}
    ]
  }},
  "loan_request": {{
    "purpose": "mục đích vay như ghi trên giấy",
    "purpose_category": "mua_nha | xay_sua_nha | mua_o_to | tieu_dung | kinh_doanh | khac | ''",
    "product": "tên sản phẩm vay nếu giấy ghi, hoặc ''",
    "amount": <số tiền đề nghị vay, số như in trên giấy, hoặc null>,
    "tenor_months": <số tháng, hoặc null>,
    "interest_rate_pct_year": <lãi suất %/năm, hoặc null>,
    "interest_rate_as_printed": "nguyên văn lãi suất trên giấy, hoặc ''",
    "repayment_method": "goc_deu_lai_giam_dan | tra_gop_deu | goc_cuoi_ky | khac | ''",
    "grace_months": <số tháng ân hạn gốc, hoặc null>,
    "disbursement_method": "vd: chuyển khoản cho bên bán, hoặc ''",
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "page": <số nguyên hoặc null>
  }},
  "purpose_financing": {{
    "total_cost": <tổng nhu cầu vốn/giá mua nhà, xe..., số như in trên giấy, hoặc null>,
    "own_contribution": <vốn tự có tham gia, hoặc null>,
    "loan_amount": <vốn vay trong cơ cấu nguồn vốn, hoặc null>,
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "page": <số nguyên hoặc null>
  }},
  "declared_income": {{
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "items": [
      {{"earner": "người có thu nhập (khách hàng, vợ/chồng...)",
        "source_type": "luong | huu_tri | cho_thue | kinh_doanh | khac",
        "description": "vd: Lương tại Công ty X",
        "monthly_amount": <số như in trên giấy, hoặc null>}}
    ],
    "page": <số nguyên hoặc null>
  }},
  "declared_expenses": {{
    "monthly_living_expenses": <chi phí sinh hoạt hằng tháng của hộ, hoặc null>,
    "other_monthly_expenses": <chi phí khác hằng tháng (học phí, thuê nhà...), hoặc null>,
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "page": <số nguyên hoặc null>
  }},
  "existing_debts": {{
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "items": [
      {{"lender": "tên TCTD/công ty tài chính", "product": "vd: vay mua nhà, thẻ tín dụng",
        "limit": <hạn mức nếu là thẻ/hạn mức, hoặc null>,
        "outstanding": <dư nợ, hoặc null>, "monthly_payment": <số tiền trả hằng tháng, hoặc null>,
        "maturity": "ngày/tháng đáo hạn, hoặc ''"}}
    ],
    "page": <số nguyên hoặc null>
  }},
  "repayment_plan": {{
    "sources": "nguồn trả nợ như ghi trên giấy, hoặc ''",
    "monthly_payment_declared": <số tiền dự kiến trả hằng tháng nếu giấy nêu, hoặc null>,
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "page": <số nguyên hoặc null>
  }},
  "collateral": {{
    "items": [
      {{"type": "bat_dong_san | o_to | giay_to_co_gia | khac",
        "description": "mô tả tài sản (địa chỉ thửa đất, loại xe, số sổ...)",
        "owner": "chủ sở hữu, hoặc ''",
        "value": <giá trị định giá/tạm tính, số như in trên giấy, hoặc null>,
        "valuation_basis": "vd: chứng thư định giá số..., giá hợp đồng mua bán, KH tự kê khai, hoặc ''",
        "is_purchase_asset": <true nếu chính tài sản hình thành từ vốn vay, false nếu không, null nếu không rõ>}}
    ],
    "total_value": <tổng giá trị TSBĐ nếu giấy ghi, hoặc null>,
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "page": <số nguyên hoặc null>
  }},
  "extraction_notes": ["ghi chú về mục thiếu, không chắc chắn, hoặc OCR kém"]
}}
"""

REPAYMENT_METHODS = ("goc_deu_lai_giam_dan", "tra_gop_deu", "goc_cuoi_ky", "khac")
# Highest %/năm treated as a real rate. Consumer-finance pricing can run high,
# but nothing above this is a yearly rate somebody actually wrote down.
MAX_PLAUSIBLE_RATE = 60


def _scale(value: Any, multiplier: int) -> Any:
    """Scale one figure to đồng and round; leave anything else alone."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(value * multiplier, 2)


def normalize_amounts(result: dict[str, Any]) -> dict[str, Any]:
    """Convert every amount to đồng using each block's own source_unit, in place.

    The prompt asks for the figure exactly as printed plus the unit the page
    states; the conversion is here and nowhere else. Blocks are scaled
    independently because the form genuinely mixes units between them: the
    house price in tỷ, the salary in triệu.
    """

    for block_name, fields in _AMOUNT_FIELDS.items():
        block = result.get(block_name)
        if not isinstance(block, dict):
            continue
        multiplier = resolve_money_multiplier(block.get("source_unit"))
        for field_name in fields:
            block[field_name] = _scale(block.get(field_name), multiplier)

    for block_name, list_field, keys in _LIST_AMOUNT_FIELDS:
        block = result.get(block_name)
        if not isinstance(block, dict):
            continue
        multiplier = resolve_money_multiplier(block.get("source_unit"))
        for item in block.get(list_field) or []:
            if not isinstance(item, dict):
                continue
            for key in keys:
                item[key] = _scale(item.get(key), multiplier)
    return result


def normalize_terms(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the loan terms on the scale the calculators expect, and say so.

    Two mistakes are common enough to guard: a rate typed per month into the
    per-year field (0.8 meaning 0.8%/tháng, which would price a mortgage at
    under 1% a year), and a tenor written in years (5 meaning 5 năm). Both are
    left alone unless they are implausible, and every change is written into
    extraction_notes rather than made quietly.
    """

    request = result.get("loan_request")
    if not isinstance(request, dict):
        return result
    notes = result.setdefault("extraction_notes", [])
    if not isinstance(notes, list):
        notes = result["extraction_notes"] = []

    rate = request.get("interest_rate_pct_year")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        if 0 < rate < 1 and rate * 100 <= MAX_PLAUSIBLE_RATE:
            # Below 1 it can only be a fraction (0.105 meaning 10,5%) — a
            # per-year rate under 1% does not exist in retail lending.
            request["interest_rate_pct_year"] = round(rate * 100, 4)
            notes.append(
                f"loan_request.interest_rate_pct_year: đọc được {rate} — hiểu là "
                f"tỷ lệ thập phân và quy về {rate * 100:g}%/năm."
            )
        elif 0 < rate < 1 or rate > MAX_PLAUSIBLE_RATE:
            # 0.8 is most likely 0.8%/tháng typed into the yearly field: as a
            # fraction it would be 80%/năm, as a yearly rate 0.8%. Neither is
            # usable, so the calculator falls back to its stress rate.
            request["interest_rate_pct_year"] = None
            notes.append(
                f"loan_request.interest_rate_pct_year: {rate} không phải lãi suất "
                "%/năm hợp lệ — đã bỏ, hệ thống dùng lãi suất giả định."
            )

    tenor = request.get("tenor_months")
    if isinstance(tenor, (int, float)) and not isinstance(tenor, bool):
        if tenor <= 0 or tenor > 420:
            request["tenor_months"] = None
            notes.append(
                f"loan_request.tenor_months: {tenor} tháng ngoài khoảng hợp lý — đã bỏ."
            )

    method = str(request.get("repayment_method") or "").strip()
    if method and method not in REPAYMENT_METHODS:
        request["repayment_method"] = "khac"
    return result


def build_proposal_extraction_chain(llm: Any):
    """Build the JSON-output extraction chain for the loan application form."""

    return build_extraction_chain(PROPOSAL_EXTRACTION_SYSTEM_PROMPT, llm)


def extract_proposal_structured_data(
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
        "No proposal extraction LLM configured.",
    )
    if result is None:
        return None, error
    return normalize_terms(normalize_amounts(result)), ""
