"""LLM extraction of the retail appraisal / field verification report into JSON.

Runs once per document whose matrix type is flagged ``sitevisit_extraction``
(see src/matrix/document_matrix.yaml and ``is_sitevisit_type``). For an
individual borrower this is the relationship officer's "báo cáo thẩm định" or
"biên bản xác minh": the only document in the dossier written *after* someone
went and checked — that the borrower lives where the papers say, works where
the contract says, earns roughly what the payslips say, and that the collateral
exists and is in the state described. Every specialist consumes some part of
it, which is why all five read this block.

Two things make it different from the other passes:

- It is prose, not a form, so the prompt names the facts to find rather than
  section numbers.
- It mixes verification with judgement. The officer's verdict — the risks
  noted, the recommendation — sits in ``conclusion`` and nowhere else, kept
  apart on purpose: it is one person's opinion, and an agent that cites it as
  document evidence would be telling the reader the file says something it
  does not. The block header says so out loud.

Amounts (verified income, business turnover, collateral value) are read as
printed and scaled here from each block's ``source_unit``. The conversion
happens here and only here — a prompt that asks the model to convert *and*
record the unit invites the figure to be scaled twice.
"""

from __future__ import annotations

from typing import Any

from src.agents.structured_extraction import (
    build_extraction_chain,
    resolve_money_multiplier,
    run_extraction,
)

REQUIRED_TOP_LEVEL_KEYS = {
    "survey_info",
    "residence_verification",
    "employment_verification",
    "business_verification",
    "collateral_verification",
    "conclusion",
}

_AMOUNT_FIELDS: dict[str, tuple[str, ...]] = {
    "employment_verification": ("verified_monthly_income",),
    "business_verification": (
        "monthly_revenue",
        "monthly_expenses",
        "monthly_profit",
    ),
    "collateral_verification": ("estimated_value",),
}

SITEVISIT_EXTRACTION_SYSTEM_PROMPT = """
Bạn trích xuất BÁO CÁO THẨM ĐỊNH / BIÊN BẢN XÁC MINH THỰC TẾ khách hàng CÁ NHÂN
vay vốn thành JSON.

Đây là văn bản tự do, không phải biểu mẫu có ô cố định. Thông tin nằm rải rác
trong các đoạn văn — hãy tìm theo NỘI DUNG, đừng tìm theo số thứ tự mục.

QUY TẮC BẮT BUỘC:
- Chỉ ghi thông tin CÓ THẬT trong tài liệu. Không suy đoán, không điền giá trị
  "hợp lý". Không tìm thấy thì để null (hoặc [] với danh sách, '' với chuỗi).
- Tách bạch XÁC MINH và Ý KIẾN. Mọi đánh giá, nhận định, khuyến nghị của cán bộ
  chỉ được đặt trong khối "conclusion". Các khối còn lại chỉ chứa điều cán bộ
  đã KIỂM TRA/QUAN SÁT/ĐỐI CHIẾU được (gặp ai, xem giấy tờ gì, thấy gì).
- "verification_method" ghi CÁCH xác minh như tài liệu nêu: "đến tận nơi",
  "gọi điện cho phòng nhân sự", "đối chiếu sao kê", "chưa xác minh"...
- KHÔNG tự quy đổi đơn vị tiền. Ghi con số ĐÚNG NHƯ IN trên giấy: "25 triệu
  đồng/tháng" thì trả 25 với source_unit "trieu dong". Hệ thống tự nhân theo
  "source_unit" — bạn quy đổi thêm lần nữa là số sai gấp triệu lần.
- "source_unit" của TỪNG khối: "dong" | "nghin dong" | "trieu dong" |
  "ty dong". Không chú thích đơn vị nào thì để "dong".
- Khối "business_verification" chỉ điền khi khách hàng có hoạt động kinh doanh
  (hộ kinh doanh, buôn bán). Không có thì để các trường null/''/[].

Trả về JSON đúng cấu trúc sau, không kèm giải thích:

{{
  "survey_info": {{
    "survey_date": "YYYY-MM-DD hoặc nguyên văn nếu không rõ, null nếu không có",
    "officers": ["họ tên cán bộ thẩm định"],
    "locations_visited": ["địa điểm đã đến"],
    "people_met": ["họ tên - quan hệ/chức vụ người đã gặp"]
  }},
  "residence_verification": {{
    "address": "địa chỉ nơi ở thực tế đã xác minh, hoặc ''",
    "matches_documents": <true nếu khớp giấy tờ cư trú, false nếu không khớp, null nếu không nêu>,
    "ownership": "nhà sở hữu | ở cùng gia đình | nhà thuê | khác | ''",
    "years_at_address": <số năm cư trú, hoặc null>,
    "household_members": <số người trong hộ, hoặc null>,
    "dependents": <số người phụ thuộc, hoặc null>,
    "observations": ["điều quan sát được tại nơi ở"]
  }},
  "employment_verification": {{
    "employer": "đơn vị công tác, hoặc ''",
    "position": "chức vụ, hoặc ''",
    "years_in_job": <số năm công tác, hoặc null>,
    "verification_method": "cách xác minh, hoặc ''",
    "verified_monthly_income": <thu nhập hằng tháng cán bộ xác minh được, số như in trên giấy, hoặc null>,
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "observations": ["điều xác minh được về công việc, thu nhập"]
  }},
  "business_verification": {{
    "business_name": "tên hộ kinh doanh/cửa hàng, hoặc ''",
    "business_line": "ngành hàng, hoặc ''",
    "location": "địa điểm kinh doanh, hoặc ''",
    "years_operating": <số năm kinh doanh, hoặc null>,
    "monthly_revenue": <doanh thu bình quân tháng, số như in trên giấy, hoặc null>,
    "monthly_expenses": <chi phí bình quân tháng, hoặc null>,
    "monthly_profit": <lợi nhuận/thu nhập ròng bình quân tháng, hoặc null>,
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "observations": ["quan sát tại điểm kinh doanh: hàng tồn, lượng khách, sổ sách..."]
  }},
  "collateral_verification": {{
    "description": "tài sản đã xem, hoặc ''",
    "location": "vị trí tài sản, hoặc ''",
    "current_use": "hiện trạng sử dụng (đang ở, cho thuê, đất trống...), hoặc ''",
    "matches_documents": <true | false | null>,
    "estimated_value": <giá trị cán bộ ước tính/ghi nhận, số như in trên giấy, hoặc null>,
    "source_unit": "dong | nghin dong | trieu dong | ty dong",
    "observations": ["tình trạng pháp lý, tranh chấp, quy hoạch, khả năng thanh khoản quan sát được"]
  }},
  "conclusion": {{
    "overall_assessment": "đánh giá chung của cán bộ",
    "risks_noted": ["rủi ro cán bộ ghi nhận"],
    "recommendation": "đề xuất của cán bộ (số tiền, thời hạn, điều kiện nếu có)",
    "conditions": ["điều kiện kèm theo nếu có"]
  }},
  "extraction_notes": ["ghi chú về mục thiếu, không chắc chắn, hoặc OCR kém"]
}}
"""


def _scale(value: Any, multiplier: int) -> Any:
    """Scale to đồng and round, leaving anything that is not a number alone."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    return round(value * multiplier, 2)


def normalize_amounts(result: dict[str, Any]) -> dict[str, Any]:
    """Convert every amount to đồng using each block's own source_unit, in place."""

    for block_name, fields in _AMOUNT_FIELDS.items():
        block = result.get(block_name)
        if not isinstance(block, dict):
            continue
        multiplier = resolve_money_multiplier(block.get("source_unit"))
        for field_name in fields:
            block[field_name] = _scale(block.get(field_name), multiplier)
    return result


def build_sitevisit_extraction_chain(llm: Any):
    """Build the JSON-output extraction chain for the appraisal report."""

    return build_extraction_chain(SITEVISIT_EXTRACTION_SYSTEM_PROMPT, llm)


def extract_sitevisit_structured_data(
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
        "No sitevisit extraction LLM configured.",
    )
    if result is None:
        return None, error
    return normalize_amounts(result), ""
