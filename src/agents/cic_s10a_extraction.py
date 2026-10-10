"""LLM extraction of the borrower's CIC credit report into structured JSON.

Runs once per document whose matrix type is flagged ``cic_s10a_extraction``
(see src/matrix/document_matrix.yaml and ``is_cic_s10a_type``) — for a retail
file, the CIC credit information report of the individual borrower.

The module keeps its original name after the CIC credit-relationship form code
(S10A) it was first written for. The report layout CIC issues for individuals
carries the same blocks this schema reads — current balances per lender, the
monthly debt trend, card balances, bad-debt and late-payment history, and a
credit score/rating — so one schema serves both; the prompt finds them by
content rather than by section number. The collateral report (R20/R21) shares
nothing but the letterhead and has its own module.

Retail adds one block the corporate form never needed: ``the_tin_dung``, the
credit cards with their limits. A card is an obligation whether or not it is
drawn, and its limit is what the affordability calculator turns into an assumed
monthly payment when the application does not declare one.

Money is transcribed exactly as printed and scaled here rather than in the
prompt. The report's unit convention is fixed — VND columns are in triệu đồng,
foreign-currency columns are in the currency's own unit — so the multiplication
is a known constant, and a known constant belongs in code, not in a dozen
opportunities for a model to drop a zero.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from src.agents.structured_extraction import build_extraction_chain, run_extraction

REQUIRED_TOP_LEVEL_KEYS = {
    "bao_cao",
    "khach_hang",
    "du_no_hien_tai",
    "du_no_12_thang",
    "xep_hang_tin_dung",
}

# The whole report states VND in triệu đồng. Foreign currency is quoted in its
# own unit ("1 đơn vị tiền tệ đối với ngoại tệ"), so it is never scaled — using
# this multiplier on a USD column would be wrong by a factor of a million.
VND_UNIT_MULTIPLIER = 10**6
VND_CURRENCY_CODES = frozenset({"VND", "VNĐ", "VN", ""})

CIC_S10A_EXTRACTION_SYSTEM_PROMPT = """
Bạn trích xuất BÁO CÁO THÔNG TIN TÍN DỤNG của Trung tâm Thông tin Tín dụng Quốc
gia (CIC) về một khách hàng CÁ NHÂN thành JSON, phục vụ thẩm định cho vay bán
lẻ. Báo cáo có thể là mẫu báo cáo quan hệ tín dụng (vd S10A) hoặc mẫu báo cáo
tín dụng cá nhân khác của CIC — tìm thông tin theo NỘI DUNG, đừng dựa vào số
thứ tự mục.

Văn bản là OCR của bản scan nên có thể lệch dòng, dính cột. Chỉ trích thứ đọc
được trên giấy. Không suy diễn, không bịa số.

QUY TẮC ĐỌC SỐ — phần dễ sai nhất, đọc kỹ:

1. Dấu "," trong báo cáo này là DẤU PHÂN CÁCH HÀNG NGHÌN, KHÔNG phải dấu thập
   phân. "12,345" là mười hai nghìn ba trăm bốn mươi lăm -> ghi 12345.
   "1,234,567" -> ghi 1234567. Đọc thành 12.345 là sai một nghìn lần.

2. GHI ĐÚNG CON SỐ IN TRÊN GIẤY, chỉ bỏ dấu phân cách hàng nghìn. Số tiền VND
   trong báo cáo CIC in theo đơn vị TRIỆU ĐỒNG — TUYỆT ĐỐI KHÔNG tự nhân lên,
   không tự quy đổi đơn vị; chương trình làm việc đó sau.

3. Ô ghi "(-)" nghĩa là "Thiếu kỳ báo cáo số liệu" -> để null.
   Ô TRỐNG -> để null.
   TUYỆT ĐỐI không thay bằng 0: "không có số liệu" và "dư nợ bằng 0" là hai
   chuyện trái ngược nhau khi đọc biểu đồ dư nợ.

4. Cột VND và cột ngoại tệ là hai cột riêng biệt. Không cộng vào nhau, không
   quy đổi qua lại — báo cáo không in tỉ giá nào.

5. Bảng diễn biến dư nợ 12 tháng (nếu có) thường có chú thích dư nợ ĐÃ BAO GỒM
   dư nợ ngoại tệ quy đổi. Chép đúng cột đang có, KHÔNG cộng thêm số từ bảng
   khác vào — như vậy là tính hai lần.

6. Giữ nguyên thứ tự dòng như trên giấy. Không tự sắp xếp lại.

7. Nếu tài liệu KHÔNG có bảng diễn biến dư nợ theo tháng thì để
   "du_no_12_thang": [] và ghi lý do vào "extraction_notes". TUYỆT ĐỐI không
   bịa ra chuỗi 12 tháng.

8. THẺ TÍN DỤNG: mỗi thẻ (hoặc mỗi TCTD phát hành thẻ) một dòng trong
   "the_tin_dung", ghi HẠN MỨC thẻ nếu báo cáo in, và dư nợ thẻ. Thẻ có hạn
   mức nhưng dư nợ bằng 0 vẫn phải liệt kê — hạn mức thẻ là nghĩa vụ tiềm năng.

9. Điểm tín dụng/xếp hạng tín dụng cá nhân (nếu báo cáo in) ghi vào
   "xep_hang_tin_dung"; không có thì để [].

QUY TẮC GHI THÁNG:
Mọi nhãn tháng ghi dạng "MM/YYYY" (hai chữ số tháng), ví dụ "03/2026".

"page" là số trang chứa bảng, đọc từ mốc "--- Page N ---" trong văn bản OCR;
không xác định được thì để null. Không bịa số trang.

Trả về CHÍNH XÁC JSON theo schema sau, không thêm text nào khác:
{{
  "bao_cao": {{
    "so_hieu": "số hiệu báo cáo, hoặc ''",
    "ngay_gui": "dd/mm/yyyy hoặc ''",
    "don_vi_tra_cuu": "tên tổ chức tra cứu hoặc ''",
    "page": <số nguyên hoặc null>
  }},
  "khach_hang": {{
    "ten": "...", "ma_cic": "...", "so_giay_to": "số CCCD/CMND/hộ chiếu",
    "ngay_sinh": "dd/mm/yyyy hoặc ''", "dia_chi": "...",
    "page": <số nguyên hoặc null>
  }},
  "du_no_hien_tai": [
    {{"tctd": "tên tổ chức tín dụng và chi nhánh",
      "ngay_bao_cao": "dd/mm/yyyy hoặc ''",
      "khoan_muc": "vd: Dư nợ cho vay ngắn hạn | Dư nợ cho vay trung hạn | Dư nợ cho vay dài hạn | Tổng cộng",
      "nhom_no": "vd: Nợ đủ tiêu chuẩn, hoặc ''",
      "vnd": <số như in trên giấy, hoặc null>,
      "ngoai_te": <số như in trên giấy, hoặc null>,
      "loai_ngoai_te": "vd: USD, hoặc ''",
      "page": <số nguyên hoặc null>}}
  ],
  "the_tin_dung": [
    {{"tctd": "tổ chức phát hành thẻ",
      "han_muc": <hạn mức thẻ, số như in trên giấy, hoặc null>,
      "du_no": <dư nợ thẻ, số như in trên giấy, hoặc null>,
      "so_ngay_cham_thanh_toan": <số ngày chậm thanh toán nếu có, hoặc null>,
      "ngay_bao_cao": "dd/mm/yyyy hoặc ''",
      "page": <số nguyên hoặc null>}}
  ],
  "du_no_12_thang": [
    {{"thang": "MM/YYYY",
      "du_no_vay": <số như in trên giấy, hoặc null>,
      "du_no_the": <số như in trên giấy, hoặc null>,
      "tong_du_no": <số như in trên giấy, hoặc null>,
      "page": <số nguyên hoặc null>}}
  ],
  "cam_ket_ngoai_bang": [
    {{"tctd": "...", "gia_tri": <số như in trên giấy, hoặc null>,
      "loai_tien": "VND | USD | ...", "nhom_no": "...",
      "ngay_bao_cao": "dd/mm/yyyy hoặc ''", "page": <số nguyên hoặc null>}}
  ],
  "xep_hang_tin_dung": [
    {{"nam": "YYYY", "hang": "vd: hạng/điểm tín dụng như in", "diem": <điểm số nếu có, hoặc null>,
      "pd_phan_tram": <số hoặc null>, "dien_giai": "...", "page": <số nguyên hoặc null>}}
  ],
  "canh_bao": {{
    "no_xau_5_nam": "nguyên văn kết luận về nợ xấu 5 năm gần nhất, hoặc ''",
    "no_can_chu_y_12_thang": "nguyên văn kết luận về nợ cần chú ý 12 tháng gần nhất, hoặc ''",
    "cham_thanh_toan_the": "nguyên văn kết luận về chậm thanh toán thẻ, hoặc ''",
    "du_no_the_tin_dung": "nguyên văn kết luận về dư nợ thẻ tín dụng, hoặc ''",
    "khac": "cảnh báo khác in trên báo cáo (bảo lãnh cho người khác, nợ đã xử lý rủi ro...), hoặc ''"
  }},
  "extraction_notes": ["ghi chú về mục thiếu, không chắc chắn, hoặc OCR kém"]
}}
"""


def build_cic_s10a_extraction_chain(llm: Any):
    """Build the JSON-output extraction chain for the CIC S10A report."""

    return build_extraction_chain(CIC_S10A_EXTRACTION_SYSTEM_PROMPT, llm)


# --------------------------------------------------------------------------
# Month labels
# --------------------------------------------------------------------------

_YEAR = r"(?:19|20)\d{2}"
_SEP = r"\s*[/\-.]\s*"
# Tried in this order. Day-month-year first so "31/03/2026" resolves on the
# month, not on a "31/03" that a bare month/year pattern would never match
# anyway — but the explicit rule documents the intent.
_DATE_PATTERNS = (
    re.compile(rf"(?<!\d)(?:0?[1-9]|[12]\d|3[01]){_SEP}(0?[1-9]|1[0-2]){_SEP}({_YEAR})(?!\d)"),
    re.compile(rf"(?<!\d)(0?[1-9]|1[0-2]){_SEP}({_YEAR})(?!\d)"),
)
# Year-first forms put the groups the other way round, so they are matched
# separately rather than bent into the tuple above.
_YEAR_FIRST_PATTERNS = (
    re.compile(rf"(?<!\d)({_YEAR}){_SEP}(0?[1-9]|1[0-2]){_SEP}(?:0?[1-9]|[12]\d|3[01])(?!\d)"),
    re.compile(rf"(?<!\d)({_YEAR}){_SEP}(0?[1-9]|1[0-2])(?!\d)"),
)
# Whole digit runs only. Matching a fixed width instead would carve a plausible
# date out of the middle of a longer number — an eight-digit window slides
# straight across a twelve-digit ID number.
_DIGIT_RUN = re.compile(r"(?<!\d)\d+(?!\d)")
_YEAR_ONLY = re.compile(rf"(?<!\d){_YEAR}(?!\d)")


def _month_year_from_digit_run(run: str) -> tuple[str, str] | None:
    """Read (month, year) out of a date written without separators.

    "032026" is March 2026 and "31032026" is 31 March 2026, but a twelve-digit
    ID number is neither. Only 6- and 8-digit runs qualify, and only when the
    leftover digits form a plausible day/month.
    """

    if len(run) == 6:
        if _YEAR_ONLY.fullmatch(run[2:]) and 1 <= int(run[:2]) <= 12:
            return run[:2], run[2:]
        if _YEAR_ONLY.fullmatch(run[:4]) and 1 <= int(run[4:]) <= 12:
            return run[4:], run[:4]
        return None
    if len(run) == 8:
        if (
            _YEAR_ONLY.fullmatch(run[4:])
            and 1 <= int(run[:2]) <= 31
            and 1 <= int(run[2:4]) <= 12
        ):
            return run[2:4], run[4:]
        if (
            _YEAR_ONLY.fullmatch(run[:4])
            and 1 <= int(run[4:6]) <= 12
            and 1 <= int(run[6:]) <= 31
        ):
            return run[4:6], run[:4]
    return None


def normalize_month_label(raw: Any) -> str:
    """Render any month label the model produced as a single "MM/YYYY" form.

    The source table already prints "03/2026", but OCR and models between them
    produce "T3/2026", "3/2026", "032026" and "31/03/2026" for the same cell.
    These strings are the join key between the debt series and the income series,
    so two spellings of one month silently become two points on the chart.

    A label that yields no month is returned unchanged — a slightly odd label
    beats a silently dropped data point. Idempotent.
    """

    text = str(raw or "").strip()
    if not text:
        return text

    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return f"{int(match.group(1)):02d}/{match.group(2)}"
    for pattern in _YEAR_FIRST_PATTERNS:
        match = pattern.search(text)
        if match:
            return f"{int(match.group(2)):02d}/{match.group(1)}"

    for run in _DIGIT_RUN.findall(text):
        if len(run) in (6, 8) and (found := _month_year_from_digit_run(run)):
            return f"{int(found[0]):02d}/{found[1]}"
    return text


def month_sort_key(label: str) -> tuple[int, int]:
    """Chronological sort key for a "MM/YYYY" label.

    Anything unparseable sorts last rather than raising, so one malformed row
    cannot take down a whole series.
    """

    match = re.fullmatch(r"(\d{1,2})/(\d{4})", str(label or "").strip())
    if not match:
        return (9999, 99)
    return (int(match.group(2)), int(match.group(1)))


# --------------------------------------------------------------------------
# Unit scaling
# --------------------------------------------------------------------------


def _scale_vnd(value: Any) -> Any:
    """Turn a printed triệu-đồng figure into đồng. Leaves null and text alone."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return value * VND_UNIT_MULTIPLIER


def normalize_amounts(result: dict[str, Any]) -> dict[str, Any]:
    """Scale every VND amount from triệu đồng to đồng, in place.

    Foreign-currency figures are left untouched: the report quotes them in their
    own unit. Scaling is driven by which column a number came from, not by which
    table it sits in, because section 2.5 puts VND and USD rows in one table and
    distinguishes them only by the "Loại tiền" cell.
    """

    for row in result.get("du_no_hien_tai") or []:
        if isinstance(row, dict):
            row["vnd"] = _scale_vnd(row.get("vnd"))
            # "ngoai_te" is deliberately not scaled.

    for row in result.get("the_tin_dung") or []:
        if isinstance(row, dict):
            row["han_muc"] = _scale_vnd(row.get("han_muc"))
            row["du_no"] = _scale_vnd(row.get("du_no"))

    for row in result.get("du_no_12_thang") or []:
        if isinstance(row, dict):
            row["thang"] = normalize_month_label(row.get("thang"))
            for key in ("du_no_vay", "du_no_the", "tong_du_no"):
                row[key] = _scale_vnd(row.get(key))

    for row in result.get("cam_ket_ngoai_bang") or []:
        if not isinstance(row, dict):
            continue
        currency = str(row.get("loai_tien") or "").strip().upper()
        if currency in VND_CURRENCY_CODES:
            row["gia_tri"] = _scale_vnd(row.get("gia_tri"))

    return result


def extract_cic_s10a_structured_data(
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
        "No CIC S10A extraction LLM configured.",
    )
    if result is None:
        return None, error
    return normalize_amounts(result), ""


# --------------------------------------------------------------------------
# Merging into a chart-ready series
# --------------------------------------------------------------------------


def _first_number(row: dict[str, Any], *keys: str) -> float | None:
    """First numeric value among ``keys``, or None when every one is missing.

    "Tổng dư nợ" is the column to chart, but a report whose card column is empty
    sometimes leaves the total empty too and fills only "Dư nợ vay". Falling back
    keeps that month on the chart instead of punching a hole in it.
    """

    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def merge_debt_series(
    extractions: Iterable[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge per-file S10A extractions into one ascending monthly debt series.

    Takes ``(filename, extraction)`` pairs and returns
    ``[{"thang", "du_no", "nguon"}]`` sorted oldest-first — the table is printed
    newest-first, and a chart read right to left is a chart read wrong.

    Months with no figure are dropped rather than zero-filled: section 2.6 marks
    a missing reporting period with "(-)", which the extraction turns into null,
    and drawing that as zero would invent a debt collapse. The renderer breaks
    the line across the gap instead.

    When two files report the same month, the first non-null figure wins and both
    filenames are recorded, so a citation can name the file a figure came from.
    """

    by_month: dict[str, dict[str, Any]] = {}
    for filename, extraction in extractions:
        if not isinstance(extraction, dict):
            continue
        for row in extraction.get("du_no_12_thang") or []:
            if not isinstance(row, dict):
                continue
            month = normalize_month_label(row.get("thang"))
            if not month:
                continue
            value = _first_number(row, "tong_du_no", "du_no_vay")
            entry = by_month.setdefault(
                month, {"thang": month, "du_no": None, "nguon": []}
            )
            if entry["du_no"] is None and value is not None:
                entry["du_no"] = value
            if filename and filename not in entry["nguon"]:
                entry["nguon"].append(filename)

    return [
        entry
        for entry in sorted(by_month.values(), key=lambda e: month_sort_key(e["thang"]))
        if entry["du_no"] is not None
    ]
