"""LLM extraction of the CIC R20/R21 collateral report into structured JSON.

Runs once per document whose matrix type is flagged ``cic_r21_extraction``
(see src/matrix/document_matrix.yaml and ``is_cic_r21_type``).

Named after the CIC form code the plan reserved for it rather than "CIC"
generally, because CIC issues several unrelated forms against the same
customer and they share nothing but a letterhead: S10A is the
credit-relationship report (``cic_khach_hang_vay``), this one is the
collateral report (``cic_tai_san_bao_dam``). The printed form code is actually
"R20" — the number in the module name follows the earlier decision to keep it,
not the paper — so the classifier keywords and this module's own regex accept
both spellings rather than assuming either one.

Money is transcribed exactly as printed and scaled here rather than in the
prompt: "Giá trị TS" is stated in triệu đồng, a fixed convention for the whole
report, so the multiplication is a known constant that belongs in code.
"""

from __future__ import annotations

from typing import Any

from src.agents.structured_extraction import build_extraction_chain, run_extraction

REQUIRED_TOP_LEVEL_KEYS = {
    "bao_cao",
    "khach_hang",
    "danh_sach_tctd",
    "tai_san_bao_dam",
}

# The whole report states collateral value in triệu đồng ("Giá trị TS (Triệu
# VNĐ)"). No foreign-currency column exists on this form, unlike S10A, so
# there is only one multiplier and no column-by-column branching.
VND_UNIT_MULTIPLIER = 10**6

CIC_R21_EXTRACTION_SYSTEM_PROMPT = """
Bạn trích xuất BÁO CÁO THÔNG TIN BẢO ĐẢM TIỀN VAY (mã phiếu R20, đôi khi ghi
R21) của Trung tâm Thông tin Tín dụng Quốc gia (CIC), phục vụ thẩm định tín
dụng khách hàng cá nhân. Đây LÀ báo cáo khác hẳn "Báo cáo chi tiết quan hệ tín dụng" (mã
S10A) — S10A nói về dư nợ, báo cáo này chỉ nói về TÀI SẢN BẢO ĐẢM.

Văn bản là OCR của bản scan nên có thể lệch dòng, dính cột. Chỉ trích thứ đọc
được trên giấy. Không suy diễn, không bịa số.

QUY TẮC ĐỌC SỐ — phần dễ sai nhất, đọc kỹ:

1. Dấu "," trong báo cáo này là DẤU PHÂN CÁCH HÀNG NGHÌN, KHÔNG phải dấu thập
   phân. "12,345" là mười hai nghìn ba trăm bốn mươi lăm -> ghi 12345.

2. GHI ĐÚNG CON SỐ IN TRÊN GIẤY cho "Giá trị TS (Triệu VNĐ)", chỉ bỏ dấu phân
   cách hàng nghìn. TUYỆT ĐỐI KHÔNG tự nhân lên triệu, không tự quy đổi đơn vị
   — việc quy đổi do chương trình làm sau. Bạn chỉ cần chép đúng.

3. "Loại tài sản" là MÃ SỐ hai chữ số in trên giấy (ví dụ "01", "08", "21"),
   KHÔNG PHẢI số thứ tự. Chép nguyên văn CHUỖI KÝ TỰ, giữ số 0 ở đầu nếu có
   ("08" phải ghi "08", không ghi "8"). Không tự đoán hoặc tự dịch nghĩa mã
   này — báo cáo không in bảng chú giải mã, ghi sai nghĩa còn tệ hơn không ghi.

4. Ô "Mã số tài sản" hoặc "Ngày giải chấp" TRỐNG -> để null. Ô trống ở "Ngày
   giải chấp" nghĩa là tài sản CHƯA ĐƯỢC GIẢI CHẤP (vẫn đang thế chấp), không
   phải thiếu dữ liệu — đây là trạng thái có ý nghĩa, không phải lỗi OCR.

5. Nếu "Mô tả tài sản" ghi đúng câu "Không có bảo đảm tiền vay bằng tài sản"
   (hoặc câu tương đương xác nhận TCTD đó không nhận tài sản bảo đảm nào), vẫn
   GHI LẠI khối đó nguyên vẹn — đây là thông tin có thật (xác nhận TCTD không
   có TSBĐ), KHÔNG được bỏ qua khối này. Các trường số/ngày trong khối đó để
   null. Khối này thường nằm ở CUỐI danh sách; đừng nhầm nó với phần ghi chú
   pháp lý đứng sau mục "3. THÔNG TIN KHÁC VỀ KHÁCH HÀNG VAY" — hai phần đó
   khác nhau, khối tài sản luôn nằm TRƯỚC tiêu đề mục 3 đó.

6. QUAN TRỌNG — ĐẾM SỐ KHỐI TÀI SẢN BẰNG CỤM "Danh sách Tài sản bảo đảm:",
   KHÔNG dùng số thứ tự đầu dòng: OCR bản scan thường đọc sai số thứ tự (ví
   dụ đọc nhầm "3." thành "5." hoặc lặp lại đúng "3." ở hai khối khác nhau).
   Nếu tin vào số thứ tự, hai khối trông giống hệt nhau (cùng số, cùng TCTD)
   sẽ bị hiểu nhầm là MỘT khối lặp lại và bị gộp mất — trong khi cụm nhãn cố
   định "Danh sách Tài sản bảo đảm:" luôn xuất hiện đúng một lần cho mỗi khối
   thật, kể cả khi số thứ tự phía trên nó bị lỗi. Trước khi trả lời, ĐẾM xem
   cụm này xuất hiện bao nhiêu lần trong toàn văn bản — mảng "tai_san_bao_dam"
   phải có ĐÚNG bấy nhiêu phần tử, không hơn không kém.
   Sau khi đếm đủ số khối, mới chép số thứ tự in trên giấy vào "stt" của từng
   khối theo đúng thứ tự xuất hiện; nếu số đó không đọc được, để null thay vì
   đoán. Số thứ tự này đánh LIÊN TỤC xuyên suốt toàn báo cáo — không reset về 1
   ở mỗi tổ chức tín dụng mới.

   Hai khối LIÊN TIẾP của CÙNG một TCTD có thể có "Mã số tài sản" trông giống
   nhau (chỉ khác vài ký tự cuối, ví dụ "MD01234567" và "MD01234BDS") mà vẫn
   là HAI TÀI SẢN HOÀN TOÀN KHÁC NHAU — đọc kỹ "Mô tả tài sản" của từng khối
   để phân biệt, TUYỆT ĐỐI không tự bỏ bớt một khối vì tưởng là OCR đọc trùng
   một tài sản hai lần. Số khối trả về phải khớp đúng số đếm được ở trên,
   không được ít hơn vì lý do "trông giống trùng lặp".

7. Nếu tài liệu KHÔNG có mục "THÔNG TIN ĐẢM BẢO TIỀN VAY" (ví dụ đây thực ra
   là báo cáo S10A hoặc một loại báo cáo CIC khác) thì để "tai_san_bao_dam": []
   và ghi lý do vào "extraction_notes". TUYỆT ĐỐI không bịa ra danh sách tài
   sản.

"page" là số trang chứa dữ kiện, đọc từ mốc "--- Page N ---" trong văn bản
OCR; không xác định được thì để null. Không bịa số trang.

Trả về CHÍNH XÁC JSON theo schema sau, không thêm text nào khác:
{{
  "bao_cao": {{
    "so_hieu": "vd: 2026/R20, hoặc ''",
    "ngay_gui": "dd/mm/yyyy hoặc ''",
    "don_vi_tra_cuu": "tên tổ chức tra cứu hoặc ''",
    "page": <số nguyên hoặc null>
  }},
  "khach_hang": {{
    "ten": "...", "ma_cic": "...", "so_giay_to": "số CCCD/CMND/hộ chiếu",
    "dia_chi": "...",
    "page": <số nguyên hoặc null>
  }},
  "danh_sach_tctd": [
    {{"stt": <số nguyên hoặc null>, "tctd": "tên tổ chức/chi nhánh",
      "ma_tctd": "...", "ngay_bao_cao_du_no": "dd/mm/yyyy hoặc ''",
      "page": <số nguyên hoặc null>}}
  ],
  "tai_san_bao_dam": [
    {{"stt": <số in trên giấy nếu đọc rõ, hoặc null nếu mờ/không chắc>,
      "ma_tctd": "...", "tctd": "tên tổ chức/chi nhánh",
      "ngay_bao_cao_tai_san": "dd/mm/yyyy hoặc ''",
      "ma_so_tai_san": "... hoặc null",
      "loai_tai_san": "mã hai chữ số dạng chuỗi, vd '08', hoặc null",
      "mo_ta_tai_san": "nguyên văn mô tả trên giấy",
      "chu_so_huu": "... hoặc ''",
      "gia_tri_trieu_vnd": <số như in trên giấy, hoặc null>,
      "ngay_the_chap": "dd/mm/yyyy hoặc ''",
      "ngay_giai_chap": "dd/mm/yyyy hoặc null nếu trống",
      "page": <số nguyên hoặc null>}}
  ],
  "thong_tin_khac": "nguyên văn mục 3 nếu đọc được, hoặc ''",
  "extraction_notes": ["ghi chú về mục thiếu, không chắc chắn, hoặc OCR kém"]
}}
"""


def build_cic_r21_extraction_chain(llm: Any):
    """Build the JSON-output extraction chain for the CIC R20/R21 report."""

    return build_extraction_chain(CIC_R21_EXTRACTION_SYSTEM_PROMPT, llm)


def _scale_vnd(value: Any) -> Any:
    """Turn a printed triệu-đồng figure into đồng. Leaves null and text alone."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return value * VND_UNIT_MULTIPLIER


def normalize_amounts(result: dict[str, Any]) -> dict[str, Any]:
    """Scale every collateral value from triệu đồng to đồng, in place."""

    for row in result.get("tai_san_bao_dam") or []:
        if isinstance(row, dict):
            row["gia_tri_trieu_vnd"] = _scale_vnd(row.get("gia_tri_trieu_vnd"))

    return result


def extract_cic_r21_structured_data(
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
        "No CIC R21 extraction LLM configured.",
    )
    if result is None:
        return None, error
    return normalize_amounts(result), ""
