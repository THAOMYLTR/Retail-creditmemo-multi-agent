"""Extraction prompts for BRD items 1-4 ("Pháp lý doanh nghiệp").

One prompt per BRD item, each returning one JSON shape whatever the document
(GCN ĐKDN, điều lệ, sổ cổ đông, báo cáo thực địa, bản tra cứu Cổng ĐKDN...).
The deterministic checks in ``checks.py`` then compare those shapes across
sources — that is where "khớp / không khớp" is decided, never in the prompt.

Shared rules, stated in every prompt because each runs as its own call:
- Only what is printed. No inference, no "reasonable" values; missing → null/''/[].
- Money exactly as printed plus ``source_unit``; scaling happens in code
  (same arrangement as every retail pass — see structured_extraction.py).
- Dates as dd/mm/yyyy when the day is known, otherwise as printed.
- ``page`` from the "--- Page N ---" markers, null when unknown.

Braces are doubled because these strings go through ChatPromptTemplate.
"""

_COMMON_RULES = """
QUY TẮC BẮT BUỘC:
- Chỉ ghi thông tin CÓ THẬT trong văn bản. Không suy đoán, không điền giá trị
  "hợp lý". Không tìm thấy thì để null (số/bool), '' (chuỗi) hoặc [] (danh sách).
- Giữ NGUYÊN VĂN tên doanh nghiệp, tên người, địa chỉ như in trên giấy (có dấu).
- Mã số doanh nghiệp/MST ghi đúng từng chữ số, giữ dấu gạch nối của chi nhánh
  (ví dụ "0101234567-001"). Không tự thêm/bớt chữ số.
- Ngày tháng ghi "dd/mm/yyyy"; chỉ có tháng/năm thì "mm/yyyy"; chỉ có năm thì "yyyy".
- KHÔNG tự quy đổi đơn vị tiền. Ghi con số ĐÚNG NHƯ IN (chỉ bỏ dấu phân cách
  hàng nghìn) và ghi "source_unit": "dong" | "nghin dong" | "trieu dong" |
  "ty dong". Không chú thích đơn vị thì "dong". Hệ thống tự nhân theo đơn vị.
- "page" là số trang đọc từ mốc "--- Page N ---"; không xác định được thì null.
- Văn bản không chứa nội dung của mục này thì trả các trường rỗng và ghi lý do
  vào "extraction_notes".
"""

LEGAL_BASIC_PROMPT = """
Bạn trích xuất THÔNG TIN PHÁP LÝ CƠ BẢN của một DOANH NGHIỆP vay vốn tại ngân
hàng Việt Nam (BRD mục 1). Văn bản có thể là GCN ĐKDN, điều lệ, sổ cổ đông,
báo cáo thực địa, hoặc bản tra cứu Cổng thông tin quốc gia về ĐKDN / cơ quan
thuế (masothue.com, bocaodientu.com.vn).
""" + _COMMON_RULES + """
QUY TẮC RIÊNG:
- "operating_status": chỉ điền khi văn bản NÊU RÕ tình trạng (thường là bản tra
  cứu cơ quan thuế/Cổng ĐKDN): "dang_hoat_dong" | "tam_ngung" | "ngung_hoat_dong"
  | "giai_the" | "pha_san" | "khac". GCN ĐKDN không ghi tình trạng thì để ''.
- "change_history": mỗi lần thay đổi đăng ký NÊU TRONG VĂN BẢN là một dòng.
  "fields_changed" chọn trong: "name" | "capital" | "address" | "legal_rep" |
  "members" | "business_lines" | "other". GCN ghi "Đăng ký thay đổi lần thứ N,
  ngày ..." mà không nói thay đổi gì thì vẫn ghi một dòng với fields_changed [].
- "site_visit": CHỈ điền khi văn bản là báo cáo thực địa/khảo sát. Ghi điều cán
  bộ QUAN SÁT được, không ghi ý kiến.

Trả về CHÍNH XÁC JSON sau, không thêm text:
{{
  "company_name": "tên doanh nghiệp tiếng Việt, hoặc ''",
  "company_name_foreign": "tên nước ngoài, hoặc ''",
  "abbreviation": "tên viết tắt, hoặc ''",
  "tax_code": "mã số doanh nghiệp/MST, hoặc ''",
  "legal_form": "cong_ty_co_phan | tnhh_mot_thanh_vien | tnhh_hai_thanh_vien_tro_len | doanh_nghiep_tu_nhan | cong_ty_hop_danh | khac | ''",
  "establishment_date": "ngày đăng ký lần đầu, hoặc ''",
  "registration_change_number": <lần đăng ký thay đổi gần nhất (số), hoặc null>,
  "latest_change_date": "ngày đăng ký thay đổi gần nhất, hoặc ''",
  "charter_capital": <vốn điều lệ, số như in, hoặc null>,
  "source_unit": "dong | nghin dong | trieu dong | ty dong",
  "head_office_address": "địa chỉ trụ sở chính, hoặc ''",
  "legal_representatives": ["họ tên người đại diện theo pháp luật"],
  "operating_status": "dang_hoat_dong | tam_ngung | ngung_hoat_dong | giai_the | pha_san | khac | ''",
  "status_as_of": "ngày tra cứu/ngày ghi nhận tình trạng, hoặc ''",
  "change_history": [
    {{"date": "...", "change_number": <số hoặc null>, "fields_changed": ["name | capital | address | legal_rep | members | business_lines | other"],
      "before": "nội dung trước, hoặc ''", "after": "nội dung sau, hoặc ''", "page": <số hoặc null>}}
  ],
  "site_visit": {{
    "visit_date": "...",
    "address_visited": "...",
    "actual_operation_confirmed": <true nếu cán bộ xác nhận DN hoạt động thực tế tại địa chỉ, false nếu không, null nếu không nêu>,
    "coordinates": "tọa độ nếu có, hoặc ''",
    "signage_observed": <true | false | null>,
    "observations": ["điều quan sát được: biển hiệu, văn phòng, kho xưởng, nhân sự..."]
  }},
  "page": <trang chứa thông tin chính, hoặc null>,
  "extraction_notes": ["ghi chú về thông tin thiếu, mờ, không chắc"]
}}
"""

SHAREHOLDERS_PROMPT = """
Bạn trích xuất CƠ CẤU CỔ ĐÔNG / THÀNH VIÊN GÓP VỐN của một DOANH NGHIỆP vay vốn
(BRD mục 2). Văn bản có thể là GCN ĐKDN, điều lệ, sổ đăng ký cổ đông/thành viên,
hợp đồng chuyển nhượng vốn, nghị quyết, báo cáo thực địa, bản tra cứu Cổng ĐKDN.
""" + _COMMON_RULES + """
QUY TẮC RIÊNG:
- Mỗi cổ đông/thành viên là một dòng "members". "member_type": "ca_nhan" (cá
  nhân) hoặc "phap_nhan" (tổ chức/doanh nghiệp).
- "ownership_pct": tỷ lệ sở hữu ĐÚNG NHƯ IN, dạng số phần trăm (35 nghĩa là
  35%). Văn bản không in tỷ lệ thì để null — KHÔNG tự chia vốn góp cho vốn điều lệ.
- "acquisition_method": "gop_von_thanh_lap" (góp khi thành lập) |
  "nhan_chuyen_nhuong" | "tang_von" | "thua_ke_tang_cho" | "khac" | ''.
- "transfers": mọi giao dịch chuyển nhượng vốn góp/cổ phần nêu trong văn bản.
- "relationships": chỉ ghi quan hệ giữa các chủ sở hữu khi văn bản NÊU RÕ (vợ
  chồng, cha con, anh em, nhân viên, đối tác, công ty mẹ-con...). Kèm câu trích
  ngắn làm căn cứ.
- "beneficial_owner_notes": CHỈ từ báo cáo thực địa/RBO — ai là chủ thực sự,
  người đứng tên hộ... ghi đúng như cán bộ ghi nhận.

Trả về CHÍNH XÁC JSON sau, không thêm text:
{{
  "document_date": "...",
  "source_unit": "dong | nghin dong | trieu dong | ty dong",
  "charter_capital": <vốn điều lệ, số như in, hoặc null>,
  "members": [
    {{"name": "...", "member_type": "ca_nhan | phap_nhan", "id_number": "CCCD/hộ chiếu/MSDN nếu có, hoặc ''",
      "capital_amount": <vốn góp, số như in, hoặc null>, "ownership_pct": <số hoặc null>,
      "acquisition_date": "...", "acquisition_method": "gop_von_thanh_lap | nhan_chuyen_nhuong | tang_von | thua_ke_tang_cho | khac | ''",
      "page": <số hoặc null>}}
  ],
  "transfers": [
    {{"date": "...", "transferor": "...", "transferee": "...", "capital_amount": <số như in hoặc null>,
      "ownership_pct": <số hoặc null>, "page": <số hoặc null>}}
  ],
  "relationships": [
    {{"member_a": "...", "member_b": "...", "relation": "...", "evidence": "trích ngắn"}}
  ],
  "beneficial_owner_notes": ["..."],
  "extraction_notes": ["..."]
}}
"""

LEGAL_REP_PROMPT = """
Bạn trích xuất NGƯỜI ĐẠI DIỆN THEO PHÁP LUẬT và NGƯỜI ĐIỀU HÀNH THỰC TẾ của một
DOANH NGHIỆP vay vốn (BRD mục 3). Văn bản có thể là GCN ĐKDN, điều lệ, quyết định
bổ nhiệm, nghị quyết, báo cáo thực địa/phỏng vấn RBO, bản tra cứu Cổng ĐKDN.
""" + _COMMON_RULES + """
QUY TẮC RIÊNG:
- "legal_representatives": người đại diện theo pháp luật HIỆN TẠI theo văn bản.
- "previous_representatives": người đại diện trước đây khi văn bản nêu (lịch
  sử thay đổi trên Cổng ĐKDN, quyết định miễn nhiệm...).
- "people_met" và "actual_operator": CHỈ điền từ báo cáo thực địa/phỏng vấn —
  ai là người tiếp xúc, ai quyết định kinh doanh, ai hiểu hoạt động và tài
  chính của DN. Ghi đúng như cán bộ ghi nhận.
- KHÔNG đánh giá năng lực, kinh nghiệm của ban lãnh đạo (đã bỏ khỏi BRD).

Trả về CHÍNH XÁC JSON sau, không thêm text:
{{
  "legal_representatives": [
    {{"name": "...", "title": "chức danh, hoặc ''", "id_number": "CCCD/hộ chiếu nếu có, hoặc ''",
      "appointed_date": "...", "appointing_document": "số/ngày quyết định, nghị quyết nếu có, hoặc ''",
      "page": <số hoặc null>}}
  ],
  "previous_representatives": [
    {{"name": "...", "title": "...", "until_date": "...", "page": <số hoặc null>}}
  ],
  "people_met": [
    {{"name": "...", "role": "...", "decides_business": <true | false | null>,
      "understands_operations_finance": <true | false | null>}}
  ],
  "actual_operator": {{"name": "người điều hành thực tế, hoặc ''", "role": "...", "evidence": "trích ngắn, hoặc ''"}},
  "extraction_notes": ["..."]
}}
"""

LIABILITIES_PROMPT = """
Bạn trích xuất THÔNG TIN BẤT LỢI của một DOANH NGHIỆP vay vốn (BRD mục 4): nợ
thuế, nợ BHXH, phạt vi phạm hành chính (thuế, môi trường, PCCC, lao động...),
cưỡng chế, kết luận kiểm tra/thanh tra, tranh chấp/kiện tụng và tình trạng khắc
phục. Văn bản có thể là thông báo nợ thuế, xác nhận nghĩa vụ thuế, thông báo
BHXH, quyết định xử phạt, biên bản kiểm tra, hồ sơ tòa án/trọng tài, thuyết
minh BCTC kiểm toán, báo cáo thực địa, bản tra cứu công khai.
""" + _COMMON_RULES + """
QUY TẮC RIÊNG:
- Xác nhận "không còn nợ"/"đã hoàn thành nghĩa vụ" thì ghi vào
  "clearance_confirmations", KHÔNG tạo dòng nợ với số tiền 0.
- "status" của khoản nợ: "con_no" | "da_nop" | "gia_han" | "dang_cuong_che" | ''.
- "remediation_status" của vi phạm: "da_khac_phuc" | "dang_khac_phuc" |
  "chua_khac_phuc" | ''. Chỉ điền khi văn bản nêu.
- "domain": "thue" | "bhxh" | "moi_truong" | "pccc" | "lao_dong" | "dat_dai" | "khac".
- Tranh chấp: "role" là vai trò của DOANH NGHIỆP vay vốn: "nguyen_don" |
  "bi_don" | "nguoi_lien_quan" | ''. "affects_operations_or_collateral": true
  chỉ khi văn bản nêu tranh chấp liên quan tài sản, hợp đồng chính, dòng tiền.
- Thuyết minh BCTC: chỉ lấy các khoản thuế/BHXH quá hạn, nợ tiềm tàng, vụ kiện
  được thuyết minh — không chép số dư thuế phải nộp thông thường.
- "site_risks": CHỈ từ báo cáo thực địa — rủi ro môi trường/PCCC quan sát được.

Trả về CHÍNH XÁC JSON sau, không thêm text:
{{
  "as_of_date": "...",
  "source_unit": "dong | nghin dong | trieu dong | ty dong",
  "tax_debts": [
    {{"authority": "...", "tax_type": "...", "amount": <số như in hoặc null>, "overdue_days": <số hoặc null>,
      "as_of": "...", "status": "con_no | da_nop | gia_han | dang_cuong_che | ''", "page": <số hoặc null>}}
  ],
  "social_insurance_debts": [
    {{"agency": "...", "amount": <số như in hoặc null>, "months_overdue": <số hoặc null>, "as_of": "...",
      "status": "con_no | da_nop | gia_han | dang_cuong_che | ''", "page": <số hoặc null>}}
  ],
  "clearance_confirmations": [
    {{"issuer": "...", "scope": "thue | bhxh", "as_of": "...", "statement": "trích ngắn", "page": <số hoặc null>}}
  ],
  "administrative_penalties": [
    {{"decision_no": "...", "date": "...", "issuer": "...", "domain": "thue | bhxh | moi_truong | pccc | lao_dong | dat_dai | khac",
      "violation": "...", "amount": <số như in hoặc null>, "remediation_status": "da_khac_phuc | dang_khac_phuc | chua_khac_phuc | ''",
      "page": <số hoặc null>}}
  ],
  "enforcement_actions": [
    {{"decision_no": "...", "date": "...", "issuer": "...", "measure": "biện pháp cưỡng chế", "amount": <số như in hoặc null>, "page": <số hoặc null>}}
  ],
  "inspections": [
    {{"date": "...", "agency": "...", "scope": "...", "findings": "...", "page": <số hoặc null>}}
  ],
  "disputes": [
    {{"case_no": "...", "forum": "tòa án/trọng tài", "role": "nguyen_don | bi_don | nguoi_lien_quan | ''", "counterparty": "...",
      "subject": "...", "claim_amount": <số như in hoặc null>, "status": "...", "date": "...",
      "affects_operations_or_collateral": <true | false | null>, "page": <số hoặc null>}}
  ],
  "site_risks": ["..."],
  "extraction_notes": ["..."]
}}
"""

PROMPTS = {
    "legal_basic": LEGAL_BASIC_PROMPT,
    "shareholders": SHAREHOLDERS_PROMPT,
    "legal_rep": LEGAL_REP_PROMPT,
    "liabilities": LIABILITIES_PROMPT,
}

REQUIRED_KEYS = {
    "legal_basic": {"company_name", "tax_code", "legal_representatives"},
    "shareholders": {"members"},
    "legal_rep": {"legal_representatives"},
    "liabilities": {"tax_debts", "social_insurance_debts", "administrative_penalties", "disputes"},
}
