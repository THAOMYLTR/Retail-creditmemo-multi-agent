"""Verify the corp_income agent (BRD items 1-4) end to end without an LLM.

Fake chains return fixed JSON per (file, pass), so this checks everything the
agent does in code: classification, which passes run on which document, unit
scaling, merging with internal/external data, and every B6 cross-check that
the BRD names — MST khớp tuyệt đối, thay đổi trong 12 tháng, tổng tỷ lệ sở hữu,
KH cá nhân mới nhận vốn góp, người đại diện thay đổi, nợ thuế/BHXH, tranh chấp.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.agents.corp_income import CorpIncomeAgent
from src.agents.corp_income.checks import parse_date, within_lookback
from src.agents.corp_income.documents import classify, load_types

failures = []


def expect(condition, message):
    print(("   ✓ " if condition else "   ✗ ") + message)
    if not condition:
        failures.append(message)


# --- 1. catalogue + classification -------------------------------------------
print("1. Phân loại hồ sơ")
types = load_types()
expect(len(types) >= 10, f"{len(types)} loại hồ sơ KHDN")
cases = {
    ("GCN_DKDN_ABC.pdf", ""): "gcn_dkdn",
    ("scan01.pdf", "GIẤY CHỨNG NHẬN ĐĂNG KÝ DOANH NGHIỆP CÔNG TY TNHH"): "gcn_dkdn",
    ("so_dang_ky_thanh_vien.pdf", ""): "so_dang_ky_co_dong",
    ("bao_cao_thuc_dia.docx.pdf", ""): "bao_cao_thuc_dia",
    ("tra_cuu_masothue.png", ""): "tra_cuu_dkdn",
    ("thong_bao_no_thue.pdf", ""): "thong_bao_no_thue",
    ("QD_xu_phat_moi_truong.pdf", ""): "quyet_dinh_xu_phat",
    ("anh_chup.jpg", "nội dung không liên quan"): "khac",
}
for (filename, content), wanted in cases.items():
    got = classify(filename, content)["type"]
    expect(got == wanted, f"{filename!r} → {got} (kỳ vọng {wanted})")
expect(classify("x.pdf", "", "dieu_le")["method"] == "override", "type override được tôn trọng")

# --- 2. date helpers ---------------------------------------------------------
print("\n2. Ngày tháng & cửa sổ 12 tháng")
as_of = parse_date("13/10/2026")
expect(parse_date("2026-03-05") == parse_date("05/03/2026"), "dd/mm/yyyy == yyyy-mm-dd")
expect(within_lookback("15/11/2025", as_of), "15/11/2025 nằm trong 12 tháng")
expect(not within_lookback("01/09/2025", as_of), "01/09/2025 ngoài 12 tháng")
expect(within_lookback("10/2025", as_of), "tháng 10/2025 tính cuối tháng → trong 12 tháng")

# --- 3. end-to-end with fake chains ------------------------------------------
GCN = {
    "company_name": "CÔNG TY TNHH THƯƠNG MẠI ABC", "company_name_foreign": "", "abbreviation": "ABC CO",
    "tax_code": "0101234567", "legal_form": "tnhh_hai_thanh_vien_tro_len", "establishment_date": "12/05/2015",
    "registration_change_number": 6, "latest_change_date": "20/03/2026", "charter_capital": 10,
    "source_unit": "ty dong", "head_office_address": "Số 1 Láng Hạ, Đống Đa, Hà Nội",
    "legal_representatives": ["Trần Thị B"], "operating_status": "", "status_as_of": "",
    "change_history": [{"date": "20/03/2026", "change_number": 6, "fields_changed": ["legal_rep", "members"],
                        "before": "", "after": "", "page": 1}],
    "site_visit": {}, "page": 1, "extraction_notes": [],
}
EXTERNAL_LOOKUP = {**GCN, "tax_code": "0101234568", "operating_status": "dang_hoat_dong",
                   "status_as_of": "10/10/2026", "change_history": []}
SITE_BASIC = {**{k: "" for k in GCN}, "legal_representatives": [], "change_history": [], "charter_capital": None,
              "source_unit": "dong", "registration_change_number": None, "page": None, "extraction_notes": [],
              "site_visit": {"visit_date": "01/10/2026", "address_visited": "Số 1 Láng Hạ",
                             "actual_operation_confirmed": True, "coordinates": "", "signage_observed": True,
                             "observations": ["Có biển hiệu, kho hàng"]}}
MEMBERS = {
    "document_date": "20/03/2026", "source_unit": "trieu dong", "charter_capital": 10000,
    "members": [
        {"name": "Nguyễn Văn A", "member_type": "ca_nhan", "id_number": "001080000001", "capital_amount": 6000,
         "ownership_pct": 60, "acquisition_date": "20/03/2026", "acquisition_method": "nhan_chuyen_nhuong", "page": 2},
        {"name": "Công ty CP Đầu tư XYZ", "member_type": "phap_nhan", "id_number": "0109999999",
         "capital_amount": 3000, "ownership_pct": 30, "acquisition_date": "12/05/2015",
         "acquisition_method": "gop_von_thanh_lap", "page": 2},
    ],
    "transfers": [{"date": "20/03/2026", "transferor": "Lê Văn C", "transferee": "Nguyễn Văn A",
                   "capital_amount": 6000, "ownership_pct": 60, "page": 3}],
    "relationships": [], "beneficial_owner_notes": [], "extraction_notes": [],
}
REP_GCN = {"legal_representatives": [{"name": "Trần Thị B", "title": "Giám đốc", "id_number": "",
                                      "appointed_date": "20/03/2026", "appointing_document": "", "page": 1}],
           "previous_representatives": [{"name": "Lê Văn C", "title": "Giám đốc", "until_date": "20/03/2026", "page": 1}],
           "people_met": [], "actual_operator": {"name": "", "role": "", "evidence": ""}, "extraction_notes": []}
REP_SITE = {"legal_representatives": [], "previous_representatives": [],
            "people_met": [{"name": "Nguyễn Văn A", "role": "Chủ tịch HĐTV", "decides_business": True,
                            "understands_operations_finance": True}],
            "actual_operator": {"name": "Nguyễn Văn A", "role": "Chủ tịch HĐTV", "evidence": "quyết định mọi hợp đồng"},
            "extraction_notes": []}
LIAB_TAX = {"as_of_date": "30/09/2026", "source_unit": "dong",
            "tax_debts": [{"authority": "Chi cục Thuế Đống Đa", "tax_type": "GTGT", "amount": 125000000,
                           "overdue_days": 120, "as_of": "30/09/2026", "status": "con_no", "page": 1}],
            "social_insurance_debts": [], "clearance_confirmations": [], "administrative_penalties": [],
            "enforcement_actions": [], "inspections": [], "disputes": [], "site_risks": [], "extraction_notes": []}
LIAB_SITE = {**{k: [] for k in LIAB_TAX if k not in ("as_of_date", "source_unit")}, "as_of_date": "", "source_unit": "dong",
             "site_risks": ["Kho chứa hàng dễ cháy, bình chữa cháy hết hạn"]}
EMPTY_MEMBERS = {"document_date": "", "source_unit": "dong", "charter_capital": None, "members": [],
                 "transfers": [], "relationships": [], "beneficial_owner_notes": [], "extraction_notes": []}

ANSWERS = {
    ("GCN_DKDN_ABC.pdf", "legal_basic"): GCN,
    ("GCN_DKDN_ABC.pdf", "shareholders"): MEMBERS,
    ("GCN_DKDN_ABC.pdf", "legal_rep"): REP_GCN,
    ("tra_cuu_masothue.txt", "legal_basic"): EXTERNAL_LOOKUP,
    ("tra_cuu_masothue.txt", "shareholders"): EMPTY_MEMBERS,
    ("tra_cuu_masothue.txt", "legal_rep"): {**REP_GCN, "previous_representatives": []},
    ("bao_cao_thuc_dia.txt", "legal_basic"): SITE_BASIC,
    ("bao_cao_thuc_dia.txt", "shareholders"): EMPTY_MEMBERS,
    ("bao_cao_thuc_dia.txt", "legal_rep"): REP_SITE,
    ("bao_cao_thuc_dia.txt", "liabilities"): LIAB_SITE,
    ("thong_bao_no_thue.txt", "liabilities"): LIAB_TAX,
}
calls = []


class FakeChain:
    def __init__(self, pass_name):
        self.pass_name = pass_name

    def invoke(self, payload):
        calls.append((payload["filename"], self.pass_name))
        import copy
        return copy.deepcopy(ANSWERS[(payload["filename"], self.pass_name)])


agent = CorpIncomeAgent(chains={p: FakeChain(p) for p in ("legal_basic", "shareholders", "legal_rep", "liabilities")})
result = agent.run(
    texts={name: "nội dung OCR" for name in ("GCN_DKDN_ABC.pdf", "tra_cuu_masothue.txt",
                                              "bao_cao_thuc_dia.txt", "thong_bao_no_thue.txt")},
    loan_date="13/10/2026",
    borrower_names=["Nguyễn Văn A"],
    internal_data={"legal_basic": {**GCN, "head_office_address": "Số 5 Cầu Giấy, Hà Nội", "source_unit": "ty dong"}},
    watchlist=[{"name": "Công ty CP Đầu tư XYZ", "tax_code": "0109999999", "list": "WL"}],
)
codes = {item: {f["code"]: f for f in result.items[item]["findings"]} for item in result.items}

print("\n3. Pass chạy theo loại hồ sơ")
expect(sorted(c for c in calls if c[0] == "thong_bao_no_thue.txt") == [("thong_bao_no_thue.txt", "liabilities")],
       "thông báo nợ thuế chỉ chạy pass liabilities")
expect(len([c for c in calls if c[0] == "GCN_DKDN_ABC.pdf"]) == 3, "GCN ĐKDN chạy 3 pass (mục 1-3)")

print("\n4. Mục 1 — pháp lý cơ bản")
expect(codes[1].get("tax_code_mismatch", {}).get("severity") == "alert", "MST lệch 1 chữ số → cảnh báo")
expect("status_active" in codes[1], "trạng thái đang hoạt động lấy từ bản tra cứu")
expect("changes_12m" in codes[1], "thay đổi ĐKDN ngày 20/03/2026 bị đánh dấu trong 12 tháng")
expect("internal_stale" in codes[1], "địa chỉ nội bộ khác GCN → nội bộ chưa cập nhật")
expect("site_operating" in codes[1], "báo cáo thực địa xác nhận hoạt động thực tế")
capital = result.items[1]["summary"]["Vốn điều lệ"]["value"]
expect(capital == "10.000.000.000 đ", f"vốn điều lệ 10 tỷ quy đổi đúng ({capital})")

print("\n5. Mục 2 — cổ đông, thành viên góp vốn")
expect(codes[2].get("ownership_not_100", {}).get("message", "").startswith("Tổng tỷ lệ sở hữu = 90"),
       "tổng sở hữu 90% → cần làm rõ")
expect(codes[2].get("borrower_new_member", {}).get("severity") == "alert",
       "KH cá nhân mới nhận chuyển nhượng 60% trong 12 tháng → cảnh báo")
expect("transfer_12m" in codes[2], "chuyển nhượng vốn gần thời điểm vay")
expect("corporate_members" in codes[2], "cổ đông pháp nhân → làm rõ sở hữu nhiều tầng")
expect(codes[2].get("watchlist_member", {}).get("severity") == "alert", "cổ đông thuộc WL → cảnh báo")
member_a = result.items[2]["summary"]["members"][0]
expect(member_a["capital_amount"] == 6_000_000_000, "vốn góp 6.000 triệu → 6 tỷ đồng")

print("\n6. Mục 3 — người đại diện pháp luật")
expect(codes[3].get("rep_changed_12m", {}).get("severity") == "alert", "đổi người đại diện 20/03/2026 → cảnh báo")
expect("operator_differs" in codes[3], "người điều hành thực tế (A) khác ĐDPL (B)")
expect("rep_match" in codes[3], "ĐDPL khớp giữa GCN và tra cứu")

print("\n7. Mục 4 — nợ thuế, BHXH, phạt, tranh chấp")
expect(codes[4].get("thue_debt", {}).get("severity") == "alert", "còn nợ thuế 125 triệu → cảnh báo")
expect("quá hạn > 90 ngày" in codes[4].get("thue_debt", {}).get("message", ""), "nợ quá hạn > 90 ngày được ghi rõ")
expect("site_risk" in codes[4], "rủi ro PCCC từ thực địa")
expect("bhxh_unverified" in codes[4], "chưa có xác nhận BHXH → ghi nhận")

print("\n8. Báo cáo")
md = result.markdown
for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## Hồ sơ đã xử lý"):
    expect(heading in md, f"báo cáo có '{heading}'")
expect(result.to_dict()["items"]["1"]["title"].startswith("Thông tin cơ bản"), "JSON serialisable")

print("\n8b. Template báo cáo (src/templates/corp-income-structure.md)")
from src.agents.corp_income.report import (DEFAULT_TEMPLATE, TEMPLATE_PATH, load_template,
                                           placeholder_names, render_markdown, unknown_placeholders)
expect(TEMPLATE_PATH.is_file(), "có file template mẫu trong src/templates")
expect(not unknown_placeholders(TEMPLATE_PATH.read_text(encoding="utf-8")), "template mẫu chỉ dùng placeholder hợp lệ")
expect(result.meta["template"].endswith("corp-income-structure.md"), "agent dùng template trong src/templates")
expect("<!--" not in md, "ghi chú trong template không lọt vào báo cáo")
builtin = render_markdown({"meta": result.meta, "documents": result.documents, "items": result.items}, DEFAULT_TEMPLATE)
expect(md.strip() == builtin.strip(), "template mẫu cho kết quả giống bố cục mặc định")
custom = "MST: {{tax_code}} ({{tax_code_source}})\nKết luận 2: {{item2_conclusion}}\nSai: {{khong_co}}"
rendered = render_markdown({"meta": result.meta, "documents": result.documents, "items": result.items}, custom)
expect(rendered.startswith("MST: 0101234568 (tra_cuu_masothue.txt"), "template tùy chỉnh điền đúng MST + nguồn")
expect(rendered.endswith("Sai: —"), "placeholder lạ → '—'")
expect(unknown_placeholders(custom) == ["khong_co"], "placeholder lạ được liệt kê")
expect(len(placeholder_names()) >= 50, f"{len(placeholder_names())} placeholder")

print("\n9. Không có LLM → không sập, ghi lỗi")
bare = CorpIncomeAgent().run(texts={"GCN_DKDN_ABC.pdf": "x"}, loan_date="13/10/2026")
expect(bare.documents[0]["errors"].get("legal_basic", "").startswith("No corp_income"), "lỗi thiếu LLM được ghi lại")
expect(bare.items[1]["findings"][0]["code"] == "no_data", "mục 1 báo không có dữ liệu")

if failures:
    print(f"\n❌ {len(failures)} lỗi")
    sys.exit(1)
print("\n✅ corp_income OK")
