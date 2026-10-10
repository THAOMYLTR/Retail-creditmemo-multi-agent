"""Chứng minh không nhân đôi đơn vị, và số trong báo cáo được trình bày đúng (triệu VNĐ)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.agents.sitevisit_extraction import (  # noqa: E402
    normalize_amounts as sv_normalize,
    SITEVISIT_EXTRACTION_SYSTEM_PROMPT,
)
from src.agents import income_extraction as inc  # noqa: E402
from src.agents import proposal_extraction as prop  # noqa: E402
from src.utils.formatting import convert_amounts_in_text  # noqa: E402

fails = []


def expect(label, got, want):
    ok = got == want
    print(f"   {'✅' if ok else '❌'} {label}: {got!r}" + ("" if ok else f" (mong đợi {want!r})"))
    if not ok:
        fails.append(label)


print("=" * 68)
print("1. Thẩm định: model trả số như in + đơn vị, code quy đổi đúng một lần")
block = {"employment_verification": {"source_unit": "trieu dong", "verified_monthly_income": 35}}
expect("35 + 'trieu dong'", sv_normalize(block)["employment_verification"]["verified_monthly_income"], 35_000_000)
for unit in ("", "dong", None):
    block = {"employment_verification": {"source_unit": unit, "verified_monthly_income": 35_000_000}}
    expect(f"mặc định đồng ({unit!r})",
           sv_normalize(block)["employment_verification"]["verified_monthly_income"], 35_000_000)
block = {"collateral_verification": {"source_unit": "tỷ VNĐ", "estimated_value": 3.2}}
expect("3,2 'tỷ VNĐ'", sv_normalize(block)["collateral_verification"]["estimated_value"], 3_200_000_000)

print("\n2. Chứng từ thu nhập: một đơn vị cho cả tài liệu, tháng chuẩn hoá")
record = inc.normalize_income_record({
    "document_kind": "Sao ke tai khoan",
    "income_owner": {"name": "A"},
    "source_unit": "trieu dong",
    "monthly_income": [
        {"month": "T3/2026", "source_type": "Lương", "amount": 32.5},
        {"month": "01/2026", "source_type": "tiền thuê nhà", "amount": 10},
        {"month": "02/2026", "source_type": "luong", "amount": None},
    ],
    "recurring_debits": [{"monthly_amount": 4, "months_seen": 3}],
    "contract_terms": {"monthly_amount": 30, "monthly_allowances": None},
})
expect("document_kind chuẩn hoá", record["document_kind"], "sao_ke_tai_khoan")
expect("dòng thiếu số bị bỏ, còn lại sắp theo tháng",
       [(r["month"], r["source_type"], r["amount"]) for r in record["monthly_income"]],
       [("01/2026", "cho_thue", 10_000_000), ("03/2026", "luong", 32_500_000)])
expect("recurring_debits quy đổi", record["recurring_debits"][0]["monthly_amount"], 4_000_000)
expect("contract_terms quy đổi", record["contract_terms"]["monthly_amount"], 30_000_000)

print("\n3. Giấy đề nghị: mỗi khối một đơn vị riêng")
result = prop.normalize_terms(prop.normalize_amounts({
    "loan_request": {"source_unit": "ty dong", "amount": 1.5, "tenor_months": 240,
                     "interest_rate_pct_year": 0.095, "repayment_method": "tra gop"},
    "declared_income": {"source_unit": "trieu dong", "items": [{"monthly_amount": 35}]},
    "existing_debts": {"source_unit": "dong", "items": [{"monthly_payment": 4_000_000}]},
    "collateral": {"source_unit": "ty dong", "items": [{"value": 3.2}], "total_value": None},
}))
expect("số tiền vay 1,5 tỷ", result["loan_request"]["amount"], 1_500_000_000)
expect("thu nhập kê khai 35 triệu", result["declared_income"]["items"][0]["monthly_amount"], 35_000_000)
expect("nợ hiện hữu giữ nguyên đồng", result["existing_debts"]["items"][0]["monthly_payment"], 4_000_000)
expect("TSBĐ 3,2 tỷ", result["collateral"]["items"][0]["value"], 3_200_000_000)
expect("lãi suất thập phân 0.095 -> 9,5%/năm", result["loan_request"]["interest_rate_pct_year"], 9.5)
expect("phương thức lạ -> 'khac'", result["loan_request"]["repayment_method"], "khac")
bad = prop.normalize_terms({"loan_request": {"interest_rate_pct_year": 0.8, "tenor_months": 0}})
expect("0.8 (lãi tháng gõ nhầm) bị bỏ", bad["loan_request"]["interest_rate_pct_year"], None)
expect("thời hạn 0 bị bỏ", bad["loan_request"]["tenor_months"], None)

print("\n4. Prompt không bảo model tự quy đổi")
for prompt, name in ((SITEVISIT_EXTRACTION_SYSTEM_PROMPT, "sitevisit"),
                     (prop.PROPOSAL_EXTRACTION_SYSTEM_PROMPT, "proposal"),
                     (inc.INCOME_EXTRACTION_SYSTEM_PROMPT, "income")):
    ok = "KHÔNG tự quy đổi" in prompt or "KHÔNG tự quy đổi đơn vị" in prompt
    print(f"   {'✅' if ok else '❌'} {name}: cấm tự quy đổi")
    fails += [] if ok else [f"{name} thiếu luật đơn vị"]
    ok = '"dong"' in prompt
    print(f"   {'✅' if ok else '❌'} {name}: nêu mặc định \"dong\"")
    fails += [] if ok else [f"{name} thiếu mặc định"]

print("\n5. Báo cáo hiển thị bằng triệu VNĐ")
for src, want, why in (
    ("Dư nợ 1.850.000.000 VNĐ.", "Dư nợ 1.850,00 triệu VNĐ.", "số tiền lớn"),
    ("Lương 25.500.000 đồng/tháng.", "Lương 25,50 triệu VNĐ/tháng.", "lương tháng"),
    ("Diện tích 85.000 m2.", "Diện tích 85.000 m2.", "số nhỏ không phải tiền giữ nguyên"),
    ("Đã ghi 25,50 triệu VNĐ.", "Đã ghi 25,50 triệu VNĐ.", "số đã có đơn vị giữ nguyên"),
    ("CCCD 001234567890.", "CCCD 001234567890.", "dãy số định danh không bị đụng"),
):
    got = convert_amounts_in_text(src)
    print(f"   {'✅' if got == want else '❌'} {why}: {got!r}")
    if got != want:
        fails.append(f"convert: {src!r} -> {got!r}")

# --- Trình bày số trong báo cáo giao cho người đọc ---------------------------
# Hai luật đặt ra cho mọi agent, chốt lại bằng mã ở đây vì luật trong prompt chỉ
# là lời nhờ. Đáng canh nhất là chỗ chúng đụng EVIDENCE RULE: dấu "-" nghĩa là số
# ĐỌC ĐƯỢC và bằng không, còn ô TRỐNG nghĩa là hồ sơ không nêu. Lẫn hai thứ này
# là báo cáo nói dối người đọc, nên có riêng một khẳng định cho nó.
from src.utils.markdown_fixups import tidy_numbers  # noqa: E402
from src.agents.specialist import SpecialistAgent  # noqa: E402

print("\n6. Trình bày số: bỏ thập phân toàn 0, ô bằng không thành '-'")
for src, want, why in (
    ("Tỷ trọng 8,00% và 8,0%.", "Tỷ trọng 8% và 8%.", "bỏ thập phân toàn 0"),
    ("Tỷ trọng 35,2% giữ nguyên.", "Tỷ trọng 35,2% giữ nguyên.", "1 chữ số có nghĩa thì giữ"),
    ("Tỷ trọng 35,20% giữ nguyên.", "Tỷ trọng 35,20% giữ nguyên.", "0 sau chữ số có nghĩa thì giữ"),
    ("| Doanh thu | 0,00 | 376,64 |", "| Doanh thu | - | 376,64 |", "ô bằng 0 -> gạch ngang"),
    ("| Tỷ trọng | 0,00% | 8,0% |", "| Tỷ trọng | - | 8% |", "cả hai luật cùng lúc"),
    ("| Nợ quá hạn | 0 | 1,20 |", "| Nợ quá hạn | - | 1,20 |", "số 0 trần"),
    ("|---|---:|---:|", "|---|---:|---:|", "hàng phân cách không bị đụng"),
    ("| Ghi chú | | Chưa nêu |", "| Ghi chú | | Chưa nêu |", "Ô TRỐNG vẫn trống, không thành '-'"),
    ("| Năm | 2020 | 100 |", "| Năm | 2020 | 100 |", "số khác 0 không bị đụng"),
    ("Khách hàng có 0 người phụ thuộc.", "Khách hàng có 0 người phụ thuộc.",
     "số 0 trong câu văn không thành gạch ngang"),
    ("```mermaid\nA -->|8,0%| B\n```", "```mermaid\nA -->|8%| B\n```",
     "nhãn sơ đồ cũng là thứ người đọc nhìn thấy"),
):
    got = tidy_numbers(src)
    print(f"   {'✅' if got == want else '❌'} {why}")
    if got != want:
        fails.append(f"tidy_numbers: {src!r} -> {got!r}, mong {want!r}")

rules = SpecialistAgent.__init__.__doc__ or ""
import inspect  # noqa: E402
prompt_src = inspect.getsource(SpecialistAgent)
for needed, why in (
    ("NUMBER FORMAT RULE", "luật có trong prompt chung của mọi agent"),
    ('nghĩa là hồ sơ nêu và bằng không', "prompt nói rõ '-' khác ô trống"),
):
    ok = needed in prompt_src
    print(f"   {'✅' if ok else '❌'} {why}")
    if not ok:
        fails.append(f"prompt thiếu: {needed!r}")

print("\n" + "=" * 68)
if fails:
    print("❌ HỎNG:", *fails, sep="\n   - "); sys.exit(1)
print("✅ Tất cả kiểm chứng đơn vị + làm tròn đều đạt")
