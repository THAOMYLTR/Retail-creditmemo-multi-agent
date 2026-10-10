"""The repayment-capacity and maximum-loan tables compute what they claim.

These two tables decide the loan, and every agent is told to copy their figures
rather than recompute them — so the arithmetic has to be right here, and the
rules that make it conservative have to hold:

1. Instalment formulas: declining balance, annuity and bullet, plus their
   inverses (a payment turned back into principal round-trips exactly).
2. Evidence ranking: for the same month, money credited on a statement beats a
   payslip; at equal rank the lower figure wins; a contract only counts when
   nothing better covers that income type.
3. Obligations take the LARGEST of declared, statement and CIC-implied, plus an
   assumed card payment when CIC shows a card the application does not declare.
4. Living costs never fall below the policy floor.
5. The maximum loan is the minimum of its limits, names the one that binds, and
   at that amount the DSR sits exactly on the cap when DSR binds.
6. Age shortens the tenor before the capacity-based amount is computed.
7. An unnamed product uses the strictest limits across all products.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.agents.affordability_calculator import (  # noqa: E402
    build_affordability,
    monthly_payment,
    principal_for_payment,
)
from src.agents.loan_sizing_calculator import build_loan_sizing_table  # noqa: E402
from src.policy.retail_policy import load_policy  # noqa: E402
from src.types import ClassifiedDocument  # noqa: E402

fails: list[str] = []
TODAY = date(2026, 10, 10)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'✅' if ok else '❌'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def close(a, b, tol=1.0):
    return a is not None and b is not None and abs(a - b) <= tol


def doc(name: str, program: str = "MUA_NHA", **fields) -> ClassifiedDocument:
    d = ClassifiedDocument(
        path=f"/x/{name}", filename=name, content="", agent="INCOME_ANALYSIS_AGENT",
        reasoning="", confidence=1.0, loan_program=program,
    )
    for key, value in fields.items():
        setattr(d, key, value)
    return d


def income_doc(name, kind, rows, **extra):
    record = {"document_kind": kind, "income_owner": {"name": "A"},
              "source_unit": "dong", "monthly_income": rows}
    record.update(extra)
    return doc(name, is_income=True, income_extraction=record)


def row(label, result):
    return next((r for r in result.rows if r.label.startswith(label)), None)


# --- 1. instalment formulas --------------------------------------------------
print("1. Công thức trả nợ")
p = monthly_payment(1_200_000_000, 120, 12, "goc_deu_lai_giam_dan")
check("gốc đều: kỳ đầu = gốc/n + lãi tháng trên toàn bộ dư nợ", close(p, 10_000_000 + 12_000_000))
p = monthly_payment(1_000_000_000, 240, 9.6, "tra_gop_deu")
r = 0.096 / 12
check("trả góp đều: đúng công thức annuity",
      close(p, 1_000_000_000 * r / (1 - (1 + r) ** -240)))
check("gốc cuối kỳ: chỉ tiền lãi", close(monthly_payment(600_000_000, 36, 12, "goc_cuoi_ky"), 6_000_000))
check("phương thức lạ -> tính như gốc đều (bảo thủ nhất)",
      close(monthly_payment(1_200_000_000, 120, 12, "khac"), 22_000_000))
for method in ("goc_deu_lai_giam_dan", "tra_gop_deu"):
    pay = monthly_payment(1_500_000_000, 300, 9.5, method)
    back = principal_for_payment(pay, 300, 9.5, method)
    check(f"{method}: đảo ngược ra đúng gốc", close(back, 1_500_000_000))
check("lãi suất 0: annuity chia đều", close(monthly_payment(120, 12, 0, "tra_gop_deu"), 10))
check("thiếu thời hạn -> None", monthly_payment(1, None, 10, "tra_gop_deu") is None)

# --- shared fixture ------------------------------------------------------------
statement = income_doc(
    "SAO_KE.pdf", "sao_ke_tai_khoan",
    [{"month": f"{m:02d}/2026", "source_type": "luong", "amount": 30_000_000} for m in range(1, 7)],
    recurring_debits=[{"description": "Trả góp FE", "monthly_amount": 5_000_000, "months_seen": 6}],
)
payslip = income_doc(
    "LUONG.pdf", "xac_nhan_luong",
    [{"month": "06/2026", "source_type": "luong", "amount": 45_000_000},
     {"month": "07/2026", "source_type": "luong", "amount": 31_000_000}],
)
rental = income_doc("HD_THUE.pdf", "hop_dong_cho_thue", [],
                    contract_terms={"monthly_amount": 10_000_000})
labour = income_doc("HDLD.pdf", "hop_dong_lao_dong", [],
                    contract_terms={"monthly_amount": 99_000_000})
proposal = doc("DE_NGHI.pdf", is_proposal=True, proposal_extraction={
    "borrower": {"date_of_birth": "01/01/1980", "dependents": 1, "household_size": 3},
    "loan_request": {"amount": 2_000_000_000, "tenor_months": 360,
                     "interest_rate_pct_year": 10, "repayment_method": "goc_deu_lai_giam_dan"},
    "purpose_financing": {"total_cost": 2_500_000_000, "own_contribution": 500_000_000},
    "declared_income": {"items": [{"monthly_amount": 60_000_000}]},
    "declared_expenses": {"monthly_living_expenses": 6_000_000},
    "existing_debts": {"items": [{"lender": "FE", "product": "vay tiêu dùng", "monthly_payment": 2_000_000}]},
    "collateral": {"items": [{"value": 2_500_000_000}]},
})
cic = doc("CIC.pdf", is_cic_s10a=True, cic_s10a_extraction={
    "du_no_hien_tai": [
        {"tctd": "FE Credit", "khoan_muc": "Dư nợ cho vay ngắn hạn", "vnd": 24_000_000},
        {"tctd": "FE Credit", "khoan_muc": "Tổng cộng", "vnd": 24_000_000},
    ],
    "the_tin_dung": [{"tctd": "VCB", "han_muc": 40_000_000, "du_no": 0}],
})
DOCS = [statement, payslip, rental, labour, proposal, cic]
result = build_affordability(DOCS, "MUA_NHA", today=TODAY)
policy = load_policy()

# --- 2. evidence ranking -------------------------------------------------------
print("\n2. Thứ bậc chứng từ thu nhập")
salary = next(s for s in result.income_sources if s.source_type == "luong")
june = next(m for m in salary.months if m[0] == "06/2026")
check("tháng 06: sao kê (30tr) thắng phiếu lương (45tr)", june[1] == 30_000_000 and june[2] == "sao kê")
july = next(m for m in salary.months if m[0] == "07/2026")
check("tháng 07 chỉ có phiếu lương -> vẫn được dùng", july[1] == 31_000_000)
check("cửa sổ 6 tháng gần nhất", [m[0] for m in salary.months][0] == "02/2026" and len(salary.months) == 6)
check("hợp đồng lao động 99tr KHÔNG thay được thu nhập đã có chứng từ",
      salary.average < 40_000_000, f"bình quân {salary.average:,.0f}")
rent = next(s for s in result.income_sources if s.source_type == "cho_thue")
check("cho thuê chỉ có hợp đồng -> dùng mức hợp đồng, nhãn 'hợp đồng'",
      rent.average == 10_000_000 and rent.evidence == "hợp đồng")
want = salary.average * policy.income_recognition("luong") + 10_000_000 * policy.income_recognition("cho_thue")
check("thu nhập ghi nhận = Σ bình quân × tỷ lệ ghi nhận", close(result.recognised_income, want))
check("có cảnh báo thu nhập chỉ có hợp đồng",
      any("chỉ có" in w and "hợp đồng" in w for w in result.warnings))

# --- 3. obligations ----------------------------------------------------------
print("\n3. Nghĩa vụ trả nợ hiện hữu")
card = 40_000_000 * policy.defaults["card_payment_rate"]
check("lấy số lớn nhất (sao kê 5tr > kê khai 2tr) + thẻ 5% hạn mức",
      close(result.existing_obligations, 5_000_000 + card),
      f"{result.existing_obligations:,.0f}")
check("dư nợ CIC: dòng 'Tổng cộng' không bị cộng hai lần", result.cic_loan_balance == 24_000_000)
check("cảnh báo khoản trả nợ trên sao kê chưa kê khai",
      any("Sao kê cho thấy" in w for w in result.warnings))

# --- 4. living costs ---------------------------------------------------------
print("\n4. Chi phí sinh hoạt")
floor = 2 * policy.defaults["living_cost_per_adult"] + 1 * policy.defaults["living_cost_per_dependant"]
living = row("Chi phí sinh hoạt", result)
check("kê khai 6tr < mức sàn -> dùng mức sàn, nguồn 'chính sách'",
      living.value == floor and living.source == "chính sách", f"{living.value:,.0f}")

# --- 5 & 6. loan sizing ------------------------------------------------------
print("\n5. Số tiền vay tối đa")
table = build_loan_sizing_table(DOCS, result, "MUA_NHA")
limits = policy.limits_for("MUA_NHA")
age = TODAY.year - 1980
tenor_by_age = (limits.max_age_at_maturity - age) * 12
check("thời hạn bị cắt theo tuổi trước khi tính",
      table.tenor_months == min(360, limits.max_tenor_months, tenor_by_age),
      f"{table.tenor_months:g} tháng")
capacity = result.recognised_income * limits.max_dsr - result.existing_obligations
by_dsr = principal_for_payment(capacity, table.tenor_months, 10, "goc_deu_lai_giam_dan")
candidates = {
    "khả năng trả nợ (DSR)": by_dsr,
    "giá trị tài sản bảo đảm (LTV)": 2_500_000_000 * limits.max_ltv,
    "vốn tự có tối thiểu": 2_500_000_000 * (1 - limits.min_own_contribution),
    "số tiền khách hàng đề nghị": 2_000_000_000,
}
binding = min(candidates, key=candidates.get)
check("số tiền tối đa = min các giới hạn", close(table.max_loan, candidates[binding]),
      f"{table.max_loan:,.0f}")
check("nêu đúng giới hạn chặn", table.binding_constraint == binding, table.binding_constraint)
dsr_row = next(r for r in table.rows if r.label == "DSR tại số tiền đề xuất")
if binding.startswith("khả năng"):
    check("DSR chặn -> DSR tại số đề xuất đúng bằng trần", close(dsr_row.value, limits.max_dsr, 1e-6))
else:
    check("DSR tại số đề xuất không vượt trần", dsr_row.value <= limits.max_dsr + 1e-9)

print("\n6. Tuổi khi đáo hạn")
maturity = row("Tuổi khách hàng khi kết thúc", result)
check("tuổi đáo hạn với thời hạn đề nghị", close(maturity.value, age + 30, 1e-6))
check("vượt tuổi tối đa -> 'Vượt ngưỡng'", maturity.status == "Vượt ngưỡng")

# --- 7. strictest limits without a product ----------------------------------
print("\n7. Không nêu sản phẩm -> ngưỡng chặt nhất")
strict = policy.limits_for(None)
check("DSR tối đa = thấp nhất", strict.max_dsr == min(l.max_dsr for l in policy.products.values()))
check("vốn tự có tối thiểu = cao nhất",
      strict.min_own_contribution == max(l.min_own_contribution for l in policy.products.values()
                                         if l.min_own_contribution is not None))
check("LTV bỏ qua sản phẩm tín chấp (null)", strict.max_ltv is not None)

check("trần số tiền/bội số thu nhập của sản phẩm tín chấp không áp cho sản phẩm chưa rõ",
      strict.max_amount is None and strict.max_income_multiple is None)
inferred = build_affordability(DOCS[:-2] + [doc("DE_NGHI2.pdf", program="", is_proposal=True,
    proposal_extraction={**proposal.proposal_extraction,
                         "loan_request": {**proposal.proposal_extraction["loan_request"],
                                          "purpose_category": "mua_o_to"}})],
    None, today=TODAY)
check("không nêu sản phẩm nhưng mục đích 'mua_o_to' -> áp ngưỡng vay mua ô tô, ghi rõ là suy ra",
      inferred.product_id == "MUA_O_TO" and "suy ra" in inferred.product_label, inferred.product_label)
named = build_affordability(DOCS, "TIEU_DUNG", today=TODAY)
check("sản phẩm nêu trong yêu cầu thắng mục đích trên giấy đề nghị", named.product_id == "TIEU_DUNG")

# --- 8. degraded inputs ------------------------------------------------------
print("\n8. Hồ sơ thiếu")
only_declared = build_affordability([proposal], "TIEU_DUNG", today=TODAY)
share = policy.income_recognition("khac")
check("không chứng từ -> kê khai × tỷ lệ thấp nhất, có cảnh báo",
      close(only_declared.recognised_income, 60_000_000 * share)
      and any("Không có chứng từ" in w for w in only_declared.warnings))
empty = build_affordability([], None, today=TODAY)
sizing = build_loan_sizing_table([], empty, None)
check("hồ sơ rỗng không lỗi, báo thiếu dữ liệu",
      sizing.max_loan is None and any("Không đủ dữ liệu" in w for w in sizing.warnings))
check("không có giá trị NaN/inf trong bảng",
      all(r.value is None or math.isfinite(r.value) for r in result.rows + table.rows))

print("\n" + "=" * 66)
if fails:
    print("❌ HỎNG:", *fails, sep="\n   - ")
    sys.exit(1)
print("✅ Bảng khả năng trả nợ và số tiền vay tối đa tính đúng và bảo thủ")
