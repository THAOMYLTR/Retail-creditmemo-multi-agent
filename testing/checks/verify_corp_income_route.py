"""Verify the corp_income route inside the supervisor graph.

- A corporate request goes straight from discover_documents to the corp_income
  node: no retail classification, no retail evidence-gap check.
- Retail requests (including ones mentioning taxes) never take that route.
- process(..., corp_income_options=...) reaches CorpIncomeAgent.run.
"""

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.agents.corp_income import CorpIncomeAgent
from src.agents.supervisor import Supervisor
from src.config import Config

failures = []


def expect(condition, message):
    print(("   ✓ " if condition else "   ✗ ") + message)
    if not condition:
        failures.append(message)


print("1. Nhận diện yêu cầu doanh nghiệp")
corp_queries = [
    "Kiểm tra pháp lý doanh nghiệp cho hồ sơ này",
    "phap ly doanh nghiep",
    "Corp income assessment mục 1-4",
    "Trích xuất thành viên góp vốn và người đại diện theo pháp luật",
    "Hồ sơ KHDN công ty ABC",
]
retail_queries = [
    "Lập tờ trình cho khách hàng vay mua nhà",
    "Phân tích khả năng trả nợ",
    "Khách hàng có nợ thuế TNCN không?",
    "Đánh giá rủi ro tín dụng",
    "Xin chào",
]
for q in corp_queries:
    expect(Supervisor._is_corp_income_request(q), f"corp: {q!r}")
for q in retail_queries:
    expect(not Supervisor._is_corp_income_request(q), f"retail: {q!r}")

print("\n2. Route → workflow_mode → node")
expect(Supervisor._workflow_mode_for_route("CORP_INCOME_AGENT") == "single_corp_income", "mode single_corp_income")
expect(Supervisor._passes_needed_for_route("CORP_INCOME_AGENT") == set(), "không chạy pass trích xuất bán lẻ nào")

print("\n3. Chạy qua graph")
seen = []


class FakeChain:
    def __init__(self, name):
        self.name = name

    def invoke(self, payload):
        seen.append((payload["filename"], self.name))
        if self.name == "legal_basic":
            return {"company_name": "CÔNG TY TNHH ABC", "tax_code": "0101234567",
                    "legal_representatives": ["Trần Thị B"], "operating_status": "dang_hoat_dong",
                    "source_unit": "dong", "change_history": []}
        if self.name == "shareholders":
            return {"members": [{"name": "Nguyễn Văn A", "member_type": "ca_nhan", "ownership_pct": 100,
                                 "acquisition_date": "12/05/2015", "acquisition_method": "gop_von_thanh_lap"}],
                    "source_unit": "dong"}
        if self.name == "legal_rep":
            return {"legal_representatives": [{"name": "Trần Thị B", "appointed_date": "12/05/2015"}]}
        return {"tax_debts": [], "social_insurance_debts": [], "administrative_penalties": [], "disputes": []}


sup = Supervisor(Config())
sup.corp_income_agent = CorpIncomeAgent(chains={p: FakeChain(p) for p in
                                                ("legal_basic", "shareholders", "legal_rep", "liabilities")})
with tempfile.TemporaryDirectory() as folder:
    Path(folder, "GCN_DKDN_ABC.txt").write_text("GIẤY CHỨNG NHẬN ĐĂNG KÝ DOANH NGHIỆP", encoding="utf-8")
    out = sup.process(
        "Kiểm tra pháp lý doanh nghiệp",
        [folder],
        corp_income_options={"loan_date": "13/10/2026", "borrower_names": ["Nguyễn Văn A"]},
    )
expect(out["agent_name"] == "CORP_INCOME_AGENT", f"agent_name = {out['agent_name']}")
expect(out["decision"].get("route") == "CORP_INCOME_AGENT", "decision.route")
expect("corp_income" in out and out["corp_income"]["meta"]["as_of"] == "13/10/2026", "loan_date tới được agent")
expect(out["response"].startswith("# Corp Income Assessment"), "response là báo cáo corp_income")
expect(not any(s.startswith("Loan program") for s in out["steps"]), "không chạy bộ phân loại bán lẻ")
expect(not any("evidence gap" in s.lower() for s in out["steps"]), "không chạy kiểm tra thiếu hồ sơ bán lẻ")
expect(len(seen) == 3, f"GCN chạy 3 pass ({len(seen)})")

print("\n4. Tùy chọn sai → trả lỗi, không sập")
bad = sup.process("pháp lý doanh nghiệp", [], corp_income_options={"internal_data": {"not_a_pass": {}}})
expect(bad["status"] == "error" and "corp_income" in bad["response"], f"status = {bad['status']}")

if failures:
    print(f"\n❌ {len(failures)} lỗi")
    sys.exit(1)
print("\n✅ corp_income route OK")
