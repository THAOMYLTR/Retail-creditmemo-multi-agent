"""Matrix, policy and request routing agree with each other.

Three files have to stay in step and nothing else enforces it:

- src/matrix/document_matrix.yaml names the loan products (columns) and which
  agent each document reaches;
- src/policy/retail_products.yaml holds each product's limits, keyed by the same
  product ids;
- supervisor.py routes a request to an agent by its keywords.

A product added to one YAML and not the other would silently fall back to the
strictest limits; a document type no agent requires would never be anyone's core
evidence; a Vietnamese request phrased the usual way landing on the wrong agent
would produce the wrong report with nothing in the output saying so.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.agents.document_classification import detect_loan_program  # noqa: E402
from src.agents.supervisor import Supervisor  # noqa: E402
from src.config import Config  # noqa: E402
from src.matrix.document_matrix import (  # noqa: E402
    agent_relevance_for_type,
    is_income_type,
    load_matrix,
)
from src.policy.retail_policy import load_policy  # noqa: E402
from src.types import SPECIALIST_DOCUMENT_AGENTS  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'✅' if ok else '❌'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


matrix = load_matrix()
policy = load_policy()

print("1. Ma trận hồ sơ và chính sách sản phẩm khớp nhau")
check("cùng bộ mã sản phẩm",
      set(matrix.loan_programs) == set(policy.products),
      f"matrix={sorted(matrix.loan_programs)} policy={sorted(policy.products)}")
for agent in sorted(SPECIALIST_DOCUMENT_AGENTS):
    required = [t for t in matrix.types if agent_relevance_for_type(t).get(agent) == "R"]
    check(f"{agent}: có loại hồ sơ bắt buộc", bool(required), f"{len(required)} loại")
income_types = [t for t in matrix.types if is_income_type(t)]
check("mọi loại chứng từ thu nhập đều tới INCOME_ANALYSIS_AGENT",
      all("INCOME_ANALYSIS_AGENT" in agent_relevance_for_type(t) for t in income_types),
      f"{len(income_types)} loại")
for product in matrix.loan_programs:
    relevance = agent_relevance_for_type("giay_chung_nhan_qsdd", product)
    want = "R" if product == "MUA_NHA" else "O"
    check(f"sổ đỏ với {product}: {want}", relevance.get("CREDIT_PROPOSAL_AGENT") == want)

print("\n2. Nhận diện sản phẩm vay trong câu hỏi")
for query, expected in [
    ("Lập tờ trình cho khách hàng vay mua nhà", "MUA_NHA"),
    ("Thẩm định khoản vay mua căn hộ chung cư", "MUA_NHA"),
    ("Khách vay mua ô tô 7 chỗ", "MUA_O_TO"),
    ("Đánh giá hồ sơ vay tín chấp", "TIEU_DUNG"),
    ("Hộ kinh doanh tạp hoá cần vay vốn", "HO_KINH_DOANH"),
    ("Phân tích khả năng trả nợ", None),
    ("So sánh vay mua nhà với vay mua ô tô", None),
]:
    got = detect_loan_program(query)["program"]
    check(f"{query!r} -> {expected}", got == expected, f"nhận {got}")

print("\n3. Định tuyến yêu cầu tới agent")
sup = Supervisor(Config())
for query, expected in [
    ("Lập tờ trình thẩm định cho khách hàng này", "CREDIT_MEMO"),
    ("Tạo báo cáo thẩm định", "CREDIT_MEMO"),
    ("Khách hàng được vay tối đa bao nhiêu?", "CREDIT_PROPOSAL_AGENT"),
    ("Đề xuất khoản vay và tính LTV", "CREDIT_PROPOSAL_AGENT"),
    ("Đánh giá rủi ro tín dụng", "RISK_ASSESSMENT_AGENT"),
    ("Kiểm tra lịch sử tín dụng CIC và thẻ tín dụng", "CREDIT_RELATIONSHIP_AGENT"),
    ("Phân tích thu nhập và khả năng trả nợ", "INCOME_ANALYSIS_AGENT"),
    ("Tính DSR cho khách hàng", "INCOME_ANALYSIS_AGENT"),
    ("Tóm tắt thông tin nhân thân và nơi cư trú", "CUSTOMER_PROFILE_AGENT"),
]:
    got = sup._deterministic_route(query)
    check(f"{query!r} -> {expected}", got == expected, f"nhận {got}")
    mode = sup._workflow_mode_for_route(got or "")
    check("   có nhánh đồ thị tương ứng",
          mode in {"single_customer_profile", "single_credit_relationship",
                   "single_income_analysis", "single_risk_assessment",
                   "single_credit_proposal", "full_credit_memo"}, mode)

print("\n4. Thiếu chứng từ thu nhập thì dừng trước khi tốn LLM")
gap = sup._analyze_evidence_gaps([], "CREDIT_MEMO")
check("tờ trình không có chứng từ thu nhập -> không chạy",
      gap["can_proceed"] is False
      and any(m["type"] == "income_documents" for m in gap["missing_evidence"]))
gap = sup._analyze_evidence_gaps([], "CREDIT_RELATIONSHIP_AGENT")
check("quan hệ tín dụng không bắt buộc chứng từ thu nhập", gap["can_proceed"] is True)

print("\n" + "=" * 66)
if fails:
    print("❌ HỎNG:", *fails, sep="\n   - ")
    sys.exit(1)
print("✅ Ma trận, chính sách và định tuyến nhất quán")
