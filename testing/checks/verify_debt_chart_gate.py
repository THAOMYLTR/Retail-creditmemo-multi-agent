"""Verify the debt/income chart only appears when CREDIT_RELATIONSHIP_AGENT ran.

The income line is read straight from the proof-of-income extraction (the same
monthly series the affordability calculator uses), so the title must promise the
income line exactly when there is one to draw.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.agents.supervisor import Supervisor
from src.types import ClassifiedDocument

CIC_DOC = ClassifiedDocument(
    filename="CIC_KHCN.pdf",
    path="/tmp/CIC_KHCN.pdf",
    content="",
    agent="CREDIT_RELATIONSHIP_AGENT",
    reasoning="",
    confidence=0.9,
    is_cic_s10a=True,
    cic_s10a_extraction={
        "du_no_12_thang": [
            {"thang": "01/2026", "tong_du_no": 500_000_000},
            {"thang": "02/2026", "tong_du_no": 480_000_000},
            {"thang": "03/2026", "tong_du_no": 470_000_000},
        ]
    },
)

INCOME_DOC = ClassifiedDocument(
    filename="SAO_KE_LUONG.pdf",
    path="/tmp/SAO_KE_LUONG.pdf",
    content="",
    agent="INCOME_ANALYSIS_AGENT",
    reasoning="",
    confidence=0.9,
    is_income=True,
    income_extraction={
        "document_kind": "sao_ke_tai_khoan",
        "income_owner": {"name": "Nguyen Van A"},
        "source_unit": "dong",
        "monthly_income": [
            {"month": "01/2026", "source_type": "luong", "amount": 30_000_000},
            {"month": "02/2026", "source_type": "luong", "amount": 31_000_000},
            {"month": "03/2026", "source_type": "luong", "amount": 32_000_000},
        ],
    },
)

# A report with no credit-relationship heading — what PF/IA/CP/RISK produce.
PLAIN_REPORT = "# Bao cao\n\n## 1. Tong quan\n\nnoi dung"

MEMO = {
    "CUSTOMER_PROFILE_AGENT": "x",
    "CREDIT_RELATIONSHIP_AGENT": "x",
    "INCOME_ANALYSIS_AGENT": "x",
    "CREDIT_PROPOSAL_AGENT": "x",
    "RISK_ASSESSMENT_AGENT": "x",
    "CREDIT_MEMO": "x",
}

FLOWS = [
    ("single CUSTOMER_PROFILE", {"CUSTOMER_PROFILE_AGENT": "x"}, False),
    ("single INCOME_ANALYSIS", {"INCOME_ANALYSIS_AGENT": "x"}, False),
    ("single CREDIT_PROPOSAL", {"CREDIT_PROPOSAL_AGENT": "x"}, False),
    ("single RISK_ASSESSMENT", {"RISK_ASSESSMENT_AGENT": "x"}, False),
    ("single CREDIT_RELATIONSHIP", {"CREDIT_RELATIONSHIP_AGENT": "x"}, True),
    ("full CREDIT_MEMO", MEMO, True),
]

failures = []

print(f"{'FLOW':29} {'CHART?':8} {'EXPECTED':10} RESULT")
print("-" * 62)
for name, subs, should_build in FLOWS:
    block, title = Supervisor._build_debt_chart_block([CIC_DOC, INCOME_DOC], subs)
    built = bool(block)
    ok = built == should_build
    if not ok:
        failures.append(f"{name}: built={built}, expected={should_build}")
    print(
        f"{name:29} {('YES' if built else 'no'):8} "
        f"{('YES' if should_build else 'no'):10} {'PASS' if ok else 'FAIL'}"
    )
    if not built and title:
        failures.append(f"{name}: no block but title={title!r}")

print("\n--- anchor placement (chart must land in its section) ---")
block, title = Supervisor._build_debt_chart_block(
    [CIC_DOC, INCOME_DOC], {"CREDIT_RELATIONSHIP_AGENT": "x"}
)
for label, report, expected in [
    ("CR report (### 2.5 Dien bien du no)", "# CR\n\n### 2.5 Diễn biến dư nợ\n\nx", "dien bien du no"),
    ("Memo (## 3. Quan hệ tín dụng)", "# Memo\n\n## 3. Quan hệ tín dụng\n\nx", "quan he tin dung"),
]:
    _, anchor = Supervisor._insert_debt_chart(report, block, title)
    ok = anchor == expected
    if not ok:
        failures.append(f"{label}: anchor={anchor!r}, expected={expected!r}")
    print(f"{label:38} -> {anchor:20} {'PASS' if ok else 'FAIL'}")

print("\n--- title tracks the data actually drawn ---")
for label, documents, want_income in [
    ("income documents present", [CIC_DOC, INCOME_DOC], True),
    ("no income documents", [CIC_DOC], False),
]:
    block, title = Supervisor._build_debt_chart_block(
        documents, {"CREDIT_RELATIONSHIP_AGENT": "x"}
    )
    columns = next(l for l in block.splitlines() if l.startswith("columns:"))
    embedded = next(l for l in block.splitlines() if l.startswith("title:"))
    unit = next(l for l in block.splitlines() if l.startswith("unit:"))
    has_income_col = "Thu nhập" in columns
    income_in_title = "thu nhập" in title
    ok = has_income_col == want_income and income_in_title == want_income
    ok = ok and embedded == f"title: {title}"
    # Retail balances and salaries both read naturally in triệu.
    ok = ok and unit == "unit: triệu VNĐ"
    if not ok:
        failures.append(f"{label}: title={title!r}, {columns}, {unit}")
    print(f"{label:32} {'PASS' if ok else 'FAIL'}")
    print(f"    title:   {title}")
    print(f"    {columns}")

    text, anchor = Supervisor._insert_debt_chart(PLAIN_REPORT, block, title)
    assert anchor == "appended"
    if f"## {title}" not in text:
        failures.append(f"{label}: fallback heading missing {title!r}")

print("\n--- income figures are the extracted ones, not re-typed ---")
block, _ = Supervisor._build_debt_chart_block(
    [CIC_DOC, INCOME_DOC], {"CREDIT_RELATIONSHIP_AGENT": "x"}
)
row = next(l for l in block.splitlines() if l.startswith("02/2026"))
ok = row == "02/2026 | 480,00 | 31,00"
print(f"{row:32} {'PASS' if ok else 'FAIL'}")
if not ok:
    failures.append(f"02/2026 row: {row!r}")

print()
if failures:
    print("FAILURES:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("All checks passed.")
