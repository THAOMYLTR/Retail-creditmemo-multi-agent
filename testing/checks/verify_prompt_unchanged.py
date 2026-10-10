"""Every agent's prompt must be identical, character for character, after a move.

This is the decisive check for a pure refactor: if the string handed to the LLM
does not change by one character, behaviour cannot change either. The baseline is
recorded by this same file (`--record`) before the move and compared after it.

    python testing/checks/verify_prompt_unchanged.py --record   # record baseline
    python testing/checks/verify_prompt_unchanged.py            # compare to it

The input documents are a synthetic retail dossier built in this file rather
than customer files, so the baseline is committable and runs on a fresh machine.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datetime import date as _date  # noqa: E402

from src.agents import affordability_calculator  # noqa: E402
from src.agents.supervisor import Supervisor  # noqa: E402
from src.config import Config  # noqa: E402
from src.matrix.document_matrix import agent_relevance_for_type  # noqa: E402
from src.types import ClassifiedDocument  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "fixtures" / "prompt_baseline.json"

AGENTS = (
    "CUSTOMER_PROFILE_AGENT",
    "INCOME_ANALYSIS_AGENT",
    "CREDIT_RELATIONSHIP_AGENT",
    "CREDIT_PROPOSAL_AGENT",
    "RISK_ASSESSMENT_AGENT",
)

# The affordability table computes the borrower's age from today's date, so the
# prompt would drift every new year. Pinned for the comparison.
FIXED_TODAY = _date(2026, 10, 10)


class _FixedDate(_date):
    @classmethod
    def today(cls):
        return FIXED_TODAY


affordability_calculator.date = _FixedDate


def _doc(type_id: str, filename: str, content: str, **fields) -> ClassifiedDocument:
    doc = ClassifiedDocument(
        path=f"/fixture/{filename}",
        filename=filename,
        content=content,
        agent="RISK_ASSESSMENT_AGENT",
        reasoning="fixture",
        confidence=0.9,
        document_type=type_id,
        loan_program="MUA_NHA",
        agent_relevance=agent_relevance_for_type(type_id, "MUA_NHA"),
    )
    for key, value in fields.items():
        setattr(doc, key, value)
    return doc


def _documents() -> list[ClassifiedDocument]:
    """A synthetic retail dossier touching every prompt block.

    Built in code rather than from customer files so the baseline is
    committable and runs on a fresh machine.
    """

    pad = " x" * 400
    return [
        _doc("giay_to_tuy_than", "CCCD.jpg", "CĂN CƯỚC CÔNG DÂN" + pad),
        _doc("xac_nhan_cu_tru", "CT07.pdf", "XÁC NHẬN THÔNG TIN VỀ CƯ TRÚ" + pad),
        _doc("hop_dong_lao_dong", "HDLD.pdf", "HỢP ĐỒNG LAO ĐỘNG" + pad, is_income=True,
             income_extraction={"document_kind": "hop_dong_lao_dong",
                                "income_owner": {"name": "Nguyễn Văn A"}, "source_unit": "dong",
                                "monthly_income": [],
                                "contract_terms": {"monthly_amount": 38_000_000}}),
        _doc("sao_ke_tai_khoan", "SAO_KE.pdf", "SAO KÊ TÀI KHOẢN" + pad, is_income=True,
             income_extraction={"document_kind": "sao_ke_tai_khoan",
                                "income_owner": {"name": "Nguyễn Văn A"}, "source_unit": "dong",
                                "monthly_income": [
                                    {"month": f"{m:02d}/2026", "source_type": "luong",
                                     "amount": 34_000_000 + m * 250_000, "page": 1}
                                    for m in range(1, 7)],
                                "recurring_debits": [
                                    {"description": "Trả góp công ty tài chính",
                                     "monthly_amount": 3_500_000, "months_seen": 6}]}),
        _doc("de_nghi_vay_von", "DE_NGHI.pdf", "GIẤY ĐỀ NGHỊ VAY VỐN" + pad, is_proposal=True,
             proposal_extraction={
                 "borrower": {"name": "Nguyễn Văn A", "date_of_birth": "15/04/1986",
                              "dependents": 1, "household_size": 3},
                 "loan_request": {"purpose_category": "mua_nha", "amount": 1_800_000_000,
                                  "tenor_months": 300, "interest_rate_pct_year": 9.2,
                                  "repayment_method": "goc_deu_lai_giam_dan"},
                 "purpose_financing": {"total_cost": 2_600_000_000,
                                       "own_contribution": 800_000_000},
                 "declared_income": {"items": [{"monthly_amount": 45_000_000}]},
                 "declared_expenses": {"monthly_living_expenses": 15_000_000},
                 "existing_debts": {"items": []},
                 "collateral": {"items": [{"value": 2_600_000_000,
                                           "is_purchase_asset": True}]}}),
        _doc("hop_dong_mua_ban_nha_dat", "HDMB.pdf", "HỢP ĐỒNG MUA BÁN CĂN HỘ" + pad),
        _doc("cic_khach_hang_vay", "CIC.pdf", "BÁO CÁO CIC" + pad, is_cic_s10a=True,
             cic_s10a_extraction={
                 "du_no_hien_tai": [{"tctd": "Công ty tài chính X",
                                     "khoan_muc": "Dư nợ cho vay trung hạn",
                                     "vnd": 60_000_000}],
                 "the_tin_dung": [{"tctd": "Ngân hàng Y", "han_muc": 50_000_000,
                                   "du_no": 12_000_000}],
                 "du_no_12_thang": [{"thang": f"{m:02d}/2026",
                                     "tong_du_no": 80_000_000 - m * 2_000_000}
                                    for m in range(1, 7)]}),
        _doc("bao_cao_tham_dinh", "THAM_DINH.pdf", "BÁO CÁO THẨM ĐỊNH" + pad,
             is_sitevisit=True,
             sitevisit_extraction={
                 "survey_info": {"survey_date": "2026-09-30"},
                 "residence_verification": {"address": "Hà Nội", "matches_documents": True},
                 "employment_verification": {"employer": "Công ty ABC",
                                             "verification_method": "gọi điện"},
                 "business_verification": {},
                 "collateral_verification": {"description": "Căn hộ 2PN"},
                 "conclusion": {"recommendation": "Cho vay 1,6 tỷ"}}),
    ]


def _prompts() -> dict[str, str]:
    supervisor = Supervisor(Config())
    docs = _documents()
    return {
        agent: supervisor._build_user_input(
            "Lập tờ trình thẩm định cho khách hàng vay mua nhà", docs, "", agent, "", {}
        )
        for agent in AGENTS
    }


def main() -> int:
    prompts = _prompts()
    digests = {a: hashlib.sha256(p.encode()).hexdigest() for a, p in prompts.items()}

    if "--record" in sys.argv:
        BASELINE.write_text(
            json.dumps({"sha256": digests, "lengths": {a: len(p) for a, p in prompts.items()}},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Đã chụp mốc cho {len(digests)} agent -> {BASELINE.name}")
        for agent, digest in digests.items():
            print(f"   {agent:28} {len(prompts[agent]):>7,} ký tự  {digest[:16]}")
        return 0

    if not BASELINE.is_file():
        raise FileNotFoundError(BASELINE)

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    failures = []
    for agent in AGENTS:
        want, got = baseline["sha256"].get(agent), digests[agent]
        want_len, got_len = baseline["lengths"].get(agent), len(prompts[agent])
        if want == got:
            print(f"   ✅ {agent:28} {got_len:>7,} ký tự — giống hệt")
        else:
            failures.append(agent)
            print(f"   ❌ {agent:28} {want_len:>7,} -> {got_len:>7,} ký tự — ĐÃ ĐỔI")

    print()
    if failures:
        print("Prompt đã đổi:", ", ".join(failures))
        print("Một refactor thuần tuý không được đổi chuỗi nào. Nếu thay đổi là cố ý,")
        print("chạy lại với --record và nói rõ trong commit vì sao prompt đổi.")
        return 1
    print("✅ Prompt của cả 5 agent không đổi một ký tự nào")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
