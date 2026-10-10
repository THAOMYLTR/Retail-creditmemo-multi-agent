"""Deterministic repayment-capacity (affordability) table for a retail borrower.

Retail underwriting turns on a handful of numbers — how much income reliably
arrives each month, what the household must spend to live, what it already pays
other lenders, and what the new loan would cost per month. Left to a model those
are a dozen chances to produce a plausible wrong number, and the DSR built on
them decides the loan. So they are computed here and handed to the agents as
data, the same division of labour the corporate version had for financial
ratios.

Three rules drive the design.

**Evidence beats declaration, and the stronger evidence wins.** The same salary
can appear on the labour contract, the employer's confirmation and the bank
statement. For each income source and month the figure from the strongest kind
of document is kept (money actually credited to the account outranks a payslip,
which outranks an agreed contract figure); the loan application's own declared
income is used only when nothing evidences it, and is labelled so.

**Every row says where it came from.** Living-cost floors, the card payment rate
and the stress interest rate are policy defaults, not facts about the customer.
A reviewer must be able to tell "chính sách" from "sao kê" at a glance —
printing both the same way would be presenting an assumption as evidence.

**When in doubt, the conservative figure.** Existing obligations take the
largest of what the customer declared, what the bank statement shows being paid
out, and what the CIC balances imply. The new instalment is the first (largest)
payment of a declining-balance schedule. Recognised income discounts sources
that are harder to verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from src.agents.cic_s10a_extraction import month_sort_key
from src.policy.retail_policy import ProductLimits, RetailPolicy, load_policy
from src.types import ClassifiedDocument
from src.utils.common import normalize_text
from src.utils.formatting import format_vn_number, to_million_vnd

# Source labels, written into the "Nguồn" column the agents must repeat.
SRC_STATEMENT = "sao kê"
SRC_PAYSLIP = "xác nhận lương"
SRC_CONTRACT = "hợp đồng"
SRC_TAX = "thuế TNCN"
SRC_BUSINESS = "sổ sách HKD"
SRC_OTHER_DOC = "chứng từ thu nhập"
SRC_APPRAISAL = "thẩm định"
SRC_PROPOSAL = "đề nghị vay"
SRC_CIC = "CIC"
SRC_POLICY = "chính sách"
SRC_DERIVED = "tính toán"

# How much each kind of proof is trusted for a monthly figure. Money that
# actually reached the account beats what an employer says it pays, which beats
# what a contract says it will pay.
EVIDENCE_RANK: dict[str, int] = {
    "sao_ke_tai_khoan": 5,
    "xac_nhan_luong": 4,
    "huu_tri": 4,
    "ho_kinh_doanh": 3,
    "tien_gui": 3,
    "khac": 2,
    "thue_tncn": 1,
    "hop_dong_lao_dong": 0,
    "hop_dong_cho_thue": 0,
}
EVIDENCE_LABEL: dict[str, str] = {
    "sao_ke_tai_khoan": SRC_STATEMENT,
    "xac_nhan_luong": SRC_PAYSLIP,
    "huu_tri": SRC_OTHER_DOC,
    "ho_kinh_doanh": SRC_BUSINESS,
    "tien_gui": SRC_OTHER_DOC,
    "khac": SRC_OTHER_DOC,
    "thue_tncn": SRC_TAX,
    "hop_dong_lao_dong": SRC_CONTRACT,
    "hop_dong_cho_thue": SRC_CONTRACT,
}
SOURCE_TYPE_LABEL: dict[str, str] = {
    "luong": "Lương, thưởng, phụ cấp",
    "huu_tri": "Lương hưu, trợ cấp",
    "cho_thue": "Cho thuê tài sản",
    "kinh_doanh": "Kinh doanh",
    "khac": "Thu nhập khác",
}
# Contract documents carry the agreed figure in contract_terms; this is the
# income type that figure belongs to.
CONTRACT_SOURCE_TYPE = {
    "hop_dong_lao_dong": "luong",
    "hop_dong_cho_thue": "cho_thue",
}
# The application's purpose category, mapped onto a product id of the policy,
# for requests that name no product. Only used when the request itself is
# silent, and the table says the product was inferred.
PURPOSE_TO_PRODUCT = {
    "mua_nha": "MUA_NHA",
    "xay_sua_nha": "MUA_NHA",
    "mua_o_to": "MUA_O_TO",
    "tieu_dung": "TIEU_DUNG",
    "kinh_doanh": "HO_KINH_DOANH",
}
REPAYMENT_METHOD_LABEL = {
    "goc_deu_lai_giam_dan": "gốc đều, lãi giảm dần",
    "tra_gop_deu": "trả góp đều (gốc + lãi)",
    "goc_cuoi_ky": "trả gốc cuối kỳ",
}


@dataclass
class Row:
    """One line of a calculator table: a value, its unit, and where it came from."""

    label: str
    unit: str
    value: float | None = None
    source: str = ""
    note: str = ""
    # Optional policy limit the value is judged against, and the verdict.
    limit: float | None = None
    status: str = ""

    def __post_init__(self) -> None:
        for attr in ("value", "limit"):
            number = getattr(self, attr)
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                object.__setattr__(self, attr, round(float(number), 4))

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "unit": self.unit,
            "value": self.value,
            "limit": self.limit,
            "status": self.status,
            "source": self.source,
            "note": self.note,
        }


@dataclass
class IncomeSource:
    """One income source after evidence de-duplication."""

    source_type: str
    months: list[tuple[str, float, str, str]]  # (month, amount, evidence label, filename)
    average: float | None
    minimum: float | None
    recognition: float
    recognised: float | None
    evidence: str
    files: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class AffordabilityResult:
    product_label: str = ""
    # The product whose limits were applied: the one the request named, else
    # the one the application's purpose implies, else "" (strictest limits).
    product_id: str = ""
    rows: list[Row] = field(default_factory=list)
    income_sources: list[IncomeSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Facts the loan-sizing calculator builds on. Kept as plain values so the
    # second table is computed from exactly what the first one showed.
    recognised_income: float | None = None
    living_costs: float | None = None
    existing_obligations: float | None = None
    new_payment: float | None = None
    loan_amount: float | None = None
    tenor_months: float | None = None
    rate_pct_year: float | None = None
    repayment_method: str = ""
    borrower_age: float | None = None
    cic_loan_balance: float | None = None
    # Evidenced income per month, every source summed, before recognition —
    # the series the debt/income chart draws.
    monthly_income_series: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "product": self.product_label,
            "product_id": self.product_id,
            "rows": [row.as_dict() for row in self.rows],
            "income_sources": [
                {
                    "source_type": s.source_type,
                    "months": [
                        {"month": m, "amount": a, "evidence": e, "file": f}
                        for m, a, e, f in s.months
                    ],
                    "average": s.average,
                    "minimum": s.minimum,
                    "recognition": s.recognition,
                    "recognised": s.recognised,
                    "evidence": s.evidence,
                    "files": s.files,
                    "note": s.note,
                }
                for s in self.income_sources
            ],
            "warnings": self.warnings,
            "recognised_income": self.recognised_income,
            "living_costs": self.living_costs,
            "existing_obligations": self.existing_obligations,
            "new_payment": self.new_payment,
            "loan_amount": self.loan_amount,
            "tenor_months": self.tenor_months,
            "rate_pct_year": self.rate_pct_year,
            "repayment_method": self.repayment_method,
            "borrower_age": self.borrower_age,
            "cic_loan_balance": self.cic_loan_balance,
            "monthly_income_series": self.monthly_income_series,
        }


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def money(value: float | None) -> str:
    """Đồng amount for a note, on the memo's display scale (triệu VNĐ)."""

    return "" if value is None else to_million_vnd(value)


def pct(value: float | None, decimals: int = 1) -> str:
    """A 0-1 ratio as a Vietnamese percentage for a note or warning."""

    return "" if value is None else f"{format_vn_number(value * 100, decimals)}%"


def num(value: Any) -> float | None:
    """A finite number, or None. Booleans are not numbers here."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _sum(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def monthly_payment(
    principal: float | None,
    tenor_months: float | None,
    rate_pct_year: float | None,
    method: str,
) -> float | None:
    """The instalment used for the DSR: the largest payment of the schedule.

    For declining-balance repayment that is the first one (full principal share
    plus a month's interest on the whole amount); for an annuity every payment
    is equal; for bullet principal it is the monthly interest alone. Unknown
    methods are treated as declining-balance, which gives the highest first
    payment of the three and therefore the most conservative DSR.
    """

    if principal is None or principal <= 0 or not tenor_months or tenor_months <= 0:
        return None
    rate = (rate_pct_year or 0.0) / 100 / 12
    if method == "tra_gop_deu":
        if rate == 0:
            return principal / tenor_months
        return principal * rate / (1 - (1 + rate) ** (-tenor_months))
    if method == "goc_cuoi_ky":
        return principal * rate
    return principal / tenor_months + principal * rate


def principal_for_payment(
    payment: float | None,
    tenor_months: float | None,
    rate_pct_year: float | None,
    method: str,
) -> float | None:
    """Inverse of monthly_payment: the largest principal a payment can carry."""

    if payment is None or payment <= 0 or not tenor_months or tenor_months <= 0:
        return None
    rate = (rate_pct_year or 0.0) / 100 / 12
    if method == "tra_gop_deu":
        if rate == 0:
            return payment * tenor_months
        return payment * (1 - (1 + rate) ** (-tenor_months)) / rate
    if method == "goc_cuoi_ky":
        return payment / rate if rate else None
    return payment / (1 / tenor_months + rate)


def parse_birth_year(text: Any) -> int | None:
    """Year out of a dd/mm/yyyy (or any) date string, or None."""

    digits = "".join(ch if ch.isdigit() else " " for ch in str(text or "")).split()
    for run in reversed(digits):
        if len(run) == 4 and 1900 <= int(run) <= 2100:
            return int(run)
    return None


# --------------------------------------------------------------------------
# Readers over the extracted records
# --------------------------------------------------------------------------


def first_extraction(documents: list[ClassifiedDocument], attr: str, flag: str) -> dict[str, Any] | None:
    return next(
        (
            getattr(doc, attr)
            for doc in documents
            if getattr(doc, flag) and isinstance(getattr(doc, attr), dict)
        ),
        None,
    )


def _block(record: dict[str, Any] | None, name: str) -> dict[str, Any]:
    block = (record or {}).get(name)
    return block if isinstance(block, dict) else {}


def cic_loan_rows(cic_records: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    """(lender, bucket, balance) per lender, totals winning over detail lines.

    Each lender's block may carry both per-facility lines and a "Tổng cộng"
    line; summing everything would count that lender twice. The bucket is
    "ngan_han" or "dai_han" and only matters for the assumed remaining term.
    """

    per_lender: dict[str, dict[str, Any]] = {}
    for record in cic_records:
        for row in (record or {}).get("du_no_hien_tai") or []:
            if not isinstance(row, dict):
                continue
            amount = num(row.get("vnd"))
            if amount is None:
                continue
            lender = str(row.get("tctd") or "").strip() or "(không rõ TCTD)"
            item = normalize_text(str(row.get("khoan_muc") or ""))
            bucket = per_lender.setdefault(
                lender, {"total": 0.0, "short": 0.0, "long": 0.0}
            )
            if "tong cong" in item:
                bucket["total"] = max(bucket["total"], amount)
            elif "ngan han" in item:
                bucket["short"] += amount
            else:
                bucket["long"] += amount

    rows: list[tuple[str, str, float]] = []
    for lender, bucket in per_lender.items():
        details = bucket["short"] + bucket["long"]
        if bucket["total"] and not details:
            # Only a total was printed: no way to tell the term, so assume the
            # longer one — a smaller assumed payment, flagged as an assumption.
            rows.append((lender, "dai_han", bucket["total"]))
            continue
        if bucket["short"]:
            rows.append((lender, "ngan_han", bucket["short"]))
        if bucket["long"]:
            rows.append((lender, "dai_han", bucket["long"]))
    return rows


def cic_cards(cic_records: list[dict[str, Any]]) -> list[tuple[str, float | None, float | None]]:
    """(issuer, limit, balance) for every card in the CIC report(s)."""

    cards = []
    for record in cic_records:
        for row in (record or {}).get("the_tin_dung") or []:
            if not isinstance(row, dict):
                continue
            limit, balance = num(row.get("han_muc")), num(row.get("du_no"))
            if limit is None and balance is None:
                continue
            cards.append((str(row.get("tctd") or "").strip(), limit, balance))
    return cards


def _declares_card(proposal: dict[str, Any] | None) -> bool:
    for item in _block(proposal, "existing_debts").get("items") or []:
        if not isinstance(item, dict):
            continue
        tokens = normalize_text(
            f"{item.get('product') or ''} {item.get('lender') or ''}"
        ).split()
        if "the" in tokens or "card" in tokens:
            return True
    return False


# --------------------------------------------------------------------------
# Income
# --------------------------------------------------------------------------


def collect_income_sources(
    documents: list[ClassifiedDocument],
    policy: RetailPolicy,
    sitevisit: dict[str, Any] | None,
) -> tuple[list[IncomeSource], dict[str, float], list[str]]:
    """Evidence-ranked income per source, the monthly series, and warnings."""

    warnings: list[str] = []
    best: dict[tuple[str, str], tuple[int, float, str, str]] = {}
    contract_terms: dict[str, list[tuple[float, str]]] = {}
    annual: dict[str, list[tuple[float, str, str]]] = {}

    for doc in documents:
        record = doc.income_extraction if doc.is_income else None
        if not isinstance(record, dict):
            continue
        kind = str(record.get("document_kind") or "khac")
        rank = EVIDENCE_RANK.get(kind, 2)
        label = EVIDENCE_LABEL.get(kind, SRC_OTHER_DOC)
        for row in record.get("monthly_income") or []:
            if not isinstance(row, dict):
                continue
            amount = num(row.get("amount"))
            month = str(row.get("month") or "")
            if amount is None or amount <= 0 or month_sort_key(month)[0] == 9999:
                continue
            key = (str(row.get("source_type") or "khac"), month)
            current = best.get(key)
            # Stronger evidence wins; at equal strength keep the LOWER figure,
            # because two documents of the same rank disagreeing is exactly the
            # case where the conservative reading is wanted.
            if (
                current is None
                or rank > current[0]
                or (rank == current[0] and amount < current[1])
            ):
                best[key] = (rank, amount, label, doc.filename)

        terms = record.get("contract_terms")
        if isinstance(terms, dict) and kind in CONTRACT_SOURCE_TYPE:
            agreed = _sum([num(terms.get("monthly_amount")), num(terms.get("monthly_allowances"))])
            if agreed:
                contract_terms.setdefault(CONTRACT_SOURCE_TYPE[kind], []).append(
                    (agreed, doc.filename)
                )
        for row in record.get("annual_income") or []:
            if isinstance(row, dict) and num(row.get("amount")):
                annual.setdefault(str(row.get("source_type") or "luong"), []).append(
                    (float(row["amount"]), str(row.get("year") or ""), doc.filename)
                )

    lookback = int(policy.defaults.get("income_lookback_months", 6))
    min_months = int(policy.defaults.get("min_income_months", 3))
    drop_alert = float(policy.defaults.get("income_drop_alert", 0.3))

    by_type: dict[str, list[tuple[str, float, str, str]]] = {}
    for (source_type, month), (_, amount, label, filename) in best.items():
        by_type.setdefault(source_type, []).append((month, amount, label, filename))

    sources: list[IncomeSource] = []
    series: dict[str, float] = {}
    for source_type, rows in by_type.items():
        rows.sort(key=lambda r: month_sort_key(r[0]))
        for month, amount, _, _ in rows:
            series[month] = series.get(month, 0.0) + amount
        recent = rows[-lookback:]
        amounts = [r[1] for r in recent]
        average = sum(amounts) / len(amounts)
        minimum = min(amounts)
        evidence = ", ".join(sorted({r[2] for r in recent}))
        files = sorted({r[3] for r in recent})
        notes = []
        if len(recent) < min_months:
            notes.append(f"chỉ có {len(recent)} tháng chứng từ")
            warnings.append(
                f"Thu nhập \"{SOURCE_TYPE_LABEL.get(source_type, source_type)}\" chỉ "
                f"có {len(recent)} tháng chứng từ, ít hơn {min_months} tháng theo "
                "chính sách — mức bình quân chưa đủ tin cậy."
            )
        low = [m for m, a, _, _ in recent if a < average * (1 - drop_alert)]
        if low:
            notes.append("có tháng giảm mạnh: " + ", ".join(low))
            warnings.append(
                f"Thu nhập \"{SOURCE_TYPE_LABEL.get(source_type, source_type)}\" "
                f"giảm hơn {pct(drop_alert, 0)} so với bình quân ở tháng "
                + ", ".join(low)
                + " — thu nhập không ổn định."
            )
        share = policy.income_recognition(source_type)
        sources.append(
            IncomeSource(
                source_type=source_type,
                months=recent,
                average=average,
                minimum=minimum,
                recognition=share,
                recognised=average * share,
                evidence=evidence,
                files=files,
                note="; ".join(notes),
            )
        )

    covered = {s.source_type for s in sources}
    # Weaker fallbacks, only for source types no monthly record covered.
    for source_type, items in contract_terms.items():
        if source_type in covered:
            continue
        amount = min(a for a, _ in items)
        share = policy.income_recognition(source_type)
        sources.append(
            IncomeSource(
                source_type=source_type,
                months=[],
                average=amount,
                minimum=amount,
                recognition=share,
                recognised=amount * share,
                evidence=SRC_CONTRACT,
                files=sorted({f for _, f in items}),
                note="mức thỏa thuận trên hợp đồng, chưa có chứng từ nhận tiền",
            )
        )
        covered.add(source_type)
        warnings.append(
            f"Thu nhập \"{SOURCE_TYPE_LABEL.get(source_type, source_type)}\" chỉ có "
            "mức thỏa thuận trên hợp đồng, chưa có sao kê/bảng lương chứng minh tiền "
            "thực nhận."
        )
    for source_type, items in annual.items():
        if source_type in covered:
            continue
        latest = max(items, key=lambda item: item[1])
        amount = latest[0] / 12
        share = policy.income_recognition(source_type)
        sources.append(
            IncomeSource(
                source_type=source_type,
                months=[],
                average=amount,
                minimum=amount,
                recognition=share,
                recognised=amount * share,
                evidence=SRC_TAX,
                files=[latest[2]],
                note=f"thu nhập chịu thuế năm {latest[1]} chia 12",
            )
        )
        covered.add(source_type)

    employment = _block(sitevisit, "employment_verification")
    verified_salary = num(employment.get("verified_monthly_income"))
    if verified_salary and "luong" not in covered:
        share = policy.income_recognition("luong")
        sources.append(
            IncomeSource(
                source_type="luong",
                months=[],
                average=verified_salary,
                minimum=verified_salary,
                recognition=share,
                recognised=verified_salary * share,
                evidence=SRC_APPRAISAL,
                note="thu nhập do cán bộ thẩm định xác minh, không có chứng từ gốc",
            )
        )
        covered.add("luong")
    business = _block(sitevisit, "business_verification")
    verified_profit = num(business.get("monthly_profit"))
    if verified_profit and "kinh_doanh" not in covered:
        share = policy.income_recognition("kinh_doanh")
        sources.append(
            IncomeSource(
                source_type="kinh_doanh",
                months=[],
                average=verified_profit,
                minimum=verified_profit,
                recognition=share,
                recognised=verified_profit * share,
                evidence=SRC_APPRAISAL,
                note="lợi nhuận bình quân tháng do cán bộ thẩm định ghi nhận",
            )
        )
    sources.sort(key=lambda s: list(SOURCE_TYPE_LABEL).index(s.source_type)
                 if s.source_type in SOURCE_TYPE_LABEL else 99)
    return sources, dict(sorted(series.items(), key=lambda kv: month_sort_key(kv[0]))), warnings


def declared_income_total(proposal: dict[str, Any] | None) -> float | None:
    items = _block(proposal, "declared_income").get("items") or []
    return _sum(num(i.get("monthly_amount")) for i in items if isinstance(i, dict))


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


def policy_breach(row: Row) -> str:
    """One warning line for a row that fails its policy limit."""

    if row.unit == "%":
        shown, limit = pct(row.value), pct(row.limit)
    elif row.unit == "VNĐ":
        shown, limit = money(row.value), money(row.limit)
    else:
        shown = f"{format_vn_number(row.value or 0, 1)} {row.unit}"
        limit = f"{format_vn_number(row.limit or 0, 1)} {row.unit}"
    verb = "vượt ngưỡng" if row.status == "Vượt ngưỡng" else "không đạt mức tối thiểu"
    return f"{row.label}: {shown} — {verb} chính sách ({limit})."


def resolve_product(
    loan_program: str | None,
    proposal: dict[str, Any] | None,
    policy: RetailPolicy,
) -> tuple[str, bool]:
    """(product id, inferred?) — the request's product, else the purpose's."""

    if loan_program and loan_program in policy.products:
        return loan_program, False
    category = str(_block(proposal, "loan_request").get("purpose_category") or "")
    product = PURPOSE_TO_PRODUCT.get(category, "")
    if product in policy.products:
        return product, True
    return "", False


def _status(value: float | None, limit: float | None, *, upper: bool) -> str:
    if value is None or limit is None:
        return ""
    ok = value <= limit if upper else value >= limit
    return "Đạt" if ok else "Vượt ngưỡng" if upper else "Không đạt"


def build_affordability(
    documents: list[ClassifiedDocument],
    loan_program: str | None = None,
    policy: RetailPolicy | None = None,
    today: date | None = None,
) -> AffordabilityResult:
    """Compute the repayment-capacity table from every extracted document."""

    policy = policy or load_policy()
    today = today or date.today()
    usable = [d for d in documents if d.extraction_status == "success"]
    proposal = first_extraction(usable, "proposal_extraction", "is_proposal")
    sitevisit = first_extraction(usable, "sitevisit_extraction", "is_sitevisit")
    cic_records = [
        d.cic_s10a_extraction
        for d in usable
        if d.is_cic_s10a and isinstance(d.cic_s10a_extraction, dict)
    ]
    product_id, inferred = resolve_product(loan_program, proposal, policy)
    limits: ProductLimits = policy.limits_for(product_id or None)
    label = limits.label + (" (suy ra từ mục đích vay trên giấy đề nghị)" if inferred else "")
    result = AffordabilityResult(product_label=label, product_id=product_id)
    rows = result.rows

    # ---- Income -----------------------------------------------------------
    sources, series, warnings = collect_income_sources(usable, policy, sitevisit)
    result.income_sources = sources
    result.monthly_income_series = series
    result.warnings.extend(warnings)

    for source in sources:
        rows.append(
            Row(
                label=f"Thu nhập bình quân tháng — {SOURCE_TYPE_LABEL.get(source.source_type, source.source_type)}",
                unit="VNĐ",
                value=source.average,
                source=source.evidence,
                note=(
                    (f"{len(source.months)} tháng gần nhất; " if source.months else "")
                    + f"ghi nhận {pct(source.recognition, 0)}"
                    + (f"; {source.note}" if source.note else "")
                ),
            )
        )

    declared = declared_income_total(proposal)
    evidenced = _sum(s.average for s in sources)
    recognised = _sum(s.recognised for s in sources)
    if recognised is None and declared:
        # Nothing evidences any income: fall back to the declaration, at the
        # most conservative recognition share, and say so loudly.
        share = policy.income_recognition("khac")
        recognised = declared * share
        result.warnings.append(
            "Không có chứng từ thu nhập nào được trích xuất — thu nhập ghi nhận "
            f"tạm lấy {pct(share, 0)} mức khách hàng tự kê khai trên giấy đề nghị."
        )
    if declared is not None:
        rows.append(
            Row(
                label="Thu nhập khách hàng tự kê khai (giấy đề nghị)",
                unit="VNĐ",
                value=declared,
                source=SRC_PROPOSAL,
            )
        )
        if evidenced and declared > evidenced * 1.2:
            result.warnings.append(
                "Thu nhập kê khai cao hơn thu nhập có chứng từ trên 20% — phần "
                "chênh lệch không được dùng để tính khả năng trả nợ."
            )
    rows.append(
        Row(
            label="Thu nhập ghi nhận để tính khả năng trả nợ",
            unit="VNĐ",
            value=recognised,
            source=SRC_DERIVED if sources else SRC_PROPOSAL,
            note="tổng thu nhập bình quân × tỷ lệ ghi nhận theo từng nguồn",
            limit=limits.min_monthly_income,
            status=_status(recognised, limits.min_monthly_income, upper=False),
        )
    )
    result.recognised_income = recognised

    # ---- Living costs -----------------------------------------------------
    expenses = _block(proposal, "declared_expenses")
    borrower = _block(proposal, "borrower")
    residence = _block(sitevisit, "residence_verification")
    dependants = num(borrower.get("dependents"))
    if dependants is None:
        dependants = num(residence.get("dependents"))
    household = num(borrower.get("household_size")) or num(residence.get("household_members"))
    co_borrowers = [
        i for i in _block(proposal, "co_borrowers").get("items") or [] if isinstance(i, dict)
    ]
    if household:
        adults = max(1.0, household - (dependants or 0))
    else:
        adults = 1.0 + len(co_borrowers)
    floor = (
        adults * float(policy.defaults.get("living_cost_per_adult", 0))
        + (dependants or 0) * float(policy.defaults.get("living_cost_per_dependant", 0))
    )
    declared_living = num(expenses.get("monthly_living_expenses"))
    living = max(v for v in (declared_living, floor) if v is not None) if (declared_living or floor) else None
    living_source = (
        SRC_PROPOSAL if declared_living is not None and declared_living >= floor else SRC_POLICY
    )
    rows.append(
        Row(
            label="Chi phí sinh hoạt hằng tháng",
            unit="VNĐ",
            value=living,
            source=living_source,
            note=(
                f"mức sàn chính sách {money(floor)} cho {adults:g} người lớn"
                + (f" và {dependants:g} người phụ thuộc" if dependants else "")
                + (
                    f"; khách hàng kê khai {money(declared_living)}"
                    if declared_living is not None
                    else "; khách hàng không kê khai"
                )
            ),
        )
    )
    other_expenses = num(expenses.get("other_monthly_expenses"))
    if other_expenses:
        rows.append(
            Row(
                label="Chi phí cố định khác hằng tháng",
                unit="VNĐ",
                value=other_expenses,
                source=SRC_PROPOSAL,
            )
        )
    result.living_costs = _sum([living, other_expenses])

    # ---- Existing obligations ---------------------------------------------
    declared_debt = _sum(
        num(i.get("monthly_payment"))
        for i in _block(proposal, "existing_debts").get("items") or []
        if isinstance(i, dict)
    )
    statement_debt = _sum(
        num(row.get("monthly_amount"))
        for doc in usable
        if doc.is_income and isinstance(doc.income_extraction, dict)
        for row in doc.income_extraction.get("recurring_debits") or []
        if isinstance(row, dict)
    )
    stress_rate = float(policy.defaults.get("stress_rate_pct_year", 12.0))
    loans = cic_loan_rows(cic_records)
    result.cic_loan_balance = _sum(balance for _, _, balance in loans)
    cic_estimate = None
    if loans:
        short_months = float(policy.defaults.get("assumed_remaining_months_short_term", 12))
        long_months = float(policy.defaults.get("assumed_remaining_months_long_term", 60))
        cic_estimate = _sum(
            monthly_payment(
                balance,
                short_months if bucket == "ngan_han" else long_months,
                stress_rate,
                "goc_deu_lai_giam_dan",
            )
            for _, bucket, balance in loans
        )
    if declared_debt is not None:
        rows.append(Row("Nghĩa vụ trả nợ hiện hữu khách hàng kê khai", "VNĐ", declared_debt, SRC_PROPOSAL))
    if statement_debt is not None:
        rows.append(
            Row(
                "Khoản trả nợ định kỳ thấy trên sao kê",
                "VNĐ",
                statement_debt,
                SRC_STATEMENT,
                note="ghi nợ lặp lại hằng tháng có dấu hiệu trả nợ",
            )
        )
    if cic_estimate is not None:
        rows.append(
            Row(
                "Nghĩa vụ ước tính từ dư nợ vay trên CIC",
                "VNĐ",
                cic_estimate,
                f"{SRC_CIC} + {SRC_POLICY}",
                note=(
                    f"dư nợ vay {money(result.cic_loan_balance)}, giả định trả gốc "
                    "đều trong thời hạn còn lại theo chính sách và lãi suất "
                    f"{format_vn_number(stress_rate, 1)}%/năm"
                ),
            )
        )
    loan_obligation = max(
        (v for v in (declared_debt, statement_debt, cic_estimate) if v is not None),
        default=None,
    )
    if loan_obligation is not None and cic_estimate and (declared_debt or 0) < cic_estimate * 0.5:
        result.warnings.append(
            "Nghĩa vụ trả nợ khách hàng kê khai thấp hơn nhiều so với mức suy ra "
            "từ dư nợ trên CIC — cần làm rõ các khoản vay chưa kê khai."
        )
    if statement_debt and (declared_debt or 0) < statement_debt * 0.8:
        result.warnings.append(
            "Sao kê cho thấy các khoản trả nợ định kỳ lớn hơn mức khách hàng kê "
            "khai — có thể có khoản vay chưa kê khai (công ty tài chính, vay cá nhân)."
        )

    card_estimate = None
    cards = cic_cards(cic_records)
    if cards and not _declares_card(proposal):
        rate = float(policy.defaults.get("card_payment_rate", 0.05))
        card_estimate = _sum(
            (limit if limit is not None else balance or 0.0) * rate
            for _, limit, balance in cards
        )
        rows.append(
            Row(
                "Nghĩa vụ ước tính thẻ tín dụng",
                "VNĐ",
                card_estimate,
                f"{SRC_CIC} + {SRC_POLICY}",
                note=f"{pct(rate, 0)} hạn mức thẻ (hoặc dư nợ khi không có hạn mức), {len(cards)} thẻ",
            )
        )
    obligations = _sum([loan_obligation, card_estimate])
    rows.append(
        Row(
            "Tổng nghĩa vụ trả nợ hiện hữu hằng tháng",
            "VNĐ",
            obligations or 0.0,
            SRC_DERIVED,
            note="lấy số LỚN NHẤT giữa kê khai, sao kê và ước tính từ CIC, cộng nghĩa vụ thẻ",
        )
    )
    result.existing_obligations = obligations or 0.0

    # ---- New loan ---------------------------------------------------------
    request = _block(proposal, "loan_request")
    financing = _block(proposal, "purpose_financing")
    amount = num(request.get("amount")) or num(financing.get("loan_amount"))
    tenor = num(request.get("tenor_months"))
    rate = num(request.get("interest_rate_pct_year"))
    rate_source = SRC_PROPOSAL
    if rate is None:
        rate, rate_source = stress_rate, SRC_POLICY
    method = str(request.get("repayment_method") or "")
    if method not in REPAYMENT_METHOD_LABEL:
        method_note = "không nêu phương thức — tính theo gốc đều, lãi giảm dần"
        method = "goc_deu_lai_giam_dan"
    else:
        method_note = REPAYMENT_METHOD_LABEL[method]
    result.loan_amount, result.tenor_months = amount, tenor
    result.rate_pct_year, result.repayment_method = rate, method

    if amount is not None:
        rows.append(Row("Số tiền đề nghị vay", "VNĐ", amount, SRC_PROPOSAL))
    if tenor is not None:
        rows.append(
            Row(
                "Thời hạn vay đề nghị",
                "tháng",
                tenor,
                SRC_PROPOSAL,
                limit=limits.max_tenor_months,
                status=_status(tenor, limits.max_tenor_months, upper=True),
            )
        )
    rows.append(Row("Lãi suất dùng để tính", "%/năm", rate, rate_source,
                    note="" if rate_source == SRC_PROPOSAL else "giấy đề nghị không nêu lãi suất — dùng lãi suất giả định"))
    new_payment = monthly_payment(amount, tenor, rate, method)
    result.new_payment = new_payment
    rows.append(
        Row(
            "Số tiền trả nợ hằng tháng khoản vay mới (kỳ cao nhất)",
            "VNĐ",
            new_payment,
            SRC_DERIVED,
            note=method_note,
        )
    )
    declared_payment = num(_block(proposal, "repayment_plan").get("monthly_payment_declared"))
    if declared_payment and new_payment and declared_payment < new_payment * 0.9:
        result.warnings.append(
            "Số tiền trả nợ hằng tháng khách hàng dự kiến trên giấy đề nghị thấp "
            "hơn kỳ trả nợ cao nhất hệ thống tính được — phương án trả nợ có thể "
            "đang tính theo kỳ cuối hoặc chưa tính lãi."
        )
    if method == "goc_cuoi_ky":
        result.warnings.append(
            "Phương thức trả gốc cuối kỳ: DSR chỉ phản ánh tiền lãi; nguồn trả "
            "gốc một lần cuối kỳ cần được chứng minh riêng."
        )

    # ---- Ratios -----------------------------------------------------------
    total_service = _sum([obligations, new_payment])
    dsr = _div(total_service, recognised)
    rows.append(
        Row(
            "DSR — tổng trả nợ hằng tháng / thu nhập ghi nhận",
            "%",
            dsr,
            SRC_DERIVED,
            note="(nghĩa vụ hiện hữu + khoản vay mới) / thu nhập ghi nhận",
            limit=limits.max_dsr,
            status=_status(dsr, limits.max_dsr, upper=True),
        )
    )
    surplus = None
    if recognised is not None:
        surplus = recognised - (result.living_costs or 0) - (total_service or 0)
    rows.append(
        Row(
            "Thu nhập còn lại sau chi phí sinh hoạt và trả nợ",
            "VNĐ",
            surplus,
            SRC_DERIVED,
            note="thu nhập ghi nhận − chi phí sinh hoạt, chi phí khác − tổng trả nợ",
            status="" if surplus is None else ("Dương" if surplus >= 0 else "Âm"),
        )
    )
    if surplus is not None and surplus < 0:
        result.warnings.append(
            "Thu nhập còn lại ÂM: thu nhập ghi nhận không đủ trang trải chi phí "
            "sinh hoạt và toàn bộ nghĩa vụ trả nợ."
        )
    total_debt = _sum([result.cic_loan_balance, amount])
    leverage = _div(total_debt, recognised * 12 if recognised else None)
    if leverage is not None:
        rows.append(
            Row(
                "Tổng dư nợ (CIC + khoản vay mới) / thu nhập ghi nhận năm",
                "lần",
                leverage,
                SRC_DERIVED,
            )
        )

    # ---- Age at maturity ----------------------------------------------------
    birth_year = parse_birth_year(borrower.get("date_of_birth"))
    if birth_year is None:
        birth_year = next(
            (
                parse_birth_year(_block(record, "khach_hang").get("ngay_sinh"))
                for record in cic_records
                if parse_birth_year(_block(record, "khach_hang").get("ngay_sinh"))
            ),
            None,
        )
    if birth_year is not None:
        age = today.year - birth_year
        result.borrower_age = float(age)
        rows.append(Row("Tuổi khách hàng hiện tại", "tuổi", age, SRC_PROPOSAL if borrower.get("date_of_birth") else SRC_CIC))
        if tenor:
            maturity_age = age + tenor / 12
            rows.append(
                Row(
                    "Tuổi khách hàng khi kết thúc khoản vay",
                    "tuổi",
                    maturity_age,
                    SRC_DERIVED,
                    limit=limits.max_age_at_maturity,
                    status=_status(maturity_age, limits.max_age_at_maturity, upper=True),
                )
            )
    for row in rows:
        if row.status in ("Vượt ngưỡng", "Không đạt"):
            result.warnings.append(policy_breach(row))
    return result
