"""Deterministic maximum-loan table for the Credit Proposal agent.

A retail loan is capped by several independent limits at once, and the proposal
is only as good as its arithmetic on each:

- **repayment capacity** — the instalment the recognised income can carry under
  the product's DSR cap, after existing obligations, turned back into principal
  over the tenor actually allowed;
- **collateral** — the product's LTV cap on the appraised value;
- **own contribution** — the share of the purchase price the borrower must fund;
- **product caps** — a flat maximum, or a multiple of income for unsecured loans;
- **the request itself** — the bank never lends more than was asked for.

The table computes each, names the one that binds, and reports the instalment and
DSR at the resulting amount. The tenor is cut first when the borrower's age at
maturity would exceed policy, because a shorter tenor raises the instalment and
therefore lowers the capacity-based amount — doing it the other way round
overstates the loan.

Every row carries its source, as in the affordability table: a reviewer must see
at a glance that "LTV tối đa 70%" is policy and "giá trị TSBĐ" came from a
valuation certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.affordability_calculator import (
    SRC_APPRAISAL,
    SRC_DERIVED,
    SRC_POLICY,
    SRC_PROPOSAL,
    AffordabilityResult,
    Row,
    first_extraction,
    monthly_payment,
    num,
    pct,
    principal_for_payment,
)
from src.utils.formatting import format_vn_number
from src.policy.retail_policy import ProductLimits, RetailPolicy, load_policy
from src.types import ClassifiedDocument


@dataclass
class LoanSizingTable:
    product_label: str = ""
    rows: list[Row] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    max_loan: float | None = None
    binding_constraint: str = ""
    tenor_months: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "product": self.product_label,
            "rows": [row.as_dict() for row in self.rows],
            "warnings": self.warnings,
            "max_loan": self.max_loan,
            "binding_constraint": self.binding_constraint,
            "tenor_months": self.tenor_months,
        }


def _block(record: dict[str, Any] | None, name: str) -> dict[str, Any]:
    block = (record or {}).get(name)
    return block if isinstance(block, dict) else {}


def collateral_value(
    proposal: dict[str, Any] | None,
    sitevisit: dict[str, Any] | None,
) -> tuple[float | None, str]:
    """Total collateral value and where it was read.

    The application's own list is preferred — it is the list the loan will be
    secured by, often with the valuation certificate's figure. The appraisal
    report's estimate is the fallback.
    """

    block = _block(proposal, "collateral")
    total = num(block.get("total_value"))
    if total is None:
        values = [
            num(item.get("value"))
            for item in block.get("items") or []
            if isinstance(item, dict)
        ]
        values = [v for v in values if v is not None]
        total = sum(values) if values else None
    if total:
        return total, SRC_PROPOSAL
    estimate = num(_block(sitevisit, "collateral_verification").get("estimated_value"))
    if estimate:
        return estimate, SRC_APPRAISAL
    return None, ""


def build_loan_sizing_table(
    documents: list[ClassifiedDocument],
    affordability: AffordabilityResult,
    loan_program: str | None = None,
    policy: RetailPolicy | None = None,
) -> LoanSizingTable:
    """Compute the maximum loan under every policy limit, and which one binds."""

    policy = policy or load_policy()
    # The affordability table already resolved the product (named, or inferred
    # from the purpose); using the same one keeps the two tables consistent.
    product_id = affordability.product_id or (loan_program or "")
    limits: ProductLimits = policy.limits_for(product_id or None)
    usable = [d for d in documents if d.extraction_status == "success"]
    proposal = first_extraction(usable, "proposal_extraction", "is_proposal")
    sitevisit = first_extraction(usable, "sitevisit_extraction", "is_sitevisit")
    table = LoanSizingTable(product_label=affordability.product_label or limits.label)
    rows = table.rows
    candidates: list[tuple[str, float]] = []

    requested = affordability.loan_amount
    rate = affordability.rate_pct_year
    method = affordability.repayment_method or "goc_deu_lai_giam_dan"

    # ---- Tenor --------------------------------------------------------------
    tenor_requested = affordability.tenor_months
    tenor_caps: list[tuple[str, float]] = []
    if limits.max_tenor_months:
        tenor_caps.append(("chính sách sản phẩm", limits.max_tenor_months))
        rows.append(Row("Thời hạn tối đa theo sản phẩm", "tháng", limits.max_tenor_months, SRC_POLICY))
    if affordability.borrower_age is not None and limits.max_age_at_maturity:
        by_age = max(0.0, (limits.max_age_at_maturity - affordability.borrower_age) * 12)
        tenor_caps.append(("tuổi khi đáo hạn", by_age))
        rows.append(
            Row(
                "Thời hạn tối đa theo tuổi khách hàng",
                "tháng",
                by_age,
                SRC_DERIVED,
                note=f"tuổi tối đa khi kết thúc khoản vay {limits.max_age_at_maturity:g}",
            )
        )
    tenor = tenor_requested
    if tenor is None and tenor_caps:
        tenor = min(cap for _, cap in tenor_caps)
        table.warnings.append(
            "Giấy đề nghị không nêu thời hạn vay — tính theo thời hạn tối đa cho phép."
        )
    elif tenor is not None and tenor_caps:
        cap_name, cap = min(tenor_caps, key=lambda item: item[1])
        if tenor > cap:
            table.warnings.append(
                f"Thời hạn đề nghị {tenor:g} tháng vượt mức tối đa {cap:g} tháng "
                f"({cap_name}) — các số dưới tính theo {cap:g} tháng."
            )
            tenor = cap
    table.tenor_months = tenor
    rows.append(Row("Thời hạn áp dụng để tính", "tháng", tenor, SRC_DERIVED))

    # ---- Repayment capacity -------------------------------------------------
    income = affordability.recognised_income
    obligations = affordability.existing_obligations or 0.0
    if income is not None and limits.max_dsr is not None:
        capacity = income * limits.max_dsr - obligations
        rows.append(
            Row(
                "Số tiền trả nợ tối đa hằng tháng cho khoản vay mới",
                "VNĐ",
                capacity,
                SRC_DERIVED,
                note=(
                    f"thu nhập ghi nhận × DSR tối đa {pct(limits.max_dsr, 0)} − "
                    "nghĩa vụ trả nợ hiện hữu"
                ),
            )
        )
        if capacity <= 0:
            table.warnings.append(
                "Nghĩa vụ trả nợ hiện hữu đã chiếm hết hạn mức DSR — không còn "
                "khả năng trả nợ cho khoản vay mới."
            )
            candidates.append(("khả năng trả nợ (DSR)", 0.0))
            rows.append(Row("Số tiền vay tối đa theo khả năng trả nợ", "VNĐ", 0.0, SRC_DERIVED))
        else:
            by_capacity = principal_for_payment(capacity, tenor, rate, method)
            if by_capacity is not None:
                candidates.append(("khả năng trả nợ (DSR)", by_capacity))
            rows.append(
                Row(
                    "Số tiền vay tối đa theo khả năng trả nợ",
                    "VNĐ",
                    by_capacity,
                    SRC_DERIVED,
                    note=(
                        f"lãi suất {format_vn_number(rate, 2)}%/năm, {tenor:g} tháng"
                        if rate is not None and tenor
                        else ""
                    ),
                )
            )

    # ---- Collateral ---------------------------------------------------------
    value, value_source = collateral_value(proposal, sitevisit)
    if value is not None:
        rows.append(Row("Giá trị tài sản bảo đảm", "VNĐ", value, value_source))
    if limits.max_ltv is not None:
        rows.append(Row("Tỷ lệ cho vay/giá trị TSBĐ tối đa (LTV)", "%", limits.max_ltv, SRC_POLICY))
        if value is not None:
            by_ltv = value * limits.max_ltv
            candidates.append(("giá trị tài sản bảo đảm (LTV)", by_ltv))
            rows.append(Row("Số tiền vay tối đa theo LTV", "VNĐ", by_ltv, SRC_DERIVED))
        elif product_id and product_id != "TIEU_DUNG":
            table.warnings.append(
                "Sản phẩm yêu cầu tài sản bảo đảm nhưng hồ sơ chưa có giá trị "
                "định giá — chưa tính được giới hạn theo LTV."
            )

    # ---- Own contribution ---------------------------------------------------
    financing = _block(proposal, "purpose_financing")
    total_cost = num(financing.get("total_cost"))
    if total_cost is not None:
        rows.append(Row("Tổng nhu cầu vốn của phương án", "VNĐ", total_cost, SRC_PROPOSAL))
        own = num(financing.get("own_contribution"))
        if own is not None:
            rows.append(
                Row(
                    "Vốn tự có khách hàng tham gia",
                    "VNĐ",
                    own,
                    SRC_PROPOSAL,
                    note=f"{pct(own / total_cost)} tổng nhu cầu vốn" if total_cost else "",
                )
            )
        if limits.min_own_contribution is not None:
            by_own = total_cost * (1 - limits.min_own_contribution)
            candidates.append(("vốn tự có tối thiểu", by_own))
            rows.append(
                Row(
                    "Số tiền vay tối đa theo tỷ lệ vốn tự có tối thiểu",
                    "VNĐ",
                    by_own,
                    SRC_DERIVED,
                    note=f"vốn tự có tối thiểu {pct(limits.min_own_contribution, 0)} theo chính sách",
                )
            )

    # ---- Product caps -------------------------------------------------------
    if limits.max_amount is not None:
        candidates.append(("trần sản phẩm", limits.max_amount))
        rows.append(Row("Số tiền vay tối đa của sản phẩm", "VNĐ", limits.max_amount, SRC_POLICY))
    if limits.max_income_multiple is not None and income is not None:
        by_multiple = income * limits.max_income_multiple
        candidates.append(("bội số thu nhập", by_multiple))
        rows.append(
            Row(
                "Số tiền vay tối đa theo bội số thu nhập",
                "VNĐ",
                by_multiple,
                SRC_DERIVED,
                note=f"{limits.max_income_multiple:g} lần thu nhập ghi nhận tháng (chính sách)",
            )
        )
    if requested is not None:
        candidates.append(("số tiền khách hàng đề nghị", requested))

    # ---- Result -------------------------------------------------------------
    # Caps alone say how much a product could ever lend, not how much this
    # borrower can: without income, a request or a collateral value there is
    # nothing to size, and the cap must not be reported as a recommendation.
    caps_only = {"trần sản phẩm", "bội số thu nhập"}
    if candidates and all(name in caps_only for name, _ in candidates):
        candidates = []
    if candidates:
        name, amount = min(candidates, key=lambda item: item[1])
        amount = max(0.0, amount)
        table.max_loan, table.binding_constraint = amount, name
        rows.append(
            Row(
                "Số tiền vay tối đa đề xuất",
                "VNĐ",
                amount,
                SRC_DERIVED,
                note=f"giới hạn chặn: {name}",
            )
        )
        payment = monthly_payment(amount, tenor, rate, method)
        rows.append(Row("Trả nợ hằng tháng tại số tiền đề xuất (kỳ cao nhất)", "VNĐ", payment, SRC_DERIVED))
        if income:
            dsr = ((payment or 0.0) + obligations) / income
            rows.append(
                Row(
                    "DSR tại số tiền đề xuất",
                    "%",
                    dsr,
                    SRC_DERIVED,
                    limit=limits.max_dsr,
                    status=(
                        "" if limits.max_dsr is None
                        else "Đạt" if dsr <= limits.max_dsr + 1e-9 else "Vượt ngưỡng"
                    ),
                )
            )
        if value:
            rows.append(Row("LTV tại số tiền đề xuất", "%", amount / value, SRC_DERIVED,
                            limit=limits.max_ltv,
                            status="" if limits.max_ltv is None else
                            "Đạt" if amount / value <= limits.max_ltv + 1e-9 else "Vượt ngưỡng"))
        if requested is not None and amount < requested - 1:
            table.warnings.append(
                f"Số tiền vay tối đa thấp hơn số tiền đề nghị — giới hạn chặn là {name}."
            )
    else:
        table.warnings.append(
            "Không đủ dữ liệu để tính số tiền vay tối đa (thiếu thu nhập, giá trị "
            "TSBĐ và số tiền đề nghị)."
        )
    return table
