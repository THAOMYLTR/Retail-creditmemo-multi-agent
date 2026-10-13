# Architecture — Retail Credit Memo

This repository is the **retail (khách hàng cá nhân)** adaptation of
[hodongtu/SME-creditmemo-multi-agent](https://github.com/hodongtu/SME-creditmemo-multi-agent).
The orchestration, OCR, classification, citation and rendering machinery are kept;
everything that encoded *what an SME credit file is* was replaced with what a retail
credit file is.

The entry point is the notebook `local_underwriting_agents.ipynb`; all logic lives in the
importable package `src/`.

## What changed from the SME version

| SME concept | Retail replacement | Where |
|---|---|---|
| `BUSINESS_ACTIVITY_AGENT` — operations, suppliers, customers | `CUSTOMER_PROFILE_AGENT` — identity, household, residence, employment, purpose fit | `specialist.py`, `customer-profile-*.md` |
| `FINANCIAL_ANALYSIS_AGENT` — financial statements, ratios | `INCOME_ANALYSIS_AGENT` — proof of income, living costs, obligations, DSR | `specialist.py`, `income-analysis-*.md` |
| BCTC extraction (balance sheet, P&L, cash flow) | Proof-of-income extraction (contract, payslip, bank statement, PIT, rental, household business) | `income_extraction.py` |
| `FinancialRatioCalculator` (CCC, ROE, leverage…) | Affordability calculator (recognised income, obligations, new instalment, DSR, surplus, age at maturity) | `affordability_calculator.py` |
| Working-capital credit-need table (LC, guarantees) | Maximum-loan table (DSR, LTV, own contribution, product caps, binding limit) | `loan_sizing_calculator.py` |
| Corporate credit application (sections B, C, D) | Retail loan application with repayment plan | `proposal_extraction.py` |
| Site-visit report (industry, GSO, supply chain) | Appraisal / verification report (residence, workplace, income, collateral) | `sitevisit_extraction.py` |
| CIC S10A corporate report | CIC individual credit report, plus credit cards and limits | `cic_s10a_extraction.py` |
| Debt vs VAT revenue chart | Debt vs evidenced monthly income chart | `prompt_blocks.py` |
| Loan programs B1CP / MISA / PL++ / PLO | Products `TIEU_DUNG`, `MUA_NHA`, `MUA_O_TO`, `HO_KINH_DOANH` | `document_matrix.yaml` |
| — | Product policy limits (DSR, LTV, tenor, age, own contribution) | `src/policy/retail_products.yaml` |
| Memo money unit: tỷ VNĐ | triệu VNĐ | `formatting.py` |
| PDF / XLSX / XML input | + phone photos (JPG/PNG/TIFF) via OCR | `ocr.py`, `extractors.py` |

Removed because they have no retail counterpart: e-tax XML parsing (`tax_xml.py`), VAT
revenue transcription (`vat_revenue.py`), industry knowledge decks
(`industry_knowledge.py`, `scripts/ingest_industry_knowledge.py`), and the unused legacy
flow diagram (`utils/flow.py`).

## Module map

| Module | Role |
|---|---|
| `src/agents/supervisor.py` | LangGraph orchestrator: routing, classification, gap check, extraction, agents, finalize |
| `src/agents/document_classification.py` | Keyword scoring against the matrix, LLM fallback, loan-product detection |
| `src/matrix/document_matrix.yaml` | **Source of truth** for document type → agents (R/O per product) |
| `src/policy/retail_products.yaml` | **Source of truth** for product limits used by the calculators |
| `src/policy/retail_policy.py` | Loads/validates the policy; strictest limits when no product is known |
| `src/agents/income_extraction.py` | Proof of income → monthly income record |
| `src/agents/proposal_extraction.py` | Loan application → amount, tenor, rate, method, declared income/expenses/debts, collateral |
| `src/agents/cic_s10a_extraction.py` | CIC credit report → balances, cards, 12-month trend, warnings |
| `src/agents/cic_r21_extraction.py` | CIC collateral report |
| `src/agents/sitevisit_extraction.py` | Appraisal report → verified facts, officer's opinion kept apart |
| `src/agents/affordability_calculator.py` | Deterministic repayment-capacity table |
| `src/agents/loan_sizing_calculator.py` | Deterministic maximum-loan table |
| `src/agents/prompt_blocks.py` | Renders every structured block into the agents' prompts, builds the chart |
| `src/agents/specialist.py` | The five specialists and the memo composer |
| `src/templates/*-structure.md` / `*-guidance.md` | Report layout (copied) / analysis rules (never copied) per agent |
| `src/agents/corp_income/` | **Corporate (KHDN)** legal-information agent, BRD items 1–4 — standalone, see `docs/CORP_INCOME.md` |
| `src/agents/tools.py` | Allow-listed SQL tools over retail tables (customer, income, CASA, T24, cards, CIC, collateral) |

## Two rules that shape the design

**Arithmetic lives in code, not in prompts.** Recognised income, obligations, the new
instalment, DSR, surplus, LTV and the maximum loan are computed deterministically and
handed to the agents as tables they must copy. Every row carries a *source* (`sao kê`,
`xác nhận lương`, `hợp đồng`, `đề nghị vay`, `CIC`, `chính sách`, `tính toán`) so a
reviewer can tell evidence from assumption.

**When in doubt, conservative.** For the same month, money credited on a statement beats a
payslip, which beats a contract, which beats the borrower's own declaration. Existing
obligations take the largest of declared, statement-visible and CIC-implied. The new
instalment is the first (largest) payment of the schedule. An unnamed product gets the
strictest limits — unless the application's purpose identifies the product, in which case
that product's limits apply and the table says so.

## Rule of thumb

- Which agent sees which document → edit `document_matrix.yaml`.
- DSR cap, LTV, tenor, age, living-cost floor, income recognition shares → edit
  `retail_products.yaml` (the shipped values are placeholders, not any bank's policy).
- What a report section contains → edit the agent's `*-structure.md`; how to analyse it →
  `*-guidance.md`.
- Run `python testing/run_checks.py` after any change.
