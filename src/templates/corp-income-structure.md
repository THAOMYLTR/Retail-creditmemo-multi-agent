<!--
corp_income report template. Replace this file with your own layout.
Write a placeholder as a name inside double curly braces, e.g. company_name
inside double curly braces becomes the company name read from the documents.
This comment block is removed from the report.
A name the agent does not know becomes "—" and is listed in
result.meta["unknown_placeholders"]. Delete this file to use the built-in layout.

Available names:
  abbreviation, abbreviation_source, actual_operator, administrative_penalties_table
  as_of, borrower_ownership, borrowers, charter_capital
  charter_capital_source, company_name, company_name_foreign, company_name_foreign_source
  company_name_source, disputes_table, documents_table, enforcement_actions_table
  establishment_date, establishment_date_source, head_office_address, head_office_address_source
  item1_conclusion, item1_findings, item1_table, item1_title
  item2_conclusion, item2_findings, item2_title, item3_conclusion
  item3_findings, item3_title, item4_conclusion, item4_findings
  item4_title, latest_change_date, latest_change_date_source, legal_form
  legal_form_source, legal_rep_table, legal_representatives, legal_representatives_source
  members_table, operating_status, operating_status_source, registration_change_number
  registration_change_number_source, scope_note, social_insurance_debts_table, summary_table
  tax_code, tax_code_source, tax_debts_table

Full description: docs/CORP_INCOME.md, section "Report template".
-->

# Corp Income Assessment — Pháp lý doanh nghiệp (mục 1–4)

- Ngày tham chiếu (thời điểm vay): **{{as_of}}** — cửa sổ kiểm tra 12 tháng tính đến ngày này
- KH cá nhân vay vốn: {{borrowers}}

## Tóm tắt

{{summary_table}}

## 1. {{item1_title}}

{{item1_table}}

{{item1_findings}}

## 2. {{item2_title}}

Vốn điều lệ: **{{charter_capital}}**

{{members_table}}

{{item2_findings}}

## 3. {{item3_title}}

{{legal_rep_table}}

Người điều hành thực tế: {{actual_operator}}

{{item3_findings}}

## 4. {{item4_title}}

**Nợ thuế:** {{tax_debts_table}}

**Nợ BHXH:** {{social_insurance_debts_table}}

**Xử phạt vi phạm hành chính:** {{administrative_penalties_table}}

**Cưỡng chế:** {{enforcement_actions_table}}

**Tranh chấp / kiện tụng:** {{disputes_table}}

{{item4_findings}}

## Hồ sơ đã xử lý

{{documents_table}}

> {{scope_note}}
