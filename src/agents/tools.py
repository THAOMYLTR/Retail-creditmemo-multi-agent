"""Database query tools for specialist agents (extracted from the notebook)."""

from __future__ import annotations

import json
from typing import Any

from collections.abc import Callable

from langchain_core.tools import tool


DatabaseExecutor = Callable[[str, dict[str, Any]], Any]
DATABASE_EXECUTOR: DatabaseExecutor | None = None
DEFAULT_QUERY_LIMIT = 20
MAX_QUERY_LIMIT = 100


# Point these table/column names at your own database schema. Retail
# customers are keyed by CIF and national ID number (CCCD), not tax code.
TABLE_CONFIGS = {
    "customer_profile": {
        "table": "retail_customer_master",
        "columns": [
            "customer_id",
            "customer_name",
            "id_number",
            "date_of_birth",
            "gender",
            "marital_status",
            "dependents",
            "residence_address",
            "residence_status",
            "phone",
            "occupation",
            "employer",
            "customer_segment",
            "kyc_status",
            "updated_at",
        ],
        "filter_columns": ["customer_id", "id_number", "customer_name"],
        "order_by": "updated_at DESC",
    },
    "income_record": {
        "table": "retail_income_record",
        "columns": [
            "customer_id",
            "id_number",
            "period",
            "income_type",
            "payer",
            "amount",
            "is_payroll_account",
            "updated_at",
        ],
        "filter_columns": ["customer_id", "id_number", "period"],
        "order_by": "period DESC",
    },
    "account_activity": {
        "table": "casa_monthly_activity",
        "columns": [
            "customer_id",
            "id_number",
            "account_number",
            "period",
            "total_credits",
            "total_debits",
            "average_balance",
            "salary_credits",
            "updated_at",
        ],
        "filter_columns": ["customer_id", "id_number", "account_number", "period"],
        "order_by": "period DESC",
    },
    "t24_credit_relationship": {
        "table": "t24_credit_relationship",
        "columns": [
            "customer_id",
            "id_number",
            "facility_id",
            "product_type",
            "approved_amount",
            "outstanding_balance",
            "monthly_installment",
            "currency",
            "interest_rate",
            "overdue_days",
            "debt_group",
            "repayment_status",
            "start_date",
            "maturity_date",
            "updated_at",
        ],
        "filter_columns": ["customer_id", "id_number", "facility_id"],
        "order_by": "updated_at DESC",
    },
    "credit_card": {
        "table": "credit_card_account",
        "columns": [
            "customer_id",
            "id_number",
            "card_id",
            "card_product",
            "credit_limit",
            "outstanding_balance",
            "minimum_payment",
            "days_past_due",
            "status",
            "updated_at",
        ],
        "filter_columns": ["customer_id", "id_number", "card_id"],
        "order_by": "updated_at DESC",
    },
    "cic_bureau": {
        "table": "cic_bureau",
        "columns": [
            "customer_id",
            "id_number",
            "bureau_report_date",
            "credit_institution",
            "facility_type",
            "approved_limit",
            "outstanding_balance",
            "overdue_balance",
            "overdue_days",
            "debt_group",
            "credit_score",
            "note",
            "updated_at",
        ],
        "filter_columns": ["customer_id", "id_number", "bureau_report_date"],
        "order_by": "bureau_report_date DESC, updated_at DESC",
    },
    "collateral": {
        "table": "collateral",
        "columns": [
            "customer_id",
            "id_number",
            "collateral_id",
            "collateral_type",
            "description",
            "appraised_value",
            "ltv",
            "ownership_status",
            "valuation_date",
            "updated_at",
        ],
        "filter_columns": ["customer_id", "id_number", "collateral_id"],
        "order_by": "updated_at DESC",
    },
}


def configure_database_executor(executor: DatabaseExecutor | None) -> None:
    """Inject database query function có sẵn vào notebook tools."""

    global DATABASE_EXECUTOR
    DATABASE_EXECUTOR = executor


def get_database_tools() -> list[Any]:
    """Return all database tools that can be attached to agents."""

    return [
        query_customer_profile,
        query_income_records,
        query_account_activity,
        query_t24_credit_relationship_data,
        query_credit_card_data,
        query_cic_bureau_data,
        query_collateral_data,
        query_configured_table,
    ]


def build_select_query(
    logical_table: str,
    filters: dict[str, Any],
    limit: int = DEFAULT_QUERY_LIMIT,
) -> tuple[str, dict[str, Any]]:
    """Build a parameterized SELECT from allowlisted table configs."""

    table_config = TABLE_CONFIGS.get(logical_table)
    if not table_config:
        allowed = ", ".join(sorted(TABLE_CONFIGS))
        raise ValueError(f"Unsupported logical_table. Allowed values: {allowed}")

    cleaned_filters = clean_query_filters(
        filters,
        table_config["filter_columns"],
    )
    if not cleaned_filters:
        raise ValueError("At least one non-empty filter is required.")

    where_clauses = []
    params = {}
    for column, value in cleaned_filters.items():
        param_name = f"p_{column}"
        if column.endswith("name"):
            where_clauses.append(f"LOWER({column}) LIKE LOWER(:{param_name})")
            params[param_name] = f"%{value}%"
        else:
            where_clauses.append(f"{column} = :{param_name}")
            params[param_name] = value

    params["limit"] = min(max(int(limit or DEFAULT_QUERY_LIMIT), 1), MAX_QUERY_LIMIT)
    columns = ", ".join(table_config["columns"])
    sql = (
        f"SELECT {columns} "
        f"FROM {table_config['table']} "
        f"WHERE {' AND '.join(where_clauses)} "
        f"ORDER BY {table_config['order_by']} "
        "LIMIT :limit"
    )
    return sql, params


def clean_query_filters(
    filters: dict[str, Any],
    allowed_columns: list[str],
) -> dict[str, Any]:
    """Keep only allowlisted, non-empty filters."""

    cleaned = {}
    for key, value in (filters or {}).items():
        if key not in allowed_columns or value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned[key] = value.strip() if isinstance(value, str) else value
    return cleaned


def execute_database_query(sql: str, params: dict[str, Any]) -> Any:
    """Run SQL through the injected database executor."""

    if DATABASE_EXECUTOR is None:
        raise RuntimeError(
            "Database executor is not configured. Call "
            "`configure_database_executor(executor)` first."
        )
    return DATABASE_EXECUTOR(sql, params)


def normalize_database_rows(rows: Any) -> list[dict[str, Any]]:
    """Normalize common database result shapes into dictionaries."""

    if rows is None:
        return []
    if isinstance(rows, str):
        return [{"text": rows}]
    if isinstance(rows, dict):
        if isinstance(rows.get("rows"), list):
            return normalize_database_rows(rows["rows"])
        return [rows]

    normalized_rows = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row)
        elif hasattr(row, "_asdict"):
            normalized_rows.append(row._asdict())
        elif hasattr(row, "_mapping"):
            normalized_rows.append(dict(row._mapping))
        else:
            normalized_rows.append({"value": row})
    return normalized_rows


def format_database_tool_response(
    tool_name: str,
    sql: str,
    params: dict[str, Any],
    rows: Any,
) -> str:
    """Return stable JSON for downstream agents."""

    normalized_rows = normalize_database_rows(rows)
    payload = {
        "status": "success",
        "tool": tool_name,
        "row_count": len(normalized_rows),
        "rows": normalized_rows,
        "query": {"sql": sql, "params": params},
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def format_database_tool_error(tool_name: str, error: Exception) -> str:
    """Return stable JSON error payload for downstream agents."""

    payload = {
        "status": "error",
        "tool": tool_name,
        "error_type": type(error).__name__,
        "error": str(error),
        "rows": [],
    }
    return json.dumps(payload, ensure_ascii=False)


def run_configured_table_query(
    logical_table: str,
    filters: dict[str, Any] | None = None,
    limit: int = DEFAULT_QUERY_LIMIT,
) -> str:
    """Run a safe read query against one configured logical table."""

    tool_name = f"query_{logical_table}"
    try:
        sql, params = build_select_query(logical_table, filters or {}, limit)
        rows = execute_database_query(sql, params)
        return format_database_tool_response(tool_name, sql, params, rows)
    except Exception as exc:
        return format_database_tool_error(tool_name, exc)


@tool
def query_customer_profile(
    id_number: str = "",
    customer_id: str = "",
    customer_name: str = "",
    limit: int = DEFAULT_QUERY_LIMIT,
) -> str:
    """Query retail customer master/profile data by ID number, CIF or name."""

    return run_configured_table_query(
        "customer_profile",
        {
            "id_number": id_number,
            "customer_id": customer_id,
            "customer_name": customer_name,
        },
        limit,
    )


@tool
def query_income_records(
    id_number: str = "",
    customer_id: str = "",
    period: str = "",
    limit: int = DEFAULT_QUERY_LIMIT,
) -> str:
    """Query income records (payroll credits, declared income) for a customer."""

    return run_configured_table_query(
        "income_record",
        {
            "id_number": id_number,
            "customer_id": customer_id,
            "period": period,
        },
        limit,
    )


@tool
def query_account_activity(
    id_number: str = "",
    customer_id: str = "",
    account_number: str = "",
    period: str = "",
    limit: int = DEFAULT_QUERY_LIMIT,
) -> str:
    """Query monthly current-account activity (credits, debits, balances)."""

    return run_configured_table_query(
        "account_activity",
        {
            "id_number": id_number,
            "customer_id": customer_id,
            "account_number": account_number,
            "period": period,
        },
        limit,
    )


@tool
def query_t24_credit_relationship_data(
    id_number: str = "",
    customer_id: str = "",
    facility_id: str = "",
    limit: int = DEFAULT_QUERY_LIMIT,
) -> str:
    """Query internal T24 loans for a retail customer."""

    return run_configured_table_query(
        "t24_credit_relationship",
        {
            "id_number": id_number,
            "customer_id": customer_id,
            "facility_id": facility_id,
        },
        limit,
    )


@tool
def query_credit_card_data(
    id_number: str = "",
    customer_id: str = "",
    card_id: str = "",
    limit: int = DEFAULT_QUERY_LIMIT,
) -> str:
    """Query internal credit card accounts (limit, balance, days past due)."""

    return run_configured_table_query(
        "credit_card",
        {
            "id_number": id_number,
            "customer_id": customer_id,
            "card_id": card_id,
        },
        limit,
    )


@tool
def query_cic_bureau_data(
    id_number: str = "",
    customer_id: str = "",
    bureau_report_date: str = "",
    limit: int = DEFAULT_QUERY_LIMIT,
) -> str:
    """Query CIC bureau credit data for a retail customer."""

    return run_configured_table_query(
        "cic_bureau",
        {
            "id_number": id_number,
            "customer_id": customer_id,
            "bureau_report_date": bureau_report_date,
        },
        limit,
    )


@tool
def query_collateral_data(
    id_number: str = "",
    customer_id: str = "",
    collateral_id: str = "",
    limit: int = DEFAULT_QUERY_LIMIT,
) -> str:
    """Query collateral records for a retail customer."""

    return run_configured_table_query(
        "collateral",
        {
            "id_number": id_number,
            "customer_id": customer_id,
            "collateral_id": collateral_id,
        },
        limit,
    )


@tool
def query_configured_table(
    logical_table: str,
    filters_json: str,
    limit: int = DEFAULT_QUERY_LIMIT,
) -> str:
    """Query any allowlisted logical table using JSON filters."""

    try:
        filters = json.loads(filters_json or "{}")
        if not isinstance(filters, dict):
            raise ValueError("filters_json must decode to a JSON object.")
    except Exception as exc:
        return format_database_tool_error("query_configured_table", exc)

    return run_configured_table_query(logical_table, filters, limit)


DATABASE_TOOLS = get_database_tools()
INCOME_DATABASE_TOOLS = [
    query_customer_profile,
    query_income_records,
    query_account_activity,
]
CUSTOMER_PROFILE_DATABASE_TOOLS = [
    query_customer_profile,
]
CREDIT_RELATIONSHIP_DATABASE_TOOLS = [
    query_customer_profile,
    query_t24_credit_relationship_data,
    query_credit_card_data,
    query_cic_bureau_data,
]
RISK_ASSESSMENT_DATABASE_TOOLS = [
    query_customer_profile,
    query_income_records,
    query_account_activity,
    query_t24_credit_relationship_data,
    query_credit_card_data,
    query_cic_bureau_data,
    query_collateral_data,
]


# Example: uncomment and adapt this to your existing database object.
# def database_executor(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
#     return database.query(sql, params)
#
# configure_database_executor(database_executor)

print("Database tools embedded:", [tool.name for tool in DATABASE_TOOLS])
