"""Run the BRD extraction passes (items 1-4) on one classified corporate document.

Each document runs only the passes its type lists in ``document_types.yaml``.
Every result is scaled to đồng here — the only place amounts are converted —
and tagged with the document it came from, so the cross-checks downstream can
always say *which paper* disagreed with which.

Never raises: a failed pass is recorded as an error on the document and the
rest of the run carries on, the same contract as the retail passes
(``run_extraction`` in structured_extraction.py).
"""

from __future__ import annotations

from typing import Any

from src.agents.corp_income.prompts import PROMPTS, REQUIRED_KEYS
from src.agents.structured_extraction import (
    build_extraction_chain,
    resolve_money_multiplier,
    run_extraction,
)

# Per pass: (list key or None for top level, amount fields inside it).
_AMOUNT_FIELDS: dict[str, tuple[tuple[str | None, tuple[str, ...]], ...]] = {
    "legal_basic": ((None, ("charter_capital",)),),
    "shareholders": (
        (None, ("charter_capital",)),
        ("members", ("capital_amount",)),
        ("transfers", ("capital_amount",)),
    ),
    "legal_rep": (),
    "liabilities": (
        ("tax_debts", ("amount",)),
        ("social_insurance_debts", ("amount",)),
        ("administrative_penalties", ("amount",)),
        ("enforcement_actions", ("amount",)),
        ("disputes", ("claim_amount",)),
    ),
}


def _scale(value: Any, multiplier: int) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(value * multiplier, 2)


def normalize_amounts(pass_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Scale every amount of one pass result to đồng, in place."""

    multiplier = resolve_money_multiplier(result.get("source_unit"))
    for list_key, fields in _AMOUNT_FIELDS.get(pass_name, ()):
        rows = [result] if list_key is None else result.get(list_key) or []
        for row in rows:
            if isinstance(row, dict):
                for field_name in fields:
                    row[field_name] = _scale(row.get(field_name), multiplier)
    if "source_unit" in result:
        result["source_unit"] = "dong"
    return result


def build_chains(llm: Any) -> dict[str, Any]:
    """One JSON extraction chain per BRD pass, or ``{}`` without an LLM."""

    if llm is None:
        return {}
    return {name: build_extraction_chain(prompt, llm) for name, prompt in PROMPTS.items()}


def extract_document(
    chains: dict[str, Any],
    filename: str,
    content: str,
    passes: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Run ``passes`` on one document. Returns ``(results, errors)`` by pass name."""

    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for pass_name in passes:
        result, error = run_extraction(
            chains.get(pass_name),
            filename,
            content,
            REQUIRED_KEYS[pass_name],
            "No corp_income extraction LLM configured.",
        )
        if result is None:
            errors[pass_name] = error
            continue
        results[pass_name] = normalize_amounts(pass_name, result)
    return results, errors
