"""Shared plumbing for the LLM passes that turn a scanned document into JSON.

Five documents get this treatment — the financial statements, the credit
application, the two CIC reports and the site-visit report — and they all need
the same things: a prompt/LLM/JSON-parser chain, a check that the model returned
the shape that was asked for, a wrapper that never raises so a failed extraction
degrades to "use the raw OCR" instead of taking the whole run down, and one
reading of the money units the forms print.

Only that scaffolding lives here. The prompts and schemas stay in their own
modules, because those are where the domain knowledge is.
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import normalize_text


def resolve_money_multiplier(source_unit: Any) -> int:
    """Turn a unit a form printed into the multiplier that takes it to đồng.

    Matched on tokens rather than the whole string, because the two passes that
    used an exact-match table silently failed on the spellings nobody thought to
    list. "tỷ đồng" was in the table and "tỷ VNĐ" was not, so a plan written in
    tỷ VNĐ came through unscaled — off by a factor of a billion, and quiet about
    it. Any phrasing carrying the word survives this version.

    Unknown units mean 1, and the prompts say so explicitly: a page that prints
    no unit is in đồng. That default has to be stated in both places — a prompt
    that leaves it open invites the model to guess "triệu", and the guess is
    indistinguishable here from a page that really said triệu.
    """

    tokens = normalize_text(str(source_unit or "")).split()
    if "ty" in tokens:
        return 10 ** 9
    if "trieu" in tokens:
        return 10 ** 6
    if "nghin" in tokens:
        return 10 ** 3
    return 1


def build_extraction_chain(system_prompt: str, llm: Any):
    """Wire a system prompt to the LLM with a JSON-parsing tail."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
                Tên file: {filename}

                Văn bản OCR:
                {content}

                Trích xuất theo đúng schema JSON đã mô tả.
                """,
            ),
        ]
    )
    return prompt | llm | JsonOutputParser()


def run_extraction(
    chain: Any,
    filename: str,
    content: str,
    required_keys: set[str],
    missing_llm_message: str,
) -> tuple[dict[str, Any] | None, str]:
    """Invoke an extraction chain and validate its shape.

    Never raises. Returns ``(None, reason)`` on any failure so the caller always
    has a clean signal to fall back to the document's raw text — losing the
    structured view is a degradation, losing the document is a bug.
    """

    if chain is None:
        return None, missing_llm_message
    try:
        result = chain.invoke({"filename": filename, "content": content})
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:500]

    if not isinstance(result, dict):
        return None, f"Extraction returned non-dict result: {type(result).__name__}"
    missing = required_keys - result.keys()
    if missing:
        return None, f"Extraction result missing keys: {sorted(missing)}"
    return result, ""
