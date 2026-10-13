"""Optional LangSmith tracing wrapper (extracted from the notebook)."""

from __future__ import annotations

import os

from typing import Any


def _env_flag(name: str, default: str = "false") -> bool:
    """Read a boolean-like flag from environment variables."""

    return os.getenv(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


# Set True/False to force tracing on/off from this notebook.
# Keep None to read RUN_LANGSMITH_TRACE or LANGSMITH_TRACING from .env.
LANGSMITH_TRACE_OVERRIDE = None

if LANGSMITH_TRACE_OVERRIDE is None:
    RUN_LANGSMITH_TRACE = _env_flag(
        "RUN_LANGSMITH_TRACE",
        os.getenv("LANGSMITH_TRACING", "false"),
    )
else:
    RUN_LANGSMITH_TRACE = bool(LANGSMITH_TRACE_OVERRIDE)
LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "Retail_UW_Local_Notebook",
)
LANGSMITH_RUN_NAME = os.getenv(
    "LANGSMITH_RUN_NAME",
    "local_underwriting_workflow",
)

if RUN_LANGSMITH_TRACE and not os.getenv("LANGSMITH_API_KEY", "").strip():
    print("LangSmith tracing disabled: missing LANGSMITH_API_KEY.")
    RUN_LANGSMITH_TRACE = False

os.environ["LANGSMITH_TRACING"] = "true" if RUN_LANGSMITH_TRACE else "false"
os.environ["LANGCHAIN_TRACING_V2"] = (
    "true" if RUN_LANGSMITH_TRACE else "false"
)
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

try:
    from langsmith import traceable
except ImportError:
    traceable = None
    if RUN_LANGSMITH_TRACE:
        print("LangSmith tracing disabled: package 'langsmith' is missing.")
        RUN_LANGSMITH_TRACE = False
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


def run_supervisor_with_optional_trace(
    supervisor: Any,
    query: str,
    input_paths: list[str],
    conversation_history: list[dict[str, str]],
    corp_income_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the supervisor with an optional LangSmith root trace."""

    if not RUN_LANGSMITH_TRACE or traceable is None:
        return supervisor.process(
            query, input_paths, conversation_history,
            corp_income_options=corp_income_options,
        )

    @traceable(name=LANGSMITH_RUN_NAME, run_type="chain")
    def _run_traced_workflow(
        traced_query: str,
        traced_input_paths: list[str],
        traced_conversation_history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Create one parent run for the complete notebook workflow."""

        return supervisor.process(
            traced_query,
            traced_input_paths,
            traced_conversation_history,
            corp_income_options=corp_income_options,
        )

    return _run_traced_workflow(query, input_paths, conversation_history)


print(
    "LangSmith tracing:",
    {
        "enabled": RUN_LANGSMITH_TRACE,
        "project": LANGSMITH_PROJECT,
        "run_name": LANGSMITH_RUN_NAME,
    },
)
