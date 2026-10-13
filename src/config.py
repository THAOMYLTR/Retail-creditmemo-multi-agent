"""LLM client factory + runtime Config (extracted from the notebook).

Import-safe: constructing the LLM clients (config = Config(...)) stays in the
notebook driver cell so importing this module never needs live credentials.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI


def env_flag(name: str, default: str = "false") -> bool:
    """Read a boolean env var.

    ``os.getenv(name, False)`` returns the raw *string*, so "false" is truthy —
    read every boolean flag through here instead.
    """

    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


class CountingRateLimiter(InMemoryRateLimiter):
    """Rate limiter that also reports how long it made callers wait.

    Throttling is invisible from the outside: a run that spends a minute waiting
    for tokens looks identical to one that hung. These counters go into the run
    output so the delay is attributable, and so there is data to decide which
    calls to cut if the quota gets tighter.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stats_lock = threading.Lock()
        self.acquired_count = 0
        self.waited_seconds = 0.0

    def acquire(self, *, blocking: bool = True) -> bool:
        start = time.monotonic()
        result = super().acquire(blocking=blocking)
        with self._stats_lock:
            self.acquired_count += 1
            self.waited_seconds += time.monotonic() - start
        return result

    async def aacquire(self, *, blocking: bool = True) -> bool:
        start = time.monotonic()
        result = await super().aacquire(blocking=blocking)
        with self._stats_lock:
            self.acquired_count += 1
            self.waited_seconds += time.monotonic() - start
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            per_minute = self.requests_per_second * 60
            return {
                "requests_per_minute": round(per_minute, 2),
                "llm_calls": self.acquired_count,
                # Summed over every caller, so with parallel agents this exceeds
                # the run's wall-clock time — four threads blocked for ten
                # seconds each counts as forty.
                "throttled_seconds_all_threads": round(self.waited_seconds, 1),
                # The floor this quota puts under any run of this size, which is
                # the number worth comparing against the observed duration.
                "minimum_seconds_for_these_calls": (
                    round(self.acquired_count / per_minute * 60, 1)
                    if per_minute
                    else 0.0
                ),
            }


@lru_cache(maxsize=1)
def shared_rate_limiter() -> CountingRateLimiter:
    """The single rate limiter every LLM client must share.

    The provider's quota counts requests per *endpoint*, not per client object.
    Giving each of the seven clients its own limiter would multiply the budget
    sevenfold and defeat the whole thing — hence one cached instance.

    max_bucket_size=1 forbids bursting, which is what makes overrunning the
    quota impossible rather than merely unlikely: four parallel analysis agents
    still leave calls spaced a full interval apart.
    """

    per_minute = float(os.getenv("LLM_REQUESTS_PER_MINUTE", "9"))
    limiter = CountingRateLimiter(
        requests_per_second=per_minute / 60.0,
        check_every_n_seconds=0.5,
        max_bucket_size=1,
    )
    # The bucket starts empty, so without this the first call of every process
    # waits a full interval for nothing. One token of credit at startup stays
    # inside the headroom left below the real quota.
    limiter.available_tokens = 1.0
    return limiter


def build_llm(model_env: str,
              temperature: float = 0.1,
              timeout_env: str = "LLM_TIMEOUT_SECONDS"):
    """Build one ChatOpenAI client from environment variables."""

    model = os.getenv(model_env, "")

    kwargs = {
        "model": model,
        "base_url": os.getenv("OPENAI_API_BASE"),
        "temperature": temperature,
        "timeout": float(os.getenv(timeout_env, "60")),
        # The limiter keeps our own traffic under quota; retries are the safety
        # net for a 429 that arrives anyway. The OpenAI SDK backs off on 429 and
        # honours the Retry-After header.
        "max_retries": int(os.getenv("LLM_CLIENT_MAX_RETRIES", "3")),
        # Shared across every client — see shared_rate_limiter.
        "rate_limiter": shared_rate_limiter(),
    }
    return ChatOpenAI(**kwargs)

@dataclass
class Config:
    """Runtime config for the local notebook workflow."""

    decision_llm: Any = None
    document_llm: Any = None
    analysis_llm: Any = None
    credit_memo_llm: Any = None
    # Powers LocalGuardrails' input/output safety checks (see guardrails.py).
    guardrail_llm: Any = None
    income_extraction_llm: Any = None
    proposal_extraction_llm: Any = None
    cic_s10a_extraction_llm: Any = None
    cic_r21_extraction_llm: Any = None
    sitevisit_extraction_llm: Any = None
    # Corporate (KHDN) legal-information agent — src/agents/corp_income/.
    corp_income_extraction_llm: Any = None
    max_files: int = 50
    max_chars_per_document: int = 120_000
    max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_CONCURRENCY", "3"))
    )
    document_classifier_rule_confidence_threshold: float = 0.65
    enable_plan_and_execute: bool = True
    enable_self_ask_gap_analysis: bool = True
    enable_safety_guardrails: bool = field(
        default_factory=lambda: env_flag("RUN_SAFETY_GUARDRAILS")
    )
    enable_web_search: bool = field(
        default_factory=lambda: env_flag("RUN_WEB_SEARCH")
    )
    agent_input_char_budgets: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.agent_input_char_budgets is None:
            self.agent_input_char_budgets = {
                # Income analysis reads every proof-of-income document as
                # extracted JSON plus the application, so most of its budget
                # goes to structured blocks — a six-month bank statement with
                # recurring debits listed is the largest of them.
                "INCOME_ANALYSIS_AGENT": 40_000,
                "CUSTOMER_PROFILE_AGENT": 20_000,
                "CREDIT_RELATIONSHIP_AGENT": 20_000,
                # The routing matrix routes almost every document type to the
                # risk agent, so it sees more evidence than any other
                # specialist and needs headroom to match.
                "RISK_ASSESSMENT_AGENT": 40_000,
                # Two computed tables, the application JSON and the purpose and
                # collateral papers.
                "CREDIT_PROPOSAL_AGENT": 25_000,
                "CREDIT_MEMO": 80_000,
                "GENERAL_CONTEXT": 12_000,
            }

