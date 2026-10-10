"""Shared types and small helpers for the underwriting workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict


AgentName = Literal[
    "INCOME_ANALYSIS_AGENT",
    "CUSTOMER_PROFILE_AGENT",
    "CREDIT_RELATIONSHIP_AGENT",
    "CREDIT_PROPOSAL_AGENT",
    "RISK_ASSESSMENT_AGENT",
    "CREDIT_MEMO",
]
WorkflowMode = Literal[
    "single_customer_profile",
    "single_credit_relationship",
    "single_income_analysis",
    "single_risk_assessment",
    "single_credit_proposal",
    "full_credit_memo",
]
DocumentAgentName = Literal[
    "INCOME_ANALYSIS_AGENT",
    "CUSTOMER_PROFILE_AGENT",
    "CREDIT_RELATIONSHIP_AGENT",
    "CREDIT_PROPOSAL_AGENT",
    "RISK_ASSESSMENT_AGENT",
    "GENERAL_CONTEXT",
]

# Agents a document can be routed to by the matrix. GENERAL_CONTEXT is
# deliberately excluded: it is the fallback bucket for documents no matrix row
# matched, never something the matrix itself assigns.
SPECIALIST_DOCUMENT_AGENTS = frozenset({
    "INCOME_ANALYSIS_AGENT",
    "CUSTOMER_PROFILE_AGENT",
    "CREDIT_RELATIONSHIP_AGENT",
    "CREDIT_PROPOSAL_AGENT",
    "RISK_ASSESSMENT_AGENT",
})
VALID_DOCUMENT_AGENTS = SPECIALIST_DOCUMENT_AGENTS | {"GENERAL_CONTEXT"}


@dataclass
class ClassifiedDocument:
    """Extracted document text plus its target agent classification."""

    path: str
    filename: str
    content: str
    agent: DocumentAgentName
    reasoning: str
    confidence: float
    file_hash: str = ""
    # One line naming this document for the reader of the finished memo, written
    # by the classifier from the text it already had in front of it. Empty when
    # the filename settled the type on its own and no LLM ever read the file.
    description: str = ""
    extraction_status: str = "success"
    extraction_error: str = ""
    classifier_error_type: str = ""
    classifier_error: str = ""
    # Matrix row this document matched (src/matrix/document_matrix.yaml), plus
    # the group it belongs to. Empty when nothing matched — the document then
    # falls back to agent == GENERAL_CONTEXT and is shared with every agent.
    document_type: str = ""
    document_group: str = ""
    # Keyword score per candidate document type, kept for monitoring: it shows
    # how close the runner-up was and whether the matrix needs a new keyword.
    type_scores: dict[str, float] = field(default_factory=dict)
    # {agent: "R"|"O"} straight from the matrix — who consumes this document and
    # how strongly. Drives both the fan-out and the prompt budget weighting.
    agent_relevance: dict[str, str] = field(default_factory=dict)
    # Which loan product agent_relevance above was resolved under. Empty means
    # no product was named in the request, so the strongest level across every
    # program was used — recorded per document so the monitoring output explains
    # its own R/O values instead of leaving them unattributable.
    loan_program: str = ""
    # Numeric mirror of agent_relevance (R -> 2, O -> 1) so ranking code can
    # sort by relevance without carrying the R/O vocabulary around.
    agent_scores: dict[str, float] = field(default_factory=dict)
    # Every agent this document is evidence for == sorted(agent_relevance).
    # Cached at classification time so it shows up in document_classifications
    # output for monitoring, and downstream consumers don't recompute it.
    relevant_agents: list[str] = field(default_factory=list)
    # True when document_type is flagged income_extraction in the matrix: a
    # labour contract, salary confirmation, bank statement, PIT return, rental
    # contract or household-business books. income_extraction holds the
    # structured monthly income record, or stays None on failure/no LLM
    # configured — callers must fall back to raw `content` in that case.
    is_income: bool = False
    # Same arrangement for the loan application form: purpose, amount, tenor,
    # repayment plan, declared income/expenses and collateral are pulled into
    # JSON so the proposal agent reads figures rather than OCR.
    is_proposal: bool = False
    # And for the borrower's CIC credit report, the only monthly debt series in
    # the whole file set. Flagged per CIC form rather than per issuer: the
    # collateral report (R20/R21) shares CIC's letterhead and nothing else, so it
    # needs its own pass, not this one.
    is_cic_s10a: bool = False
    # The other CIC form sharing that letterhead — R20/R21, the collateral
    # report. Same reasoning as is_cic_s10a: different structure, own pass.
    is_cic_r21: bool = False
    # The appraisal / field verification report. The odd one out: the others
    # are forms with fixed boxes, this is prose, so the pass separates what the
    # officer *verified* (residence, workplace, income, collateral) from what
    # the officer *concluded* — the conclusion block is opinion and agents must
    # not cite it as evidence.
    is_sitevisit: bool = False
    income_extraction: dict[str, Any] | None = None
    income_extraction_error: str = ""
    proposal_extraction: dict[str, Any] | None = None
    proposal_extraction_error: str = ""
    cic_s10a_extraction: dict[str, Any] | None = None
    cic_s10a_extraction_error: str = ""
    cic_r21_extraction: dict[str, Any] | None = None
    cic_r21_extraction_error: str = ""
    sitevisit_extraction: dict[str, Any] | None = None
    sitevisit_extraction_error: str = ""


class UnderwritingGraphState(TypedDict, total=False):
    """State passed between LangGraph workflow nodes."""

    query: str
    input_paths: list[str]
    conversation_history: list[dict[str, str]]
    history_context: str
    files: list[str]
    # Retail loan product named in the request (see detect_loan_program), or ""
    # when none was named / two were named. Selects which column of the document
    # matrix decides each document's R/O relevance, and which policy limits the
    # affordability and loan-sizing calculators apply.
    loan_program: str
    loan_program_detection: dict[str, Any]
    documents: list[ClassifiedDocument]
    document_routes: set[str]
    document_summary: str
    decision: dict[str, Any]
    workflow_mode: WorkflowMode
    gap_analysis: dict[str, Any]
    execution_plan: dict[str, Any]
    web_context: str
    steps: list[str]
    output_state: dict[str, Any]


def to_dict_list(items: list[Any]) -> list[dict[str, Any]]:
    """Convert dataclasses to JSON-friendly dictionaries."""

    return [
        asdict(item) if hasattr(item, "__dataclass_fields__") else item
        for item in items
    ]


def extract_text_from_agent_output(output: Any) -> str:
    """Normalize LangChain agent or chain output into plain text."""

    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if hasattr(output, "content"):
        return str(output.content)
    if isinstance(output, dict):
        for key in ["response", "output", "content"]:
            if key in output:
                return extract_text_from_agent_output(output[key])
        messages = output.get("messages")
        if messages:
            return extract_text_from_agent_output(messages[-1])
    return str(output)


def truncate_text(text: str, limit: int) -> str:
    """Truncate long prompt/evidence blocks."""

    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"
