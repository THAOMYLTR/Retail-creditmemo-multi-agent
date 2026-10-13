"""Supervisor orchestration for retail (individual borrower) underwriting.

Canonical AI-agent workflow: routing, document prep/classification, gap analysis,
memo workflow, finalize + triệu-VNĐ formatting. See docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import END, StateGraph

from src.utils.common import normalize_text
from src.utils.extractors import extract_document_text
from src.utils.formatting import convert_amounts_in_text
from src.utils.template_leak import check_template_leakage
from src.utils.citations import (
    AGENT_LABEL_PREFIXES,
    FootnoteAudit,
    consolidate_footnotes,
    format_footnote_findings,
    namespace_footnotes,
)
from src.utils.markdown_fixups import (
    ensure_blank_line_before_lists,
    tidy_numbers,
)

from src.config import Config, shared_rate_limiter
from src.types import (
    AgentName,
    ClassifiedDocument,
    UnderwritingGraphState,
    WorkflowMode,
    extract_text_from_agent_output,
    to_dict_list,
    truncate_text,
)
from src.agents.document_classification import (
    VALID_DOCUMENT_AGENTS,
    build_document_classification_prompt,
    compute_file_hash,
    detect_loan_program,
    discover_documents,
    rule_classify_document,
)
from src.matrix.document_matrix import (
    agent_relevance_for_type,
    get_type,
    is_income_type,
    is_cic_r21_type,
    is_sitevisit_type,
    is_cic_s10a_type,
    is_proposal_type,
    primary_agent_for_type,
)
from src.agents.income_extraction import (
    build_income_extraction_chain,
    extract_income_structured_data,
)
from src.agents.proposal_extraction import (
    build_proposal_extraction_chain,
    extract_proposal_structured_data,
)
from src.agents.cic_s10a_extraction import (
    build_cic_s10a_extraction_chain,
    extract_cic_s10a_structured_data,
)
from src.agents.cic_r21_extraction import (
    build_cic_r21_extraction_chain,
    extract_cic_r21_structured_data,
)
from src.agents.sitevisit_extraction import (
    build_sitevisit_extraction_chain,
    extract_sitevisit_structured_data,
)
from src.agents import prompt_blocks
from src.agents.corp_income import CorpIncomeAgent
from src.agents.specialist import (
    CustomerProfileAnalysis,
    CreditMemoComposerAgent,
    CreditProposalAnalysis,
    CreditRelationshipAnalysis,
    IncomeAnalysis,
    RiskAssessment,
)
from src.agents.guardrails import (
    LocalGuardrails,
    WebSearchProcessorAgent,
)

DECISION_SYSTEM_PROMPT = """
You are an expert in retail (individual borrower) loan underwriting at a bank.
Your task is to serve as a supervisor/planner for a multi-agent team.

Available routes:
- CUSTOMER_PROFILE_AGENT: borrower profile — identity, age, marital status and
  dependants, residence stability, employment or household business, and
  whether the stated loan purpose fits the person.
- INCOME_ANALYSIS_AGENT: income and repayment capacity — salary confirmations,
  payslips, labour contracts, bank statements, PIT returns, rental or business
  income, living costs, existing debt payments, DSR.
- CREDIT_RELATIONSHIP_AGENT: credit relationship analysis using internal T24
  credit data and the CIC credit report (loans, credit cards, debt group,
  overdue history).
- CREDIT_PROPOSAL_AGENT: loan proposal — product, amount, tenor, rate,
  repayment method, maximum loan by DSR/LTV/own contribution, collateral and
  credit conditions.
- RISK_ASSESSMENT_AGENT: credit risk assessment only. It returns risk analysis,
  not a Credit Memo.
- CREDIT_MEMO: create, draft, prepare, or generate a full underwriting Credit
  Memo / tờ trình / báo cáo thẩm định.

Rules:
- If uploaded documents are mostly proof of income and the user does not
  specify a task, prefer INCOME_ANALYSIS_AGENT.
- If uploaded documents are mostly identity/residence papers and the user does
  not specify a task, prefer CUSTOMER_PROFILE_AGENT.
- If the user only asks about credit relationship, T24, CIC, bureau, credit
  cards, existing loans, outstanding balance, or repayment history, use
  CREDIT_RELATIONSHIP_AGENT.
- If the user only asks how much can be lent, the loan amount, tenor, LTV or a
  loan proposal, use CREDIT_PROPOSAL_AGENT.
- If the user asks for risk assessment, credit risk, approval view, or risk
  conclusion, use RISK_ASSESSMENT_AGENT.
- If the user explicitly asks to create, draft, prepare, or generate a Credit
  Memo, tờ trình or báo cáo thẩm định, use CREDIT_MEMO.
- CREDIT_MEMO must run the full underwriting workflow:
  CUSTOMER_PROFILE_AGENT -> CREDIT_RELATIONSHIP_AGENT ->
  INCOME_ANALYSIS_AGENT -> CREDIT_PROPOSAL_AGENT ->
  RISK_ASSESSMENT_AGENT -> CREDIT_MEMO.
- Return JSON only with route, reasoning, and confidence.
"""

# Where routing lands when nothing else decides: the decision LLM returned an id
# that is not a route, or no keyword matched. Income analysis because a retail
# request with documents but no stated task is far more often "can this person
# afford it" than anything else.
DEFAULT_ROUTE: AgentName = "INCOME_ANALYSIS_AGENT"
DEFAULT_WORKFLOW_MODE: WorkflowMode = "single_income_analysis"

# Answer for a request that carries no documents and asks for no analysis — a
# greeting, or a question about the tool itself. Written out rather than routed
# to an agent: there is nothing to analyse, so an LLM call could only produce
# chat, and this system is not a chatbot.
OUT_OF_SCOPE_RESPONSE = (
    "Hệ thống này chỉ thực hiện thẩm định tín dụng khách hàng cá nhân "
    "(bán lẻ) trên hồ sơ khách hàng.\n\n"
    "Để bắt đầu, vui lòng:\n"
    "- Cung cấp hồ sơ (CCCD, xác nhận cư trú, hợp đồng lao động, xác nhận "
    "lương, sao kê tài khoản, báo cáo CIC, giấy đề nghị vay vốn, giấy tờ tài "
    "sản bảo đảm…), và\n"
    "- Nêu rõ yêu cầu, ví dụ: *\"Lập tờ trình thẩm định cho khách hàng vay mua "
    "nhà này\"* hoặc *\"Phân tích khả năng trả nợ\"*."
)

CREDIT_MEMO_KEYWORDS = [
    "credit memo",
    "báo cáo thẩm định",
    "báo cáo thẩm định tín dụng",
    "lập báo cáo thẩm định",
    "tạo báo cáo thẩm định",
    "soạn báo cáo thẩm định",
    "tờ trình",
    "lập tờ trình",
    "tờ trình thẩm định",
    "generate credit memo",
    "create credit memo",
    "draft credit memo",
    "prepare credit memo",
]
CREDIT_PROPOSAL_ROUTE_KEYWORDS = [
    "credit proposal",
    "loan proposal",
    "loan amount",
    "how much can",
    "maximum loan",
    "phân tích đề xuất tín dụng",
    "đề xuất tín dụng",
    "đề xuất cấp tín dụng",
    "đề xuất khoản vay",
    "số tiền vay tối đa",
    "số tiền cho vay",
    "vay tối đa",
    "cho vay tối đa",
    "được vay bao nhiêu",
    "hạn mức đề xuất",
    "số tiền đề xuất",
    "ltv",
]
RISK_ASSESSMENT_ROUTE_KEYWORDS = [
    "risk assessment",
    "credit risk",
    "risk analysis",
    "approval view",
    "underwriting conclusion",
    "đánh giá rủi ro",
    "rủi ro tín dụng",
    "thẩm định tín dụng",
    "nhận xét rủi ro",
    "phân tích rủi ro",
    "kết luận rủi ro",
    "phê duyệt",
]
INCOME_ROUTE_KEYWORDS = [
    "income",
    "affordability",
    "repayment capacity",
    "dsr",
    "dti",
    "thu nhập",
    "khả năng trả nợ",
    "nguồn trả nợ",
    "lương",
    "sao kê",
    "chi phí sinh hoạt",
    "phân tích tài chính",
    "tài chính cá nhân",
]
CUSTOMER_PROFILE_ROUTE_KEYWORDS = [
    "customer profile",
    "borrower profile",
    "kyc",
    "thông tin khách hàng",
    "hồ sơ khách hàng",
    "nhân thân",
    "cư trú",
    "hôn nhân",
    "nghề nghiệp",
    "công việc",
    "mục đích vay",
]
CREDIT_RELATIONSHIP_ROUTE_KEYWORDS = [
    "credit relationship",
    "credit history",
    "bureau",
    "cic",
    "t24",
    "credit card",
    "outstanding balance",
    "repayment status",
    "overdue",
    "quan hệ tín dụng",
    "lịch sử tín dụng",
    "thẻ tín dụng",
    "dư nợ",
    "nợ quá hạn",
    "nhóm nợ",
    "tổ chức tín dụng",
]
# Requests for the corporate (KHDN) legal-information agent. Matched on
# accent-stripped text (normalize_text) so "phap ly doanh nghiep" works as well
# as the accented form. Checked BEFORE every retail route and right after file
# discovery: a corporate dossier must never reach the retail classifier or the
# retail evidence-gap check, which would stop it for missing proof of income.
CORP_INCOME_ROUTE_KEYWORDS = [
    "corp income",
    "corp_income",
    "corporate legal",
    "phap ly doanh nghiep",
    "khach hang doanh nghiep",
    "khdn",
    "gcn dkdn",
    "dang ky doanh nghiep",
    "co cau co dong",
    "thanh vien gop von",
    "nguoi dai dien phap luat",
    "nguoi dai dien theo phap luat",
]
# Deliberately absent: bare "cổ đông", "nợ thuế", "nợ BHXH". A retail question
# ("khách hàng có nợ thuế TNCN không?") can contain them, and this route is
# checked first — a false match here would take a retail file off its pipeline.
VIETNAMESE_CHARS = set(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)



class Supervisor:
    """Local  supervisor without API, cache, or database dependencies."""


    # The computed tables are in triệu VNĐ while the extracted JSON blocks are
    # in đồng, so one salary appears as both "25,50" and "25500000" in the same
    # prompt. Said out loud whenever more than one of them is present.
    MIXED_UNIT_WARNING = (
        "LƯU Ý ĐƠN VỊ TIỀN: các khối dữ liệu dưới đây dùng ĐƠN VỊ KHÁC NHAU và "
        "mỗi khối tự ghi đơn vị của nó ở ngay đầu khối (bảng tính sẵn dùng "
        "triệu VNĐ, dữ liệu trích xuất dùng đồng). Đọc đúng đơn vị của khối "
        "đang trích. TUYỆT ĐỐI không lấy số của khối này đặt cạnh số của khối "
        "khác khi chưa quy đổi."
    )

    # Which agents read proof-of-income documents as structured JSON. Being in
    # this table has two inseparable consequences: the agent gets the
    # [DỮ LIỆU THU NHẬP ĐÃ TRÍCH XUẤT] block, AND the document's raw OCR is
    # replaced by a pointer to that block. Enabling only one of the two would
    # bill the same content twice, so both read this single table.
    INCOME_JSON_AGENTS = ("INCOME_ANALYSIS_AGENT",)

    # Same arrangement for the loan application form: these agents read it as
    # JSON and therefore do NOT get its raw OCR. Income analysis is here
    # because the declared incomes, living costs and existing debts are what it
    # checks the evidence against; the profile agent keeps the raw form for
    # the borrower details at the top of it.
    PROPOSAL_JSON_AGENTS = (
        "INCOME_ANALYSIS_AGENT",
        "CREDIT_PROPOSAL_AGENT",
        "RISK_ASSESSMENT_AGENT",
    )

    # And for the borrower's CIC report. Credit relationship is the obvious
    # consumer — the report *is* its subject matter. Risk assessment is here
    # because the debt group, the bad-debt history and the card balances are
    # its evidence. Same inseparable pair of consequences as the tables above.
    CIC_S10A_JSON_AGENTS = ("CREDIT_RELATIONSHIP_AGENT", "RISK_ASSESSMENT_AGENT")

    # And for the CIC R20/R21 collateral report — the two agents the matrix
    # routes cic_tai_san_bao_dam to.
    CIC_R21_JSON_AGENTS = ("CREDIT_RELATIONSHIP_AGENT", "RISK_ASSESSMENT_AGENT")

    # The appraisal report goes to everyone. It is the one document recording
    # what the bank itself verified: the profile agent takes residence and
    # employment, income analysis the verified income, credit proposal the
    # collateral check, credit relationship the officer's view of the
    # borrower's other lenders, and risk the officer's findings.
    SITEVISIT_JSON_AGENTS = (
        "CUSTOMER_PROFILE_AGENT",
        "INCOME_ANALYSIS_AGENT",
        "CREDIT_RELATIONSHIP_AGENT",
        "CREDIT_PROPOSAL_AGENT",
        "RISK_ASSESSMENT_AGENT",
    )

    # The prompt blocks live in src/agents/prompt_blocks.py. Rebound here so
    # call sites read them through Supervisor.
    AFFORDABILITY_BLOCK_AGENTS = prompt_blocks.AFFORDABILITY_BLOCK_AGENTS
    LOAN_SIZING_BLOCK_AGENTS = prompt_blocks.LOAN_SIZING_BLOCK_AGENTS
    AFFORDABILITY_BLOCK_HEADING = prompt_blocks.AFFORDABILITY_BLOCK_HEADING
    LOAN_SIZING_BLOCK_HEADING = prompt_blocks.LOAN_SIZING_BLOCK_HEADING
    INCOME_BLOCK_HEADING = prompt_blocks.INCOME_BLOCK_HEADING
    PROPOSAL_BLOCK_HEADING = prompt_blocks.PROPOSAL_BLOCK_HEADING
    CIC_S10A_BLOCK_HEADING = prompt_blocks.CIC_S10A_BLOCK_HEADING
    CIC_R21_BLOCK_HEADING = prompt_blocks.CIC_R21_BLOCK_HEADING
    SITEVISIT_BLOCK_HEADING = prompt_blocks.SITEVISIT_BLOCK_HEADING
    SOURCE_LIST_BLOCK_HEADING = prompt_blocks.SOURCE_LIST_BLOCK_HEADING
    DEBT_CHART_TITLE = prompt_blocks.DEBT_CHART_TITLE
    DEBT_CHART_TITLE_DEBT_ONLY = prompt_blocks.DEBT_CHART_TITLE_DEBT_ONLY
    DEBT_CHART_COLUMNS = prompt_blocks.DEBT_CHART_COLUMNS

    _document_block_header = staticmethod(prompt_blocks._document_block_header)
    _build_source_list_block = staticmethod(prompt_blocks._build_source_list_block)
    _build_affordability_block = staticmethod(prompt_blocks._build_affordability_block)
    _build_loan_sizing_block = staticmethod(prompt_blocks._build_loan_sizing_block)
    _build_income_structured_block = staticmethod(prompt_blocks._build_income_structured_block)
    _build_proposal_structured_block = staticmethod(prompt_blocks._build_proposal_structured_block)
    _build_debt_chart_block = staticmethod(prompt_blocks._build_debt_chart_block)
    _build_cic_s10a_structured_block = staticmethod(prompt_blocks._build_cic_s10a_structured_block)
    _build_cic_r21_structured_block = staticmethod(prompt_blocks._build_cic_r21_structured_block)
    _build_sitevisit_structured_block = staticmethod(prompt_blocks._build_sitevisit_structured_block)

    # Agents that need an extraction to have *run* without reading its JSON
    # block: the computed tables are built from every extracted document, so a
    # single-agent route still has to pay for the passes its table reads. Kept
    # apart from the *_JSON_AGENTS tables so the two reasons cannot be confused
    # later — being here does NOT strip the document's raw OCR.
    CALCULATOR_PASS_READERS: dict[str, tuple[str, ...]] = {
        "Income": prompt_blocks.AFFORDABILITY_BLOCK_AGENTS,
        "Proposal": prompt_blocks.AFFORDABILITY_BLOCK_AGENTS,
        "CIC S10A": prompt_blocks.AFFORDABILITY_BLOCK_AGENTS,
        "Sitevisit": prompt_blocks.AFFORDABILITY_BLOCK_AGENTS,
    }

    # The flag each extraction pass selects documents by. Single-sourced because
    # three things key off the same pass names: which passes a route needs, which
    # ones then run, and how many calls the skipped ones would have cost.
    EXTRACTION_PASS_FLAGS = {
        "Income": "is_income",
        "Proposal": "is_proposal",
        "CIC S10A": "is_cic_s10a",
        "CIC R21": "is_cic_r21",
        "Sitevisit": "is_sitevisit",
    }

    # Which agents each route actually runs. Structured extraction costs one LLM
    # call per matching document, so the run needs to know who is going to read
    # the result before it pays for it — see _passes_needed_for_route.
    ROUTE_AGENTS: dict[str, tuple[str, ...]] = {
        "CUSTOMER_PROFILE_AGENT": ("CUSTOMER_PROFILE_AGENT",),
        "CREDIT_RELATIONSHIP_AGENT": ("CREDIT_RELATIONSHIP_AGENT",),
        "INCOME_ANALYSIS_AGENT": ("INCOME_ANALYSIS_AGENT",),
        "RISK_ASSESSMENT_AGENT": ("RISK_ASSESSMENT_AGENT",),
        "CREDIT_PROPOSAL_AGENT": ("CREDIT_PROPOSAL_AGENT",),
        "CREDIT_MEMO": (
            "CUSTOMER_PROFILE_AGENT",
            "CREDIT_RELATIONSHIP_AGENT",
            "INCOME_ANALYSIS_AGENT",
            "CREDIT_PROPOSAL_AGENT",
            "RISK_ASSESSMENT_AGENT",
        ),
        # Runs its own OCR/classification/extraction (src/agents/corp_income),
        # so no retail extraction pass is needed for it.
        "CORP_INCOME_AGENT": ("CORP_INCOME_AGENT",),
    }

    # Where to anchor the debt/income chart, most specific first. The composer
    # renumbers headings when it merges the five specialist reports, so matching
    # on an exact "## 2. ..." string would work in a single-agent run and quietly
    # fail in a full memo. Matched on accent-stripped substrings of the heading.
    DEBT_CHART_ANCHORS = ("dien bien du no", "quan he tin dung")

    # Numeric form of the matrix's R/O relevance levels, so ranking and budget
    # code can compare relevance without carrying the R/O vocabulary around.
    RELEVANCE_SCORE = {"R": 2.0, "O": 1.0}

    # Relative per-document budget weights in _build_user_input. The matrix says
    # which documents an agent *requires* (R) versus may use (O); required
    # evidence must not be squeezed out by optional evidence. Documents that
    # matched no matrix type are shared with everyone and get the optional
    # weight — they are context, not this agent's core evidence.
    PRIMARY_DOC_BUDGET_WEIGHT = 3
    SECONDARY_DOC_BUDGET_WEIGHT = 1

    # Fixed prompt scaffolding around the document blocks. Named constants so
    # the budget arithmetic in _build_user_input measures exactly what it emits.
    DOC_SECTION_HEADER = "Uploaded document extracted content:\n\n"
    DOC_BLOCK_SEPARATOR = "\n\n---\n\n"

    def __init__(self, config: Config):
        self.config = config
        self.guardrails = (
            LocalGuardrails(config.guardrail_llm)
            if config.enable_safety_guardrails
            else None
        )
        self.web_search_agent = (
            WebSearchProcessorAgent()
            if config.enable_web_search
            else None
        )
        self.decision_chain = (
            self._build_decision_chain() 
            if config.decision_llm 
            else None
        )
        self.document_classifier_chain = (
            self._build_document_classifier_chain()
            if config.document_llm
            else None
        )
        self.income_extraction_chain = (
            build_income_extraction_chain(config.income_extraction_llm)
            if config.income_extraction_llm
            else None
        )
        self.proposal_extraction_chain = (
            build_proposal_extraction_chain(config.proposal_extraction_llm)
            if config.proposal_extraction_llm
            else None
        )
        self.cic_s10a_extraction_chain = (
            build_cic_s10a_extraction_chain(config.cic_s10a_extraction_llm)
            if config.cic_s10a_extraction_llm
            else None
        )
        self.cic_r21_extraction_chain = (
            build_cic_r21_extraction_chain(config.cic_r21_extraction_llm)
            if config.cic_r21_extraction_llm
            else None
        )
        self.sitevisit_extraction_chain = (
            build_sitevisit_extraction_chain(config.sitevisit_extraction_llm)
            if config.sitevisit_extraction_llm
            else None
        )
        self.corp_income_agent = CorpIncomeAgent(
            llm=config.corp_income_extraction_llm,
            max_workers=config.max_concurrency,
            max_chars_per_document=config.max_chars_per_document,
        )
        self.workflow_graph = self._build_workflow_graph()

    def _build_workflow_graph(self):
        """Build the deterministic LangGraph underwriting workflow."""

        workflow = StateGraph(UnderwritingGraphState)
        workflow.add_node("prepare_input", self._graph_prepare_input)
        workflow.add_node("input_guardrail", self._graph_input_guardrail)
        workflow.add_node("discover_documents", self._graph_discover_documents)
        workflow.add_node("classify_documents", self._graph_classify_documents)
        workflow.add_node("decide_workflow", self._graph_decide_workflow)
        workflow.add_node("evidence_gap_check", self._graph_evidence_gap_check)
        workflow.add_node("extract_documents", self._graph_extract_documents)
        workflow.add_node("web_search", self._graph_web_search)
        workflow.add_node(
            "single_customer_profile",
            self._graph_run_customer_profile,
        )
        workflow.add_node(
            "single_credit_relationship",
            self._graph_run_credit_relationship,
        )
        workflow.add_node(
            "single_income_analysis",
            self._graph_run_income_analysis,
        )
        workflow.add_node(
            "single_risk_assessment",
            self._graph_run_risk_assessment,
        )
        workflow.add_node(
            "single_credit_proposal",
            self._graph_run_credit_proposal,
        )
        workflow.add_node("full_credit_memo", self._graph_run_credit_memo)
        workflow.add_node("single_corp_income", self._graph_run_corp_income)

        workflow.set_entry_point("prepare_input")
        workflow.add_edge("prepare_input", "input_guardrail")
        workflow.add_conditional_edges(
            "input_guardrail",
            self._graph_after_input_guardrail,
            {"blocked": END, "continue": "discover_documents"},
        )
        workflow.add_conditional_edges(
            "discover_documents",
            self._graph_after_discover_documents,
            {
                "out_of_scope": END,
                "corp_income": "single_corp_income",
                "continue": "classify_documents",
            },
        )
        workflow.add_edge("classify_documents", "decide_workflow")
        workflow.add_edge("decide_workflow", "evidence_gap_check")
        workflow.add_conditional_edges(
            "evidence_gap_check",
            self._graph_after_evidence_gap_check,
            {"blocked": END, "continue": "extract_documents"},
        )
        workflow.add_edge("extract_documents", "web_search")
        workflow.add_conditional_edges(
            "web_search",
            self._graph_select_workflow_branch,
            {
                "single_customer_profile": "single_customer_profile",
                "single_credit_relationship": "single_credit_relationship",
                "single_income_analysis": "single_income_analysis",
                "single_risk_assessment": "single_risk_assessment",
                "single_credit_proposal": "single_credit_proposal",
                "full_credit_memo": "full_credit_memo",
            },
        )
        for node in [
            "single_customer_profile",
            "single_credit_relationship",
            "single_income_analysis",
            "single_risk_assessment",
            "single_credit_proposal",
            "full_credit_memo",
            "single_corp_income",
        ]:
            workflow.add_edge(node, END)
        
        return workflow.compile()

    def _graph_prepare_input(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Normalize the user request and conversation context."""

        query = state.get("query", "")
        input_paths = state.get("input_paths") or []
        conversation_history = state.get("conversation_history") or []
        steps = ["Received request"]
        history_context = self._format_history(conversation_history)
        steps.append("Built compact conversation context")
        return {
            **state,
            "query": query,
            "input_paths": input_paths,
            "conversation_history": conversation_history,
            "history_context": history_context,
            "steps": steps,
        }

    def _graph_input_guardrail(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Stop early when input guardrails reject the request."""

        query = state.get("query", "")
        steps = state.get("steps", [])
        if self.guardrails and query.strip():
            allowed, message = self.guardrails.check_input(query)
            steps.append("Checked input guardrails")
            if not allowed:
                return {
                    **state,
                    "steps": steps,
                    "output_state": self._build_state(
                        query,
                        message,
                        "INPUT_GUARDRAILS",
                        steps=steps + ["Blocked by input guardrails"],
                    ),
                }
        return {**state, "steps": steps}

    @staticmethod
    def _graph_after_input_guardrail(state: UnderwritingGraphState) -> str:
        """Choose whether to continue after input guardrail checks."""

        return "blocked" if state.get("output_state") else "continue"

    def _graph_discover_documents(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Discover supported files from the provided input paths."""

        files = discover_documents(
            state.get("input_paths") or [],
            self.config.max_files,
        )
        steps = state.get("steps", [])
        steps.append(f"Discovered {len(files)} supported file(s)")
        if self._is_corp_income_request(state.get("query", "")):
            # Decided here, before any OCR: the corporate route has its own
            # classifier and checks, and the retail ones would misread the file.
            steps.append("Selected route: CORP_INCOME_AGENT (corporate legal request)")
            return {
                **state,
                "files": files,
                "steps": steps,
                "decision": {
                    "route": "CORP_INCOME_AGENT",
                    "workflow_mode": "single_corp_income",
                    "reasoning": "Deterministic keyword route (corporate legal request).",
                    "confidence": 1.0,
                    "requires_credit_memo": False,
                },
                "workflow_mode": "single_corp_income",
            }
        if files or self._has_analysis_intent(state.get("query", "")):
            return {**state, "files": files, "steps": steps}

        # Nothing to analyse and nothing asked for — answer from a constant and
        # stop. Sitting here rather than earlier because only now is "no usable
        # document" certain: the caller may have passed a folder holding nothing
        # this system reads. Still ahead of every cost, since discovery only
        # lists filenames and all OCR belongs to classify_documents.
        steps.append("Out of scope: no documents and no analysis request")
        return {
            **state,
            "files": files,
            "steps": steps,
            "output_state": self._build_state(
                state.get("query", ""),
                OUT_OF_SCOPE_RESPONSE,
                "OUT_OF_SCOPE",
                {"route": "OUT_OF_SCOPE", "reasoning": "No documents, no analysis request."},
                [],
                {},
                {},
                {},
                "",
                steps,
            ),
        }

    @staticmethod
    def _graph_after_discover_documents(state: UnderwritingGraphState) -> str:
        """Stop when there is nothing to analyse and nothing was asked."""

        if state.get("output_state"):
            return "out_of_scope"
        if state.get("workflow_mode") == "single_corp_income":
            return "corp_income"
        return "continue"

    @staticmethod
    def _is_corp_income_request(query: str) -> bool:
        """True when the request asks for the corporate legal-information agent."""

        normalized = f" {normalize_text(query)} "
        return any(f" {keyword} " in normalized for keyword in CORP_INCOME_ROUTE_KEYWORDS)

    @classmethod
    def _has_analysis_intent(cls, query: str) -> bool:
        """True when the request names work one of the specialists can do.

        Read off the same keyword lists routing already uses, so a request that
        would have reached an agent can never be turned away here. Kept separate
        from "are there documents": asking for an income analysis without
        attaching payslips should still reach the evidence gap check, which
        answers with the specific documents that are missing.
        """

        normalized = (query or "").lower()
        return any(
            cls._contains_any(normalized, keywords)
            for keywords in (
                CREDIT_MEMO_KEYWORDS,
                CREDIT_PROPOSAL_ROUTE_KEYWORDS,
                RISK_ASSESSMENT_ROUTE_KEYWORDS,
                CREDIT_RELATIONSHIP_ROUTE_KEYWORDS,
                INCOME_ROUTE_KEYWORDS,
                CUSTOMER_PROFILE_ROUTE_KEYWORDS,
            )
        )

    def _graph_classify_documents(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Extract and classify uploaded documents."""

        steps = state.get("steps", [])
        # Resolved once per run, before any document is classified: it picks
        # which column of the matrix decides every document's R/O relevance.
        detection = detect_loan_program(
            state.get("query", ""),
            state.get("history_context", ""),
        )
        loan_program = detection["program"] or ""
        steps.append(self._format_loan_program_step(detection))
        documents = self._prepare_documents(
            state.get("files") or [],
            state.get("query", ""),
            state.get("history_context", ""),
            steps,
            loan_program,
        )
        # Structured extraction deliberately does NOT happen here. It costs one
        # LLM call per matching document, and which of those results anybody
        # reads depends on the route, which is not decided until two nodes from
        # now — so it runs in _graph_extract_documents instead.
        document_routes: set[str] = set()
        for doc in documents:
            if doc.agent == "GENERAL_CONTEXT":
                document_routes.add("GENERAL_CONTEXT")
                continue
            document_routes.update(doc.relevant_agents)
        document_summary = self._format_document_summary(documents)
        return {
            **state,
            "loan_program": loan_program,
            "loan_program_detection": detection,
            "documents": documents,
            "document_routes": document_routes,
            "document_summary": document_summary,
        }

    @staticmethod
    def _format_loan_program_step(detection: dict[str, Any]) -> str:
        """Explain the loan program decision in the run's step log.

        Detection comes from free text the user typed, so it has to be visible:
        a silently wrong (or silently absent) program would quietly change how
        documents are prioritised with nothing in the output to point at.
        """

        if detection["program"]:
            return (
                f"Loan program: {detection['program']} "
                f"(matched \"{detection['matched_alias']}\" in the "
                f"{detection['source']})"
            )
        if detection["candidates"]:
            return (
                "Loan program: ambiguous — the request names "
                f"{', '.join(detection['candidates'])}. Using the strongest "
                "relevance across all programs."
            )
        return (
            "Loan program: not specified in the request. Using the strongest "
            "relevance across all programs (documents can only be "
            "over-prioritised, never dropped)."
        )

    def _graph_decide_workflow(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Select route and deterministic workflow mode."""

        documents = state.get("documents") or []
        decision = self._decide(
            state.get("query", ""),
            bool(documents),
            state.get("history_context", ""),
            state.get("document_summary", ""),
            state.get("document_routes") or set(),
        )
        steps = state.get("steps", [])
        steps.append(f"Selected route: {decision['route']}")
        steps.append(f"Selected workflow mode: {decision['workflow_mode']}")
        return {
            **state,
            "decision": decision,
            "workflow_mode": decision["workflow_mode"],
            "steps": steps,
        }

    def _graph_evidence_gap_check(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Analyze required evidence before running expensive agents."""

        query = state.get("query", "")
        decision = state.get("decision") or {}
        documents = state.get("documents") or []
        steps = state.get("steps", [])
        gap_analysis = self._analyze_evidence_gaps(
            documents,
            decision.get("route", DEFAULT_ROUTE),
        )
        execution_plan = self._build_execution_plan(decision, gap_analysis)
        steps.append("Built Self-Ask evidence gap analysis")
        steps.append("Built LangGraph execution plan")
        if not execution_plan.get("can_answer_now", True):
            response = self._missing_evidence_response(query, gap_analysis)
            output_state = self._build_state(
                query,
                response,
                "EVIDENCE_GAP_CHECK",
                decision,
                documents,
                {},
                execution_plan,
                gap_analysis,
                "",
                steps
                + [
                    "Stopped before agent execution because required evidence "
                    "is missing"
                ],
            )
            return {
                **state,
                "gap_analysis": gap_analysis,
                "execution_plan": execution_plan,
                "steps": steps,
                "output_state": output_state,
            }
        return {
            **state,
            "gap_analysis": gap_analysis,
            "execution_plan": execution_plan,
            "steps": steps,
        }

    @staticmethod
    def _graph_after_evidence_gap_check(
        state: UnderwritingGraphState,
    ) -> str:
        """Choose whether missing required evidence blocks execution."""

        return "blocked" if state.get("output_state") else "continue"

    def _graph_extract_documents(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Run only the structured extractions this route will actually read.

        Placed after the evidence gap check rather than beside classification so
        two kinds of waste disappear: a run blocked for missing evidence pays for
        no extraction at all, and a route whose agents never read a given block
        does not pay for it either — a single CUSTOMER_PROFILE run would otherwise
        spend an LLM call per CIC report and bank statement that nothing
        downstream then opened.

        Safe to defer because nothing between classification and here reads an
        extraction *result*: routing and the gap check both work off the document
        types the matrix assigned.
        """

        documents = state.get("documents") or []
        steps = state.get("steps", [])
        route = (state.get("decision") or {}).get("route", DEFAULT_ROUTE)
        needed = self._passes_needed_for_route(route)

        for label, run_pass in (
            ("Income", self._extract_income_documents),
            ("Proposal", self._extract_proposal_documents),
            ("CIC S10A", self._extract_cic_s10a_documents),
            ("CIC R21", self._extract_cic_r21_documents),
            ("Sitevisit", self._extract_sitevisit_documents),
        ):
            if label in needed:
                run_pass(documents, steps)

        skipped = self._describe_skipped_passes(documents, needed)
        if skipped:
            # Say it out loud: a missing extraction block otherwise looks
            # identical to one that failed, and the two need different fixes.
            steps.append(
                f"Saved LLM calls — no agent on route {route} reads: {skipped}"
            )
        return {
            **state,
            "documents": documents,
            # Rebuilt because the tags in it report extraction state, which only
            # exists now — the copy the routing node read was written before any
            # of this ran.
            "document_summary": self._format_document_summary(documents),
            "steps": steps,
        }

    @classmethod
    def _passes_needed_for_route(cls, route: str) -> set[str]:
        """Which extraction passes the agents on this route actually consume."""

        agents = set(cls.ROUTE_AGENTS.get(route, ()))
        consumers = {
            "Income": set(cls.INCOME_JSON_AGENTS),
            "Proposal": set(cls.PROPOSAL_JSON_AGENTS),
            "CIC S10A": set(cls.CIC_S10A_JSON_AGENTS),
            "CIC R21": set(cls.CIC_R21_JSON_AGENTS),
            "Sitevisit": set(cls.SITEVISIT_JSON_AGENTS),
        }
        # The computed tables read several passes without the agent reading
        # their JSON — the affordability table needs income, the application,
        # CIC and the appraisal report whoever it is shown to.
        for label, readers in cls.CALCULATOR_PASS_READERS.items():
            consumers[label] |= set(readers)
        return {
            label
            for label, readers in consumers.items()
            if agents & readers
        }

    @classmethod
    def _describe_skipped_passes(
        cls,
        documents: list[ClassifiedDocument],
        needed: set[str],
    ) -> str:
        """Name the skipped passes and how many documents each would have cost.

        Counts documents rather than just naming passes so the step log shows the
        saving, not merely the decision.
        """

        parts = []
        for label, flag_attr in cls.EXTRACTION_PASS_FLAGS.items():
            if label in needed:
                continue
            count = sum(1 for doc in documents if getattr(doc, flag_attr))
            if count:
                parts.append(f"{label} ({count} document(s), {count} LLM call(s))")
        return ", ".join(parts)

    def _graph_web_search(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Optionally enrich state with web search context."""

        web_context = self._maybe_run_web_search(
            state.get("query", ""),
            state.get("conversation_history") or [],
            state.get("decision") or {},
            state.get("document_summary", ""),
            state.get("steps", []),
        )
        return {**state, "web_context": web_context}

    @staticmethod
    def _graph_select_workflow_branch(
        state: UnderwritingGraphState,
    ) -> WorkflowMode:
        """Route graph execution by deterministic workflow mode."""

        return state.get("workflow_mode", DEFAULT_WORKFLOW_MODE)

    def _graph_run_customer_profile(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Run the single customer profile branch."""

        return {
            **state,
            "output_state": self._run_single_agent_branch(
                state,
                "CUSTOMER_PROFILE_AGENT",
            ),
        }

    def _graph_run_credit_relationship(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Run the single credit relationship branch."""

        return {
            **state,
            "output_state": self._run_single_agent_branch(
                state,
                "CREDIT_RELATIONSHIP_AGENT",
            ),
        }

    def _graph_run_income_analysis(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Run the single income analysis branch."""

        return {
            **state,
            "output_state": self._run_single_agent_branch(
                state,
                "INCOME_ANALYSIS_AGENT",
            ),
        }

    def _graph_run_risk_assessment(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Run the single risk assessment branch."""

        return {
            **state,
            "output_state": self._run_single_agent_branch(
                state,
                "RISK_ASSESSMENT_AGENT",
            ),
        }

    def _graph_run_credit_proposal(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Run the standalone Credit Proposal branch."""

        return {
            **state,
            "output_state": self._run_single_agent(
                "CREDIT_PROPOSAL_AGENT",
                state.get("query", ""),
                state.get("history_context", ""),
                state.get("decision") or {},
                state.get("documents") or [],
                state.get("web_context", ""),
                state.get("execution_plan") or {},
                state.get("gap_analysis") or {},
                state.get("steps", []),
            ),
        }

    def _graph_run_credit_memo(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Run the full Credit Memo underwriting branch."""

        return {
            **state,
            "output_state": self._run_credit_memo_workflow(
                state.get("query", ""),
                state.get("history_context", ""),
                state.get("decision") or {},
                state.get("documents") or [],
                state.get("web_context", ""),
                state.get("execution_plan") or {},
                state.get("gap_analysis") or {},
                state.get("steps", []),
            ),
        }

    def _graph_run_corp_income(
        self,
        state: UnderwritingGraphState,
    ) -> UnderwritingGraphState:
        """Run the corporate legal-information agent (BRD Corp Income items 1-4).

        Its report is returned as written, not passed through _finalize: that
        step converts đồng to triệu VNĐ, consolidates footnotes and inserts the
        retail debt/income chart, none of which applies to this report — its
        amounts already carry a unit ("10.000.000.000 đ") and converting them
        again would garble them. Output guardrails still apply.
        """

        query = state.get("query", "")
        steps = list(state.get("steps", []))
        options = state.get("corp_income_options") or {}
        try:
            result = self.corp_income_agent.run(state.get("files") or [], **options)
        except Exception as exc:  # bad options, e.g. an unknown type override
            message = f"corp_income không chạy được: {type(exc).__name__}: {exc}"
            output = self._build_state(
                query, message, "CORP_INCOME_AGENT", state.get("decision") or {},
                steps=steps + ["corp_income failed"],
            )
            output["status"] = "error"
            return {**state, "output_state": output}

        response = result.markdown
        agent_name = "CORP_INCOME_AGENT"
        if self.guardrails:
            allowed, response = self.guardrails.check_output(response, query)
            if not allowed:
                agent_name = "OUTPUT_GUARDRAILS"

        classified = sum(1 for doc in result.documents if doc["type"] != "khac")
        failed = [doc["filename"] for doc in result.documents if doc["errors"]]
        steps.append(
            f"corp_income: {len(result.documents)} file(s), {classified} classified, "
            f"{len(result.records)} extraction record(s)"
        )
        if failed:
            steps.append(f"corp_income: errors on {', '.join(failed)}")
        rate_limit = shared_rate_limiter().snapshot()
        steps.append(
            f"LLM: {rate_limit['llm_calls']} calls at a "
            f"{rate_limit['requests_per_minute']}/min limit"
        )
        output = self._build_state(
            query,
            response,
            agent_name,
            state.get("decision") or {},
            steps=steps + ["Built final response"],
            rate_limit=rate_limit,
        )
        # Full structured result (documents, records, findings per item) for
        # monitoring or for writing into a Databricks table.
        output["corp_income"] = result.to_dict()
        return {**state, "output_state": output}

    def _run_single_agent_branch(
        self,
        state: UnderwritingGraphState,
        agent_name: str,
    ) -> dict[str, Any]:
        """Adapt graph state to the existing single-agent runner."""

        return self._run_single_agent(
            agent_name,
            state.get("query", ""),
            state.get("history_context", ""),
            state.get("decision") or {},
            state.get("documents") or [],
            state.get("web_context", ""),
            state.get("execution_plan") or {},
            state.get("gap_analysis") or {},
            state.get("steps", []),
        )

    def process(
        self,
        query: str,
        input_paths: list[str] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        corp_income_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one local notebook request through the agent workflow.

        ``corp_income_options`` only matters on the corporate route — keyword
        arguments for CorpIncomeAgent.run (loan_date, borrower_names,
        internal_data, external_data, watchlist, type_overrides).
        """

        initial_state: UnderwritingGraphState = {
            "query": query,
            "input_paths": input_paths or [],
            "conversation_history": conversation_history or [],
            "corp_income_options": corp_income_options or {},
        }
        result = self.workflow_graph.invoke(initial_state)
        output_state = result.get("output_state")
        if output_state:
            return output_state
        return self._build_state(
            query,
            "Workflow graph finished without an output state.",
            "WORKFLOW_GRAPH",
            steps=result.get("steps", []),
        )

    def _prepare_documents(
        self,
        files: list[str],
        input_text: str,
        history_context: str,
        steps: list[str],
        loan_program: str = "",
    ) -> list[ClassifiedDocument]:
        documents = []
        seen_hashes: set[str] = set()
        for file_path in files:
            filename = Path(file_path).name
            file_hash = compute_file_hash(file_path)
            if file_hash in seen_hashes:
                steps.append(f"Skipped duplicate document: {filename}")
                continue
            seen_hashes.add(file_hash)
            try:
                steps.append(f"Extracting document content: {filename}")
                content = extract_document_text(
                    file_path,
                    ocr_timeout_seconds=float(
                        os.getenv("OCR_TIMEOUT_SECONDS", "120")
                    ),
                )
                extraction_status = "success"
                extraction_error = ""
            except Exception as exc:
                content = (
                    f"[Document text extraction failed for {filename}: {exc}]"
                )
                extraction_status = "failed"
                extraction_error = f"{type(exc).__name__}: {exc}"

            content = truncate_text(
                content,
                self.config.max_chars_per_document,
            )
            classification = self._classify_document(
                filename,
                content,
                input_text,
                history_context,
            )
            document_type = classification.get("document_type", "")
            type_scores = classification.get("scores", {})
            # The matrix decides the fan-out. A document that matched no type
            # stays GENERAL_CONTEXT, which _select_documents_for_agent already
            # shares with every agent, so nothing is dropped on a miss.
            # An empty loan_program resolves to the strongest level across all
            # programs — see agent_relevance_for_type.
            agent_relevance = agent_relevance_for_type(
                document_type,
                loan_program or None,
            )
            relevant_agents = sorted(agent_relevance)
            matched = get_type(document_type)
            agent = primary_agent_for_type(document_type) or "GENERAL_CONTEXT"
            is_income = is_income_type(document_type)
            is_proposal = is_proposal_type(document_type)
            is_cic_s10a = is_cic_s10a_type(document_type)
            is_cic_r21 = is_cic_r21_type(document_type)
            is_sitevisit = is_sitevisit_type(document_type)
            steps.append(
                f"Classified document: {filename} -> "
                + (f"{document_type} " if document_type else "(no type matched) ")
                + f"-> {', '.join(relevant_agents) or 'GENERAL_CONTEXT'}"
                + (" [THU NHẬP]" if is_income else "")
            )
            documents.append(
                ClassifiedDocument(
                    path=file_path,
                    filename=filename,
                    content=content,
                    agent=agent,
                    reasoning=classification.get("reasoning", ""),
                    confidence=float(classification.get("confidence", 0.0)),
                    file_hash=file_hash,
                    description=str(classification.get("description", "")).strip(),
                    extraction_status=extraction_status,
                    extraction_error=extraction_error,
                    classifier_error_type=classification.get(
                        "classifier_error_type",
                        "",
                    ),
                    classifier_error=classification.get("classifier_error", ""),
                    document_type=document_type,
                    document_group=matched.group_id if matched else "",
                    type_scores=type_scores,
                    agent_relevance=agent_relevance,
                    loan_program=loan_program,
                    agent_scores={
                        name: self.RELEVANCE_SCORE[level]
                        for name, level in agent_relevance.items()
                    },
                    relevant_agents=relevant_agents,
                    is_income=is_income,
                    is_proposal=is_proposal,
                    is_cic_s10a=is_cic_s10a,
                    is_cic_r21=is_cic_r21,
                    is_sitevisit=is_sitevisit,
                )
            )
        return documents

    def _run_structured_extraction(
        self,
        documents: list[ClassifiedDocument],
        steps: list[str],
        *,
        flag_attr: str,
        chain: Any,
        extract: Any,
        result_attr: str,
        error_attr: str,
        label: str,
        missing_llm_message: str,
    ) -> None:
        """Run one structured-extraction pass over the documents it applies to.

        Several files can need the same pass in one run (six monthly payslips,
        say), so it goes concurrently, bounded by max_concurrency —
        the same pattern as the parallel analysis agents. Results are written
        onto each document in place and nothing raises, so a failed or
        unconfigured extraction always leaves _build_user_input a clean signal
        to fall back to the raw OCR text.

        Shared by all five passes: they differ only in which documents they
        apply to and where the result is stored.
        """

        targets = [doc for doc in documents if getattr(doc, flag_attr)]
        if not targets:
            return
        if not chain:
            for doc in targets:
                setattr(doc, error_attr, missing_llm_message)
            steps.append(
                f"Skipped {label} extraction for {len(targets)} document(s): "
                f"{missing_llm_message}"
            )
            return

        def _run(doc: ClassifiedDocument):
            result, error = extract(chain, doc.filename, doc.content, doc.path)
            return doc, result, error

        max_workers = max(1, min(len(targets), self.config.max_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run, doc) for doc in targets]
            for future in futures:
                doc, result, error = future.result()
                setattr(doc, result_attr, result)
                setattr(doc, error_attr, error)
                steps.append(
                    f"{label} extraction: {doc.filename} -> "
                    + ("ok" if result is not None else f"failed: {error}")
                )

    def _extract_income_documents(
        self,
        documents: list[ClassifiedDocument],
        steps: list[str],
    ) -> None:
        self._run_structured_extraction(
            documents,
            steps,
            flag_attr="is_income",
            chain=self.income_extraction_chain,
            extract=extract_income_structured_data,
            result_attr="income_extraction",
            error_attr="income_extraction_error",
            label="Income",
            missing_llm_message="No income_extraction_llm configured.",
        )

    def _extract_proposal_documents(
        self,
        documents: list[ClassifiedDocument],
        steps: list[str],
    ) -> None:
        self._run_structured_extraction(
            documents,
            steps,
            flag_attr="is_proposal",
            chain=self.proposal_extraction_chain,
            extract=extract_proposal_structured_data,
            result_attr="proposal_extraction",
            error_attr="proposal_extraction_error",
            label="Proposal",
            missing_llm_message="No proposal_extraction_llm configured.",
        )

    def _extract_cic_s10a_documents(
        self,
        documents: list[ClassifiedDocument],
        steps: list[str],
    ) -> None:
        self._run_structured_extraction(
            documents,
            steps,
            flag_attr="is_cic_s10a",
            chain=self.cic_s10a_extraction_chain,
            extract=extract_cic_s10a_structured_data,
            result_attr="cic_s10a_extraction",
            error_attr="cic_s10a_extraction_error",
            label="CIC S10A",
            missing_llm_message="No cic_s10a_extraction_llm configured.",
        )

    def _extract_cic_r21_documents(
        self,
        documents: list[ClassifiedDocument],
        steps: list[str],
    ) -> None:
        self._run_structured_extraction(
            documents,
            steps,
            flag_attr="is_cic_r21",
            chain=self.cic_r21_extraction_chain,
            extract=extract_cic_r21_structured_data,
            result_attr="cic_r21_extraction",
            error_attr="cic_r21_extraction_error",
            label="CIC R21",
            missing_llm_message="No cic_r21_extraction_llm configured.",
        )

    def _extract_sitevisit_documents(
        self,
        documents: list[ClassifiedDocument],
        steps: list[str],
    ) -> None:
        self._run_structured_extraction(
            documents,
            steps,
            flag_attr="is_sitevisit",
            chain=self.sitevisit_extraction_chain,
            extract=extract_sitevisit_structured_data,
            result_attr="sitevisit_extraction",
            error_attr="sitevisit_extraction_error",
            label="Sitevisit",
            missing_llm_message="No sitevisit_extraction_llm configured.",
        )

    @staticmethod
    def _document_sample(content: str, limit: int = 2_400) -> str:
        """Sample head+middle+tail so classification is not cover-page biased."""

        text = content or ""
        if len(text) <= limit:
            return text
        chunk = limit // 3
        head = text[:chunk]
        mid_start = max(chunk, (len(text) - chunk) // 2)
        middle = text[mid_start:mid_start + chunk]
        tail = text[-chunk:]
        return f"{head}\n...\n{middle}\n...\n{tail}"

    def _classify_document(
        self,
        filename: str,
        content: str,
        input_text: str,
        history_context: str,
    ) -> dict[str, Any]:
        rule = rule_classify_document(filename, content)
        if rule["document_type"] and (
            rule["confidence"]
            >= self.config.document_classifier_rule_confidence_threshold
            # Even a low-confidence rule result is worth keeping when every
            # plausible type routes to the same agents: the LLM could only
            # change the label, not where the document goes.
            or rule["routing_unambiguous"]
            # Or when the filename names exactly one type and the body agreed.
            # Confidence is a margin measure, so a well-named file whose text
            # happens to quote a neighbouring type's vocabulary can dip under
            # the threshold with nothing actually in doubt.
            or rule["filename_decisive"]
        ):
            return rule

        if self.document_classifier_chain:
            try:
                result = self.document_classifier_chain.invoke(
                    {
                        "input_text": input_text or "No explicit user request.",
                        "history_context": (
                            history_context or "No previous conversation."
                        ),
                        "filename": filename,
                        "content_sample": (
                            self._document_sample(content)
                            or "No text extracted from the document."
                        ),
                    }
                )
                document_type = result.get("document_type") or ""
                # An id the matrix doesn't define would silently route the
                # document nowhere, so fall back to the rule result instead.
                if document_type and get_type(document_type) is None:
                    return rule
                result["document_type"] = document_type
                # Keyword scores always come from the rule pass, so monitoring
                # can see the runner-up types even when the LLM label wins.
                result["scores"] = rule.get("scores", {})
                return result
            except Exception as exc:
                rule["classifier_error_type"] = type(exc).__name__
                rule["classifier_error"] = str(exc)[:500]
                rule["reasoning"] = (
                    "Fallback keyword classification because document "
                    "classifier failed."
                )
        return rule

    def _build_decision_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DECISION_SYSTEM_PROMPT),
                (
                    "human",
                    """
                    Recent conversation history:
                    {history_context}

                    User query: {input_text}

                    Has uploaded file: {has_file}

                    Uploaded document classification summary:
                    {document_summary}

                    Choose the route.
                    """,
                ),
            ]
        )
        return prompt | self.config.decision_llm | JsonOutputParser()

    def _build_document_classifier_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", build_document_classification_prompt()),
                (
                    "human",
                    """
                    User request:
                    {input_text}

                    Recent conversation history:
                    {history_context}

                    Filename: {filename}

                    Extracted document text sample:
                    {content_sample}

                    Classify this document.
                    """,
                ),
            ]
        )
        return prompt | self.config.document_llm | JsonOutputParser()

    def _deterministic_route(self, input_text: str) -> AgentName | None:
        """Return a route decided purely by keywords, or None if ambiguous.

        Mirrors the keyword branches of ``_override_route`` (which already take
        precedence over the decision LLM). When a keyword matches we can skip the
        decision-LLM round-trip entirely — the routing outcome is identical.
        """
        normalized = input_text.lower()
        keyword_routes = [
            (CREDIT_MEMO_KEYWORDS, "CREDIT_MEMO"),
            (CREDIT_PROPOSAL_ROUTE_KEYWORDS, "CREDIT_PROPOSAL_AGENT"),
            (RISK_ASSESSMENT_ROUTE_KEYWORDS, "RISK_ASSESSMENT_AGENT"),
            (CREDIT_RELATIONSHIP_ROUTE_KEYWORDS, "CREDIT_RELATIONSHIP_AGENT"),
            (INCOME_ROUTE_KEYWORDS, "INCOME_ANALYSIS_AGENT"),
            (CUSTOMER_PROFILE_ROUTE_KEYWORDS, "CUSTOMER_PROFILE_AGENT"),
        ]
        for keywords, route in keyword_routes:
            if self._contains_any(normalized, keywords):
                return route
        return None

    def _decide(
        self,
        input_text: str,
        has_file: bool,
        history_context: str,
        document_summary: str,
        document_routes: set[str],
    ) -> dict[str, Any]:
        deterministic_route = self._deterministic_route(input_text)
        if deterministic_route is not None:
            decision = {
                "reasoning": "Deterministic keyword route (decision LLM skipped).",
                "confidence": 1.0,
            }
            route = deterministic_route
        elif self.decision_chain:
            try:
                decision = self.decision_chain.invoke(
                    {
                        "input_text": input_text,
                        "has_file": str(has_file),
                        "history_context": (
                            history_context or "No previous conversation."
                        ),
                        "document_summary": (
                            document_summary or "No uploaded documents."
                        ),
                    }
                )
                route = decision.get("route", DEFAULT_ROUTE)
            except Exception:
                decision = {
                    "reasoning": (
                        "Fallback route selected because supervisor "
                        "classification failed."
                    ),
                    "confidence": 0.0,
                }
                route = self._fallback_route(input_text, has_file)
        else:
            decision = {
                "reasoning": (
                    "Rule-based route selected because decision LLM is "
                    "not configured."
                ),
                "confidence": 0.0,
            }
            route = self._fallback_route(input_text, has_file)

        route = self._override_route(
            route,
            input_text,
            has_file,
            document_routes,
        )
        workflow_mode = self._workflow_mode_for_route(route)
        return {
            "route": route,
            "workflow_mode": workflow_mode,
            "reasoning": decision.get("reasoning", ""),
            "confidence": float(decision.get("confidence", 0.0)),
            "requires_credit_memo": route == "CREDIT_MEMO",
        }

    @staticmethod
    def _workflow_mode_for_route(route: str) -> WorkflowMode:
        """Map a route to the deterministic workflow branch."""

        route_modes = {
            "CUSTOMER_PROFILE_AGENT": "single_customer_profile",
            "CREDIT_RELATIONSHIP_AGENT": "single_credit_relationship",
            "INCOME_ANALYSIS_AGENT": "single_income_analysis",
            "RISK_ASSESSMENT_AGENT": "single_risk_assessment",
            "CREDIT_PROPOSAL_AGENT": "single_credit_proposal",
            "CREDIT_MEMO": "full_credit_memo",
            "CORP_INCOME_AGENT": "single_corp_income",
        }
        return route_modes.get(route, DEFAULT_WORKFLOW_MODE)

    def _override_route(
        self,
        route: str,
        input_text: str,
        has_file: bool,
        document_routes: set[str],
    ) -> AgentName:
        normalized = input_text.lower()
        if self._contains_any(normalized, CREDIT_MEMO_KEYWORDS):
            return "CREDIT_MEMO"
        if self._contains_any(normalized, CREDIT_PROPOSAL_ROUTE_KEYWORDS):
            return "CREDIT_PROPOSAL_AGENT"
        if self._contains_any(normalized, RISK_ASSESSMENT_ROUTE_KEYWORDS):
            return "RISK_ASSESSMENT_AGENT"
        if self._contains_any(normalized, CREDIT_RELATIONSHIP_ROUTE_KEYWORDS):
            return "CREDIT_RELATIONSHIP_AGENT"
        if self._contains_any(normalized, INCOME_ROUTE_KEYWORDS):
            return "INCOME_ANALYSIS_AGENT"
        if self._contains_any(normalized, CUSTOMER_PROFILE_ROUTE_KEYWORDS):
            return "CUSTOMER_PROFILE_AGENT"
        valid_routes = {
            "INCOME_ANALYSIS_AGENT",
            "CUSTOMER_PROFILE_AGENT",
            "CREDIT_RELATIONSHIP_AGENT",
            "RISK_ASSESSMENT_AGENT",
            "CREDIT_PROPOSAL_AGENT",
            "CREDIT_MEMO",
        }
        # The request named no task, so let the evidence decide. Previously
        # reached via route == CONVERSATION_AGENT, which was how the decision
        # step said "this is not analysis"; with that route gone, the same
        # situation shows up as a route it does not recognise.
        if route not in valid_routes and has_file:
            both_analysis_docs = {
                "INCOME_ANALYSIS_AGENT",
                "CUSTOMER_PROFILE_AGENT",
            }.issubset(document_routes)
            if both_analysis_docs:
                return "RISK_ASSESSMENT_AGENT"
            for candidate in [
                "INCOME_ANALYSIS_AGENT",
                "CUSTOMER_PROFILE_AGENT",
                "CREDIT_RELATIONSHIP_AGENT",
                "CREDIT_PROPOSAL_AGENT",
                "RISK_ASSESSMENT_AGENT",
            ]:
                if candidate in document_routes:
                    return candidate
        if route in valid_routes:
            return route
        return DEFAULT_ROUTE

    def _fallback_route(self, input_text: str, has_file: bool) -> AgentName:
        normalized = input_text.lower()
        if self._contains_any(normalized, CREDIT_MEMO_KEYWORDS):
            return "CREDIT_MEMO"
        if self._contains_any(normalized, CREDIT_PROPOSAL_ROUTE_KEYWORDS):
            return "CREDIT_PROPOSAL_AGENT"
        if self._contains_any(
            normalized,
            RISK_ASSESSMENT_ROUTE_KEYWORDS
            + ["underwriting", "approve", "approval"],
        ):
            return "RISK_ASSESSMENT_AGENT"
        if self._contains_any(normalized, CREDIT_RELATIONSHIP_ROUTE_KEYWORDS):
            return "CREDIT_RELATIONSHIP_AGENT"
        if self._contains_any(
            normalized,
            CUSTOMER_PROFILE_ROUTE_KEYWORDS + ["identity", "residence", "employment"],
        ):
            return "CUSTOMER_PROFILE_AGENT"
        if has_file or self._contains_any(
            normalized,
            INCOME_ROUTE_KEYWORDS
            + ["salary", "payslip", "bank statement", "repayment", "expenses"],
        ):
            return "INCOME_ANALYSIS_AGENT"
        # No keyword matched and no file. Requests like this are turned away
        # after discovery, so reaching here means the caller drove the graph
        # some other way — answer with the default rather than inventing a
        # route the workflow has no branch for.
        return DEFAULT_ROUTE

    def _analyze_evidence_gaps(
        self,
        documents: list[ClassifiedDocument],
        route: str,
    ) -> dict[str, Any]:
        inventory = {
            "income_documents": [],
            "customer_profile_documents": [],
            "credit_relationship_documents": [],
            "credit_proposal_documents": [],
            "risk_assessment_documents": [],
            "general_context": [],
        }
        agent_to_bucket = {
            "INCOME_ANALYSIS_AGENT": "income_documents",
            "CUSTOMER_PROFILE_AGENT": "customer_profile_documents",
            "CREDIT_RELATIONSHIP_AGENT": "credit_relationship_documents",
            "CREDIT_PROPOSAL_AGENT": "credit_proposal_documents",
            "RISK_ASSESSMENT_AGENT": "risk_assessment_documents",
            "GENERAL_CONTEXT": "general_context",
        }
        for doc in documents:
            if doc.agent == "GENERAL_CONTEXT":
                inventory["general_context"].append(doc.filename)
                continue
            # Count the document toward every agent it's real evidence for, not
            # just its primary label, so "missing evidence" doesn't fire for
            # coverage that's actually present in a combined document.
            for agent in doc.relevant_agents:
                bucket = agent_to_bucket.get(agent, "general_context")
                inventory[bucket].append(doc.filename)

        # Income evidence is the one gap that blocks: without any proof of
        # income there is no repayment capacity to assess, and every agent that
        # ran would be writing about a DSR nobody can compute.
        required = {
            "INCOME_ANALYSIS_AGENT": ["income_documents"],
            "CUSTOMER_PROFILE_AGENT": ["customer_profile_documents"],
            "CREDIT_PROPOSAL_AGENT": ["income_documents"],
            "RISK_ASSESSMENT_AGENT": [
                "income_documents",
                "customer_profile_documents",
            ],
            "CREDIT_MEMO": [
                "income_documents",
                "customer_profile_documents",
            ],
        }.get(route, [])

        missing = []
        for evidence_type in required:
            if inventory.get(evidence_type):
                continue
            missing.append(
                {
                    "type": evidence_type,
                    "severity": (
                        "high"
                        if evidence_type == "income_documents"
                        else "medium"
                    ),
                    "can_continue_without_it": (
                        evidence_type != "income_documents"
                    ),
                    "reason": self._gap_reason(evidence_type),
                }
            )

        can_proceed = not any(
            item["severity"] == "high"
            and not item["can_continue_without_it"]
            for item in missing
        )
        available = (
            ", ".join(
                f"{bucket}: {len(files)}"
                for bucket, files in inventory.items()
                if files
            )
            or "none"
        )
        missing_summary = (
            ", ".join(item["type"] for item in missing)
            if missing
            else "none"
        )
        return {
            "evidence_inventory": inventory,
            "missing_evidence": missing,
            "recommended_actions": ["run_available_analysis_agents"],
            "can_proceed": can_proceed,
            "summary": (
                f"Route {route}. Available evidence: {available}. "
                f"Missing evidence: {missing_summary}."
            ),
        }

    def _build_execution_plan(
        self,
        decision: dict[str, Any],
        gap_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        route = decision["route"]
        workflow_mode = decision.get(
            "workflow_mode",
            self._workflow_mode_for_route(route),
        )
        requires_credit_memo = route == "CREDIT_MEMO"
        if route == "CREDIT_MEMO":
            agents = [
                "CUSTOMER_PROFILE_AGENT",
                "CREDIT_RELATIONSHIP_AGENT",
                "INCOME_ANALYSIS_AGENT",
                "CREDIT_PROPOSAL_AGENT",
                "RISK_ASSESSMENT_AGENT",
                "CREDIT_MEMO",
            ]
            order = [
                "customer_profile_analysis",
                "credit_relationship_analysis",
                "income_analysis",
                "credit_proposal_calculation",
                "risk_assessment",
                "credit_memo_composition",
                "reflection",
            ]
        elif route == "RISK_ASSESSMENT_AGENT":
            agents = ["RISK_ASSESSMENT_AGENT"]
            order = ["risk_assessment", "reflection"]
        elif route == "CREDIT_PROPOSAL_AGENT":
            agents = ["CREDIT_PROPOSAL_AGENT"]
            order = ["credit_proposal_calculation", "reflection"]
        else:
            # Every remaining route is a single specialist. Unknown ids land
            # here too and are answered with the default rather than a plan
            # naming an agent that no longer exists.
            agents = [route if route in self.ROUTE_AGENTS else DEFAULT_ROUTE]
            order = [
                (
                    "income_analysis"
                    if agents[0] == "INCOME_ANALYSIS_AGENT"
                    else "credit_relationship_analysis"
                    if agents[0] == "CREDIT_RELATIONSHIP_AGENT"
                    else "customer_profile_analysis"
                ),
                "reflection",
            ]

        return {
            "plan_version": "notebook-local-v1",
            "intent": workflow_mode,
            "route": route,
            "workflow_mode": workflow_mode,
            "required_agents": agents,
            "execution_order": order,
            "requires_credit_memo": requires_credit_memo,
            "can_answer_now": bool(gap_analysis.get("can_proceed", True)),
            "missing_information": gap_analysis.get("missing_evidence", []),
            "reasoning": decision.get("reasoning", ""),
        }

    def _maybe_run_web_search(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
        decision: dict[str, Any],
        document_summary: str,
        steps: list[str],
    ) -> str:
        if not self.web_search_agent:
            return ""
        steps.append("Running WEB_SEARCH_AGENT")
        web_query = f"{query}\n\n{document_summary}"
        return self.web_search_agent.process_web_search_results(
            web_query,
            conversation_history,
        )

    def _run_single_agent(
        self,
        agent_name: str,
        input_text: str,
        history_context: str,
        decision: dict[str, Any],
        documents: list[ClassifiedDocument],
        web_context: str,
        execution_plan: dict[str, Any],
        gap_analysis: dict[str, Any],
        steps: list[str],
    ) -> dict[str, Any]:
        steps.append(f"Running {agent_name}")
        user_input = self._build_user_input(
            input_text,
            documents,
            history_context,
            agent_name,
            web_context,
            gap_analysis,
        )
        if agent_name == "INCOME_ANALYSIS_AGENT":
            agent = IncomeAnalysis(self.config.analysis_llm)
        elif agent_name == "CREDIT_RELATIONSHIP_AGENT":
            agent = CreditRelationshipAnalysis(self.config.analysis_llm)
        elif agent_name == "RISK_ASSESSMENT_AGENT":
            agent = RiskAssessment(self.config.analysis_llm)
        elif agent_name == "CREDIT_PROPOSAL_AGENT":
            agent = CreditProposalAnalysis(self.config.analysis_llm)
        else:
            agent = CustomerProfileAnalysis(self.config.analysis_llm)
        # Namespaced even though a single-agent run has nobody to collide with:
        # the labels a reviewer reads in the .md should mean the same thing here
        # as in a full memo, and this is the one path all five modes share.
        response = namespace_footnotes(
            extract_text_from_agent_output(agent.analyze(user_input)),
            AGENT_LABEL_PREFIXES.get(agent_name, ""),
        )

        return self._finalize(
            input_text,
            response,
            agent_name,
            decision,
            documents,
            {agent_name: response},
            web_context,
            history_context,
            execution_plan,
            gap_analysis,
            steps,
        )

    def _run_credit_memo_workflow(
        self,
        input_text: str,
        history_context: str,
        decision: dict[str, Any],
        documents: list[ClassifiedDocument],
        web_context: str,
        execution_plan: dict[str, Any],
        gap_analysis: dict[str, Any],
        steps: list[str],
    ) -> dict[str, Any]:
        # Customer profile, credit relationship, income analysis and credit
        # proposal each read their own documents and are independent, so run them
        # concurrently (bounded by the provider rate limit via
        # LLM_MAX_CONCURRENCY). RISK + the memo composer below still receive all
        # four outputs, so report quality is unchanged.
        steps.append(
            "Running CUSTOMER_PROFILE / CREDIT_RELATIONSHIP / INCOME_ANALYSIS / "
            "CREDIT_PROPOSAL_AGENT agents in parallel "
            f"(max_concurrency={self.config.max_concurrency})"
        )

        # Each specialist numbers its own footnotes from 1, so their labels are
        # namespaced the moment the text exists — before the composer merges
        # them and a shared "[^1]" would silently resolve to one agent's source.
        def _run_profile() -> str:
            profile_input = self._build_user_input(
                input_text,
                documents,
                history_context,
                "CUSTOMER_PROFILE_AGENT",
                web_context,
                gap_analysis,
            )
            return namespace_footnotes(
                extract_text_from_agent_output(
                    CustomerProfileAnalysis(self.config.analysis_llm).analyze(
                        profile_input
                    )
                ),
                AGENT_LABEL_PREFIXES["CUSTOMER_PROFILE_AGENT"],
            )

        def _run_credit_relationship() -> str:
            credit_relationship_input = f"""
            {self._build_user_input(
                input_text,
                documents,
                history_context,
                "CREDIT_RELATIONSHIP_AGENT",
                web_context,
                gap_analysis,
            )}

            Instruction:
            - Use T24 and CIC/bureau database tools when customer identifiers are
              available.
            - If tool data is unavailable, clearly state the limitation.
            """
            return namespace_footnotes(
                extract_text_from_agent_output(
                    CreditRelationshipAnalysis(
                        self.config.analysis_llm
                    ).analyze(credit_relationship_input)
                ),
                AGENT_LABEL_PREFIXES["CREDIT_RELATIONSHIP_AGENT"],
            )

        def _run_income() -> str:
            income_input = self._build_user_input(
                input_text,
                documents,
                history_context,
                "INCOME_ANALYSIS_AGENT",
                web_context,
                gap_analysis,
            )
            return namespace_footnotes(
                extract_text_from_agent_output(
                    IncomeAnalysis(self.config.analysis_llm).analyze(
                        income_input
                    )
                ),
                AGENT_LABEL_PREFIXES["INCOME_ANALYSIS_AGENT"],
            )

        def _run_proposal() -> str:
            proposal_input = self._build_user_input(
                input_text,
                documents,
                history_context,
                "CREDIT_PROPOSAL_AGENT",
                web_context,
                gap_analysis,
            )
            return namespace_footnotes(
                extract_text_from_agent_output(
                    CreditProposalAnalysis(self.config.analysis_llm).analyze(
                        proposal_input
                    )
                ),
                AGENT_LABEL_PREFIXES["CREDIT_PROPOSAL_AGENT"],
            )

        max_workers = max(1, min(4, self.config.max_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            profile_future = executor.submit(_run_profile)
            credit_relationship_future = executor.submit(_run_credit_relationship)
            income_future = executor.submit(_run_income)
            proposal_future = executor.submit(_run_proposal)
            profile_text = profile_future.result()
            credit_relationship_text = credit_relationship_future.result()
            income_text = income_future.result()
            credit_proposal_text = proposal_future.result()

        steps.append("Running RISK_ASSESSMENT_AGENT")
        risk_context = self._build_user_input(
            input_text,
            documents,
            history_context,
            "RISK_ASSESSMENT_AGENT",
            web_context,
            gap_analysis,
        )
        risk_input = f"""
        {risk_context}

        Customer profile analysis from CUSTOMER_PROFILE_AGENT:
        {profile_text}

        Credit relationship analysis from CREDIT_RELATIONSHIP_AGENT:
        {credit_relationship_text}

        Income and repayment capacity analysis from INCOME_ANALYSIS_AGENT:
        {income_text}

        Credit proposal from CREDIT_PROPOSAL_AGENT:
        {credit_proposal_text}
        """
        risk_text = namespace_footnotes(
            extract_text_from_agent_output(
                RiskAssessment(self.config.analysis_llm).analyze(risk_input)
            ),
            AGENT_LABEL_PREFIXES["RISK_ASSESSMENT_AGENT"],
        )

        steps.append("Running CREDIT_MEMO_COMPOSER_AGENT")
        response = CreditMemoComposerAgent(
            self.config.credit_memo_llm,
            max_input_chars=self.config.agent_input_char_budgets.get(
                "CREDIT_MEMO",
                80_000,
            ),
        ).compose(
            input_text=input_text,
            customer_profile=profile_text,
            credit_relationship_analysis=credit_relationship_text,
            income_analysis=income_text,
            credit_proposal=credit_proposal_text,
            risk_assessment=risk_text,
        )
        sub_outputs = {
            "CUSTOMER_PROFILE_AGENT": profile_text,
            "CREDIT_RELATIONSHIP_AGENT": credit_relationship_text,
            "INCOME_ANALYSIS_AGENT": income_text,
            "CREDIT_PROPOSAL_AGENT": credit_proposal_text,
            "RISK_ASSESSMENT_AGENT": risk_text,
            "CREDIT_MEMO": response,
        }
        return self._finalize(
            input_text,
            response,
            (
                "CUSTOMER_PROFILE_AGENT, CREDIT_RELATIONSHIP_AGENT, "
                "INCOME_ANALYSIS_AGENT, CREDIT_PROPOSAL_AGENT, "
                "RISK_ASSESSMENT_AGENT, CREDIT_MEMO_COMPOSER_AGENT"
            ),
            decision,
            documents,
            sub_outputs,
            web_context,
            history_context,
            execution_plan,
            gap_analysis,
            steps,
        )

    def _finalize(
        self,
        input_text: str,
        response: str,
        agent_name: str,
        decision: dict[str, Any],
        documents: list[ClassifiedDocument],
        sub_agent_outputs: dict[str, str],
        web_context: str,
        history_context: str,
        execution_plan: dict[str, Any],
        gap_analysis: dict[str, Any],
        steps: list[str],
    ) -> dict[str, Any]:
        # First, so everything below (chart insertion, footnote/template-leak
        # findings) reads a report whose citations already sit in one list at
        # the end, not scattered mid-section. Returns the audit too — gathering
        # the definitions collapses repeated labels, so a checker running
        # afterwards could no longer see that two agents had claimed the same
        # one.
        response, footnote_audit = consolidate_footnotes(response)
        # Same reasoning as above, for a different renderer quirk: a bullet
        # list glued to the line above it with no blank line renders as a
        # bare "-" instead of a <ul>. Fixing it here means every "Nhận định"
        # bullet is correct on the page even if a specialist forgot the blank
        # line the prompt asks for.
        response = ensure_blank_line_before_lists(response)
        # "8,00%" -> "8%", and a table cell holding only a zero -> "-". Asked of
        # every agent in NUMBER FORMAT RULE and applied here too, because a rule
        # in a prompt is a request.
        response = tidy_numbers(response)

        if self.guardrails:
            allowed, checked_response = self.guardrails.check_output(
                response,
                input_text,
            )
            if not allowed:
                agent_name = "OUTPUT_GUARDRAILS"
            response = checked_response

        # Convert all raw đồng amounts to triệu VNĐ for display, after the output
        # safety check above has already seen the raw-number response.
        response = convert_amounts_in_text(response)
        # Prompt rules are not enforcement: report the citations that did not
        # resolve, rather than quietly dropping or inventing them.
        response += self._format_footnote_findings(footnote_audit)
        response += self._format_template_findings(response)
        # Last, and deliberately so: everything above this line reads or edits
        # what the *model* wrote, and this block is written by the pipeline from
        # extracted JSON. It is not the checkers' business, and the text rewriter
        # has no business in it either.
        #
        # Measured rather than assumed — with today's formatting the block
        # survives all three untouched: convert_amounts_in_text only matches
        # thousands-grouped integers and every value here carries two decimals,
        # and both assertion checks stay silent on it. So this ordering is a
        # guard against a future change to either side, not a fix for a live bug.
        chart_block, chart_title = self._build_debt_chart_block(
            documents,
            sub_agent_outputs,
        )
        if chart_block:
            response, anchor = self._insert_debt_chart(
                response,
                chart_block,
                chart_title,
            )
            steps.append(
                "Inserted debt/income chart "
                + (
                    "at the end (no credit-relationship heading found)"
                    if anchor == "appended"
                    else f'under the heading matching "{anchor}"'
                )
            )
        # detect_loan_program is a pure string match over the same two strings
        # the classify node used (input guardrails pass `query` through
        # untouched), so recomputing here is exact and avoids threading the
        # detection dict through five graph nodes and three runner signatures.
        # The documents carry the program that was actually applied, so cross-
        # check rather than trust the recomputation blindly.
        detection = detect_loan_program(input_text, history_context)
        applied = next((doc.loan_program for doc in documents), "")
        if documents and (detection["program"] or "") != applied:
            detection = {
                "program": applied or None,
                "matched_alias": "",
                "candidates": [],
                "source": "applied-at-classification",
            }
        # Throttling is otherwise indistinguishable from a hang: report how many
        # LLM calls the run made and how long the rate limiter held them back.
        rate_limit = shared_rate_limiter().snapshot()
        return self._build_state(
            input_text,
            response,
            agent_name,
            decision,
            documents,
            sub_agent_outputs,
            execution_plan,
            gap_analysis,
            web_context,
            steps
            + [
                f"LLM: {rate_limit['llm_calls']} calls at a "
                f"{rate_limit['requests_per_minute']}/min limit — this run "
                f"could not finish faster than "
                f"{rate_limit['minimum_seconds_for_these_calls']}s",
                "Built final response",
            ],
            self._build_document_selections(documents),
            prompt_blocks.affordability_data(documents),
            loan_program=applied,
            loan_program_detection=detection,
            rate_limit=rate_limit,
            loan_sizing=prompt_blocks.loan_sizing_data(documents),
        )

    @staticmethod
    def _format_template_findings(response: str) -> str:
        """Flag layout scaffolding or placeholders copied into the report."""

        findings = check_template_leakage(response)
        if not findings:
            return ""
        lines = ["", "**Lỗi bám mẫu báo cáo:**", ""]
        lines += [f"- {finding}" for finding in findings]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_footnote_findings(audit: FootnoteAudit) -> str:
        """Append the citations that did not resolve against their definitions.

        Takes the audit rather than the text: it was produced by the same pass
        that gathered the definitions, which is the only point where a label
        claimed by two agents is still visible.
        """

        findings = format_footnote_findings(audit)
        if not findings:
            return ""
        lines = [
            "",
            "**Chú thích nguồn:**",
            "",
        ]
        lines += [f"- {finding}" for finding in findings]
        return "\n".join(lines) + "\n"

    def _build_user_input(
        self,
        input_text: str,
        documents: list[ClassifiedDocument],
        history_context: str,
        target_agent: str,
        web_context: str,
        gap_analysis: dict[str, Any],
    ) -> str:
        budget = self.config.agent_input_char_budgets.get(target_agent, 12_000)
        language_instruction = (
            "Response language instruction:\n"
            "- Answer in the same language as the current user request.\n"
            "- If the current user request is Vietnamese, answer fully in "
            "Vietnamese.\n"
            "- If the user request has no clear language, answer in Vietnamese "
            "by default.\n\n"
        )
        base = (
            "Recent conversation history:\n"
            f"{history_context or 'No previous conversation.'}\n\n"
            f"{language_instruction}"
            "Evidence sufficiency summary:\n"
            f"{gap_analysis.get('summary', '')}\n\n"
            "Current user request:\n"
            f"{input_text}"
        )
        if web_context:
            base += (
                "\n\nExternal context from web search (employer, property "
                f"market, regulation):\n{web_context}"
            )

        selected = self._select_documents_for_agent(documents, target_agent)
        # Only successfully-extracted documents become evidence; failed ones are
        # still listed in the document summary but must not pollute agent input.
        usable = [doc for doc in selected if doc.extraction_status == "success"]

        def _is_primary(doc: ClassifiedDocument) -> bool:
            """Required evidence for this agent, per the matrix."""

            return doc.agent_relevance.get(target_agent) == "R"

        source_list_block = self._build_source_list_block(usable)
        # The computed tables read every extracted document, not this agent's
        # routed subset: income, obligations and the maximum loan are facts about
        # the borrower, built from documents some of these agents never see.
        affordability_block = self._build_affordability_block(documents, target_agent)
        loan_sizing_block = self._build_loan_sizing_block(documents, target_agent)
        reads_income_json = target_agent in self.INCOME_JSON_AGENTS
        income_block = (
            self._build_income_structured_block(usable)
            if reads_income_json
            else ""
        )
        reads_proposal_json = target_agent in self.PROPOSAL_JSON_AGENTS
        proposal_block = (
            self._build_proposal_structured_block(usable)
            if reads_proposal_json
            else ""
        )
        reads_cic_s10a_json = target_agent in self.CIC_S10A_JSON_AGENTS
        cic_s10a_block = (
            self._build_cic_s10a_structured_block(usable)
            if reads_cic_s10a_json
            else ""
        )
        reads_cic_r21_json = target_agent in self.CIC_R21_JSON_AGENTS
        cic_r21_block = (
            self._build_cic_r21_structured_block(usable)
            if reads_cic_r21_json
            else ""
        )
        reads_sitevisit_json = target_agent in self.SITEVISIT_JSON_AGENTS
        sitevisit_block = (
            self._build_sitevisit_structured_block(usable)
            if reads_sitevisit_json
            else ""
        )
        structured_blocks = (
            affordability_block,
            loan_sizing_block,
            income_block,
            proposal_block,
            cic_s10a_block,
            cic_r21_block,
            sitevisit_block,
        )
        # The units only clash once two of these blocks are present, so the
        # warning appears exactly then.
        money_blocks = sum(1 for block in structured_blocks if block)
        unit_warning = (
            f"{self.MIXED_UNIT_WARNING}\n\n" if money_blocks > 1 else ""
        )
        # Every block spends characters on its own header plus a separator,
        # before any content. That overhead has to come out of the budget up
        # front: without it the assembled prompt overshoots and the final
        # truncate_text lops whole documents off the end.
        block_overhead = sum(
            len(self._document_block_header(doc, target_agent)) for doc in usable
        ) + len(self.DOC_BLOCK_SEPARATOR) * max(0, len(usable) - 1)
        remaining = max(
            1_000,
            budget
            - len(base)
            - len(source_list_block)
            - sum(len(block) for block in structured_blocks)
            - len(self.DOC_SECTION_HEADER)
            - len(unit_warning)
            - block_overhead,
        )
        # Required evidence gets more budget than optional evidence, so a
        # document the matrix marks optional for this agent can't crowd out the
        # documents it actually needs.
        weight_sum = sum(
            self.PRIMARY_DOC_BUDGET_WEIGHT
            if _is_primary(doc)
            else self.SECONDARY_DOC_BUDGET_WEIGHT
            for doc in usable
        ) or 1
        unit_budget = remaining / weight_sum

        def _doc_budget(doc: ClassifiedDocument) -> int:
            weight = (
                self.PRIMARY_DOC_BUDGET_WEIGHT
                if _is_primary(doc)
                else self.SECONDARY_DOC_BUDGET_WEIGHT
            )
            return max(1_000, int(unit_budget * weight))

        def _pointer(heading: str) -> str:
            return (
                "Extracted document content: đã trích xuất có cấu trúc "
                f"— xem {heading} bên dưới."
            )

        blocks = []
        for doc in usable:
            # A successfully-extracted document is represented by its structured
            # JSON, not its raw OCR dump — that's the whole point of the pass.
            # Gated on the same table as the block itself, because sending one
            # without the other either bills the content twice or loses it.
            # Anything else (no pass, or extraction failed) keeps raw content so
            # no evidence is ever silently dropped.
            if reads_income_json and doc.is_income and doc.income_extraction:
                content_section = _pointer(self.INCOME_BLOCK_HEADING)
            elif reads_proposal_json and doc.is_proposal and doc.proposal_extraction:
                content_section = _pointer(self.PROPOSAL_BLOCK_HEADING)
            elif reads_cic_s10a_json and doc.is_cic_s10a and doc.cic_s10a_extraction:
                content_section = _pointer(self.CIC_S10A_BLOCK_HEADING)
            elif reads_cic_r21_json and doc.is_cic_r21 and doc.cic_r21_extraction:
                content_section = _pointer(self.CIC_R21_BLOCK_HEADING)
            elif (
                reads_sitevisit_json
                and doc.is_sitevisit
                and doc.sitevisit_extraction
            ):
                content_section = _pointer(self.SITEVISIT_BLOCK_HEADING)
            else:
                content_section = "\n".join(
                    [
                        "Extracted document content:",
                        truncate_text(doc.content, _doc_budget(doc)),
                    ]
                )
            blocks.append(
                self._document_block_header(doc, target_agent) + content_section
            )
        docs_text = self.DOC_BLOCK_SEPARATOR.join(blocks)
        # Emitted only when there is something to say: an empty slot would add
        # blank lines to every prompt that lacks that document.
        sections = "".join(f"{block}\n\n" for block in structured_blocks if block)
        return truncate_text(
            (
                f"{base}\n\n"
                # Near the top on purpose: the final truncate trims the tail, and
                # a source list that got cut is the exact failure this replaced.
                f"{source_list_block}\n\n"
                f"{unit_warning}"
                f"{sections}"
                f"{self.DOC_SECTION_HEADER}"
                f"{docs_text}"
            ),
            budget,
        )

    @classmethod
    def _insert_debt_chart(
        cls,
        response: str,
        block: str,
        title: str,
    ) -> tuple[str, str]:
        """Place the chart under the credit-relationship heading.

        Returns ``(text, where)``; ``where`` names the anchor that matched so the
        step log can say whether the chart landed in its section or was appended
        as a fallback.

        ``title`` is only used by that fallback, and is taken from the caller
        rather than the class constant so the heading cannot promise an income
        series the block itself leaves out.

        Appending is the last resort rather than the failure case: a chart at the
        end of the memo is worse than one in its section, but far better than a
        chart that silently disappears because the composer reworded a heading.
        """

        if not block:
            return response, ""
        lines = (response or "").splitlines()
        for anchor in cls.DEBT_CHART_ANCHORS:
            for index, line in enumerate(lines):
                if not line.lstrip().startswith("#"):
                    continue
                if anchor in normalize_text(line):
                    return (
                        "\n".join(
                            lines[: index + 1] + ["", block] + lines[index + 1:]
                        ),
                        anchor,
                    )
        return (
            f"{response}\n\n## {title}\n\n{block}\n",
            "appended",
        )





    @staticmethod
    def _select_documents_for_agent(
        documents: list[ClassifiedDocument],
        target_agent: str,
    ) -> list[ClassifiedDocument]:
        # Membership comes from the matrix, not from which agent "owns" the
        # document: one document legitimately feeds several agents (a bank
        # statement is evidence for both INCOME and RISK). Required evidence goes
        # first so it wins the char budget when many files are uploaded.
        required: list[ClassifiedDocument] = []
        optional: list[ClassifiedDocument] = []
        # Documents that matched no matrix type are shared with every agent, so
        # a classification miss degrades to "everyone sees it" instead of
        # silently hiding evidence from the agent that needed it.
        unmatched: list[ClassifiedDocument] = []
        for doc in documents:
            level = doc.agent_relevance.get(target_agent)
            if level == "R":
                required.append(doc)
            elif level:
                optional.append(doc)
            elif doc.agent == "GENERAL_CONTEXT":
                unmatched.append(doc)
        return required + optional + unmatched

    @classmethod
    def _build_document_selections(
        cls,
        documents: list[ClassifiedDocument],
    ) -> dict[str, dict[str, list[str]]]:
        """Snapshot, per specialist agent, exactly what _build_user_input would
        feed it — for monitoring/testing, not for prompt construction itself.
        Recomputed via the same _select_documents_for_agent used at call time,
        so it can never drift from the real selection."""

        selections: dict[str, dict[str, list[str]]] = {}
        for target_agent in sorted(VALID_DOCUMENT_AGENTS - {"GENERAL_CONTEXT"}):
            selected = cls._select_documents_for_agent(documents, target_agent)
            selections[target_agent] = {
                "required": [
                    f"{doc.filename} [{doc.document_type}]"
                    for doc in selected
                    if doc.agent_relevance.get(target_agent) == "R"
                ],
                "optional": [
                    f"{doc.filename} [{doc.document_type}]"
                    for doc in selected
                    if doc.agent_relevance.get(target_agent) == "O"
                ],
                "unmatched_shared": [
                    doc.filename
                    for doc in selected
                    if target_agent not in doc.agent_relevance
                ],
            }
        return selections

    @staticmethod
    def _format_history(
        history: list[dict[str, str]],
        limit: int = 20,
    ) -> str:
        if not history:
            return ""
        lines = []
        for item in history[-limit:]:
            content = truncate_text(item.get("content", ""), 1_000)
            lines.append(f"{item.get('role', 'user')}: {content}")
        return "\n".join(lines)

    # (flag attribute, result attribute, label) for each structured-extraction
    # pass, so the summary tag is written once instead of once per pass. A third
    # pass added later shows up here and nowhere else.
    EXTRACTION_PASSES = (
        ("is_income", "income_extraction", "income_extraction_error", "THU NHẬP"),
        (
            "is_proposal",
            "proposal_extraction",
            "proposal_extraction_error",
            "ĐỀ NGHỊ",
        ),
        (
            "is_cic_s10a",
            "cic_s10a_extraction",
            "cic_s10a_extraction_error",
            "CIC S10A",
        ),
        (
            "is_cic_r21",
            "cic_r21_extraction",
            "cic_r21_extraction_error",
            "CIC R21",
        ),
        (
            "is_sitevisit",
            "sitevisit_extraction",
            "sitevisit_extraction_error",
            "KHẢO SÁT",
        ),
    )

    @classmethod
    def _extraction_tags(cls, doc: ClassifiedDocument) -> str:
        """Mark which structured extractions ran on a document, and which failed.

        A failed extraction silently drops the agent back to raw OCR — the report
        still comes out, just without the structured figures — so the failure has
        to be visible in the summary the decision LLM reads.
        """

        tags = []
        for flag_attr, _, error_attr, label in cls.EXTRACTION_PASSES:
            if not getattr(doc, flag_attr):
                continue
            # Keyed on the error, not on a missing result: this summary is built
            # twice, and the first time — for the routing node — no pass has run
            # yet. Testing the result alone would report every document as a
            # failed extraction at that point, which is the opposite of true.
            if getattr(doc, error_attr):
                tags.append(f" [{label}, trích xuất lỗi]")
            else:
                tags.append(f" [{label}]")
        return "".join(tags)

    @classmethod
    def _format_document_summary(cls, documents: list[ClassifiedDocument]) -> str:
        lines = []
        for doc in documents:
            routing = (
                ", ".join(
                    f"{agent}:{level}"
                    for agent, level in sorted(doc.agent_relevance.items())
                )
                or "GENERAL_CONTEXT (không khớp loại nào)"
            )
            lines.append(
                f"- {doc.filename}: {doc.document_type or '-'} -> {routing}"
                f"{cls._extraction_tags(doc)} (confidence={doc.confidence:.2f}, "
                f"extraction={doc.extraction_status})"
            )
        return "\n".join(lines)

    @staticmethod
    def _gap_reason(evidence_type: str) -> str:
        reasons = {
            "income_documents": (
                "Cần chứng từ chứng minh thu nhập: hợp đồng lao động, xác nhận "
                "lương/bảng lương, sao kê tài khoản 3-6 tháng gần nhất, thuế "
                "TNCN, hợp đồng cho thuê hoặc sổ sách hộ kinh doanh."
            ),
            "customer_profile_documents": (
                "Thông tin khách hàng đầy đủ hơn khi có CCCD, xác nhận cư trú, "
                "giấy tờ tình trạng hôn nhân và giấy đề nghị vay vốn."
            ),
            "credit_relationship_documents": (
                "Phân tích quan hệ tín dụng dùng báo cáo CIC của khách hàng "
                "(và người đồng vay), hoặc truy vấn T24/CIC qua công cụ dữ liệu."
            ),
            "credit_proposal_documents": (
                "Đề xuất khoản vay cần giấy đề nghị vay vốn (số tiền, thời hạn, "
                "mục đích), hợp đồng mua bán và giấy tờ, định giá tài sản bảo đảm."
            ),
            "risk_assessment_documents": (
                "Đánh giá rủi ro đầy đủ hơn khi có giấy đề nghị vay vốn, báo cáo "
                "CIC, chứng từ thu nhập và hồ sơ tài sản bảo đảm."
            ),
        }
        return reasons.get(evidence_type, "Relevant evidence was not found.")

    @staticmethod
    def _missing_evidence_response(
        input_text: str,
        gap_analysis: dict[str, Any],
    ) -> str:
        missing = gap_analysis.get("missing_evidence", [])
        lines = [
            f"- **{item.get('type', 'evidence')}**: "
            f"{item.get('reason', 'Missing evidence.')}"
            for item in missing
        ]
        if Supervisor._prefer_vietnamese_response(input_text):
            return (
                "## Cần bổ sung thêm tài liệu\n\n"
                "Mình chưa thể tiếp tục phân tích vì đang thiếu thông tin "
                "bắt buộc.\n\n"
                "### Tài liệu cần bổ sung\n"
                + ("\n".join(lines) if lines else "- Chưa xác định.")
            )
        return (
            "## Additional Documents Required\n\n"
            "Required evidence is missing.\n\n"
            + ("\n".join(lines) if lines else "- Not specified.")
        )

    @staticmethod
    def _prefer_vietnamese_response(text: str) -> bool:
        normalized = (text or "").lower()
        return "hãy" in normalized or any(
            char in VIETNAMESE_CHARS for char in normalized
        )

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _build_state(
        input_text: str,
        response: str,
        agent_name: str,
        decision: dict[str, Any] | None = None,
        documents: list[ClassifiedDocument] | None = None,
        sub_agent_outputs: dict[str, str] | None = None,
        execution_plan: dict[str, Any] | None = None,
        gap_analysis: dict[str, Any] | None = None,
        web_context: str = "",
        steps: list[str] | None = None,
        document_selections: dict[str, dict[str, list[str]]] | None = None,
        affordability: dict[str, Any] | None = None,
        # Appended last, and always passed by keyword: two call sites supply the
        # earlier parameters positionally, so inserting anywhere above would
        # silently shift their arguments.
        loan_program: str = "",
        loan_program_detection: dict[str, Any] | None = None,
        rate_limit: dict[str, Any] | None = None,
        loan_sizing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "response": response,
            "agent_name": agent_name,
            "decision": decision or {},
            # Which loan program column of the document matrix was used, and how
            # it was determined — the R/O levels in document_classifications are
            # only interpretable against it.
            "loan_program": loan_program,
            "loan_program_detection": loan_program_detection or {},
            # LLM call count and seconds spent waiting on the rate limiter, so a
            # slow run can be told apart from a stuck one.
            "rate_limit": rate_limit or {},
            "document_classifications": to_dict_list(documents or []),
            # Per-agent snapshot of _select_documents_for_agent's output
            # (primary vs secondary/shared filenames) — for monitoring/testing
            # which documents actually feed each specialist's LLM input.
            "document_selections": document_selections or {},
            # Deterministic repayment-capacity table as data (not just the
            # markdown block) so a finished run can be audited afterwards.
            # Money stays in đồng here even though the prompt shows triệu VNĐ
            # — a monitoring dump should carry the figure, not its presentation.
            "affordability": affordability or {},
            # The maximum-loan table as data, same convention.
            "loan_sizing": loan_sizing or {},
            "sub_agent_outputs": sub_agent_outputs or {},
            "execution_plan": execution_plan or {},
            "gap_analysis": gap_analysis or {},
            "web_context": web_context,
            "steps": steps or [],
        }
