"""Loader for the document routing matrix (src/matrix/document_matrix.yaml).

The matrix is the single source of truth for "which agents consume which kind of
document". Classification only has to answer the *objective* question — what
kind of document is this — and the matrix turns that answer into the agent
fan-out. Before this, keyword scores were aimed straight at agents, which made
the fan-out an emergent property of two threshold constants instead of a
business decision.

Everything is validated at load time and raises DocumentMatrixError on the first
problem: a silently mis-parsed matrix would drop documents from an agent's
evidence without any visible failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.types import SPECIALIST_DOCUMENT_AGENTS
from src.utils.common import normalize_text


MATRIX_PATH = Path(__file__).resolve().parent / "document_matrix.yaml"

RELEVANCE_LEVELS = ("R", "O")

# Used only to pick a display "primary" agent when a document type's own agent
# set does not contain its group's primary_agent (e.g. the car purchase
# contract routes to CREDIT_PROPOSAL and RISK only). Most specific consumer first.
PRIMARY_AGENT_PRECEDENCE = (
    "CREDIT_PROPOSAL_AGENT",
    "CREDIT_RELATIONSHIP_AGENT",
    "INCOME_ANALYSIS_AGENT",
    "CUSTOMER_PROFILE_AGENT",
    "RISK_ASSESSMENT_AGENT",
)

# Ranked so "strongest wins" comparisons stay explicit rather than relying on
# string ordering ("O" < "R" is true but accidental).
_LEVEL_RANK = {"O": 1, "R": 2}


class DocumentMatrixError(RuntimeError):
    """Raised when the matrix file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class DocumentType:
    """One row of the matrix: a kind of document and who consumes it."""

    id: str
    stt: str
    label: str
    # A caption for the reader of a finished report. `label` cannot serve: it is
    # the regulatory sentence from the checklist ("Sao kê tài khoản nhận lương/
    # tài khoản thanh toán ... 03-06 tháng gần nhất"), which is what the document must satisfy,
    # not what it is called. Falls back to `label` when the YAML omits it, so a
    # new type is never nameless — only verbose.
    short_label: str
    group_id: str
    group_stt: str
    group_label: str
    keywords: tuple[str, ...]
    # {agent: {loan_program: "R"|"O"}} — always fully expanded across programs,
    # so callers never have to know whether the YAML used the scalar shorthand.
    agents: dict[str, dict[str, str]]
    primary_agent: str
    requirement: dict[str, str] = field(default_factory=dict)
    # Proof of income (labour contract, salary confirmation, bank statement, PIT
    # return, rental contract, household-business books): turned into a
    # monthly income record the affordability calculator reads.
    income_extraction: bool = False
    proposal_extraction: bool = False
    # Named after the CIC form code, not "cic": CIC issues several unrelated
    # reports against one customer and they share only a letterhead, so the
    # collateral report gets its own flag rather than widening this one.
    cic_s10a_extraction: bool = False
    # The collateral report. Printed form code is actually "R20"; kept the
    # "r21" name the module was already registered under rather than rename
    # mid-project — the flag name is internal, only the matrix YAML key and the
    # Python module need to agree with each other.
    cic_r21_extraction: bool = False
    # The appraisal / field verification report. Unlike the forms above it is
    # written prose, so the pass pulls out what the officer verified (residence,
    # workplace, income, collateral) and keeps the officer's own verdict in a
    # block of its own, labelled as opinion.
    sitevisit_extraction: bool = False

    @property
    def routing_signature(self) -> frozenset[str]:
        """The set of agents this type feeds, ignoring R/O and loan program.

        Two types with the same signature are interchangeable as far as routing
        is concerned, so confusing one for the other costs nothing.
        """

        return frozenset(self.agents)


@dataclass(frozen=True)
class DocumentMatrix:
    version: int
    loan_programs: tuple[str, ...]
    loan_program_labels: dict[str, str]
    # {normalized alias: loan program id} — how a user might name each program
    # in a prompt. Normalized at load so detection never re-does the work.
    loan_program_aliases: dict[str, str]
    types: dict[str, DocumentType]

    def get(self, type_id: str) -> DocumentType | None:
        return self.types.get(type_id)


def _fail(message: str) -> None:
    raise DocumentMatrixError(f"{MATRIX_PATH.name}: {message}")


def _parse_agents(
    raw: Any,
    programs: tuple[str, ...],
    where: str,
) -> dict[str, dict[str, str]]:
    """Expand the `agents` block into {agent: {program: level}}.

    Accepts a scalar level (same across every loan program) or an explicit
    per-program map. A partial map is rejected rather than back-filled: the
    whole point of splitting the loan products is that a divergence must be
    stated, not guessed.
    """

    if not isinstance(raw, dict) or not raw:
        _fail(f"{where}: 'agents' must be a non-empty mapping")

    agents: dict[str, dict[str, str]] = {}
    for agent, value in raw.items():
        if agent not in SPECIALIST_DOCUMENT_AGENTS:
            _fail(
                f"{where}: unknown agent {agent!r}; expected one of "
                f"{sorted(SPECIALIST_DOCUMENT_AGENTS)}"
            )
        if isinstance(value, str):
            level = value.strip().upper()
            if level not in RELEVANCE_LEVELS:
                _fail(f"{where}/{agent}: level {value!r} must be R or O")
            agents[agent] = {program: level for program in programs}
            continue
        if isinstance(value, dict):
            missing = [p for p in programs if p not in value]
            extra = [p for p in value if p not in programs]
            if missing or extra:
                _fail(
                    f"{where}/{agent}: per-program map must list exactly "
                    f"{list(programs)}"
                    + (f"; missing {missing}" if missing else "")
                    + (f"; unknown {extra}" if extra else "")
                )
            levels = {}
            for program in programs:
                level = str(value[program]).strip().upper()
                if level not in RELEVANCE_LEVELS:
                    _fail(
                        f"{where}/{agent}/{program}: level "
                        f"{value[program]!r} must be R or O"
                    )
                levels[program] = level
            agents[agent] = levels
            continue
        _fail(f"{where}/{agent}: expected 'R'/'O' or a per-program mapping")
    return agents


def _resolve_primary_agent(
    declared: str | None,
    agents: dict[str, dict[str, str]],
    where: str,
) -> str:
    """Pick the agent this document type is displayed as belonging to.

    The group's primary_agent wins when it actually consumes the type; otherwise
    fall back to precedence order over the type's own agents, preferring an
    agent that requires the document over one that merely may use it.
    """

    if declared:
        if declared not in SPECIALIST_DOCUMENT_AGENTS:
            _fail(f"{where}: unknown primary_agent {declared!r}")
        if declared in agents:
            return declared

    def _pick(level: str) -> str | None:
        for agent in PRIMARY_AGENT_PRECEDENCE:
            levels = agents.get(agent)
            if levels and level in levels.values():
                return agent
        return None

    return _pick("R") or _pick("O") or sorted(agents)[0]


def _load(path: Path) -> DocumentMatrix:
    if not path.exists():
        _fail(f"matrix file not found at {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DocumentMatrixError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        _fail("top level must be a mapping")

    version = raw.get("version")
    if not isinstance(version, int):
        _fail("'version' must be an integer")

    raw_programs = raw.get("loan_programs")
    if not isinstance(raw_programs, list) or not raw_programs:
        _fail("'loan_programs' must be a non-empty list")
    programs: list[str] = []
    program_labels: dict[str, str] = {}
    program_aliases: dict[str, str] = {}
    for entry in raw_programs:
        if not isinstance(entry, dict) or not entry.get("id"):
            _fail("each loan_programs entry needs an 'id'")
        program_id = str(entry["id"])
        if program_id in program_labels:
            _fail(f"duplicate loan program id {program_id!r}")
        programs.append(program_id)
        label = str(entry.get("label") or program_id)
        program_labels[program_id] = label
        # The id and label are always recognised; `aliases` only adds to them.
        raw_aliases = entry.get("aliases") or []
        if not isinstance(raw_aliases, list):
            _fail(f"loan program {program_id!r}: 'aliases' must be a list")
        for alias in [program_id, label, *raw_aliases]:
            normalized = normalize_text(str(alias))
            if not normalized:
                _fail(
                    f"loan program {program_id!r}: alias {alias!r} normalizes to "
                    "nothing — it can never match"
                )
            owner = program_aliases.get(normalized)
            if owner and owner != program_id:
                # Two programs answering to the same words would make detection
                # a coin flip, so refuse to load rather than pick one.
                _fail(
                    f"alias {alias!r} (normalized {normalized!r}) is claimed by "
                    f"both {owner!r} and {program_id!r}"
                )
            program_aliases[normalized] = program_id

    raw_groups = raw.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        _fail("'groups' must be a non-empty list")

    types: dict[str, DocumentType] = {}
    group_ids: set[str] = set()
    for group in raw_groups:
        if not isinstance(group, dict):
            _fail("each group must be a mapping")
        group_id = str(group.get("id") or "")
        if not group_id:
            _fail("each group needs an 'id'")
        if group_id in group_ids:
            _fail(f"duplicate group id {group_id!r}")
        group_ids.add(group_id)

        raw_types = group.get("types")
        if not isinstance(raw_types, list) or not raw_types:
            _fail(f"group {group_id!r}: 'types' must be a non-empty list")

        for entry in raw_types:
            if not isinstance(entry, dict):
                _fail(f"group {group_id!r}: each type must be a mapping")
            type_id = str(entry.get("id") or "")
            if not type_id:
                _fail(f"group {group_id!r}: a type is missing 'id'")
            if type_id in types:
                _fail(f"duplicate document type id {type_id!r}")
            where = f"type {type_id!r}"

            label = str(entry.get("label") or "").strip()
            if not label:
                _fail(f"{where}: 'label' is required")
            short_label = str(entry.get("short_label") or "").strip() or label

            raw_keywords = entry.get("keywords")
            if not isinstance(raw_keywords, list) or not raw_keywords:
                _fail(f"{where}: 'keywords' must be a non-empty list")
            keywords: list[str] = []
            for keyword in raw_keywords:
                text = str(keyword).strip()
                if not text:
                    _fail(f"{where}: empty keyword")
                if text in keywords:
                    _fail(f"{where}: duplicate keyword {text!r}")
                keywords.append(text)

            agents = _parse_agents(entry.get("agents"), tuple(programs), where)
            requirement = entry.get("requirement") or {}
            if not isinstance(requirement, dict):
                _fail(f"{where}: 'requirement' must be a mapping")

            types[type_id] = DocumentType(
                id=type_id,
                stt=str(entry.get("stt") or ""),
                label=label,
                short_label=short_label,
                group_id=group_id,
                group_stt=str(group.get("stt") or ""),
                group_label=str(group.get("label") or group_id),
                keywords=tuple(keywords),
                agents=agents,
                primary_agent=_resolve_primary_agent(
                    entry.get("primary_agent") or group.get("primary_agent"),
                    agents,
                    where,
                ),
                requirement={
                    str(key): str(value or "")
                    for key, value in requirement.items()
                },
                income_extraction=bool(entry.get("income_extraction", False)),
                proposal_extraction=bool(
                    entry.get("proposal_extraction", False)
                ),
                cic_s10a_extraction=bool(
                    entry.get("cic_s10a_extraction", False)
                ),
                cic_r21_extraction=bool(
                    entry.get("cic_r21_extraction", False)
                ),
                sitevisit_extraction=bool(
                    entry.get("sitevisit_extraction", False)
                ),
            )

    if not types:
        _fail("no document types defined")
    return DocumentMatrix(
        version=version,
        loan_programs=tuple(programs),
        loan_program_labels=program_labels,
        loan_program_aliases=program_aliases,
        types=types,
    )


@lru_cache(maxsize=1)
def load_matrix() -> DocumentMatrix:
    """Load and validate the matrix once per process."""

    return _load(MATRIX_PATH)


def reload_matrix() -> DocumentMatrix:
    """Drop the cache and re-read the file (for tests / notebook iteration)."""

    load_matrix.cache_clear()
    return load_matrix()


def all_types() -> dict[str, DocumentType]:
    return load_matrix().types


def get_type(type_id: str) -> DocumentType | None:
    return load_matrix().get(type_id)


def document_type_keywords() -> dict[str, tuple[str, ...]]:
    """{document_type_id: keywords} — the input to rule-based classification."""

    return {type_id: doc.keywords for type_id, doc in all_types().items()}


def agent_relevance_for_type(
    type_id: str,
    loan_program: str | None = None,
) -> dict[str, str]:
    """Return {agent: "R"|"O"} for a document type.

    ``loan_program=None`` resolves to the strongest level across every program.
    Taking the strongest level means an unnamed product can only
    over-prioritise a document, never quietly demote real evidence.
    """

    doc = get_type(type_id)
    if doc is None:
        return {}
    if loan_program is not None:
        if loan_program not in load_matrix().loan_programs:
            raise DocumentMatrixError(
                f"unknown loan program {loan_program!r}; expected one of "
                f"{list(load_matrix().loan_programs)}"
            )
        return {
            agent: levels[loan_program] for agent, levels in doc.agents.items()
        }
    return {
        agent: max(levels.values(), key=lambda level: _LEVEL_RANK[level])
        for agent, levels in doc.agents.items()
    }


def primary_agent_for_type(type_id: str) -> str | None:
    doc = get_type(type_id)
    return doc.primary_agent if doc else None


def is_income_type(type_id: str) -> bool:
    """True when this document type is proof of income to be extracted."""

    doc = get_type(type_id)
    return bool(doc and doc.income_extraction)


def is_proposal_type(type_id: str) -> bool:
    """True when this document type is a loan application to be extracted."""

    doc = get_type(type_id)
    return bool(doc and doc.proposal_extraction)


def is_cic_s10a_type(type_id: str) -> bool:
    """True when this document type is the borrower's CIC credit report."""

    doc = get_type(type_id)
    return bool(doc and doc.cic_s10a_extraction)


def is_cic_r21_type(type_id: str) -> bool:
    """True when this document type is a CIC R20/R21 collateral report."""

    doc = get_type(type_id)
    return bool(doc and doc.cic_r21_extraction)


def is_sitevisit_type(type_id: str) -> bool:
    """True when this document type is an appraisal/verification report."""

    doc = get_type(type_id)
    return bool(doc and doc.sitevisit_extraction)


def routing_signature(type_id: str) -> frozenset[str]:
    doc = get_type(type_id)
    return doc.routing_signature if doc else frozenset()


def describe_types_for_prompt() -> str:
    """Render the type catalogue for the classifier LLM prompt.

    Built from the matrix so the prompt can never list a type the matrix does
    not define (or miss one it does).
    """

    lines: list[str] = []
    current_group: str | None = None
    for doc in all_types().values():
        if doc.group_id != current_group:
            current_group = doc.group_id
            lines.append(f"\n{doc.group_stt}. {doc.group_label}")
        label = " ".join(doc.label.split())
        if len(label) > 180:
            label = label[:180].rstrip() + "..."
        lines.append(f'- "{doc.id}" ({doc.stt}): {label}')
    return "\n".join(lines).strip()
