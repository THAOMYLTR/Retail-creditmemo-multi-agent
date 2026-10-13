"""Corporate document types and keyword classification for the corp_income agent.

The catalogue lives in ``document_types.yaml`` next to this file. Scoring
mirrors the retail classifier (src/agents/document_classification.py): text and
keywords are both accent-stripped and lowercased with ``normalize_text``, a
keyword found in the filename counts three times one found in the content, and
the highest score wins.

Unlike the retail classifier there is no LLM fallback here: a corporate file
that matches nothing is reported as ``khac`` (unclassified) and no extraction
pass runs on it. The caller can force a type per file with ``type_overrides``
when the filename and content say nothing useful — cheaper and more
predictable than asking a model to guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.common import normalize_text

TYPES_PATH = Path(__file__).with_name("document_types.yaml")

PASSES = ("legal_basic", "shareholders", "legal_rep", "liabilities")
SOURCES = ("customer", "field", "external", "internal")
UNCLASSIFIED = "khac"
FILENAME_KEYWORD_WEIGHT = 3


@dataclass(frozen=True)
class CorpDocumentType:
    id: str
    label: str
    source: str
    keywords: tuple[str, ...]
    passes: tuple[str, ...]


@lru_cache(maxsize=1)
def load_types() -> dict[str, CorpDocumentType]:
    """Load and validate the catalogue. Fails loudly on a malformed entry."""

    raw = yaml.safe_load(TYPES_PATH.read_text(encoding="utf-8")) or {}
    types: dict[str, CorpDocumentType] = {}
    for entry in raw.get("types", []):
        type_id = entry["id"]
        if type_id in types:
            raise ValueError(f"document_types.yaml: duplicate id {type_id!r}")
        source = entry.get("source", "customer")
        if source not in SOURCES:
            raise ValueError(f"{type_id}: unknown source {source!r}")
        passes = tuple(entry.get("passes", []))
        unknown = set(passes) - set(PASSES)
        if unknown:
            raise ValueError(f"{type_id}: unknown passes {sorted(unknown)}")
        types[type_id] = CorpDocumentType(
            id=type_id,
            label=entry.get("label", type_id),
            source=source,
            keywords=tuple(normalize_text(k) for k in entry.get("keywords", []) if k),
            passes=passes,
        )
    return types


def _count(keyword: str, padded_text: str) -> int:
    # Whole-token match: "dieu le" must not fire inside "dieu lenh".
    return padded_text.count(f" {keyword} ")


def type_scores(filename: str, content: str) -> dict[str, int]:
    name = f" {normalize_text(Path(filename).stem)} "
    # The first pages carry the title; scoring the whole of a 40-page charter
    # lets incidental mentions ("theo điều lệ...") outvote the real type.
    body = f" {normalize_text(content[:6000])} "
    scores: dict[str, int] = {}
    for doc_type in load_types().values():
        score = 0
        for keyword in doc_type.keywords:
            score += FILENAME_KEYWORD_WEIGHT * _count(keyword, name)
            score += min(_count(keyword, body), 3)
        if score:
            scores[doc_type.id] = score
    return scores


def classify(filename: str, content: str, override: str | None = None) -> dict[str, Any]:
    """Return ``{"type", "label", "source", "passes", "scores", "method"}``."""

    types = load_types()
    if override:
        if override not in types:
            raise ValueError(f"Unknown corp document type override: {override!r}")
        chosen, method, scores = override, "override", {}
    else:
        scores = type_scores(filename, content)
        if scores:
            chosen = max(scores, key=lambda k: (scores[k], -list(types).index(k)))
            method = "keyword"
        else:
            chosen, method = UNCLASSIFIED, "none"

    doc_type = types.get(chosen)
    return {
        "type": chosen,
        "label": doc_type.label if doc_type else "Chưa phân loại",
        "source": doc_type.source if doc_type else "customer",
        "passes": list(doc_type.passes) if doc_type else [],
        "scores": scores,
        "method": method,
    }
