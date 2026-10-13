"""CorpIncomeAgent — OCR and check the "Pháp lý doanh nghiệp" items (1-4) of the BRD.

Pipeline (mirrors B1-B6 of the BRD, sheet "Corp income (detail)"):

    files ──► B1 OCR (src/utils/extractors.py, same engine as retail)
          ──► classify (document_types.yaml, keyword scoring)
          ──► LLM extraction, only the passes the type needs (one call each)
          ──► + B2 internal data / B3 external data passed in as JSON
          ──► + B4 watchlist (AML/BL/WL) passed in as a list
          ──► B6 deterministic cross-checks (checks.py)
          ──► Markdown report + JSON

The agent is standalone: it does not touch the retail Supervisor graph or the
retail document matrix, so adding it cannot change how a retail file is routed.
Plugging it into the supervisor later is a matter of calling ``run`` from a node.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from src.agents.corp_income.checks import parse_date, run_all_checks
from src.agents.corp_income.documents import PASSES, classify
from src.agents.corp_income.extraction import build_chains, extract_document, normalize_amounts
from src.agents.corp_income.report import load_template, render_markdown, unknown_placeholders
from src.utils.common import SUPPORTED_EXTENSIONS

AGENT_NAME = "CORP_INCOME_AGENT"

_PROVIDED_LABELS = {
    "internal": "Internal data (BCDE/BEP/DBX)",
    "external": "External data (Cổng ĐKDN/cơ quan thuế/BHXH)",
}


@dataclass
class CorpIncomeResult:
    meta: dict[str, Any]
    documents: list[dict[str, Any]]
    records: list[dict[str, Any]]
    items: dict[int, dict[str, Any]]
    markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "documents": self.documents,
            "records": self.records,
            "items": {str(k): v for k, v in self.items.items()},
        }

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {"markdown": out / "corp_income_report.md", "json": out / "corp_income_result.json"}
        paths["markdown"].write_text(self.markdown, encoding="utf-8")
        paths["json"].write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return paths


@dataclass
class CorpIncomeAgent:
    """Extraction LLM (any LangChain chat model) plus run settings.

    ``chains`` can be passed instead of ``llm`` — one object with ``.invoke``
    per pass name — which is how the check script runs without an API key.
    """

    llm: Any = None
    chains: dict[str, Any] | None = None
    max_workers: int = 3
    max_chars_per_document: int = 120_000
    ocr_timeout_seconds: float | None = None
    # Report layout. None = src/templates/corp-income-structure.md if it exists,
    # otherwise the built-in layout (report.DEFAULT_TEMPLATE).
    template_path: str | None = None
    _chains: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._chains = self.chains if self.chains is not None else build_chains(self.llm)

    # ------------------------------------------------------------------ #
    @staticmethod
    def discover(paths: list[str | Path]) -> list[Path]:
        files: list[Path] = []
        for raw in paths:
            path = Path(raw).expanduser()
            candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
            files += [p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        return files

    def _read_text(self, path: Path) -> str:
        # Imported here so that classification and checks work (and can be
        # tested) on a machine without Tesseract.
        from src.utils.extractors import extract_document_text

        return extract_document_text(str(path), ocr_timeout_seconds=self.ocr_timeout_seconds)

    def _process(self, filename: str, text: str, override: str | None, path: str) -> tuple[dict, list[dict]]:
        text = (text or "")[: self.max_chars_per_document]
        doc_class = classify(filename, text, override)
        results, errors = extract_document(self._chains, filename, text, doc_class["passes"])
        document = {
            "filename": filename,
            "path": path,
            "type": doc_class["type"],
            "label": doc_class["label"],
            "source": doc_class["source"],
            "classification_method": doc_class["method"],
            "classification_scores": doc_class["scores"],
            "passes_done": sorted(results),
            "errors": errors,
            "chars": len(text),
        }
        records = [
            {"source": doc_class["source"], "document": filename, "document_type": doc_class["type"],
             "pass": pass_name, "data": data}
            for pass_name, data in results.items()
        ]
        return document, records

    @staticmethod
    def _provided_records(kind: str, provided: dict[str, Any] | None) -> list[dict[str, Any]]:
        records = []
        for pass_name, data in (provided or {}).items():
            if pass_name not in PASSES or not isinstance(data, dict):
                raise ValueError(f"{kind}_data: key must be one of {PASSES}, got {pass_name!r}")
            records.append({
                "source": kind, "document": _PROVIDED_LABELS[kind], "document_type": f"{kind}_data",
                "pass": pass_name, "data": normalize_amounts(pass_name, json.loads(json.dumps(data))),
            })
        return records

    # ------------------------------------------------------------------ #
    def run(
        self,
        paths: list[str | Path] | None = None,
        *,
        texts: dict[str, str] | None = None,
        loan_date: str | date | None = None,
        borrower_names: list[str] | None = None,
        internal_data: dict[str, Any] | None = None,
        external_data: dict[str, Any] | None = None,
        watchlist: list[dict[str, Any]] | None = None,
        type_overrides: dict[str, str] | None = None,
    ) -> CorpIncomeResult:
        """Process the dossier and return the four BRD items.

        paths          files or folders (PDF, photos, XLSX, TXT... — whatever the retail extractor reads)
        texts          {filename: already-extracted text}, skips OCR (tests, re-runs)
        loan_date      the 12-month windows end here; today if omitted
        borrower_names individual borrower(s) of this loan, for the BRD item-2 warning
        internal_data  {pass: JSON in the pass schema} from BCDE/BEP/DBX (KHDN ETC TCB)
        external_data  {pass: JSON in the pass schema} from Cổng ĐKDN/thuế/BHXH APIs
        watchlist      [{"name", "tax_code", "id_number", "list": "AML|BL|WL"}]
        type_overrides {filename: document type id} when keywords cannot tell
        """

        as_of = loan_date if isinstance(loan_date, date) else (parse_date(loan_date) or date.today())
        overrides = type_overrides or {}
        jobs: list[tuple[str, str | None, str | None]] = []  # (filename, text or None, path)
        for path in self.discover(paths or []):
            jobs.append((path.name, None, str(path)))
        for filename, text in (texts or {}).items():
            jobs.append((filename, text, None))

        def work(job):
            filename, text, path = job
            if text is None:
                try:
                    text = self._read_text(Path(path))
                except Exception as exc:  # one unreadable scan must not end the run
                    return ({"filename": filename, "path": path, "type": "khac", "label": "Không đọc được",
                             "source": "customer", "classification_method": "none", "classification_scores": {},
                             "passes_done": [], "errors": {"ocr": f"{type(exc).__name__}: {exc}"[:300]}, "chars": 0}, [])
            return self._process(filename, text, overrides.get(filename), path or "")

        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, len(jobs) or 1))) as pool:
            outcomes = list(pool.map(work, jobs))

        documents = [doc for doc, _ in outcomes]
        records = [rec for _, recs in outcomes for rec in recs]
        records += self._provided_records("internal", internal_data)
        records += self._provided_records("external", external_data)

        template_text, template_source = load_template(self.template_path)
        items = run_all_checks(records, as_of=as_of, borrower_names=borrower_names, watchlist=watchlist)
        result = CorpIncomeResult(
            meta={
                "agent": AGENT_NAME,
                "as_of": as_of.strftime("%d/%m/%Y"),
                "borrower_names": list(borrower_names or []),
                "has_internal_data": bool(internal_data),
                "has_external_data": bool(external_data),
                "watchlist_size": len(watchlist or []),
                "template": template_source,
                "unknown_placeholders": unknown_placeholders(template_text),
            },
            documents=documents,
            records=records,
            items=items,
        )
        result.markdown = render_markdown({"meta": result.meta, "documents": documents, "items": items},
                                          template_text)
        return result
