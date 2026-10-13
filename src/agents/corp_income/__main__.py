"""Command line: python -m src.agents.corp_income <files/folders> [options]

    python -m src.agents.corp_income testing/samples/khdn_abc \
        --loan-date 13/10/2026 --borrower "Nguyễn Văn A" \
        --internal internal.json --external external.json --watchlist watchlist.json \
        --out logs/corp_income
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str | None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Corp income — pháp lý doanh nghiệp (BRD mục 1-4)")
    parser.add_argument("paths", nargs="+", help="files or folders of the corporate dossier")
    parser.add_argument("--loan-date", help="dd/mm/yyyy; 12-month windows end here (default: today)")
    parser.add_argument("--borrower", action="append", default=[], help="individual borrower name (repeatable)")
    parser.add_argument("--internal", help="JSON {pass: data} from internal systems")
    parser.add_argument("--external", help="JSON {pass: data} from Cổng ĐKDN/thuế/BHXH")
    parser.add_argument("--watchlist", help="JSON list [{name, tax_code, id_number, list}]")
    parser.add_argument("--type", action="append", default=[], metavar="FILE=TYPE",
                        help="force a document type, e.g. scan01.pdf=gcn_dkdn")
    parser.add_argument("--out", default="logs/corp_income", help="output folder")
    args = parser.parse_args()

    from dotenv import load_dotenv

    from src.agents.corp_income import CorpIncomeAgent
    from src.config import build_llm

    load_dotenv()
    agent = CorpIncomeAgent(llm=build_llm("MODEL_CORP_INCOME_EXTRACTION", temperature=0.0))
    result = agent.run(
        args.paths,
        loan_date=args.loan_date,
        borrower_names=args.borrower,
        internal_data=_load_json(args.internal),
        external_data=_load_json(args.external),
        watchlist=_load_json(args.watchlist),
        type_overrides=dict(item.split("=", 1) for item in args.type),
    )
    paths = result.save(args.out)
    for number, item in result.items.items():
        print(f"{number}. {item['title']}: {item['conclusion']}")
    print(f"\nReport: {paths['markdown']}\nJSON:   {paths['json']}")


if __name__ == "__main__":
    main()
