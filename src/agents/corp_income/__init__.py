"""corp_income — corporate (KHDN) legal-information agent, BRD "Corp Income Assessment" items 1-4.

    from src.agents.corp_income import CorpIncomeAgent
    from src.config import build_llm

    agent = CorpIncomeAgent(llm=build_llm("MODEL_CORP_INCOME_EXTRACTION", temperature=0.0))
    result = agent.run(["hoso_KHDN/"], loan_date="13/10/2026", borrower_names=["Nguyễn Văn A"])
    result.save("logs/corp_income")

See docs/CORP_INCOME.md.
"""

from src.agents.corp_income.agent import AGENT_NAME, CorpIncomeAgent, CorpIncomeResult

__all__ = ["AGENT_NAME", "CorpIncomeAgent", "CorpIncomeResult"]
