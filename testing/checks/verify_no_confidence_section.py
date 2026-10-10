"""Verify '## Kiểm chứng độ tin cậy' never appears in _finalize's response.

Originally this also asserted the hallucination check still ran and left its
data in the state dict. a53f0f4 removed that guardrail outright, so the second
half of the premise is gone — what remains worth guarding is that the section
does not come back.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env", override=True)

from src.config import Config
from src.agents.supervisor import Supervisor

# hallucination_llm gỡ khỏi Config ở a53f0f4.
config = Config()

supervisor = Supervisor(config)

fake_response = """## Báo cáo thông tin khách hàng

### CÔNG TY MẪU ABC

**Nhận định**:

- Doanh thu tăng **20%** so với cùng kỳ.
"""

result = supervisor._finalize(
    input_text="Tóm tắt thông tin khách hàng.",
    response=fake_response,
    agent_name="CUSTOMER_PROFILE_AGENT",
    decision={"route": "SINGLE_AGENT"},
    documents=[],
    sub_agent_outputs={},
    web_context="",
    history_context="",
    execution_plan={},
    gap_analysis={"summary": "Đủ dữ liệu."},
    steps=[],
)

final_text = result["response"]
print("\n=== FINAL RESPONSE ===")
print(final_text)
assert "Kiểm chứng độ tin cậy" not in final_text, "confidence section still present!"
assert "hallucination_check" not in result, "hallucination_check khôi phục lại rồi?"
print("\nOK: không có mục kiểm chứng độ tin cậy, không còn dữ liệu hallucination")
