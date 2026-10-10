"""Every template must survive the code that will meet it.

A guidance or structure file is not prose: it is pasted into a LangChain prompt
template, and the blocks it shows the model are copied into real reports almost
verbatim. Three ways that went wrong in the corporate version of this project,
each invisible until a report was generated:

- Mermaid fences split across two lines or indented four spaces, so no diagram
  was drawn.
- A guidance file naming an injected data block that no code produced, so the
  model was told to read something it never received.
- (New with the retail rewrite) a single brace in a template. The system prompt
  goes through ChatPromptTemplate, where "{x}" is a required input variable — a
  stray brace in guidance text turns into a missing-variable error at the first
  LLM call. Placeholders must be written "{{Name}}".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.diagrams import mermaid_to_html  # noqa: E402

fails: list[str] = []
TEMPLATES = sorted((REPO / "src" / "templates").glob("*.md"))
AGENTS = (
    "customer-profile",
    "income-analysis",
    "credit-relationship",
    "credit-proposal",
    "risk-assessment",
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'✅' if ok else '❌'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def template(name: str) -> str:
    return (REPO / "src/templates" / f"{name}.md").read_text(encoding="utf-8")


print("1. Mỗi agent có đủ cặp guidance + structure")
for name in AGENTS:
    check(
        f"{name}: có đủ hai file",
        (REPO / f"src/templates/{name}-guidance.md").is_file()
        and (REPO / f"src/templates/{name}-structure.md").is_file(),
    )

print("\n2. Không có dấu ngoặc nhọn đơn (ChatPromptTemplate coi là biến)")
for path in TEMPLATES:
    body = path.read_text(encoding="utf-8")
    stripped = re.sub(r"\{\{[A-Za-z0-9_]+\}\}", "", body)
    stray = re.findall(r"[{}]", stripped)
    check(f"{path.name}: không có ngoặc đơn", not stray, f"{len(stray)} ký tự")

print("\n3. Ví dụ Mermaid dựng được thành SVG")
for path in TEMPLATES:
    body = path.read_text(encoding="utf-8")
    if "```mermaid" not in body:
        continue
    fences = re.findall(r"^```mermaid$", body, re.M)
    # A fence named only inline in prose ("hàng rào ```mermaid") is a mention,
    # not a block; only line-start fences are diagrams.
    if not fences:
        continue
    rendered = mermaid_to_html(body)
    check(f"{path.name}: mọi sơ đồ ra SVG",
          rendered.count("<svg") == len(fences),
          f"{rendered.count('<svg')}/{len(fences)}")
check(
    "income-analysis-structure có sơ đồ dòng tiền",
    bool(re.search(r"^```mermaid$", template("income-analysis-structure"), re.M)),
)

print("\n4. Guidance không viện dẫn khối dữ liệu không tồn tại")
code = "\n".join(
    p.read_text(encoding="utf-8") for p in (REPO / "src").rglob("*.py")
)
for name in AGENTS:
    body = template(f"{name}-guidance")
    # Bracketed text with no lower-case letter: that is how the prompt names its
    # injected blocks ([BẢNG TÍNH KHẢ NĂNG TRẢ NỢ], ...).
    labels = {m.strip() for m in re.findall(r"\[([^\]a-z]{6,60})\]", body)}
    missing = sorted(lbl for lbl in labels if f"[{lbl}]" not in code)
    check(f"{name}: mọi khối viện dẫn đều có trong mã",
          not missing, f"thiếu: {missing}" if missing else f"{len(labels)} khối")

print("\n5. Không còn dấu vết SME trong template")
for path in TEMPLATES:
    body = path.read_text(encoding="utf-8")
    leftovers = [
        word for word in ("BCTC", "doanh thu VAT", "vat-doanh-thu", "GSO", "sổ 131", "SME")
        if word in body
    ]
    check(f"{path.name}: sạch", not leftovers, ", ".join(leftovers))

print("\n" + "=" * 66)
if fails:
    print("❌ HỎNG:", *fails, sep="\n   - ")
    sys.exit(1)
print("✅ Mọi template đều chạy qua được đúng bộ phân tích của nó")
