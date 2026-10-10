"""Verify duplicate-source citations merge without damaging the audit."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.citations import consolidate_footnotes

failures = []


def sources(text):
    return [l for l in text.splitlines() if l.startswith("[^")]


# --- 1. the reported case ----------------------------------------------------
REPORT = """## Phân tích

- Dữ kiện A[^pf1] và dữ kiện B[^pf2].
- Dữ kiện C[^pf3], dữ kiện D[^pf4].
- Dữ kiện E[^cr1] và F[^cr2].

---

[^pf1]: BAO CAO KHAO SAT.pdf, trang 1
[^pf2]: BAO CAO KHAO SAT.pdf, trang 1
[^pf3]: SO 154.xlsx, Sheet 1
[^pf4]: SO 131.xlsx, Sheet 1
[^cr1]: BAO CAO KHAO SAT.pdf, trang 1
[^cr2]: BAO CAO KHAO SAT.pdf, trang 2
"""
out, audit = consolidate_footnotes(REPORT)
got = sources(out)
print("1. Danh sách nguồn sau khi gộp")
for i, line in enumerate(got, 1):
    print(f"   {i}. {line}")
expected = [
    "[^pf1]: BAO CAO KHAO SAT.pdf, trang 1",
    "[^pf3]: SO 154.xlsx, Sheet 1",
    "[^pf4]: SO 131.xlsx, Sheet 1",
    "[^cr2]: BAO CAO KHAO SAT.pdf, trang 2",
]
if got != expected:
    failures.append(f"list mismatch:\n  got={got}\n  want={expected}")

# --- 2. body markers remapped, nothing dangling ------------------------------
body = out.split("\n---\n")[0]
print()
print("2. Thân bài sau khi viết lại")
for line in body.splitlines():
    if "[^" in line:
        print(f"   {line}")
if "[^pf2]" in body or "[^cr1]" in body:
    failures.append("merged labels still referenced in body")
if audit.orphan_references:
    failures.append(f"orphans after merge: {audit.orphan_references}")
if audit.unused_definitions:
    failures.append(f"unexpected unused: {audit.unused_definitions}")

# --- 3. idempotent -----------------------------------------------------------
twice, audit2 = consolidate_footnotes(out)
print()
print(f"3. Idempotent: {'OK' if twice == out else 'FAIL'}")
if twice != out:
    failures.append("second consolidate changed the text")

# --- 4. genuinely different sources stay apart -------------------------------
print("4. Không gộp nhầm:", end=" ")
distinct = [s for s in got if "trang 1" in s or "trang 2" in s]
if len(distinct) != 2:
    failures.append(f"trang 1 / trang 2 collapsed: {distinct}")
if not any("SO 154" in s for s in got) or not any("SO 131" in s for s in got):
    failures.append("SO 154 / SO 131 collapsed")
print("trang1≠trang2, SO154≠SO131 OK")

# --- 5. code fences untouched ------------------------------------------------
FENCED = """Câu A[^x1] và câu B[^x2].

```text
ví dụ cú pháp: [^x2] và [^x2]: nguồn giả
```

---

[^x1]: F.pdf, trang 1
[^x2]: F.pdf, trang 1
"""
fenced_out, _ = consolidate_footnotes(FENCED)
print()
print("5. Khối code")
inside = fenced_out.split("```text")[1].split("```")[0]
print(f"   nội dung fence: {inside.strip()!r}")
if "[^x2]" not in inside:
    failures.append("rewrote a marker inside a code fence")
if len(sources(fenced_out)) != 1:
    failures.append(f"fenced case should still merge to 1: {sources(fenced_out)}")

# --- 6. audit still catches the three real problems --------------------------
PROBLEMS = """Câu A[^p1], câu B[^p9].

---

[^p1]: A.pdf, trang 1
[^p1]: A.pdf, trang 7
[^p2]: B.pdf, trang 3
"""
_, audit3 = consolidate_footnotes(PROBLEMS)
print()
print("6. Audit")
print(f"   orphan_references    : {audit3.orphan_references}")
print(f"   unused_definitions   : {audit3.unused_definitions}")
print(f"   duplicate_definitions: {audit3.duplicate_definitions}")
if audit3.orphan_references != ["p9"]:
    failures.append(f"orphan not caught: {audit3.orphan_references}")
if [l for l, _ in audit3.unused_definitions] != ["p2"]:
    failures.append(f"unused not caught: {audit3.unused_definitions}")
if [l for l, _ in audit3.duplicate_definitions] != ["p1"]:
    failures.append(f"duplicate not caught: {audit3.duplicate_definitions}")

# --- 7. redundant unused definition is NOT reported --------------------------
REDUNDANT = """Câu A[^r1].

---

[^r1]: C.pdf, trang 2
[^r2]: C.pdf, trang 2
"""
red_out, audit4 = consolidate_footnotes(REDUNDANT)
print()
print("7. Định nghĩa thừa trùng nguồn (không nên báo unused)")
print(f"   sources: {sources(red_out)}")
print(f"   unused : {audit4.unused_definitions}")
if len(sources(red_out)) != 1:
    failures.append(f"redundant definition not merged: {sources(red_out)}")
if audit4.unused_definitions:
    failures.append(f"redundant definition reported as unused: {audit4.unused_definitions}")

print()
if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
