"""Bullets sit on the body's left edge, and no list marker hangs off the margin.

The Nhận định block is a bullet list, and the user agent's default padding-left
of 40px stepped it in from the paragraph that introduces it, so it read as a
sub-level of that sentence rather than as its substance. Removing the step is one
line of CSS; keeping it removed is what this file is for, because an outside list
marker is drawn to the LEFT of the content box and every marker is a different
width. Two regressions were measured while writing it: at 6pt padding "1." was
printed 4.6pt outside the page margin, and at 14pt "10." was still 1.8pt outside.
Neither is visible in the HTML — only in the PDF.

Everything below is measured off a rendered PDF for that reason, and the margin
is taken from where a plain paragraph actually lands rather than hardcoded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:
    import weasyprint
except OSError:
    # macOS: WeasyPrint dlopens its Pango/GObject libraries through cffi, and
    # dyld reads DYLD_* only at exec — setting it in-process would be too late.
    HOMEBREW = "/opt/homebrew/lib"
    if os.path.isdir(HOMEBREW) and os.environ.get("_LIST_INDENT_REEXEC") != "1":
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
            HOMEBREW + ":" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        )
        os.environ["_LIST_INDENT_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    raise

import fitz  # noqa: E402
import markdown as md  # noqa: E402

from src.utils.report_style import REPORT_CSS  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'✅' if ok else '❌'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def render(body_html: str) -> fitz.Page:
    html = (
        "<html><head><meta charset='utf-8'>"
        f"<style>{REPORT_CSS}</style></head><body>{body_html}</body></html>"
    )
    out = REPO / "logs" / "list_indent_probe.pdf"
    out.parent.mkdir(exist_ok=True)
    weasyprint.HTML(string=html).write_pdf(out)
    return fitz.open(out)[0]


def words(page: fitz.Page) -> list[tuple[str, float]]:
    return [(w[4], w[0]) for w in page.get_text("words")]


def first_x(page: fitz.Page, token: str) -> float:
    return next(x for t, x in words(page) if t == token)


MD = """MỐC lề trái của thân bài.

- ĐẦU một gạch đầu dòng dài, đủ để phải gãy sang dòng thứ hai và cho thấy phần
  thân của nó treo lề ra sao khi in ra giấy A4 trong bản báo cáo thật.
- **Hồ sơ**:
    - LỒNG một mục con
- CUỐI gạch cuối cùng.
"""

page = render(md.markdown(MD, extensions=["tables", "fenced_code", "footnotes"]))
margin = first_x(page, "MỐC")
bullets = sorted(x for t, x in words(page) if t == "•")
nested = [x for t, x in words(page) if t == "◦"]

print(f"1. Gạch đầu dòng thẳng lề thân bài (lề đo được: {margin:.1f}pt)")
check("dấu chấm nằm đúng mép chữ thân bài",
      abs(bullets[0] - margin) < 0.6, f"{bullets[0]:.1f} vs {margin:.1f}")
check("mọi gạch cùng cấp cùng một lề",
      max(bullets) - min(bullets) < 0.6, f"{bullets}")

print("\n2. Dòng gãy vẫn treo lề, không tụt về dưới dấu chấm")


def wrapped_pair(page: fitz.Page, opener: str) -> tuple[float, float]:
    """Left edge of the line starting with `opener`, and of the line after it.

    By line bbox, not by looking up a word: the first attempt searched for a word
    from the second line and found the same word mid-sentence on the first,
    reporting a 63pt discrepancy that was entirely the test's own.
    """

    for block in page.get_text("dict")["blocks"]:
        lines = block.get("lines", [])
        for index, line in enumerate(lines[:-1]):
            text = "".join(sp["text"] for sp in line["spans"]).strip()
            if text.startswith(opener):
                return line["bbox"][0], lines[index + 1]["bbox"][0]
    raise AssertionError(f"không thấy dòng nào mở đầu bằng {opener!r}")


head, wrapped = wrapped_pair(page, "ĐẦU")
check("dòng sau thẳng với dòng đầu của cùng gạch",
      abs(wrapped - head) < 0.6, f"{wrapped:.1f} vs {head:.1f}")
check("thân gạch lùi vào so với dấu chấm", head > bullets[0] + 1.0,
      f"{head:.1f} > {bullets[0]:.1f}")

print("\n3. Cấp lồng vẫn phải trông như cấp lồng")
check("có mục lồng", bool(nested))
check("bước lùi đủ thấy được", nested and nested[0] - bullets[0] >= 10.0,
      f"{nested[0] - bullets[0]:.1f}pt" if nested else "")

print("\n4. Không marker nào thò ra ngoài lề trang")
# Twelve items, so the two-digit numerals are actually printed. "10." is the
# widest marker the report can produce and it is the one that kept escaping.
items = "".join(f"<li>Mục {i}</li>" for i in range(1, 13))
ol_page = render(f"<p>MỐC lề</p><ol>{items}</ol>")
ol_margin = first_x(ol_page, "MỐC")
markers = [(t, x) for t, x in words(ol_page)
           if t.endswith(".") and t[:-1].isdigit()]
outside = [(t, x) for t, x in markers if x < ol_margin - 0.1]
check("danh sách đánh số 1..12 nằm trọn trong lề", not outside,
      f"thò ra: {outside}" if outside else f"trái nhất {min(x for _, x in markers):.1f}"
      f" >= {ol_margin:.1f}")
check("gạch đầu dòng cũng vậy", bullets[0] >= margin - 0.1)

print("\n5. Footnote giữ luật riêng của nó")
fn = render(md.markdown("Câu có dẫn nguồn.[^1]\n\n[^1]: Sao kê tài khoản 2026, trang 4.\n",
                        extensions=["footnotes"]))
fn_x = next(x for t, x in words(fn) if t == "1.")
check("danh sách footnote không bị luật chung kéo theo",
      fn_x > first_x(fn, "Câu"), f"{fn_x:.1f}")

print("\n" + "=" * 66)
if fails:
    print("❌ HỎNG:", *fails, sep="\n   - ")
    sys.exit(1)
print("✅ Gạch đầu dòng thẳng lề thân bài, không marker nào tràn ra ngoài lề")
