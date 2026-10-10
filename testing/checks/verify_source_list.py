"""The source list is computed in code, names every file, and hides none of them."""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import markdown as md  # noqa: E402

from src.agents import prompt_blocks  # noqa: E402
from src.matrix.document_matrix import get_type  # noqa: E402
from src.types import ClassifiedDocument  # noqa: E402
from src.utils.source_list import MAX_DESCRIPTION_WORDS, build_source_lines  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"   {'✅' if ok else '❌'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


PAYSLIP = "Phiếu lương tháng tại Công ty ABC"
TKT16 = [(f"PHIEU LUONG {m:02d}.2025.pdf", "xac_nhan_luong", PAYSLIP) for m in range(1, 13)] \
    + [(f"PHIEU LUONG {m:02d}.2026.pdf", "xac_nhan_luong", PAYSLIP) for m in range(1, 5)]

# The invariant, stated once and applied to everything below. The old version of
# this file had to decompress a period range ("01-12/2025" -> twelve) to count
# what a line stood for, so the check carried arithmetic of its own that could be
# wrong in the same way the code was. Now the files are printed, so the test is
# a set comparison and has nowhere to hide a mistake.
def missing(cases):
    lines = build_source_lines(cases)
    body = "\n".join(lines)
    return [n for n, _, _ in cases if n.replace("|", r"\|") not in body], lines


print("1. Không tệp nào biến mất")
CASES = {
    "16 phiếu lương": TKT16,
    "2 sao kê không mô tả": [(f"SAO KE {y}.pdf", "sao_ke_tai_khoan", "") for y in (2025, 2026)],
    "3 hợp đồng cho thuê, mô tả khác nhau": [
        ("HD_THUE_1.txt", "hop_dong_cho_thue", "Hợp đồng cho thuê căn hộ A12 với ông Lê C, ký 01/2025"),
        ("HD_THUE_2.txt", "hop_dong_cho_thue", "Hợp đồng cho thuê mặt bằng 45 Trần Phú với bà Đỗ D"),
        ("HD_THUE_3.pdf", "hop_dong_cho_thue", "Hợp đồng cho thuê xe 7 chỗ với Công ty E, ký 03/2026"),
    ],
    "trộn nhiều loại": TKT16[:3] + [
        ("SAO KE 2026.pdf", "sao_ke_tai_khoan", ""),
        ("HDLD.pdf", "hop_dong_lao_dong", "Hợp đồng lao động không xác định thời hạn với Công ty ABC"),
    ],
    "không rõ loại": [("La_thu.pdf", "", ""), ("Ghi_chu.docx", "", "")],
    "trùng tên khác đuôi": [("CCCD.pdf", "giay_to_tuy_than", ""), ("CCCD.jpg", "giay_to_tuy_than", "")],
    # A pipe anywhere in the line ends the table cell it lands in, and the
    # header's column count then discards the rest — a document gone from the
    # list without a trace.
    "có dấu | trong tên và mô tả": [
        ("Hop|dong 2024.pdf", "hop_dong_cho_thue", "Hợp đồng A|B với ông Lê C"),
    ],
}
for name, cases in CASES.items():
    gone, lines = missing(cases)
    check(f"{name:30} {len(cases):2} tệp -> {len(lines)} dòng", not gone,
          f"THIẾU {gone}" if gone else "")

print("\n2. Gộp hay tách là do mô tả quyết định")
check("cùng mô tả -> một dòng", len(build_source_lines(TKT16)) == 1)
check("khác mô tả -> mỗi tệp một dòng",
      len(build_source_lines(CASES["3 hợp đồng cho thuê, mô tả khác nhau"])) == 3)
check("không mô tả -> một dòng, dùng short_label",
      build_source_lines(CASES["2 sao kê không mô tả"])
      == [f"**{get_type('sao_ke_tai_khoan').short_label}** —  "
          "<em>SAO KE 2025.pdf, SAO KE 2026.pdf</em>"],
      str(build_source_lines(CASES["2 sao kê không mô tả"])))

print("\n3. Mô tả quá dài bị bỏ, KHÔNG bị cắt cụt")
long_desc = " ".join(["từ"] * (MAX_DESCRIPTION_WORDS + 5))
line = build_source_lines([("x.pdf", "sao_ke_tai_khoan", long_desc)])[0]
check(f"quá {MAX_DESCRIPTION_WORDS} từ -> lùi về short_label",
      line.startswith(f'**{get_type("sao_ke_tai_khoan").short_label}**'), line[:60])
check("không có dấu chấm lửng", "…" not in line and "..." not in line)
ok_desc = " ".join(["từ"] * MAX_DESCRIPTION_WORDS)
check(f"đúng {MAX_DESCRIPTION_WORDS} từ -> vẫn dùng",
      build_source_lines([("x.pdf", "sao_ke_tai_khoan", ok_desc)])[0]
      .startswith(f"**{ok_desc}**"))

print("\n3b. Phần mô tả in đậm, tên tệp in nghiêng")
for name, cases in CASES.items():
    _, lines = missing(cases)
    check(f"{name:30} mọi dòng có ** và <em>",
          all(l.startswith("**") and "** —  <em>" in l and l.endswith("</em>") for l in lines),
          lines[0][:60] if lines else "")

print("\n4. Đuôi tệp còn nguyên")
for name, cases in CASES.items():
    _, lines = missing(cases)
    body = "\n".join(lines)
    exts = {n.rsplit(".", 1)[-1] for n, _, _ in cases if "." in n}
    check(f"{name:30} giữ đuôi {sorted(exts)}",
          all(f".{e}" in body for e in exts))

print("\n5. Ô bảng dựng được thật — đây là chỗ phương án danh sách thật đã hỏng")
docs = [ClassifiedDocument(path=f"/x/{n}", filename=n, content="c",
                           agent="INCOME_ANALYSIS_AGENT", reasoning="",
                           confidence=1.0, document_type=t, description=d)
        for n, t, d in CASES["trộn nhiều loại"]]
block = prompt_blocks._build_source_list_block(docs)
rows = [l[2:] for l in block.splitlines() if l.startswith("- ")]
cell = "<br>".join(f"- {r}" for r in rows)
table = ("| Thông tin chung | &nbsp; |\n| --- | --- |\n"
         f"| *Nguồn dữ liệu* | {cell} |\n")
html = md.markdown(table, extensions=["tables", "fenced_code", "footnotes"])
check("khối prompt sinh ra các dòng", bool(rows), f"{len(rows)} dòng")
check("mọi dòng vào đúng MỘT ô", html.count("<td>") == 2, str(html.count("<td>")))
check("số dòng giữ nguyên sau khi dựng", html.count("<br") == len(rows) - 1,
      f"{html.count('<br')} <br> cho {len(rows)} dòng")
check("thẻ <em> sống sót", html.count("<em>") >= len(rows))
check("phần mô tả in đậm sống sót", html.count("<strong>") == len(rows),
      f"{html.count('<strong>')} <strong> cho {len(rows)} dòng")
check("không tệp nào rơi khi dựng",
      all(n in html for n, _, _ in CASES["trộn nhiều loại"]))
# The shape that was rejected: a real list inside the cell.
broken = md.markdown("| a | b |\n| --- | --- |\n| *Nguồn* | - dòng 1\n- dòng 2 |\n",
                     extensions=["tables"])
check("bằng chứng: danh sách thật trong ô bảng vẫn hỏng", broken.count("<li>") == 0)

print("\n" + "=" * 66)
if fails:
    print("❌ HỎNG:", *fails, sep="\n   - ")
    sys.exit(1)
print("✅ Danh sách nguồn: đủ tệp, đủ đuôi, gộp đúng, dựng được trong ô bảng")
