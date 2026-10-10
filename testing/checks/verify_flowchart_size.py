"""Flowcharts lay out into the page width instead of being scaled down to it.

The Business Activity section 1 supply chain is nine boxes in a row. It used to
come out 1,649px wide, get scaled to 45% to reach A4, and print at 4.7pt against
9.5pt body text — legible on screen, unreadable on paper.

The fix is the contract charts.py already uses: fix the width, let the content
arrange itself into it. What the constants say is then what prints.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.utils.graph_svg as G  # noqa: E402
from src.utils.diagrams import (  # noqa: E402
    _parse, _percent_only_edge_labels, mermaid_to_html)

fails = []


def check(name, ok, detail=""):
    print(f"   {'✅' if ok else '❌'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def measure(source):
    svg = G.render_svg(_parse(source))
    m = re.search(r'viewBox="0 0 (\d+) (\d+)" width="(\d+)" height="(\d+)"', svg)
    view_w, _, draw_w, _ = (int(x) for x in m.groups())
    # Boxes in one row are centred against the tallest, so a two-line label gives
    # its box a different y from its neighbours. Counting distinct y values would
    # report three rows where there are two — group into bands instead.
    ys = sorted(float(y) for y in re.findall(r'<rect x="[\d.]+" y="([\d.]+)"', svg))
    bands = []
    for y in ys:
        if not bands or y - bands[-1] > G.LINE_HEIGHT * 2:
            bands.append(y)
    return {
        "svg": svg,
        "width": view_w,
        "scale": draw_w / view_w,
        "font": G.FONT_SIZE * draw_w / view_w,
        "rows": len(bands),
    }


BA_SECTION_1 = (
    "flowchart LR\n  P1[Gỗ MDF phủ Melamine] --> KV[Công ty TNHH Gỗ An Cường] "
    "--> A[Đầu vào] --> B[Sản xuất] --> C[Tồn kho] --> D[Đầu ra] --> E[Thu tiền] "
    "--> KR[Chuỗi siêu thị nội thất Nhà Xinh] --> P2[Bàn ghế, tủ kệ]"
)
BA_SECTION_3 = (
    "flowchart LR\n  A[Nhận đơn hàng] --> B[Ký hợp đồng] --> C[Sản xuất] "
    "--> D[Giao hàng & thanh toán]"
)
WITH_EDGE_LABELS = (
    "flowchart LR\n  A[Đầu vào] -->|thanh toán 30 ngày| B[Sản xuất] "
    "-->|giao hàng| C[Đầu ra]"
)
BRANCHING = (
    "flowchart LR\n  A[Nhà cung cấp] --> B[Nhà máy]\n  A --> C[Kho]\n"
    "  B --> D[Khách hàng]\n  C --> D"
)

print("1. Sơ đồ mục 1 — không còn thu nhỏ")
one = measure(BA_SECTION_1)
check("scale = 1,00", abs(one["scale"] - 1.0) < 0.001, f"{one['scale']:.2f}")
check("cỡ chữ = đúng hằng số", abs(one["font"] - G.FONT_SIZE) < 0.01, f"{one['font']:.1f}px")
check("cỡ chữ trên GIẤY ≥ 10pt (px x 0,75)", one["font"] * G.PX_TO_PT >= 10.0,
      f"{one['font']:.1f}px = {one['font'] * G.PX_TO_PT:.1f}pt")
check("bọc thành nhiều dòng", one["rows"] >= 2, f"{one['rows']} dòng")
check("vừa khổ giấy", one["width"] <= G.PAGE_CONTENT_WIDTH, f"{one['width']}px")

print("\n2. Sơ đồ mục 3 — đủ hẹp, không bọc thừa")
three = measure(BA_SECTION_3)
check("vẫn 1 dòng", three["rows"] == 1, f"{three['rows']} dòng")
check("scale = 1,00", abs(three["scale"] - 1.0) < 0.001)

print("\n3. Sơ đồ có nhãn cạnh — khoảng cách rank vẫn rộng")
labelled = measure(WITH_EDGE_LABELS)
bare = measure("flowchart LR\n  A[Đầu vào] --> B[Sản xuất] --> C[Đầu ra]")
check("rộng hơn cùng sơ đồ không nhãn", labelled["width"] > bare["width"],
      f"{labelled['width']} vs {bare['width']}")
check("nhãn cạnh có mặt trong SVG", "thanh toán 30 ngày" in labelled["svg"])

print("\n4. Sơ đồ rẽ nhánh — không bọc dòng")
branch = measure(BRANCHING)
check("không bị bọc dòng (bố cục tầng giữ nguyên)",
      G._is_linear_chain.__call__ is not None and branch["scale"] == 1.0)
check("không thu nhỏ", abs(branch["scale"] - 1.0) < 0.001)

print("\n5. Chiều mũi tên qua chỗ bọc dòng")
paths = re.findall(r'<path d="([^"]+)"', one["svg"])
wrap = [p for p in paths if len(re.findall(r"[ML][\d.]+ [\d.]+", p)) == 4
        and float(re.findall(r"[ML]([\d.]+) ([\d.]+)", p)[0][1])
        < float(re.findall(r"[ML]([\d.]+) ([\d.]+)", p)[-1][1])]
check("số đường nối xuống dòng = số dòng - 1",
      len(wrap) == one["rows"] - 1, f"{len(wrap)} đường / {one['rows']} dòng")
if wrap:
    pts = [(float(x), float(y)) for x, y in re.findall(r"[ML]([\d.]+) ([\d.]+)", wrap[0])]
    check("đi xuống rồi sang TRÁI (không đảo chiều chuỗi)",
          pts[-1][0] < pts[0][0] and pts[-1][1] > pts[0][1],
          f"x {pts[0][0]:.0f}->{pts[-1][0]:.0f}, y {pts[0][1]:.0f}->{pts[-1][1]:.0f}")

print("\n6. Sơ đồ quá rộng: vẫn vẽ, không kèm chú thích nào")
deep = "flowchart LR\n" + "\n".join(
    f"  N{i}[Công đoạn sản xuất số {i}] --> N{i+1}[Công đoạn sản xuất số {i+1}]"
    for i in range(1, 10)) + "\n  N1 --> N5"
import contextlib as _ctx, io as _io  # noqa: E402
_buf = _io.StringIO()
with _ctx.redirect_stdout(_buf):
    wide = measure(deep)
check("vẫn vẽ ra sơ đồ", wide["width"] > 0)
check("KHÔNG in chú thích dưới hình", "mmd-note" not in wide["svg"])
check("KHÔNG dùng comment ẩn", "<!-- diagram scaled" not in wide["svg"])
check("KHÔNG in cảnh báo ra stdout", _buf.getvalue() == "", repr(_buf.getvalue()[:60]))

print("\n7. Bảng màu theo báo cáo, không phải mặc định mermaid")
check("bỏ tím #9370DB", "#9370DB" not in one["svg"])
check("dùng xanh của báo cáo #2f6f9f", "#2f6f9f" in one["svg"])

print("\n8. Phong cách: màu theo tầng, bóng vẽ tay")
import re as _re  # noqa: E402
pairs = _re.findall(r'fill="(#[0-9a-f]{6})" stroke="(#[0-9a-f]{6})"', one["svg"])
check("mỗi tầng một màu khác nhau", len({p for p in pairs}) >= 4, f"{len(set(pairs))} tổ hợp")
check("dùng đúng bảng LEVEL_COLOURS",
      all(p in G.LEVEL_COLOURS for p in pairs), str([p for p in pairs if p not in G.LEVEL_COLOURS][:2]))

shadows = _re.findall(r'fill="' + G.SHADOW_COLOUR + r'" opacity="([\d.]+)"', one["svg"])
check("mỗi khối có một bóng vẽ tay", len(shadows) == len(pairs), f"{len(shadows)} bóng / {len(pairs)} khối")
check("bóng mờ, không đặc", all(0 < float(o) < 0.3 for o in shadows), str(set(shadows)))
check("KHÔNG dùng feDropShadow (WeasyPrint bỏ qua không báo)",
      "feDropShadow" not in one["svg"])

print("\n9. Style do sơ đồ tự khai vẫn thắng bảng màu theo tầng")
warn = measure("flowchart LR\n  A[Khách A] --> B[Doanh nghiệp]\n  style A fill:#FFC000,stroke:#333")
check("giữ nguyên màu cảnh báo tập trung", "#FFC000" in warn["svg"])

print("\n10. Lề quanh chữ đồng đều ở MỌI khối, kể cả nhãn dài")
import re as _r  # noqa: E402
src10 = ("flowchart LR\n  A[Gỗ MDF phủ Melamine] --> B[Đầu ra] --> C[Công ty TNHH Gỗ An Cường] "
         "--> D[Thu tiền]")
svg10 = G.render_svg(_parse(src10))
chart10 = _parse(src10)
pads = []
for nid, lab in chart10.labels.items():
    lines = G._text_lines(lab)
    tw = max(G.text_width(l, G.FONT_SIZE) for l in lines)
    w = min(G.NODE_MAX_WIDTH, max(G.NODE_MIN_WIDTH, tw + 2 * G.NODE_PADDING_X))
    pads.append(round((w - tw) / 2, 1))
check("lề mỗi bên như nhau ở mọi khối", len(set(pads)) == 1, f"{sorted(set(pads))}px")
check("lề đúng bằng NODE_PADDING_X", pads[0] == G.NODE_PADDING_X, f"{pads[0]} vs {G.NODE_PADDING_X}")
check("đo bằng font thật, không đếm ký tự", G._measure_font() is not None)

print("\n11. Dây nối vào cùng một tầng thì dừng ở cùng một mép")
svg11 = G.render_svg(_parse(
    "flowchart LR\n  A[Nhà cung cấp A] --> M[Nhà máy]\n  B[Nhà cung cấp B] --> M\n"
    "  M --> C[Khách hàng lớn 78%]\n  M --> D[Khách hàng khác]"))
lefts = {round(float(x)) for x, y in
         _r.findall(r'<rect x="([\d.]+)" y="([\d.]+)"', svg11)[1::2]}
check("3 tầng -> đúng 3 mép trái", len(lefts) == 3, str(sorted(lefts)))
ends = {round([(float(a), float(b)) for a, b in
               _r.findall(r"[ML]([\d.-]+) ([\d.-]+)", d)][-1][0])
        for d in _r.findall(r'<path d="([^"]+)"', svg11)}
check("dây nối kết thúc ở đúng các mép đó", ends <= lefts, f"{sorted(ends)} vs {sorted(lefts)}")

print("\n12. Dây nối dài ngắn theo nhãn, và chừa chỗ cho mũi tên")
for name, source, label in (
    ("ngắn", "flowchart LR\n  A[Đầu vào] -->|giao hàng| B[Sản xuất]", "giao hàng"),
    ("vừa", "flowchart LR\n  A[Đầu vào] -->|thanh toán 30 ngày| B[Sản xuất]", "thanh toán 30 ngày"),
    ("dài", "flowchart LR\n  A[X] -->|thanh toán trong vòng 45 ngày kể từ ngày nghiệm thu| B[Y]",
     "thanh toán trong vòng 45 ngày kể từ ngày nghiệm thu"),
):
    svg12 = G.render_svg(_parse(source))
    boxes12 = sorted({(float(x), float(x) + float(w)) for x, w in
                      re.findall(r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)"', svg12)})[1::2]
    widest12 = max(G.text_width(l, G.EDGE_FONT_SIZE)
                   for l in G._text_lines(label))
    gap12 = boxes12[1][0] - boxes12[0][1]
    side = (gap12 - widest12 - G.ARROW_LENGTH) / 2
    # Ít nhất, không phải đúng bằng: RANK_GAP là sàn, nên nhãn ngắn được rộng
    # hơn mức tối thiểu. Điều phải giữ là không bao giờ HẸP hơn.
    check(f"nhãn {name}: chừa ít nhất {G.EDGE_LABEL_CLEARANCE}px mỗi bên",
          side >= G.EDGE_LABEL_CLEARANCE - 0.5, f"{side:.1f}px")

# nhãn phải nằm gọn giữa hộp nguồn và đầu mũi tên
svg12 = G.render_svg(_parse("flowchart LR\n  A[Đầu vào] -->|thanh toán 30 ngày| B[Sản xuất]"))
lx = float(re.search(r'<text x="([\d.]+)"[^>]*font-size="' + str(G.EDGE_FONT_SIZE), svg12).group(1))
boxes12 = sorted({(float(x), float(x) + float(w)) for x, w in
                  re.findall(r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)"', svg12)})[1::2]
half = G.text_width("thanh toán 30 ngày", G.EDGE_FONT_SIZE) / 2
check("nhãn không chạm hộp nguồn", lx - half > boxes12[0][1], f"{lx - half:.1f} > {boxes12[0][1]:.1f}")
check("nhãn không chạm đầu mũi tên",
      lx + half < boxes12[1][0] - G.ARROW_LENGTH,
      f"{lx + half:.1f} < {boxes12[1][0] - G.ARROW_LENGTH:.1f}")

print("\n13. Nhãn dài được vẽ ĐẦY ĐỦ, không cắt, không dấu …")
long_label = "thanh toán trong vòng 45 ngày kể từ ngày nghiệm thu"
svg_long = G.render_svg(_parse(f"flowchart LR\n  X[Ký hợp đồng] -->|{long_label}| Y[Nghiệm thu]"))
lines_long = re.findall(
    r'<text x="[\d.]+" y="[\d.-]+" font-size="' + str(G.EDGE_FONT_SIZE) + r'"[^>]*>([^<]*)</text>',
    svg_long)
check("ghép các dòng lại đúng bằng nhãn gốc",
      " ".join(lines_long) == long_label, " ".join(lines_long))
check("không có dấu … ở bất kỳ dòng nào",
      not any("…" in line for line in lines_long), str(lines_long))
guidance = (REPO / "src/templates/income-analysis-guidance.md").read_text(encoding="utf-8")
check("guidance đặt trần 10 từ cho nhãn mũi tên", "TỐI ĐA 10 TỪ" in guidance)
check("guidance nói rõ hệ thống vẽ đầy đủ", "vẽ ĐẦY ĐỦ nhãn" in guidance)

print("\n14. Nhãn nhiều dòng nằm gọn trong khung")
for label13 in ("giao hàng", "thanh toán trong vòng 45 ngày kể từ ngày nghiệm thu"):
    svg13 = G.render_svg(_parse(f"flowchart LR\n  X[Ký hợp đồng] -->|{label13}| Y[Nghiệm thu]"))
    h13 = int(re.search(r'viewBox="0 0 \d+ (\d+)"', svg13).group(1))
    ys13 = [float(m.group(1)) for m in
            re.finditer(r'<text x="[\d.]+" y="([\d.-]+)" font-size="'
                        + str(G.EDGE_FONT_SIZE), svg13)]
    top13 = min(ys13) - G.EDGE_FONT_SIZE
    check(f"{len(G._text_lines(label13))} dòng: không tràn trên",
          top13 >= 0, f"đỉnh chữ y={top13:.1f}")
    check(f"{len(G._text_lines(label13))} dòng: không tràn dưới",
          max(ys13) <= h13, f"đáy y={max(ys13):.1f} / khung {h13}")

print("\n15. Nhãn KHÔNG đè lên dây — có margin thật")
for label15 in ("giao hàng", "trả chậm 30 ngày",
                "thanh toán trong vòng 45 ngày kể từ ngày nghiệm thu"):
    svg15 = G.render_svg(_parse(f"flowchart LR\n  A[Đầu vào] -->|{label15}| B[Sản xuất]"))
    wire = float(re.findall(r"[ML][\d.-]+ ([\d.-]+)",
                            re.search(r'<path d="([^"]+)"', svg15).group(1))[0])
    ys15 = [float(m.group(1)) for m in re.finditer(
        r'<text x="[\d.]+" y="([\d.-]+)" font-size="' + str(G.EDGE_FONT_SIZE), svg15)]
    bottom = max(ys15) + G.EDGE_FONT_SIZE * G.DESCENDER_RATIO
    n = len(ys15)
    check(f"{n} dòng: đáy chữ cách dây ≥ {G.EDGE_LABEL_MARGIN}px",
          wire - bottom >= G.EDGE_LABEL_MARGIN - 0.1, f"{wire - bottom:.1f}px")
    check(f"{n} dòng: mọi dòng nằm TRÊN dây", max(ys15) < wire,
          f"đáy baseline {max(ys15):.1f} < dây {wire:.1f}")

print("\n16. Cùng tầng thì cùng bề rộng — dây thẳng cả hai mép")
svg16 = G.render_svg(_parse(
    "flowchart LR\n  A[Công ty TNHH Thương mại Toàn Cầu] --> M[Nhà máy]\n"
    "  B[Gỗ An Cường] --> M\n  M --> C[Khách hàng lớn 78%]\n  M --> D[Khách hàng khác]"))
import collections as _c  # noqa: E402
rects16 = [(float(x), float(y), float(w), float(h)) for x, y, w, h in
           re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"',
                      svg16)][1::2]
ranks16 = _c.defaultdict(list)
for x, y, w, h in rects16:
    ranks16[round(x)].append((w, h))
check("mỗi tầng đúng một bề rộng",
      all(len({w for w, _ in v}) == 1 for v in ranks16.values()),
      str({k: sorted({w for w, _ in v}) for k, v in ranks16.items()}))
rights = {round(x + w) for x, _, w, _ in rects16 if round(x) == min(ranks16)}
check("tầng đầu: mép PHẢI thẳng hàng", len(rights) == 1, str(sorted(rights)))

print("\n17. Chiều cao khối theo số dòng thật, và căn giữa theo trục kia")
heights16 = {h for _, _, _, h in rects16}
check("khối 1 dòng và 2 dòng cao khác nhau", len(heights16) > 1, str(sorted(heights16)))
first = [(y, h) for x, y, _, h in rects16 if round(x) == min(ranks16)]
centres = {round(y + h / 2) for y, h in first}
check("nhưng vẫn có khối cao khác nhau trong cùng tầng", len({h for _, h in first}) > 1)

print("\n18. Ngắt dòng theo bề rộng đo được, không đếm ký tự")
limit18 = G.NODE_MAX_WIDTH - 2 * G.NODE_PADDING_X
for probe in ("Công ty Cổ phần Đầu tư và Phát triển Công nghệ Cao Việt Nam",
              "WWWWWWWWWWWWWWWWWWWWW", "iiiiiiiiiiiiiiiiiiiiiiiiiiiiii"):
    over = [l for l in G._text_lines(probe) if G.text_width(l, G.FONT_SIZE) > limit18 + 0.5]
    check(f"{len(G._text_lines(probe))} dòng, không dòng nào tràn hộp", not over, str(over))

print("\n19. Cỡ chữ nhãn: nhỏ hơn chữ trong khối, vẫn đọc được trên giấy")
check("nhãn nhỏ hơn chữ trong khối", G.EDGE_FONT_SIZE < G.FONT_SIZE,
      f"{G.EDGE_FONT_SIZE} < {G.FONT_SIZE}")
check("in ra giấy ≥ 8pt", G.EDGE_FONT_SIZE * G.PX_TO_PT >= 8.0,
      f"{G.EDGE_FONT_SIZE * G.PX_TO_PT:.2f}pt")
check("trên ngưỡng đọc được", G.EDGE_FONT_SIZE >= G.MIN_READABLE_FONT,
      f"{G.EDGE_FONT_SIZE} >= {G.MIN_READABLE_FONT}")

print("\n19c. Cảnh báo tập trung: hộp VÀ nhãn cạnh của nó, đậm và đổi màu")
# Through mermaid_to_html, not render_svg(_parse(...)): the warning style is
# applied by _auto_color_concentration in between, so the shorter path renders a
# chart nothing has flagged and the assertions below would pass on any code.
warn_svg = mermaid_to_html(
    "```mermaid\nflowchart LR\n  NM[Nhà máy] -->|52%| KH1[Nhà Xinh]\n"
    "  NM -->|18%| KH2[Hoà Phát]\n  NM -->|8%| KH3[Khác]\n```")
spans = re.findall(r"<text ([^>]*)>([^<]*)</text>", warn_svg)
styled = {t: (re.search(r'fill="([^"]+)"', a).group(1), "bold" in a) for a, t in spans}
WARN = "#c1616f"
check("nhãn 52% đổi sang màu cảnh báo", styled["52%"][0] == WARN, styled["52%"][0])
check("nhãn 52% in đậm", styled["52%"][1])
check("hộp Nhà Xinh in đậm", styled["Nhà Xinh"][1])
check("hộp Nhà Xinh giữ màu cảnh báo", styled["Nhà Xinh"][0] == WARN, styled["Nhà Xinh"][0])
for quiet in ("18%", "8%", "Hoà Phát", "Khác", "Nhà máy"):
    check(f"{quiet!r} không bị kéo theo",
          styled[quiet][0] != WARN and not styled[quiet][1], str(styled[quiet]))

print("\n19e. Nhãn cạnh không bao giờ bị hộp nuốt")
# Boxes are painted after the edges, so a label that runs back over one does not
# overlap it — it vanishes. That is what happened in a real report: the gap
# between two ranks was sized as though the label sat in the middle of it, but a
# fan's label sits on one side of the shared trunk and so has only half the gap.
REAL = ("flowchart LR\n"
        "  S1[CÔNG TY CỔ PHẦN TÔN ĐÔNG Á] -->|6,03%<br/>3,65 tỷ| NM[CƠ KHÍ QUY NHƠN]\n"
        "  S2[CÔNG TY TNHH 2TV THƯƠNG MẠI DỊCH VỤ PHÚ THỊNH] -->|5,75%<br/>4,33 tỷ| NM")
WIDE = ("flowchart LR\n"
        "  S1[Nhà cung cấp A] -->|16,03% doanh thu<br/>3.650 triệu đồng| NM[Nhà máy]\n"
        "  S2[Nhà cung cấp B] -->|5,75%<br/>4,33 tỷ| NM")
OUTWARD = ("flowchart LR\n  NM[Nhà máy] -->|8%<br/>2,48 tỷ| O1[Sắt thép, tôn, inox]\n"
           "  NM -->|6%<br/>1,85 tỷ| O2[Sắt thép, tôn, inox]")
CHAIN = "flowchart LR\n  A[Đầu vào] -->|trả chậm 30 ngày| B[Sản xuất] -->|45 ngày| C[Tồn kho]"


def label_clearance(mermaid_src):
    """Smallest horizontal gap between an edge label and any box, in px.

    Negative means the label is inside a box's footprint. Measured on the SVG
    the renderer emits, not on the gap constants, because the constants are what
    were wrong.
    """

    svg = G.render_svg(_parse(mermaid_src))
    boxes = [(float(x), float(x) + float(w)) for x, w in re.findall(
        r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)"[^>]*rx="3" '
        r'fill="#[0-9a-fA-F]{3,6}" stroke=', svg)]
    worst = float("inf")
    for x, text in re.findall(
        r'<text x="([\d.]+)" y="[\d.-]+" font-size="' + str(G.EDGE_FONT_SIZE)
        + r'"[^>]*>([^<]*)<', svg
    ):
        half = G.text_width(text, G.EDGE_FONT_SIZE) / 2
        left, right = float(x) - half, float(x) + half
        for bx, br in boxes:
            # Distance from the label to the box, positive when they are apart.
            # Written the other way round the first time — br - left for a box
            # sitting to the LEFT — which reported every diagram as broken,
            # including ones a separate measurement had just shown to be clean.
            worst = min(worst, left - br if br <= left else
                        bx - right if bx >= right else
                        -min(br - left, right - bx))
    return worst


for case_name, case_src in (("hồ sơ thật", REAL), ("nhãn rộng hơn", WIDE),
                            ("quạt ra", OUTWARD), ("chuỗi thẳng", CHAIN)):
    gap = label_clearance(case_src)
    check(f"{case_name}: nhãn nằm ngoài mọi hộp", gap > 0, f"hở {gap:.1f}px")

print("\n19f. Nhãn tỷ trọng chỉ còn phần trăm")
for raw, want in (
    ("6,03%<br/>3,65 tỷ", "6,03%"),
    ("16,03% doanh thu<br/>3.650 triệu đồng", "16,03% doanh thu"),
    ("5,75%<br/>2,48 tỉ VNĐ", "5,75%"),
    ("12%<br/>1.850.000 đồng", "12%"),
    ("6,03% (3,65 tỷ)", "6,03%"),
    # Nothing to do with a share of revenue: these must survive untouched.
    ("45 ngày", "45 ngày"),
    ("trả chậm 30 ngày", "trả chậm 30 ngày"),
    ("3,65 tỷ", "3,65 tỷ"),
    ("6,03%<br/>tồn kho 45 ngày", "6,03%<br/>tồn kho 45 ngày"),
    ("2,48 tỷ<br/>1,85 tỷ", "2,48 tỷ<br/>1,85 tỷ"),
    ("6,03%<br/>45", "6,03%<br/>45"),
):
    got = _percent_only_edge_labels(
        _parse(f"flowchart LR\n  A[X] -->|{raw}| B[Y]")).edges[0][1]
    check(f"{raw!r} -> {want!r}", got == want, repr(got))

print("\n19d. Style nhà thắng style của model ở đúng ca cảnh báo")
_HEX = r"#[0-9a-fA-F]{3,6}"


def box_fills(mermaid_src):
    """{nhãn hộp: (nền, viền)} đọc theo thứ tự tài liệu của SVG.

    Chấp cả #333 lẫn #333333: model viết mã 3 ký tự, và một regex đòi đủ 6 đã
    lặng lẽ bỏ qua đúng những hộp cần kiểm — chúng biến mất khỏi kết quả trông
    y như thể chúng không được tô.
    """

    svg = mermaid_to_html(mermaid_src)
    out, pending = {}, None
    for m in re.finditer(
        rf'<rect [^>]*fill="({_HEX})" stroke="({_HEX})"[^>]*/>'
        r'|<text [^>]*font-size="13\.5"[^>]*>([^<]*)</text>', svg
    ):
        if m.group(1):
            pending = (m.group(1), m.group(2))
        elif pending:
            out[m.group(3)] = pending
            pending = None
    return out


WARN_FILL, WARN_STROKE, MODEL_AMBER = "#f7e2e5", "#c1616f", "#FFC000"
CLASSDEF = "  classDef warn fill:#FFC000,stroke:#333\n"

# The two diagrams from one real specialist run: the model marked the 55%
# customer itself and forgot the 68% supplier, so the report carried two
# different colours for one finding.
hit = box_fills("```mermaid\nflowchart LR\n" + CLASSDEF
                + "  KH[Nhà máy] -->|55%| R1[Đối tác A]:::warn\n"
                  "  KH -->|25%| R2[Đối tác B]\n  KH -->|20%| R3[Đối tác C]\n```")
miss = box_fills("```mermaid\nflowchart LR\n" + CLASSDEF
                 + "  V1[NCC A] -->|68%| KH[Nhà máy]\n  V2[NCC B] -->|32%| KH\n```")
check("hộp model tự tô :::warn đổi sang style nhà",
      hit["Đối tác A"] == (WARN_FILL, WARN_STROKE), str(hit["Đối tác A"]))
check("hộp model bỏ sót cũng ra đúng style ấy",
      miss["NCC A"] == (WARN_FILL, WARN_STROKE), str(miss["NCC A"]))
check("hai sơ đồ cùng một màu cảnh báo",
      hit["Đối tác A"] == miss["NCC A"])
check("không còn màu hổ phách của model",
      MODEL_AMBER not in {f for f, _ in list(hit.values()) + list(miss.values())})
check("trục hub không bị tô dù dây 68% chạm ngưỡng",
      miss["Nhà máy"][0] != WARN_FILL, miss["Nhà máy"][0])

# Below the threshold this function has nothing to say, and it says nothing.
low = box_fills("```mermaid\nflowchart LR\n" + CLASSDEF
                + "  KH[Nhà máy] -->|30%| R1[Đối tác A]:::warn\n"
                  "  KH -->|25%| R2[Đối tác B]\n  KH -->|20%| R3[Đối tác C]\n```")
check("dưới ngưỡng thì giữ nguyên style của model",
      low["Đối tác A"] == (MODEL_AMBER, "#333"), str(low["Đối tác A"]))
other = box_fills("```mermaid\nflowchart LR\n  classDef hub fill:#BDD7EE,stroke:#333\n"
                  "  KH[Nhà máy]:::hub -->|30%| R1[Đối tác A]\n"
                  "  KH -->|25%| R2[Đối tác B]\n  KH -->|20%| R3[Đối tác C]\n```")
check("style model không phải cảnh báo thì không đụng tới",
      other["Nhà máy"] == ("#BDD7EE", "#333"), str(other["Nhà máy"]))

print("\n19b. Nhãn căn giữa và cách đều — cả quạt ra lẫn quạt vào")
for fan_name, fan_src in (
    ("quạt ra", "flowchart LR\n  NM[Nhà máy] -->|52%| A[Nhà Xinh]\n"
                "  NM -->|18%| B[Hoà Phát]\n  NM -->|8%| C[Khác]"),
    ("quạt vào", "flowchart LR\n  A[Gỗ An Cường] -->|30%| NM[Nhà máy]\n"
                 "  B[Thép Hoà Phát] -->|25%| NM\n  C[Sơn Nippon] -->|5%| NM"),
):
    fan = G.render_svg(_parse(fan_src))
    rows = re.findall(
        r'<text x="([\d.]+)" y="([\d.-]+)" font-size="' + str(G.EDGE_FONT_SIZE)
        + r'"[^>]*text-anchor="(\w+)"', fan)
    check(f"{fan_name}: mọi nhãn anchor=middle",
          {a for _, _, a in rows} == {"middle"}, str({a for _, _, a in rows}))
    check(f"{fan_name}: cùng một x",
          len({round(float(x)) for x, _, _ in rows}) == 1,
          str(sorted({round(float(x)) for x, _, _ in rows})))
    fys = sorted(float(y) for _, y, _ in rows)
    fgaps = {round(fys[i + 1] - fys[i], 1) for i in range(len(fys) - 1)}
    check(f"{fan_name}: khoảng cách đều", len(fgaps) == 1, str(sorted(fgaps)))
    # Centring the labels put them on the shared vertical trunk, which is exactly
    # where the midpoint of a doglegged connector falls. Nothing here caught that
    # — the labels were centred, level and evenly spaced, and unreadable.
    trunk = min(round(float(m))
                for m in re.findall(r"L([\d.]+) [\d.]+ L\1 ", fan))
    texts = re.findall(
        r'<text x="([\d.]+)"[^>]*font-size="' + str(G.EDGE_FONT_SIZE)
        + r'"[^>]*>([^<]*)<', fan)
    clear = min(abs(float(x) - trunk) - G.text_width(t, G.EDGE_FONT_SIZE) / 2
                for x, t in texts)
    check(f"{fan_name}: nhãn không đè trục dọc", clear > 0, f"cách {clear:.1f}px")

print("\n20. Highlight tập trung: hồng, và KHÔNG trùng màu tầng nào")
from src.utils.diagrams import _auto_color_concentration, _AUTO_WARN_STYLE  # noqa: E402
flagged = _auto_color_concentration(_parse(
    "flowchart LR\n  NM[Nhà máy] -->|52%| KH1[Nhà Xinh]\n  NM -->|18%| KH2[Hoà Phát]"))
check("khối vượt ngưỡng được gán style", "KH1" in flagged.node_style)
check("khối dưới ngưỡng thì không", "KH2" not in flagged.node_style)
warn_pair = (_AUTO_WARN_STYLE["fill"], _AUTO_WARN_STYLE["stroke"])
check("màu highlight KHÔNG nằm trong bảng màu tầng",
      warn_pair not in G.LEVEL_COLOURS, str(warn_pair))
check("chữ trong khối cũng đổi màu",
      _AUTO_WARN_STYLE.get("color") == _AUTO_WARN_STYLE["stroke"],
      str(_AUTO_WARN_STYLE.get("color")))
svg20 = G.render_svg(flagged)
check("SVG vẽ chữ bằng màu đó",
      f'fill="{_AUTO_WARN_STYLE["color"]}"' in
      "".join(re.findall(r"<text[^>]*>", svg20)))
check("5 màu tầng vẫn khác nhau đôi một", len(set(G.LEVEL_COLOURS)) == 5)

print("\n21. Guidance và bố cục có sơ đồ khớp với bộ vẽ")
# The retail templates carry one diagram: the monthly cash-flow chart in the
# income analysis. Colour is the renderer's job (_colour_by_level skips any node
# the model styled), and an indented fence is not a fence — markdown reads four
# spaces as a code block and mermaid_to_html never sees the language tag.
for name in ("income-analysis-guidance", "income-analysis-structure"):
    body = (REPO / f"src/templates/{name}.md").read_text(encoding="utf-8")
    check(f"{name}: không dạy model viết classDef",
          not re.search(r"classDef\s+\w+\s+fill:", body))
    check(f"{name}: không có hàng rào ```mermaid bị thụt",
          not re.search(r"^[ \t]+```mermaid", body, re.M))

print("\n" + "=" * 66)
if fails:
    print("❌ HỎNG:", *fails, sep="\n   - ")
    sys.exit(1)
print("✅ Flowchart bố trí vào bề rộng trang, chữ đúng cỡ hằng số")
