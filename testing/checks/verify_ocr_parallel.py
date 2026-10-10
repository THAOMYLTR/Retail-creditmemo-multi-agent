"""Verify batched parallel OCR produces identical text and fixes the rotation bug."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import importlib
import os
import re
import tempfile
import time

os.chdir(REPO)

# A private cache dir, so the timings below measure OCR rather than whatever the
# developer's own cache happens to hold.
CACHE = tempfile.mkdtemp(prefix="ocr-check-")
os.environ["OCR_CACHE_DIR"] = CACHE

failures = []
# testing/samples is gitignored (customer files), so on a fresh machine this
# reports missing data and run_checks.py counts it as SKIPPED, not failed.
# The page-level expectations below were calibrated on the scanned statement
# this OCR stage was first tuned on in the SME project; point it at that file
# (or adapt section 3b to a scan of your own) to run it.
SHORT = str(REPO / "testing/samples/case_1/BCTC_VVS_2025_short.pdf")
if not os.path.isfile(SHORT):
    raise FileNotFoundError(SHORT)


def fresh(**env):
    for key in ("OCR_MAX_WORKERS", "OCR_AUTO_ROTATE"):
        os.environ.pop(key, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    import src.utils.ocr as ocr
    importlib.reload(ocr)
    return ocr


def clear_cache():
    for name in os.listdir(CACHE):
        os.remove(os.path.join(CACHE, name))


# --- 3. auto_rotate default ---------------------------------------------------
ocr = fresh()
cfg = ocr._ocr_config()
print(f"3. auto_rotate mặc định : {cfg['auto_rotate']}   (phải là False)")
print(f"   max_workers mặc định : {cfg['max_workers']}")
if cfg["auto_rotate"] is not False:
    failures.append("auto_rotate should default to False")

# --- 1 & 2. identical text, page order ---------------------------------------
clear_cache()
ocr1 = fresh(OCR_MAX_WORKERS=1)
t = time.perf_counter(); text1 = ocr1.ocr_pdf(SHORT); seq = time.perf_counter() - t

clear_cache()
ocr8 = fresh(OCR_MAX_WORKERS=8)
t = time.perf_counter(); text8 = ocr8.ocr_pdf(SHORT); par = time.perf_counter() - t

print()
print("1. Text 1 luồng vs 8 luồng")
print(f"   1 luồng: {seq:5.1f}s, {len(text1)} ký tự")
print(f"   8 luồng: {par:5.1f}s, {len(text8)} ký tự   -> nhanh {seq/par:.1f}x")
print(f"   giống hệt từng ký tự: {text1 == text8}")
if text1 != text8:
    failures.append("parallel text differs from sequential")

pages = [int(n) for n in re.findall(r"--- Page (\d+) ---", text8)]
print()
print(f"2. Thứ tự trang: {pages}")
if pages != list(range(1, len(pages) + 1)):
    failures.append(f"page order broken: {pages}")

# --- 3b. the four previously-corrupted pages now read correctly ---------------
VN = ["tài sản", "nguồn vốn", "cộng", "tổng", "số tiền", "chỉ tiêu", "năm",
      "doanh thu", "nợ", "vốn", "thuyết minh", "báo cáo", "công ty", "ngày"]
chunks = re.split(r"--- Page \d+ ---", text8)[1:]
print()
print("3b. Bốn trang trước đây bị xoay hỏng (4-7)")
for page in (4, 5, 6, 7):
    body = chunks[page - 1].lower()
    score = sum(body.count(w) for w in VN)
    print(f"   trang {page}: điểm từ khoá VN = {score}")
    if score == 0:
        failures.append(f"page {page} still unreadable")

# --- 4. flag still works ------------------------------------------------------
clear_cache()
ocr_rot = fresh(OCR_MAX_WORKERS=8, OCR_AUTO_ROTATE=1)
print()
print(f"4. Bật lại cờ: auto_rotate = {ocr_rot._ocr_config()['auto_rotate']}")
if ocr_rot._ocr_config()["auto_rotate"] is not True:
    failures.append("OCR_AUTO_ROTATE=1 did not re-enable the feature")

# --- 5. memory: batch never exceeds worker count ------------------------------
ocr5 = fresh(OCR_MAX_WORKERS=4)
imgs = ocr5._render_pdf_pages(SHORT, 300, start=0, count=4)
print()
print(f"5. Bộ nhớ: render theo lô trả {len(imgs)} ảnh (yêu cầu 4)")
if len(imgs) != 4:
    failures.append(f"batch render returned {len(imgs)} images, expected 4")
tail = ocr5._render_pdf_pages(SHORT, 300, start=8, count=4)
print(f"   lô cuối (trang 9-11) trả {len(tail)} ảnh — không vượt quá số trang")
if len(tail) != 3:
    failures.append(f"tail batch returned {len(tail)}, expected 3")

# --- 7. cache ------------------------------------------------------------------
ocr7 = fresh(OCR_MAX_WORKERS=8)
clear_cache()
t = time.perf_counter(); first = ocr7.ocr_pdf(SHORT); miss = time.perf_counter() - t
t = time.perf_counter(); again = ocr7.ocr_pdf(SHORT); hit = time.perf_counter() - t
print()
print(f"7. Cache: lần 1 (miss) {miss:.2f}s -> lần 2 (hit) {hit:.3f}s, text khớp: {first == again}")
if hit > 0.5 or first != again:
    failures.append(f"cache hit took {hit:.2f}s / text match={first == again}")

print()
if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
