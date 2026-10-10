"""OCR helpers for PDF documents.

Pipeline per page: rasterize -> ruled-line removal -> image preprocessing
(grayscale, denoise, deskew, Otsu threshold, optional upscale) -> layout-aware
Tesseract OCR -> post-OCR text cleanup. Results are cached on disk keyed by file
content hash + OCR config so re-runs of the notebook do not re-OCR the same
document.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pypdfium2 as pdfium
import pytesseract

# Homebrew's location on Apple Silicon when it exists, otherwise whatever is on
# PATH (/usr/bin/tesseract on Debian/Ubuntu). TESSERACT_CMD overrides both.
DEFAULT_TESSERACT_CMD = (
    "/opt/homebrew/bin/tesseract"
    if os.path.exists("/opt/homebrew/bin/tesseract")
    else shutil.which("tesseract") or "tesseract"
)
# pypdfium2 bundles the pdfium binary, so no system poppler install is required.
PDF_POINTS_PER_INCH = 72.0


def _ocr_config() -> dict[str, object]:
    """Read OCR settings from environment with backward-compatible defaults.

    Image preprocessing is OFF by default: on reasonably rendered PDFs Tesseract's
    own internal thresholding beats external binarization/denoising (which blur
    Vietnamese diacritics). Enable it (and its sub-steps) only for poor scans.
    """
    return {
        "tesseract_cmd": os.getenv("TESSERACT_CMD", DEFAULT_TESSERACT_CMD),
        "dpi": int(os.getenv("OCR_DPI", "300")),
        "lang": os.getenv("OCR_LANG", "vie+eng"),
        "psm": os.getenv("OCR_PSM", "6"),
        "oem": os.getenv("OCR_OEM", "3"),
        "preprocess": os.getenv("OCR_PREPROCESS", "0") != "0",
        "deskew": os.getenv("OCR_DESKEW", "1") != "0",
        "denoise": os.getenv("OCR_DENOISE", "0") != "0",
        "binarize": os.getenv("OCR_BINARIZE", "0") != "0",
        "layout": os.getenv("OCR_LAYOUT", "1") != "0",
        "upscale": float(os.getenv("OCR_UPSCALE", "1.0")),
        "cache_dir": os.getenv("OCR_CACHE_DIR", ""),
        # Ruled-table borders confuse Tesseract's line-height estimate badly
        # enough to merge two table rows into one unreadable blob — see
        # _remove_ruled_lines. Unlike denoise/binarize below, this is not a
        # scan-quality tradeoff: measured on a clean 300dpi render, it took a
        # 12-row balance table from 1 correct row to 12. On independent by
        # design (thin straight rules vs. character strokes), so it stays on
        # even when the rest of ``preprocess`` is off.
        "deline": os.getenv("OCR_DELINE", "1") != "0",
        # OFF by default, and the reason is a measured split rather than a
        # verdict on the detector as a whole. Measured on the scanned sample set
        # this OCR stage was first tuned on (corporate financial statements):
        #
        #   180° verdicts: 4 of 4 wrong. On BCTC_VVS_2025_short.pdf the detector
        #     asked to flip 4 of 11 upright pages, turning "Công ty cổ phần Đầu
        #     tư Phát triển May Việt Nam" into "é fy BA uaẩn8h) { FTN LAAN)".
        #   90° verdicts: 4 of 4 right. On BCTC_VVS_2024.pdf pages 29-31 and 33
        #     are genuinely sideways — 0 Vietnamese keywords upright, 10-25 once
        #     rotated.
        #
        # So leaving this off is a real trade: it stops the 180° damage and
        # costs the 90° repair. Those four pages of BCTC_VVS_2024.pdf reach the
        # agents as noise while it is off. Enable it for a scan set where pages
        # are genuinely sideways and the 180° risk is worth accepting.
        #
        # The earlier claim here — "measured to have no false positives on an
        # already-correct page" — is the one thing now known to be false; it is
        # what kept this on while it was corrupting a third of a statement.
        #
        # Speed is NOT the argument either way: once pages OCR concurrently the
        # extra Tesseract pass costs ~2.2s on a 43-page file (24.0s vs 21.8s).
        "auto_rotate": os.getenv("OCR_AUTO_ROTATE", "0") != "0",
        # OCR is the pipeline's slowest step and Tesseract runs out-of-process,
        # so threads scale it nearly linearly. Bounded rather than unbounded:
        # each worker holds a 300dpi page (~25 MB) plus its own tesseract
        # process.
        "max_workers": max(
            1,
            int(os.getenv("OCR_MAX_WORKERS", "0")) or min(8, os.cpu_count() or 1),
        ),
    }


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def _remove_ruled_lines(gray: np.ndarray, dpi: int = 300) -> np.ndarray:
    """Paint over long straight table rules, leaving the text between them.

    Discovered on a CIC report: a 12-row balance table OCR'd as 1 correct row
    and 11 blobs of garbage. ``image_to_data`` showed why — normal text lines
    were ~30-45px tall, but every row touching a ruling line was boxed at
    ~101-106px, more than the ~66px gap to the next row. Tesseract's line finder
    was fusing the horizontal rule into the row's own text height, so each
    "line" it fed the recognizer was really two rows of digits bled together.

    The fix removes the cause rather than the symptom: find long horizontal and
    vertical runs by morphological opening (a run of foreground pixels wider —
    or taller — than the kernel survives; an isolated character stroke does
    not), then whiten only those pixels. Kernel lengths scale with DPI so the
    same table is caught whether rendered at 150 or 600.
    """
    factor = dpi / 300.0
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, round(60 * factor)), 1)
    )
    # 40px (the original tuning, against an A4-sized CIC report) cut into tall
    # letter strokes on a CIC report printed on a larger page (~11x14in): the
    # same absolute DPI renders larger glyphs on a larger sheet, so "Ngân hàng"
    # came back "Ngân àng" — the vertical kernel was reading a letter's stem as
    # a rule. Measured across both documents at several sizes: the balance
    # table this constant was tuned on stays 12/12 correct rows from 40px all
    # the way to 100px, so raising it here traded nothing away.
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(10, round(60 * factor)))
    )
    lines_mask = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel),
    )
    # Dilated by one pass so the light anti-aliased fringe around each rule
    # (which survives the Otsu cut as faint grey, not solid black) is cleared
    # too — left in place, that fringe is exactly what re-inflates the next
    # line-height estimate.
    lines_mask = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8))
    cleaned = gray.copy()
    cleaned[lines_mask > 0] = 255
    return cleaned


def _detect_rotation_angle(image: Image.Image) -> int:
    """0/90/180/270 — degrees to rotate the page clockwise to correct it.

    Separate from _deskew on purpose: that one straightens a few degrees of
    scan tilt, this one catches a page rotated a full quarter-turn or upside
    down, which minAreaRect cannot see as "tilt" — the whole block of text has
    rotated with the page, so there is no skew angle left to measure.

    Never raises. Measured directly rather than assumed: on a page with too
    little text to judge (a blank or near-blank crop), Tesseract's OSD raises
    rather than guessing — "too few characters, skipping this page" — so that
    failure is read as "leave the page alone", not routed to a fallback guess.
    orientation_conf is deliberately not used as a threshold: measured across a
    clean document and a noisy one, the noisy one scored *lower* confidence
    while still getting all four rotations right, so filtering on confidence
    would have discarded the exact cases most worth correcting.
    """
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        return int(osd["rotate"]) % 360
    except Exception:
        return 0


def _apply_rotation(image: Image.Image, angle: int) -> Image.Image:
    """Rotate a page clockwise by the angle OSD reported is needed to fix it.

    PIL's own ``rotate()`` turns counter-clockwise for a positive angle, while
    OSD's "rotate" is the clockwise correction needed — hence the sign flip.
    """
    if angle % 360 == 0:
        return image
    return image.rotate(-angle, expand=True)


def _deskew(gray: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate the image to correct small scan skew, if any is detected."""
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(
        inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 50:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    # Only correct meaningful, plausible skew; skip near-zero / extreme values.
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return gray

    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(
    image: Image.Image,
    deskew: bool = True,
    denoise: bool = False,
    binarize: bool = False,
    upscale: float = 1.0,
) -> Image.Image:
    """Prepare a scanned page image for OCR (opt-in; steps are individually gated).

    Only the steps that empirically help a given scan should be enabled. Denoising
    and forced binarization are OFF by default because they blur small diacritics on
    already-clean renders; enable them for photographed or low-quality scans.
    """
    array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)
    if deskew:
        gray = _deskew(gray)
    if upscale and upscale > 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_CUBIC,
        )
    if binarize:
        _, gray = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    return Image.fromarray(gray)


# ---------------------------------------------------------------------------
# Layout-aware OCR (keeps table columns separated)
# ---------------------------------------------------------------------------
def image_to_layout_text(
    image: Image.Image,
    lang: str,
    config: str,
    timeout: int | None,
) -> str:
    """OCR a page and rebuild text preserving line and column structure.

    Uses ``image_to_data`` so tokens keep their bounding boxes; tokens are
    grouped into physical lines and separated by whitespace proportional to the
    horizontal gap between them, so financial-statement columns do not collapse.
    Falls back to plain ``image_to_string`` on any failure.
    """
    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            timeout=timeout,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return pytesseract.image_to_string(
            image, lang=lang, config=config, timeout=timeout
        )

    lines: dict[tuple[int, int, int], list[tuple[int, int, str]]] = {}
    char_widths: list[float] = []
    for index, word in enumerate(data["text"]):
        if not word or not word.strip():
            continue
        try:
            conf = float(data["conf"][index])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        left = data["left"][index]
        width = data["width"][index]
        lines.setdefault(key, []).append((left, left + width, word))
        if word:
            char_widths.append(width / max(len(word), 1))

    if not lines:
        return pytesseract.image_to_string(
            image, lang=lang, config=config, timeout=timeout
        )

    approx_char_width = np.median(char_widths) if char_widths else 10.0
    gap_per_space = max(approx_char_width, 1.0)

    rendered_lines = []
    for key in sorted(lines):
        tokens = sorted(lines[key], key=lambda item: item[0])
        parts = []
        previous_right: int | None = None
        for left, right, word in tokens:
            if previous_right is not None:
                gap = left - previous_right
                spaces = int(gap / gap_per_space) if gap > 0 else 0
                parts.append(" " * max(1, min(spaces, 8)))
            parts.append(word)
            previous_right = right
        rendered_lines.append("".join(parts))
    return "\n".join(rendered_lines)


# ---------------------------------------------------------------------------
# Post-OCR text cleanup
# ---------------------------------------------------------------------------
_NUMERIC_RUN = re.compile("(?<=\\d)[ \t\u00a0](?=\\d{3}\\b)")
_MULTISPACE = re.compile("[ \t\u00a0]{2,}")
_DEHYPHENATE = re.compile(r"(\w)-\n(\w)")


def _line_is_noise(line: str) -> bool:
    """Detect OCR garbage lines dominated by punctuation/symbols."""
    stripped = line.strip()
    if not stripped:
        return False
    meaningful = sum(char.isalnum() for char in stripped)
    return meaningful / len(stripped) < 0.25 and len(stripped) >= 4


def _strip_repeated_headers_footers(pages: list[str]) -> list[str]:
    """Remove header/footer lines that repeat across most pages."""
    if len(pages) < 3:
        return pages

    from collections import Counter

    edge_counter: Counter[str] = Counter()
    for page in pages:
        page_lines = [line.strip() for line in page.splitlines() if line.strip()]
        edges = page_lines[:2] + page_lines[-2:]
        for line in edges:
            if len(line) >= 4:
                edge_counter[line] += 1

    threshold = max(3, int(len(pages) * 0.6))
    repeated = {line for line, count in edge_counter.items() if count >= threshold}
    if not repeated:
        return pages

    cleaned = []
    for page in pages:
        kept = [
            line
            for line in page.splitlines()
            if line.strip() not in repeated
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def clean_ocr_text(text: str) -> str:
    """Normalize raw OCR text before it reaches the LLM and ratio parser."""
    if not text:
        return ""

    # Join words split across a line break by hyphenation.
    text = _DEHYPHENATE.sub(r"\1\2", text)

    cleaned_lines = []
    for line in text.splitlines():
        if _line_is_noise(line):
            continue
        # Merge space-split thousands groups ("1 234 567" -> "1234567").
        line = _NUMERIC_RUN.sub("", line)
        line = _MULTISPACE.sub("  ", line.rstrip())
        cleaned_lines.append(line)

    # Collapse runs of blank lines to a single blank line.
    result_lines: list[str] = []
    for line in cleaned_lines:
        if not line.strip() and result_lines and not result_lines[-1].strip():
            continue
        result_lines.append(line)
    return "\n".join(result_lines).strip()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _file_sha256(path: str) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _cache_dir(settings: dict[str, object]) -> Path:
    configured = str(settings.get("cache_dir") or "")
    if configured:
        directory = Path(configured)
    else:
        directory = Path(tempfile.gettempdir()) / "retail_ocr_cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cache_key(pdf_path: str, settings: dict[str, object]) -> str:
    signature = {
        key: settings[key]
        for key in (
            "dpi", "lang", "psm", "oem", "preprocess",
            "deskew", "denoise", "binarize", "layout", "upscale", "deline",
            "auto_rotate",
        )
    }
    payload = f"{_file_sha256(pdf_path)}|{json.dumps(signature, sort_keys=True)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# PDF rasterization (pypdfium2 — no system poppler dependency)
# ---------------------------------------------------------------------------
def _render_pdf_pages(
    pdf_path: str,
    dpi: int,
    start: int = 0,
    count: int | None = None,
) -> list[Image.Image]:
    """Render a slice of PDF pages to PIL images using the bundled pdfium engine.

    Rendering a *slice* rather than the whole document is what keeps memory
    bounded: a 300dpi page is ~25 MB, so the 87-page statement in
    testing/samples would need ~2.1 GB if every page were held at once. Callers
    render one batch, OCR it, and drop it before asking for the next.
    """

    scale = dpi / PDF_POINTS_PER_INCH
    document = pdfium.PdfDocument(pdf_path)
    try:
        stop = len(document) if count is None else min(start + count, len(document))
        images = []
        for index in range(start, stop):
            page = document[index]
            bitmap = page.render(scale=scale)
            images.append(bitmap.to_pil().convert("RGB"))
        return images
    finally:
        document.close()


def _pdf_page_count(pdf_path: str) -> int:
    document = pdfium.PdfDocument(pdf_path)
    try:
        return len(document)
    finally:
        document.close()


def _ocr_page(
    image: Image.Image,
    settings: dict[str, object],
    lang: str,
    tess_config: str,
    timeout: int | None,
    page_number: int,
) -> str:
    """Everything one page needs, from raw render to cleaned text.

    Split out of ocr_pdf so the pages can run concurrently: Tesseract does its
    work in a subprocess, so threads here really do run in parallel rather than
    taking turns on the GIL.
    """

    working = image
    # First of all the fixes, because every step after this assumes the page's
    # horizontal/vertical axes match the reading direction: a ruled-line kernel
    # or a small-skew estimate is meaningless applied to a page still rotated a
    # quarter turn. Off by default — see the auto_rotate note in _ocr_config.
    if settings["auto_rotate"]:
        angle = _detect_rotation_angle(working)
        if angle:
            working = _apply_rotation(working, angle)
            print(
                f"[ocr_pdf] WARNING: trang {page_number} bị xoay {angle}°, "
                "đã tự động chỉnh — kiểm tra file gốc nếu kết quả OCR vẫn "
                "bất thường."
            )
    # Ahead of the opt-in ``preprocess`` bundle and unconditional on it: this
    # targets table rules, not scan noise, so it runs on the untouched render
    # even when nothing else does. Applied before deskew/denoise/binarize —
    # line detection assumes axis-aligned rules, so a page that also needs
    # deskewing should have that run first in a later pass; these renders come
    # straight from pdfium rather than a photographed scan, so skew is not the
    # case this is tuned for.
    if settings["deline"]:
        gray = cv2.cvtColor(np.array(working.convert("RGB")), cv2.COLOR_RGB2GRAY)
        working = Image.fromarray(_remove_ruled_lines(gray, int(settings["dpi"])))
    prepared = (
        preprocess_image(
            working,
            deskew=bool(settings["deskew"]),
            denoise=bool(settings["denoise"]),
            binarize=bool(settings["binarize"]),
            upscale=float(settings["upscale"]),
        )
        if settings["preprocess"]
        else working
    )
    if settings["layout"]:
        text = image_to_layout_text(prepared, lang, tess_config, timeout)
    else:
        text = pytesseract.image_to_string(
            prepared, lang=lang, config=tess_config, timeout=timeout
        )
    return clean_ocr_text(text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ocr_pdf(pdf_path: str, timeout_seconds: float | None = None) -> str:
    """Extract Vietnamese and English text from a PDF via Tesseract OCR."""
    settings = _ocr_config()
    pytesseract.pytesseract.tesseract_cmd = str(settings["tesseract_cmd"])
    timeout = int(timeout_seconds) if timeout_seconds else None

    cache_directory = _cache_dir(settings)
    cache_file = cache_directory / f"{_cache_key(pdf_path, settings)}.txt"
    if cache_file.is_file():
        return cache_file.read_text(encoding="utf-8")

    tess_config = f"--oem {settings['oem']} --psm {settings['psm']}"
    lang = str(settings["lang"])
    workers = int(settings["max_workers"])
    page_count = _pdf_page_count(pdf_path)

    # One batch of pages in flight at a time: render it, OCR the batch across
    # threads, then let the images go before rendering the next. Batching is
    # what makes the concurrency affordable — the whole document rendered up
    # front was ~2.1 GB on the largest statement in testing/samples, while a
    # batch is workers x ~25 MB.
    #
    # executor.map preserves input order, which is load-bearing twice over:
    # _strip_repeated_headers_footers compares page against page, and the
    # "--- Page N ---" markers are what citations resolve a page number from.
    page_texts: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, page_count, workers):
            images = _render_pdf_pages(
                pdf_path,
                int(settings["dpi"]),
                start=start,
                count=workers,
            )
            page_texts.extend(
                executor.map(
                    lambda item: _ocr_page(
                        item[1],
                        settings,
                        lang,
                        tess_config,
                        timeout,
                        item[0],
                    ),
                    enumerate(images, start=start + 1),
                )
            )
            del images

    page_texts = _strip_repeated_headers_footers(page_texts)
    full_text = "".join(
        f"\n--- Page {index + 1} ---\n{text}"
        for index, text in enumerate(page_texts)
    )

    try:
        cache_file.write_text(full_text, encoding="utf-8")
    except OSError:
        pass
    return full_text


# Image files a retail dossier arrives as: phone photos of the ID card, the
# land-use certificate or a payslip are at least as common as scanned PDFs.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})


def ocr_image(image_path: str, timeout_seconds: float | None = None) -> str:
    """OCR a photographed or scanned image file into the same text shape as a PDF.

    A phone photo differs from a pdfium render in the two ways that matter
    here: it carries its orientation in EXIF rather than in the pixels, and it
    is noisier. So the EXIF rotation is applied first, and denoising and
    deskewing run whatever the PDF-tuned settings say. A multi-frame TIFF is
    read frame by frame, one "--- Page N ---" marker per frame, so citations
    resolve to a page exactly as they do for a PDF.
    """

    from PIL import ImageOps, ImageSequence

    settings = dict(_ocr_config())
    settings.update({"preprocess": True, "deskew": True, "denoise": True})
    pytesseract.pytesseract.tesseract_cmd = str(settings["tesseract_cmd"])
    timeout = int(timeout_seconds) if timeout_seconds else None

    cache_directory = _cache_dir(settings)
    cache_file = cache_directory / f"{_cache_key(image_path, settings)}.txt"
    if cache_file.is_file():
        return cache_file.read_text(encoding="utf-8")

    tess_config = f"--oem {settings['oem']} --psm {settings['psm']}"
    lang = str(settings["lang"])
    page_texts: list[str] = []
    with Image.open(image_path) as source:
        for index, frame in enumerate(ImageSequence.Iterator(source), start=1):
            page = ImageOps.exif_transpose(frame.convert("RGB"))
            page_texts.append(
                _ocr_page(page, settings, lang, tess_config, timeout, index)
            )
    full_text = "".join(
        f"\n--- Page {index + 1} ---\n{text}"
        for index, text in enumerate(page_texts)
    )
    try:
        cache_file.write_text(full_text, encoding="utf-8")
    except OSError:
        pass
    return full_text
