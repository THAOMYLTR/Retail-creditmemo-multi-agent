"""Small text matching helpers.

Deliberately free of project imports so the lowest-level modules (the document
matrix loader, classification) can all share ``normalize_text`` without an
import cycle.
"""

from __future__ import annotations

import re
import unicodedata


# Every extension the pipeline will ingest; discovery rejects the rest before a
# file is ever classified. Kept here rather than beside the discovery code
# because two things need it and they sit on opposite sides of the layering:
# classification filters a folder with it, and the source list uses it to strip
# an extension off a filename it is about to print. Stripping by "text after the
# last dot" would eat the year off "SAO_KE_01.2026", a file with no extension
# at all — knowing the real set is what makes that safe.
SUPPORTED_EXTENSIONS = frozenset(
    {
        ".pdf", ".xlsx", ".xls", ".csv", ".txt", ".md", ".pptx", ".xml",
        # Phone photos and scans: ID cards, land-use certificates, payslips.
        ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
    }
)


def contains_any(text: str, keywords: list[str]) -> bool:
    """Return True when any keyword appears in the input text."""
    return any(keyword in text for keyword in keywords)


def normalize_text(text: str) -> str:
    """Lowercase and strip Vietnamese accents for robust keyword matching.

    OCR output frequently drops or mangles diacritics, so matching must be
    accent-insensitive (e.g. "báo cáo" and "bao cao" must match the same key).
    """

    # Vietnamese "đ"/"Đ" do NOT decompose under NFD, so map them explicitly.
    lowered = (text or "").lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    # Collapse separators/punctuation to single spaces so filename tokens
    # ("SAO_KE_2026", "de_nghi_vay_von") match space-delimited keywords.
    return re.sub(r"[^0-9a-z]+", " ", stripped).strip()


def show_graph(graph, xray=False):
    """Display a LangGraph Mermaid diagram with fallback rendering."""
    from IPython.display import Image, Markdown

    drawable_graph = graph.get_graph(xray=xray)

    try:
        return Image(drawable_graph.draw_mermaid_png())
    except Exception as api_error:
        print(
            "Default renderer failed "
            f"({api_error}), falling back to pyppeteer..."
        )

    try:
        # Pyppeteer needs the notebook event loop to allow nested async calls.
        import nest_asyncio
        nest_asyncio.apply()

        from langchain_core.runnables.graph import MermaidDrawMethod

        return Image(
            drawable_graph.draw_mermaid_png(
                draw_method=MermaidDrawMethod.PYPPETEER
            )
        )
    except Exception as local_error:
        print(
            "Local pyppeteer renderer failed "
            f"({local_error}), showing Mermaid source instead."
        )
        mermaid_code = drawable_graph.draw_mermaid()
        return Markdown(f"```mermaid\n{mermaid_code}\n```")
