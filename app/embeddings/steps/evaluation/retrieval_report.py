import random
import re
from pathlib import Path
from urllib.parse import quote

from embeddings.steps.patot.debug_report import (
    BODY_FONT,
    TITLE_FONT,
    UI_FONT,
    draw_pdf_line,
    ensure_report_fonts,
    start_page,
    wrap_text_for_pdf,
)

DOC_TEXT_MAX_LINES = 3
DOC_TEXT_TRUNCATION_SUFFIX = "…"

# Column x-positions (left margin = 40)
_COL_RANK = 40
_COL_SCORE = 64
_COL_MARKER = 112
_COL_REF = 130
_COL_TEXT = 260
_RIGHT_MARGIN = 40
_ROW_PADDING = 3
_LEADING = 12


def _sefaria_url(ref: str) -> str:
    # "Mishnah Berakhot 1:1" → "Mishnah_Berakhot.1.1"
    # "Genesis 1:1" → "Genesis.1.1"
    match = re.match(r"^(.*?)\s+(\d.*)$", ref)
    if match:
        book = match.group(1).replace(" ", "_")
        locator = re.sub(r"[:\s,]", ".", match.group(2))
        url_ref = f"{book}.{locator}"
    else:
        url_ref = ref.replace(" ", "_")
    return f"https://www.sefaria.org/{quote(url_ref, safe='.')}"


def _row_bg_color(rank: int, top_k: int):
    from reportlab.lib.colors import Color

    t = (rank - 1) / max(top_k - 1, 1)
    r = 0.62 + 0.33 * t
    g = 0.91 + 0.08 * t
    b = 0.74 + 0.22 * t
    return Color(r, g, b)


def _truncate_text(text: str, font: str, size: int, max_width: float, max_lines: int) -> list[str]:
    lines = wrap_text_for_pdf(text, font, size, max_width)
    if len(lines) <= max_lines:
        return lines
    truncated = lines[:max_lines]
    last = truncated[-1]
    while last and wrap_text_for_pdf(last + DOC_TEXT_TRUNCATION_SUFFIX, font, size, max_width)[0] != last + DOC_TEXT_TRUNCATION_SUFFIX:
        last = last[:-1]
    truncated[-1] = last + DOC_TEXT_TRUNCATION_SUFFIX
    return truncated


def write_retrieval_visualization_pdf(
    output_path: Path,
    per_query_results: list[dict],
    documents_by_id: dict[str, dict],
    sample_count: int,
    sample_seed: int,
) -> dict:
    if not per_query_results:
        raise ValueError("No query results to visualize.")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")

    try:
        from reportlab.lib.colors import Color, HexColor, black, white
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("Missing dependency: reportlab") from exc

    ensure_report_fonts()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_size = min(sample_count, len(per_query_results))
    sampled = random.Random(sample_seed).sample(per_query_results, sample_size)

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    page_width, page_height = A4
    pdf.setTitle("Embedding Retrieval Visualization")

    for query_index, query_row in enumerate(sampled, start=1):
        query_id = str(query_row.get("query_id", ""))
        query_text = str(query_row.get("query_text", ""))
        query_type = str(query_row.get("query_type") or "")
        query_lang = str(query_row.get("query_lang") or "")
        top_k_results = query_row.get("top_10") or []
        top_k = len(top_k_results)
        text_col_width = page_width - _COL_TEXT - _RIGHT_MARGIN

        y = start_page(pdf, f"Query {query_index} of {sample_size}", page_width, page_height)

        # Query metadata
        pdf.setFont(UI_FONT, 9)
        meta = f"query_id={query_id}  type={query_type}  lang={query_lang}"
        pdf.drawString(_COL_RANK, y, meta)
        y -= _LEADING

        # Query text
        ref_width = page_width - _COL_RANK - _RIGHT_MARGIN
        pdf.setFont(BODY_FONT, 10)
        for line in wrap_text_for_pdf(query_text, BODY_FONT, 10, ref_width):
            if y < 60:
                y = start_page(pdf, f"Query {query_index} of {sample_size} (cont.)", page_width, page_height)
            draw_pdf_line(pdf, line, _COL_RANK, y, ref_width)
            y -= _LEADING

        y -= 6

        # Column header
        if y < 80:
            pdf.showPage()
            y = start_page(pdf, f"Query {query_index} of {sample_size} (cont.)", page_width, page_height)

        pdf.setFont(TITLE_FONT, 9)
        pdf.drawString(_COL_RANK, y, "#")
        pdf.drawString(_COL_SCORE, y, "Score")
        pdf.drawString(_COL_MARKER, y, "✓")
        pdf.drawString(_COL_REF, y, "Ref")
        pdf.drawString(_COL_TEXT, y, "Document text")
        y -= 4
        pdf.setStrokeColor(HexColor("#cccccc"))
        pdf.setLineWidth(0.5)
        pdf.line(_COL_RANK, y, page_width - _RIGHT_MARGIN, y)
        y -= 8

        # Result rows
        for result in top_k_results:
            rank = int(result.get("rank", 0))
            score = float(result.get("score", 0.0))
            is_relevant = bool(result.get("is_relevant", False))
            doc_id = str(result.get("doc_id", ""))
            doc = documents_by_id.get(doc_id, {})
            doc_text = str(doc.get("text", ""))
            ref = str((doc.get("metadata") or {}).get("ref") or doc_id)

            text_lines = _truncate_text(doc_text, BODY_FONT, 10, text_col_width, DOC_TEXT_MAX_LINES)
            row_height = _ROW_PADDING * 2 + _LEADING * max(len(text_lines), 1)

            if y - row_height < 45:
                pdf.showPage()
                y = start_page(pdf, f"Query {query_index} of {sample_size} (cont.)", page_width, page_height)

            row_bottom = y - row_height
            bg = _row_bg_color(rank, top_k)
            pdf.setFillColor(bg)
            pdf.rect(_COL_RANK - 4, row_bottom, page_width - _COL_RANK - _RIGHT_MARGIN + 8, row_height, fill=1, stroke=0)

            text_y = y - _ROW_PADDING - 10

            # Rank
            pdf.setFillColor(black)
            pdf.setFont(UI_FONT, 9)
            pdf.drawString(_COL_RANK, text_y, str(rank))

            # Score
            pdf.drawString(_COL_SCORE, text_y, f"{score:.4f}")

            # Relevance marker
            if is_relevant:
                pdf.setFont(TITLE_FONT, 10)
                pdf.drawString(_COL_MARKER, text_y, "★")
                pdf.setFont(UI_FONT, 9)

            # Ref — rendered as a clickable Sefaria hyperlink
            from reportlab.pdfbase import pdfmetrics
            ref_col_width = _COL_TEXT - _COL_REF - 8
            ref_lines = wrap_text_for_pdf(ref, BODY_FONT, 8, ref_col_width)
            sefaria_url = _sefaria_url(ref)
            pdf.setFont(BODY_FONT, 8)
            pdf.setFillColor(HexColor("#1a5ea8"))
            ref_y = text_y
            for ref_line in ref_lines[:2]:
                draw_pdf_line(pdf, ref_line, _COL_REF, ref_y, ref_col_width)
                line_w = pdfmetrics.stringWidth(ref_line, BODY_FONT, 8)
                pdf.setStrokeColor(HexColor("#1a5ea8"))
                pdf.setLineWidth(0.4)
                pdf.line(_COL_REF, ref_y - 1, _COL_REF + line_w, ref_y - 1)
                pdf.linkURL(sefaria_url, (_COL_REF, ref_y - 2, _COL_REF + line_w, ref_y + 8), relative=0)
                ref_y -= 10
            pdf.setFillColor(black)
            pdf.setStrokeColor(black)

            # Doc text
            pdf.setFont(BODY_FONT, 10)
            line_y = text_y
            for line in text_lines:
                draw_pdf_line(pdf, line, _COL_TEXT, line_y, text_col_width)
                line_y -= _LEADING

            y = row_bottom - 2

        pdf.showPage()

    pdf.save()
    return {
        "sampled_query_count": sample_size,
        "total_query_count": len(per_query_results),
        "output_path": str(output_path),
    }
