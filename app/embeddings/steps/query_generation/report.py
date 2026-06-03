import random
from pathlib import Path

from embeddings.steps.patot.debug_report import BODY_FONT, TITLE_FONT, draw_pdf_line, ensure_report_fonts, start_page, wrap_text_for_pdf


def _doc_header_lines(doc: dict, query_count: int) -> list[str]:
    metadata = dict(doc.get("metadata") or {})
    return [
        f"doc_id={doc['doc_id']}",
        f"lang={metadata.get('lang') or 'he'}",
        f"ref={metadata.get('ref') or ''}",
        f"versionTitle={metadata.get('versionTitle') or ''}",
        f"query_count={query_count}",
    ]


def write_query_dataset_pdf(
    output_path: Path,
    documents: list[dict],
    queries_by_doc_id: dict[str, list[dict]],
    sample_count: int,
    sample_seed: int,
) -> list[str]:
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    if not documents:
        raise ValueError("No documents available for query dataset visualization.")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("Missing dependency: reportlab\nInstall it with:\npython -m pip install reportlab") from exc

    ensure_report_fonts()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_size = min(sample_count, len(documents))
    if sample_size == len(documents):
        sampled_documents = list(documents)
    else:
        sampled_documents = random.Random(sample_seed).sample(documents, sample_size)
    selected_doc_ids = [str(doc["doc_id"]) for doc in sampled_documents]

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    page_width, page_height = A4
    pdf.setTitle("Query Dataset Visualization")

    for index, doc in enumerate(sampled_documents, start=1):
        doc_id = str(doc["doc_id"])
        queries = queries_by_doc_id.get(doc_id, [])

        y = start_page(pdf, f"Query Dataset Sample {index}", page_width, page_height)
        usable_width = page_width - 80

        for line in _doc_header_lines(doc, len(queries)):
            for wrapped in wrap_text_for_pdf(line, BODY_FONT, 10, usable_width):
                draw_pdf_line(pdf, wrapped, 40, y, usable_width)
                y -= 12

        y -= 6
        pdf.setFont(TITLE_FONT, 11)
        pdf.drawString(40, y, "Document")
        y -= 16
        pdf.setFont(BODY_FONT, 10)
        for wrapped in wrap_text_for_pdf(str(doc["text"]), BODY_FONT, 10, usable_width):
            if y < 45:
                pdf.showPage()
                y = start_page(pdf, f"Query Dataset Sample {index}", page_width, page_height)
            draw_pdf_line(pdf, wrapped, 40, y, usable_width)
            y -= 12

        y -= 4
        pdf.setFont(TITLE_FONT, 11)
        if y < 60:
            pdf.showPage()
            y = start_page(pdf, f"Query Dataset Sample {index}", page_width, page_height)
        pdf.drawString(40, y, "Generated Queries")
        y -= 16

        pdf.setFont(BODY_FONT, 10)
        if not queries:
            draw_pdf_line(pdf, "No queries found for this document.", 40, y, usable_width)
            y -= 12
        else:
            for query in queries:
                query_lines = [
                    f"- type={query['type']} lang={query['lang']} id={query['query_id']}",
                    str(query["text"]),
                ]
                reason = str(query.get("reason") or "").strip()
                if reason:
                    query_lines.append(f"reason={reason}")
                query_lines.append("")
                for line in query_lines:
                    wrapped_lines = wrap_text_for_pdf(line, BODY_FONT, 10, usable_width)
                    if y - 12 * len(wrapped_lines) < 45:
                        pdf.showPage()
                        y = start_page(pdf, f"Query Dataset Sample {index}", page_width, page_height)
                    for wrapped in wrapped_lines:
                        draw_pdf_line(pdf, wrapped, 40, y, usable_width)
                        y -= 12

        pdf.showPage()

    pdf.save()
    return selected_doc_ids
