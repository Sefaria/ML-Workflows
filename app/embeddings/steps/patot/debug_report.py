from pathlib import Path

from bidi.algorithm import get_display

from .chunker import PatotChunkResult


TITLE_FONT = "Helvetica-Bold"
UI_FONT = "Helvetica"
BODY_FONT = "DebugReportBody"
BODY_FONT_PATHS = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansHebrew-Regular.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
]


def ensure_report_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if BODY_FONT in pdfmetrics.getRegisteredFontNames():
        return

    for path in BODY_FONT_PATHS:
        if path.exists():
            pdfmetrics.registerFont(TTFont(BODY_FONT, str(path)))
            return

    raise RuntimeError(
        "Could not find a Unicode font for PDF output. "
        f"Tried: {', '.join(str(path) for path in BODY_FONT_PATHS)}"
    )


def contains_hebrew(text: str) -> bool:
    return any("\u0590" <= char <= "\u05FF" for char in text)


def to_pdf_display_text(text: str) -> str:
    if contains_hebrew(text):
        return get_display(text)
    return text


def wrap_text_for_pdf(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    from reportlab.pdfbase import pdfmetrics

    if not text:
        return [""]

    lines = []
    for paragraph in text.splitlines() or [text]:
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split(" ")
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            if pdfmetrics.stringWidth(to_pdf_display_text(candidate), font_name, font_size) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_pdf_line(pdf, text: str, x: float, y: float, max_width: float) -> None:
    display_text = to_pdf_display_text(text)
    if contains_hebrew(text):
        pdf.drawRightString(x + max_width, y, display_text)
    else:
        pdf.drawString(x, y, display_text)


def start_page(pdf, title: str, page_width: float, page_height: float) -> float:
    pdf.setFont(TITLE_FONT, 16)
    pdf.drawString(40, page_height - 40, title)
    pdf.setFont(BODY_FONT, 10)
    return page_height - 65


def add_summary_page(pdf, result: PatotChunkResult, tref: str, lang: str, config, page_width: float, page_height: float) -> None:
    y = start_page(pdf, "Patot Segment Chunker Report", page_width, page_height)
    usable_width = page_width - 80
    summary_lines = [
        f"tref={tref}",
        f"lang={lang}",
        f"detected_lang={result.debug_trace.detected_lang}",
        f"input_segment_count={result.input_segment_count}",
        f"final_chunk_count={result.final_chunk_count}",
        f"min_split_tokens={config.min_split_tokens}",
        f"max_split_tokens={config.max_split_tokens}",
        "",
        "The annotated table shows which segment/unit ended up in which final chunk.",
        "Gray lines mark original segment boundaries.",
        "Blue lines mark final chunk boundaries.",
    ]

    for line in summary_lines:
        for wrapped in wrap_text_for_pdf(line, BODY_FONT, 10, usable_width):
            draw_pdf_line(pdf, wrapped, 40, y, usable_width)
            y -= 12
    pdf.showPage()


def build_table_rows(result: PatotChunkResult) -> list[dict]:
    prepared_by_ref = {segment.tref: segment for segment in result.debug_trace.prepared_segments}
    segment_number_by_ref = {segment.tref: i for i, segment in enumerate(result.debug_trace.prepared_segments, start=1)}
    pass2_by_ref = {segment.tref: segment for segment in result.debug_trace.pass2_segments}
    pass3_by_ref = {}
    for adjustment in result.debug_trace.pass3_adjustments:
        if len(adjustment.source_segment_refs) != 1:
            continue
        tref = adjustment.source_segment_refs[0]
        pass3_by_ref.setdefault(tref, []).append(adjustment)

    segment_states = {}
    for tref, pass2_segment in pass2_by_ref.items():
        units = pass2_segment.fallback_splits or pass2_segment.initial_splits or [prepared_by_ref[tref].processed_text]
        segment_states[tref] = {
            "units": units,
            "unit_cursor": 0,
            "chunk_cursor": 0,
            "split_counts": [chunk.split_count for chunk in pass2_segment.final_chunks],
            "returned_single_segment": pass2_segment.returned_single_segment,
            "pass3_queue": list(pass3_by_ref.get(tref, [])),
            "active_pass3": None,
        }

    last_final_chunk_index_by_ref = {}
    for final_chunk_index, chunk in enumerate(result.chunks, start=1):
        if len(chunk.source_segment_refs) == 1:
            last_final_chunk_index_by_ref[chunk.source_segment_refs[0]] = final_chunk_index

    rows = []
    for final_chunk_index, chunk in enumerate(result.chunks, start=1):
        if len(chunk.source_segment_refs) > 1:
            for local_index, tref in enumerate(chunk.source_segment_refs, start=1):
                rows.append(
                    {
                        "segment_number": segment_number_by_ref[tref],
                        "unit_number": "-",
                        "chunk_number": final_chunk_index,
                        "text": prepared_by_ref[tref].processed_text,
                        "segment_boundary_after": True,
                        "final_chunk_boundary_after": local_index == len(chunk.source_segment_refs),
                    }
                )
            continue

        tref = chunk.source_segment_refs[0]
        segment_number = segment_number_by_ref[tref]
        state = segment_states.get(tref)

        if chunk.pass_number == 3 and chunk.kind == "hard_max_split" and state:
            active_pass3 = state["active_pass3"]
            if active_pass3 is None:
                if not state["pass3_queue"]:
                    raise RuntimeError(f"Missing pass-3 adjustment metadata for {tref}")
                active_pass3 = {"adjustment": state["pass3_queue"].pop(0), "produced_index": 0}
                state["active_pass3"] = active_pass3

            adjustment = active_pass3["adjustment"]
            produced_index = active_pass3["produced_index"]
            rows.append(
                {
                    "segment_number": segment_number,
                    "unit_number": "P3",
                    "chunk_number": final_chunk_index,
                    "text": chunk.text,
                    "segment_boundary_after": final_chunk_index == last_final_chunk_index_by_ref.get(tref),
                    "final_chunk_boundary_after": True,
                }
            )
            active_pass3["produced_index"] += 1
            if active_pass3["produced_index"] == len(adjustment.produced_chunks):
                if state["chunk_cursor"] < len(state["split_counts"]):
                    split_count = state["split_counts"][state["chunk_cursor"]]
                    state["unit_cursor"] = min(state["unit_cursor"] + split_count, len(state["units"]))
                    state["chunk_cursor"] += 1
                elif state["returned_single_segment"]:
                    state["chunk_cursor"] += 1
                state["active_pass3"] = None
            continue

        if chunk.kind == "intra_segment" and state and state["chunk_cursor"] < len(state["split_counts"]):
            split_count = state["split_counts"][state["chunk_cursor"]]
            start = state["unit_cursor"]
            end = min(start + split_count, len(state["units"]))
            chunk_units = state["units"][start:end]
            for local_index, unit_text in enumerate(chunk_units, start=1):
                absolute_unit_number = start + local_index
                rows.append(
                    {
                        "segment_number": segment_number,
                        "unit_number": absolute_unit_number,
                        "chunk_number": final_chunk_index,
                        "text": unit_text,
                        "segment_boundary_after": absolute_unit_number == len(state["units"]),
                        "final_chunk_boundary_after": local_index == len(chunk_units),
                    }
                )
            state["unit_cursor"] = end
            state["chunk_cursor"] += 1
            continue

        rows.append(
            {
                "segment_number": segment_number,
                "unit_number": "P3" if chunk.pass_number == 3 else "-",
                "chunk_number": final_chunk_index,
                "text": chunk.text,
                "segment_boundary_after": final_chunk_index == last_final_chunk_index_by_ref.get(tref),
                "final_chunk_boundary_after": True,
            }
        )

    return rows


def add_annotated_table_page(pdf, rows: list[dict], page_width: float, page_height: float) -> None:
    from reportlab.lib import colors

    left_margin = 40
    right_margin = 40
    bottom_y = 45
    leading = 12
    row_padding = 4
    segment_x = left_margin
    unit_x = left_margin + 40
    chunk_x = left_margin + 78
    text_x = left_margin + 120
    text_width = page_width - text_x - right_margin
    row_colors = [colors.HexColor("#f7f9fc"), colors.HexColor("#eef5fb")]

    def start_table_page() -> float:
        y = start_page(pdf, "Annotated Structure", page_width, page_height)
        pdf.setFont(TITLE_FONT, 9)
        pdf.drawString(segment_x, y, "Seg")
        pdf.drawString(unit_x, y, "Unit")
        pdf.drawString(chunk_x, y, "Chunk")
        pdf.drawString(text_x, y, "Text")
        return y - 12

    y = start_table_page()
    for row_index, row in enumerate(rows, start=1):
        wrapped_text = wrap_text_for_pdf(row["text"], BODY_FONT, 10, text_width)
        row_height = row_padding * 2 + leading * len(wrapped_text)
        boundary_extra = 8 if row["segment_boundary_after"] or row["final_chunk_boundary_after"] else 0

        if y - row_height - boundary_extra < bottom_y:
            pdf.showPage()
            y = start_table_page()

        row_top = y
        row_bottom = y - row_height
        pdf.setFillColor(row_colors[(row_index - 1) % len(row_colors)])
        pdf.rect(left_margin - 4, row_bottom, page_width - left_margin - right_margin + 8, row_height, fill=1, stroke=0)
        pdf.setFillColor(colors.black)

        pdf.setFont(UI_FONT, 9)
        pdf.drawString(segment_x, row_top - 11, f"S{row['segment_number']}")
        pdf.drawString(unit_x, row_top - 11, str(row["unit_number"]))
        pdf.drawString(chunk_x, row_top - 11, f"C{row['chunk_number']}")

        pdf.setFont(BODY_FONT, 10)
        text_y = row_top - 11
        for wrapped_line in wrapped_text:
            draw_pdf_line(pdf, wrapped_line, text_x, text_y, text_width)
            text_y -= leading

        y = row_bottom - 4

        if row["segment_boundary_after"]:
            pdf.setStrokeColor(colors.HexColor("#888888"))
            pdf.setLineWidth(1)
            pdf.line(left_margin - 4, y, page_width - right_margin + 4, y)
            y -= 3

        if row["final_chunk_boundary_after"]:
            pdf.setStrokeColor(colors.HexColor("#1f5aa6"))
            pdf.setLineWidth(2.2)
            pdf.line(left_margin - 4, y, page_width - right_margin + 4, y)
            y -= 5

        pdf.setStrokeColor(colors.black)
        pdf.setLineWidth(1)

    pdf.showPage()


def add_final_chunks_page(pdf, result: PatotChunkResult, page_width: float, page_height: float) -> None:
    y = start_page(pdf, "Final Chunks", page_width, page_height)
    usable_width = page_width - 80

    for i, chunk in enumerate(result.chunks, start=1):
        lines = [
            f"{i}. kind={chunk.kind} pass={chunk.pass_number} refs={chunk.source_segment_refs} token_count={chunk.token_count}",
            chunk.text,
            "",
        ]
        for line in lines:
            wrapped_lines = wrap_text_for_pdf(line, BODY_FONT, 10, usable_width)
            if y - 12 * len(wrapped_lines) < 45:
                pdf.showPage()
                y = start_page(pdf, "Final Chunks", page_width, page_height)
            for wrapped in wrapped_lines:
                draw_pdf_line(pdf, wrapped, 40, y, usable_width)
                y -= 12
    pdf.showPage()


def write_debug_pdf(output_path: Path, result: PatotChunkResult, tref: str, lang: str, config) -> None:
    if not result.debug_trace:
        raise RuntimeError("Debug trace is required for PDF output. Run the chunker with debug=True.")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("Missing dependency: reportlab\nInstall it with:\npython -m pip install reportlab") from exc

    ensure_report_fonts()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    page_width, page_height = A4
    pdf.setTitle("Patot Segment Chunker Report")

    add_summary_page(pdf, result, tref, lang, config, page_width, page_height)
    add_annotated_table_page(pdf, build_table_rows(result), page_width, page_height)
    add_final_chunks_page(pdf, result, page_width, page_height)
    pdf.save()


def write_debug_pdf_bundle(output_path: Path, entries: list[dict]) -> None:
    if not entries:
        raise ValueError("At least one entry is required for PDF bundle output.")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("Missing dependency: reportlab\nInstall it with:\npython -m pip install reportlab") from exc

    ensure_report_fonts()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    page_width, page_height = A4
    pdf.setTitle("Patot Chunking Visualization Bundle")

    for entry in entries:
        result = entry["result"]
        if not result.debug_trace:
            raise RuntimeError("Debug trace is required for PDF output. Run the chunker with debug=True.")
        tref = entry["tref"]
        lang = entry["lang"]
        config = entry["config"]
        add_summary_page(pdf, result, tref, lang, config, page_width, page_height)
        add_annotated_table_page(pdf, build_table_rows(result), page_width, page_height)
        add_final_chunks_page(pdf, result, page_width, page_height)

    pdf.save()
