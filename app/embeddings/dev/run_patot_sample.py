import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

from app.embeddings.steps.patot.config import ChunkerConfig
from app.embeddings.steps.patot.chunker import PatotChunker
from app.embeddings.steps.patot.json_loader import load_segment_records_from_json_file
from transformers import AutoTokenizer


DEFAULT_INPUT_PATH = Path("app/embeddings/temp_data/sampled_library_sections.json")
DEFAULT_OUTPUT_PATH = Path("app/embeddings/temp_data/output/patot_sample_output.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PATOT chunker against the sample library sections JSON.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=1, help="Number of sections to process.")
    parser.add_argument("--section-ref", help="Process only the section with this exact ref.")
    parser.add_argument("--api-key", help="Gemini API key. Falls back to GEMINI_API_KEY.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose PATOT debug output.")
    return parser.parse_args()


def _load_sections(json_path: Path) -> list[dict]:
    payload = json.loads(json_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a top-level list in {json_path}")
    return payload


def _select_sections(sections: list[dict], section_ref: Optional[str], limit: int) -> list[dict]:
    if section_ref:
        matching_sections = [section for section in sections if section.get("ref") == section_ref]
        if not matching_sections:
            raise ValueError(f"Could not find section_ref={section_ref!r} in sample data")
        return matching_sections[:1]
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    return sections[:limit]


def _build_output_rows(selected_sections: list[dict], all_segment_rows: list[list], chunker: PatotChunker) -> list[dict]:
    output_rows: list[dict] = []
    for section, segment_rows in zip(selected_sections, all_segment_rows):
        result = chunker.chunk_segments(segment_rows)
        output_rows.append(
            {
                "section_ref": section["ref"],
                "url": section.get("url"),
                "version_title": section.get("versionTitle"),
                "language": section.get("language"),
                "input_segment_count": result.input_segment_count,
                "pass1_chunk_count": result.pass1_chunk_count,
                "final_chunk_count": result.final_chunk_count,
                "chunks": [asdict(chunk) for chunk in result.chunks],
            }
        )
    return output_rows


def _token_length(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _verify_chunk_token_bounds(output_rows: list[dict], tokenizer_model: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    max_tokens = tokenizer.model_max_length
    violations = []

    for row in output_rows:
        for i, chunk in enumerate(row["chunks"], start=1):
            token_count = _token_length(tokenizer, chunk["text"])
            chunk["verified_token_count"] = token_count
            if token_count > max_tokens:
                violations.append(
                    {
                        "section_ref": row["section_ref"],
                        "chunk_index": i,
                        "token_count": token_count,
                        "max_tokens": max_tokens,
                        "source_segment_refs": chunk["source_segment_refs"],
                        "text": chunk["text"],
                    }
                )

    if violations:
        lines = [f"BEREL token bound verification failed: {len(violations)} chunk(s) exceed {max_tokens} tokens."]
        for violation in violations[:10]:
            lines.append(
                " ".join(
                    [
                        f"section_ref={violation['section_ref']!r}",
                        f"chunk_index={violation['chunk_index']}",
                        f"token_count={violation['token_count']}",
                        f"max_tokens={violation['max_tokens']}",
                        f"source_segment_refs={violation['source_segment_refs']}",
                    ]
                )
            )
        raise SystemExit("\n".join(lines))


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing Gemini API key. Pass --api-key or set GEMINI_API_KEY or GOOGLE_API_KEY.")

    sections = _load_sections(args.input_path)
    selected_sections = _select_sections(sections, args.section_ref, args.limit)
    section_refs = {section["ref"] for section in selected_sections}

    all_segment_rows = []
    for section, segment_rows in zip(sections, load_segment_records_from_json_file(args.input_path)):
        if section["ref"] in section_refs:
            all_segment_rows.append(segment_rows)

    config = replace(ChunkerConfig(), debug=args.debug)
    chunker = PatotChunker(api_key=api_key, config=config)
    output_rows = _build_output_rows(selected_sections, all_segment_rows, chunker)
    _verify_chunk_token_bounds(output_rows, config.tokenizer_model)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2))

    print(f"Processed {len(output_rows)} section(s)")
    print(f"Wrote output to {args.output_path}")
    for row in output_rows:
        print(
            " ".join(
                [
                    f"section_ref={row['section_ref']}",
                    f"input_segments={row['input_segment_count']}",
                    f"final_chunks={row['final_chunk_count']}",
                ]
            )
        )


if __name__ == "__main__":
    main()
