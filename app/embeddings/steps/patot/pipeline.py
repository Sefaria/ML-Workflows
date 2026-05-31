import hashlib
import re
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Iterable, Iterator

from .chunker import PatotChunker
from .config import ChunkerConfig
from .json_loader import load_segment_records_from_section


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def base_text_ref_for_source_segment(source_segment_ref: str) -> str:
    return source_segment_ref.split("::fn:", 1)[0]


def relevant_base_text_mappings(section: dict[str, Any], source_segment_refs: list[str]) -> dict[str, Any]:
    base_text_mappings = section.get("baseTextMappings") or {}
    if not isinstance(base_text_mappings, dict):
        return {}

    relevant_mappings = {}
    for source_segment_ref in source_segment_refs:
        base_text_ref = base_text_ref_for_source_segment(source_segment_ref)
        if base_text_ref in base_text_mappings:
            relevant_mappings[base_text_ref] = base_text_mappings[base_text_ref]
    return relevant_mappings


def build_chunked_documents_for_section(
    section: dict,
    api_key: str,
    config: ChunkerConfig,
) -> list[dict]:
    segment_records = load_segment_records_from_section(section)
    if not segment_records:
        return []

    chunker = PatotChunker(api_key=api_key, config=config)
    result = chunker.chunk_segments(segment_records)

    section_ref = str(section["ref"])
    section_lang = str(section.get("language") or "he")
    section_slug = slugify(section_ref)
    output_rows = []

    for chunk_index, chunk in enumerate(result.chunks, start=1):
        chunk_hash = hashlib.sha256(f"{section_ref}|{section_lang}|{chunk_index}|{chunk.text}".encode("utf-8")).hexdigest()[:12]
        metadata = {
            "ref": section_ref,
            "url": section.get("url"),
            "versionTitle": section.get("versionTitle"),
            "lang": section_lang,
            "source_segment_refs": chunk.source_segment_refs,
            "chunk_kind": chunk.kind,
            "chunk_pass_number": chunk.pass_number,
            "chunk_token_count": chunk.token_count,
            "chunk_triggered": chunk.triggered,
            "chunk_score": chunk.score,
        }
        chunk_base_text_mappings = relevant_base_text_mappings(section, chunk.source_segment_refs)
        if chunk_base_text_mappings:
            metadata["baseTextMappings"] = chunk_base_text_mappings
        output_rows.append(
            {
                "doc_id": f"chunk_{section_lang}_{section_slug}_{chunk_index}_{chunk_hash}",
                "text": chunk.text,
                "metadata": metadata,
            }
        )
    if config.runtime_analytics is not None:
        config.runtime_analytics.record_section_processed(len(output_rows))
    return output_rows


def iter_chunked_documents_parallel(
    sections: Iterable[dict],
    api_key: str,
    config: ChunkerConfig,
    max_workers: int,
) -> Iterator[dict]:
    max_pending = max(1, max_workers * 2)
    section_iter = iter(sections)
    futures: dict[Future, None] = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            section = next(section_iter)
        except StopIteration:
            return False
        futures[executor.submit(build_chunked_documents_for_section, section, api_key, config)] = None
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_pending):
            if not submit_next(executor):
                break

        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future, None)
                for row in future.result():
                    yield row
                submit_next(executor)
