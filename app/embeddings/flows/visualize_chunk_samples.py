import json
import os
import random
import tempfile
from pathlib import Path

import ijson
from prefect import flow, task

from embeddings.steps.patot.chunker import PatotChunker
from embeddings.steps.patot.config import ChunkerConfig
from embeddings.steps.patot.debug_report import write_debug_pdf_bundle
from embeddings.steps.patot.json_loader import load_segment_records_from_section
from utils.gcs import download_blob, upload_blob


def _iter_sections(local_path: str):
    with open(local_path, "rb") as handle:
        yield from ijson.items(handle, "item")


def _iter_chunked_documents(local_path: str):
    with open(local_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


@task(log_prints=True)
def download_blob_to_tmp(bucket: str, blob_path: str) -> str:
    print(f"Downloading gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


@task(log_prints=True)
def build_visualization_pdf(
    source_local_path: str,
    chunked_local_path: str,
    output_path: str,
    sample_count: int,
    sample_seed: int,
    api_key: str,
    cache_path: str,
) -> dict:
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    section_refs = {
        str(doc.get("metadata", {}).get("ref"))
        for doc in _iter_chunked_documents(chunked_local_path)
        if doc.get("metadata", {}).get("ref")
    }
    if not section_refs:
        raise ValueError("No section refs found in chunked documents input.")

    ordered_refs = sorted(section_refs)
    sample_size = min(sample_count, len(ordered_refs))
    selected_refs = random.Random(sample_seed).sample(ordered_refs, sample_size)
    selected_ref_set = set(selected_refs)
    print(f"Selected {sample_size} section(s) for visualization")
    for tref in selected_refs:
        print(f"  - {tref}")

    config = ChunkerConfig(
        debug=True,
        embedding_cache_enabled=True,
        embedding_cache_path=cache_path,
    )
    chunker = PatotChunker(api_key=api_key, config=config)

    entries = []
    for section in _iter_sections(source_local_path):
        tref = str(section["ref"])
        if tref not in selected_ref_set:
            continue
        segment_records = load_segment_records_from_section(section)
        if not segment_records:
            continue
        result = chunker.chunk_segments(segment_records)
        entries.append(
            {
                "tref": tref,
                "lang": str(section.get("language") or "he"),
                "config": config,
                "result": result,
            }
        )
        if len(entries) == sample_size:
            break

    found_refs = {entry["tref"] for entry in entries}
    missing_refs = [tref for tref in selected_refs if tref not in found_refs]
    if missing_refs:
        raise ValueError(f"Could not find selected section refs in source dataset: {missing_refs}")

    write_debug_pdf_bundle(Path(output_path), entries)
    return {
        "sample_count": sample_size,
        "sample_seed": sample_seed,
        "selected_refs": selected_refs,
        "output_path": output_path,
    }


@task(log_prints=True)
def upload_visualization_pdf(local_path: str, bucket: str, blob_path: str) -> None:
    print(f"Uploading visualization PDF to gs://{bucket}/{blob_path}")
    upload_blob(local_path, bucket, blob_path)


@flow(log_prints=True)
def visualize_chunk_samples_flow(
    source_bucket: str,
    source_blob: str,
    chunked_bucket: str,
    chunked_blob: str,
    dest_bucket: str,
    dest_blob: str,
    sample_count: int = 10,
    sample_seed: int = 0,
    cache_path: str = "/cache/patot/embedding_cache.sqlite",
) -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for chunk visualization.")

    source_local_path = download_blob_to_tmp(source_bucket, source_blob)
    chunked_local_path = download_blob_to_tmp(chunked_bucket, chunked_blob)
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf", dir="/tmp") as tmp:
        output_local_path = tmp.name

    try:
        summary = build_visualization_pdf(
            source_local_path,
            chunked_local_path,
            output_local_path,
            sample_count,
            sample_seed,
            api_key,
            cache_path,
        )
        print(
            f"Built visualization PDF for {summary['sample_count']} sampled section(s) "
            f"using seed={summary['sample_seed']}"
        )
        upload_visualization_pdf(output_local_path, dest_bucket, dest_blob)
    finally:
        Path(source_local_path).unlink(missing_ok=True)
        Path(chunked_local_path).unlink(missing_ok=True)
        Path(output_local_path).unlink(missing_ok=True)
