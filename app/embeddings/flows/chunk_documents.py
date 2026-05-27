import json
import os
import tempfile
from itertools import islice
from pathlib import Path
from typing import Optional

import ijson
from prefect import flow, task

from app.embeddings.steps.patot.config import ChunkerConfig
from app.embeddings.steps.patot.pipeline import iter_chunked_documents_parallel
from utils.gcs import download_blob, upload_blob


@task(log_prints=True)
def download_source(bucket: str, blob_path: str) -> str:
    print(f"Downloading source dataset from gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


def _iter_sections(local_path: str):
    with open(local_path, "rb") as handle:
        yield from ijson.items(handle, "item")


def _iter_limited_sections(local_path: str, section_limit: Optional[int]):
    sections = _iter_sections(local_path)
    if section_limit is None:
        yield from sections
        return
    yield from islice(sections, section_limit)


@task(log_prints=True)
def chunk_documents(
    local_path: str,
    output_path: str,
    api_key: str,
    max_workers: int,
    cache_path: str,
    section_limit: Optional[int],
) -> None:
    config = ChunkerConfig(
        debug=False,
        embedding_cache_enabled=True,
        embedding_cache_path=cache_path,
    )
    count = 0
    with open(output_path, "w") as fout:
        for row in iter_chunked_documents_parallel(
            _iter_limited_sections(local_path, section_limit),
            api_key,
            config,
            max_workers,
        ):
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} chunked documents to {output_path}")


@task(log_prints=True)
def upload_chunked_documents(local_path: str, bucket: str, blob_path: str) -> None:
    print(f"Uploading chunked documents to gs://{bucket}/{blob_path}")
    upload_blob(local_path, bucket, blob_path)


@flow(log_prints=True)
def chunk_documents_flow(
    source_bucket: str,
    source_blob: str,
    dest_bucket: str,
    dest_blob: str,
    max_workers: int = 48,
    cache_path: str = "/cache/patot/embedding_cache.sqlite",
    section_limit: Optional[int] = None,
) -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for chunking embeddings.")

    source_local_path = download_source(source_bucket, source_blob)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl", dir="/tmp") as tmp:
        output_local_path = tmp.name

    try:
        chunk_documents(source_local_path, output_local_path, api_key, max_workers, cache_path, section_limit)
        upload_chunked_documents(output_local_path, dest_bucket, dest_blob)
    finally:
        Path(source_local_path).unlink(missing_ok=True)
        Path(output_local_path).unlink(missing_ok=True)
