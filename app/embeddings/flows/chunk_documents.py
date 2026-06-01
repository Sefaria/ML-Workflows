import json
import os
import tempfile
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Optional

import ijson
from embeddings.steps.patot.analytics import ChunkingRuntimeAnalytics
from prefect import flow, task

from embeddings.steps.patot.config import ChunkerConfig
from embeddings.steps.patot.pipeline import iter_chunked_documents_parallel
from utils.gcs import download_blob, upload_blob
from utils.slack import SlackProgressReporter, SlackWebhookClient


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
def count_source_sections(local_path: str, section_limit: Optional[int]) -> int:
    if section_limit is not None:
        print(f"Using section_limit={section_limit} as chunking progress total")
        return section_limit

    print(f"Counting source sections in {local_path}")
    count = sum(1 for _ in _iter_sections(local_path))
    print(f"Counted {count} source sections")
    return count


def chunking_progress_details(snapshot: dict) -> dict:
    return {
        "Docs": snapshot["chunked_documents_written"],
        "Cache hits": snapshot["cache"]["hits"],
        "Cache misses": snapshot["cache"]["misses"],
        "Remote requests": snapshot["embeddings"]["remote_requests_succeeded"],
        "Estimated remote cost": f"${snapshot['estimated_cost']['remote_estimated_cost_usd']:.6f}",
    }


@task(log_prints=True)
def chunk_documents(
    local_path: str,
    output_path: str,
    api_key: str,
    max_workers: int,
    cache_path: str,
    section_limit: Optional[int],
    total_sections: int,
) -> dict:
    analytics = ChunkingRuntimeAnalytics()
    config = ChunkerConfig(
        debug=False,
        embedding_cache_enabled=True,
        embedding_cache_path=cache_path,
        runtime_analytics=analytics,
    )
    count = 0
    progress_log_every_sections = 100
    progress_log_every_seconds = 30.0
    last_logged_sections = 0
    last_log_time = time.monotonic()
    slack_reporter = SlackProgressReporter(
        workflow_name="Chunk documents",
        total_units=total_sections,
        client=SlackWebhookClient(username="ml-workflows"),
    )
    slack_reporter.notify_start(
        {
            "Input path": local_path,
            "Max workers": max_workers,
            "Cache path": cache_path,
        }
    )
    with open(output_path, "w") as fout:
        for row in iter_chunked_documents_parallel(
            _iter_limited_sections(local_path, section_limit),
            api_key,
            config,
            max_workers,
        ):
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            snapshot = analytics.snapshot()
            sections_processed = snapshot["sections_processed"]
            slack_reporter.notify_progress_if_due(
                sections_processed,
                chunking_progress_details(snapshot),
            )
            now = time.monotonic()
            should_log = False
            if sections_processed >= last_logged_sections + progress_log_every_sections:
                should_log = True
            elif now - last_log_time >= progress_log_every_seconds and sections_processed > last_logged_sections:
                should_log = True
            if should_log:
                progress_label = (
                    f"{sections_processed}/{total_sections}"
                    if total_sections
                    else str(sections_processed)
                )
                print(
                    "Chunking progress: "
                    f"sections={progress_label}, "
                    f"docs={snapshot['chunked_documents_written']}, "
                    f"cache_hits={snapshot['cache']['hits']}, "
                    f"cache_misses={snapshot['cache']['misses']}, "
                    f"remote_requests={snapshot['embeddings']['remote_requests_succeeded']}, "
                    f"estimated_remote_cost_usd={snapshot['estimated_cost']['remote_estimated_cost_usd']:.6f}"
                )
                last_logged_sections = sections_processed
                last_log_time = now
    print(f"Wrote {count} chunked documents to {output_path}")
    final_snapshot = analytics.snapshot()
    slack_reporter.notify_success(
        {
            "Sections": f"{final_snapshot['sections_processed']}/{total_sections}",
            **chunking_progress_details(final_snapshot),
            "Output path": output_path,
        }
    )
    return {
        "chunked_documents_written": count,
        "runtime_analytics": final_snapshot,
    }


@task(log_prints=True)
def upload_chunked_documents(local_path: str, bucket: str, blob_path: str) -> None:
    print(f"Uploading chunked documents to gs://{bucket}/{blob_path}")
    upload_blob(local_path, bucket, blob_path)


@task(log_prints=True)
def upload_runtime_analytics(report: dict, bucket: str, blob_path: str) -> None:
    print(f"Uploading chunking runtime analytics to gs://{bucket}/{blob_path}")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", dir="/tmp") as tmp:
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        local_path = tmp.name
    try:
        upload_blob(local_path, bucket, blob_path)
    finally:
        Path(local_path).unlink(missing_ok=True)


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

    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    total_sections = 0

    try:
        total_sections = count_source_sections(source_local_path, section_limit)
        chunk_result = chunk_documents(
            source_local_path,
            output_local_path,
            api_key,
            max_workers,
            cache_path,
            section_limit,
            total_sections,
        )
        upload_chunked_documents(output_local_path, dest_bucket, dest_blob)
        finished_at = datetime.now(timezone.utc).isoformat()
        duration_seconds = time.monotonic() - started_monotonic
        analytics_blob = f"{dest_blob}.runtime_analytics.json"
        analytics_report = {
            "generated_at": finished_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "source_bucket": source_bucket,
            "source_blob": source_blob,
            "dest_bucket": dest_bucket,
            "dest_blob": dest_blob,
            "cache_path": cache_path,
            "max_workers": max_workers,
            "section_limit": section_limit,
            "total_sections": total_sections,
            "chunked_documents_written": chunk_result["chunked_documents_written"],
            "runtime_analytics": chunk_result["runtime_analytics"],
        }
        summary = analytics_report["runtime_analytics"]
        print(
            "Chunking analytics summary: "
            f"sections={summary['sections_processed']}, "
            f"docs={summary['chunked_documents_written']}, "
            f"cache_hits={summary['cache']['hits']}, "
            f"cache_misses={summary['cache']['misses']}, "
            f"estimated_remote_cost_usd={summary['estimated_cost']['remote_estimated_cost_usd']:.6f}"
        )
        upload_runtime_analytics(analytics_report, dest_bucket, analytics_blob)
    except Exception as exc:
        SlackProgressReporter(
            workflow_name="Chunk documents",
            total_units=max(total_sections, 1),
            client=SlackWebhookClient(username="ml-workflows"),
        ).notify_failure(
            exc,
            {
                "Source": f"gs://{source_bucket}/{source_blob}",
                "Destination": f"gs://{dest_bucket}/{dest_blob}",
            },
        )
        raise
    finally:
        Path(source_local_path).unlink(missing_ok=True)
        Path(output_local_path).unlink(missing_ok=True)
