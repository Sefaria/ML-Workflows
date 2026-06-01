import json
import shutil
import tempfile
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Optional

from prefect import flow, task

from embeddings.steps.query_generation.analytics import QueryGenerationAnalytics
from embeddings.steps.query_generation.cache import flush_cache
from embeddings.steps.query_generation import QueryGenerationConfig, generate_queries_and_qrels
from utils.gcs import download_blob, upload_directory
from utils.slack import notify_workflow_started


def _read_jsonl(path: str, document_limit: Optional[int] = None) -> list[dict]:
    rows = []
    with open(path, "r") as fin:
        line_iter = fin if document_limit is None else islice(fin, document_limit)
        for line in line_iter:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _documents_with_qrels(documents: list[dict], qrels: list[dict]) -> list[dict]:
    relevant_doc_ids = {str(qrel["doc_id"]) for qrel in qrels}
    return [document for document in documents if str(document["doc_id"]) in relevant_doc_ids]


@task(log_prints=True)
def download_chunked_documents(bucket: str, blob_path: str) -> str:
    print(f"Downloading chunked documents from gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


@task(log_prints=True)
def build_query_dataset(
    local_path: str,
    output_dir: str,
    model: str,
    max_workers: int,
    cache_path: str,
    flush_llm_cache: bool,
    document_limit: Optional[int],
    queries_per_type_per_doc: int,
    query_types_per_doc: int,
) -> None:
    documents = _read_jsonl(local_path, document_limit=document_limit)
    analytics = QueryGenerationAnalytics()
    if flush_llm_cache:
        print(f"Flushing persistent LLM cache at {cache_path}")
        flush_cache(cache_path)
    config = QueryGenerationConfig(
        model=model,
        llm_max_workers=max_workers,
        llm_cache_enabled=True,
        llm_cache_path=cache_path,
        queries_per_type_per_doc=queries_per_type_per_doc,
        query_types_per_doc=query_types_per_doc,
        runtime_analytics=analytics,
        verbose=True,
    )
    queries, qrels, failures = generate_queries_and_qrels(documents, config)
    dataset_documents = _documents_with_qrels(documents, qrels)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("documents.jsonl", dataset_documents),
        ("queries.jsonl", queries),
        ("qrels.jsonl", qrels),
        ("failures.jsonl", failures),
    ):
        with (output_root / filename).open("w") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "input_documents_count": len(documents),
        "documents_count": len(dataset_documents),
        "dropped_documents_count": len(documents) - len(dataset_documents),
        "queries_count": len(queries),
        "qrels_count": len(qrels),
        "failures_count": len(failures),
        "query_types": list(config.query_types),
        "query_types_per_doc": config.query_types_per_doc,
        "queries_per_type_per_doc": config.queries_per_type_per_doc,
        "llm_max_workers": max_workers,
        "llm_max_retries": config.max_retries,
        "llm_request_timeout_seconds": config.request_timeout_seconds,
        "llm_cache_path": cache_path,
        "flush_llm_cache": flush_llm_cache,
        "document_limit": document_limit,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    runtime_analytics = analytics.snapshot()
    (output_root / "runtime_analytics.json").write_text(json.dumps(runtime_analytics, ensure_ascii=False, indent=2))
    print(
        "Query generation analytics summary: "
        f"documents={runtime_analytics['documents_count']}, "
        f"jobs={runtime_analytics['jobs_count']}, "
        f"queries={runtime_analytics['queries_generated']}, "
        f"cache_hits={runtime_analytics['cache']['hits']}, "
        f"cache_misses={runtime_analytics['cache']['misses']}, "
        f"estimated_remote_cost_usd={runtime_analytics['estimated_cost']['remote_estimated_cost_usd']:.6f}"
    )
    print(f"Wrote dataset artifacts to {output_root}")


@task(log_prints=True)
def upload_query_dataset(local_dir: str, bucket: str, prefix: str) -> None:
    print(f"Uploading query dataset artifacts to gs://{bucket}/{prefix}")
    upload_directory(local_dir, bucket, prefix)


@flow(log_prints=True)
def generate_query_dataset_flow(
    source_bucket: str,
    source_blob: str,
    dest_bucket: str,
    dest_prefix: str,
    model: str = "claude-sonnet-4-6",
    max_workers: int = 4,
    cache_path: str = "/cache/query_generation/llm_cache.sqlite",
    flush_llm_cache: bool = False,
    document_limit: Optional[int] = None,
    queries_per_type_per_doc: int = 1,
    query_types_per_doc: int = 2,
) -> None:
    notify_workflow_started(
        "generate-query-dataset",
        {
            "Source": f"gs://{source_bucket}/{source_blob}",
            "Destination": f"gs://{dest_bucket}/{dest_prefix}",
            "Model": model,
            "Max workers": max_workers,
            "Cache path": cache_path,
            "Flush cache": flush_llm_cache,
            "Document limit": document_limit,
            "Queries per type per doc": queries_per_type_per_doc,
            "Query types per doc": query_types_per_doc,
        },
    )
    source_local_path = download_chunked_documents(source_bucket, source_blob)
    output_dir = tempfile.mkdtemp(dir="/tmp")

    try:
        build_query_dataset(
            source_local_path,
            output_dir,
            model,
            max_workers,
            cache_path,
            flush_llm_cache,
            document_limit,
            queries_per_type_per_doc,
            query_types_per_doc,
        )
        upload_query_dataset(output_dir, dest_bucket, dest_prefix)
    finally:
        Path(source_local_path).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
