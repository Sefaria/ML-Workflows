import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ijson
from prefect import task

from embeddings.steps.patot.analytics import ChunkingRuntimeAnalytics
from embeddings.steps.patot.config import ChunkerConfig
from embeddings.steps.patot.pipeline import iter_chunked_documents_parallel
from embeddings.steps.query_generation import QueryGenerationConfig, generate_queries_and_qrels
from embeddings.steps.query_generation.analytics import QueryGenerationAnalytics
from embeddings.steps.query_generation.cache import flush_cache
from utils.gcs import download_blob, upload_directory
from utils.slack import SlackProgressReporter, SlackWebhookClient, slack_notified_flow


def _iter_raw_sections(local_path: str):
    with open(local_path, "rb") as handle:
        yield from ijson.items(handle, "item")


def _read_jsonl(local_path: str) -> list[dict]:
    rows = []
    with open(local_path, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(local_path: Path, rows: list[dict]) -> None:
    with local_path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _section_ref(section: dict) -> str:
    return str(section.get("ref") or "")


def _training_exclusion_refs(training_documents: list[dict]) -> set[str]:
    refs = set()
    for document in training_documents:
        metadata = document.get("metadata") or {}
        ref = metadata.get("ref")
        if ref:
            refs.add(str(ref))
        for source_segment_ref in metadata.get("source_segment_refs") or []:
            if source_segment_ref:
                refs.add(str(source_segment_ref).split("::fn:", 1)[0])
    return refs


def _documents_with_qrels(documents: list[dict], qrels: list[dict]) -> list[dict]:
    relevant_doc_ids = {str(qrel["doc_id"]) for qrel in qrels}
    return [document for document in documents if str(document["doc_id"]) in relevant_doc_ids]


def _with_retrieval_role(document: dict, role: str) -> dict:
    output = dict(document)
    metadata = dict(output.get("metadata") or {})
    metadata["retrieval_role"] = role
    output["metadata"] = metadata
    return output


def _query_generation_progress_details(snapshot: dict) -> dict:
    estimated_cost = snapshot.get("estimated_cost", {})
    return {
        "Queries": snapshot.get("queries_generated", 0),
        "Qrels": snapshot.get("qrels_generated", 0),
        "Cache hits": snapshot.get("cache", {}).get("hits", 0),
        "Cache misses": snapshot.get("cache", {}).get("misses", 0),
        "Remote requests": snapshot.get("llm", {}).get("remote_requests_succeeded", 0),
        "Failures": snapshot.get("llm", {}).get("remote_non_retryable_failures", 0),
        "Estimated remote cost": f"${estimated_cost.get('remote_estimated_cost_usd', 0.0):.6f}",
    }


def _chunking_progress_details(snapshot: dict) -> dict:
    return {
        "Docs": snapshot["chunked_documents_written"],
        "Cache hits": snapshot["cache"]["hits"],
        "Cache misses": snapshot["cache"]["misses"],
        "Remote requests": snapshot["embeddings"]["remote_requests_succeeded"],
        "Estimated remote cost": f"${snapshot['estimated_cost']['remote_estimated_cost_usd']:.6f}",
    }


@task(log_prints=True)
def download_source_dataset(bucket: str, blob_path: str) -> str:
    print(f"Downloading raw source sections from gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


@task(log_prints=True)
def download_training_documents(bucket: str, prefix: str) -> str:
    blob_path = f"{prefix.rstrip('/')}/documents.jsonl"
    print(f"Downloading training documents from gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


def _reservoir_sample_sections(
    source_local_path: str,
    excluded_refs: set[str],
    sample_size: int,
    sample_seed: int,
    role: str,
) -> dict:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    rng = random.Random(sample_seed)

    sampled_sections: list[dict] = []
    seen_eligible_refs = set()
    scanned_sections = 0
    missing_ref_sections = 0
    duplicate_eligible_refs = 0
    excluded_sections = 0
    eligible_sections = 0

    for section in _iter_raw_sections(source_local_path):
        scanned_sections += 1
        ref = _section_ref(section)
        if not ref:
            missing_ref_sections += 1
            continue
        if ref in excluded_refs:
            excluded_sections += 1
            continue
        if ref in seen_eligible_refs:
            duplicate_eligible_refs += 1
            continue
        seen_eligible_refs.add(ref)
        eligible_sections += 1

        if len(sampled_sections) < sample_size:
            sampled_sections.append(section)
            continue

        replacement_index = rng.randrange(eligible_sections)
        if replacement_index < sample_size:
            sampled_sections[replacement_index] = section

    if len(sampled_sections) < sample_size:
        print(
            "Held-out sample smaller than requested: "
            f"requested={sample_size}, sampled={len(sampled_sections)}, eligible={eligible_sections}"
        )

    sampled_sections.sort(key=lambda section: _section_ref(section))
    sampled_refs = [_section_ref(section) for section in sampled_sections]
    report = {
        "role": role,
        "strategy": "reservoir_sample_excluding_training_refs",
        "sample_size_requested": sample_size,
        "sample_seed": sample_seed,
        "sampled_sections_count": len(sampled_sections),
        "sampled_refs": sampled_refs,
        "source_sections_scanned": scanned_sections,
        "excluded_refs_count": len(excluded_refs),
        "excluded_sections_count": excluded_sections,
        "eligible_sections_count": eligible_sections,
        "missing_ref_sections_count": missing_ref_sections,
        "duplicate_eligible_refs_count": duplicate_eligible_refs,
    }
    print(
        f"{role} sampling summary: "
        f"scanned={scanned_sections}, eligible={eligible_sections}, excluded={excluded_sections}, "
        f"sampled={len(sampled_sections)}"
    )
    return {
        "sections": sampled_sections,
        "report": report,
    }


@task(log_prints=True)
def sample_heldout_sections(
    source_local_path: str,
    training_documents_local_path: str,
    sample_size: int,
    sample_seed: int,
    distractor_sample_size: int,
    distractor_sample_seed: int,
) -> dict:
    if distractor_sample_size < 0:
        raise ValueError("distractor_sample_size must be non-negative.")

    training_documents = _read_jsonl(training_documents_local_path)
    training_excluded_refs = _training_exclusion_refs(training_documents)
    positive_payload = _reservoir_sample_sections(
        source_local_path=source_local_path,
        excluded_refs=training_excluded_refs,
        sample_size=sample_size,
        sample_seed=sample_seed,
        role="positive",
    )
    positive_sections = list(positive_payload["sections"])
    positive_refs = {_section_ref(section) for section in positive_sections if _section_ref(section)}
    distractor_payload = {"sections": [], "report": None}
    if distractor_sample_size > 0:
        distractor_payload = _reservoir_sample_sections(
            source_local_path=source_local_path,
            excluded_refs=training_excluded_refs | positive_refs,
            sample_size=distractor_sample_size,
            sample_seed=distractor_sample_seed,
            role="distractor",
        )

    distractor_sections = list(distractor_payload["sections"])
    report = {
        "strategy": "positive_then_distractor_reservoir_sample_excluding_training_refs",
        "training_documents_count": len(training_documents),
        "training_excluded_refs_count": len(training_excluded_refs),
        "positive": positive_payload["report"],
        "distractor": distractor_payload["report"],
        "positive_sections_count": len(positive_sections),
        "distractor_sections_count": len(distractor_sections),
    }
    print(
        "Eval sampling summary: "
        f"training_docs={len(training_documents)}, "
        f"positive_sections={len(positive_sections)}, "
        f"distractor_sections={len(distractor_sections)}"
    )
    return {
        "query_sections": positive_sections,
        "distractor_sections": distractor_sections,
        "sections": positive_sections,
        "report": report,
    }


@task(log_prints=True)
def build_eval_dataset_artifacts(
    sampled_payload: dict,
    output_dir: str,
    api_key: str,
    chunk_max_workers: int,
    chunk_cache_path: str,
    model: str,
    query_max_workers: int,
    query_cache_path: str,
    flush_llm_cache: bool,
    queries_per_type_per_doc: int,
    query_types_per_doc: int,
    query_type_sample_seed: int,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    query_sections = list(sampled_payload["query_sections"])
    distractor_sections = list(sampled_payload.get("distractor_sections") or [])
    sampled_sections = query_sections + distractor_sections
    sampling_report = dict(sampled_payload["report"])
    _write_jsonl(output_root / "source_sections.jsonl", sampled_sections)
    _write_jsonl(output_root / "query_source_sections.jsonl", query_sections)
    _write_jsonl(output_root / "distractor_source_sections.jsonl", distractor_sections)

    chunking_analytics = ChunkingRuntimeAnalytics()
    chunk_config = ChunkerConfig(
        debug=False,
        embedding_cache_enabled=True,
        embedding_cache_path=chunk_cache_path,
        runtime_analytics=chunking_analytics,
    )

    chunk_reporter = SlackProgressReporter(
        workflow_name="Generate eval query dataset chunking",
        total_units=len(sampled_sections),
        client=SlackWebhookClient(username="ml-workflows"),
        unit_label="sections",
    )
    chunk_reporter.notify_start(
        {
            "Sections": len(sampled_sections),
            "Query sections": len(query_sections),
            "Distractor sections": len(distractor_sections),
            "Max workers": chunk_max_workers,
            "Cache path": chunk_cache_path,
        }
    )

    positive_chunked_documents = []
    for row in iter_chunked_documents_parallel(
        query_sections,
        api_key,
        chunk_config,
        chunk_max_workers,
    ):
        positive_chunked_documents.append(_with_retrieval_role(row, "positive"))
        snapshot = chunking_analytics.snapshot()
        chunk_reporter.notify_progress_if_due(
            snapshot["sections_processed"],
            _chunking_progress_details(snapshot),
        )

    distractor_chunked_documents = []
    if distractor_sections:
        for row in iter_chunked_documents_parallel(
            distractor_sections,
            api_key,
            chunk_config,
            chunk_max_workers,
        ):
            distractor_chunked_documents.append(_with_retrieval_role(row, "distractor"))
            snapshot = chunking_analytics.snapshot()
            chunk_reporter.notify_progress_if_due(
                snapshot["sections_processed"],
                _chunking_progress_details(snapshot),
            )

    corpus_documents = positive_chunked_documents + distractor_chunked_documents

    _write_jsonl(output_root / "chunked_documents.jsonl", corpus_documents)
    _write_jsonl(output_root / "positive_chunked_documents.jsonl", positive_chunked_documents)
    _write_jsonl(output_root / "distractor_chunked_documents.jsonl", distractor_chunked_documents)
    chunking_snapshot = chunking_analytics.snapshot()
    chunk_reporter.notify_success(
        {
            "Sections": f"{chunking_snapshot['sections_processed']}/{len(sampled_sections)}",
            **_chunking_progress_details(chunking_snapshot),
            "Positive docs": len(positive_chunked_documents),
            "Distractor docs": len(distractor_chunked_documents),
            "Corpus docs": len(corpus_documents),
        }
    )

    query_analytics = QueryGenerationAnalytics()
    if flush_llm_cache:
        print(f"Flushing persistent eval query-generation LLM cache at {query_cache_path}")
        flush_cache(query_cache_path)

    query_config = QueryGenerationConfig(
        model=model,
        llm_max_workers=query_max_workers,
        llm_cache_enabled=True,
        llm_cache_path=query_cache_path,
        queries_per_type_per_doc=queries_per_type_per_doc,
        query_types_per_doc=query_types_per_doc,
        query_type_sample_seed=query_type_sample_seed,
        runtime_analytics=query_analytics,
        verbose=True,
    )
    total_query_jobs = len(positive_chunked_documents) * min(query_config.query_types_per_doc, len(query_config.query_types))
    query_reporter = SlackProgressReporter(
        workflow_name="Generate eval query dataset queries",
        total_units=total_query_jobs,
        client=SlackWebhookClient(username="ml-workflows"),
        unit_label="jobs",
    )
    query_reporter.notify_start(
        {
            "Documents": len(positive_chunked_documents),
            "Corpus documents": len(corpus_documents),
            "Distractor documents": len(distractor_chunked_documents),
            "Model": model,
            "Max workers": query_max_workers,
            "Cache path": query_cache_path,
            "Flush cache": flush_llm_cache,
            "Query type seed": query_type_sample_seed,
        }
    )

    progress_log_every_jobs = 100
    progress_log_every_seconds = 30.0
    last_logged_jobs = 0
    last_log_time = time.monotonic()

    def report_query_progress(completed_jobs: int, total_jobs: int, snapshot: dict) -> None:
        nonlocal last_logged_jobs, last_log_time
        query_reporter.notify_progress_if_due(completed_jobs, _query_generation_progress_details(snapshot))
        now = time.monotonic()
        if completed_jobs < last_logged_jobs + progress_log_every_jobs and now - last_log_time < progress_log_every_seconds:
            return
        estimated_cost = snapshot.get("estimated_cost", {})
        print(
            "Eval query generation progress: "
            f"jobs={completed_jobs}/{total_jobs}, "
            f"queries={snapshot.get('queries_generated', 0)}, "
            f"qrels={snapshot.get('qrels_generated', 0)}, "
            f"cache_hits={snapshot.get('cache', {}).get('hits', 0)}, "
            f"cache_misses={snapshot.get('cache', {}).get('misses', 0)}, "
            f"remote_requests={snapshot.get('llm', {}).get('remote_requests_succeeded', 0)}, "
            f"failures={snapshot.get('llm', {}).get('remote_non_retryable_failures', 0)}, "
            f"estimated_remote_cost_usd={estimated_cost.get('remote_estimated_cost_usd', 0.0):.6f}"
        )
        last_logged_jobs = completed_jobs
        last_log_time = now

    query_config = replace(query_config, progress_callback=report_query_progress)
    queries, qrels, failures = generate_queries_and_qrels(positive_chunked_documents, query_config)
    documents_with_qrels = _documents_with_qrels(corpus_documents, qrels)

    _write_jsonl(output_root / "documents.jsonl", corpus_documents)
    _write_jsonl(output_root / "documents_with_qrels.jsonl", documents_with_qrels)
    _write_jsonl(output_root / "queries.jsonl", queries)
    _write_jsonl(output_root / "qrels.jsonl", qrels)
    _write_jsonl(output_root / "failures.jsonl", failures)

    query_snapshot = query_analytics.snapshot()
    query_reporter.notify_success(
        {
            "Corpus documents": len(corpus_documents),
            "Documents with qrels": len(documents_with_qrels),
            "Distractor documents": len(distractor_chunked_documents),
            **_query_generation_progress_details(query_snapshot),
        }
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_stage": "eval_query_dataset_generation",
        "sampling": sampling_report,
        "chunking": {
            "input_sections_count": len(sampled_sections),
            "query_sections_count": len(query_sections),
            "distractor_sections_count": len(distractor_sections),
            "chunked_documents_count": len(corpus_documents),
            "positive_documents_count": len(positive_chunked_documents),
            "distractor_documents_count": len(distractor_chunked_documents),
            "corpus_documents_count": len(corpus_documents),
            "documents_with_qrels_count": len(documents_with_qrels),
            "implicit_negative_documents_count": len(corpus_documents) - len(documents_with_qrels),
            "chunk_max_workers": chunk_max_workers,
            "chunk_cache_path": chunk_cache_path,
            "runtime_analytics": chunking_snapshot,
        },
        "query_generation": {
            "model": model,
            "queries_count": len(queries),
            "qrels_count": len(qrels),
            "failures_count": len(failures),
            "query_types": list(query_config.query_types),
            "query_types_per_doc": query_config.query_types_per_doc,
            "query_type_sample_seed": query_config.query_type_sample_seed,
            "queries_per_type_per_doc": query_config.queries_per_type_per_doc,
            "llm_max_workers": query_max_workers,
            "llm_max_retries": query_config.max_retries,
            "llm_request_timeout_seconds": query_config.request_timeout_seconds,
            "llm_cache_path": query_cache_path,
            "flush_llm_cache": flush_llm_cache,
            "runtime_analytics": query_snapshot,
        },
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(
        "Eval query dataset summary: "
        f"source_sections={len(sampled_sections)}, "
        f"query_sections={len(query_sections)}, "
        f"distractor_sections={len(distractor_sections)}, "
        f"corpus_documents={len(corpus_documents)}, "
        f"positive_documents={len(positive_chunked_documents)}, "
        f"distractor_documents={len(distractor_chunked_documents)}, "
        f"documents_with_qrels={len(documents_with_qrels)}, "
        f"queries={len(queries)}, "
        f"qrels={len(qrels)}, "
        f"failures={len(failures)}"
    )
    return metadata


@task(log_prints=True)
def upload_eval_dataset(local_dir: str, bucket: str, prefix: str) -> None:
    print(f"Uploading eval query dataset artifacts to gs://{bucket}/{prefix}")
    upload_directory(local_dir, bucket, prefix)


@slack_notified_flow(workflow_name="sample-chunk-eval-query-dataset", log_prints=True)
def sample_chunk_eval_query_dataset_flow(
    source_bucket: str,
    source_blob: str,
    training_dataset_bucket: str,
    training_dataset_prefix: str,
    dest_bucket: str,
    dest_prefix: str,
    sample_size: int = 100,
    sample_seed: int = 613,
    distractor_sample_size: int = 1000,
    distractor_sample_seed: int = 614,
    chunk_max_workers: int = 48,
    chunk_cache_path: str = "/cache/patot/embedding_cache.sqlite",
    model: str = "claude-sonnet-4-6",
    query_max_workers: int = 4,
    query_cache_path: str = "/cache/query_generation/eval_llm_cache.sqlite",
    flush_llm_cache: bool = False,
    queries_per_type_per_doc: int = 1,
    query_types_per_doc: int = 2,
    query_type_sample_seed: int = 613,
) -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for chunking embeddings.")

    source_local_path = download_source_dataset(source_bucket, source_blob)
    training_documents_local_path = download_training_documents(training_dataset_bucket, training_dataset_prefix)
    output_dir = tempfile.mkdtemp(dir="/tmp")

    try:
        sampled_payload = sample_heldout_sections(
            source_local_path,
            training_documents_local_path,
            sample_size,
            sample_seed,
            distractor_sample_size,
            distractor_sample_seed,
        )
        metadata = build_eval_dataset_artifacts(
            sampled_payload,
            output_dir,
            api_key,
            chunk_max_workers,
            chunk_cache_path,
            model,
            query_max_workers,
            query_cache_path,
            flush_llm_cache,
            queries_per_type_per_doc,
            query_types_per_doc,
            query_type_sample_seed,
        )
        metadata["source"] = {
            "source_bucket": source_bucket,
            "source_blob": source_blob,
            "training_dataset_bucket": training_dataset_bucket,
            "training_dataset_prefix": training_dataset_prefix,
            "dest_bucket": dest_bucket,
            "dest_prefix": dest_prefix,
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "distractor_sample_size": distractor_sample_size,
            "distractor_sample_seed": distractor_sample_seed,
        }
        Path(output_dir, "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        upload_eval_dataset(output_dir, dest_bucket, dest_prefix)
    finally:
        Path(source_local_path).unlink(missing_ok=True)
        Path(training_documents_local_path).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
