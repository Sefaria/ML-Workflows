import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic

from .analytics import QueryGenerationAnalytics
from .cache import cache_lookup, cache_update
from .config import QueryGenerationConfig
from .prompts import build_query_type_prompt, build_system_prompt
from .sampling import choose_query_types_for_doc
from .types import ChunkedDocument, QrelRecord, QueryGenerationFailure, QueryGenerationResponse, QueryRecord


_thread_local = threading.local()


def log(message: str, config: QueryGenerationConfig) -> None:
    if config.verbose:
        print(message)


def parse_json_response(content: str) -> dict:
    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*", "", normalized)
        normalized = re.sub(r"\s*```$", "", normalized)
    return json.loads(normalized)


def create_structured_llm(config: QueryGenerationConfig):
    model = ChatAnthropic(
        model=config.model,
        temperature=0,
        max_retries=config.max_retries,
        timeout=config.request_timeout_seconds,
    )
    return model.with_structured_output(QueryGenerationResponse, include_raw=True)


def get_thread_structured_llm(config: QueryGenerationConfig):
    client_key = (
        config.model,
        config.max_retries,
        config.request_timeout_seconds,
    )
    client = getattr(_thread_local, "query_generation_structured_client", None)
    configured_key = getattr(_thread_local, "query_generation_structured_client_key", None)
    if client is None or configured_key != client_key:
        client = create_structured_llm(config)
        _thread_local.query_generation_structured_client = client
        _thread_local.query_generation_structured_client_key = client_key
    return client


def _extract_usage(raw_message: Any) -> dict:
    usage_metadata = getattr(raw_message, "usage_metadata", None) or {}
    response_metadata = getattr(raw_message, "response_metadata", None) or {}
    token_usage = response_metadata.get("usage") or response_metadata.get("token_usage") or {}
    return {
        "input_tokens": int(
            usage_metadata.get("input_tokens")
            or token_usage.get("input_tokens")
            or token_usage.get("prompt_tokens")
            or 0
        ),
        "output_tokens": int(
            usage_metadata.get("output_tokens")
            or token_usage.get("output_tokens")
            or token_usage.get("completion_tokens")
            or 0
        ),
    }


def _parse_cached_response(content: str) -> QueryGenerationResponse:
    try:
        return QueryGenerationResponse.model_validate_json(content)
    except ValueError:
        return QueryGenerationResponse.model_validate(parse_json_response(content))


def call_structured_llm_for_query_type(
    client,
    doc: ChunkedDocument,
    query_type: str,
    config: QueryGenerationConfig,
) -> QueryGenerationResponse:
    prompt = build_query_type_prompt(doc, query_type, config)
    system_prompt = build_system_prompt()
    llm_string = (
        f"langchain_anthropic_structured|model={config.model}|"
        "temperature=0|"
        f"schema=QueryGenerationResponse|"
        f"system={system_prompt}"
    )
    analytics = config.runtime_analytics
    if config.llm_cache_enabled and config.llm_cache_path:
        cached = cache_lookup(prompt, llm_string, config.llm_cache_path)
        if cached is not None:
            if analytics is not None:
                analytics.record_cache_hit(cached.get("usage"))
            return _parse_cached_response(cached["content"])
        if analytics is not None:
            analytics.record_cache_miss()

    log(f"Calling Anthropic structured output {config.model} for {doc.doc_id} {query_type}", config)
    response = client.invoke(
        [
            ("system", system_prompt),
            ("user", prompt),
        ]
    )
    raw_message = response.get("raw") if isinstance(response, dict) else None
    usage = _extract_usage(raw_message)
    if analytics is not None:
        analytics.record_remote_success(usage["input_tokens"], usage["output_tokens"])

    parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
    parsed = response.get("parsed") if isinstance(response, dict) else None
    if parsing_error is not None:
        raise ValueError(f"Structured output parsing failed: {parsing_error}")
    if parsed is None:
        raise ValueError("Structured output parsing failed: no parsed response returned")

    if config.llm_cache_enabled and config.llm_cache_path:
        cache_update(prompt, llm_string, parsed.model_dump_json(), usage, config.llm_cache_path)
        if analytics is not None:
            analytics.record_cache_write()
    return parsed


def call_llm_for_query_type(
    doc: ChunkedDocument,
    query_type: str,
    config: QueryGenerationConfig,
) -> QueryGenerationResponse:
    client = get_thread_structured_llm(config)
    return call_structured_llm_for_query_type(client, doc, query_type, config)


def normalize_query_type_output(
    output: QueryGenerationResponse,
    doc: ChunkedDocument,
    query_type: str,
    job_index: int,
    config: QueryGenerationConfig,
) -> Tuple[List[QueryRecord], List[QrelRecord]]:
    queries: List[QueryRecord] = []
    qrels: List[QrelRecord] = []
    query_lang = doc.lang

    if output.skip is True:
        log(f"Skipping {doc.doc_id} {query_type}: {output.skip_reason or 'no reason provided'}", config)
        return queries, qrels

    for query_index, raw_query in enumerate(output.queries):
        if query_index >= config.queries_per_type_per_doc:
            break
        query_text = raw_query.text.strip()
        if not query_text:
            continue
        query_id = f"q_{query_lang}_{query_type}_{job_index}_{query_index}"
        queries.append(
            QueryRecord(
                query_id=query_id,
                text=query_text,
                type=query_type,
                lang=query_lang,
            )
        )
        qrels.append(
            QrelRecord(
                query_id=query_id,
                doc_id=doc.doc_id,
                relevance=1,
                reason=raw_query.reason.strip(),
            )
        )
    return queries, qrels


def build_failure(job: Dict, error: Exception) -> QueryGenerationFailure:
    doc = job["doc"]
    return QueryGenerationFailure(
        doc_id=doc.doc_id,
        query_type=str(job["query_type"]),
        job_index=int(job["index"]),
        error_type=error.__class__.__name__,
        error_message=str(error),
        ref=str(doc.metadata.get("ref") or ""),
    )


def generate_queries_for_documents(
    documents: List[ChunkedDocument],
    config: QueryGenerationConfig,
) -> Tuple[List[QueryRecord], List[QrelRecord], List[QueryGenerationFailure]]:
    jobs = [
        {"doc": doc, "query_type": query_type, "index": index}
        for index, (doc, query_type) in enumerate(
            (doc, query_type)
            for doc in documents
            for query_type in choose_query_types_for_doc(doc, config)
        )
    ]
    if config.runtime_analytics is not None:
        config.runtime_analytics.record_documents_count(len(documents))
        config.runtime_analytics.record_jobs_count(len(jobs))

    all_queries: List[QueryRecord] = []
    all_qrels: List[QrelRecord] = []
    all_failures: List[QueryGenerationFailure] = []
    start = time.perf_counter()

    def run_job(job: Dict) -> Tuple[List[QueryRecord], List[QrelRecord], Optional[QueryGenerationFailure]]:
        try:
            output = call_llm_for_query_type(job["doc"], job["query_type"], config)
            queries, qrels = normalize_query_type_output(output, job["doc"], job["query_type"], job["index"], config)
            if config.runtime_analytics is not None:
                config.runtime_analytics.record_generated_counts(len(queries), len(qrels))
            return queries, qrels, None
        except Exception as error:
            if config.runtime_analytics is not None:
                config.runtime_analytics.record_remote_non_retryable_failure()
            return [], [], build_failure(job, error)

    with ThreadPoolExecutor(max_workers=config.llm_max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for completed_jobs, future in enumerate(as_completed(futures), start=1):
            queries, qrels, failure = future.result()
            all_queries.extend(queries)
            all_qrels.extend(qrels)
            if failure is not None:
                all_failures.append(failure)
            if config.progress_callback is not None:
                snapshot = config.runtime_analytics.snapshot() if config.runtime_analytics is not None else {}
                try:
                    config.progress_callback(completed_jobs, len(jobs), snapshot)
                except Exception as error:
                    print(f"Query generation progress callback failed: {type(error).__name__}: {error}")

    log(
        f"Generated {len(all_queries)} queries and {len(all_qrels)} qrels "
        f"with {len(all_failures)} failures in {time.perf_counter() - start:.2f}s",
        config,
    )
    return all_queries, all_qrels, all_failures


def generate_queries_and_qrels(
    documents: List[dict],
    config: QueryGenerationConfig = QueryGenerationConfig(),
) -> Tuple[List[dict], List[dict], List[dict]]:
    chunked_documents = [
        ChunkedDocument(
            doc_id=str(document["doc_id"]),
            text=str(document["text"]),
            metadata=dict(document.get("metadata") or {}),
        )
        for document in documents
    ]
    queries, qrels, failures = generate_queries_for_documents(chunked_documents, config)
    return [asdict(query) for query in queries], [asdict(qrel) for qrel in qrels], [asdict(failure) for failure in failures]
