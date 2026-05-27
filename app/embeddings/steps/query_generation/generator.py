import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, List, Tuple

from anthropic import Anthropic

from .config import QueryGenerationConfig
from .prompts import build_query_type_prompt, build_system_prompt
from .sampling import choose_query_types_for_doc
from .types import ChunkedDocument, QrelRecord, QueryRecord


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


def create_llm_client(config: QueryGenerationConfig):
    return Anthropic()


def get_thread_llm_client(config: QueryGenerationConfig):
    client = getattr(_thread_local, "query_generation_client", None)
    if client is None:
        client = create_llm_client(config)
        _thread_local.query_generation_client = client
    return client


def call_anthropic_for_query_type(
    client: Anthropic,
    doc: ChunkedDocument,
    query_type: str,
    config: QueryGenerationConfig,
) -> dict:
    prompt = build_query_type_prompt(doc, query_type, config)
    log(f"Calling Anthropic {config.model} for {doc.doc_id} {query_type}", config)
    response = client.messages.create(
        model=config.model,
        temperature=0,
        max_tokens=config.max_tokens,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )
    content = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return parse_json_response(content)


def call_llm_for_query_type(
    doc: ChunkedDocument,
    query_type: str,
    config: QueryGenerationConfig,
) -> dict:
    client = get_thread_llm_client(config)
    return call_anthropic_for_query_type(client, doc, query_type, config)


def normalize_query_type_output(
    output: dict,
    doc: ChunkedDocument,
    query_type: str,
    job_index: int,
    config: QueryGenerationConfig,
) -> Tuple[List[QueryRecord], List[QrelRecord]]:
    queries: List[QueryRecord] = []
    qrels: List[QrelRecord] = []
    query_lang = doc.lang

    for query_index, raw_query in enumerate(output.get("queries", [])):
        if query_index >= config.queries_per_type_per_doc:
            break
        query_text = str(raw_query.get("text") or "").strip()
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
                reason=str(raw_query.get("reason") or "").strip(),
            )
        )
    return queries, qrels


def generate_queries_for_documents(
    documents: List[ChunkedDocument],
    config: QueryGenerationConfig,
) -> Tuple[List[QueryRecord], List[QrelRecord]]:
    jobs = [
        {"doc": doc, "query_type": query_type, "index": index}
        for index, (doc, query_type) in enumerate(
            (doc, query_type)
            for doc in documents
            for query_type in choose_query_types_for_doc(doc, config)
        )
    ]

    all_queries: List[QueryRecord] = []
    all_qrels: List[QrelRecord] = []
    start = time.perf_counter()

    def run_job(job: Dict) -> Tuple[List[QueryRecord], List[QrelRecord]]:
        output = call_llm_for_query_type(job["doc"], job["query_type"], config)
        return normalize_query_type_output(output, job["doc"], job["query_type"], job["index"], config)

    with ThreadPoolExecutor(max_workers=config.llm_max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            queries, qrels = future.result()
            all_queries.extend(queries)
            all_qrels.extend(qrels)

    log(
        f"Generated {len(all_queries)} queries and {len(all_qrels)} qrels in {time.perf_counter() - start:.2f}s",
        config,
    )
    return all_queries, all_qrels


def generate_queries_and_qrels(
    documents: List[dict],
    config: QueryGenerationConfig = QueryGenerationConfig(),
) -> Tuple[List[dict], List[dict]]:
    chunked_documents = [
        ChunkedDocument(
            doc_id=str(document["doc_id"]),
            text=str(document["text"]),
            metadata=dict(document.get("metadata") or {}),
        )
        for document in documents
    ]
    queries, qrels = generate_queries_for_documents(chunked_documents, config)
    return [asdict(query) for query in queries], [asdict(qrel) for qrel in qrels]
