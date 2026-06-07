from __future__ import annotations

import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from embeddings.steps.query_generation.cache import cache_lookup, cache_update
from embeddings.steps.query_generation.generator import _extract_usage


SYSTEM_PROMPT = """You judge retrieval relevance for a search evaluation.

Given a user query and one retrieved document, decide whether the document is relevant.
Mark relevant=true only when the document directly answers the query, supplies information that would satisfy the query, or is clearly semantically about the same specific subject.
Mark relevant=false when the document is merely topically adjacent, shares broad vocabulary, or does not answer what the query asks.
"""


class RelevanceJudgment(BaseModel):
    relevant: bool = Field(description="True if the document is relevant to the query.")
    reason: str = Field(description="Brief explanation for the binary decision.")


@dataclass
class LlmRelevanceAnalytics:
    input_cost_per_million_tokens_usd: float = 3.0
    output_cost_per_million_tokens_usd: float = 15.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    jobs_count: int = 0
    completed_jobs: int = 0
    relevant_judgments: int = 0
    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    remote_requests_succeeded: int = 0
    remote_failures: int = 0
    remote_input_tokens: int = 0
    remote_output_tokens: int = 0
    cached_input_tokens_saved: int = 0
    cached_output_tokens_saved: int = 0

    def record_jobs_count(self, count: int) -> None:
        with self._lock:
            self.jobs_count = count

    def record_completed(self, relevant: bool) -> None:
        with self._lock:
            self.completed_jobs += 1
            if relevant:
                self.relevant_judgments += 1

    def record_cache_hit(self, usage: Optional[dict]) -> None:
        with self._lock:
            self.cache_lookups += 1
            self.cache_hits += 1
            if usage:
                self.cached_input_tokens_saved += int(usage.get("input_tokens") or 0)
                self.cached_output_tokens_saved += int(usage.get("output_tokens") or 0)

    def record_cache_miss(self) -> None:
        with self._lock:
            self.cache_lookups += 1
            self.cache_misses += 1

    def record_cache_write(self) -> None:
        with self._lock:
            self.cache_writes += 1

    def record_remote_success(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.remote_requests_succeeded += 1
            self.remote_input_tokens += input_tokens
            self.remote_output_tokens += output_tokens

    def record_remote_failure(self) -> None:
        with self._lock:
            self.remote_failures += 1

    def _cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000.0) * self.input_cost_per_million_tokens_usd
            + (output_tokens / 1_000_000.0) * self.output_cost_per_million_tokens_usd
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            remote_cost = self._cost_usd(self.remote_input_tokens, self.remote_output_tokens)
            saved_cost = self._cost_usd(self.cached_input_tokens_saved, self.cached_output_tokens_saved)
            return {
                "jobs_count": self.jobs_count,
                "completed_jobs": self.completed_jobs,
                "relevant_judgments": self.relevant_judgments,
                "cache": {
                    "lookups": self.cache_lookups,
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "writes": self.cache_writes,
                    "hit_rate": self.cache_hits / self.cache_lookups if self.cache_lookups else 0.0,
                },
                "llm": {
                    "remote_requests_succeeded": self.remote_requests_succeeded,
                    "remote_failures": self.remote_failures,
                    "remote_input_tokens": self.remote_input_tokens,
                    "remote_output_tokens": self.remote_output_tokens,
                    "cached_input_tokens_saved": self.cached_input_tokens_saved,
                    "cached_output_tokens_saved": self.cached_output_tokens_saved,
                },
                "estimated_cost": {
                    "pricing_source": "https://docs.anthropic.com/en/docs/about-claude/pricing",
                    "pricing_note": "Assumes the configured Claude model uses Sonnet 4 pricing.",
                    "input_cost_per_million_tokens_usd": self.input_cost_per_million_tokens_usd,
                    "output_cost_per_million_tokens_usd": self.output_cost_per_million_tokens_usd,
                    "remote_estimated_cost_usd": remote_cost,
                    "cache_saved_estimated_cost_usd": saved_cost,
                    "estimated_total_cost_without_cache_usd": remote_cost + saved_cost,
                },
            }


@dataclass(frozen=True)
class LlmRelevanceConfig:
    model: str = "claude-sonnet-4-6"
    top_k: int = 10
    max_workers: int = 4
    max_retries: int = 6
    request_timeout_seconds: float = 120.0
    cache_enabled: bool = True
    cache_path: str = "/cache/evaluation/llm_relevance_cache.sqlite"
    verbose: bool = True
    runtime_analytics: LlmRelevanceAnalytics | None = None
    progress_callback: Callable[[int, int, dict], None] | None = None


_thread_local = threading.local()


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _get_thread_client(config: LlmRelevanceConfig):
    client_key = (config.model, config.max_retries, config.request_timeout_seconds)
    client = getattr(_thread_local, "llm_relevance_client", None)
    configured_key = getattr(_thread_local, "llm_relevance_client_key", None)
    if client is None or configured_key != client_key:
        client = ChatAnthropic(
            model=config.model,
            temperature=0,
            max_retries=config.max_retries,
            timeout=config.request_timeout_seconds,
        ).with_structured_output(RelevanceJudgment, include_raw=True)
        _thread_local.llm_relevance_client = client
        _thread_local.llm_relevance_client_key = client_key
    return client


def build_judgment_prompt(query: dict, document: dict) -> str:
    metadata = document.get("metadata") or {}
    return "\n\n".join(
        [
            "Query:",
            str(query.get("query_text") or query.get("text") or ""),
            "Retrieved document:",
            str(document.get("text") or ""),
            "Document metadata:",
            json.dumps(
                {
                    "doc_id": document.get("doc_id"),
                    "ref": metadata.get("ref"),
                    "lang": metadata.get("lang"),
                    "retrieval_role": metadata.get("retrieval_role"),
                },
                ensure_ascii=False,
            ),
            "Return a binary relevance judgment.",
        ]
    )


def _parse_cached_judgment(content: str) -> RelevanceJudgment:
    try:
        return RelevanceJudgment.model_validate_json(content)
    except ValueError:
        return RelevanceJudgment.model_validate(json.loads(content))


def judge_query_document_relevance(
    query: dict,
    document: dict,
    config: LlmRelevanceConfig,
) -> RelevanceJudgment:
    prompt = build_judgment_prompt(query, document)
    llm_string = (
        f"langchain_anthropic_structured|model={config.model}|"
        "temperature=0|"
        "schema=RelevanceJudgment|"
        f"system={SYSTEM_PROMPT}"
    )
    analytics = config.runtime_analytics
    if config.cache_enabled and config.cache_path:
        cached = cache_lookup(prompt, llm_string, config.cache_path)
        if cached is not None:
            if analytics is not None:
                analytics.record_cache_hit(cached.get("usage"))
            return _parse_cached_judgment(cached["content"])
        if analytics is not None:
            analytics.record_cache_miss()

    if config.verbose:
        print(f"Calling Anthropic structured output {config.model} for LLM relevance judgment")
    response = _get_thread_client(config).invoke(
        [
            ("system", SYSTEM_PROMPT),
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

    if config.cache_enabled and config.cache_path:
        cache_update(prompt, llm_string, parsed.model_dump_json(), usage, config.cache_path)
        if analytics is not None:
            analytics.record_cache_write()
    return parsed


def _top_key(config: LlmRelevanceConfig) -> str:
    return f"top_{config.top_k}"


def _ranked_rows_for_query(query_row: dict, documents_by_id: dict[str, dict], config: LlmRelevanceConfig) -> list[dict]:
    ranked = query_row.get(_top_key(config)) or query_row.get("top_10") or []
    return [
        {
            "query": query_row,
            "ranking": ranking,
            "document": documents_by_id.get(str(ranking.get("doc_id"))),
        }
        for ranking in ranked[: config.top_k]
        if documents_by_id.get(str(ranking.get("doc_id"))) is not None
    ]


def summarize_llm_judged_rows(rows: list[dict], top_k: int) -> dict:
    query_ids = sorted({str(row["query_id"]) for row in rows})
    relevant_count_by_query = {
        query_id: sum(1 for row in rows if str(row["query_id"]) == query_id and row["llm_relevant"])
        for query_id in query_ids
    }
    count_values = list(relevant_count_by_query.values())
    total_judgments = len(rows)
    relevant_judgments = sum(count_values)
    return {
        "query_count": len(query_ids),
        "judged_pairs": total_judgments,
        "top_k": top_k,
        f"llm_relevant_count@{top_k}": sum(count_values) / len(count_values) if count_values else 0.0,
        f"llm_precision@{top_k}": relevant_judgments / total_judgments if total_judgments else 0.0,
        f"llm_any_relevant_rate@{top_k}": (
            sum(1 for value in count_values if value > 0) / len(count_values) if count_values else 0.0
        ),
        "llm_relevant_judgments": relevant_judgments,
        "llm_non_relevant_judgments": total_judgments - relevant_judgments,
    }


def evaluate_llm_judged_top_k(
    *,
    documents_path: str,
    retrieval_results_path: str,
    retrieval_report_path: str,
    output_dir: str,
    config: LlmRelevanceConfig,
) -> dict:
    documents = read_jsonl(documents_path)
    retrieval_rows = read_jsonl(retrieval_results_path)
    retrieval_report = json.loads(Path(retrieval_report_path).read_text())
    documents_by_id = {str(document["doc_id"]): document for document in documents}
    jobs = [
        job
        for query_row in retrieval_rows
        for job in _ranked_rows_for_query(query_row, documents_by_id, config)
    ]
    if config.runtime_analytics is not None:
        config.runtime_analytics.record_jobs_count(len(jobs))

    judged_rows: list[dict] = []
    failures: list[dict] = []

    def run_job(job: dict) -> dict:
        query = job["query"]
        ranking = job["ranking"]
        document = job["document"]
        judgment = judge_query_document_relevance(query, document, config)
        if config.runtime_analytics is not None:
            config.runtime_analytics.record_completed(judgment.relevant)
        metadata = document.get("metadata") or {}
        return {
            "query_id": str(query["query_id"]),
            "query_text": query.get("query_text", ""),
            "doc_id": str(document["doc_id"]),
            "doc_text": document.get("text", ""),
            "doc_ref": metadata.get("ref"),
            "retrieval_role": metadata.get("retrieval_role"),
            "rank": int(ranking["rank"]),
            "retrieval_score": float(ranking["score"]),
            "original_qrel_relevant": bool(ranking.get("is_relevant")),
            "llm_relevant": bool(judgment.relevant),
            "llm_reason": judgment.reason,
        }

    with ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for completed_jobs, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                judged_rows.append(future.result())
            except Exception as exc:
                if config.runtime_analytics is not None:
                    config.runtime_analytics.record_remote_failure()
                query = job["query"]
                ranking = job["ranking"]
                failures.append(
                    {
                        "query_id": str(query.get("query_id")),
                        "doc_id": str(ranking.get("doc_id")),
                        "rank": ranking.get("rank"),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            if config.progress_callback is not None:
                snapshot = config.runtime_analytics.snapshot() if config.runtime_analytics is not None else {}
                try:
                    config.progress_callback(completed_jobs, len(jobs), snapshot)
                except Exception as error:
                    print(f"LLM relevance progress callback failed: {type(error).__name__}: {error}")

    judged_rows.sort(key=lambda row: (row["query_id"], row["rank"]))
    summary = summarize_llm_judged_rows(judged_rows, config.top_k)
    analytics = config.runtime_analytics.snapshot() if config.runtime_analytics is not None else {}
    report = {
        "status": "success",
        "primary_metric": f"llm_relevant_count@{config.top_k}",
        "metric_notes": {
            f"llm_relevant_count@{config.top_k}": "Average number of Claude-judged relevant documents retrieved per query in the top K.",
            f"llm_precision@{config.top_k}": "Fraction of judged top-K query/document pairs marked relevant by Claude.",
        },
        "summary": summary,
        "retrieval_summary": retrieval_report.get("summary", {}),
        "dataset": retrieval_report.get("dataset", {}),
        "judge": {
            "model": config.model,
            "top_k": config.top_k,
            "max_workers": config.max_workers,
            "cache_enabled": config.cache_enabled,
            "cache_path": config.cache_path,
            "failures_count": len(failures),
        },
        "analytics": analytics,
        "failures": failures,
    }
    output_path = Path(output_dir)
    write_jsonl(output_path / "llm_judged_pairs.jsonl", judged_rows)
    write_csv(output_path / "llm_relevance_summary.csv", [summary])
    write_json(output_path / "llm_relevance_summary.json", summary)
    write_json(output_path / "run_config.json", report)
    return report
