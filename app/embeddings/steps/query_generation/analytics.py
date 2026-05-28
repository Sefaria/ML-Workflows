from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional


@dataclass
class QueryGenerationAnalytics:
    input_cost_per_million_tokens_usd: float = 3.0
    output_cost_per_million_tokens_usd: float = 15.0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    documents_count: int = 0
    jobs_count: int = 0
    queries_generated: int = 0
    qrels_generated: int = 0
    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    remote_requests_succeeded: int = 0
    remote_retryable_failures: int = 0
    remote_non_retryable_failures: int = 0
    remote_input_tokens: int = 0
    remote_output_tokens: int = 0
    cached_input_tokens_saved: int = 0
    cached_output_tokens_saved: int = 0

    def record_documents_count(self, count: int) -> None:
        with self._lock:
            self.documents_count = count

    def record_jobs_count(self, count: int) -> None:
        with self._lock:
            self.jobs_count = count

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

    def record_remote_retryable_failure(self) -> None:
        with self._lock:
            self.remote_retryable_failures += 1

    def record_remote_non_retryable_failure(self) -> None:
        with self._lock:
            self.remote_non_retryable_failures += 1

    def record_generated_counts(self, queries: int, qrels: int) -> None:
        with self._lock:
            self.queries_generated += queries
            self.qrels_generated += qrels

    def _cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000.0) * self.input_cost_per_million_tokens_usd
            + (output_tokens / 1_000_000.0) * self.output_cost_per_million_tokens_usd
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            hit_rate = self.cache_hits / self.cache_lookups if self.cache_lookups else 0.0
            remote_cost = self._cost_usd(self.remote_input_tokens, self.remote_output_tokens)
            saved_cost = self._cost_usd(self.cached_input_tokens_saved, self.cached_output_tokens_saved)
            return {
                "documents_count": self.documents_count,
                "jobs_count": self.jobs_count,
                "queries_generated": self.queries_generated,
                "qrels_generated": self.qrels_generated,
                "cache": {
                    "lookups": self.cache_lookups,
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "writes": self.cache_writes,
                    "hit_rate": hit_rate,
                },
                "llm": {
                    "remote_requests_succeeded": self.remote_requests_succeeded,
                    "remote_retryable_failures": self.remote_retryable_failures,
                    "remote_non_retryable_failures": self.remote_non_retryable_failures,
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
