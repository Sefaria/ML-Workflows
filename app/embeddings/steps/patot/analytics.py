from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class ChunkingRuntimeAnalytics:
    estimated_cost_per_million_tokens_usd: float = 0.15
    estimated_characters_per_token: float = 4.0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    sections_processed: int = 0
    chunked_documents_written: int = 0
    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_hit_characters: int = 0
    cache_miss_characters: int = 0
    remote_requests_succeeded: int = 0
    remote_retryable_responses: int = 0
    remote_non_retryable_failures: int = 0
    remote_success_characters: int = 0

    def record_section_processed(self, chunk_count: int) -> None:
        with self._lock:
            self.sections_processed += 1
            self.chunked_documents_written += chunk_count

    def record_cache_hit(self, text: str) -> None:
        with self._lock:
            self.cache_lookups += 1
            self.cache_hits += 1
            self.cache_hit_characters += len(text)

    def record_cache_miss(self, text: str) -> None:
        with self._lock:
            self.cache_lookups += 1
            self.cache_misses += 1
            self.cache_miss_characters += len(text)

    def record_cache_write(self) -> None:
        with self._lock:
            self.cache_writes += 1

    def record_remote_success(self, text: str) -> None:
        with self._lock:
            self.remote_requests_succeeded += 1
            self.remote_success_characters += len(text)

    def record_remote_retryable_response(self) -> None:
        with self._lock:
            self.remote_retryable_responses += 1

    def record_remote_non_retryable_failure(self) -> None:
        with self._lock:
            self.remote_non_retryable_failures += 1

    def _estimate_tokens(self, characters: int) -> float:
        if self.estimated_characters_per_token <= 0:
            return 0.0
        return characters / self.estimated_characters_per_token

    def _estimate_cost_usd(self, token_count: float) -> float:
        return (token_count / 1_000_000.0) * self.estimated_cost_per_million_tokens_usd

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            remote_estimated_tokens = self._estimate_tokens(self.remote_success_characters)
            cache_saved_estimated_tokens = self._estimate_tokens(self.cache_hit_characters)
            total_estimated_tokens_without_cache = remote_estimated_tokens + cache_saved_estimated_tokens
            hit_rate = self.cache_hits / self.cache_lookups if self.cache_lookups else 0.0
            return {
                "sections_processed": self.sections_processed,
                "chunked_documents_written": self.chunked_documents_written,
                "cache": {
                    "lookups": self.cache_lookups,
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "writes": self.cache_writes,
                    "hit_rate": hit_rate,
                    "hit_characters": self.cache_hit_characters,
                    "miss_characters": self.cache_miss_characters,
                },
                "embeddings": {
                    "remote_requests_succeeded": self.remote_requests_succeeded,
                    "remote_retryable_responses": self.remote_retryable_responses,
                    "remote_non_retryable_failures": self.remote_non_retryable_failures,
                    "remote_success_characters": self.remote_success_characters,
                },
                "estimated_cost": {
                    "pricing_model": "Gemini Developer API gemini-embedding-001 standard pricing",
                    "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
                    "pricing_units": "USD per 1M input tokens",
                    "estimation_note": "Token counts are estimated from characters using assumed_characters_per_token; Gemini billing tokenization may differ.",
                    "cost_per_million_tokens_usd": self.estimated_cost_per_million_tokens_usd,
                    "assumed_characters_per_token": self.estimated_characters_per_token,
                    "remote_estimated_input_tokens": remote_estimated_tokens,
                    "remote_estimated_cost_usd": self._estimate_cost_usd(remote_estimated_tokens),
                    "cache_saved_estimated_input_tokens": cache_saved_estimated_tokens,
                    "cache_saved_estimated_cost_usd": self._estimate_cost_usd(cache_saved_estimated_tokens),
                    "estimated_total_cost_without_cache_usd": self._estimate_cost_usd(total_estimated_tokens_without_cache),
                },
            }
