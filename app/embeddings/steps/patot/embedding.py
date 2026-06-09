import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import requests

from .cache import cache_lookup, cache_update


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE_REQUEST_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


class EmbeddingError(Exception):
    pass


@dataclass(frozen=True)
class GeminiEmbeddingConfig:
    model: str
    embedding_task_setup: str
    output_dimensionality: int
    doc_text_variant: str
    query_text_variant: str
    similarity_metric: str
    normalize_embeddings: bool
    query_task_type: Optional[str] = None
    doc_task_type: Optional[str] = None


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        cache_enabled: bool = False,
        cache_path: Optional[str] = None,
        cache_max_entries: Optional[int] = None,
        timeout_seconds: int = 60,
        max_retries: int = 5,
        initial_backoff_seconds: float = 1.0,
    ):
        self.api_key = api_key
        self.cache_enabled = cache_enabled
        self.cache_path = cache_path
        self.cache_max_entries = cache_max_entries
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.session = requests.Session()

    def _sleep_before_retry(self, attempt: int) -> None:
        backoff = self.initial_backoff_seconds * (2**attempt)
        jitter = random.uniform(0, self.initial_backoff_seconds)
        time.sleep(backoff + jitter)

    def list_models(self) -> list[str]:
        url = f"{GEMINI_API_BASE}/models"
        response = self.session.get(
            url,
            params={"key": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return [model.get("name", "").split("/")[-1] for model in payload.get("models", [])]

    def supports_model(self, model: str) -> bool:
        return model in self.list_models()

    def embed_text(
        self,
        model: str,
        text: str,
        output_dimensionality: int,
        task_type: Optional[str],
        runtime_analytics=None,
    ) -> list[float]:
        llm_string = (
            f"gemini_embedding|model={model}|"
            f"output_dimensionality={output_dimensionality}|task_type={task_type or ''}"
        )
        if self.cache_enabled and self.cache_path:
            cached = cache_lookup(text, llm_string, self.cache_path)
            if cached is not None:
                if runtime_analytics is not None:
                    runtime_analytics.record_cache_hit(text)
                return cached
            if runtime_analytics is not None:
                runtime_analytics.record_cache_miss(text)

        body = {
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": output_dimensionality,
        }
        if task_type is not None:
            body["taskType"] = task_type
        url = f"{GEMINI_API_BASE}/models/{model}:embedContent"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    params={"key": self.api_key},
                    json=body,
                    timeout=self.timeout_seconds,
                )
            except RETRYABLE_REQUEST_EXCEPTIONS as exc:
                last_error = exc
                if runtime_analytics is not None:
                    runtime_analytics.record_remote_retryable_response()
                if attempt < self.max_retries - 1:
                    self._sleep_before_retry(attempt)
                    continue
                break

            if response.status_code in {429, 500, 502, 503, 504}:
                if runtime_analytics is not None:
                    runtime_analytics.record_remote_retryable_response()
                if attempt < self.max_retries - 1:
                    self._sleep_before_retry(attempt)
                    continue
                break
            if not response.ok:
                if runtime_analytics is not None:
                    runtime_analytics.record_remote_non_retryable_failure()
                raise EmbeddingError(f"Embedding call failed: {response.status_code} {response.text}")
            payload = response.json()
            embedding = payload.get("embedding") or {}
            values = embedding.get("values")
            if values is None:
                raise EmbeddingError(f"Missing embedding values in response: {payload}")
            if runtime_analytics is not None:
                runtime_analytics.record_remote_success(text)
            if self.cache_enabled and self.cache_path:
                cache_update(text, llm_string, values, self.cache_path, self.cache_max_entries)
                if runtime_analytics is not None:
                    runtime_analytics.record_cache_write()
            return values

        detail = f": {last_error}" if last_error is not None else ""
        raise EmbeddingError(f"Embedding call failed after {self.max_retries} attempts for model={model}{detail}")


def l2_normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
