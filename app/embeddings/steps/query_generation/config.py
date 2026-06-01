from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .analytics import QueryGenerationAnalytics


@dataclass(frozen=True)
class QueryGenerationConfig:
    model: str = "claude-sonnet-4-6"
    max_retries: int = 6
    request_timeout_seconds: float = 120.0
    llm_max_workers: int = 4
    llm_cache_enabled: bool = True
    llm_cache_path: str = "/cache/query_generation/llm_cache.sqlite"
    queries_per_type_per_doc: int = 3
    query_types: Tuple[str, ...] = ("keyword", "question", "sentence")
    query_types_per_doc: int = 2
    query_type_sample_seed: int = 613
    verbose: bool = True
    runtime_analytics: "QueryGenerationAnalytics | None" = None
    query_language_names: dict = field(
        default_factory=lambda: {
            "en": "English",
            "he": "Hebrew",
        }
    )
