from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class QueryGenerationConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    llm_max_workers: int = 4
    queries_per_type_per_doc: int = 3
    query_types: Tuple[str, ...] = ("keyword", "question", "sentence")
    query_types_per_doc: int = 2
    query_type_sample_seed: int = 613
    verbose: bool = True
    query_language_names: dict = field(
        default_factory=lambda: {
            "en": "English",
            "he": "Hebrew",
        }
    )
