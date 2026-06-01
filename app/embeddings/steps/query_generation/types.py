from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ChunkedDocument:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def lang(self) -> str:
        return str(self.metadata.get("lang") or "he")


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    text: str
    type: str
    lang: str


@dataclass(frozen=True)
class QrelRecord:
    query_id: str
    doc_id: str
    relevance: int
    reason: str


@dataclass(frozen=True)
class QueryGenerationFailure:
    doc_id: str
    query_type: str
    job_index: int
    error_type: str
    error_message: str
    ref: str


class GeneratedQuery(BaseModel):
    text: str = Field(description="The generated retrieval query.")
    reason: str = Field(description="Why the commentary is relevant beyond the base text.")


class QueryGenerationResponse(BaseModel):
    skip: bool = Field(
        default=False,
        description="True when no good commentary-specific query can be generated.",
    )
    skip_reason: Optional[str] = Field(
        default=None,
        description="Why the example should be skipped when skip is true.",
    )
    queries: list[GeneratedQuery] = Field(default_factory=list)
