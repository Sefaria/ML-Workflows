from dataclasses import dataclass, field
from typing import Any, Dict


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
