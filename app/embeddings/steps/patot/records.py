from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SegmentRecord:
    tref: str
    text: str
    segment_index: int
    kind: str = "base"
    base_tref: Optional[str] = None

