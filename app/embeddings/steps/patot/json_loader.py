import json
import re
from pathlib import Path
from typing import Any, Iterable, Tuple, Union

from .records import SegmentRecord


SEGMENT_SUFFIX_RE = re.compile(r"^\d+(?::\d+)*$")


def _segment_order_key(
    section_ref: str,
    segment_ref: str,
    fallback_index: int,
) -> Tuple[int, Union[Tuple[int, ...], Tuple[str, ...]], int]:
    prefix = f"{section_ref}:"
    suffix = segment_ref[len(prefix) :] if segment_ref.startswith(prefix) else segment_ref.rsplit(":", 1)[-1]
    if SEGMENT_SUFFIX_RE.fullmatch(suffix):
        return (0, tuple(int(part) for part in suffix.split(":")), fallback_index)
    return (1, tuple(suffix.split(":")), fallback_index)


def _ordered_segment_items(section_ref: str, segments: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    indexed_items = list(enumerate(segments.items()))
    for _, item in sorted(
        indexed_items,
        key=lambda indexed_item: _segment_order_key(section_ref, indexed_item[1][0], indexed_item[0]),
    ):
        yield item


def load_segment_records_from_section(section: dict[str, Any]) -> list[SegmentRecord]:
    section_ref = str(section["ref"])
    segments = section.get("segments") or {}
    if not isinstance(segments, dict):
        raise ValueError(f"Expected 'segments' to be an object for section {section_ref}")

    rows: list[SegmentRecord] = []
    for i, (segment_ref, text) in enumerate(_ordered_segment_items(section_ref, segments), start=1):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        rows.append(
            SegmentRecord(
                tref=segment_ref,
                text=cleaned,
                segment_index=i,
                kind="base",
                base_tref=section_ref,
            )
        )
    return rows


def load_segment_records_from_json_file(json_path: Union[str, Path]) -> list[list[SegmentRecord]]:
    path = Path(json_path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level list in {path}")
    return [load_segment_records_from_section(section) for section in payload]
