"""How a roshei tevot example is turned into model input.

Query (the context tower):

    [CLS] {category} | {title} [SEP] ...before <abbr> {surface} </abbr> after... [SEP]
          segment 0                  segment 1

The metadata is a separate segment so truncation never reaches it, the abbreviation is wrapped in two added marker tokens
(BEREL has no [unused] slots, and ⟦ ⟧ would both become [UNK]), and the context is cut symmetrically around the marked token.
The surface is kept as written, attached prefix included; the target is the expansion without the prefix.

Candidate (the expansion tower): [CLS] {expansion} [SEP], pooled over its tokens.
Cross-encoder: [CLS] {expansion} [SEP] {category} | {title} [SEP] ...context... [SEP], scored from CLS and the marked span.
Category and title are Hebrew (data.load_split maps them through the Sefaria table of contents).
"""
from dataclasses import dataclass

from typing import Optional

import torch

OPEN_MARKER = "<abbr>"
CLOSE_MARKER = "</abbr>"
MARKERS = [OPEN_MARKER, CLOSE_MARKER]


def add_markers(tokenizer) -> int:
    """Register the marker tokens. Returns how many were new (0 when the tokenizer already has them)."""
    return tokenizer.add_special_tokens({"additional_special_tokens": MARKERS})


def marker_ids(tokenizer) -> tuple[int, int]:
    open_id, close_id = tokenizer.convert_tokens_to_ids(MARKERS)
    if tokenizer.unk_token_id in (open_id, close_id):
        raise ValueError("Marker tokens are not in the tokenizer; call add_markers first.")
    return open_id, close_id


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"] if text.strip() else []


@dataclass
class QueryEncoding:
    input_ids: list[int]
    token_type_ids: list[int]
    span_start: int  # index of the opening marker
    span_end: int    # index of the closing marker (inclusive)


def metadata_text(category: str, title: str) -> str:
    return f"{category or ''} | {title or ''}".strip()


def encode_query(
    tokenizer,
    before: str,
    surface: str,
    after: str,
    category: str,
    title: str,
    max_length: int = 384,
    max_meta_tokens: int = 32,
    context_words: int = 100,
    candidate: Optional[str] = None,
    candidate_length: int = 24,
) -> QueryEncoding:
    """Build one query. The window is at most `context_words` words per side and then trimmed to the token budget.

    With `candidate` this is a cross-encoder input: the expansion opens the first segment, `[CLS] candidate [SEP] meta [SEP] context [SEP]`.
    """
    open_id, close_id = marker_ids(tokenizer)
    lead = _ids(tokenizer, candidate)[: candidate_length - 2] + [tokenizer.sep_token_id] if candidate is not None else []
    meta = lead + _ids(tokenizer, metadata_text(category, title))[:max_meta_tokens]
    mid = [open_id] + _ids(tokenizer, surface) + [close_id]
    before_ids = _ids(tokenizer, " ".join(before.split()[-context_words:]))
    after_ids = _ids(tokenizer, " ".join(after.split()[:context_words]))

    # [CLS] meta [SEP] ... [SEP]
    budget = max_length - len(meta) - len(mid) - 3
    if budget < 0:
        raise ValueError(f"max_length {max_length} leaves no room for the abbreviation {surface!r}.")
    half = budget // 2
    take_before = min(len(before_ids), max(half, budget - len(after_ids)))
    take_after = min(len(after_ids), budget - take_before)
    before_ids = before_ids[len(before_ids) - take_before:] if take_before else []
    after_ids = after_ids[:take_after]

    first = [tokenizer.cls_token_id] + meta + [tokenizer.sep_token_id]
    second = before_ids + mid + after_ids + [tokenizer.sep_token_id]
    start = len(first) + len(before_ids)
    return QueryEncoding(
        input_ids=first + second,
        token_type_ids=[0] * len(first) + [1] * len(second),
        span_start=start,
        span_end=start + len(mid) - 1,
    )


def encode_candidate(tokenizer, expansion: str, max_length: int = 24) -> list[int]:
    return [tokenizer.cls_token_id] + _ids(tokenizer, expansion)[: max_length - 2] + [tokenizer.sep_token_id]


def pad_batch(sequences: list[list[int]], pad_id: int, token_types: list[list[int]]):
    """Right-pad to the longest sequence. Returns input_ids, attention_mask and padded token types."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    types = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq)
        mask[i, : len(seq)] = 1
        types[i, : len(seq)] = torch.tensor(token_types[i])
    return ids, mask, types


def collate_queries(tokenizer, encodings: list[QueryEncoding]) -> dict[str, torch.Tensor]:
    ids, mask, types = pad_batch([e.input_ids for e in encodings], tokenizer.pad_token_id, [e.token_type_ids for e in encodings])
    positions = torch.arange(ids.shape[1]).unsqueeze(0)
    starts = torch.tensor([e.span_start for e in encodings]).unsqueeze(1)
    ends = torch.tensor([e.span_end for e in encodings]).unsqueeze(1)
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": types, "span_mask": ((positions >= starts) & (positions <= ends)).long()}


def collate_candidates(tokenizer, sequences: list[list[int]]) -> dict[str, torch.Tensor]:
    ids, mask, types = pad_batch(sequences, tokenizer.pad_token_id, [[0] * len(s) for s in sequences])
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
