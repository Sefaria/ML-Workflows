import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np


HEBREW_WORD_PATTERN = re.compile(r"[֐-׿]+")
TOKENIZE_BATCH_DOCUMENTS = 512


def iter_documents(path: str, document_limit: Optional[int] = None) -> Iterator[dict]:
    """Stream rows from a gzipped JSONL training file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if document_limit is not None and index >= document_limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


class WindowPacker:
    """Concatenate documents (each closed by a separator) and cut fixed windows.

    Every window is [CLS] + (seq_len - 2) tokens + [SEP]. The trailing remainder
    shorter than a full window is dropped by the caller simply not flushing it.
    """

    def __init__(self, seq_len: int, cls_token_id: int, sep_token_id: int):
        self.body_len = seq_len - 2
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.buffer: list[int] = []

    def add_document(self, ids: list[int]) -> Iterator[np.ndarray]:
        self.buffer.extend(ids)
        self.buffer.append(self.sep_token_id)
        while len(self.buffer) >= self.body_len:
            window = [self.cls_token_id, *self.buffer[: self.body_len], self.sep_token_id]
            del self.buffer[: self.body_len]
            yield np.asarray(window, dtype=np.int32)


def pack_token_windows(
    token_id_documents: Iterable[list[int]],
    seq_len: int,
    cls_token_id: int,
    sep_token_id: int,
) -> Iterator[np.ndarray]:
    packer = WindowPacker(seq_len, cls_token_id, sep_token_id)
    for ids in token_id_documents:
        yield from packer.add_document(ids)


class PackedShardWriter:
    """Write packed windows as fixed-width int32 .npy shards plus counts."""

    def __init__(self, output_dir: str, name: str, seq_len: int, shard_sequences: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.seq_len = seq_len
        self.shard_sequences = shard_sequences
        self._rows: list[np.ndarray] = []
        self.shard_files: list[str] = []
        self.sequence_count = 0

    def add(self, window: np.ndarray) -> None:
        self._rows.append(window)
        self.sequence_count += 1
        if len(self._rows) >= self.shard_sequences:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        file_name = f"{self.name}-{len(self.shard_files):05d}.npy"
        np.save(self.output_dir / file_name, np.stack(self._rows))
        self.shard_files.append(file_name)
        self._rows = []

    def close(self) -> dict:
        self.flush()
        return {
            "name": self.name,
            "files": self.shard_files,
            "sequences": self.sequence_count,
            "tokens": self.sequence_count * self.seq_len,
        }


def _batched(documents: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for document in documents:
        batch.append(document)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_corpus(
    documents_path: str,
    tokenizer,
    output_dir: str,
    seq_len: int,
    shard_sequences: int,
    document_limit: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    skip_document: Optional[Callable[[dict], bool]] = None,
) -> dict:
    """Tokenize and pack the corpus.

    Train documents are packed together across sources. Eval documents are
    packed per source so the eval loss can be reported per source. Documents for
    which skip_document returns True are left out and counted per split and source.
    """
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    if cls_token_id is None or sep_token_id is None:
        raise ValueError("Tokenizer must define cls_token_id and sep_token_id for window packing.")

    writers: dict[str, PackedShardWriter] = {}
    packers: dict[str, WindowPacker] = {}
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "words": 0, "tokens": 0, "stored_tokens": 0})

    def packer_for(group: str) -> WindowPacker:
        if group not in packers:
            writers[group] = PackedShardWriter(output_dir, group, seq_len, shard_sequences)
            packers[group] = WindowPacker(seq_len, cls_token_id, sep_token_id)
        return packers[group]

    skipped: dict[str, int] = defaultdict(int)
    documents_seen = 0
    for batch in _batched(iter_documents(documents_path, document_limit), TOKENIZE_BATCH_DOCUMENTS):
        documents_seen += len(batch)
        if skip_document is not None:
            kept = []
            for document in batch:
                if skip_document(document):
                    skipped[f"{document['split']}:{document['source']}"] += 1
                else:
                    kept.append(document)
            batch = kept
            if not batch:
                continue
        encoded = tokenizer([doc["text"] for doc in batch], add_special_tokens=False)["input_ids"]
        for document, ids in zip(batch, encoded):
            split = document["split"]
            group = "train" if split == "train" else f"eval-{document['source']}"
            for window in packer_for(group).add_document(ids):
                writers[group].add(window)
            for key in {split, f"{split}:{document['source']}"}:
                entry = stats[key]
                entry["documents"] += 1
                entry["words"] += int(document.get("words") or 0)
                entry["tokens"] += len(ids)
                entry["stored_tokens"] += int(document.get("tokens") or 0)
        if progress_callback is not None:
            progress_callback(documents_seen)

    shard_reports = {group: writers[group].close() for group in sorted(writers)}
    return {
        "seq_len": seq_len,
        "documents_seen": documents_seen,
        "skipped_documents": dict(skipped),
        "groups": shard_reports,
        "stats": dict(stats),
    }


def hebrew_tokenization_report(
    documents_path: str,
    tokenizer,
    sample_documents: int = 20000,
    max_words: int = 20000,
) -> dict:
    """Measure how many tokens inline Hebrew words cost compared with other words.

    Hebrew-heavy documents were filtered from the corpus, but English text with
    inline Hebrew remains. A high tokens-per-word figure means the tokenizer
    fragments it badly and extending the vocabulary may be worth considering.
    """
    hebrew_words: list[str] = []
    documents_with_hebrew = 0
    documents_scanned = 0
    for document in iter_documents(documents_path, sample_documents):
        documents_scanned += 1
        found = HEBREW_WORD_PATTERN.findall(document["text"])
        if found:
            documents_with_hebrew += 1
            if len(hebrew_words) < max_words:
                hebrew_words.extend(found[: max_words - len(hebrew_words)])

    report: dict = {
        "documents_scanned": documents_scanned,
        "documents_with_hebrew": documents_with_hebrew,
        "hebrew_words_sampled": len(hebrew_words),
    }
    if not hebrew_words:
        return report

    encoded = tokenizer([f" {word}" for word in hebrew_words], add_special_tokens=False)["input_ids"]
    lengths = np.asarray([len(ids) for ids in encoded])
    report.update(
        {
            "mean_tokens_per_hebrew_word": round(float(lengths.mean()), 3),
            "p90_tokens_per_hebrew_word": float(np.percentile(lengths, 90)),
            "mean_tokens_per_hebrew_letter": round(
                float(lengths.sum() / sum(len(word) for word in hebrew_words)), 3
            ),
        }
    )
    return report
