"""Version quality from the error detector: share of words flagged in a fixed random sample of each version's segments."""
import csv
import heapq
import io
import json
from collections import defaultdict
from typing import Optional

import numpy as np

from .data import chunk_rows, dataset_file
from .evaluation import predict_words

SAMPLE_FILE = "segments.jsonl"


def load_segments(sample_repo: str, max_segments: Optional[int] = None) -> list[dict]:
    """Rows of {id, ref, version, book, license, tokens}; built by sample_versions.py in translation-engine. Every word is labelled 0 (unknown)."""
    rows = []
    with open(dataset_file(sample_repo, SAMPLE_FILE), encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({**row, "source": "sample", "labels": [0] * len(row["tokens"]), "kinds": [None] * len(row["tokens"]), "fixes": []})
            if max_segments and len(rows) >= max_segments:
                break
    return rows


def segment_counts(words: list[dict], threshold: float, counts: Optional[dict] = None) -> dict:
    """Add per-word predictions into {segment id: [words, flagged, sum of scores]} (chunk ids are '<segment id>#<n>')."""
    out = counts if counts is not None else defaultdict(lambda: [0, 0, 0.0])
    for w in words:
        acc = out[w["chunk"].rsplit("#", 1)[0]]
        acc[0] += 1; acc[1] += int(w["score"] >= threshold); acc[2] += w["score"]
    return out


def bootstrap_ci(words: np.ndarray, flagged: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of flagged words per 1,000, resampling segments."""
    if len(words) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(words), size=(n, len(words)))
    rates = 1000 * flagged[idx].sum(1) / np.maximum(words[idx].sum(1), 1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def aggregate(rows: list[dict], counts: dict, keys=("version",)) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["id"] in counts:
            groups[tuple(row[k] for k in keys)].append(counts[row["id"]])
    out = []
    for key, segs in groups.items():
        words = np.array([s[0] for s in segs], dtype=float)
        flagged = np.array([s[1] for s in segs], dtype=float)
        lo, hi = bootstrap_ci(words, flagged)
        out.append({**dict(zip(keys, key)), "segments": len(segs), "words": int(words.sum()), "flagged": int(flagged.sum()),
                    "flagged_per_1k": round(1000 * flagged.sum() / max(words.sum(), 1), 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                    "mean_score": round(sum(s[2] for s in segs) / max(words.sum(), 1), 5)})
    return sorted(out, key=lambda r: -r["flagged_per_1k"])


class TopExamples:
    """The highest-scoring flagged words of each version, with context, for reading the top and bottom of the list by hand."""

    def __init__(self, per_version: int = 5, context: int = 6):
        self.per_version, self.context = per_version, context
        self.heaps: dict = defaultdict(list)

    def add(self, rows_by_id: dict, words: list[dict], threshold: float) -> None:
        for w in words:
            if w["score"] < threshold:
                continue
            heap = self.heaps[w["version"]]
            if len(heap) < self.per_version or w["score"] > heap[0][0]:
                row = rows_by_id[w["chunk"].rsplit("#", 1)[0]]
                i = w["word_index"]
                item = (w["score"], row["ref"], row["tokens"][i], " ".join(row["tokens"][max(0, i - self.context):i]),
                        " ".join(row["tokens"][i + 1:i + 1 + self.context]))
                (heapq.heappush if len(heap) < self.per_version else heapq.heapreplace)(heap, item)

    def rows(self) -> list[dict]:
        return [{"version": v, "ref": ref, "score": round(score, 4), "word": word, "before": before, "after": after}
                for v, heap in self.heaps.items() for score, ref, word, before, after in sorted(heap, reverse=True)]


def to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def score_segments(model, tokenizer, rows: list[dict], device: str, max_length: int, threshold: float, batch_size: int = 128,
                   block: int = 2000, log_every: int = 10) -> tuple[dict, TopExamples]:
    """Score segments in blocks so per-word records never accumulate (millions of words)."""
    counts, top = defaultdict(lambda: [0, 0, 0.0]), TopExamples()
    for b, start in enumerate(range(0, len(rows), block)):
        part = rows[start:start + block]
        words = predict_words(model, tokenizer, chunk_rows(tokenizer, part, max_length), device, batch_size)
        segment_counts(words, threshold, counts)
        top.add({row["id"]: row for row in part}, words, threshold)
        if b % log_every == 0:
            print(f"scored {start + len(part)}/{len(rows)} segments")
    return counts, top
