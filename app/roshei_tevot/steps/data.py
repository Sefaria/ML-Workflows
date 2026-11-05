import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from .encoding import collate_candidates, collate_queries, encode_candidate, encode_query

DATASET_REPO = "Sefaria/berel-roshei-tevot"
KEPT_FIELDS = ("id", "source", "surface", "base_expansion", "key", "candidates", "before", "after", "title", "category")


def load_split(split: str, repo: str = DATASET_REPO, context_words: int = 64, max_rows: Optional[int] = None) -> list[dict]:
    """Read one split, keeping only the fields training and evaluation use and trimming the context to the window."""
    path = hf_hub_download(repo, f"{split}.jsonl", repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if not raw.get("base_expansion"):
                continue
            row = {name: raw.get(name) for name in KEPT_FIELDS}
            row["before"] = " ".join(row["before"].split()[-context_words:])
            row["after"] = " ".join(row["after"].split()[:context_words])
            row["candidates"] = row["candidates"] or []
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def sampling_weights(rows: list[dict], key_alpha: float = 0.5, synthetic_weight: float = 0.5) -> np.ndarray:
    """Weight 1 / count(key)^alpha so a few very frequent abbreviations do not dominate; synthetic rows count for less."""
    counts = Counter(row["key"] for row in rows)
    weights = np.array([counts[row["key"]] ** -key_alpha * (synthetic_weight if row["source"] == "swapped" else 1.0) for row in rows])
    return weights / weights.sum()


class ResampledDataset(Dataset):
    """A fixed, seeded draw (with replacement) of `size` rows. Deterministic, so a resumed Trainer skips the right batches."""

    def __init__(self, rows: list[dict], size: int, weights: Optional[np.ndarray] = None, seed: int = 0):
        self.rows = rows
        self.order = np.random.default_rng(seed).choice(len(rows), size=size, p=weights) if weights is not None else np.arange(min(size, len(rows)))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        return self.rows[int(self.order[index])]


class ExpansionCollator:
    """Turns rows into a query batch plus a candidate pool.

    The pool is the batch's positives and up to `hard_negatives` same-key lexicon candidates per row, deduplicated by text. The
    positive of a row is never one of its own negatives, and an expansion shared by two rows appears once, so it is never a false negative.
    """

    def __init__(self, tokenizer, max_length: int = 256, context_words: int = 64, candidate_length: int = 24, hard_negatives: int = 7, seed: int = 0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_words = context_words
        self.candidate_length = candidate_length
        self.hard_negatives = hard_negatives
        self.rng = random.Random(seed)

    def __call__(self, rows: list[dict]) -> dict:
        encodings = [
            encode_query(self.tokenizer, r["before"], r["surface"], r["after"], r["category"], r["title"], self.max_length, context_words=self.context_words)
            for r in rows
        ]
        pool: dict[str, int] = {}
        for r in rows:
            pool.setdefault(r["base_expansion"], len(pool))
        targets = [pool[r["base_expansion"]] for r in rows]
        for r in rows:
            negatives = [c for c in r["candidates"] if c != r["base_expansion"]]
            for text in self.rng.sample(negatives, min(self.hard_negatives, len(negatives))):
                pool.setdefault(text, len(pool))
        texts = list(pool)
        return {
            "queries": collate_queries(self.tokenizer, encodings),
            "candidates": collate_candidates(self.tokenizer, [encode_candidate(self.tokenizer, t, self.candidate_length) for t in texts]),
            "targets": torch.tensor(targets),
        }


def encode_texts(model, tokenizer, texts: list[str], device: str, batch_size: int = 512, max_length: int = 24) -> torch.Tensor:
    """Embed expansions (the candidate tower). Returns a [len(texts), dim] float32 CPU tensor."""
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for start in range(0, len(texts), batch_size):
            batch = collate_candidates(tokenizer, [encode_candidate(tokenizer, t, max_length) for t in texts[start:start + batch_size]])
            out.append(model.encode_candidates(**{k: v.to(device) for k, v in batch.items()}).cpu())
    return torch.cat(out) if out else torch.zeros(0, 1)


def encode_rows(model, tokenizer, rows: list[dict], device: str, batch_size: int = 128, max_length: int = 256, context_words: int = 64) -> torch.Tensor:
    """Embed contexts (the query tower). Returns a [len(rows), dim] float32 CPU tensor."""
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for start in range(0, len(rows), batch_size):
            encodings = [
                encode_query(tokenizer, r["before"], r["surface"], r["after"], r["category"], r["title"], max_length, context_words=context_words)
                for r in rows[start:start + batch_size]
            ]
            batch = collate_queries(tokenizer, encodings)
            out.append(model.encode_queries(**{k: v.to(device) for k, v in batch.items()}).cpu())
    return torch.cat(out) if out else torch.zeros(0, 1)
