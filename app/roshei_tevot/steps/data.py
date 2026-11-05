import json
import os
import random
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from .encoding import collate_candidates, collate_queries, encode_candidate, encode_query
from .index import Lexicon
from .vendor.spelling import same_expansion, stem_key

DATASET_REPO = "Sefaria/berel-roshei-tevot"
# The training pods have no Hugging Face token, so the flows read a GCS mirror of the (private) dataset by default.
GCS_DATASET = "gs://sefaria-ml-models/roshei-tevot/data/berel-roshei-tevot"
LOCAL_DATA_DIR = os.environ.get("ROSHEI_TEVOT_DATA_DIR", "/tmp/roshei-tevot-data")
LEXICON_FILE = "lexicon/surface_index.jsonl"
HEBREW_TITLES_FILE = "metadata/hebrew_titles.json"
KEPT_FIELDS = ("id", "source", "surface", "base_expansion", "key", "before", "after", "title", "category")


def dataset_file(repo: str, filename: str) -> str:
    """Local path of one dataset file. `repo` is a Hugging Face dataset id or a gs://bucket/prefix mirror of the same files.
    GCS files are downloaded once into LOCAL_DATA_DIR and reused."""
    if not repo.startswith("gs://"):
        return hf_hub_download(repo, filename, repo_type="dataset")
    from utils.gcs import download_blob_to_path

    bucket, _, prefix = repo[len("gs://"):].partition("/")
    local = Path(LOCAL_DATA_DIR) / bucket / prefix / filename
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".partial")
        download_blob_to_path(bucket, f"{prefix.rstrip('/')}/{filename}", str(partial))
        partial.rename(local)
    return str(local)


def load_lexicon(repo: str = DATASET_REPO) -> Lexicon:
    return Lexicon.load(dataset_file(repo, LEXICON_FILE))


def load_hebrew_names(repo: str = DATASET_REPO) -> dict:
    with open(dataset_file(repo, HEBREW_TITLES_FILE), encoding="utf-8") as handle:
        return json.load(handle)


def load_split(split: str, repo: str = DATASET_REPO, context_words: int = 100, max_rows: Optional[int] = None,
               hebrew_names: Optional[dict] = None) -> list[dict]:
    """Read one split, keep the fields training and evaluation use, trim the context, and put the title and category in Hebrew
    (falling back to English for a title the table of contents does not know)."""
    names = hebrew_names if hebrew_names is not None else load_hebrew_names(repo)
    titles, categories = names["titles"], names["categories"]
    path = dataset_file(repo, f"{split}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if not raw.get("base_expansion"):
                continue
            row = {name: raw.get(name) for name in KEPT_FIELDS}
            row["before"] = " ".join((row["before"] or "").split()[-context_words:])
            row["after"] = " ".join((row["after"] or "").split()[:context_words])
            row["title"] = titles.get(row["title"] or "", row["title"] or "")
            row["category"] = categories.get(row["category"] or "", row["category"] or "")
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def attach_candidates(rows: list[dict], lexicon: Lexicon, min_candidates: int = 2, training: bool = True) -> tuple[list[dict], dict]:
    """Give each row the candidates inference would see (lexicon lookup plus letter check) and drop rows with fewer than `min_candidates`.

    `candidates` is the lookup, most frequent first, with `freqs` alongside. `target` is the candidate that the spelling folds match to the
    gold expansion, or None when the lexicon has no such candidate. For training, a row without a matching candidate keeps its gold text as
    an extra candidate, so it is still a usable example; evaluation leaves it out of the candidates and counts it as unanswerable.
    """
    kept, dropped, unmatched = [], 0, 0
    for row in rows:
        found = lexicon.candidates(row["surface"])
        texts = sorted(found, key=lambda t: -found[t])
        target = next((t for t in texts if same_expansion(t, row["base_expansion"])), None)
        if target is None:
            unmatched += 1
            if training:
                texts = texts + [row["base_expansion"]]
                found = {**found, row["base_expansion"]: 0}
                target = row["base_expansion"]
        if len({stem_key(t) for t in texts}) < min_candidates:
            dropped += 1
            continue
        kept.append({**row, "candidates": texts, "freqs": [found[t] for t in texts], "target": target})
    return kept, {"rows_in": len(rows), "rows_kept": len(kept), "dropped_below_min_candidates": dropped, "gold_not_in_lexicon_lookup": unmatched}


def sampling_weights(rows: list[dict], key_alpha: float = 0.5, synthetic_weight: float = 0.5) -> np.ndarray:
    """Weight 1 / count(key)^alpha so a few very frequent abbreviations do not dominate; synthetic swapped-back rows count for less."""
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


def _negatives(rng: random.Random, row: dict, limit: int) -> list[str]:
    negatives = [c for c in row["candidates"] if not same_expansion(c, row["target"])]
    return rng.sample(negatives, min(limit, len(negatives)))


def _query(tokenizer, row: dict, max_length: int, context_words: int, candidate: Optional[str] = None):
    return encode_query(tokenizer, row["before"], row["surface"], row["after"], row["category"], row["title"], max_length,
                        context_words=context_words, candidate=candidate)


class SiameseCollator:
    """A query batch plus one candidate pool for the whole batch.

    The pool is every row's target and up to `hard_negatives` of its mined candidates, deduplicated by folded spelling, so the same expansion
    is never a negative for itself. `own[i, j]` marks pool entries that are row i's own candidates, for the within-candidates loss term.
    """

    def __init__(self, tokenizer, max_length: int = 384, context_words: int = 100, candidate_length: int = 24, hard_negatives: int = 7, seed: int = 0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_words = context_words
        self.candidate_length = candidate_length
        self.hard_negatives = hard_negatives
        self.rng = random.Random(seed)

    def __call__(self, rows: list[dict]) -> dict:
        pool: dict[tuple, int] = {}
        texts: list[str] = []

        def add(text: str) -> int:
            key = stem_key(text)
            if key not in pool:
                pool[key] = len(texts)
                texts.append(text)
            return pool[key]

        targets = [add(r["target"]) for r in rows]
        members = [[targets[i]] + [add(t) for t in _negatives(self.rng, r, self.hard_negatives)] for i, r in enumerate(rows)]
        own = torch.zeros((len(rows), len(texts)), dtype=torch.bool)
        for i, ids in enumerate(members):
            own[i, ids] = True
        return {
            "queries": collate_queries(self.tokenizer, [_query(self.tokenizer, r, self.max_length, self.context_words) for r in rows]),
            "candidates": collate_candidates(self.tokenizer, [encode_candidate(self.tokenizer, t, self.candidate_length) for t in texts]),
            "targets": torch.tensor(targets),
            "own": own,
        }


class CrossCollator:
    """One (context, candidate) sequence per candidate: each row's target first, then up to `hard_negatives` mined candidates.
    `group` maps each sequence to its row; the model scores the sequences and takes cross-entropy within each row, target at position 0."""

    def __init__(self, tokenizer, max_length: int = 384, context_words: int = 100, hard_negatives: int = 7, seed: int = 0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_words = context_words
        self.hard_negatives = hard_negatives
        self.rng = random.Random(seed)

    def __call__(self, rows: list[dict]) -> dict:
        encodings, group = [], []
        for i, row in enumerate(rows):
            for text in [row["target"]] + _negatives(self.rng, row, self.hard_negatives):
                encodings.append(_query(self.tokenizer, row, self.max_length, self.context_words, candidate=text))
                group.append(i)
        return {"pairs": collate_queries(self.tokenizer, encodings), "group": torch.tensor(group), "rows": len(rows)}


def encode_texts(model, tokenizer, texts: list[str], device: str, batch_size: int = 512, max_length: int = 24) -> torch.Tensor:
    """Embed expansions (the siamese candidate tower). Returns a [len(texts), dim] float32 CPU tensor."""
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for start in range(0, len(texts), batch_size):
            batch = collate_candidates(tokenizer, [encode_candidate(tokenizer, t, max_length) for t in texts[start:start + batch_size]])
            out.append(model.encode_candidates(**{k: v.to(device) for k, v in batch.items()}).cpu())
    return torch.cat(out) if out else torch.zeros(0, 1)


def encode_rows(model, tokenizer, rows: list[dict], device: str, batch_size: int = 128, max_length: int = 384, context_words: int = 100) -> torch.Tensor:
    """Embed contexts (the siamese query tower). Returns a [len(rows), dim] float32 CPU tensor."""
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for start in range(0, len(rows), batch_size):
            batch = collate_queries(tokenizer, [_query(tokenizer, r, max_length, context_words) for r in rows[start:start + batch_size]])
            out.append(model.encode_queries(**{k: v.to(device) for k, v in batch.items()}).cpu())
    return torch.cat(out) if out else torch.zeros(0, 1)


def score_pairs(model, tokenizer, rows: list[dict], device: str, batch_size: int = 256, max_length: int = 384, context_words: int = 100) -> list[list[float]]:
    """Cross-encoder scores for every candidate of every row, in each row's candidate order."""
    pairs = [(i, text) for i, row in enumerate(rows) for text in row["candidates"]]
    scores: list[float] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            batch = collate_queries(tokenizer, [_query(tokenizer, rows[i], max_length, context_words, candidate=t) for i, t in chunk])
            scores.extend(model.score(**{k: v.to(device) for k, v in batch.items()}).float().cpu().tolist())
    out, cursor = [], 0
    for row in rows:
        out.append(scores[cursor:cursor + len(row["candidates"])])
        cursor += len(row["candidates"])
    return out
