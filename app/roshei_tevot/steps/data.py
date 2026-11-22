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
KEPT_FIELDS = ("id", "source", "kind", "surface", "base_expansion", "key", "sense", "before", "after", "title", "category")
# Candidate fields the dataset builder writes (train-only lists, so no held-out label or count reaches them); kept apart from the
# `candidates` that attach_candidates sets, which is what training and evaluation rank.
DATASET_CANDIDATE_FIELDS = {"candidates": "dataset_candidates", "candidate_train_counts": "dataset_candidate_counts",
                            "candidate_senses": "dataset_candidate_senses", "independent_candidates": "dataset_independent_candidates",
                            "independent_candidate_senses": "dataset_independent_candidate_senses"}
CANDIDATE_SOURCES = ("dataset", "lexicon")


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


FOLD_SPLIT_PREFIX = "train_fold:"


def fold_of(raw: dict, n_folds: int) -> int:
    """Cross-validation fold of a dataset row: a hash of its text version and top-level section, so neighbouring passages (which share
    context) fall in the same fold while the rest of the work can still be in training."""
    import hashlib
    group = f"{raw.get('vid')}:{(raw.get('path') or [None])[0]}"
    return int(hashlib.md5(group.encode()).hexdigest()[:8], 16) % n_folds


def parse_fold_split(split: str) -> Optional[tuple[int, int]]:
    """`train_fold:I/N` -> (I, N): fold I of N of train, real examples only (for scoring held-out train rows). Else None."""
    if not split.startswith(FOLD_SPLIT_PREFIX):
        return None
    index, _, n = split[len(FOLD_SPLIT_PREFIX):].partition("/")
    return int(index), int(n)


def load_split(split: str, repo: str = DATASET_REPO, context_words: int = 100, max_rows: Optional[int] = None,
               hebrew_names: Optional[dict] = None, holdout_fold: Optional[int] = None, n_folds: int = 5) -> list[dict]:
    """Read one split, keep the fields training and evaluation use, trim the context, and put the title and category in Hebrew
    (falling back to English for a title the table of contents does not know).

    `holdout_fold` (train only) leaves out the rows of that fold (`fold_of`), for cross-validated training. A split named
    `train_fold:I/N` reads train and keeps only fold I's real (not swapped) rows, which a model trained with holdout_fold=I never saw."""
    only_fold = parse_fold_split(split)
    if only_fold:
        split, (only_index, n_folds) = "train", only_fold
    names = hebrew_names if hebrew_names is not None else load_hebrew_names(repo)
    titles, categories = names["titles"], names["categories"]
    path = dataset_file(repo, f"{split}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if not raw.get("base_expansion"):
                continue
            if only_fold and (raw.get("source") == "swapped" or fold_of(raw, n_folds) != only_index):
                continue
            if holdout_fold is not None and fold_of(raw, n_folds) == holdout_fold:
                continue
            row = {name: raw.get(name) for name in KEPT_FIELDS}
            row.update({ours: raw.get(theirs) or [] for theirs, ours in DATASET_CANDIDATE_FIELDS.items()})
            row["before"] = " ".join((row["before"] or "").split()[-context_words:])
            row["after"] = " ".join((row["after"] or "").split()[:context_words])
            row["title"] = titles.get(row["title"] or "", row["title"] or "")
            row["category"] = categories.get(row["category"] or "", row["category"] or "")
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def dataset_senses(row: dict) -> list[dict]:
    """The dataset's candidates for a row grouped into sense classes: one entry per class, {"unit", "text", "freq", "members"}.

    The candidates are the key's train-only list with train counts, or for an unseen key (empty list) its readings from independent sources
    (kizur, seed, gematria, supported edition tables) at count 0. Readings in the same lexicon sense class (spelling and inflection variants,
    איכא דאמרי / אית דאמרין) become one candidate: its text is the class's most-trained reading, its frequency the class's summed train count.
    A reading with no sense class is a class of its own. Entries come most frequent first."""
    if row.get("dataset_candidates"):
        texts, counts, senses = row["dataset_candidates"], row.get("dataset_candidate_counts") or [], row.get("dataset_candidate_senses") or []
    else:
        texts, counts, senses = row.get("dataset_independent_candidates") or [], [], row.get("dataset_independent_candidate_senses") or []
    groups: dict[str, dict] = {}
    for i, text in enumerate(texts):
        sense = senses[i] if i < len(senses) else None
        unit = f"sense:{sense}" if sense is not None else f"text:{text}"
        count = int(counts[i]) if i < len(counts) else 0
        group = groups.setdefault(unit, {"unit": unit, "text": text, "freq": 0, "members": [], "best": -1})
        if count > group["best"]:  # the list is most-trained first, so ties keep the earlier reading
            group["text"], group["best"] = text, count
        group["freq"] += count
        group["members"].append(text)
    return sorted(({k: v for k, v in g.items() if k != "best"} for g in groups.values()), key=lambda g: -g["freq"])


def _sense_target(row: dict, groups: list[dict]) -> Optional[str]:
    """The unit of the row's label: its sense class, or for a label with no sense class the class whose readings match it by spelling folds."""
    if row.get("sense") is not None:
        unit = f"sense:{row['sense']}"
        return unit if any(g["unit"] == unit for g in groups) else None
    return next((g["unit"] for g in groups if any(same_expansion(m, row["base_expansion"]) for m in g["members"])), None)


def attach_candidates(rows: list[dict], lexicon: Optional[Lexicon], min_candidates: int = 2, training: bool = True,
                      candidate_source: str = "dataset") -> tuple[list[dict], dict]:
    """Give each row the candidates to rank and drop rows with fewer than `min_candidates`.

    `candidate_source="dataset"` (default) ranks sense classes: the dataset builder's train-only candidates for the key, grouped by the
    lexicon's sense classes (`dataset_senses`), with train counts as the frequency prior. No held-out label or count reaches a candidate list,
    its order or the prior. Each row also gets `units` (one id per candidate) and `target_unit`; the target is the label's sense class, so a
    variant spelling of the right sense is never a negative and any reading of that sense is right.
    `"lexicon"` is the older lookup in `surface_index.jsonl` plus the letter check: candidates are readings, not classes, and that index counts
    every labelled example, held-out ones included, so its scores are optimistic.

    `candidates` is most frequent first, with `freqs` alongside. `target` is the candidate text of the label (None when no candidate is the
    label). For training, a row without a matching candidate keeps its gold text as an extra candidate, so it is still a usable example;
    evaluation leaves it out of the candidates and counts it as unanswerable.

    A training row whose answer is a number (`kind == "number"`) is kept with one candidate: a pure numeral (תקצ"ו) has no abbreviation reading, so the number is its
    only class, and the other rows in the batch supply the negatives. Evaluation still drops it, since one candidate is not a choice.
    """
    if candidate_source not in CANDIDATE_SOURCES:
        raise ValueError(f"candidate_source must be one of {CANDIDATE_SOURCES}, not {candidate_source!r}")
    if candidate_source == "lexicon" and lexicon is None:
        raise ValueError("candidate_source='lexicon' needs a lexicon")
    kept, dropped, unmatched, by_sense, dropped_right = [], 0, 0, 0, 0
    for row in rows:
        slim = {k: v for k, v in row.items() if k not in DATASET_CANDIDATE_FIELDS.values()}
        needed = 1 if training and row.get("kind") == "number" else min_candidates
        if candidate_source == "dataset":
            groups = dataset_senses(row)
            target_unit = _sense_target(row, groups)
            if target_unit is None:
                unmatched += 1
                if training:
                    target_unit = f"gold:{row['base_expansion']}"
                    groups = groups + [{"unit": target_unit, "text": row["base_expansion"], "freq": 0, "members": [row["base_expansion"]]}]
            if len(groups) < needed:
                dropped += 1
                dropped_right += target_unit is not None and not target_unit.startswith("gold:")  # one class, and it is the label's: right by default
                continue
            by_sense += row.get("sense") is not None and target_unit is not None and target_unit.startswith("sense:")
            target = next((g["text"] for g in groups if g["unit"] == target_unit), None)
            kept.append({**slim, "candidates": [g["text"] for g in groups], "freqs": [g["freq"] for g in groups],
                         "units": [g["unit"] for g in groups], "target": target, "target_unit": target_unit})
            continue
        found = lexicon.candidates(row["surface"])
        texts = sorted(found, key=lambda t: -found[t])
        target = next((t for t in texts if same_expansion(t, row["base_expansion"])), None)
        if target is None:
            unmatched += 1
            if training:
                texts = texts + [row["base_expansion"]]
                found = {**found, row["base_expansion"]: 0}
                target = row["base_expansion"]
        if len({stem_key(t) for t in texts}) < needed:
            dropped += 1
            continue
        kept.append({**slim, "candidates": texts, "freqs": [found[t] for t in texts], "target": target})
    return kept, {"rows_in": len(rows), "rows_kept": len(kept), "dropped_below_min_candidates": dropped, "gold_not_in_lexicon_lookup": unmatched,
                  "candidate_source": candidate_source, "target_is_sense_class": by_sense,
                  "dropped_and_right_by_default": dropped_right}


def sampling_weights(rows: list[dict], key_alpha: float = 0.5, synthetic_weight: float = 0.5, sense_beta: float = 0.0,
                     sense_max_boost: float = 10.0) -> np.ndarray:
    """Weight 1 / count(key)^alpha so a few very frequent abbreviations do not dominate; synthetic swapped-back rows count for less.

    With `sense_beta` > 0, rows are also balanced across the senses of their key: weight x (share of the row's target class among its key's
    rows)^-beta, capped at `sense_max_boost`, so a rare sense of a common key is seen more often. A key's total weight is unchanged in
    proportion to the others only up to that boost; the frequency prior (chosen on val at evaluation) puts the real base rates back."""
    counts = Counter(row["key"] for row in rows)
    weights = np.array([counts[row["key"]] ** -key_alpha * (synthetic_weight if row["source"] == "swapped" else 1.0) for row in rows])
    if sense_beta > 0:
        unit_of = [row.get("target_unit") or row.get("target") for row in rows]
        per_sense = Counter(zip((row["key"] for row in rows), unit_of))
        boost = np.array([min(sense_max_boost, (per_sense[(row["key"], u)] / counts[row["key"]]) ** -sense_beta) for row, u in zip(rows, unit_of)])
        weights = weights * boost
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
    """Other candidates of the row: other sense classes when the row has units, never a spelling of the target."""
    units = row.get("units")
    negatives = [c for i, c in enumerate(row["candidates"])
                 if (units is None or units[i] != row["target_unit"]) and not same_expansion(c, row["target"])]
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
