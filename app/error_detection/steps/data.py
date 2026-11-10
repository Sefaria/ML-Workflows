"""Data for the Hebrew error detector: load `Sefaria/hebrew-text-corrections`, normalize, build paired negatives, chunk into model inputs.

A dataset row is a window of whitespace tokens with per-token labels (0 ok, 1 erroneous, 2 word missing before this token) and the list of
fixes that turned it into the corrected text. The detector is binary: label 2 is mapped to 0 (there are only a handful of omissions).
"""
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from .vendor.normalize import normalize_word

DATASET_REPO = "Sefaria/hebrew-text-corrections"
# The training pods have no Hugging Face token, so the flows read a GCS mirror of the (private) dataset by default.
GCS_DATASET = "gs://sefaria-ml-models/error-detection/data/hebrew-text-corrections"
LOCAL_DATA_DIR = os.environ.get("ERROR_DETECTION_DATA_DIR", "/tmp/error-detection-data")
FILES = {
    ("real", "train"): "real/train.jsonl",
    ("real", "validation"): "real/validation.jsonl",
    ("real", "test"): "real/test.jsonl",
    ("synthetic", "train"): "synthetic/train.jsonl",
}
FIX_KINDS = ("typo", "spacing", "ocr", "junk", "omission")
IGNORE = -100


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


def word_kinds(row: dict) -> list[Optional[str]]:
    """The fix kind of each labelled token (None for unlabelled tokens), for recall by kind."""
    kinds: list[Optional[str]] = [None] * len(row["tokens"])
    for fix in row["fixes"]:
        if fix.get("labelled"):
            for i in range(max(0, fix["i1"]), min(fix["i2"], len(kinds))):
                if row["labels"][i] == 1:
                    kinds[i] = fix["kind"]
    return kinds


def load_rows(config: str, split: str, repo: str = GCS_DATASET, max_rows: Optional[int] = None) -> list[dict]:
    rows = []
    with open(dataset_file(repo, FILES[(config, split)]), encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                "id": row["id"], "source": row["source"], "ref": row["ref"], "version": row["version"],
                "tokens": row["tokens"], "labels": [1 if label == 1 else 0 for label in row["labels"]],
                "fixes": row["fixes"], "kinds": word_kinds(row),
            })
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def corrected(row: dict) -> dict:
    """The same window after its fixes, every token labelled 0: a negative paired with the erroneous window. Fixes are applied right to
    left so earlier indices stay valid; a fix running past the window end is clipped."""
    tokens = list(row["tokens"])
    for fix in sorted(row["fixes"], key=lambda f: f["i1"], reverse=True):
        i1, i2 = max(0, fix["i1"]), min(fix["i2"], len(tokens))
        if i1 > len(tokens):
            continue
        tokens[i1:i2] = fix["after"]
    return {**row, "id": f"{row['id']}:corrected", "source": "corrected", "tokens": tokens, "labels": [0] * len(tokens),
            "kinds": [None] * len(tokens)}


def chunk_row(tokenizer, row: dict, max_length: int = 256) -> list[dict]:
    """Normalize the words, drop words that normalize to nothing, and pack whole words into chunks of at most `max_length` pieces
    (with [CLS] and [SEP]). The first piece of each word carries its label; other pieces and specials are IGNORE. Each chunk keeps the
    original token index of every labelled position, so predictions map back to words."""
    words, labels, kinds, index = [], [], [], []
    for i, token in enumerate(row["tokens"]):
        word = normalize_word(token)
        if word:
            words.append(word); labels.append(row["labels"][i]); kinds.append(row["kinds"][i]); index.append(i)
    if not words:
        return []
    pieces = tokenizer(words, add_special_tokens=False)["input_ids"]
    budget = max_length - 2
    chunks, current, used = [], [], 0
    for w, ids in enumerate(pieces):
        ids = ids[:budget] or [tokenizer.unk_token_id]
        if current and used + len(ids) > budget:
            chunks.append(current); current, used = [], 0
        current.append((w, ids)); used += len(ids)
    if current:
        chunks.append(current)

    out = []
    for c, chunk in enumerate(chunks):
        input_ids, label_ids, word_index, word_kind = [tokenizer.cls_token_id], [IGNORE], [], []
        for w, ids in chunk:
            input_ids += ids
            label_ids += [labels[w]] + [IGNORE] * (len(ids) - 1)
            word_index.append(index[w]); word_kind.append(kinds[w])
        input_ids.append(tokenizer.sep_token_id); label_ids.append(IGNORE)
        # numpy arrays: ~170k training chunks as Python int lists would cost GBs per DataLoader worker.
        out.append({"id": f"{row['id']}#{c}", "row_id": row["id"], "source": row["source"], "version": row["version"],
                    "input_ids": np.array(input_ids, dtype=np.int32), "labels": np.array(label_ids, dtype=np.int16),
                    "word_index": word_index, "word_kind": word_kind})
    return out


def chunk_rows(tokenizer, rows: Iterable[dict], max_length: int = 256) -> list[dict]:
    return [chunk for row in rows for chunk in chunk_row(tokenizer, row, max_length)]


def build_training_rows(real: list[dict], synthetic: list[dict], paired_negatives: bool = True) -> list[dict]:
    rows = list(real)
    if paired_negatives:
        rows += [corrected(row) for row in real]
    return rows + list(synthetic)


def sampling_weights(chunks: list[dict], synthetic_weight: float = 0.5, negative_weight: float = 1.0) -> np.ndarray:
    """Per-chunk draw weight: real erroneous windows 1, their corrected copies `negative_weight`, synthetic `synthetic_weight`."""
    by_source = {"real": 1.0, "corrected": negative_weight, "synthetic": synthetic_weight}
    weights = np.array([by_source[chunk["source"]] for chunk in chunks], dtype=float)
    if weights.sum() == 0:
        raise ValueError("All sampling weights are zero.")
    return weights / weights.sum()


class ResampledDataset(Dataset):
    """A fixed, seeded draw (with replacement) of `size` chunks. Deterministic, so a resumed Trainer skips the right batches."""

    def __init__(self, chunks: list[dict], size: int, weights: Optional[np.ndarray] = None, seed: int = 0):
        self.chunks = chunks
        if weights is None:
            self.order = np.arange(min(size, len(chunks)))
        else:
            self.order = np.random.default_rng(seed).choice(len(chunks), size=size, p=weights)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        return self.chunks[int(self.order[index])]


class TokenCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, chunks: list[dict]) -> dict:
        width = max(len(chunk["input_ids"]) for chunk in chunks)
        input_ids = torch.full((len(chunks), width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(chunks), width), dtype=torch.long)
        labels = torch.full((len(chunks), width), IGNORE, dtype=torch.long)
        for r, chunk in enumerate(chunks):
            n = len(chunk["input_ids"])
            input_ids[r, :n] = torch.as_tensor(np.asarray(chunk["input_ids"]), dtype=torch.long)
            attention[r, :n] = 1
            labels[r, :n] = torch.as_tensor(np.asarray(chunk["labels"]), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
