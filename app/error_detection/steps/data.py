"""Data for the Hebrew error detector: load `Sefaria/hebrew-text-corrections`, normalize, build paired negatives, chunk into model inputs.

A dataset row is a window of whitespace tokens with per-token labels (0 ok, 1 erroneous, 2 word missing before this token) and the list of
fixes that turned it into the corrected text. The detector is binary: label 2 is mapped to 0 (there are only a handful of omissions).

A row may also carry `judged`: {token index: 1 | -1} for words nobody corrected that LLM judges looked at after a model flagged them.
1 makes the word an error (kind "judged"); -1 (judges could not decide) leaves it out of training loss and of metrics.
"""
import hashlib
import json
import os
import re
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
FIX_KINDS = ("typo", "spacing", "ocr", "junk", "omission", "judged")
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
    for index, verdict in (row.get("judged") or {}).items():
        if verdict == 1 and int(index) < len(kinds):
            kinds[int(index)] = "judged"
    for fix in row["fixes"]:
        if fix.get("labelled"):
            for i in range(max(0, fix["i1"]), min(fix["i2"], len(kinds))):
                if row["labels"][i] == 1:
                    kinds[i] = fix["kind"]
    return kinds


def binary_labels(row: dict) -> list[int]:
    """1 for corrected words and judged errors, IGNORE for undecided judged words, else 0."""
    labels = [1 if label == 1 else 0 for label in row["labels"]]
    for index, verdict in (row.get("judged") or {}).items():
        i = int(index)
        if i < len(labels) and labels[i] == 0:
            labels[i] = 1 if verdict == 1 else IGNORE
    return labels


BOOK = re.compile(r"^(.*?)(?: \d+[ab]?(?:[:.]\S*)?(?:-\S+)?)?$")


def fold_of(row: dict, n_folds: int) -> int:
    """Cross-validation fold of a row, by (version, book) like the dataset splits, so a book's text never sits on both sides."""
    book = BOOK.match(row["ref"]).group(1)
    return int(hashlib.md5(f"fold|{row['version']}|{book}".encode()).hexdigest(), 16) % n_folds


def select_fold(rows: list[dict], n_folds: int, fold: int, held_out: bool) -> list[dict]:
    """The rows of `fold` (held_out=True) or every other fold (held_out=False). n_folds 0 keeps everything."""
    if not n_folds:
        return rows
    return [row for row in rows if (fold_of(row, n_folds) == fold) == held_out]


def load_rows(config: str, split: str, repo: str = GCS_DATASET, max_rows: Optional[int] = None) -> list[dict]:
    rows = []
    with open(dataset_file(repo, FILES[(config, split)]), encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                "id": row["id"], "source": row["source"], "ref": row["ref"], "version": row["version"],
                "tokens": row["tokens"], "labels": binary_labels(row), "judged": row.get("judged") or {},
                "fixes": row["fixes"], "kinds": word_kinds(row),
                # where the edited segment sits in `tokens` (older rows without the field are all segment)
                "segment_span": row.get("segment_span") or [0, len(row["tokens"])],
            })
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def corrected(row: dict) -> dict:
    """The same window after its fixes: a negative paired with the erroneous window. Every token is labelled 0 except judged words
    (errors nobody fixed are still there after the fixes), which keep their judged label at their shifted position. Fixes are applied right
    to left so earlier indices stay valid; a fix running past the window end is clipped."""
    tokens = list(row["tokens"])
    start, end = row.get("segment_span") or (0, len(tokens))
    judged = {int(i): v for i, v in (row.get("judged") or {}).items()}
    for fix in sorted(row["fixes"], key=lambda f: f["i1"], reverse=True):
        i1, i2 = max(0, fix["i1"]), min(fix["i2"], len(tokens))
        if i1 > len(tokens):
            continue
        tokens[i1:i2] = fix["after"]
        shift = len(fix["after"]) - (i2 - i1)
        if i1 < end:                                    # fixes inside the segment move its end
            end += shift
        judged = {(i + shift if i >= i2 else i): v for i, v in judged.items() if not i1 <= i < i2}
    labels, kinds = [0] * len(tokens), [None] * len(tokens)
    for i, v in judged.items():
        if i < len(tokens):
            labels[i] = 1 if v == 1 else IGNORE
            kinds[i] = "judged" if v == 1 else None
    return {**row, "id": f"{row['id']}:corrected", "source": "corrected", "tokens": tokens, "labels": labels,
            "kinds": kinds, "judged": {str(i): v for i, v in judged.items()}, "segment_span": [start, max(start, end)]}


LABEL_SCOPES = ("all", "segment")


def chunk_row(tokenizer, row: dict, max_length: int = 256, label_scope: str = "all") -> list[dict]:
    """Normalize the words, drop words that normalize to nothing, and pack whole words into chunks of at most `max_length` pieces
    (with [CLS] and [SEP]). The first piece of each word carries its label; other pieces and specials are IGNORE. Each chunk keeps the
    original token index of every labelled position, so predictions map back to words, and whether that word is in the edited segment.

    label_scope "segment": in real windows and their corrected copies, words outside the edited segment (neighbouring segments, current
    library text whose errors nobody has fixed) stay in the input as context but carry no label. Synthetic rows keep every label, since
    their errors are injected and known."""
    if label_scope not in LABEL_SCOPES:
        raise ValueError(f"label_scope must be one of {LABEL_SCOPES}")
    start, end = row.get("segment_span") or (0, len(row["tokens"]))
    context_only = label_scope == "segment" and row["source"] in ("real", "corrected")
    words, labels, kinds, index, scored, in_seg = [], [], [], [], [], []
    for i, token in enumerate(row["tokens"]):
        word = normalize_word(token)
        if word:
            seg = start <= i < end
            words.append(word); labels.append(row["labels"][i]); kinds.append(row["kinds"][i]); index.append(i)
            in_seg.append(seg); scored.append((seg or not context_only) and row["labels"][i] != IGNORE)
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
        input_ids, label_ids, word_index, word_kind, word_in_segment = [tokenizer.cls_token_id], [IGNORE], [], [], []
        for w, ids in chunk:
            input_ids += ids
            label_ids += [labels[w] if scored[w] else IGNORE] + [IGNORE] * (len(ids) - 1)
            if scored[w]:
                word_index.append(index[w]); word_kind.append(kinds[w]); word_in_segment.append(in_seg[w])
        input_ids.append(tokenizer.sep_token_id); label_ids.append(IGNORE)
        # numpy arrays: ~170k training chunks as Python int lists would cost GBs per DataLoader worker.
        out.append({"id": f"{row['id']}#{c}", "row_id": row["id"], "source": row["source"], "version": row["version"],
                    "input_ids": np.array(input_ids, dtype=np.int32), "labels": np.array(label_ids, dtype=np.int16),
                    "word_index": word_index, "word_kind": word_kind, "word_in_segment": word_in_segment})
    return out


def chunk_rows(tokenizer, rows: Iterable[dict], max_length: int = 256, label_scope: str = "all") -> list[dict]:
    return [chunk for row in rows for chunk in chunk_row(tokenizer, row, max_length, label_scope)]


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
