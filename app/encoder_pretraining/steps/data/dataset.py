import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .masking import mask_input_ids


MANIFEST_NAME = "manifest.json"


def read_manifest(corpus_dir: str) -> dict:
    return json.loads((Path(corpus_dir) / MANIFEST_NAME).read_text())


class PackedTokenDataset(Dataset):
    """Memory-mapped view over the packed .npy shards of one group (for example "train")."""

    def __init__(self, corpus_dir: str, group: str, manifest: dict):
        files = manifest["groups"][group]["files"]
        self.shards = [np.load(Path(corpus_dir) / name, mmap_mode="r") for name in files]
        self.offsets = np.cumsum([0] + [len(shard) for shard in self.shards]).tolist()

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index = bisect.bisect_right(self.offsets, index) - 1
        row = self.shards[shard_index][index - self.offsets[shard_index]]
        return {"input_ids": torch.from_numpy(np.array(row, dtype=np.int64))}


def load_eval_windows(corpus_dir: str, manifest: dict, max_sequences: int) -> dict[str, np.ndarray]:
    """Load up to max_sequences eval windows per source, split evenly across the eval groups."""
    groups = sorted(name for name in manifest["groups"] if name.startswith("eval-"))
    if not groups:
        raise ValueError("Corpus manifest has no eval groups.")
    per_group = max(1, max_sequences // len(groups))
    windows = {}
    for group in groups:
        shards = [np.load(Path(corpus_dir) / name) for name in manifest["groups"][group]["files"]]
        windows[group] = np.concatenate(shards)[:per_group] if shards else np.empty((0, manifest["seq_len"]), np.int32)
    return windows


class MaskingCollator:
    """Stack packed windows and apply fresh dynamic masking each time a batch is built.

    Examples that already carry labels (the static eval set) are only stacked.
    """

    def __init__(self, tokenizer, mlm_probability: float):
        self.special_token_ids = tokenizer.all_special_ids
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)
        self.mlm_probability = mlm_probability

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([example["input_ids"] for example in examples])
        if "labels" in examples[0]:
            return {"input_ids": input_ids, "labels": torch.stack([example["labels"] for example in examples])}
        masked_ids, labels = mask_input_ids(
            input_ids,
            special_token_ids=self.special_token_ids,
            mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size,
            mlm_probability=self.mlm_probability,
        )
        return {"input_ids": masked_ids, "labels": labels}
