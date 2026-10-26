import collections
import copy
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
from tokenizers import Tokenizer, pre_tokenizers, trainers, models
from transformers import PreTrainedTokenizerFast


# Hebrew points and cantillation, without the punctuation marks that live in the same block
# (maqaf U+05BE, paseq U+05C0, sof pasuq U+05C3, nun hafukha U+05C6).
HEBREW_MARKS_PATTERN = r"[֑-ׇֽֿׁׂׅׄ]"
HEBREW_LETTERS = re.compile(r"[א-ת]+")
HEBREW_MARKS = re.compile(HEBREW_MARKS_PATTERN)
NEW_TOKEN_INIT_FILE = "new_token_init.json"


def count_hebrew_words(documents: Iterable[dict], split: str = "train") -> collections.Counter:
    """Count Hebrew words (letters only, vowel marks removed) in documents of one split."""
    counts: collections.Counter = collections.Counter()
    for document in documents:
        if document["split"] == split:
            counts.update(HEBREW_LETTERS.findall(HEBREW_MARKS.sub("", document["text"])))
    return counts


def learn_hebrew_merges(word_counts: collections.Counter, num_merges: int) -> list[tuple[str, str]]:
    """Learn byte-level BPE merges on Hebrew words as ModernBERT's pre-tokenizer sees them.

    Words are fed with a leading space, so the merges apply to running text. The returned list
    starts with the merges that build single Hebrew letters out of bytes; those already exist in
    ModernBERT and are kept so that later merges can build on them.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    trainer = trainers.BpeTrainer(
        vocab_size=256 + num_merges,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
    )

    def lines() -> Iterator[str]:
        buffer: list[str] = []
        for word, count in word_counts.items():
            buffer.extend([word] * count)
            if len(buffer) > 2000:
                yield " " + " ".join(buffer)
                buffer = []
        if buffer:
            yield " " + " ".join(buffer)

    tokenizer.train_from_iterator(lines(), trainer)
    merges = json.loads(tokenizer.to_str())["model"]["merges"]
    return [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in merges]


def build_extended_tokenizer(
    base_tokenizer_json: dict,
    merges: list[tuple[str, str]],
    num_new_tokens: int,
    first_new_id: int,
) -> tuple[dict, list[int]]:
    """Return (tokenizer json, new token ids): vowel marks stripped, Hebrew merges prepended.

    The Hebrew merges run before every existing merge. They only involve Hebrew bytes, so
    English tokenization is unchanged. A merge whose result is already in the vocabulary
    reuses that token; every other merge adds one new token, up to num_new_tokens.
    """
    tokenizer_json = copy.deepcopy(base_tokenizer_json)
    vocab = tokenizer_json["model"]["vocab"]
    # ModernBERT's added tokens ([CLS], [MASK], the unused slots, ...) occupy ids above the BPE vocabulary.
    # Without entries in the vocabulary, `tokenizers` hands those ids out again after the new tokens.
    for added in tokenizer_json["added_tokens"]:
        vocab.setdefault(added["content"], added["id"])
    new_ids: list[int] = []
    kept: list[list[str]] = []
    for left, right in merges:
        merged = left + right
        if merged not in vocab:
            if len(new_ids) >= num_new_tokens:
                break
            vocab[merged] = first_new_id + len(new_ids)
            new_ids.append(vocab[merged])
        kept.append([left, right])
    kept_pairs = {tuple(pair) for pair in kept}
    existing = [
        merge for merge in tokenizer_json["model"]["merges"]
        if tuple(merge if isinstance(merge, list) else merge.split(" ")) not in kept_pairs
    ]
    tokenizer_json["model"]["merges"] = kept + existing
    tokenizer_json["normalizer"] = {
        "type": "Sequence",
        "normalizers": [
            tokenizer_json["normalizer"],
            {"type": "Replace", "pattern": {"Regex": HEBREW_MARKS_PATTERN}, "content": ""},
        ],
    }
    return tokenizer_json, new_ids


def assert_special_tokens_unchanged(base_tokenizer, extended_tokenizer) -> None:
    """Every token the base tokenizer knows by id must keep that id, or the base model's weights no longer line up."""
    for token, token_id in base_tokenizer.get_vocab().items():
        if extended_tokenizer.convert_tokens_to_ids(token) != token_id:
            raise RuntimeError(f"Token {token!r} moved from id {token_id} to {extended_tokenizer.convert_tokens_to_ids(token)}")
    for name in ("cls_token", "sep_token", "pad_token", "mask_token", "unk_token"):
        if getattr(base_tokenizer, name + "_id") != getattr(extended_tokenizer, name + "_id"):
            raise RuntimeError(f"{name} id changed")


def save_extended_tokenizer(tokenizer_json: dict, base_tokenizer, output_dir: str) -> PreTrainedTokenizerFast:
    extended = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(tokenizer_json)),
        cls_token=base_tokenizer.cls_token,
        sep_token=base_tokenizer.sep_token,
        pad_token=base_tokenizer.pad_token,
        mask_token=base_tokenizer.mask_token,
        unk_token=base_tokenizer.unk_token,
        model_max_length=base_tokenizer.model_max_length,
    )
    extended.save_pretrained(output_dir)
    return extended


def new_token_init_map(base_tokenizer, extended_tokenizer, new_ids: list[int]) -> dict[str, list[int]]:
    """For each new token, the base tokenizer's token ids for the same text.

    The model starts a new token's embedding as the mean of these, so it begins close to
    what the model already knows about those letters.
    """
    mapping = {}
    for new_id in new_ids:
        text = extended_tokenizer.convert_tokens_to_string([extended_tokenizer.convert_ids_to_tokens(new_id)])
        mapping[str(new_id)] = base_tokenizer(text, add_special_tokens=False)["input_ids"]
    return mapping


def hebrew_token_usage(
    tokenizer: PreTrainedTokenizerFast,
    word_counts: collections.Counter,
    eval_word_counts: collections.Counter,
    new_ids: list[int],
) -> dict:
    """How well the tokenizer compresses Hebrew and how often each new token occurs in one epoch."""
    words = list(word_counts)
    counts = np.array([word_counts[w] for w in words])
    encoded = tokenizer([" " + w for w in words], add_special_tokens=False)["input_ids"]
    lengths = np.array([len(ids) for ids in encoded])
    usage: collections.Counter = collections.Counter()
    for ids, count in zip(encoded, counts):
        for token_id in ids:
            usage[token_id] += int(count)
    uses = np.array([usage.get(i, 0) for i in new_ids]) if new_ids else np.array([0])

    eval_words = list(eval_word_counts)
    eval_counts = np.array([eval_word_counts[w] for w in eval_words])
    eval_lengths = np.array([len(ids) for ids in tokenizer([" " + w for w in eval_words], add_special_tokens=False)["input_ids"]])
    return {
        "new_tokens": len(new_ids),
        "train_hebrew_words": int(counts.sum()),
        "train_hebrew_tokens": int((lengths * counts).sum()),
        "train_tokens_per_hebrew_word": float((lengths * counts).sum() / counts.sum()),
        "eval_tokens_per_hebrew_word": float((eval_lengths * eval_counts).sum() / eval_counts.sum()),
        "eval_single_token_word_share": float(eval_counts[eval_lengths == 1].sum() / eval_counts.sum()),
        "new_token_uses_per_epoch_min": int(uses.min()),
        "new_token_uses_per_epoch_p5": int(np.percentile(uses, 5)),
        "new_token_uses_per_epoch_median": int(np.median(uses)),
        "new_tokens_used_under_100": int((uses < 100).sum()),
        "new_tokens_unused": int((uses == 0).sum()),
    }
